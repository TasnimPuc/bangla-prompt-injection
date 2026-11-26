"""Fine-tune a multilingual encoder for injection (1) vs legit (0)."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

log = logging.getLogger(__name__)


def _metrics(eval_pred):
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    logits, y = eval_pred
    p = torch.softmax(torch.tensor(logits), -1)[:, 1].numpy()
    out = {"accuracy": accuracy_score(y, p >= 0.5), "f1": f1_score(y, p >= 0.5)}
    if 0 < y.sum() < len(y):
        out["auroc"] = roc_auc_score(y, p)
    return out


def train(cfg, train_df: pd.DataFrame, val_df: pd.DataFrame, out_dir: str | Path, model_name: str | None = None,
          store=None):
    """Fine-tunes; every epoch is mirrored to the bucket (if a store is given) so a new session resumes."""
    import hashlib

    from datasets import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer,
                              TrainerCallback, TrainingArguments)
    name = model_name or cfg.model_name
    tok = AutoTokenizer.from_pretrained(name)
    kw = dict(num_labels=2, id2label={0: "LEGIT", 1: "INJECTION"}, label2id={"LEGIT": 0, "INJECTION": 1},
              ignore_mismatched_sizes=True)
    # master weights must be fp32 for mixed-precision training (transformers 5.x would keep the checkpoint's fp16)
    try:
        model = AutoModelForSequenceClassification.from_pretrained(name, dtype=torch.float32, **kw)
    except TypeError:
        model = AutoModelForSequenceClassification.from_pretrained(name, torch_dtype=torch.float32, **kw)
    model = model.float()

    def enc(df):
        ds = Dataset.from_pandas(df[["text", "label"]].rename(columns={"label": "labels"}).reset_index(drop=True))
        return ds.map(lambda b: tok(b["text"], truncation=True, max_length=cfg.max_len), batched=True,
                      remove_columns=["text"])
    cuda = torch.cuda.is_available()
    args = TrainingArguments(
        output_dir=str(out_dir), num_train_epochs=cfg.epochs, learning_rate=cfg.lr, weight_decay=0.01,
        per_device_train_batch_size=cfg.batch_size, per_device_eval_batch_size=64,
        warmup_steps=int(0.1 * cfg.epochs * -(-len(train_df) // cfg.batch_size)),   # works on transformers 4.x and 5.x
        eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True, metric_for_best_model="f1",
        save_total_limit=1, fp16=cuda, logging_steps=20, report_to=[], seed=cfg.seed, dataloader_num_workers=2)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = hashlib.md5(("".join(train_df.text) + str(train_df.label.sum()) + name + str(cfg.epochs)).encode()).hexdigest()[:8]
    ck = f"ckpt/{cfg.run_name}/{out_dir.name}-{fp}"                     # other data / model -> fresh checkpoints

    class Mirror(TrainerCallback):
        def on_save(self, args, state, control, **kw):
            if store is None:
                return
            last = max(out_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
            store.put_dir(last, f"{ck}/{last.name}")
            (out_dir / "LATEST").write_text(last.name)
            store.put(out_dir / "LATEST", f"{ck}/LATEST")
            log.info("checkpoint %s mirrored to bucket", last.name)

    if store is not None and not list(out_dir.glob("checkpoint-*")) and store.get(f"{ck}/LATEST", out_dir / "LATEST"):
        cname = (out_dir / "LATEST").read_text().strip()
        for k in store.list(f"{ck}/{cname}"):
            store.get(k, out_dir / cname / Path(k).relative_to(f"{ck}/{cname}"))
        log.info("RESUME from bucket checkpoint %s", cname)
    tr = Trainer(model=model, args=args, train_dataset=enc(train_df), eval_dataset=enc(val_df),
                 processing_class=tok, data_collator=DataCollatorWithPadding(tok), compute_metrics=_metrics,
                 callbacks=[Mirror()])
    ckpts = sorted(out_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    saved = _saved_epochs(ckpts[-1] if ckpts else None)
    if saved:
        log.info("loaded checkpoint %s: %d of %d epochs already done", ckpts[-1].name, saved, cfg.epochs)
    tr.train(resume_from_checkpoint=str(ckpts[-1]) if ckpts else None)      # resumes if the session died
    hist = _epoch_table(tr.state.log_history, "f1", tr.state.best_metric, saved)
    hist.to_csv(out_dir / "history.csv", index=False)
    log.info("training history (all epochs):\n%s", hist.to_string(index=False) if len(hist) else "-")
    return tr.model.eval(), tok


@torch.no_grad()
def predict(model, tok, texts, max_len=256, bs=64) -> np.ndarray:
    dev = next(model.parameters()).device
    pos = 1
    lab = {int(k): str(v).upper() for k, v in getattr(model.config, "id2label", {}).items()}
    for k, v in lab.items():
        if "INJECT" in v:
            pos = k
    out = []
    for b in range(0, len(texts), bs):
        e = tok(list(texts[b:b + bs]), truncation=True, max_length=max_len, padding=True, return_tensors="pt").to(dev)
        out.append(torch.softmax(model(**e).logits.float(), -1)[:, pos].cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def load_hub(name):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return AutoModelForSequenceClassification.from_pretrained(name).to(dev).eval(), AutoTokenizer.from_pretrained(name)


def _epoch_table(log_history, best_metric_key, best_metric, saved_epochs):
    """Trainer log_history -> one row per epoch (train loss + validation metrics)."""
    import math
    loss, rows = {}, []
    for h in log_history:
        ep = h.get("epoch")
        if ep is None:
            continue
        if "loss" in h:
            loss[max(1, math.ceil(ep - 1e-9))] = h["loss"]
        if "eval_loss" in h:
            e = int(round(ep))
            r = {"epoch": e, "train_loss": loss.get(e)}
            r.update({k[5:]: v for k, v in h.items() if k.startswith("eval_") and
                      not k.endswith(("runtime", "per_second"))})
            r["source"] = "checkpoint" if e <= saved_epochs else "this session"
            rows.append(r)
    df = pd.DataFrame(rows)
    if len(df) and best_metric is not None and best_metric_key in df:
        hit = np.flatnonzero(np.isclose(df[best_metric_key], best_metric))
        df["best"] = ["★" if i == (hit[0] if len(hit) else -1) else "" for i in range(len(df))]   # first best = kept
    return df


def _saved_epochs(ckpt_dir) -> int:
    import json
    f = Path(ckpt_dir) / "trainer_state.json" if ckpt_dir else None
    return int(round(json.loads(f.read_text()).get("epoch", 0))) if f and f.exists() else 0


def history(out_dir) -> pd.DataFrame:
    """Per-epoch values of the last training in out_dir, including epochs restored from a checkpoint."""
    f = Path(out_dir) / "history.csv"
    return pd.read_csv(f) if f.exists() else pd.DataFrame({"info": ["no training history yet"]})
