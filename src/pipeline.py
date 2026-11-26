"""dataset -> (zero-shot English detector) -> mDeBERTa on original only -> mDeBERTa on augmented -> eval -> export."""
from __future__ import annotations

import json
import logging
import os
import shutil
import time
from pathlib import Path

import pandas as pd
import torch

from . import data, evaluate
from .config import Config
from .train import history, load_hub, predict, train

log = logging.getLogger("bpi")
REPO = Path(__file__).resolve().parents[1]


def _hf_token(cfg):
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(cfg.hf_token_secret)
    except Exception:
        return os.environ.get(cfg.hf_token_secret)


def run(cfg: Config, translate=None, hub_baseline: bool = True) -> dict:
    t0 = time.time()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)
    out = Path(cfg.work_dir)
    for d in ("figures", "model", "report", "data"):
        (out / d).mkdir(parents=True, exist_ok=True)
    # %% Build the dataset | deepset/prompt-injections → Bangla (M2M100) → Banglish → code-mixed, plus the native red-team set.
    # show: df.groupby(["split", "variant"]).size().unstack(0)
    df = data.build(cfg, translate)
    store = None
    if cfg.project_id or cfg.local_store:                # checkpoints in the bucket -> resume in a new session
        from .store import make_store
        store = make_store(cfg)
    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
    results = []

    # %% Baseline 1: English detector, zero-shot | Does an off-the-shelf detector work on Bangla?
    if hub_baseline and cfg.baseline_hub_model:
        try:
            m, t = load_hub(cfg.baseline_hub_model)
            results.append(evaluate.by_variant(te, predict(m, t, te.text.tolist(), cfg.max_len), cfg.threshold,
                                               "English detector (zero-shot)"))
            del m; torch.cuda.empty_cache()
        except Exception as e:
            log.warning("zero-shot baseline skipped: %s", e)

    # %% Baseline 2: mDeBERTa on the original data | Same model, no Bangla augmentation. Table = every epoch, also those loaded from the checkpoint.
    # show: history(out / "ckpt_original")
    orig = {"en", "de"}
    m0, t0k = train(cfg, tr[tr.variant.isin(orig)], va[va.variant.isin(orig)], out / "ckpt_original", store=store)
    results.append(evaluate.by_variant(te, predict(m0, t0k, te.text.tolist(), cfg.max_len), cfg.threshold,
                                       "mDeBERTa, original data"))
    del m0; torch.cuda.empty_cache()

    # %% Ours: mDeBERTa + Bangla/Banglish augmentation | Trained on Bangla, Banglish, code-mixed and native red-team text.
    # show: history(out / "ckpt_augmented")
    # show: table[table.variant == "ALL"]
    model, tok = train(cfg, tr, va, out / "ckpt_augmented", store=store)
    p = predict(model, tok, te.text.tolist(), cfg.max_len)
    results.append(evaluate.by_variant(te, p, cfg.threshold, "mDeBERTa, + Bangla/Banglish augmentation (ours)"))
    table = pd.concat(results, ignore_index=True)
    table.to_csv(out / "report" / "results_by_variant.csv", index=False)
    evaluate.plot(table, out / "figures" / "f1_by_variant.png")

    # errors on our own native red-team set (all synthetic text written for this repo -> safe to show)
    nat = te.assign(p=p)[te.variant.str.startswith("native")]
    errs = nat[(nat.p >= cfg.threshold).astype(int) != nat.label][["text", "label", "p"]]

    # %% Export + report | Hugging Face folder, `.pth`, report, optional Hub upload and bucket copy.
    md = out / "model"
    model.save_pretrained(md / "hf"); tok.save_pretrained(md / "hf")
    torch.save({"state_dict": model.state_dict(), "base_model": cfg.model_name, "max_len": cfg.max_len,
                "id2label": {0: "LEGIT", 1: "INJECTION"}}, md / "bangla_injection_mdeberta.pth")
    shutil.copy(REPO / "space" / "MODEL_CARD.md", md / "hf" / "README.md") if (REPO / "space" / "MODEL_CARD.md").exists() else None
    df.to_parquet(out / "data" / "dataset.parquet", index=False)

    summary = dict(n_rows=len(df), splits=df.split.value_counts().to_dict(),
                   variants=df.variant.value_counts().to_dict(), results=table.to_dict("records"),
                   runtime_min=round((time.time() - t0) / 60, 1), config=cfg.to_dict())
    (out / "metrics.json").write_text(json.dumps(summary, indent=1, default=float, ensure_ascii=False))
    from .report import build as build_report
    build_report(out, table, errs, summary)

    # 5. optional: push model / dataset / demo Space to the Hugging Face Hub (public is fine: Apache-2.0 data)
    tok_hf = _hf_token(cfg)
    if tok_hf and (cfg.hub_model_repo or cfg.hub_dataset_repo or cfg.hub_space_repo):
        from .hub import push_all
        summary["hub"] = push_all(cfg, out, tok_hf)
    # optional: copy results to the GCS bucket used by the other projects
    if store is not None:
        store.put_dir(out / "report", f"runs/{cfg.run_name}/report")
        store.put_dir(out / "figures", f"runs/{cfg.run_name}/figures")
        store.put(md / "bangla_injection_mdeberta.pth", f"runs/{cfg.run_name}/model/bangla_injection_mdeberta.pth")
    ours = table[(table.model.str.contains("ours")) & (table.variant == "ALL")].iloc[0]
    log.info("DONE in %.1f min | ours: F1 %.3f AUROC %.3f on all test variants", summary["runtime_min"], ours.f1, ours.auroc)
    return summary
