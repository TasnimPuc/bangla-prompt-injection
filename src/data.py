"""Build the multilingual dataset: deepset/prompt-injections (en/de) -> Bangla (MT) -> Banglish (rule-based)
-> code-mixed, plus own red-team seeds. All variants of one source example stay in the same split (no leakage)."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .translit import to_banglish

log = logging.getLogger(__name__)
M2M_LANG = {"en": "en", "de": "de"}


def detect_lang(text: str) -> str:
    import langid
    langid.set_languages(["en", "de"])
    return langid.classify(text)[0]


def load_source(cfg) -> pd.DataFrame:
    from datasets import load_dataset
    ds = load_dataset(cfg.hf_dataset)
    rows = []
    for split in ("train", "test"):
        for i, r in enumerate(ds[split]):
            rows.append({"src_id": f"{split}_{i}", "text": r["text"], "label": int(r["label"]), "orig_split": split})
    df = pd.DataFrame(rows)
    df["src_lang"] = df.text.map(detect_lang)
    log.info("Source: %d rows | %s | labels %s", len(df), df.src_lang.value_counts().to_dict(),
             df.label.value_counts().to_dict())
    return df


class Translator:
    """Local M2M100 translation to Bangla (runs on the Kaggle GPU; no external API)."""

    def __init__(self, cfg):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self.cfg = cfg
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(cfg.mt_model)
        dt = torch.float16 if self.dev == "cuda" else torch.float32
        try:                                    # transformers 5.x: `dtype`; 4.x: `torch_dtype`
            m = AutoModelForSeq2SeqLM.from_pretrained(cfg.mt_model, dtype=dt)
        except TypeError:
            m = AutoModelForSeq2SeqLM.from_pretrained(cfg.mt_model, torch_dtype=dt)
        self.model = m.to(self.dev).eval()

    def __call__(self, texts: list[str], src_langs: list[str]) -> list[str]:
        import torch
        out = [""] * len(texts)
        for lang in set(src_langs):
            idx = [i for i, l in enumerate(src_langs) if l == lang]
            self.tok.src_lang = M2M_LANG.get(lang, "en")
            for b in range(0, len(idx), self.cfg.mt_batch):
                j = idx[b:b + self.cfg.mt_batch]
                enc = self.tok([texts[k] for k in j], return_tensors="pt", padding=True, truncation=True,
                               max_length=self.cfg.mt_max_len).to(self.dev)
                with torch.inference_mode():
                    gen = self.model.generate(**enc, forced_bos_token_id=self.tok.get_lang_id("bn"),
                                              max_new_tokens=self.cfg.mt_max_len, num_beams=4)
                for k, t in zip(j, self.tok.batch_decode(gen, skip_special_tokens=True)):
                    out[k] = t
        return out


def _sentences(t: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?।])\s+", t.strip()) if s]


def code_mix(en: str, bn: str, rng) -> str:
    """Sentence-level mixing of the English/German original and its Banglish/Bangla version."""
    a, b = _sentences(en), _sentences(bn)
    if len(a) == len(b) and len(a) > 1:
        return " ".join(x if rng.random() < 0.5 else y for x, y in zip(a, b))
    wa, wb = en.split(), bn.split()
    h = max(1, len(wb) // 2)
    return " ".join(wb[:h] + wa[len(wa) // 2:]) if rng.random() < 0.5 else " ".join(wa[:len(wa) // 2] + wb[h:])


def augment(src: pd.DataFrame, bn: list[str], cfg) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    rows = []
    for r, b in zip(src.itertuples(index=False), bn):
        base = dict(src_id=r.src_id, label=r.label, orig_split=r.orig_split)
        rows.append(base | {"text": r.text, "variant": r.src_lang})
        if b.strip():
            bl = to_banglish(b)
            rows.append(base | {"text": b, "variant": "bn"})
            rows.append(base | {"text": bl, "variant": "banglish"})
            rows.append(base | {"text": code_mix(r.text, b if rng.random() < 0.5 else bl, rng), "variant": "mixed"})
    return pd.DataFrame(rows)


def load_redteam(cfg) -> pd.DataFrame:
    p = Path(cfg.redteam_path)
    if not p.is_absolute() and not p.exists():
        p = Path(__file__).resolve().parents[1] / cfg.redteam_path
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    df["src_id"] = [f"rt_{i}" for i in range(len(df))]
    df["variant"] = "native_" + df["lang"]
    df["orig_split"] = "redteam"
    return df[["src_id", "text", "label", "variant", "orig_split"]]


def assign_splits(df: pd.DataFrame, cfg) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    ids = df.src_id.unique()
    u = dict(zip(ids, rng.random(len(ids))))
    def split(r):
        if r.orig_split == "test":
            return "test"
        if r.orig_split == "redteam":
            return "train" if u[r.src_id] < cfg.native_train_frac else "test"
        return "val" if u[r.src_id] < cfg.val_frac else "train"
    df = df.copy()
    df["split"] = df.apply(split, axis=1)
    return df


def build(cfg, translate=None, cache: Path | None = None) -> pd.DataFrame:
    cache = cache or Path(cfg.work_dir) / "data" / "dataset.parquet"
    if cache.exists():
        log.info("Dataset from cache %s", cache)
        return pd.read_parquet(cache)
    src = load_source(cfg)
    translate = translate or Translator(cfg)
    log.info("Translating %d texts to Bangla with %s ...", len(src), cfg.mt_model)
    bn = translate(src.text.tolist(), src.src_lang.tolist())
    df = pd.concat([augment(src, bn, cfg), load_redteam(cfg)], ignore_index=True)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    df = assign_splits(df, cfg)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    log.info("Dataset: %d rows\n%s", len(df), pd.crosstab([df.split, df.variant], df.label))
    return df
