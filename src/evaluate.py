from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

ORDER = ["en", "de", "bn", "banglish", "mixed", "native_bn", "native_banglish", "native_mixed"]


def by_variant(test: pd.DataFrame, p: np.ndarray, thr: float, model_name: str) -> pd.DataFrame:
    rows = []
    t = test.assign(p=p, pred=(p >= thr).astype(int))
    groups = [(v, t[t.variant == v]) for v in ORDER if (t.variant == v).any()] + [("ALL", t)]
    for v, d in groups:
        y = d.label.to_numpy()
        tp, fp = int(((d.pred == 1) & (y == 1)).sum()), int(((d.pred == 1) & (y == 0)).sum())
        fn, tn = int(((d.pred == 0) & (y == 1)).sum()), int(((d.pred == 0) & (y == 0)).sum())
        rows.append(dict(model=model_name, variant=v, n=len(d), f1=f1_score(y, d.pred, zero_division=0),
                         auroc=roc_auc_score(y, d.p) if 0 < y.sum() < len(y) else np.nan,
                         attack_miss_rate=fn / max(tp + fn, 1), false_alarm_rate=fp / max(fp + tn, 1)))
    return pd.DataFrame(rows)


def plot(table: pd.DataFrame, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    piv = table.pivot(index="variant", columns="model", values="f1")
    piv = piv.reindex([v for v in ORDER + ["ALL"] if v in piv.index])
    ax = piv.plot.bar(figsize=(9, 3.8), rot=25, width=0.8)
    ax.set_ylabel("F1 (test)"); ax.set_ylim(0, 1.02); ax.legend(fontsize=8, loc="lower left")
    ax.set_title("Prompt-injection detection by language variant")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
