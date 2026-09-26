"""Decision threshold tuning for pair-level F0.5 on the validation pairs."""
import json

import numpy as np
import pandas as pd

from . import config
from .features import get_feature_names
from .score_f05 import compute_f05

BEST_THRESHOLD_FILE = config.ARTIFACTS_DIR / "best_threshold.json"


def find_best_threshold(val_df: pd.DataFrame, model, beta: float = config.F05_BETA) -> float:
    """Threshold in [0.900, 0.999] (step 0.001) maximizing pair-level F-beta on val_df.

    The model is trained with scale_pos_weight ~93, which pushes scores toward 1, so
    the useful range is high. Pass validation pairs only: training pairs overstate it.
    """
    scores = model.predict_proba(val_df[get_feature_names()])[:, 1]
    y = val_df["is_match"].to_numpy() == 1
    n_true = max(int(y.sum()), 1)

    best, rows = None, []
    for t in np.round(np.arange(0.900, 0.9995, 0.001), 3):
        pred = scores >= t
        tp = int((pred & y).sum())
        precision = tp / max(int(pred.sum()), 1)
        recall = tp / n_true
        f = compute_f05(precision, recall, beta)
        rows.append((t, precision, recall, f))
        if best is None or f > best["f05"]:
            best = {"threshold": float(t), "f05": f, "precision": precision, "recall": recall}

    print("threshold | precision | recall | F0.5")
    for t, p, r, f in rows:
        if t >= 0.99 or round(t * 100, 3).is_integer():
            print(f"  {t:.3f}   |  {p:.4f}   | {r:.4f} | {f:.4f}")
    print(f"Best threshold: {best['threshold']:.3f}")
    print(f"  F0.5={best['f05']:.4f} precision={best['precision']:.4f} recall={best['recall']:.4f}")
    if best["threshold"] in (0.9, 0.999):
        print("  note: best threshold is at the edge of the search range")
    BEST_THRESHOLD_FILE.write_text(json.dumps(best, indent=2))
    print(f"Saved -> {BEST_THRESHOLD_FILE}")
    return best["threshold"]
