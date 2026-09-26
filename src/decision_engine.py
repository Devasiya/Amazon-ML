"""Layer 5 — turn scored candidate pairs into per-S1 match lists, and tune on train.

Every S2/S3 record matches at most one S1 entity in the ground truth, so by
default (best_only) a query is only ever assigned to its highest-scoring S1.
Queries with an empty address can get their own threshold: address similarity
dominates the model, so their scores are not comparable to the rest.
"""
import json
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from . import config
from .data_loader import load_ground_truth
from .score_f05 import score_single

BEST_THRESHOLD_ENTITY_FILE = config.ARTIFACTS_DIR / "best_threshold_entity.json"
NEVER = 1.01  # a threshold no score reaches: "never match"


def _pair_thresholds(pairs: pd.DataFrame, threshold: float,
                     threshold_empty_addr: Optional[float]) -> np.ndarray:
    t = np.full(len(pairs), threshold, dtype=np.float64)
    if threshold_empty_addr is not None and "f_addr_empty_q" in pairs.columns:
        t[pairs["f_addr_empty_q"].to_numpy() == 1] = threshold_empty_addr
    return t


def _best_per_query(scored_pairs: pd.DataFrame) -> pd.DataFrame:
    """Each query's single highest-scoring candidate."""
    order = scored_pairs.sort_values(["query_entity_id", "score"], ascending=[True, False])
    return order.drop_duplicates("query_entity_id", keep="first")


def scores_to_predictions(scored_pairs: pd.DataFrame, threshold: float,
                          best_only: bool = True,
                          threshold_empty_addr: Optional[float] = None) -> Dict[str, List[str]]:
    """{S1 id: [matched S2/S3 ids]} for every S1 id appearing as a candidate.

    best_only: keep only each query's top-scoring candidate before thresholding.
    threshold_empty_addr: threshold for pairs whose query has no address
    (default: same as threshold).
    """
    predictions = {s1: [] for s1 in scored_pairs["candidate_s1_id"].astype(str).unique()}
    pairs = _best_per_query(scored_pairs) if best_only else scored_pairs
    keep = pairs["score"].to_numpy() >= _pair_thresholds(pairs, threshold, threshold_empty_addr)
    kept = pairs.loc[keep, ["candidate_s1_id", "query_entity_id"]].astype(str)
    for s1, q in zip(kept["candidate_s1_id"].tolist(), kept["query_entity_id"].tolist()):
        predictions[s1].append(q)
    return predictions


def _restrict_gt(ground_truth: dict, query_ids: Iterable[str]) -> dict:
    qs = set(query_ids)
    return {s1: [m for m in ms if m in qs] for s1, ms in ground_truth.items()}


def score_predictions_on_train(predictions: Dict[str, List[str]],
                               ground_truth: Optional[dict] = None,
                               query_ids: Optional[Iterable[str]] = None,
                               verbose: bool = True) -> dict:
    """Competition F0.5 (per S1 entity, macro-averaged) against the train ground truth.

    query_ids: when only a sample of S2/S3 records was run through inference, pass
    it. The ground truth is then cut down to matches among those records and the
    average is taken over "touched" S1 entities: those with a true match or a
    predicted match in the sample. (Averaging over all 2.2M S1 would be dominated by
    entities that are trivially singletons in the sample.) This estimates the
    non-singleton part of the metric; it is not the leaderboard number.
    """
    gt = load_ground_truth() if ground_truth is None else ground_truth
    if query_ids is not None:
        gt = _restrict_gt(gt, query_ids)
        s1_ids = [s1 for s1, ms in gt.items() if ms] + [
            s1 for s1, ms in predictions.items() if ms and not gt.get(s1)]
    else:
        s1_ids = list(gt)

    total = 0.0
    tp = n_pred = n_true = 0
    single_ok = single_wrong = 0
    for s1 in s1_ids:
        pred, true = predictions.get(s1, ()), gt.get(s1, ())
        total += score_single(pred, true)
        pset, tset = set(pred), set(true)
        tp += len(pset & tset)
        n_pred += len(pset)
        n_true += len(tset)
        if not tset:
            single_ok += not pset
            single_wrong += bool(pset)

    metrics = {
        "f05": total / max(len(s1_ids), 1),
        "precision": tp / max(n_pred, 1),   # micro, over pairs
        "recall": tp / max(n_true, 1),      # micro, over pairs
        "n_s1_scored": len(s1_ids),
        "singletons_correct": single_ok,
        "singletons_wrong": single_wrong,
    }
    if verbose:
        scope = "touched S1 in sample" if query_ids is not None else "all S1"
        print(f"Entity-level F0.5: {metrics['f05']:.4f}  ({scope}: {len(s1_ids):,})")
        print(f"  pair precision={metrics['precision']:.4f} recall={metrics['recall']:.4f}")
        print(f"  singletons correct={single_ok:,} wrong={single_wrong:,}")
    return metrics


def find_best_threshold_entity_level(scored_pairs: pd.DataFrame, thresholds=None,
                                     ground_truth: Optional[dict] = None,
                                     query_ids: Optional[Iterable[str]] = None,
                                     best_only: bool = True,
                                     tune_empty_addr: bool = True) -> float:
    """Threshold maximizing entity-level F0.5; saves best_threshold_entity.json.

    With tune_empty_addr, a second pass then picks the threshold for empty-address
    queries (including NEVER = no matches for them) with the main one fixed.
    """
    if thresholds is None:
        thresholds = np.arange(0.90, 0.9995, 0.001)
    gt = load_ground_truth() if ground_truth is None else ground_truth
    if query_ids is not None:
        query_ids = set(query_ids)
    pairs = _best_per_query(scored_pairs) if best_only else scored_pairs

    def evaluate(t, t_empty):
        preds = scores_to_predictions(pairs, t, best_only=False, threshold_empty_addr=t_empty)
        return score_predictions_on_train(preds, gt, query_ids, verbose=False)

    print("threshold | precision | recall | F0.5")
    best_t, best_m = None, None
    for t in np.round(thresholds, 3):
        m = evaluate(float(t), None)
        print(f"  {t:.3f}   |  {m['precision']:.4f}   | {m['recall']:.4f} | {m['f05']:.4f}")
        if best_m is None or m["f05"] > best_m["f05"]:
            best_t, best_m = float(t), m

    best_empty = best_t
    if tune_empty_addr and "f_addr_empty_q" in pairs.columns:
        print("empty-address threshold | F0.5   (main threshold fixed at "
              f"{best_t:.3f})")
        for te in [0.90, 0.93, 0.95, 0.96, 0.97, 0.975, 0.98, 0.99, NEVER]:
            m = evaluate(best_t, te)
            label = "never" if te == NEVER else f"{te:.3f}"
            print(f"  {label:<7} | {m['f05']:.4f}")
            if m["f05"] > best_m["f05"]:
                best_empty, best_m = te, m

    result = {"threshold": best_t, "threshold_empty_addr": best_empty,
              "best_only": best_only, "f05": best_m["f05"],
              "precision": best_m["precision"], "recall": best_m["recall"],
              "scope": "sample (touched S1)" if query_ids is not None else "all S1"}
    BEST_THRESHOLD_ENTITY_FILE.write_text(json.dumps(result, indent=2))
    print(f"Best entity-level threshold: {best_t:.3f} (empty-address: {best_empty}) "
          f"F0.5={best_m['f05']:.4f}")
    print(f"Saved -> {BEST_THRESHOLD_ENTITY_FILE}")
    return best_t
