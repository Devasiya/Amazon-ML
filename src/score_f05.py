"""Competition metric: per-S1-entity F0.5, macro-averaged over all S1 entities."""
from typing import Dict, Iterable

from . import config


def compute_f05(precision: float, recall: float, beta: float = config.F05_BETA) -> float:
    """F-beta from precision and recall (beta=0.5 -> 1.25*P*R / (0.25*P + R))."""
    b2 = beta * beta
    denom = b2 * precision + recall
    if denom == 0:
        return 0.0
    return (1 + b2) * precision * recall / denom


def score_single(predicted_ids: Iterable[str], true_ids: Iterable[str]) -> float:
    """F0.5 for one S1 entity.

    - true empty (singleton) and predicted empty -> 1.0
    - true empty but predicted non-empty          -> 0.0 (false merge)
    - true non-empty but predicted empty          -> 0.0 (missed all matches)
    """
    pred = set(predicted_ids)
    true = set(true_ids)
    if not true:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & true)
    return compute_f05(tp / len(pred), tp / len(true))


def score_all(predictions: Dict[str, Iterable[str]],
              ground_truth: Dict[str, Iterable[str]]) -> float:
    """Macro-averaged F0.5 over every S1 entity in ground_truth.

    predictions:  {source1_entity_id: [matched_entity_ids]} (empty list = singleton).
                  An S1 entity absent from predictions is treated as an empty prediction.
    ground_truth: {source1_entity_id: [matched_entity_ids]}, e.g. from
                  data_loader.load_ground_truth().
    """
    if not ground_truth:
        return 0.0
    total = 0.0
    for s1_id, true_ids in ground_truth.items():
        total += score_single(predictions.get(s1_id, ()), true_ids)
    return total / len(ground_truth)
