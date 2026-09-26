import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.score_f05 import compute_f05, score_single, score_all


def test_exact_match():
    assert score_single(["A", "B"], ["A", "B"]) == 1.0

def test_partial_match():
    s = score_single(["A"], ["A", "B"])
    assert 0 < s < 1.0, s

def test_false_positive_on_singleton():
    assert score_single(["A"], []) == 0.0

def test_singleton_predicted_empty():
    assert score_single([], []) == 1.0

def test_missed_all_matches():
    assert score_single([], ["A", "B"]) == 0.0

def test_readme_example():
    # README: predicted [S2-00047, S2-00193, S3-00812], truth [S2-00047, S3-00812] -> 0.714
    s = score_single(["S2-00047", "S2-00193", "S3-00812"], ["S2-00047", "S3-00812"])
    assert abs(s - 0.7142857142857143) < 1e-9, s

def test_compute_f05():
    assert abs(compute_f05(1.0, 1 / 3) - (1.25 * (1 / 3)) / (0.25 + 1 / 3)) < 1e-12
    assert compute_f05(0.0, 0.0) == 0.0

def test_score_all_macro_and_missing_pred():
    gt   = {"S1-1": ["A", "B"], "S1-2": [], "S1-3": ["C"]}
    pred = {"S1-1": ["A"], "S1-3": ["C"]}  # S1-2 omitted -> treated as empty
    expected = (compute_f05(1.0, 0.5) + 1.0 + 1.0) / 3
    assert abs(score_all(pred, gt) - expected) < 1e-12


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
    print("\nAll tests passed")
