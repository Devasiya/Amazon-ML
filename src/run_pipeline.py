"""Full inference pipeline: block -> score -> decide -> write outputs.

--split train (default) verifies on TRAIN, where the ground truth is known.
Inference over all ~10.3M train queries would take about a day, so it uses a
random sample of queries the model was not trained on, and re-tunes the
entity-level thresholds on it.

--split test runs every test query with the tuned thresholds from
best_threshold_entity.json and writes the two submission files.

Run as `python src/run_pipeline.py` or `python -m src.run_pipeline` from the
project root.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402

from src import config  # noqa: E402
from src.decision_engine import (  # noqa: E402
    BEST_THRESHOLD_ENTITY_FILE, find_best_threshold_entity_level,
    score_predictions_on_train, scores_to_predictions)
from src.infer import run_inference  # noqa: E402
from src.output_writer import (  # noqa: E402
    validate_outputs, write_candidate_pairs, write_matching_results)
from src.train import TRAIN_PAIRS_FILE  # noqa: E402

VERIFY_QUERIES = 100000


def run_train_verify(topk: int, batch_size: int):
    print("=== Step 1: Load model and best threshold ===")
    joblib.load(config.MODELS_DIR / "lgbm_model.pkl")  # fail fast if missing
    threshold = json.loads((config.ARTIFACTS_DIR / "best_threshold.json").read_text())["threshold"]
    print(f"Using threshold: {threshold}")

    print(f"=== Step 2: Run inference on TRAIN (verification, {VERIFY_QUERIES:,} unseen queries) ===")
    seen = pd.read_parquet(TRAIN_PAIRS_FILE, columns=["query_entity_id"])["query_entity_id"].unique()
    scored_train = run_inference("train", topk=topk, threshold=threshold,
                                 batch_size=batch_size, max_queries=VERIFY_QUERIES,
                                 exclude_query_ids=seen)
    scored_train.to_parquet(config.ARTIFACTS_DIR / "scored_train_pairs.parquet", index=False)
    print(f"Scored {len(scored_train):,} train pairs")
    # All sampled queries, including those with no candidates (they count against recall).
    query_ids = set(scored_train.attrs["query_ids"])

    print("=== Step 3: Find best entity-level threshold on train ===")
    best_thresh = find_best_threshold_entity_level(scored_train, query_ids=query_ids)
    best = json.loads(BEST_THRESHOLD_ENTITY_FILE.read_text())

    print("=== Step 4: Convert to predictions and score on train ===")
    predictions = scores_to_predictions(scored_train, best_thresh,
                                        threshold_empty_addr=best["threshold_empty_addr"])
    metrics = score_predictions_on_train(predictions, query_ids=query_ids)
    print(f"Train entity-level metrics: {metrics}")

    print("=== Step 5: Write train outputs (for verification) ===")
    write_matching_results(predictions, "output/matching_results_train_verify.tsv", split="train")
    write_candidate_pairs(scored_train, "output/candidate_pairs_train_verify.tsv", split="train")


def run_test(topk: int, batch_size: int):
    print("=== Step 1: Load model and tuned entity-level thresholds ===")
    joblib.load(config.MODELS_DIR / "lgbm_model.pkl")  # fail fast if missing
    best = json.loads(BEST_THRESHOLD_ENTITY_FILE.read_text())
    print(f"Using threshold: {best['threshold']} (empty-address: {best['threshold_empty_addr']})")

    print("=== Step 2: Run inference on TEST (all queries) ===")
    t0 = time.time()
    # Checkpoints are per (topk, batch_size): batch numbering and contents depend on both.
    scored = run_inference("test", topk=topk, threshold=best["threshold"],
                           batch_size=batch_size,
                           checkpoint_dir=config.ARTIFACTS_DIR / "scored_test_parts"
                                          / f"topk{topk}_bs{batch_size}")
    scored.to_parquet(config.ARTIFACTS_DIR / "scored_test_pairs.parquet", index=False)
    print(f"Scored {len(scored):,} test pairs in {(time.time() - t0) / 3600:.2f}h")

    print("=== Step 3: Convert to predictions ===")
    predictions = scores_to_predictions(scored, best["threshold"],
                                        threshold_empty_addr=best["threshold_empty_addr"])

    print("=== Step 4: Write submission files ===")
    matching = write_matching_results(predictions, config.MATCHING_RESULTS_FILE, split="test")
    candidate = write_candidate_pairs(scored, config.CANDIDATE_PAIRS_FILE, split="test")

    print("=== Step 5: Validate ===")
    ok = validate_outputs(matching, candidate, "data/test")
    print(f"Validator: {'PASSED' if ok else 'FAILED'}")

    matches = pd.read_csv(matching, sep="\t", dtype=str, keep_default_na=False)
    n_matched = int((matches["matched_entity_ids"] != "").sum())
    print(f"S1 entities in matching_results.tsv: {len(matches):,}")
    print(f"  with at least one match:           {n_matched:,}")
    print(f"  predicted singletons (empty):      {len(matches) - n_matched:,}")
    for path in (matching, candidate):
        print(f"{Path(path).name}: {Path(path).stat().st_size / 1e6:.1f} MB")
    if not ok:
        sys.exit(1)
    print("SUBMISSION READY")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=config.SPLITS, default="train",
                        help="train: verify on a sample with ground truth; test: write submission")
    parser.add_argument("--topk", type=int, default=50,
                        help="max blocking candidates per query (default: 50)")
    parser.add_argument("--batch-size", type=int, default=10000,
                        help="S2/S3 queries per inference batch (default: 10000)")
    args = parser.parse_args()
    run = run_train_verify if args.split == "train" else run_test
    run(topk=args.topk, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
