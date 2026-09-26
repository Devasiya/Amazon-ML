"""Full inference pipeline: block -> score -> decide -> write outputs.

This run verifies on TRAIN, where the ground truth is known. Inference over all
~10.3M train queries would take about a day, so it uses a random sample of
queries the model was not trained on. Run as `python src/run_pipeline.py` or
`python -m src.run_pipeline` from the project root.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402

from src import config  # noqa: E402
from src.decision_engine import (  # noqa: E402
    find_best_threshold_entity_level, score_predictions_on_train, scores_to_predictions)
from src.infer import run_inference  # noqa: E402
from src.output_writer import write_candidate_pairs, write_matching_results  # noqa: E402
from src.train import TRAIN_PAIRS_FILE  # noqa: E402

VERIFY_QUERIES = 100000


def main():
    print("=== Step 1: Load model and best threshold ===")
    joblib.load(config.MODELS_DIR / "lgbm_model.pkl")  # fail fast if missing
    threshold = json.loads((config.ARTIFACTS_DIR / "best_threshold.json").read_text())["threshold"]
    print(f"Using threshold: {threshold}")

    print(f"=== Step 2: Run inference on TRAIN (verification, {VERIFY_QUERIES:,} unseen queries) ===")
    seen = pd.read_parquet(TRAIN_PAIRS_FILE, columns=["query_entity_id"])["query_entity_id"].unique()
    scored_train = run_inference("train", topk=config.BLOCKING_TOPK, threshold=threshold,
                                 batch_size=10000, max_queries=VERIFY_QUERIES,
                                 exclude_query_ids=seen)
    scored_train.to_parquet(config.ARTIFACTS_DIR / "scored_train_pairs.parquet", index=False)
    print(f"Scored {len(scored_train):,} train pairs")
    # All sampled queries, including those with no candidates (they count against recall).
    query_ids = set(scored_train.attrs["query_ids"])

    print("=== Step 3: Find best entity-level threshold on train ===")
    best_thresh = find_best_threshold_entity_level(scored_train, query_ids=query_ids)
    best = json.loads((config.ARTIFACTS_DIR / "best_threshold_entity.json").read_text())

    print("=== Step 4: Convert to predictions and score on train ===")
    predictions = scores_to_predictions(scored_train, best_thresh,
                                        threshold_empty_addr=best["threshold_empty_addr"])
    metrics = score_predictions_on_train(predictions, query_ids=query_ids)
    print(f"Train entity-level metrics: {metrics}")

    print("=== Step 5: Write train outputs (for verification) ===")
    write_matching_results(predictions, "output/matching_results_train_verify.tsv", split="train")
    write_candidate_pairs(scored_train, "output/candidate_pairs_train_verify.tsv", split="train")


if __name__ == "__main__":
    main()
