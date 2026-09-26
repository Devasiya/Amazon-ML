"""Inference: block -> features -> score, in batches of S2/S3 queries."""
import json
import time
from typing import Iterable, Optional

import joblib
import pandas as pd

from . import config
from .blocking import KEY_COLS, S1_PREFIX, block_queries, build_inverted_index, make_blocking_keys
from .data_loader import load_all_sources
from .features import compute_features_batch, get_feature_names
from .normalization import normalize_dataframe

MODEL_FILE = config.MODELS_DIR / "lgbm_model.pkl"
OUT_COLS = ["query_entity_id", "candidate_s1_id", "score", "query_country", "f_addr_empty_q"]


def _load_threshold() -> float:
    for name in ("best_threshold_entity.json", "best_threshold.json"):
        path = config.ARTIFACTS_DIR / name
        if path.exists():
            return json.loads(path.read_text())["threshold"]
    return config.MATCH_THRESHOLD


def run_inference(split: str = "test", topk: int = config.BLOCKING_TOPK,
                  threshold: Optional[float] = None, batch_size: int = 10000,
                  max_queries: Optional[int] = None,
                  exclude_query_ids: Optional[Iterable[str]] = None,
                  keep_per_query: int = 5,
                  random_seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """Scored candidate pairs: query_entity_id, candidate_s1_id, score, query_country,
    f_addr_empty_q (the decision engine uses the last one).

    max_queries: run on a random sample of S2/S3 records (stratified by source),
    after dropping exclude_query_ids (e.g. the model's training queries).
    keep_per_query: keep only each query's top-scoring candidates. Keeping all
    topk candidates for ~10M queries would be ~1B rows; decisions only ever use
    each query's best candidate.
    threshold is only reported here; decisions happen in decision_engine.
    """
    t0 = time.time()
    if threshold is None:
        threshold = _load_threshold()
    print(f"[infer] split={split} topk={topk} threshold={threshold} keep_per_query={keep_per_query}")

    df = make_blocking_keys(normalize_dataframe(load_all_sources(split)))
    is_s1   = df["entity_id"].str.startswith(S1_PREFIX)
    s1_df   = df[is_s1]
    s2s3_df = df[~is_s1]
    if exclude_query_ids is not None:
        s2s3_df = s2s3_df[~s2s3_df["entity_id"].isin(set(exclude_query_ids))]
    if max_queries is not None and max_queries < len(s2s3_df):
        frac = max_queries / len(s2s3_df)
        s2s3_df = s2s3_df.groupby("source", observed=True).sample(frac=frac, random_state=random_seed)
    print(f"[infer] S1 records: {len(s1_df):,}  queries: {len(s2s3_df):,}  "
          f"(loaded in {time.time() - t0:.0f}s)")

    indexes = {k: build_inverted_index(s1_df, k) for k in KEY_COLS}
    model = joblib.load(MODEL_FILE)
    features = get_feature_names()

    results, n_pairs = [], 0
    n_batches = (len(s2s3_df) + batch_size - 1) // batch_size
    for b, start in enumerate(range(0, len(s2s3_df), batch_size), 1):
        batch = s2s3_df.iloc[start:start + batch_size]
        pairs = block_queries(batch, indexes, topk)
        if pairs.empty:
            continue
        feats = compute_features_batch(pairs, df)
        feats["score"] = model.predict_proba(feats[features])[:, 1]
        feats = (feats.sort_values(["query_entity_id", "score"], ascending=[True, False])
                      .groupby("query_entity_id", sort=False).head(keep_per_query))
        results.append(feats[OUT_COLS].reset_index(drop=True))
        n_pairs += len(pairs)
        if b == 1 or b % 10 == 0 or b == n_batches:
            print(f"[infer] batch {b}/{n_batches}  pairs scored so far: {n_pairs:,}  "
                  f"elapsed: {time.time() - t0:.0f}s", flush=True)

    scored = pd.concat(results, ignore_index=True) if results else pd.DataFrame(columns=OUT_COLS)
    # All queries run, including those with no candidates (they still count against recall).
    scored.attrs["query_ids"] = s2s3_df["entity_id"].astype(str).tolist()
    print(f"[infer] done: {len(scored):,} pairs kept of {n_pairs:,} scored in {time.time() - t0:.0f}s")
    return scored
