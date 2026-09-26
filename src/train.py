"""Layer 4 — training data generation and the LightGBM pair classifier."""
import json
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import config
from .blocking import KEY_COLS, S1_PREFIX, block_queries, build_inverted_index, make_blocking_keys
from .data_loader import load_all_sources, load_ground_truth
from .features import compute_features_batch, get_feature_names
from .normalization import normalize_dataframe
from .score_f05 import compute_f05

FEATURE_SCHEMA_FILE = config.ARTIFACTS_DIR / "feature_schema.json"
TRAIN_PAIRS_FILE    = config.ARTIFACTS_DIR / "train_pairs.parquet"
_ID_COLS = ["query_entity_id", "candidate_s1_id", "query_country"]


def _resolve(path) -> Path:
    """Relative paths are taken from the project root, not the working directory."""
    path = Path(path)
    return path if path.is_absolute() else config.PROJECT_ROOT / path


def _gt_pairs(gt: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [(s1, m) for s1, matches in gt.items() for m in matches],
        columns=["candidate_s1_id", "query_entity_id"],
    )


def build_training_data(max_queries: int = 100000, topk: int = 50,
                        random_seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """Labeled candidate pairs with features for a random sample of S2/S3 queries."""
    t0 = time.time()
    df = make_blocking_keys(normalize_dataframe(load_all_sources("train")))
    is_s1   = df["entity_id"].str.startswith(S1_PREFIX)
    s1_df   = df[is_s1]
    s2s3_df = df[~is_s1]

    # Stratified by source, so S2 and S3 keep their share of the queries.
    frac = min(1.0, max_queries / len(s2s3_df))
    queries = s2s3_df.groupby("source", observed=True).sample(frac=frac, random_state=random_seed)
    print(f"Sampled queries: {len(queries):,}  "
          f"{queries['source'].value_counts().sort_index().to_dict()}")

    indexes = {k: build_inverted_index(s1_df, k) for k in KEY_COLS}
    pairs = block_queries(queries, indexes, topk)
    print(f"Candidate pairs: {len(pairs):,}  (blocking done at {time.time() - t0:.0f}s)")

    gt_pairs = _gt_pairs(load_ground_truth())
    gt_pairs = gt_pairs[gt_pairs["query_entity_id"].isin(set(queries["entity_id"].tolist()))]
    labeled = pairs.astype({"query_entity_id": str, "candidate_s1_id": str}).merge(
        gt_pairs.assign(is_match=1), on=["query_entity_id", "candidate_s1_id"], how="left")
    labeled["is_match"] = labeled["is_match"].fillna(0).astype(np.int8)
    print(f"True pairs for sampled queries: {len(gt_pairs):,}; "
          f"covered by blocking: {labeled['is_match'].sum() / max(len(gt_pairs), 1):.4f}")

    labeled = compute_features_batch(labeled, df)

    n_pos = int(labeled["is_match"].sum())
    n = len(labeled)
    print("Label distribution:")
    print(f"  total pairs: {n:,}")
    print(f"  positives:   {n_pos:,}")
    print(f"  negatives:   {n - n_pos:,}")
    print(f"  positive rate: {n_pos / max(n, 1):.4f}  (1 : {(n - n_pos) / max(n_pos, 1):.1f})")
    print(f"build_training_data: {time.time() - t0:.0f}s")
    return labeled[_ID_COLS + get_feature_names() + ["is_match"]]


def _split_by_query(df: pd.DataFrame, val_frac: float = 0.2, seed: int = config.RANDOM_SEED):
    """80/20 split on query_entity_id so a query's pairs never straddle train/val."""
    qids = df["query_entity_id"].unique()
    rng = np.random.default_rng(seed)
    val_q = set(rng.choice(qids, size=int(len(qids) * val_frac), replace=False).tolist())
    is_val = df["query_entity_id"].isin(val_q).to_numpy()
    return df[~is_val], df[is_val]


def train_model(labeled_df: pd.DataFrame, save_path: str = "models/lgbm_model.pkl") -> tuple:
    """Train LightGBM; returns (model, val_df with a `score` column)."""
    features = get_feature_names()
    train_df, val_df = _split_by_query(labeled_df)
    X_tr, y_tr = train_df[features], train_df["is_match"]
    X_va, y_va = val_df[features], val_df["is_match"]
    print(f"Train: {len(train_df):,} pairs / {train_df['query_entity_id'].nunique():,} queries; "
          f"Val: {len(val_df):,} pairs / {val_df['query_entity_id'].nunique():,} queries")

    scale_pos_weight = float((y_tr == 0).sum()) / max(int((y_tr == 1).sum()), 1)
    params = {
        "objective": "binary",
        "metric": "auc",
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,  # LightGBM ignores subsample unless this is > 0
        "colsample_bytree": 0.8,
        "scale_pos_weight": scale_pos_weight,
        "random_state": config.RANDOM_SEED,
        "n_jobs": -1,
        "verbose": -1,
    }
    print(f"scale_pos_weight: {scale_pos_weight:.2f}")
    model = lgb.LGBMClassifier(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
              callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(50)])

    train_auc = roc_auc_score(y_tr, model.predict_proba(X_tr)[:, 1])
    val_df = val_df.copy()
    val_df["score"] = model.predict_proba(X_va)[:, 1]
    val_auc = roc_auc_score(y_va, val_df["score"])
    print(f"Best iteration: {model.best_iteration_}")
    print(f"Train AUC: {train_auc:.5f}")
    print(f"Val AUC:   {val_auc:.5f}")

    gain = pd.Series(model.booster_.feature_importance(importance_type="gain"), index=features)
    print("Top 10 features by gain:")
    for name, g in (gain / gain.sum()).sort_values(ascending=False).head(10).items():
        print(f"  {name:<22} {g:.4f}")

    model_path = _resolve(save_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    schema = {"features": features, "n_features": len(features), "model_path": str(save_path)}
    FEATURE_SCHEMA_FILE.write_text(json.dumps(schema, indent=2))
    print(f"Saved model -> {model_path}")
    print(f"Saved schema -> {FEATURE_SCHEMA_FILE}")
    return model, val_df


def _pair_metrics(y_true: np.ndarray, pred: np.ndarray) -> tuple:
    tp = int((pred & (y_true == 1)).sum())
    precision = tp / max(int(pred.sum()), 1)
    recall = tp / max(int((y_true == 1).sum()), 1)
    return precision, recall, compute_f05(precision, recall)


def evaluate_val(val_df: pd.DataFrame, model, threshold: float = 0.5) -> dict:
    """Pair-level precision / recall / F0.5 at threshold, plus AUC (over candidate pairs)."""
    scores = model.predict_proba(val_df[get_feature_names()])[:, 1]
    y = val_df["is_match"].to_numpy()
    pred = scores >= threshold
    precision, recall, f05 = _pair_metrics(y, pred)
    auc = roc_auc_score(y, scores)
    print(f"Val @ {threshold}: precision={precision:.4f} recall={recall:.4f} "
          f"F0.5={f05:.4f} AUC={auc:.5f}")
    if "query_country" in val_df.columns:
        countries = val_df["query_country"].astype(str).to_numpy()
        for c in sorted(set(countries)):
            m = countries == c
            p, r, f = _pair_metrics(y[m], pred[m])
            print(f"  {c:<8} precision={p:.4f} recall={r:.4f} F0.5={f:.4f}  ({m.sum():,} pairs)")
    return {"precision": precision, "recall": recall, "f05": f05, "auc": auc}


if __name__ == "__main__":
    from .calibrate import find_best_threshold

    labeled_df = build_training_data(max_queries=100000, topk=50)
    labeled_df.to_parquet(TRAIN_PAIRS_FILE, index=False)
    print(f"Saved training pairs -> {TRAIN_PAIRS_FILE}")
    model, val_df = train_model(labeled_df)
    metrics = evaluate_val(val_df, model)
    print("Val F0.5:", round(metrics["f05"], 4))
    find_best_threshold(val_df, model)
