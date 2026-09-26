"""Layer 3 — pairwise features for (query S2/S3 record, candidate S1 record).

Everything is computed on whole columns: string similarities go through
rapidfuzz.process.cpdist (element-wise, in C, multi-threaded), the rest through
numpy or flat list comprehensions. Empty names/addresses score 0 similarity.
"""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist
from tqdm import tqdm

from .blocking import KEY_COLS
from .normalization import get_name_tokens

_RECORD_COLS = ["entity_id", "norm_name", "norm_address", "norm_country", "source"]

FEATURE_NAMES = [
    # name similarity
    "f_name_exact", "f_name_ratio", "f_name_partial", "f_name_token_sort",
    "f_name_token_set", "f_name_jaro",
    # token overlap
    "f_token_jaccard", "f_token_overlap_count", "f_token_q_in_s1", "f_token_s1_in_q",
    # address
    "f_addr_ratio", "f_addr_token_set", "f_addr_empty_q", "f_addr_empty_s1", "f_addr_both_empty",
    # length
    "f_name_len_diff", "f_name_len_ratio",
    # source / blocking
    "f_source_q", "f_n_keys_shared",
]


def get_feature_names() -> list:
    """Feature columns in the order they appear in the matrix."""
    return list(FEATURE_NAMES)


def _record_lookup(all_records: pd.DataFrame, ids) -> pd.DataFrame:
    """Rows of all_records for ids, indexed by entity_id (keys included if present)."""
    cols = _RECORD_COLS + [k for k in KEY_COLS if k in all_records.columns]
    sub = all_records.loc[all_records["entity_id"].isin(pd.unique(np.asarray(ids))), cols]
    return sub.drop_duplicates("entity_id").set_index("entity_id")


def _sim(a: list, b: list, scorer, valid: np.ndarray) -> np.ndarray:
    """Element-wise similarity scaled to 0-1; 0 where not valid (an empty side)."""
    scores = cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32).astype(np.float64)
    if scorer is not JaroWinkler.normalized_similarity:
        scores /= 100.0
    return np.where(valid, scores, 0.0)


def build_feature_matrix(pairs_df: pd.DataFrame, all_records: pd.DataFrame) -> pd.DataFrame:
    """pairs_df (query_entity_id, candidate_s1_id, ...) with feature columns added."""
    out = pairs_df.reset_index(drop=True).copy()
    lookup = all_records
    if lookup.index.name != "entity_id":
        lookup = _record_lookup(all_records, np.concatenate(
            [out["query_entity_id"].to_numpy(), out["candidate_s1_id"].to_numpy()]))
    q = lookup.reindex(out["query_entity_id"].to_numpy())
    c = lookup.reindex(out["candidate_s1_id"].to_numpy())

    def texts(frame, col):
        return frame[col].fillna("").astype(object).tolist()

    name_q, name_c = texts(q, "norm_name"), texts(c, "norm_name")
    addr_q, addr_c = texts(q, "norm_address"), texts(c, "norm_address")

    len_q = np.fromiter(map(len, name_q), dtype=np.float64, count=len(name_q))
    len_c = np.fromiter(map(len, name_c), dtype=np.float64, count=len(name_c))
    names_ok = (len_q > 0) & (len_c > 0)
    addr_empty_q = np.array([not s for s in addr_q], dtype=bool)
    addr_empty_c = np.array([not s for s in addr_c], dtype=bool)
    addrs_ok = ~addr_empty_q & ~addr_empty_c

    # ---- name similarity ----
    out["f_name_exact"]      = (names_ok & (np.array(name_q, dtype=object) == np.array(name_c, dtype=object))).astype(np.int8)
    out["f_name_ratio"]      = _sim(name_q, name_c, fuzz.ratio, names_ok)
    out["f_name_partial"]    = _sim(name_q, name_c, fuzz.partial_ratio, names_ok)
    out["f_name_token_sort"] = _sim(name_q, name_c, fuzz.token_sort_ratio, names_ok)
    out["f_name_token_set"]  = _sim(name_q, name_c, fuzz.token_set_ratio, names_ok)
    out["f_name_jaro"]       = _sim(name_q, name_c, JaroWinkler.normalized_similarity, names_ok)

    # ---- token overlap ----
    tok_q = [get_name_tokens(s) for s in name_q]
    tok_c = [get_name_tokens(s) for s in name_c]
    n_q = np.fromiter(map(len, tok_q), dtype=np.float64, count=len(tok_q))
    n_c = np.fromiter(map(len, tok_c), dtype=np.float64, count=len(tok_c))
    inter = np.fromiter((len(a & b) for a, b in zip(tok_q, tok_c)), dtype=np.float64, count=len(tok_q))
    union = n_q + n_c - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        out["f_token_jaccard"] = np.where(union > 0, inter / union, 0.0)
        out["f_token_q_in_s1"] = np.where(n_q > 0, inter / n_q, 0.0)
        out["f_token_s1_in_q"] = np.where(n_c > 0, inter / n_c, 0.0)
    out["f_token_overlap_count"] = inter.astype(np.int16)

    # ---- address ----
    out["f_addr_ratio"]      = _sim(addr_q, addr_c, fuzz.ratio, addrs_ok)
    out["f_addr_token_set"]  = _sim(addr_q, addr_c, fuzz.token_set_ratio, addrs_ok)
    out["f_addr_empty_q"]    = addr_empty_q.astype(np.int8)
    out["f_addr_empty_s1"]   = addr_empty_c.astype(np.int8)
    out["f_addr_both_empty"] = (addr_empty_q & addr_empty_c).astype(np.int8)

    # ---- length ----
    max_len = np.maximum(len_q, len_c)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["f_name_len_diff"]  = np.where(max_len > 0, np.abs(len_q - len_c) / max_len, 0.0)
        out["f_name_len_ratio"] = np.where(names_ok, np.minimum(len_q, len_c) / max_len, 0.0)

    # ---- source / blocking ----
    src = q["source"].fillna("").astype(object).to_numpy()
    out["f_source_q"] = np.select([src == "source2", src == "source3"], [2, 3], 0).astype(np.int8)
    shared = np.zeros(len(out), dtype=np.int8)
    for k in KEY_COLS:
        if k in lookup.columns:
            kq = q[k].fillna("").astype(object).to_numpy()
            kc = c[k].fillna("").astype(object).to_numpy()
            shared += ((kq == kc) & (kq != "")).astype(np.int8)
    out["f_n_keys_shared"] = shared
    return out


def compute_features_batch(pairs_df: pd.DataFrame, all_records: pd.DataFrame,
                           batch_size: int = 50000) -> pd.DataFrame:
    """build_feature_matrix over batches of pairs_df, concatenated.

    The record lookup is built once for all batches. No joblib: cpdist already
    uses every core (workers=-1).
    """
    if pairs_df.empty:
        return build_feature_matrix(pairs_df, all_records)
    lookup = _record_lookup(all_records, np.concatenate(
        [pairs_df["query_entity_id"].to_numpy(), pairs_df["candidate_s1_id"].to_numpy()]))
    parts = [
        build_feature_matrix(pairs_df.iloc[i:i + batch_size], lookup)
        for i in tqdm(range(0, len(pairs_df), batch_size), desc="Features", unit="batch")
    ]
    return pd.concat(parts, ignore_index=True)
