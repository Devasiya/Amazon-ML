"""Layer 2 — blocking: for each S2/S3 record, find candidate S1 entities.

Every key is prefixed with the normalized country, so candidates never cross
countries (there are zero cross-country matches in the ground truth). Nothing is
keyed on a specific country, so unseen countries (e.g. France in test) just work.
Records with an empty address still get the three name-based keys.
"""
import heapq
import random
from collections import defaultdict
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from . import config
from .data_loader import load_all_sources, load_ground_truth
from .normalization import normalize_dataframe

KEY_COLS = ["key_prefix3", "key_prefix5", "key_firsttoken", "key_token2",
            "key_bigram", "key_name_len_prefix", "key_postcode"]
S1_PREFIX = "S1-"


def _with_country(country: pd.Series, part: pd.Series) -> pd.Series:
    """country||part, or "" when part is empty (so it is skipped by the index)."""
    part = part.fillna("")
    key = country.fillna("") + "||" + part
    return key.where(part != "", "")


def make_blocking_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Add the blocking-key columns (KEY_COLS) to df in place and return it."""
    # Stay on the pyarrow string dtype: .astype(str) would build ~12GB numpy arrays.
    name    = df["norm_name"].fillna("")
    address = df["norm_address"].fillna("")
    country = df["norm_country"].fillna("")

    # norm_name tokens are single-space separated (see normalize_business_name).
    df["key_prefix3"]    = _with_country(country, name.str[:3])
    df["key_prefix5"]    = _with_country(country, name.str[:5])
    df["key_firsttoken"] = _with_country(country, name.str.extract(r"^(\S+)", expand=False))
    df["key_token2"]     = _with_country(country, name.str.extract(r"^\S+ (\S+)", expand=False))
    df["key_bigram"]     = _with_country(country, name.str.extract(r"^(\S+(?: \S+)?)", expand=False))
    len_bin = (name.str.len() // 5).astype("string[pyarrow]")
    df["key_name_len_prefix"] = _with_country(
        country, (len_bin + "||" + name.str[:4]).where(name != "", ""))
    # Postal code: last all-digit token of 4+ digits (shorter ones are street numbers).
    # Never falls back to a word token, so city/state names cannot become a key.
    df["key_postcode"]   = _with_country(
        country, address.str.extract(r".*(?<![\w-])(\d{4,})(?![\w-])", expand=False))
    return df


def build_inverted_index(s1_df: pd.DataFrame, key_col: str) -> Dict[str, List[str]]:
    """{key: [S1 entity_ids]} for one key column, skipping empty keys.

    Each list is shuffled once (seeded); that order breaks score ties in lookups.
    """
    index = defaultdict(list)
    for key, eid in zip(s1_df[key_col].tolist(), s1_df["entity_id"].tolist()):
        if key:
            index[key].append(eid)
    rng = random.Random(config.RANDOM_SEED)
    for ids in index.values():
        ids.sort()
        rng.shuffle(ids)
    return dict(index)


def _candidates_from_keys(keys, indexes: dict, key_cols: list, topk: int) -> set:
    """Top-topk S1 ids by sum over shared keys of 1/bucket_size.

    A hit in a rare bucket is a strong signal; S1 ids sharing several keys add up.
    Ties keep the seeded pre-shuffled bucket order, so the result is reproducible.
    Only the first BLOCKING_MAX_BUCKET_SCAN ids of a bucket are scanned (a random
    sample, since lists are pre-shuffled): huge buckets carry ~0 weight each, and
    scanning them in full made blocking ~9ms per query.
    """
    scores = {}
    for key, col in zip(keys, key_cols):
        if not key:
            continue
        bucket = indexes[col].get(key)
        if not bucket:
            continue
        w = 1.0 / len(bucket)
        for eid in bucket[:config.BLOCKING_MAX_BUCKET_SCAN]:
            scores[eid] = scores.get(eid, 0.0) + w
    if len(scores) <= topk:
        return set(scores)
    return set(heapq.nlargest(topk, scores, key=scores.__getitem__))


def get_candidates_for_record(record: pd.Series, indexes: dict, key_cols: list,
                              topk: int = config.BLOCKING_TOPK) -> set:
    """S1 ids sharing any blocking key with record, ranked and capped at topk."""
    return _candidates_from_keys([record[c] for c in key_cols], indexes, key_cols, topk)


def block_queries(query_df: pd.DataFrame, indexes: dict,
                  topk: int = config.BLOCKING_TOPK) -> pd.DataFrame:
    """Candidate pairs (query_entity_id, candidate_s1_id, query_country) for query_df."""
    # Many queries share the same key combination; compute each combination once.
    cache = {}
    q_ids, c_ids, q_countries = [], [], []
    rows = zip(query_df["entity_id"].tolist(), query_df["norm_country"].tolist(),
               *(query_df[k].tolist() for k in KEY_COLS))
    for eid, country, *keys in tqdm(rows, total=len(query_df), desc="Blocking", mininterval=2):
        keys = tuple(keys)
        cands = cache.get(keys)
        if cands is None:
            cands = cache[keys] = _candidates_from_keys(keys, indexes, KEY_COLS, topk)
        q_ids.extend([eid] * len(cands))
        c_ids.extend(cands)
        q_countries.extend([country] * len(cands))

    return pd.DataFrame({
        "query_entity_id": pd.Series(q_ids, dtype="string[pyarrow]"),
        "candidate_s1_id": pd.Series(c_ids, dtype="string[pyarrow]"),
        "query_country":   pd.Series(q_countries, dtype="string[pyarrow]"),
    })


def run_blocking(split: str = "train", topk: int = config.BLOCKING_TOPK,
                 max_queries: Optional[int] = None) -> pd.DataFrame:
    """Candidate pairs (query_entity_id, candidate_s1_id, query_country).

    max_queries limits the number of S2/S3 records blocked (the full train split
    can yield up to ~1B pairs at topk=100).
    """
    df = load_all_sources(split)
    df = normalize_dataframe(df)
    df = make_blocking_keys(df)

    is_s1   = df["entity_id"].str.startswith(S1_PREFIX)
    s1_df   = df[is_s1]
    s2s3_df = df[~is_s1]
    if max_queries is not None:
        s2s3_df = s2s3_df.head(max_queries)

    indexes = {k: build_inverted_index(s1_df, k) for k in tqdm(KEY_COLS, desc="Indexing S1")}
    candidate_df = block_queries(s2s3_df, indexes, topk)

    n_queries = len(s2s3_df)
    n_blocked = candidate_df["query_entity_id"].nunique()
    print(f"Total candidate pairs:        {len(candidate_df):,}")
    print(f"S2/S3 records blocked:        {n_queries:,} ({n_queries - n_blocked:,} with no candidates)")
    print(f"Avg candidates per record:    {len(candidate_df) / max(n_queries, 1):.1f}")

    if split == "train":
        queries = pd.Series(s2s3_df["norm_country"].to_numpy(), index=s2s3_df["entity_id"].to_numpy())
        recall = compute_blocking_recall(candidate_df, load_ground_truth(), queries=queries)
        print(f"Blocking recall (train):      {recall:.4f}")
    return candidate_df


def compute_blocking_recall(candidate_df: pd.DataFrame, gt: dict,
                            queries: Optional[pd.Series] = None) -> float:
    """Fraction of true (S1, S2/S3) match pairs present in candidate_df.

    queries: optional Series {S2/S3 entity_id: country} of the records that were
    blocked; only true pairs whose S2/S3 side is in it are counted, and it gives
    the country of queries that got no candidates. Defaults to the queries present
    in candidate_df (so queries with zero candidates are then left out).
    """
    gt_pairs = pd.DataFrame(
        [(s1, m) for s1, matches in gt.items() for m in matches],
        columns=["candidate_s1_id", "query_entity_id"],
    )
    if queries is None:
        queries = (candidate_df.drop_duplicates("query_entity_id")
                   .set_index("query_entity_id")["query_country"])
    gt_pairs = gt_pairs[gt_pairs["query_entity_id"].isin(queries.index)]
    if gt_pairs.empty:
        print("No ground-truth pairs to evaluate.")
        return 0.0

    covered = gt_pairs.merge(
        candidate_df[["query_entity_id", "candidate_s1_id"]].astype(str).drop_duplicates(),
        on=["query_entity_id", "candidate_s1_id"], how="left", indicator=True,
    )
    covered["hit"] = covered["_merge"] == "both"
    covered["country"] = covered["query_entity_id"].map(queries.astype(str)).fillna("unknown")

    print("Blocking recall by country:")
    for country, grp in covered.groupby("country"):
        print(f"  {country or '(empty)':<12} {grp['hit'].mean():.4f}  ({grp['hit'].sum():,}/{len(grp):,} pairs)")
    return float(covered["hit"].mean())
