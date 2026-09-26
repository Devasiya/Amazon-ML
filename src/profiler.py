"""Train-set data profile. Run: python -m src.profiler"""
import json
import time

import pandas as pd

from . import config
from .data_loader import load_all_sources, load_ground_truth

_PROFILE_FIELDS = ["business_name", "business_address", "country"]


def _empty_mask(col: pd.Series) -> pd.Series:
    """True for NaN, empty or whitespace-only values."""
    return col.isna() | (col.str.strip() == "")


def profile_dataset() -> dict:
    t0 = time.time()
    df = load_all_sources("train")
    gt = load_ground_truth()
    print(f"Loaded {len(df):,} train records and {len(gt):,} ground-truth rows "
          f"in {time.time() - t0:.1f}s\n")

    report = {"record_counts": {}, "countries": {}, "empty_rates": {}}

    # ---- Per-source stats ----
    for source, sdf in df.groupby("source", sort=True, observed=True):
        n = len(sdf)
        report["record_counts"][source] = n
        counts = sdf["country"].value_counts()
        report["countries"][source] = {
            "n_unique": int(counts.size),
            "distribution": {str(k): int(v) for k, v in counts.items()},
        }
        report["empty_rates"][source] = {
            f: round(float(_empty_mask(sdf[f]).mean()), 6) for f in _PROFILE_FIELDS
        }
    report["record_counts"]["total"] = len(df)

    # ---- Ground-truth stats ----
    sizes = [len(v) for v in gt.values()]
    n_s1 = len(sizes)
    matched_sizes = [s for s in sizes if s > 0]
    n_matched = len(matched_sizes)
    s1_ids = set(df.loc[df["source"] == "source1", "entity_id"].tolist())
    report["ground_truth"] = {
        "total_s1_entities": n_s1,
        "singletons": n_s1 - n_matched,
        "singleton_rate": round((n_s1 - n_matched) / n_s1, 6) if n_s1 else 0.0,
        "matched_groups": n_matched,
        # matches per non-singleton S1 entity (excluding the S1 record itself)
        "avg_matches_per_group": round(sum(matched_sizes) / n_matched, 4) if n_matched else 0.0,
        # full cluster size = S1 record + its matches
        "avg_group_size_incl_s1": round(sum(matched_sizes) / n_matched + 1, 4) if n_matched else 0.0,
        "max_matches_per_group": max(sizes) if sizes else 0,
        "total_match_pairs": sum(sizes),
        "gt_s1_ids_not_in_source1": len(set(gt) - s1_ids),
        "source1_ids_not_in_gt": len(s1_ids - set(gt)),
    }
    del s1_ids

    # ---- Cross-country check on every (S1, match) pair ----
    pairs = pd.DataFrame(
        [(s1, m) for s1, ms in gt.items() for m in ms], columns=["s1_id", "match_id"]
    )
    country_of = pd.Series(df["country"].to_numpy(), index=df["entity_id"].to_numpy())
    country_of = country_of[~country_of.index.duplicated()]
    pairs["s1_country"] = pairs["s1_id"].map(country_of)
    pairs["match_country"] = pairs["match_id"].map(country_of)
    unknown_ids = pairs["match_id"][pairs["match_country"].isna()].nunique()
    known = pairs.dropna(subset=["s1_country", "match_country"])
    cross = known[known["s1_country"] != known["match_country"]]
    report["cross_country_check"] = {
        "pairs_checked": len(known),
        "match_ids_not_found_in_sources": int(unknown_ids),
        "cross_country_pairs": len(cross),
        "examples": cross.head(10).to_dict("records"),
    }

    # ---- Print ----
    print("=== Record counts ===")
    for k, v in report["record_counts"].items():
        print(f"  {k:8s} {v:>12,}")
    print("\n=== Countries per source ===")
    for s, c in report["countries"].items():
        dist = ", ".join(f"{k}={v:,}" for k, v in c["distribution"].items())
        print(f"  {s}: {c['n_unique']} unique -> {dist}")
    print("\n=== Empty/null rates ===")
    for s, r in report["empty_rates"].items():
        print(f"  {s}: " + ", ".join(f"{f}={v:.4%}" for f, v in r.items()))
    print("\n=== Ground truth ===")
    for k, v in report["ground_truth"].items():
        print(f"  {k:28s} {v:,}" if isinstance(v, int) else f"  {k:28s} {v}")
    print("\n=== Cross-country check ===")
    cc = report["cross_country_check"]
    print(f"  pairs checked:                  {cc['pairs_checked']:,}")
    print(f"  match ids not found in sources: {cc['match_ids_not_found_in_sources']:,}")
    if cc["cross_country_pairs"]:
        print(f"  WARNING: {cc['cross_country_pairs']:,} cross-country pairs found in "
              f"ground truth! Examples:")
        for ex in cc["examples"]:
            print(f"    {ex}")
    else:
        print("  OK: zero cross-country pairs in ground truth")

    with open(config.DATA_PROFILE_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nSaved profile to {config.DATA_PROFILE_FILE}  ({time.time() - t0:.1f}s total)")
    return report


if __name__ == "__main__":
    profile_dataset()
