"""Writers for the two submission files, in the format utils/validate_submission.py checks.

Both files have one row per S1 entity of the split (empty list = none), with
comma-separated S2/S3 ids:
  matching_results.tsv : source1_entity_id \t matched_entity_ids
  candidate_pairs.tsv  : source1_entity_id \t candidate_entity_ids
"""
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from . import config
from .data_loader import load_source


def _resolve(path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else config.PROJECT_ROOT / path


def _s1_ids(split: str) -> List[str]:
    return load_source("source1", split)["entity_id"].astype(str).str.strip().tolist()


def _write_id_lists(path, header: str, s1_ids: List[str], lists: Dict[str, List[str]]) -> Path:
    path = _resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_nonempty = 0
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(header + "\n")
        for s1 in s1_ids:
            ids = lists.get(s1, ())
            n_nonempty += bool(ids)
            f.write(f"{s1}\t{','.join(ids)}\n")
    print(f"Wrote {path}  ({len(s1_ids):,} rows, {n_nonempty:,} non-empty)")
    return path


def write_matching_results(predictions: Dict[str, List[str]],
                           output_path: str = "output/matching_results.tsv",
                           split: str = "test", s1_ids: Optional[List[str]] = None) -> Path:
    """One row per S1 entity of the split; S1s without candidates get an empty list."""
    if s1_ids is None:
        s1_ids = _s1_ids(split)
    lists = {s1: list(dict.fromkeys(ms)) for s1, ms in predictions.items()}  # dedupe, keep order
    return _write_id_lists(output_path, "source1_entity_id\tmatched_entity_ids", s1_ids, lists)


def write_candidate_pairs(scored_pairs: pd.DataFrame,
                          output_path: str = "output/candidate_pairs.tsv",
                          split: str = "test", s1_ids: Optional[List[str]] = None) -> Path:
    """One row per S1 entity: its candidate S2/S3 ids, highest score first.

    The validator's format has no score column; per-pair scores stay in the
    scored-pairs parquet.
    """
    if s1_ids is None:
        s1_ids = _s1_ids(split)
    ordered = (scored_pairs[["candidate_s1_id", "query_entity_id", "score"]]
               .astype({"candidate_s1_id": str, "query_entity_id": str})
               .sort_values(["candidate_s1_id", "score"], ascending=[True, False])
               .drop_duplicates(["candidate_s1_id", "query_entity_id"]))
    lists = ordered.groupby("candidate_s1_id", sort=False)["query_entity_id"].agg(list).to_dict()
    return _write_id_lists(output_path, "source1_entity_id\tcandidate_entity_ids", s1_ids, lists)


def validate_outputs(matching_path: str = "output/matching_results.tsv",
                     candidate_path: str = "output/candidate_pairs.tsv",
                     test_dir: str = "data/test") -> bool:
    """Run utils/validate_submission.py; True when it passes. Test-split files only."""
    result = subprocess.run(
        [sys.executable, str(config.PROJECT_ROOT / "utils" / "validate_submission.py"),
         "--matching", str(_resolve(matching_path)),
         "--candidate", str(_resolve(candidate_path)),
         "--test-dir", str(_resolve(test_dir))],
        capture_output=True, text=True, cwd=config.PROJECT_ROOT,
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    return result.returncode == 0
