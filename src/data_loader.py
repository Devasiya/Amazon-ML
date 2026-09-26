"""Loading of source TSVs and ground truth. Reads local files only."""
import csv
from typing import Dict, List

import pandas as pd

from . import config

# pyarrow-backed strings use far less memory than Python objects (~12M train rows).
_STR_DTYPE = "string[pyarrow]"


def _read_tsv(path) -> pd.DataFrame:
    # QUOTE_NONE: names/addresses may contain stray quote characters that must not
    # swallow tabs/newlines. keep_default_na=False: literal "NA"/"null" names stay text.
    return pd.read_csv(
        path,
        sep="\t",
        dtype=_STR_DTYPE,
        quoting=csv.QUOTE_NONE,
        keep_default_na=False,
        na_values=[],
        encoding="utf-8",
    )


def load_source(source_name: str, split: str = "train") -> pd.DataFrame:
    """Load one source ("source1"/"source2"/"source3") for a split ("train"/"test")."""
    df = _read_tsv(config.source_path(source_name, split))
    missing = [c for c in config.COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{source_name}/{split} is missing columns {missing}")
    df = df[config.COLUMNS].fillna("")
    df["source"] = pd.Series(source_name, index=df.index, dtype=_STR_DTYPE)
    return df


def load_all_sources(split: str = "train") -> pd.DataFrame:
    """All three sources for a split, concatenated with a fresh index."""
    return pd.concat([load_source(s, split) for s in config.SOURCES], ignore_index=True)


def load_ground_truth() -> Dict[str, List[str]]:
    """{source1_entity_id: [matched_entity_ids]}; empty list for singletons."""
    gt = pd.read_csv(
        config.GT_FILE, sep="\t", dtype=str, quoting=csv.QUOTE_NONE,
        keep_default_na=False, na_values=[], encoding="utf-8",
    )
    result = {}
    for s1_id, raw in zip(gt[config.GT_S1_COL], gt[config.GT_MATCH_COL]):
        result[s1_id.strip()] = [x.strip() for x in raw.split(",") if x.strip()]
    return result


def get_countries(split: str = "train") -> list:
    """Sorted unique non-empty countries across all sources of a split."""
    countries = set()
    for s in config.SOURCES:
        countries.update(load_source(s, split)["country"].unique().tolist())
    countries.discard("")
    return sorted(countries)
