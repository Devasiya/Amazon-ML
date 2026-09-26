"""Layer 1 — text normalization for names, addresses and countries.

Country-agnostic: nothing here is keyed on a specific country, so unseen countries
(e.g. France in test) go through exactly the same path.
"""
import re

import pandas as pd
from tqdm import tqdm
from unidecode import unidecode

NAME_SUFFIXES = {
    "llc": "llc", "ltd": "limited", "inc": "incorporated", "corp": "corporation",
    "co": "company", "pvt": "private", "llp": "llp", "plc": "plc", "gmbh": "gmbh",
}

ADDRESS_ABBREVIATIONS = {
    "st": "street", "ave": "avenue", "blvd": "boulevard", "rd": "road", "dr": "drive",
    "ln": "lane", "ct": "court", "apt": "apartment", "ste": "suite", "fl": "floor",
    "hwy": "highway", "pkwy": "parkway",
}

_APOSTROPHES     = re.compile(r"['`]")                      # "Orelee's" -> "orelees"
_NAME_PUNCT      = re.compile(r"[^a-z0-9\s-]")               # everything but alnum/space/hyphen
_ADDR_PUNCT      = re.compile(r"[^a-z0-9\s,-]")              # also keep commas
_LOOSE_HYPHEN    = re.compile(r"(?<![a-z0-9])-|-(?![a-z0-9])")  # hyphen not between word chars
_SPACES          = re.compile(r"\s+")
_COMMA_SPACING   = re.compile(r"\s*,[\s,]*")                 # " , ,  " -> ", "
_ADDR_ABBR       = re.compile(r"\b(" + "|".join(ADDRESS_ABBREVIATIONS) + r")\b")
_DIGITS          = re.compile(r"\d+")
_ADDR_TOKEN_SPLIT = re.compile(r"[\s,]+")


def _is_missing(value) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):  # non-scalar
        return False


def _ascii_lower(value) -> str:
    return unidecode(str(value)).lower()


def normalize_business_name(name: str) -> str:
    if _is_missing(name) or name == "":
        return ""
    s = _ascii_lower(name)
    s = _APOSTROPHES.sub("", s)
    s = _NAME_PUNCT.sub(" ", s)
    s = _LOOSE_HYPHEN.sub(" ", s)
    tokens = s.split()
    # Expand the trailing run of legal-form suffixes ("xyz pvt ltd" -> "xyz private limited").
    i = len(tokens) - 1
    while i >= 0 and tokens[i] in NAME_SUFFIXES:
        tokens[i] = NAME_SUFFIXES[tokens[i]]
        i -= 1
    return " ".join(tokens)


def normalize_address(address: str) -> str:
    if _is_missing(address) or address == "":
        return ""
    s = _ascii_lower(address)
    s = _APOSTROPHES.sub("", s)
    s = _ADDR_PUNCT.sub(" ", s)
    s = _LOOSE_HYPHEN.sub(" ", s)
    s = _ADDR_ABBR.sub(lambda m: ADDRESS_ABBREVIATIONS[m.group(1)], s)
    s = _COMMA_SPACING.sub(", ", s)
    s = _SPACES.sub(" ", s)
    return s.strip(" ,")


def normalize_country(country: str) -> str:
    if _is_missing(country):
        return ""
    return str(country).strip().lower()


def _normalize_column(col: pd.Series, fn, desc: str) -> pd.Series:
    """Apply fn once per unique value (with a progress bar) and map back."""
    col = col.fillna("").astype(str)  # plain Python str, never "<NA>"/"nan"
    uniques = col.unique()
    mapping = {v: fn(v) for v in tqdm(uniques, desc=desc, unit="val", mininterval=2)}
    return col.map(mapping).astype("string[pyarrow]")


def normalize_dataframe(df: pd.DataFrame, desc: str = "Normalizing") -> pd.DataFrame:
    """Add norm_name / norm_address / norm_country to df in place and return it."""
    df["norm_name"]    = _normalize_column(df["business_name"], normalize_business_name, f"{desc} names")
    df["norm_address"] = _normalize_column(df["business_address"], normalize_address, f"{desc} addresses")
    df["norm_country"] = _normalize_column(df["country"], normalize_country, f"{desc} countries")
    return df


def get_name_tokens(norm_name: str) -> set:
    if _is_missing(norm_name) or not norm_name:
        return set()
    return {t for t in str(norm_name).split() if len(t) >= 2}


def get_address_tokens(norm_address: str) -> set:
    """Word tokens (len >= 2) plus every digit run (street numbers, postal codes)."""
    if _is_missing(norm_address) or not norm_address:
        return set()
    s = str(norm_address)
    tokens = {t for t in _ADDR_TOKEN_SPLIT.split(s) if len(t) >= 2}
    tokens.update(_DIGITS.findall(s))  # "570-13" -> "570", "13"
    return tokens
