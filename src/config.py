"""Central configuration: every path, constant and tunable lives here."""
from pathlib import Path

# ---- Paths (all relative to the project root, auto-detected) ----
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
DATA_DIR      = PROJECT_ROOT / "data"
TRAIN_DIR     = DATA_DIR / "train"
TEST_DIR      = DATA_DIR / "test"
OUTPUT_DIR    = PROJECT_ROOT / "output"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR    = PROJECT_ROOT / "models"

# ---- Data schema ----
SOURCES = ["source1", "source2", "source3"]
COLUMNS = ["entity_id", "business_name", "business_address", "country"]
SPLITS  = ["train", "test"]

GT_FILE      = TRAIN_DIR / "train_ground_truth.tsv"
GT_S1_COL    = "source1_entity_id"
GT_MATCH_COL = "matched_entity_ids"

# ---- Output / artifact files ----
MATCHING_RESULTS_FILE = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_PAIRS_FILE  = OUTPUT_DIR / "candidate_pairs.tsv"
DATA_PROFILE_FILE     = ARTIFACTS_DIR / "data_profile.json"

# ---- Metric / reproducibility ----
F05_BETA    = 0.5
RANDOM_SEED = 42

# ---- Blocking ----
BLOCKING_TOPK = 100  # max candidates per query
BLOCKING_MAX_BUCKET_SCAN = 1000  # ids scanned per bucket (lists are pre-shuffled)

# ---- Decision ----
MATCH_THRESHOLD = 0.5  # default, will be tuned later


def source_path(source_name: str, split: str = "train") -> Path:
    """Path of a source TSV, e.g. ("source2", "test") -> data/test/test_source2.tsv."""
    if source_name not in SOURCES:
        raise ValueError(f"unknown source {source_name!r}; expected one of {SOURCES}")
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    base = TRAIN_DIR if split == "train" else TEST_DIR
    return base / f"{split}_{source_name}.tsv"


for _d in [DATA_DIR, TRAIN_DIR, TEST_DIR, OUTPUT_DIR, ARTIFACTS_DIR, MODELS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)
