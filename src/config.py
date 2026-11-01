"""Central configuration. Nothing here depends on the data being present."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SUBMISSION_DIR = ROOT / "submissions"

TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_CSV = DATA_DIR / "sample_submission.csv"

ID_COL = "id"
TARGET_COL = "is_fraud"

SEED = 42
N_FOLDS = 5
