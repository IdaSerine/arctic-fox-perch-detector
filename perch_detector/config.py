"""Model and detector defaults (matches year_pipeline_1200x600/perch_standalone)."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

PERCH_HUB_URL = (
    "https://www.kaggle.com/models/google/bird-vocalization-classifier/"
    "tensorFlow2/perch_v2/2"
)

# Perch v2 embedding model sample rate (all input audio is resampled here).
PERCH_SAMPLE_RATE = 32_000
PERCH_WINDOW_SIZE_S = 5.0

DEFAULT_CHECKPOINT = REPO_ROOT / "models" / "lr_nestedcv_final_C0.01.joblib"

# Fox call classes used in the year-pipeline standalone run.
FOX_TARGET_CLASSES = [
    "fab", "fac", "facp", "fag", "fc", "fg", "fgp",
    "fsb", "ftb", "ftbp", "fw", "fwb", "fwp",
]

DEFAULT_THRESHOLD = 0.01
HIGH_CONF_THRESHOLD = 0.6
DEFAULT_THRESHOLDS = (DEFAULT_THRESHOLD, HIGH_CONF_THRESHOLD)
DEFAULT_HOP_RATIO = 0.5
DEFAULT_MERGE_IOU = 0.3
DEFAULT_CONFIDENCE_MODE = "fox_vs_noise_winner"

DETECTION_CSV_COLUMNS = [
    "start_time",
    "start_time_formatted",
    "end_time",
    "end_time_formatted",
    "class",
    "confidence",
    "num_windows",
    "file",
    "threshold",
]
