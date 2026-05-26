"""Classifier checkpoint loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import joblib


def load_checkpoint(path: Path) -> Dict[str, Any]:
    """Load joblib checkpoint with classifier, label_encoder, scaler, metadata."""
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return joblib.load(path)
