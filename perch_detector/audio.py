"""Audio loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import librosa
import numpy as np
import soundfile as sf


def load_audio(filepath: Path, target_sr: int = 32_000) -> Tuple[np.ndarray, int]:
    """
    Same as pipelines/utils/data_utils.py load_audio() in the thesis repo.

    Load WAV -> mono -> resample to target_sr (32 kHz for Perch).
    Input files may be any sample rate; do not resample them beforehand.
    """
    try:
        y, sr = sf.read(str(filepath))
    except Exception as e:
        try:
            y, sr = librosa.load(str(filepath), sr=None, mono=False)
        except Exception as e2:
            raise RuntimeError(
                f"Failed to load audio from {filepath}: soundfile={e}, librosa={e2}"
            ) from e2

    if y.ndim > 1:
        y = np.mean(y, axis=1)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return y.astype(np.float32), sr
