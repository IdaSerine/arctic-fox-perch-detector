"""
Sliding-window Perch detector for arctic fox vocalizations.

Logic matches pipelines/detect.py from the Master Thesis project
(year_pipeline_1200x600/perch_standalone configuration).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from perch_detector.audio import load_audio
from perch_detector.checkpoint import load_checkpoint
from perch_detector.config import (
    FOX_TARGET_CLASSES,
    PERCH_HUB_URL,
    PERCH_SAMPLE_RATE,
    PERCH_WINDOW_SIZE_S,
)


def format_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:05.2f}"


def interval_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def interval_iou(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = interval_overlap(a0, a1, b0, b1)
    len_a = max(0.0, a1 - a0)
    len_b = max(0.0, b1 - b0)
    union = len_a + len_b - inter
    return inter / union if union > 0 else 0.0


def merge_overlapping_detections(
    detections: List[Dict[str, Any]],
    iou_threshold: float,
) -> List[Dict[str, Any]]:
    if not detections:
        return []

    by_class: Dict[str, List[Dict[str, Any]]] = {}
    for d in detections:
        by_class.setdefault(d["class"], []).append(d)

    merged: List[Dict[str, Any]] = []
    for cls, dets in by_class.items():
        dets = sorted(dets, key=lambda x: float(x["confidence"]), reverse=True)
        used = [False] * len(dets)

        for i, det in enumerate(dets):
            if used[i]:
                continue

            group = [det]
            used[i] = True

            for j in range(i + 1, len(dets)):
                if used[j]:
                    continue
                if interval_iou(
                    det["start_time"], det["end_time"],
                    dets[j]["start_time"], dets[j]["end_time"],
                ) >= iou_threshold:
                    group.append(dets[j])
                    used[j] = True

            start_time = float(min(g["start_time"] for g in group))
            end_time = float(max(g["end_time"] for g in group))

            merged.append({
                "start_time": start_time,
                "start_time_formatted": format_time(start_time),
                "end_time": end_time,
                "end_time_formatted": format_time(end_time),
                "class": cls,
                "confidence": float(max(g["confidence"] for g in group)),
                "num_windows": len(group),
            })

    return sorted(merged, key=lambda x: float(x["start_time"]))


def find_noise_proba_column(df: pd.DataFrame) -> Optional[str]:
    for c in ("proba_noise", "proba_n", "proba_Noise", "proba_N"):
        if c in df.columns:
            return c
    return None


def build_detections_from_windows(
    windows_df: pd.DataFrame,
    threshold: float,
    merge_iou: float,
    filter_noise: bool,
    target_classes: Optional[List[str]] = None,
    confidence_mode: str = "fox_vs_noise_winner",
) -> List[Dict[str, Any]]:
    if windows_df.empty:
        return []

    df = windows_df.copy()

    if confidence_mode not in {"target_max", "fox_vs_noise_winner"}:
        raise ValueError("confidence_mode must be target_max or fox_vs_noise_winner")

    if not target_classes:
        raise ValueError("target_classes is required")

    if confidence_mode == "fox_vs_noise_winner":
        proba_cols = [f"proba_{c}" for c in target_classes]
        missing = [c for c in proba_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing probability columns: {missing}")

        noise_col = find_noise_proba_column(df)
        if noise_col is None:
            raise ValueError("fox_vs_noise_winner requires a noise class (proba_noise)")

        target_probs = df[proba_cols].to_numpy(dtype=float)
        best_idx = np.argmax(target_probs, axis=1)
        best_fox = np.max(target_probs, axis=1)
        noise_probs = df[noise_col].to_numpy(dtype=float)
        fox_idx = np.argmax(np.stack([best_fox, noise_probs], axis=1), axis=1)

        df["pred_class"] = [
            target_classes[i] if fox_idx[j] == 0 else "noise"
            for j, i in enumerate(best_idx)
        ]
        df["confidence"] = np.where(fox_idx == 0, best_fox, noise_probs)
    else:
        proba_cols = [f"proba_{c}" for c in target_classes]
        target_probs = df[proba_cols].to_numpy(dtype=float)
        best_idx = np.argmax(target_probs, axis=1)
        df["pred_class"] = [target_classes[i] for i in best_idx]
        df["confidence"] = np.max(target_probs, axis=1)

    df = df[df["confidence"] >= float(threshold)].copy()

    if filter_noise:
        df = df[~df["pred_class"].astype(str).str.lower().isin({"noise", "n"})].copy()

    detections = []
    for _, r in df.iterrows():
        detections.append({
            "file": str(r["file"]),
            "start_time": float(r["start_time"]),
            "start_time_formatted": str(r["start_time_formatted"]),
            "end_time": float(r["end_time"]),
            "end_time_formatted": str(r["end_time_formatted"]),
            "class": str(r["pred_class"]),
            "confidence": float(r["confidence"]),
            "window_idx": int(r["window_idx"]),
        })

    if merge_iou and merge_iou > 0:
        by_file: Dict[str, List[Dict[str, Any]]] = {}
        for d in detections:
            by_file.setdefault(d["file"], []).append(d)

        merged_all = []
        for wav_file, dets in by_file.items():
            merged = merge_overlapping_detections(dets, iou_threshold=merge_iou)
            for m in merged:
                m["file"] = wav_file
            merged_all.extend(merged)
        detections = merged_all

    return sorted(detections, key=lambda x: float(x["confidence"]), reverse=True)


class AudioDetector:
    """Perch embedding + logistic regression sliding-window detector."""

    def __init__(
        self,
        checkpoint_path: Path,
        hop_ratio: float = 0.5,
        merge_iou: float = 0.3,
        filter_noise: bool = True,
    ):
        self.sample_rate = PERCH_SAMPLE_RATE
        self.window_size_s = PERCH_WINDOW_SIZE_S
        self.window_samples = int(self.window_size_s * self.sample_rate)
        self.hop_ratio = float(hop_ratio)
        self.merge_iou = float(merge_iou)
        self.filter_noise = bool(filter_noise)
        self.hop_samples = int(self.window_samples * self.hop_ratio)

        self.embedding_model = None
        self._load_classifier(checkpoint_path)

    def _load_classifier(self, checkpoint_path: Path) -> None:
        data = load_checkpoint(checkpoint_path)
        self.classifier = data["classifier"]
        self.label_encoder = data["label_encoder"]
        self.scaler = data.get("scaler")
        if self.scaler is None:
            raise ValueError("Checkpoint missing 'scaler'")

    def load_embedding_model(self) -> None:
        if self.embedding_model is not None:
            return

        import tensorflow as tf
        import tensorflow_hub as hub

        try:
            tf.config.optimizer.set_jit(False)
        except Exception:
            pass

        model = hub.load(PERCH_HUB_URL)
        self.embedding_model = model.signatures["serving_default"]

    def extract_window_embedding(self, audio_window: np.ndarray) -> np.ndarray:
        import tensorflow as tf

        t = tf.convert_to_tensor([audio_window], dtype=tf.float32)
        emb = self.embedding_model(inputs=t)["embedding"].numpy().squeeze()
        return emb

    def predict_windows(self, audio_path: Path) -> pd.DataFrame:
        self.load_embedding_model()

        audio, sr = load_audio(audio_path, target_sr=self.sample_rate)

        if len(audio) < self.window_samples:
            pad = self.window_samples - len(audio)
            pad_left = pad // 2
            pad_right = pad - pad_left
            audio = np.pad(audio, (pad_left, pad_right), mode="constant")

        num_windows = int((len(audio) - self.window_samples) / self.hop_samples) + 1
        rows: List[Dict[str, Any]] = []

        for i in tqdm(range(num_windows), desc=f"Windows ({audio_path.name})"):
            start_sample = i * self.hop_samples
            end_sample = start_sample + self.window_samples
            window = audio[start_sample:end_sample]

            if len(window) < self.window_samples:
                window = np.pad(window, (0, self.window_samples - len(window)), mode="constant")

            emb = self.extract_window_embedding(window)
            emb_scaled = self.scaler.transform([emb])
            proba = self.classifier.predict_proba(emb_scaled)[0]

            max_idx = int(np.argmax(proba))
            start_time = start_sample / self.sample_rate
            end_time = end_sample / self.sample_rate

            row = {
                "file": audio_path.name,
                "window_idx": i,
                "start_time": float(start_time),
                "end_time": float(end_time),
                "start_time_formatted": format_time(start_time),
                "end_time_formatted": format_time(end_time),
                "pred_class": str(self.label_encoder.classes_[max_idx]),
                "confidence": float(proba[max_idx]),
            }
            for ci, cls in enumerate(self.label_encoder.classes_):
                row[f"proba_{cls}"] = float(proba[ci])
            rows.append(row)

        return pd.DataFrame(rows)


def configure_device(device: str, gpu_index: int) -> None:
    device = device.lower().strip()
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif device == "gpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)


def assert_gpu_available() -> None:
    import tensorflow as tf

    if not tf.config.list_physical_devices("GPU"):
        raise RuntimeError(
            "No CUDA-capable GPU visible to TensorFlow. "
            "This detector downloads Perch v2 from TF Hub and expects a GPU. "
            "Run with --device gpu on a machine with CUDA, or set CUDA_VISIBLE_DEVICES."
        )


def discover_wav_files(audio_dir: Path, recursive: bool = True) -> List[Path]:
    if recursive:
        wavs = sorted(audio_dir.rglob("*.WAV")) + sorted(audio_dir.rglob("*.wav"))
    else:
        wavs = sorted(audio_dir.glob("*.WAV")) + sorted(audio_dir.glob("*.wav"))

    seen: set[str] = set()
    unique: List[Path] = []
    for w in wavs:
        key = str(w).lower()
        if key not in seen:
            seen.add(key)
            unique.append(w)
    return sorted(unique)


def _threshold_suffix(threshold: float) -> str:
    """Filename suffix for non-default thresholds, e.g. 0.6 -> 'conf0.60'."""
    return f"conf{threshold:.2f}"


def _combined_csv_name(threshold: float, primary_threshold: float) -> str:
    if abs(threshold - primary_threshold) < 1e-9:
        return "all_detections.csv"
    return f"all_detections_{_threshold_suffix(threshold)}.csv"


def _labels_subdir(threshold: float, primary_threshold: float) -> str:
    if abs(threshold - primary_threshold) < 1e-9:
        return "labels"
    return f"labels_{_threshold_suffix(threshold)}"


def _write_detection_outputs(
    detections_df: pd.DataFrame,
    output_dir: Path,
    threshold: float,
    primary_threshold: float,
    per_file_csv: bool,
) -> Path:
    empty_columns = [
        "start_time", "start_time_formatted", "end_time", "end_time_formatted",
        "class", "confidence", "num_windows", "file", "threshold",
    ]
    if detections_df.empty:
        detections_df = pd.DataFrame(columns=empty_columns)

    combined_csv = output_dir / _combined_csv_name(threshold, primary_threshold)
    detections_df.to_csv(combined_csv, index=False)

    if per_file_csv and not detections_df.empty:
        labels_dir = output_dir / _labels_subdir(threshold, primary_threshold)
        labels_dir.mkdir(parents=True, exist_ok=True)
        for wav_name, group in detections_df.groupby("file"):
            stem = Path(str(wav_name)).stem
            out_path = labels_dir / f"{stem}_labels.csv"
            group.sort_values("start_time").to_csv(out_path, index=False)

    return combined_csv


def run_detection(
    wav_files: List[Path],
    checkpoint_path: Path,
    output_dir: Path,
    thresholds: Optional[List[float]] = None,
    hop_ratio: float = 0.5,
    merge_iou: float = 0.3,
    filter_noise: bool = True,
    confidence_mode: str = "fox_vs_noise_winner",
    target_classes: Optional[List[str]] = None,
    device: str = "gpu",
    gpu_index: int = 0,
    per_file_csv: bool = True,
) -> Dict[float, Path]:
    """
    Run detector on WAV files and write CSV label outputs for each threshold.

    Returns mapping threshold -> combined CSV path.
    """
    from perch_detector.config import DEFAULT_THRESHOLD, DEFAULT_THRESHOLDS

    if target_classes is None:
        target_classes = FOX_TARGET_CLASSES
    if thresholds is None:
        thresholds = list(DEFAULT_THRESHOLDS)

    primary_threshold = thresholds[0]
    configure_device(device, gpu_index)
    assert_gpu_available()

    output_dir.mkdir(parents=True, exist_ok=True)

    detector = AudioDetector(
        checkpoint_path=checkpoint_path,
        hop_ratio=hop_ratio,
        merge_iou=merge_iou,
        filter_noise=filter_noise,
    )

    all_windows: List[pd.DataFrame] = []
    for wav in wav_files:
        print(f"\nProcessing {wav.name}")
        windows_df = detector.predict_windows(wav)
        if not windows_df.empty:
            all_windows.append(windows_df)

    if not all_windows:
        raise RuntimeError("No window predictions were produced.")

    windows_all_df = pd.concat(all_windows, ignore_index=True)

    outputs: Dict[float, Path] = {}
    for threshold in thresholds:
        print(f"\nBuilding detections at threshold {threshold} ...")
        detections = build_detections_from_windows(
            windows_df=windows_all_df,
            threshold=threshold,
            merge_iou=merge_iou,
            filter_noise=filter_noise,
            target_classes=target_classes,
            confidence_mode=confidence_mode,
        )
        for d in detections:
            d["threshold"] = float(threshold)

        detections_df = pd.DataFrame(detections)
        combined_csv = _write_detection_outputs(
            detections_df, output_dir, threshold, primary_threshold, per_file_csv,
        )
        outputs[threshold] = combined_csv
        print(f"  {len(detections_df)} detections -> {combined_csv.name}")

    return outputs
