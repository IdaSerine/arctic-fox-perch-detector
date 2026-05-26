#!/usr/bin/env python3
"""Command-line interface for the arctic fox Perch detector."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from perch_detector.config import (
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIDENCE_MODE,
    DEFAULT_HOP_RATIO,
    DEFAULT_MERGE_IOU,
    DEFAULT_THRESHOLD,
    DEFAULT_THRESHOLDS,
    FOX_TARGET_CLASSES,
)
from perch_detector.core import discover_wav_files, run_detection


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run the arctic fox Perch sliding-window detector on WAV files. "
            "Writes one CSV label file per input WAV plus a combined all_detections.csv."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--audio-dir",
        type=Path,
        required=True,
        help="Directory containing input .wav/.WAV files (searched recursively)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for CSV outputs (all_detections.csv and labels/<stem>_labels.csv)",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Path to trained logistic-regression checkpoint (.joblib)",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Primary detection confidence threshold (writes all_detections.csv)",
    )
    p.add_argument(
        "--thresholds",
        type=str,
        default=None,
        help=(
            "Comma-separated confidence thresholds. "
            f"Default: {DEFAULT_THRESHOLD} and 0.6 (two output files)"
        ),
    )
    p.add_argument("--hop-ratio", type=float, default=DEFAULT_HOP_RATIO)
    p.add_argument("--merge-iou", type=float, default=DEFAULT_MERGE_IOU)
    p.add_argument(
        "--confidence-mode",
        choices=["fox_vs_noise_winner", "target_max"],
        default=DEFAULT_CONFIDENCE_MODE,
    )
    p.add_argument(
        "--no-filter-noise",
        action="store_true",
        help="Keep detections labeled as noise (default: filter noise out)",
    )
    p.add_argument(
        "--target-classes",
        type=str,
        default=",".join(FOX_TARGET_CLASSES),
        help="Comma-separated fox class codes",
    )
    p.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only scan top-level of --audio-dir (default: recursive)",
    )
    p.add_argument(
        "--no-per-file-csv",
        action="store_true",
        help="Only write all_detections.csv (skip per-file labels/)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.audio_dir.is_dir():
        print(f"ERROR: --audio-dir not found: {args.audio_dir}", file=sys.stderr)
        return 1
    if not args.checkpoint.exists():
        print(f"ERROR: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 1

    wav_files = discover_wav_files(args.audio_dir, recursive=not args.no_recursive)
    if not wav_files:
        print(f"ERROR: no WAV files found under {args.audio_dir}", file=sys.stderr)
        return 1

    target_classes = [c.strip() for c in args.target_classes.split(",") if c.strip()]

    print(f"Found {len(wav_files)} WAV file(s)")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output:     {args.output_dir}")
    if args.thresholds:
        thresholds = [float(t.strip()) for t in args.thresholds.split(",") if t.strip()]
    else:
        # Primary threshold from --threshold, plus 0.6 unless already included.
        thresholds = [args.threshold]
        if not any(abs(t - 0.6) < 1e-9 for t in thresholds):
            thresholds.append(0.6)

    print(f"Thresholds: {thresholds}  hop_ratio: {args.hop_ratio}  merge_iou: {args.merge_iou}")

    combined_paths = run_detection(
        wav_files=wav_files,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        thresholds=thresholds,
        hop_ratio=args.hop_ratio,
        merge_iou=args.merge_iou,
        filter_noise=not args.no_filter_noise,
        confidence_mode=args.confidence_mode,
        target_classes=target_classes,
        device=args.device,
        gpu_index=args.gpu_index,
        per_file_csv=not args.no_per_file_csv,
    )

    meta = {
        "timestamp": datetime.now().isoformat(),
        "audio_dir": str(args.audio_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "n_files": len(wav_files),
        "thresholds": thresholds,
        "hop_ratio": args.hop_ratio,
        "merge_iou": args.merge_iou,
        "confidence_mode": args.confidence_mode,
        "filter_noise": not args.no_filter_noise,
        "combined_csvs": {str(k): str(v) for k, v in combined_paths.items()},
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    print("\nDone. Combined detection files:")
    for thr, path in combined_paths.items():
        print(f"  threshold {thr}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
