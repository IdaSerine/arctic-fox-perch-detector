"""Arctic fox vocalization detector using Perch v2 embeddings + logistic regression."""

from perch_detector.core import AudioDetector, build_detections_from_windows

__all__ = ["AudioDetector", "build_detections_from_windows"]
