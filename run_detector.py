#!/usr/bin/env python3
"""Entry point: python run_detector.py --audio-dir DIR --output-dir DIR"""

import os

# Set before TensorFlow import (matches thesis pipeline).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_enable_xla_devices=false")
os.environ.setdefault("TF_DISABLE_MLIR_BRIDGE", "1")

from perch_detector.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
