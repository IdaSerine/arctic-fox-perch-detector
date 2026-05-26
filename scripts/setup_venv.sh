#!/usr/bin/env bash
# Create a Python 3.11 virtual environment and install dependencies.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python3.11}"
if ! command -v "$PYTHON" &>/dev/null; then
  PYTHON=python3
fi

"$PYTHON" -m venv .venv
# shellcheck source=/dev/null
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Virtual environment ready. Activate with:"
echo "  source $REPO_ROOT/.venv/bin/activate"
