#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/python"

# Use UV_PYTHON if set (from CI), otherwise let uv pick
PYTHON_ARG=""
if [[ -n "$UV_PYTHON" ]]; then
    PYTHON_ARG="--python $UV_PYTHON"
fi

# Sync dependencies and run all pytest tests
uv sync $PYTHON_ARG --extra dev

# Force install pyfory x86_64 wheel on macOS Intel (universal wheel doesn't work)
if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "x86_64" ]]; then
    echo "Detected macOS Intel - force installing pyfory x86_64 wheel..."
    uv pip uninstall pyfory || true
    uv pip install \
        https://files.pythonhosted.org/packages/35/c5/b2de2a2dc0d2b74002924cdd46a6e6d3bccc5380181ca0dc850855608bfe/pyfory-0.13.2-cp312-cp312-macosx_10_13_x86_64.whl
fi

# Use venv python directly to avoid uv run re-syncing deps
.venv/bin/python -m pytest -v
