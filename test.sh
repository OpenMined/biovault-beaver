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
    echo "=== macOS Intel detected ==="
    echo "Before pyfory fix:"
    uv pip show pyfory || echo "pyfory not installed"
    .venv/bin/python -c "import pyfory; print('pyfory import OK')" 2>&1 || echo "pyfory import FAILED"

    echo "Uninstalling pyfory..."
    uv pip uninstall pyfory || true

    echo "Installing pyfory x86_64 wheel..."
    uv pip install \
        https://files.pythonhosted.org/packages/35/c5/b2de2a2dc0d2b74002924cdd46a6e6d3bccc5380181ca0dc850855608bfe/pyfory-0.13.2-cp312-cp312-macosx_10_13_x86_64.whl

    echo "After pyfory fix:"
    uv pip show pyfory
    .venv/bin/python -c "import pyfory; print('pyfory import OK')" || echo "pyfory import STILL FAILED"
    echo "=== End pyfory fix ==="
fi

# Show installed packages for debugging
echo "=== Installed packages ==="
.venv/bin/pip list
echo "=========================="

# Use venv python directly to avoid uv run re-syncing deps
.venv/bin/python -m pytest -v
