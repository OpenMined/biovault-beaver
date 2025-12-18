#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export UV_VENV_CLEAR=1

VENV_PATH="$SCRIPT_DIR/.venv-libs"
PY_BIN="$VENV_PATH/bin/python"

uv venv "$VENV_PATH"
uv pip install --python "$PY_BIN" -e "$SCRIPT_DIR/python[dev,lib-support]"

# Force install pyfory x86_64 wheel on macOS Intel (universal wheel doesn't work)
if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "x86_64" ]]; then
    echo "Detected macOS Intel - force installing pyfory x86_64 wheel..."
    uv pip uninstall --python "$PY_BIN" pyfory || true
    uv pip install --python "$PY_BIN" \
        https://files.pythonhosted.org/packages/35/c5/b2de2a2dc0d2b74002924cdd46a6e6d3bccc5380181ca0dc850855608bfe/pyfory-0.13.2-cp312-cp312-macosx_10_13_x86_64.whl
fi

cd "$SCRIPT_DIR/python"
"$PY_BIN" -m pytest tests/test_lib_support.py -v
