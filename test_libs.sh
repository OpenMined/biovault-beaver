#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export UV_VENV_CLEAR=1

VENV_PATH="$SCRIPT_DIR/.venv-libs"
PY_BIN="$VENV_PATH/bin/python"

uv venv "$VENV_PATH"
uv pip install --python "$PY_BIN" -e "$SCRIPT_DIR/python[dev,lib-support]"

# Override versions via env vars (e.g., for Intel Mac compatibility)
if [[ -n "${TORCH_VERSION:-}" ]]; then
    echo "Using torch==$TORCH_VERSION from TORCH_VERSION env var"
    uv pip install --python "$PY_BIN" "torch==$TORCH_VERSION"
fi
if [[ -n "${NUMPY_SPEC:-}" ]]; then
    echo "Using numpy$NUMPY_SPEC from NUMPY_SPEC env var"
    uv pip install --python "$PY_BIN" "numpy$NUMPY_SPEC"
fi

# Force install pyfory x86_64 wheel on macOS Intel (universal wheel doesn't work)
if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "x86_64" ]]; then
    echo "=== macOS Intel detected ==="

    echo "Before pyfory fix:"
    uv pip show --python "$PY_BIN" pyfory || echo "pyfory not installed"
    "$PY_BIN" -c "import pyfory; print('pyfory import OK')" 2>&1 || echo "pyfory import FAILED"

    echo "Uninstalling pyfory..."
    uv pip uninstall --python "$PY_BIN" pyfory || true

    echo "Installing pyfory x86_64 wheel..."
    uv pip install --python "$PY_BIN" \
        https://files.pythonhosted.org/packages/35/c5/b2de2a2dc0d2b74002924cdd46a6e6d3bccc5380181ca0dc850855608bfe/pyfory-0.13.2-cp312-cp312-macosx_10_13_x86_64.whl

    echo "After pyfory fix:"
    uv pip show --python "$PY_BIN" pyfory
    "$PY_BIN" -c "import pyfory; print('pyfory import OK')" || echo "pyfory import STILL FAILED"
    echo "=== End pyfory fix ==="
fi

cd "$SCRIPT_DIR/python"

# Show installed packages for debugging
echo "=== Installed packages ==="
"$PY_BIN" -m pip list
echo "=========================="

"$PY_BIN" -m pytest tests/test_lib_support.py -v
