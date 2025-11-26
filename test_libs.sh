#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export UV_VENV_CLEAR=1

VENV_PATH="$SCRIPT_DIR/.venv-libs"
PY_BIN="$VENV_PATH/bin/python"

uv venv "$VENV_PATH"
uv pip install --python "$PY_BIN" -e "$SCRIPT_DIR/python[dev,lib-support]"

cd "$SCRIPT_DIR/python"
"$PY_BIN" -m pytest tests/test_lib_support.py -v
