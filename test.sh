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
uv run pytest -v
