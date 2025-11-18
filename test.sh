#!/bin/bash
set -e
export UV_VENV_CLEAR=1
uv venv
uv pip install -e ./python
uv pip install pytest
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/python"
uv run pytest
