#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Setup virtual environment and install dependencies
uv venv --quiet --allow-existing
uv pip install --quiet -e ./python
uv pip install --quiet pytest RestrictedPython

# Run only the security regression tests.
exec uv run pytest python/tests/test_security.py -v "$@"
