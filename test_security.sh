#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Setup virtual environment and install dependencies
uv venv --quiet --allow-existing
uv pip install --quiet -e ./python
uv pip install --quiet pytest RestrictedPython

# Force install pyfory x86_64 wheel on macOS Intel (universal wheel doesn't work)
if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "x86_64" ]]; then
    echo "Detected macOS Intel - force installing pyfory x86_64 wheel..."
    uv pip uninstall --quiet pyfory || true
    uv pip install --quiet \
        https://files.pythonhosted.org/packages/35/c5/b2de2a2dc0d2b74002924cdd46a6e6d3bccc5380181ca0dc850855608bfe/pyfory-0.13.2-cp312-cp312-macosx_10_13_x86_64.whl
fi

# Run only the security regression tests (use venv python to avoid uv run re-syncing)
exec .venv/bin/python -m pytest python/tests/test_security.py -v "$@"
