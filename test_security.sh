#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$(pwd)/python/src:${PYTHONPATH:-}"

# Run only the security regression tests.
exec python -m pytest python/tests/test_security.py "$@"
