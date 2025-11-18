#!/bin/bash
set -e

# Test beaver locally using uv
# Creates a temporary virtualenv, installs beaver, and runs the classifier

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Setting up test environment..." >&2

# Create temporary virtualenv
VENV_DIR=$(mktemp -d)
trap "rm -rf $VENV_DIR" EXIT

# Create virtualenv and install beaver
cd "$REPO_ROOT/python"
uv venv "$VENV_DIR" >&2
uv pip install --python "$VENV_DIR/bin/python" -e . >&2

echo "Running classification..." >&2
echo "" >&2

# Run with the virtualenv
cd "$REPO_ROOT/examples/apol1"
PATH="$VENV_DIR/bin:$PATH" bash process_samplesheet.sh samplesheet.csv classify_apol1.py
