#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/python"

echo "==> Installing Python 3.8..."
uv python install 3.8

echo "==> Syncing dependencies (dev + format extras)..."
uv sync --extra dev --extra format

echo "==> Running tests..."
uv run pytest

echo "==> Building package..."
uv build

echo "==> Done! Package built in python/dist/"
ls -la dist/
