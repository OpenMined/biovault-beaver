#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/python"

uv run pytest --cov=beaver --cov-report=term-missing --cov-report=html

echo ""
echo "✓ Coverage report generated!"
echo "  View HTML report: python/htmlcov/index.html"
