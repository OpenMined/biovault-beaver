#!/usr/bin/env bash
# lint.sh - Auto-fix + run quick tests (parallel, quiet on success)
# Usage: ./lint.sh [--check] [--test]
#   --check  Read-only mode for CI (no auto-fix)
#   --test   Also run tests (slower)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

CHECK_MODE=0
RUN_TESTS=0
for arg in "$@"; do
  case "$arg" in
    --check) CHECK_MODE=1 ;;
    --test) RUN_TESTS=1 ;;
  esac
done

# Setup venv (must be sequential)
uv venv --quiet 2>/dev/null || true
source .venv/bin/activate
uv pip install -e ./python --quiet 2>/dev/null
uv pip install pytest ruff mypy vulture --quiet 2>/dev/null

# Fix pyfory on macOS Intel (universal wheel doesn't work)
if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "x86_64" ]]; then
  uv pip uninstall pyfory --quiet 2>/dev/null || true
  uv pip install --quiet \
    https://files.pythonhosted.org/packages/35/c5/b2de2a2dc0d2b74002924cdd46a6e6d3bccc5380181ca0dc850855608bfe/pyfory-0.13.2-cp312-cp312-macosx_10_13_x86_64.whl \
    2>/dev/null || true
fi

cd "$ROOT_DIR/python"

TMPDIR_LINT=$(mktemp -d)
trap "rm -rf $TMPDIR_LINT" EXIT

FAILED=0
PIDS=()
TASKS=()

run_task() {
  local name="$1"
  local outfile="$TMPDIR_LINT/$name.out"
  shift
  TASKS+=("$name")
  echo -e "${CYAN}→ $name${NC}"
  (
    if "$@" > "$outfile" 2>&1; then
      echo "0" > "$outfile.status"
    else
      echo "1" > "$outfile.status"
    fi
  ) &
  PIDS+=($!)
}

wait_all() {
  local i=0
  for pid in "${PIDS[@]}"; do
    wait "$pid" || true
    local name="${TASKS[$i]}"
    local outfile="$TMPDIR_LINT/$name.out"
    if [[ -f "$outfile.status" && "$(cat "$outfile.status")" != "0" ]]; then
      echo -e "${RED}✗ $name${NC}"
      cat "$outfile"
      echo ""
      FAILED=1
    fi
    i=$((i + 1))
  done
}

if [[ "$CHECK_MODE" -eq 1 ]]; then
  run_task "ruff-format" ruff format . --check
  run_task "ruff-check" ruff check .
else
  run_task "ruff-format" ruff format .
  run_task "ruff-check" ruff check . --fix
fi

run_task "mypy" mypy src/beaver
run_task "vulture" vulture src --min-confidence 80

if [[ "$RUN_TESTS" -eq 1 ]]; then
  run_task "pytest" pytest
fi

wait_all

if [[ "$FAILED" -eq 0 ]]; then
  echo -e "${GREEN}✓ All checks passed${NC}"
else
  exit 1
fi
