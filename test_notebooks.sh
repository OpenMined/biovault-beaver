#!/usr/bin/env bash
set -euo pipefail

# Test runner for sc_test_do.ipynb and sc_test_ds.ipynb
# Sets up separate virtualenvs like dev.sh and runs notebooks in parallel

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SANDBOX_ROOT="$ROOT_DIR/sandbox"
PACKAGE_DIR="$ROOT_DIR/python"
SHARED_DIR="$SANDBOX_ROOT/shared"
NOTEBOOKS_DIR="$ROOT_DIR/notebooks"

# Client specs: name:role
CLIENT1_DIR="$SANDBOX_ROOT/client1@sandbox.local"
CLIENT2_DIR="$SANDBOX_ROOT/client2@sandbox.local"

echo "=========================================="
echo "Single-Cell Analysis Notebook Test"
echo "=========================================="

# Clean shared directory
echo ""
echo "Cleaning shared directory..."
rm -rf "$SHARED_DIR"
mkdir -p "$SHARED_DIR"

# Setup function for each client
setup_client() {
    local client_dir="$1"
    local role="$2"  # do or ds
    local notebook="sc_test_${role}.ipynb"
    local venv_path="$client_dir/.venv_test"  # Separate from dev .venv

    echo ""
    echo "[$role] Setting up $client_dir..."

    mkdir -p "$client_dir"

    # Symlink shared directory
    ln -snf ../shared "$client_dir/shared"

    # Symlink single_cell data directory
    ln -snf "$NOTEBOOKS_DIR/single_cell" "$client_dir/single_cell"

    # Copy test notebook (not symlink - we want to execute in place)
    # Remove first in case it's a symlink from dev.sh
    rm -f "$client_dir/$notebook"
    cp "$NOTEBOOKS_DIR/$notebook" "$client_dir/$notebook"

    # Create virtualenv if needed
    if [[ ! -f "$venv_path/bin/python" ]]; then
        echo "[$role] Creating virtual environment..."
        uv venv "$venv_path"
    fi

    # Install dependencies
    echo "[$role] Installing dependencies..."
    uv pip install -p "$venv_path/bin/python" -q \
        jupyter nbconvert ipykernel papermill \
        scanpy anndata matplotlib scikit-misc
    uv pip install -p "$venv_path/bin/python" -q -e "$PACKAGE_DIR"

    echo "[$role] Setup complete"
}

# Run notebook function
run_notebook() {
    local client_dir="$1"
    local role="$2"
    local notebook="sc_test_${role}.ipynb"
    local output_notebook="sc_test_${role}_output.ipynb"
    local venv_path="$client_dir/.venv_test"
    local log_file="$client_dir/test_${role}.log"

    cd "$client_dir"

    # Execute notebook using papermill (streams cell output)
    # --log-output shows print statements in real-time
    if "$venv_path/bin/python" -m papermill \
        "$notebook" \
        "$output_notebook" \
        --log-output \
        --progress-bar \
        --execution-timeout 300 \
        2>&1 | while IFS= read -r line; do
            echo "[$role] $line"
        done | tee "$log_file"; then
        echo ""
        echo "[$role] ✓ PASSED"
        return 0
    else
        echo ""
        echo "[$role] ✗ FAILED (see $log_file and $client_dir/$output_notebook)"
        return 1
    fi
}

# Setup both clients
setup_client "$CLIENT1_DIR" "do"
setup_client "$CLIENT2_DIR" "ds"

# Cleanup function
cleanup() {
    echo ""
    echo "Cleaning up..."
    # Kill any background jobs
    jobs -p | xargs -r kill 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo ""
echo "=========================================="
echo "Running test notebooks in parallel..."
echo "=========================================="

# Run both notebooks in parallel
# DO must start first and wait, DS signals ready
run_notebook "$CLIENT1_DIR" "do" &
DO_PID=$!

# Small delay to let DO start waiting
sleep 2

run_notebook "$CLIENT2_DIR" "ds" &
DS_PID=$!

echo ""
echo "Started processes:"
echo "  DO PID: $DO_PID"
echo "  DS PID: $DS_PID"
echo ""

# Wait for both to complete
DO_EXIT=0
DS_EXIT=0

wait $DO_PID || DO_EXIT=$?
wait $DS_PID || DS_EXIT=$?

echo ""
echo "=========================================="
if [[ $DO_EXIT -eq 0 && $DS_EXIT -eq 0 ]]; then
    echo "✓ ALL TESTS PASSED"
    echo "=========================================="
    exit 0
else
    echo "✗ TESTS FAILED"
    echo "  DO exit code: $DO_EXIT"
    echo "  DS exit code: $DS_EXIT"
    echo ""
    echo "Check logs:"
    echo "  $CLIENT1_DIR/test_do.log"
    echo "  $CLIENT2_DIR/test_ds.log"
    echo "=========================================="
    exit 1
fi
