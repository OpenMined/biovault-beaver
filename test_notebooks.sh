#!/usr/bin/env bash
set -euo pipefail

# Test runner for sc_test_do.ipynb and sc_test_ds.ipynb
# Sets up SyftBox sandbox structure with encryption

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SANDBOX_ROOT="$ROOT_DIR/sandbox"
PACKAGE_DIR="$ROOT_DIR/python"
NOTEBOOKS_DIR="$ROOT_DIR/notebooks"
SYFTBOX_SDK_DIR="$ROOT_DIR/syftbox-sdk/python"

# Client directories
CLIENT1_EMAIL="client1@sandbox.local"
CLIENT2_EMAIL="client2@sandbox.local"
CLIENT1_DIR="$SANDBOX_ROOT/$CLIENT1_EMAIL"
CLIENT2_DIR="$SANDBOX_ROOT/$CLIENT2_EMAIL"

echo "=========================================="
echo "Single-Cell Analysis Notebook Test"
echo "SyftBox Encrypted Mode"
echo "=========================================="

# Build syftbox-sdk first (shared across all clients)
echo ""
echo "Building syftbox-sdk..."
if [[ -d "$SYFTBOX_SDK_DIR" ]]; then
    (
        cd "$SYFTBOX_SDK_DIR"
        uv run maturin build --release 2>&1 | tail -5
    )
    SYFTBOX_WHEEL=$(find "$SYFTBOX_SDK_DIR/target/wheels" -name "*.whl" 2>/dev/null | sort -V | tail -1)
    if [[ -n "$SYFTBOX_WHEEL" ]]; then
        echo "✓ Built: $(basename $SYFTBOX_WHEEL)"
    else
        echo "✗ Failed to build syftbox-sdk wheel"
        exit 1
    fi
else
    echo "✗ syftbox-sdk not found at $SYFTBOX_SDK_DIR"
    exit 1
fi

# Clean sandbox
echo ""
echo "Cleaning sandbox directory..."
rm -rf "$SANDBOX_ROOT"
mkdir -p "$SANDBOX_ROOT"

# Setup function for each client
setup_client() {
    local client_dir="$1"
    local client_email="$2"
    local role="$3"  # do or ds
    local notebook="sc_test_${role}.ipynb"
    local venv_path="$client_dir/.venv_test"

    echo ""
    echo "[$role] Setting up $client_email..."

    # Create SyftBox directory structure
    mkdir -p "$client_dir/datasites/$client_email/shared/biovault/sessions"
    mkdir -p "$client_dir/datasites/$client_email/app_data/biovault/rpc/session"
    mkdir -p "$client_dir/datasites/$client_email/public/crypto"
    mkdir -p "$client_dir/.syc"
    mkdir -p "$client_dir/unencrypted"

    # Symlink single_cell data directory
    ln -snf "$NOTEBOOKS_DIR/single_cell" "$client_dir/single_cell"

    # Copy test notebook
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

    # Install beaver package
    uv pip install -p "$venv_path/bin/python" -q -e "$PACKAGE_DIR"

    # Install pre-built syftbox-sdk wheel
    if [[ -n "$SYFTBOX_WHEEL" ]]; then
        echo "[$role] Installing syftbox-sdk..."
        uv pip install -p "$venv_path/bin/python" -q --force-reinstall "$SYFTBOX_WHEEL"
    fi

    echo "[$role] Setup complete"
}

# Provision crypto identity
provision_identity() {
    local client_dir="$1"
    local client_email="$2"
    local venv_path="$client_dir/.venv_test"

    echo "Provisioning identity for $client_email..."
    "$venv_path/bin/python" -c "
import syftbox_sdk
result = syftbox_sdk.provision_identity(
    identity='$client_email',
    data_root='$client_dir',
)
print(f'  Identity: {result.identity}')
print(f'  Generated: {result.generated}')
print(f'  Public bundle: {result.public_bundle_path}')
"
}

# Import peer bundle
import_bundle() {
    local client_dir="$1"
    local peer_email="$2"
    local peer_dir="$3"
    local venv_path="$client_dir/.venv_test"

    echo "Importing $peer_email bundle into $(basename $client_dir)..."
    "$venv_path/bin/python" -c "
import syftbox_sdk
from pathlib import Path

peer_bundle = Path('$peer_dir/datasites/$peer_email/public/crypto/did.json')
if peer_bundle.exists():
    result = syftbox_sdk.import_bundle(
        bundle_path=str(peer_bundle),
        vault_path='$client_dir/.syc',
        expected_identity='$peer_email',
    )
    print(f'  Imported: {result}')
else:
    print(f'  Bundle not found: {peer_bundle}')
"
}

# Create peer datasite view (real directories, not symlinks)
create_peer_view() {
    local client_dir="$1"
    local peer_email="$2"
    local peer_dir="$3"

    # Create real directories for peer's datasite (SyftBox would sync these)
    mkdir -p "$client_dir/datasites/$peer_email/shared"
    mkdir -p "$client_dir/datasites/$peer_email/app_data"
    mkdir -p "$client_dir/datasites/$peer_email/public/crypto"

    # Copy peer's public crypto bundle
    if [[ -f "$peer_dir/datasites/$peer_email/public/crypto/did.json" ]]; then
        cp "$peer_dir/datasites/$peer_email/public/crypto/did.json" \
           "$client_dir/datasites/$peer_email/public/crypto/did.json"
    fi
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

    # Execute notebook using papermill
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
setup_client "$CLIENT1_DIR" "$CLIENT1_EMAIL" "do"
setup_client "$CLIENT2_DIR" "$CLIENT2_EMAIL" "ds"

# Provision identities
echo ""
echo "=========================================="
echo "Provisioning crypto identities..."
echo "=========================================="
provision_identity "$CLIENT1_DIR" "$CLIENT1_EMAIL"
provision_identity "$CLIENT2_DIR" "$CLIENT2_EMAIL"

# Create peer views (symlink each other's datasites)
echo ""
echo "Creating peer datasite views..."
create_peer_view "$CLIENT1_DIR" "$CLIENT2_EMAIL" "$CLIENT2_DIR"
create_peer_view "$CLIENT2_DIR" "$CLIENT1_EMAIL" "$CLIENT1_DIR"

# Import bundles
echo ""
echo "=========================================="
echo "Importing crypto bundles..."
echo "=========================================="
import_bundle "$CLIENT1_DIR" "$CLIENT2_EMAIL" "$CLIENT2_DIR"
import_bundle "$CLIENT2_DIR" "$CLIENT1_EMAIL" "$CLIENT1_DIR"

# Cleanup function
cleanup() {
    echo ""
    echo "Cleaning up..."
    jobs -p | xargs -r kill 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo ""
echo "=========================================="
echo "Running test notebooks in parallel..."
echo "=========================================="

# Run both notebooks in parallel
# DO waits for session request, DS sends it
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
