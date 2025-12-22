#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SANDBOX_ROOT="$ROOT_DIR/sandbox"
ENV_DIR="$ROOT_DIR/.test-notebooks"
INTERACTIVE=0
RUN_ALL=0
CONFIG_PATH=""

REQUIREMENTS=(papermill jupyter nbconvert ipykernel anndata matplotlib scikit-misc pyarrow torch torchvision safetensors)

# Override versions via env vars (e.g., for Intel Mac compatibility)
if [[ -n "${TORCH_VERSION:-}" ]]; then
    REQUIREMENTS=("${REQUIREMENTS[@]/torch/torch==$TORCH_VERSION}")
    echo "Using torch==$TORCH_VERSION from TORCH_VERSION env var"
fi
if [[ -n "${NUMPY_SPEC:-}" ]]; then
    REQUIREMENTS+=("numpy$NUMPY_SPEC")
    echo "Using numpy$NUMPY_SPEC from NUMPY_SPEC env var"
fi

# Security test mode - for 00-malicious notebooks
SECURITY_TEST=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --interactive) INTERACTIVE=1; shift ;;
        --all) RUN_ALL=1; shift ;;
        --secure) SECURITY_TEST=1; shift ;;
        *) CONFIG_PATH="$1"; shift ;;
    esac
done

# Use uv run if python3 not available (e.g., in CI)
PYTHON_CMD="python3"
if ! command -v python3 &>/dev/null; then
    PYTHON_CMD="uv run python"
fi

# Auto-detect security test mode from config path
if [[ -n "$CONFIG_PATH" && "$CONFIG_PATH" == *"00-malicious"* ]]; then
    SECURITY_TEST=1
fi

# Set environment based on mode
# BEAVER_TRUSTED_POLICY removed - notebooks must explicitly use trust_loader=True kwarg
if [[ "$SECURITY_TEST" == "1" ]]; then
    # Security test mode: auto-accept prompts, but loaders require explicit trust_loader=True
    export BEAVER_AUTO_ACCEPT="${BEAVER_AUTO_ACCEPT:-1}"
    echo "Security test mode: Testing human-review security model"
else
    # Normal mode: auto-accept prompts, loaders require explicit trust_loader=True
    export BEAVER_AUTO_ACCEPT="${BEAVER_AUTO_ACCEPT:-1}"
fi

if [[ "$RUN_ALL" == "1" ]]; then
    echo "=========================================="
    echo "Running ALL notebook tests"
    echo "=========================================="
    OVERALL_RET=0
    SKIPPED=0
    for config in "$ROOT_DIR"/notebooks/*.json; do
        echo ""
        echo ">>> $config"
        # Check if test should be skipped
        if $PYTHON_CMD -c "import json; c=json.load(open('$config')); exit(0 if c.get('skip') else 1)" 2>/dev/null; then
            SKIP_REASON=$($PYTHON_CMD -c "import json; print(json.load(open('$config')).get('skip_reason', 'marked as skip'))")
            echo "<<< SKIPPED: $SKIP_REASON"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi
        if "$0" "$config"; then
            echo "<<< PASSED: $config"
        else
            echo "<<< FAILED: $config"
            OVERALL_RET=1
        fi
    done
    echo ""
    if [[ "$OVERALL_RET" == "0" ]]; then
        echo "✓ ALL NOTEBOOK SUITES PASSED ($SKIPPED skipped)"
    else
        echo "✗ SOME NOTEBOOK SUITES FAILED ($SKIPPED skipped)"
    fi
    exit "$OVERALL_RET"
fi

[[ -z "$CONFIG_PATH" ]] && { echo "Usage: $0 <config.json> [--interactive] [--all]"; exit 1; }
CONFIG_PATH="$(realpath "$CONFIG_PATH")"
[[ ! -f "$CONFIG_PATH" ]] && { echo "Config not found: $CONFIG_PATH"; exit 1; }

echo "=========================================="
echo "Notebook Test Runner"
echo "Config: $CONFIG_PATH"
echo "=========================================="

mkdir -p "$SANDBOX_ROOT"

PARSED=$(CONFIG_PATH="$CONFIG_PATH" $PYTHON_CMD - <<'PY'
import json, os
cfg = json.load(open(os.environ["CONFIG_PATH"]))
mode = cfg.get("mode", "parallel")
print(f"MODE|{mode}")
for run in cfg.get("runs", []):
    role = run.get("role", "user")
    nb = run["notebook"]
    timeout = run.get("timeout", 600)
    email = run.get("email", f"{role}@sandbox.local")
    print(f"RUN|{role}|{nb}|{timeout}|{email}")
PY
)

RUN_LINES="$(echo "$PARSED" | grep '^RUN|' || true)"
[[ -z "$RUN_LINES" ]] && { echo "No runs found in config."; exit 1; }

RUN_PARALLEL=0
[[ "$(echo "$PARSED" | grep '^MODE|')" != "MODE|sequential" ]] && RUN_PARALLEL=1

declare -a ROLES NOTEBOOKS TIMEOUTS EMAILS CLIENT_DIRS OUTPUTS
idx=0
while IFS='|' read -r _ role nb timeout email; do
    ROLES[idx]="$role"
    NOTEBOOKS[idx]="$nb"
    TIMEOUTS[idx]="$timeout"
    EMAILS[idx]="$email"
    CLIENT_DIRS[idx]="$SANDBOX_ROOT/${email}"
    OUTPUTS[idx]="$(basename "${nb%.ipynb}")_output.ipynb"
    idx=$((idx + 1))
done <<< "$RUN_LINES"
RUN_COUNT=${#ROLES[@]}

echo "Setting up environment..."
uv venv --quiet --allow-existing "$ENV_DIR"

# On macOS Intel, install deps one-by-one for debugging
if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "x86_64" ]]; then
    echo "macOS Intel: Installing dependencies one-by-one..."
    for pkg in "${REQUIREMENTS[@]}"; do
        echo "  Installing $pkg..."
        uv pip install -p "$ENV_DIR/bin/python" "$pkg"
    done
else
    uv pip install --quiet -p "$ENV_DIR/bin/python" "${REQUIREMENTS[@]}"
fi

uv pip install --quiet -p "$ENV_DIR/bin/python" -e "$ROOT_DIR/python"

# Force install pyfory x86_64 wheel on macOS Intel (universal wheel doesn't work)
if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "x86_64" ]]; then
    echo "Detected macOS Intel - force installing pyfory x86_64 wheel..."
    uv pip uninstall --quiet -p "$ENV_DIR/bin/python" pyfory || true
    uv pip install --quiet -p "$ENV_DIR/bin/python" \
        https://files.pythonhosted.org/packages/35/c5/b2de2a2dc0d2b74002924cdd46a6e6d3bccc5380181ca0dc850855608bfe/pyfory-0.13.2-cp312-cp312-macosx_10_13_x86_64.whl
fi

PYTHON="$ENV_DIR/bin/python"

# Show installed packages for debugging
echo "=== Installed packages ==="
"$PYTHON" -m pip list
echo "=========================="

SESSION_DIR="$SANDBOX_ROOT/local_session"
SESSION_ID="test_session_$(date +%s)"

setup_session() {
    local email="$1"
    local client_dir="$2"
    mkdir -p "$SESSION_DIR" "$client_dir"

    # Create session.json in client dir
    cat > "$client_dir/session.json" <<EOF
{
    "session_id": "$SESSION_ID",
    "peer": "$email",
    "role": "solo",
    "status": "active"
}
EOF
}

run_notebook() {
    local i="$1"
    local role="${ROLES[i]}"
    local email="${EMAILS[i]}"
    local client_dir="${CLIENT_DIRS[i]}"
    local nb_rel="${NOTEBOOKS[i]}"
    local nb_name="$(basename "$nb_rel")"
    local out_nb="${OUTPUTS[i]}"
    local timeout="${TIMEOUTS[i]}"

    echo ""
    echo "[$role] Running $nb_name..."
    setup_session "$email" "$client_dir"
    cp "$ROOT_DIR/$nb_rel" "$client_dir/$nb_name"
    cd "$client_dir"

    local env_vars=(
        "BEAVER_LOCAL_MODE=1"
        "BEAVER_LOCAL_SESSION_DIR=$SESSION_DIR"
        "BEAVER_LOCAL_SHARED=1"
        "BEAVER_SESSION_ID=$SESSION_ID"
        "BEAVER_USER=$email"
    )

    if [[ "$INTERACTIVE" == "1" ]]; then
        local port=$((8888 + i))
        echo "[$role] Jupyter on port $port"
        env "${env_vars[@]}" "$PYTHON" -m jupyter notebook --no-browser --port "$port" --NotebookApp.token='' &
        return 0
    fi

    if env "${env_vars[@]}" "$PYTHON" -m papermill "$nb_name" "$out_nb" \
        --log-output --progress-bar --execution-timeout "$timeout" 2>&1 \
        | sed "s/^/[$role] /"; then
        echo "[$role] ✓ PASSED"
        return 0
    else
        echo "[$role] ✗ FAILED"
        return 1
    fi
}

echo ""
echo "Running notebooks..."
PIDS=()
RET=0

for i in "${!ROLES[@]}"; do
    if [[ "$RUN_PARALLEL" == "1" ]] && (( RUN_COUNT >= 2 )); then
        run_notebook "$i" &
        PIDS+=("$!")
    else
        run_notebook "$i" || RET=1
    fi
done

for pid in "${PIDS[@]:-}"; do
    [[ -n "$pid" ]] && { wait "$pid" || RET=1; }
done

# In interactive mode, wait for user to press Ctrl+C
if [[ "$INTERACTIVE" == "1" ]]; then
    echo ""
    echo "=========================================="
    echo "Jupyter servers running. Open in browser:"
    for i in "${!ROLES[@]}"; do
        local_port=$((8888 + i))
        echo "  [${ROLES[i]}] http://localhost:$local_port"
    done
    echo ""
    echo "Press Ctrl+C to stop all servers..."
    echo "=========================================="
    # Wait forever (until Ctrl+C)
    wait
fi

echo ""
[[ "$RET" == "0" ]] && echo "✓ ALL TESTS PASSED" || echo "✗ TESTS FAILED"
exit "$RET"
