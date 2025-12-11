#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SANDBOX_ROOT="$ROOT_DIR/sandbox"
ENV_DIR="$ROOT_DIR/.test-notebooks"
INTERACTIVE=0
RUN_ALL=0
CONFIG_PATH=""

REQUIREMENTS=(papermill jupyter nbconvert ipykernel scanpy anndata matplotlib scikit-misc pyarrow torch torchvision safetensors)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --interactive) INTERACTIVE=1; shift ;;
        --all) RUN_ALL=1; shift ;;
        *) CONFIG_PATH="$1"; shift ;;
    esac
done

if [[ "$RUN_ALL" == "1" ]]; then
    echo "=========================================="
    echo "Running ALL notebook tests"
    echo "=========================================="
    OVERALL_RET=0
    for config in "$ROOT_DIR"/notebooks/*.json; do
        echo ""
        echo ">>> $config"
        if "$0" "$config"; then
            echo "<<< PASSED: $config"
        else
            echo "<<< FAILED: $config"
            OVERALL_RET=1
        fi
    done
    echo ""
    if [[ "$OVERALL_RET" == "0" ]]; then
        echo "✓ ALL NOTEBOOK SUITES PASSED"
    else
        echo "✗ SOME NOTEBOOK SUITES FAILED"
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

PARSED=$(CONFIG_PATH="$CONFIG_PATH" python3 - <<'PY'
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
    ((idx++))
done <<< "$RUN_LINES"
RUN_COUNT=${#ROLES[@]}

echo "Setting up environment..."
uv venv --quiet --allow-existing "$ENV_DIR"
uv pip install --quiet -p "$ENV_DIR/bin/python" "${REQUIREMENTS[@]}"
uv pip install --quiet -p "$ENV_DIR/bin/python" -e "$ROOT_DIR/python"
PYTHON="$ENV_DIR/bin/python"

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

echo ""
[[ "$RET" == "0" ]] && echo "✓ ALL TESTS PASSED" || echo "✗ TESTS FAILED"
exit "$RET"
