#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SANDBOX_ROOT="$ROOT_DIR/sandbox"
ENV_DIR="$ROOT_DIR/.test-notebooks"
SYFTBOX_DIR="$ROOT_DIR/syftbox"
GO_CACHE_DIR="$SYFTBOX_DIR/.gocache"

INTERACTIVE=0
RUN_ALL=0
RESET_FLAG=1
SKIP_SYNC_CHECK=0
CONFIG_PATH=""
KEEP_ALIVE=0

DEFAULT_CLIENT1="${CLIENT1_EMAIL:-client1@sandbox.local}"
DEFAULT_CLIENT2="${CLIENT2_EMAIL:-client2@sandbox.local}"

REQUIREMENTS=(papermill jupyter nbconvert ipykernel scanpy anndata matplotlib scikit-misc pyarrow torch torchvision safetensors)

usage() {
    cat <<'EOF'
Usage: ./test_notebooks_syftbox.sh [OPTIONS] <config.json>

Runs notebook tests with a real SyftBox devstack (server + clients with sync).

Options:
  --interactive      Launch Jupyter servers instead of running notebooks
  --all              Run all notebook configs in notebooks/*.json
  --no-reset         Don't reset devstack state before starting
  --skip-sync-check  Skip waiting for sync probe
  --keep-alive       Keep devstack running after test completes
  -h, --help         Show this help message

Examples:
  ./test_notebooks_syftbox.sh notebooks/02-advanced-features.json
  ./test_notebooks_syftbox.sh --all
  ./test_notebooks_syftbox.sh --interactive notebooks/02-advanced-features.json
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --interactive) INTERACTIVE=1; shift ;;
        --all) RUN_ALL=1; shift ;;
        --no-reset) RESET_FLAG=0; shift ;;
        --skip-sync-check) SKIP_SYNC_CHECK=1; shift ;;
        --keep-alive) KEEP_ALIVE=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) CONFIG_PATH="$1"; shift ;;
    esac
done

PYTHON_CMD="python3"
if ! command -v python3 &>/dev/null; then
    PYTHON_CMD="uv run python"
fi

info() { printf "\033[1;36m[syftbox-test]\033[0m %s\n" "$1"; }

require_bin() {
    command -v "$1" >/dev/null 2>&1 || { echo "Missing required tool: $1" >&2; exit 1; }
}

cleanup() {
    if [[ "$KEEP_ALIVE" != "1" ]]; then
        info "Stopping devstack..."
        stop_devstack || true
    else
        info "Keep-alive mode: devstack still running at $SANDBOX_ROOT"
        info "Stop manually with: cd $SYFTBOX_DIR && GOCACHE=$GO_CACHE_DIR go run ./cmd/devstack stop --path $SANDBOX_ROOT"
    fi
}
trap cleanup EXIT INT TERM

stop_devstack() {
    if [[ -d "$SYFTBOX_DIR" ]]; then
        pkill -f "jupyter.*$SANDBOX_ROOT" 2>/dev/null || true
        (cd "$SYFTBOX_DIR" && GOCACHE="$GO_CACHE_DIR" go run ./cmd/devstack stop --path "$SANDBOX_ROOT") || true
    fi
}

start_devstack() {
    local clients=("$@")
    require_bin go
    [[ -d "$SYFTBOX_DIR" ]] || { echo "Missing syftbox checkout at $SYFTBOX_DIR" >&2; exit 1; }

    mkdir -p "$SANDBOX_ROOT"
    local args=(--path "$SANDBOX_ROOT" --random-ports)
    (( RESET_FLAG )) && args+=(--reset)
    (( SKIP_SYNC_CHECK )) && args+=(--skip-sync-check)

    for email in "${clients[@]}"; do
        args+=(--client "$email")
    done

    export GOCACHE="$GO_CACHE_DIR"

    info "Starting SyftBox devstack with clients: ${clients[*]}..."
    (cd "$SYFTBOX_DIR" && go run ./cmd/devstack start "${args[@]}")

    info "Devstack ready at $SANDBOX_ROOT"
}

provision_keys() {
    local clients=("$@")
    info "Provisioning crypto keys for clients..."

    for email in "${clients[@]}"; do
        local client_dir="$SANDBOX_ROOT/$email"
        info "  Generating keys for $email..."

        "$PYTHON" - "$client_dir" "$email" <<'PY'
import sys
from pathlib import Path
from beaver import provision_identity

data_dir = Path(sys.argv[1])
email = sys.argv[2]
vault_path = data_dir / ".syc"

result = provision_identity(
    data_dir=data_dir,
    email=email,
    vault_path=vault_path,
)

if result["generated"]:
    print(f"    ✓ Generated new identity")
else:
    print(f"    ✓ Using existing identity")
print(f"    Vault: {result['vault_path']}")
print(f"    Public bundle: {result['public_bundle_path']}")
PY
    done
    info "✓ All keys provisioned"
}

import_peer_bundles() {
    local clients=("$@")

    if [[ ${#clients[@]} -lt 2 ]]; then
        info "Single client mode, skipping peer bundle import"
        return 0
    fi

    info "Importing peer bundles..."

    for my_email in "${clients[@]}"; do
        local my_dir="$SANDBOX_ROOT/$my_email"
        local my_vault="$my_dir/.syc"

        for peer_email in "${clients[@]}"; do
            if [[ "$my_email" != "$peer_email" ]]; then
                local peer_bundle="$my_dir/datasites/$peer_email/public/crypto/did.json"
                info "  $my_email importing $peer_email's bundle..."

                "$PYTHON" - "$peer_bundle" "$my_vault" "$peer_email" <<'PY'
import sys
from beaver import import_peer_bundle

bundle_path = sys.argv[1]
vault_path = sys.argv[2]
expected_identity = sys.argv[3]

identity = import_peer_bundle(
    bundle_path=bundle_path,
    vault_path=vault_path,
    expected_identity=expected_identity,
)
print(f"    ✓ Imported {identity}")
PY
            fi
        done
    done
    info "✓ All peer bundles imported"
}

extract_clients_from_config() {
    local config="$1"
    $PYTHON_CMD - "$config" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
emails = set()
for run in cfg.get("runs", []):
    email = run.get("email", f"{run.get('role', 'user')}@sandbox.local")
    emails.add(email)
for email in sorted(emails):
    print(email)
PY
}

wait_for_file() {
    local path="$1"
    local timeout_s="${2:-60}"
    local waited=0
    while [[ "$waited" -lt "$timeout_s" ]]; do
        if [[ -s "$path" ]]; then
            return 0
        fi
        sleep 1
        waited=$((waited + 1))
    done
    return 1
}

wait_for_dir() {
    local path="$1"
    local timeout_s="${2:-60}"
    local waited=0
    while [[ "$waited" -lt "$timeout_s" ]]; do
        if [[ -d "$path" ]]; then
            return 0
        fi
        sleep 1
        waited=$((waited + 1))
    done
    return 1
}

wait_for_peer_sync() {
    local -a clients=("$@")
    local timeout_s="${DEVSTACK_SYNC_TIMEOUT:-60}"

    if [[ ${#clients[@]} -lt 2 ]]; then
        info "Single client mode, skipping peer sync check"
        return 0
    fi

    info "Waiting for peer key sync (timeout=${timeout_s}s)..."

    for client1 in "${clients[@]}"; do
        for client2 in "${clients[@]}"; do
            if [[ "$client1" != "$client2" ]]; then
                local c1_home="$SANDBOX_ROOT/$client1"
                local peer_bundle="$c1_home/datasites/$client2/public/crypto/did.json"
                wait_for_file "$peer_bundle" "$timeout_s" || {
                    echo "Timed out waiting for peer bundle: $peer_bundle" >&2
                    exit 1
                }
            fi
        done
    done
    info "✓ Peer keys synced!"
}

parse_state() {
    local field="$1"
    local email="$2"
    local state_file="$SANDBOX_ROOT/relay/state.json"
    $PYTHON_CMD - "$state_file" "$email" "$field" <<'PY'
import json, sys
state = json.load(open(sys.argv[1]))
email = sys.argv[2]
field = sys.argv[3]
for c in state.get("clients", []):
    if c.get("email") == email:
        print(c.get(field, ""))
        sys.exit(0)
sys.exit(1)
PY
}

if [[ "$RUN_ALL" == "1" ]]; then
    info "=========================================="
    info "Running ALL notebook tests with SyftBox"
    info "=========================================="

    declare -a ALL_CLIENTS
    for config in "$ROOT_DIR"/notebooks/*.json; do
        if $PYTHON_CMD -c "import json; c=json.load(open('$config')); exit(0 if c.get('skip') else 1)" 2>/dev/null; then
            continue
        fi
        while IFS= read -r email; do
            found=0
            for existing in "${ALL_CLIENTS[@]:-}"; do
                [[ "$existing" == "$email" ]] && { found=1; break; }
            done
            [[ "$found" == "0" ]] && ALL_CLIENTS+=("$email")
        done < <(extract_clients_from_config "$config")
    done

    if [[ ${#ALL_CLIENTS[@]} -eq 0 ]]; then
        ALL_CLIENTS=("$DEFAULT_CLIENT1" "$DEFAULT_CLIENT2")
    fi

    info "All clients needed: ${ALL_CLIENTS[*]}"

    info "Setting up Python environment..."
    uv venv --quiet --allow-existing "$ENV_DIR"
    uv pip install --quiet -p "$ENV_DIR/bin/python" "${REQUIREMENTS[@]}"

    SYFTBOX_SDK_PYTHON="$ROOT_DIR/syftbox-sdk/python"
    if [[ -d "$SYFTBOX_SDK_PYTHON" ]]; then
        info "Installing syftbox-sdk from local source (compiling Rust bindings)..."
        uv pip install --quiet -p "$ENV_DIR/bin/python" -e "$SYFTBOX_SDK_PYTHON" || {
            info "Warning: Failed to install local syftbox-sdk, falling back to PyPI version"
        }
    fi

    uv pip install --quiet -p "$ENV_DIR/bin/python" -e "$ROOT_DIR/python[lib-support]"

    # Force install pyfory x86_64 wheel on macOS Intel (universal wheel doesn't work)
    if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "x86_64" ]]; then
        info "Detected macOS Intel - force installing pyfory x86_64 wheel..."
        uv pip install --quiet -p "$ENV_DIR/bin/python" --force-reinstall \
            https://files.pythonhosted.org/packages/35/c5/b2de2a2dc0d2b74002924cdd46a6e6d3bccc5380181ca0dc850855608bfe/pyfory-0.13.2-cp312-cp312-macosx_10_13_x86_64.whl
    fi

    PYTHON="$ENV_DIR/bin/python"

    start_devstack "${ALL_CLIENTS[@]}"
    provision_keys "${ALL_CLIENTS[@]}"
    wait_for_peer_sync "${ALL_CLIENTS[@]}"
    import_peer_bundles "${ALL_CLIENTS[@]}"

    OVERALL_RET=0
    SKIPPED=0
    for config in "$ROOT_DIR"/notebooks/*.json; do
        info ""
        info ">>> $config"
        if $PYTHON_CMD -c "import json; c=json.load(open('$config')); exit(0 if c.get('skip') else 1)" 2>/dev/null; then
            SKIP_REASON=$($PYTHON_CMD -c "import json; print(json.load(open('$config')).get('skip_reason', 'marked as skip'))")
            info "<<< SKIPPED: $SKIP_REASON"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi

        if "$0" --no-reset --skip-sync-check --keep-alive "$config"; then
            info "<<< PASSED: $config"
        else
            info "<<< FAILED: $config"
            OVERALL_RET=1
        fi
    done

    info ""
    if [[ "$OVERALL_RET" == "0" ]]; then
        info "✓ ALL NOTEBOOK SUITES PASSED ($SKIPPED skipped)"
    else
        info "✗ SOME NOTEBOOK SUITES FAILED ($SKIPPED skipped)"
    fi
    exit "$OVERALL_RET"
fi

[[ -z "$CONFIG_PATH" ]] && { usage; exit 1; }
CONFIG_PATH="$(realpath "$CONFIG_PATH")"
[[ ! -f "$CONFIG_PATH" ]] && { echo "Config not found: $CONFIG_PATH"; exit 1; }

info "=========================================="
info "Notebook Test Runner (SyftBox Mode)"
info "Config: $CONFIG_PATH"
info "=========================================="

declare -a CONFIG_CLIENTS
while IFS= read -r email; do
    CONFIG_CLIENTS+=("$email")
done < <(extract_clients_from_config "$CONFIG_PATH")

if [[ ${#CONFIG_CLIENTS[@]} -eq 0 ]]; then
    CONFIG_CLIENTS=("$DEFAULT_CLIENT1" "$DEFAULT_CLIENT2")
fi

info "Clients from config: ${CONFIG_CLIENTS[*]}"

info "Setting up Python environment..."
uv venv --quiet --allow-existing "$ENV_DIR"
uv pip install --quiet -p "$ENV_DIR/bin/python" "${REQUIREMENTS[@]}"

SYFTBOX_SDK_PYTHON="$ROOT_DIR/syftbox-sdk/python"
if [[ -d "$SYFTBOX_SDK_PYTHON" ]]; then
    info "Installing syftbox-sdk from local source (compiling Rust bindings)..."
    uv pip install --quiet -p "$ENV_DIR/bin/python" -e "$SYFTBOX_SDK_PYTHON" || {
        info "Warning: Failed to install local syftbox-sdk, falling back to PyPI version"
    }
fi

uv pip install --quiet -p "$ENV_DIR/bin/python" -e "$ROOT_DIR/python[lib-support]"

# Force install pyfory x86_64 wheel on macOS Intel (universal wheel doesn't work)
if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "x86_64" ]]; then
    info "Detected macOS Intel - force installing pyfory x86_64 wheel..."
    uv pip install --quiet -p "$ENV_DIR/bin/python" --force-reinstall \
        https://files.pythonhosted.org/packages/35/c5/b2de2a2dc0d2b74002924cdd46a6e6d3bccc5380181ca0dc850855608bfe/pyfory-0.13.2-cp312-cp312-macosx_10_13_x86_64.whl
fi

PYTHON="$ENV_DIR/bin/python"

if [[ "$RESET_FLAG" == "1" ]] || [[ ! -f "$SANDBOX_ROOT/relay/state.json" ]]; then
    start_devstack "${CONFIG_CLIENTS[@]}"
fi

provision_keys "${CONFIG_CLIENTS[@]}"

if [[ "$SKIP_SYNC_CHECK" != "1" ]]; then
    wait_for_peer_sync "${CONFIG_CLIENTS[@]}"
    import_peer_bundles "${CONFIG_CLIENTS[@]}"
fi

SESSION_ID="syftbox_session_$(date +%s)"

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

get_peer_email() {
    local my_email="$1"
    for e in "${EMAILS[@]}"; do
        if [[ "$e" != "$my_email" ]]; then
            echo "$e"
            return 0
        fi
    done
    echo "$my_email"
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
    local peer_email="$(get_peer_email "$email")"

    info ""
    info "[$role] Running $nb_name... (peer=$peer_email)"

    [[ -d "$client_dir" ]] || { echo "Client dir not found: $client_dir" >&2; return 1; }

    local config_path="$client_dir/.syftbox/config.json"
    [[ -f "$config_path" ]] || { echo "SyftBox config not found: $config_path" >&2; return 1; }

    local work_dir="$client_dir/notebooks"
    mkdir -p "$work_dir"
    cp "$ROOT_DIR/$nb_rel" "$work_dir/$nb_name"
    cd "$work_dir"

    cat > "$work_dir/session.json" <<EOF
{
    "session_id": "$SESSION_ID",
    "peer": "$peer_email",
    "role": "$role",
    "status": "active"
}
EOF

    local env_vars=(
        "HOME=$client_dir"
        "SYFTBOX_EMAIL=$email"
        "SYFTBOX_DATA_DIR=$client_dir"
        "SYFTBOX_CONFIG_PATH=$config_path"
        "SYC_VAULT=$client_dir/.syc"
        "BEAVER_SESSION_ID=$SESSION_ID"
        "BEAVER_USER=$email"
        "BEAVER_AUTO_ACCEPT=1"
    )

    if [[ "$INTERACTIVE" == "1" ]]; then
        local port=$((8888 + i))
        info "[$role] Jupyter on port $port"
        env "${env_vars[@]}" "$PYTHON" -m jupyter notebook --no-browser --port "$port" --NotebookApp.token='' &
        return 0
    fi

    if env "${env_vars[@]}" "$PYTHON" -m papermill "$nb_name" "$out_nb" \
        --log-output --progress-bar --execution-timeout "$timeout" 2>&1 \
        | sed "s/^/[$role] /"; then
        info "[$role] ✓ PASSED"
        return 0
    else
        info "[$role] ✗ FAILED"
        return 1
    fi
}

info ""
info "Running notebooks (session_id=$SESSION_ID)..."
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

if [[ "$INTERACTIVE" == "1" ]]; then
    info ""
    info "=========================================="
    info "Jupyter servers running. Open in browser:"
    for i in "${!ROLES[@]}"; do
        local_port=$((8888 + i))
        info "  [${ROLES[i]}] http://localhost:$local_port"
    done
    info ""
    info "Press Ctrl+C to stop all servers..."
    info "=========================================="
    wait
fi

info ""
[[ "$RET" == "0" ]] && info "✓ ALL TESTS PASSED" || info "✗ TESTS FAILED"
exit "$RET"
