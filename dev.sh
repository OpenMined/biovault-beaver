#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./dev.sh [--clients email1,email2] [--sandbox DIR] [--reset] [--stop] [--status] [--no-jupyter] [--random-ports|--no-random-ports]

Starts a SyftBox devstack using the sbdev helper (syftbox submodule), prepares
client workspaces, installs Python deps, and launches Jupyter for each client.

Options:
  --clients list   Comma-separated client emails (default: client1@sandbox.local,client2@sandbox.local)
  --sandbox DIR    Sandbox root (default: ./sandbox)
  --reset          Remove any existing devstack state before starting (also removes sandbox on stop)
  --stop           Stop the devstack and exit (honors --reset for cleanup)
  --status         Print devstack state (relay/state.json) and exit
  --no-jupyter     Do not launch Jupyter (stack + setup only)
  --random-ports   Let the devstack pick free ports (default)
  --no-random-ports Use fixed ports for the sandbox
  -h, --help       Show this message
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYFTBOX_DIR="$ROOT_DIR/syftbox"
SANDBOX_DIR="${SANDBOX_DIR:-$ROOT_DIR/sandbox}"
PACKAGE_DIR="$ROOT_DIR/python"
SYFTBOX_SDK_WHEEL_DIR="$ROOT_DIR/syftbox-sdk/python/target/wheels"
GO_CACHE_DIR="$SYFTBOX_DIR/.gocache"

ACTION="start"
RESET_FLAG=0
START_JUPYTER=1
RANDOM_PORTS=1
RAW_CLIENTS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --clients)
      [[ $# -lt 2 ]] && { echo "Missing value for --clients" >&2; usage >&2; exit 1; }
      RAW_CLIENTS+=("$2")
      shift
      ;;
    --sandbox)
      [[ $# -lt 2 ]] && { echo "Missing value for --sandbox" >&2; usage >&2; exit 1; }
      SANDBOX_DIR="$2"
      shift
      ;;
    --reset)
      RESET_FLAG=1
      ;;
    --stop)
      ACTION="stop"
      ;;
    --status)
      ACTION="status"
      ;;
    --no-jupyter)
      START_JUPYTER=0
      ;;
    --random-ports)
      RANDOM_PORTS=1
      ;;
    --no-random-ports)
      RANDOM_PORTS=0
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
  shift
done

require_bin() {
  command -v "$1" >/dev/null 2>&1 || { echo "Missing required tool: $1" >&2; exit 1; }
}

abs_path() {
  python3 - <<'PY' "$1"
import os, sys
print(os.path.abspath(sys.argv[1]))
PY
}

SANDBOX_DIR="$(abs_path "$SANDBOX_DIR")"

declare -a CLIENTS=()
add_client() {
  local raw="$1"
  [[ -z "$raw" ]] && return
  raw="${raw#"${raw%%[![:space:]]*}"}"
  raw="${raw%"${raw##*[![:space:]]}"}"
  [[ -z "$raw" ]] && return
  raw="$(printf '%s' "$raw" | tr '[:upper:]' '[:lower:]')"
  for existing in "${CLIENTS[@]:-}"; do
    [[ "$existing" == "$raw" ]] && return
  done
  CLIENTS+=("$raw")
}

if ((${#RAW_CLIENTS[@]})); then
  for block in "${RAW_CLIENTS[@]}"; do
    IFS=',' read -r -a parts <<< "$block"
    for part in "${parts[@]}"; do
      add_client "$part"
    done
  done
fi

if ((${#CLIENTS[@]} == 0)); then
  add_client "client1@sandbox.local"
  add_client "client2@sandbox.local"
fi

STACK_STATE_FILE="$SANDBOX_DIR/relay/state.json"
SERVER_URL="${SYFTBOX_SERVER_URL:-http://localhost:8080}"
MINIO_API_PORT=""
MINIO_CONSOLE_PORT=""
declare -a STACK_CLIENTS=()
declare -a STACK_CLIENT_PORTS=()

JUPYTER_PIDS=()
declare -a JUPYTER_CLIENTS=()
declare -a JUPYTER_PORTS=()
declare -a CLIENT_VENVS_EMAILS=()
declare -a CLIENT_VENVS_PATHS=()
STACK_STARTED=0

# Shared session ID for all clients
SESSION_ID="dev_session_$(date +%s)"

# Get peer email for a given client (returns the other client)
get_peer_email() {
  local my_email="$1"
  for e in "${CLIENTS[@]}"; do
    if [[ "$e" != "$my_email" ]]; then
      echo "$e"
      return 0
    fi
  done
  # If only one client, peer is self
  echo "$my_email"
}

find_free_port() {
  local port
  while true; do
    port=$((RANDOM % 10000 + 50000))
    if ! lsof -i ":$port" >/dev/null 2>&1; then
      echo "$port"
      return
    fi
  done
}

wait_for_url() {
  local url="$1"
  local timeout="${2:-60}"
  local start=$SECONDS
  while (( SECONDS - start < timeout )); do
    if curl -sf "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

open_browser() {
  local url="$1"
  if command -v open >/dev/null 2>&1; then
    open "$url" >/dev/null 2>&1 &
  elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$url" >/dev/null 2>&1 &
  elif command -v wslview >/dev/null 2>&1; then
    wslview "$url" >/dev/null 2>&1 &
  fi
}

cleanup() {
  for pid in "${JUPYTER_PIDS[@]-}"; do
    if kill -0 "$pid" >/dev/null 2>&1; then
      kill "$pid" >/dev/null 2>&1 || true
    fi
  done

  if (( STACK_STARTED )); then
    (cd "$SYFTBOX_DIR" && GOCACHE="$GO_CACHE_DIR" just sbdev-stop --path "$SANDBOX_DIR") >/dev/null 2>&1 || true
    if (( RESET_FLAG )); then
      rm -rf "$SANDBOX_DIR"
    fi
  fi
}
trap cleanup EXIT INT TERM

wait_for_file() {
  local path="$1"
  local timeout="${2:-30}"
  local start=$SECONDS
  while (( SECONDS - start < timeout )); do
    [[ -f "$path" ]] && return 0
    sleep 1
  done
  return 1
}

start_devstack() {
  require_bin just
  require_bin go
  [[ -d "$SYFTBOX_DIR" ]] || { echo "Missing syftbox submodule at $SYFTBOX_DIR" >&2; exit 1; }

  mkdir -p "$SANDBOX_DIR"
  # Clean up stale global devstack state to avoid dangling clients
  (cd "$SYFTBOX_DIR" && GOCACHE="$GO_CACHE_DIR" just sbdev-prune) || true

  local args=(sbdev-start --path "$SANDBOX_DIR")
  (( RANDOM_PORTS )) && args+=(--random-ports)
  (( RESET_FLAG )) && args+=(--reset)
  for email in "${CLIENTS[@]}"; do
    args+=(--client "$email")
  done

  echo "Starting SyftBox devstack..."
  (cd "$SYFTBOX_DIR" && GOCACHE="$GO_CACHE_DIR" just "${args[@]}")
  STACK_STARTED=1
}

stop_devstack() {
  echo "Stopping SyftBox devstack at $SANDBOX_DIR..."
  if [[ -d "$SYFTBOX_DIR" ]]; then
    (cd "$SYFTBOX_DIR" && GOCACHE="$GO_CACHE_DIR" just sbdev-stop --path "$SANDBOX_DIR") || true
  fi
  if (( RESET_FLAG )); then
    rm -rf "$SANDBOX_DIR"
  fi
  echo "Devstack stop complete."
}

print_state() {
  if [[ ! -f "$STACK_STATE_FILE" ]]; then
    echo "State file not found at $STACK_STATE_FILE"
    exit 1
  fi
  python3 - <<'PY' "$STACK_STATE_FILE"
import json, sys, pathlib
path = pathlib.Path(sys.argv[1])
data = json.loads(path.read_text())
json.dump(data, sys.stdout, indent=2, sort_keys=True)
print()
PY
}

load_stack_info() {
  SERVER_URL="${SYFTBOX_SERVER_URL:-http://localhost:8080}"
  MINIO_API_PORT=""
  MINIO_CONSOLE_PORT=""
  STACK_CLIENTS=()
  STACK_CLIENT_PORTS=()

  [[ -f "$STACK_STATE_FILE" ]] || return

  local state_output
  if ! state_output="$(python3 - "$STACK_STATE_FILE" <<'PY'
import json, sys
path = sys.argv[1]
data = json.load(open(path))
server = data.get("server") or data.get("Server") or {}
server_port = server.get("port") or server.get("Port")
if server_port:
    print(f"server_port={server_port}")
minio = data.get("minio") or data.get("Minio") or {}
minio_api = minio.get("api_port") or minio.get("APIPort")
minio_console = minio.get("console_port") or minio.get("ConsolePort")
if minio_api:
    print(f"minio_api={minio_api}")
if minio_console:
    print(f"minio_console={minio_console}")
clients = data.get("clients") or data.get("Clients") or []
for client in clients:
    email = client.get("email") or client.get("Email")
    port = client.get("port") or client.get("Port")
    if email and port:
        print(f"client={email}|{port}")
PY
)"; then
    echo "Warning: unable to read devstack state at $STACK_STATE_FILE" >&2
    return
  fi

  while IFS= read -r line; do
    case "$line" in
      server_port=*)
        local port="${line#server_port=}"
        SERVER_URL="http://127.0.0.1:${port}"
        ;;
      minio_api=*)
        MINIO_API_PORT="${line#minio_api=}"
        ;;
      minio_console=*)
        MINIO_CONSOLE_PORT="${line#minio_console=}"
        ;;
      client=*)
        local payload="${line#client=}"
        local email="${payload%%|*}"
        local port="${payload##*|}"
        STACK_CLIENTS+=("$email")
        STACK_CLIENT_PORTS+=("$port")
        ;;
    esac
  done <<<"$state_output"
}

build_syftbox_wheel() {
  echo "Building syftbox-sdk wheel..." >&2
  (cd "$ROOT_DIR/syftbox-sdk/python" && uv tool run maturin build --release)
  ls -t "$SYFTBOX_SDK_WHEEL_DIR"/syftbox_sdk-*.whl 2>/dev/null | head -1
}

provision_client_identity() {
  local email="$1"
  local data_dir="$2"
  local venv_python="$3"
  local vault_path="$data_dir/.sbc"

  echo "[$email] Provisioning SyftBox identity..."
  "$venv_python" - "$email" "$data_dir" "$vault_path" <<'PY'
import sys

email = sys.argv[1]
data_dir = sys.argv[2]
vault_path = sys.argv[3]

import syftbox_sdk as syft

# Use keyword args with explicit vault_override to match test_notebooks_syftbox behavior
result = syft.provision_identity(
    identity=email,
    data_root=data_dir,
    vault_override=vault_path,
)
if result.generated:
    print(f"  Generated identity for {email}")
else:
    print(f"  Identity already exists for {email}")
print(f"  Vault: {result.vault_path}")
print(f"  Public bundle: {result.public_bundle_path}")
PY
}

write_permissions() {
  local email="$1"
  local perm_file="$2"
  cat > "$perm_file" <<EOF
rules:
  - pattern: 'biovault/**'
    access:
      admin:
        - '$email'
      read:
        - '*'
      write:
        - 'client1@sandbox.local'
        - 'client2@sandbox.local'
EOF
}

prepare_client_workspace() {
  local email="$1"
  local wheel="$2"
  local client_home="$SANDBOX_DIR/$email"
  local data_root="$client_home"
  local datasites_root="$client_home/datasites"
  local self_root="$datasites_root/$email"
  local shared_dir="$self_root/shared/biovault"
  local public_dir="$self_root/public/crypto"
  local session_dir="$self_root/app_data/biovault/rpc/session"
  local unencrypted_dir="$data_root/unencrypted/$email/shared/biovault"
  local config_path="$client_home/.syftbox/config.json"
  local venv_path="$client_home/.venv"
  local notebooks_dir="$ROOT_DIR/notebooks"

  wait_for_file "$config_path" 60 || { echo "SyftBox config not found for $email at $config_path" >&2; exit 1; }

  echo "[$email] Preparing workspace..."
  mkdir -p "$shared_dir" "$public_dir" "$session_dir" "$unencrypted_dir"
  mkdir -p "$data_root/.sbc/keys" "$data_root/.sbc/bundles" "$data_root/.sbc/config"

  for peer in "${CLIENTS[@]}"; do
    [[ "$peer" == "$email" ]] && continue
    mkdir -p "$datasites_root/$peer/shared/biovault"
  done

  write_permissions "$email" "$self_root/shared/syft.pub.yaml"

  # Create session.json for shared session between clients
  local peer_email
  peer_email="$(get_peer_email "$email")"
  cat > "$client_home/session.json" <<EOF
{
    "session_id": "$SESSION_ID",
    "peer": "$peer_email",
    "role": "partner",
    "status": "active"
}
EOF
  echo "[$email] Created session.json (peer=$peer_email, session=$SESSION_ID)"

  # Symlink notebooks based on client role
  if [[ "$email" == "client1@sandbox.local" ]]; then
    while IFS= read -r nb || [[ -n "$nb" ]]; do
      [[ -n "$nb" && -f "$notebooks_dir/$nb" ]] && ln -snf "$notebooks_dir/$nb" "$client_home/$nb"
    done < "$notebooks_dir/do.txt"
  elif [[ "$email" == "client2@sandbox.local" ]]; then
    while IFS= read -r nb || [[ -n "$nb" ]]; do
      [[ -n "$nb" && -f "$notebooks_dir/$nb" ]] && ln -snf "$notebooks_dir/$nb" "$client_home/$nb"
    done < "$notebooks_dir/ds.txt"
  fi

  for asset in "data.csv" "mock.csv"; do
    if [[ -f "$notebooks_dir/$asset" ]]; then
      ln -snf "$notebooks_dir/$asset" "$client_home/$asset"
    fi
  done
  if [[ -d "$notebooks_dir/single_cell" ]]; then
    ln -snf "$notebooks_dir/single_cell" "$client_home/single_cell"
  fi

  # Symlink entire notebooks folder for easy access
  ln -snf "$notebooks_dir" "$client_home/notebooks"

  # Track venv for bundle imports
  CLIENT_VENVS_EMAILS+=("$email")
  CLIENT_VENVS_PATHS+=("$venv_path/bin/python")

  echo "[$email] Creating virtual environment (Python 3.12)..."
  uv venv -p 3.12 "$venv_path"

  echo "[$email] Installing Python dependencies..."
  uv pip install -p "$venv_path/bin/python" -U jupyter pytest ipykernel
  uv pip install -p "$venv_path/bin/python" -e "$PACKAGE_DIR"
  if [[ -n "$wheel" && -f "$wheel" ]]; then
    uv pip install -p "$venv_path/bin/python" --force-reinstall "$wheel"
  fi

  # Create kernel spec with correct environment
  local kernel_name="${email%%@*}"  # e.g., "client1" from "client1@sandbox.local"
  local kernel_dir="$client_home/.local/share/jupyter/kernels/$kernel_name"
  echo "[$email] Creating Jupyter kernel '$kernel_name'..."
  mkdir -p "$kernel_dir"
  cat > "$kernel_dir/kernel.json" <<KERNEL
{
  "argv": ["$venv_path/bin/python", "-m", "ipykernel_launcher", "-f", "{connection_file}"],
  "display_name": "Python ($kernel_name)",
  "language": "python",
  "env": {
    "PATH": "$venv_path/bin:\${PATH}",
    "VIRTUAL_ENV": "$venv_path",
    "SYFTBOX_EMAIL": "$email",
    "SYFTBOX_DATA_DIR": "$client_home",
    "SYFTBOX_CONFIG_PATH": "$config_path",
    "SYFTBOX_SERVER_URL": "$SERVER_URL",
    "SBC_VAULT": "$client_home/.sbc",
    "BEAVER_SESSION_ID": "$SESSION_ID",
    "BEAVER_USER": "$email",
    "BEAVER_PEER": "$peer_email",
    "BEAVER_AUTO_ACCEPT": "1"
  }
}
KERNEL

  provision_client_identity "$email" "$client_home" "$venv_path/bin/python"
}

import_peer_bundles() {
  echo "Importing peer bundles into vaults..."
  for i in "${!CLIENT_VENVS_EMAILS[@]}"; do
    local email="${CLIENT_VENVS_EMAILS[$i]}"
    local venv_python="${CLIENT_VENVS_PATHS[$i]}"
    local vault_path="$SANDBOX_DIR/$email/.sbc"
    for peer in "${CLIENTS[@]}"; do
      [[ "$peer" == "$email" ]] && continue
      local bundle="$SANDBOX_DIR/$peer/datasites/$peer/public/crypto/did.json"
      if [[ -f "$bundle" ]]; then
        "$venv_python" - "$bundle" "$vault_path" "$peer" <<'PY'
import sys, pathlib
bundle, vault, expected = sys.argv[1:4]
import syftbox_sdk as syft
if pathlib.Path(bundle).exists():
    identity = syft.import_bundle(bundle, vault, expected_identity=expected)
    print(f"  Imported {identity} into {vault}")
PY
      else
        echo "  Skipping import for $email <- $peer (bundle missing at $bundle)"
      fi
    done
  done
}

start_jupyter() {
  local email="$1"
  local client_home="$SANDBOX_DIR/$email"
  local venv_path="$client_home/.venv"
  local config_path="$client_home/.syftbox/config.json"
  local vault_path="$client_home/.sbc"
  local peer_email
  peer_email="$(get_peer_email "$email")"
  local port
  port="$(find_free_port)"

  echo "[$email] Launching Jupyter Lab on http://127.0.0.1:$port..."

  JUPYTER_CLIENTS+=("$email")
  JUPYTER_PORTS+=("$port")

  (
    cd "$client_home"
    # Activate the venv so jupyter and python are in PATH
    source "$venv_path/bin/activate"

    # Set SyftBox environment variables for proper backend integration
    export SYFTBOX_EMAIL="$email"
    export SYFTBOX_DATA_DIR="$client_home"
    export SYFTBOX_CONFIG_PATH="$config_path"
    export SYFTBOX_SERVER_URL="$SERVER_URL"
    export SBC_VAULT="$vault_path"
    # Beaver session environment for shared sessions
    export BEAVER_AUTO_ACCEPT="${BEAVER_AUTO_ACCEPT:-1}"
    export BEAVER_SESSION_ID="$SESSION_ID"
    export BEAVER_USER="$email"
    export BEAVER_PEER="$peer_email"
    # Tell Jupyter to find kernels in client's local dir
    export JUPYTER_DATA_DIR="$client_home/.local/share/jupyter"

    exec jupyter lab \
      --no-browser \
      --ServerApp.ip=127.0.0.1 \
      --ServerApp.allow_remote_access=False \
      --ServerApp.port="$port" \
      --ServerApp.root_dir="$client_home" \
      --ServerApp.token='' \
      --ServerApp.disable_check_xsrf=True
  ) &
  JUPYTER_PIDS+=("$!")

  local url="http://127.0.0.1:${port}/lab"
  if wait_for_url "$url"; then
    echo "[$email] Jupyter running at $url"
    open_browser "$url" || true
  else
    echo "[$email] Jupyter not reachable at $url after timeout" >&2
  fi
}

if [[ "$ACTION" == "stop" ]]; then
  stop_devstack
  exit 0
fi

if [[ "$ACTION" == "status" ]]; then
  print_state
  exit 0
fi

echo "=== SyftBox devstack ==="
start_devstack
load_stack_info

echo ""
echo "=== Preparing clients ==="
SYFTBOX_WHEEL_PATH="$(build_syftbox_wheel)"
for email in "${CLIENTS[@]}"; do
  prepare_client_workspace "$email" "$SYFTBOX_WHEEL_PATH"
done
import_peer_bundles

if (( START_JUPYTER )); then
  echo ""
  echo "=== Starting Jupyter notebooks ==="
  for email in "${CLIENTS[@]}"; do
    start_jupyter "$email"
  done

  echo ""
  echo "=== Sandbox Ready ==="
  echo "SyftBox server: $SERVER_URL"
  if [[ -n "$MINIO_API_PORT" ]]; then
    echo "MinIO: http://127.0.0.1:${MINIO_API_PORT} (console http://127.0.0.1:${MINIO_CONSOLE_PORT})"
  fi
  echo "Clients:"
  for i in "${!JUPYTER_CLIENTS[@]}"; do
    echo "  ${JUPYTER_CLIENTS[$i]} -> http://127.0.0.1:${JUPYTER_PORTS[$i]}/lab"
  done
  echo "Data root: $SANDBOX_DIR"
  wait "${JUPYTER_PIDS[@]}"
else
  echo ""
  echo "=== Sandbox Ready (no Jupyter) ==="
  echo "Data root: $SANDBOX_DIR"
  echo "SyftBox server: $SERVER_URL"
fi
