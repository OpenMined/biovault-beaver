#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SANDBOX_ROOT="$ROOT_DIR/sandbox"
PACKAGE_DIR="$ROOT_DIR/python"
SYFTBOX_SDK_WHEEL="$ROOT_DIR/syftbox-sdk/python/target/wheels"
SYFTBOX_DIR="$ROOT_DIR/syftbox"
SYFTBOX_DOCKER_DIR="$SYFTBOX_DIR/docker"
SERVER_URL="${SYFTBOX_SERVER_URL:-http://localhost:8080}"

# Client emails (ports assigned dynamically)
CLIENT_EMAILS=(
  "client1@sandbox.local"
  "client2@sandbox.local"
)

# Find an available port
find_free_port() {
  local port
  while true; do
    port=$((RANDOM % 10000 + 50000))  # Random port between 50000-60000
    if ! lsof -i ":$port" >/dev/null 2>&1; then
      echo "$port"
      return
    fi
  done
}

usage() {
  cat <<'EOF'
Usage: ./dev.sh [--reset] [--no-server] [--no-jupyter]

Options:
  --reset       Full reset: stop containers, delete volumes, remove sandbox, start fresh
  --no-server   Skip starting SyftBox server (assume it's already running)
  --no-jupyter  Skip launching Jupyter notebooks
  -h, --help    Show this message

The --reset flag will:
  1. Stop all SyftBox clients (sbenv stop)
  2. Stop and remove Docker containers (server, minio)
  3. Remove Docker volumes (minio data)
  4. Delete the entire sandbox directory
  5. Rebuild and start everything fresh
EOF
}

RESET_FLAG=0
START_SERVER=1
START_JUPYTER=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --reset)
      RESET_FLAG=1
      ;;
    --no-server)
      START_SERVER=0
      ;;
    --no-jupyter)
      START_JUPYTER=0
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

wait_for_url() {
  local url="$1"
  local timeout="${2:-90}"
  local start=$SECONDS
  while (( SECONDS - start < timeout )); do
    if command -v curl >/dev/null 2>&1; then
      if curl -sf "$url" >/dev/null 2>&1; then
        return 0
      fi
    elif command -v python3 >/dev/null 2>&1; then
      if python3 - "$url" <<'PY' >/dev/null 2>&1; then
import sys
import urllib.request
try:
    urllib.request.urlopen(sys.argv[1], timeout=5)
except Exception:
    raise SystemExit(1)
raise SystemExit(0)
PY
        return 0
      fi
    else
      echo "Cannot probe $url (need curl or python)." >&2
      break
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
  else
    echo "Unable to auto-launch browser, please open $url manually."
    return 1
  fi
}

cleanup() {
  echo "Cleaning up..."
  # Stop SyftBox clients
  for client in "${CLIENT_EMAILS[@]}"; do
    local client_dir="$SANDBOX_ROOT/$client"
    if [[ -d "$client_dir" ]]; then
      (cd "$client_dir" && sbenv stop 2>/dev/null || true)
    fi
  done
  # Stop Jupyter processes
  for pid in "${JUPYTER_PIDS[@]-}"; do
    if kill -0 "$pid" >/dev/null 2>&1; then
      kill "$pid" >/dev/null 2>&1 || true
    fi
  done
}
trap cleanup EXIT INT TERM

reset_sandbox() {
  echo ""
  echo "=== Full Reset ==="
  echo ""

  # 1. Stop all SyftBox clients
  echo "Stopping SyftBox clients..."
  for client in "${CLIENT_EMAILS[@]}"; do
    local client_dir="$SANDBOX_ROOT/$client"
    if [[ -d "$client_dir" ]]; then
      echo "  Stopping $client..."
      (cd "$client_dir" && sbenv stop 2>/dev/null || true)
    fi
  done
  sleep 2

  # 2. Stop and remove Docker containers + volumes
  echo "Stopping Docker containers and removing volumes..."
  if [[ -d "$SYFTBOX_DOCKER_DIR" ]]; then
    (cd "$SYFTBOX_DOCKER_DIR" && docker compose down -v 2>/dev/null || true)
    echo "  Docker containers stopped, volumes removed"
  fi

  # 3. Delete sandbox directory
  echo "Removing sandbox directory..."
  if [[ -d "$SANDBOX_ROOT" ]]; then
    rm -rf "$SANDBOX_ROOT"
    echo "  Removed $SANDBOX_ROOT"
  fi

  echo ""
  echo "=== Reset Complete ==="
  echo ""
}

start_syftbox_server() {
  echo "Starting SyftBox server + MinIO via Docker..."
  if ! command -v docker >/dev/null 2>&1; then
    echo "Docker is required to run SyftBox server" >&2
    exit 1
  fi

  (cd "$SYFTBOX_DOCKER_DIR" && docker compose up -d --build minio server)

  echo "Waiting for SyftBox server at $SERVER_URL..."
  if wait_for_url "$SERVER_URL" 120; then
    echo "SyftBox server is ready."
  else
    echo "Server did not respond within timeout." >&2
    exit 1
  fi
}

find_syftbox_wheel() {
  # Find the latest wheel in the target/wheels directory
  local wheel
  wheel=$(ls -t "$SYFTBOX_SDK_WHEEL"/syftbox_sdk-*.whl 2>/dev/null | head -1)
  if [[ -z "$wheel" ]]; then
    echo "No syftbox-sdk wheel found. Building..." >&2
    (cd "$ROOT_DIR/syftbox-sdk/python" && uv tool run maturin build --release)
    wheel=$(ls -t "$SYFTBOX_SDK_WHEEL"/syftbox_sdk-*.whl 2>/dev/null | head -1)
  fi
  echo "$wheel"
}

setup_client_directory() {
  local email="$1"
  local client_dir="$SANDBOX_ROOT/$email"

  echo "[$email] Setting up directory structure..."

  # Create SyftBox directory structure
  mkdir -p "$client_dir/datasites/$email/public/crypto"
  mkdir -p "$client_dir/datasites/$email/shared/biovault"
  mkdir -p "$client_dir/datasites/$email/app_data/biovault/rpc/session"
  mkdir -p "$client_dir/unencrypted"
  mkdir -p "$client_dir/.syc/keys"
  mkdir -p "$client_dir/.syc/bundles"
  mkdir -p "$client_dir/.syc/config"
}

provision_client_identity() {
  local email="$1"
  local client_dir="$SANDBOX_ROOT/$email"
  local venv_python="$client_dir/.venv/bin/python"

  echo "[$email] Provisioning crypto identity..."

  "$venv_python" - "$email" "$client_dir" <<'PROVISION_PY'
import sys
import syftbox_sdk as syft

email = sys.argv[1]
data_dir = sys.argv[2]

result = syft.provision_identity(email, data_dir)
if result.generated:
    print(f"  Generated new identity for {email}")
    if result.recovery_mnemonic:
        print(f"  Recovery mnemonic: {result.recovery_mnemonic}")
else:
    print(f"  Identity already exists for {email}")
print(f"  Vault: {result.vault_path}")
print(f"  Public bundle: {result.public_bundle_path}")
PROVISION_PY
}

create_shared_permissions() {
  local email="$1"
  local client_dir="$SANDBOX_ROOT/$email"
  local perm_file="$client_dir/datasites/$email/shared/syft.pub.yaml"

  echo "[$email] Creating syft.pub.yaml permissions..."

  # Create permission file allowing both clients to read/write to biovault folder
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

exchange_bundles() {
  echo "Exchanging public bundles between clients..."

  local client1="${CLIENT_EMAILS[0]}"
  local client2="${CLIENT_EMAILS[1]}"
  local client1_dir="$SANDBOX_ROOT/$client1"
  local client2_dir="$SANDBOX_ROOT/$client2"

  # Copy client1's public bundle to client2's vault
  local client1_bundle="$client1_dir/datasites/$client1/public/crypto/did.json"
  local client2_vault="$client2_dir/.syc/bundles"
  if [[ -f "$client1_bundle" ]]; then
    cp "$client1_bundle" "$client2_vault/${client1}.json"
    echo "  Copied $client1 bundle to $client2 vault"
  fi

  # Copy client2's public bundle to client1's vault
  local client2_bundle="$client2_dir/datasites/$client2/public/crypto/did.json"
  local client1_vault="$client1_dir/.syc/bundles"
  if [[ -f "$client2_bundle" ]]; then
    cp "$client2_bundle" "$client1_vault/${client2}.json"
    echo "  Copied $client2 bundle to $client1 vault"
  fi

  # Also create views of each other's datasites
  mkdir -p "$client1_dir/datasites/$client2/shared/biovault"
  mkdir -p "$client2_dir/datasites/$client1/shared/biovault"

  echo "Bundle exchange complete."
}

start_client() {
  local client="$1"
  local client_dir="$SANDBOX_ROOT/$client"
  local venv_path="$client_dir/.venv"
  local wheel
  wheel="$(find_syftbox_wheel)"

  # Setup directory structure
  setup_client_directory "$client"

  # Symlink notebooks based on client role
  local notebooks_dir="$ROOT_DIR/notebooks"
  if [[ "$client" == "client1@sandbox.local" ]]; then
    # Data Owner notebooks from do.txt
    while IFS= read -r nb || [[ -n "$nb" ]]; do
      [[ -n "$nb" && -f "$notebooks_dir/$nb" ]] && ln -snf "$notebooks_dir/$nb" "$client_dir/$nb"
    done < "$notebooks_dir/do.txt"
  elif [[ "$client" == "client2@sandbox.local" ]]; then
    # Data Scientist notebooks from ds.txt
    while IFS= read -r nb || [[ -n "$nb" ]]; do
      [[ -n "$nb" && -f "$notebooks_dir/$nb" ]] && ln -snf "$notebooks_dir/$nb" "$client_dir/$nb"
    done < "$notebooks_dir/ds.txt"
  fi

  # Create virtual environment
  if [[ ! -f "$venv_path/bin/python" ]]; then
    echo "[$client] Creating virtual environment..."
    uv venv "$venv_path"
  fi

  # Install dependencies including syftbox-sdk wheel
  echo "[$client] Installing dependencies..."
  uv pip install -p "$venv_path/bin/python" -U jupyter pytest
  uv pip install -p "$venv_path/bin/python" -e "$PACKAGE_DIR"

  # Install syftbox-sdk wheel
  if [[ -n "$wheel" && -f "$wheel" ]]; then
    echo "[$client] Installing syftbox-sdk from wheel: $(basename "$wheel")"
    uv pip install -p "$venv_path/bin/python" --force-reinstall "$wheel"
  fi

  # Provision identity
  provision_client_identity "$client"

  # Create permissions file
  create_shared_permissions "$client"
}

build_syftbox_binary() {
  local goos goarch target
  goos="$(go env GOOS)"
  goarch="$(go env GOARCH)"
  target="$SYFTBOX_DIR/.out/syftbox_client_${goos}_${goarch}"

  if [[ -x "$target" ]]; then
    printf '%s\n' "$target"
    return
  fi

  echo "Building SyftBox client binary for ${goos}/${goarch}..." >&2
  mkdir -p "$SYFTBOX_DIR/.out"

  local version commit build_date ldflags cgo
  version="$(cd "$SYFTBOX_DIR" && git describe --tags --always 2>/dev/null || echo "dev")"
  commit="$(cd "$SYFTBOX_DIR" && git rev-parse --short HEAD 2>/dev/null || echo "unknown")"
  build_date="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  ldflags="-s -w"
  ldflags+=" -X github.com/openmined/syftbox/internal/version.Version=$version"
  ldflags+=" -X github.com/openmined/syftbox/internal/version.Revision=$commit"
  ldflags+=" -X github.com/openmined/syftbox/internal/version.BuildDate=$build_date"

  cgo=0
  [[ "$goos" == "darwin" ]] && cgo=1

  (cd "$SYFTBOX_DIR" && GOOS="$goos" GOARCH="$goarch" CGO_ENABLED="$cgo" \
    go build -trimpath --tags "go_json nomsgpack" -ldflags "$ldflags" \
    -o "$target" ./cmd/client) >&2

  printf '%s\n' "$target"
}

init_syftbox_client() {
  local client="$1"
  local client_dir="$SANDBOX_ROOT/$client"

  echo "[$client] Initializing SyftBox client..."

  # Check if sbenv is available
  if ! command -v sbenv >/dev/null 2>&1; then
    echo "Warning: sbenv not found, skipping SyftBox client init" >&2
    return 1
  fi

  # Build syftbox binary if needed
  local syftbox_binary
  syftbox_binary="$(build_syftbox_binary)"

  # Init sbenv in the client directory with the binary
  (
    cd "$client_dir"
    sbenv init --email "$client" --server-url "$SERVER_URL" --dev --quiet --binary "$syftbox_binary"
    # Create symlink to binary
    ln -sf "$syftbox_binary" ./syftbox
  )

  echo "[$client] SyftBox client initialized"
}

start_syftbox_client() {
  local client="$1"
  local client_dir="$SANDBOX_ROOT/$client"

  echo "[$client] Starting SyftBox client..."

  if ! command -v sbenv >/dev/null 2>&1; then
    echo "Warning: sbenv not found, skipping SyftBox client start" >&2
    return 1
  fi

  local syftbox_binary
  syftbox_binary="$(build_syftbox_binary)"

  (
    cd "$client_dir"
    export SYFTBOX_BINARY="$syftbox_binary"
    sbenv start --skip-login-check
  )

  echo "[$client] SyftBox client started"
}

wait_for_client_ready() {
  local client="$1"
  local client_dir="$SANDBOX_ROOT/$client"
  local pid_file="$client_dir/.syftbox/syftbox.pid"
  local log_file="$client_dir/.syftbox/daemon.log"

  echo "[$client] Waiting for SyftBox client to sync..."

  for attempt in $(seq 1 90); do
    if [[ -f "$pid_file" ]]; then
      local daemon_pid
      daemon_pid="$(tr -d ' \n\r' < "$pid_file")"
      if [[ -n "$daemon_pid" ]] && ps -p "$daemon_pid" >/dev/null 2>&1; then
        if [[ -f "$log_file" ]] && grep -q "full sync completed" "$log_file"; then
          echo "[$client] SyftBox client is syncing"
          return 0
        fi
      fi
    fi
    sleep 2
  done

  echo "[$client] SyftBox client failed to reach steady state" >&2
  [[ -f "$log_file" ]] && tail -n 20 "$log_file" >&2
  return 1
}

start_jupyter() {
  local client="$1"
  local client_dir="$SANDBOX_ROOT/$client"
  local venv_path="$client_dir/.venv"
  local port
  port="$(find_free_port)"

  echo "[$client] Launching Jupyter Lab on http://127.0.0.1:$port..."

  # Store port for summary output
  CLIENT_PORTS_KEYS+=("$client")
  CLIENT_PORTS_VALS+=("$port")
  (
    cd "$client_dir"
    exec "$venv_path/bin/jupyter" lab \
      --no-browser \
      --ServerApp.ip=127.0.0.1 \
      --ServerApp.allow_remote_access=False \
      --ServerApp.port="$port" \
      --ServerApp.root_dir="$client_dir" \
      --ServerApp.token='' \
      --ServerApp.disable_check_xsrf=True
  ) &
  JUPYTER_PIDS+=("$!")

  local url="http://127.0.0.1:${port}/lab"
  if wait_for_url "$url"; then
    echo "[$client] Jupyter running at $url"
    if open_browser "$url"; then
      echo "[$client] Opened browser tab"
    fi
  else
    echo "[$client] Jupyter not reachable at $url after timeout" >&2
  fi
}

# Main execution
declare -a JUPYTER_PIDS=()
declare -a CLIENT_PORTS_KEYS=()
declare -a CLIENT_PORTS_VALS=()

if (( RESET_FLAG )); then
  reset_sandbox
fi

if (( START_SERVER )); then
  start_syftbox_server
fi

echo ""
echo "=== Setting up clients ==="
for client in "${CLIENT_EMAILS[@]}"; do
  start_client "$client"
done

# Exchange bundles after all clients are provisioned
exchange_bundles

# Initialize and start SyftBox clients
echo ""
echo "=== Starting SyftBox clients ==="
for client in "${CLIENT_EMAILS[@]}"; do
  init_syftbox_client "$client"
  start_syftbox_client "$client"
done

# Wait for clients to be ready
echo ""
echo "=== Waiting for SyftBox clients to sync ==="
for client in "${CLIENT_EMAILS[@]}"; do
  wait_for_client_ready "$client"
done

if (( START_JUPYTER )); then
  echo ""
  echo "=== Starting Jupyter notebooks ==="
  for client in "${CLIENT_EMAILS[@]}"; do
    start_jupyter "$client"
  done

  echo ""
  echo "=== Sandbox Ready ==="
  echo "Clients:"
  for i in "${!CLIENT_PORTS_KEYS[@]}"; do
    echo "  ${CLIENT_PORTS_KEYS[$i]} -> http://127.0.0.1:${CLIENT_PORTS_VALS[$i]}/lab"
  done
  echo ""
  echo "Data root: $SANDBOX_ROOT"
  echo "SyftBox server: $SERVER_URL"
  echo ""

  wait "${JUPYTER_PIDS[@]}"
else
  echo ""
  echo "=== Sandbox Ready (no Jupyter) ==="
  echo "Data root: $SANDBOX_ROOT"
  echo "SyftBox server: $SERVER_URL"
fi
