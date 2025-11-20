#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SANDBOX_ROOT="$ROOT_DIR/sandbox"
PACKAGE_DIR="$ROOT_DIR/python"
SHARED_DIR="$SANDBOX_ROOT/shared"
CLIENT_SPECS=(
  "client1@sandbox.local:9888"
  "client2@sandbox.local:9889"
)

mkdir -p "$SHARED_DIR"

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
    elif command -v python >/dev/null 2>&1; then
      if python - "$url" <<'PY' >/dev/null 2>&1; then
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
  for pid in "${JUPYTER_PIDS[@]-}"; do
    if kill -0 "$pid" >/dev/null 2>&1; then
      kill "$pid" >/dev/null 2>&1 || true
    fi
  done
}
trap cleanup EXIT INT TERM

start_client() {
  local spec="$1"
  local client="${spec%%:*}"
  local port="${spec##*:}"
  local client_dir="$SANDBOX_ROOT/$client"
  local venv_path="$client_dir/.venv"

  mkdir -p "$client_dir"
  ln -snf ../shared "$client_dir/shared"

  if [[ ! -f "$venv_path/bin/python" ]]; then
    echo "[$client] creating virtual environment at $venv_path"
    uv venv "$venv_path"
  fi

  echo "[$client] installing dependencies"
  uv pip install -p "$venv_path/bin/python" -U jupyter pytest
  uv pip install -p "$venv_path/bin/python" -e "$PACKAGE_DIR"

  echo "[$client] launching Jupyter Lab on http://127.0.0.1:$port with root $client_dir"
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
      echo "[$client] opened browser tab at $url"
    else
      echo "[$client] please open $url manually"
    fi
  else
    echo "[$client] Jupyter not reachable at $url after timeout" >&2
  fi
}

declare -a JUPYTER_PIDS=()
for spec in "${CLIENT_SPECS[@]}"; do
  start_client "$spec"
done

wait "${JUPYTER_PIDS[@]}"
