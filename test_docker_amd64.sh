#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLATFORM=linux/amd64 \
    OUTPUT_MODE="${OUTPUT_MODE:-oci}" \
    LOAD_PLATFORM=linux/amd64 \
    "$REPO_ROOT/test_docker.sh"
