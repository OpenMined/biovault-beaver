#!/bin/bash
set -euo pipefail

# Test beaver using Docker
# Rebuilds the container and runs the classifier

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

HOST_ARCH=$(uname -m)
case "$HOST_ARCH" in
    x86_64)
        DEFAULT_PLATFORM="linux/amd64"
        ;;
    arm64|aarch64)
        DEFAULT_PLATFORM="linux/arm64"
        ;;
    *)
        echo "Unsupported host architecture: $HOST_ARCH" >&2
        exit 1
        ;;
esac

PLATFORM=${PLATFORM:-$DEFAULT_PLATFORM}
OUTPUT_MODE=${OUTPUT_MODE:-oci}
LOAD_PLATFORM=${LOAD_PLATFORM:-$PLATFORM}
BUILD_PLATFORMS=${BUILD_PLATFORMS:-linux/amd64,linux/arm64}

echo "Building Docker image..." >&2
PLATFORMS="$BUILD_PLATFORMS" \
    OUTPUT_MODE="$OUTPUT_MODE" \
    LOAD_PLATFORM="$LOAD_PLATFORM" \
    "$REPO_ROOT/docker/build.sh" >&2

echo "" >&2
echo "Running classification in Docker..." >&2
echo "" >&2

# Run with Docker
docker run --rm \
    --platform="$PLATFORM" \
    -v "$REPO_ROOT/examples/apol1:/data" \
    -w /data \
    beaver:latest \
    bash process_samplesheet.sh samplesheet.csv classify_apol1.py
