#!/bin/bash
set -euo pipefail

# Build BioVault Beaver Docker image
#
# Usage: ./build.sh [version]
#
# If no version is provided, reads from __init__.py

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

PLATFORMS=${PLATFORMS:-linux/amd64,linux/arm64}
BUILDER_NAME=${BUILDER_NAME:-beaver-builder}
REMOTE_IMAGE=${REMOTE_IMAGE:-ghcr.io/openmined/beaver}
# By default, produce a multi-arch OCI archive instead of pushing
OUTPUT_MODE=${OUTPUT_MODE:-oci} # oci|push
OUTPUT_DEST="${OUTPUT_DEST:-}"
LOAD_PLATFORM=${LOAD_PLATFORM:-auto} # auto|none|<platform>
VERIFY_MANIFEST=${VERIFY_MANIFEST:-0}

# Get version from argument or __init__.py
if [ -n "${1:-}" ]; then
    VERSION="$1"
else
    VERSION=$(grep '^__version__ = ' python/src/beaver/__init__.py | cut -d'"' -f2)
fi

if [ -z "$OUTPUT_DEST" ]; then
    OUTPUT_DEST="docker/dist/beaver-${VERSION}.oci.tar"
fi

if ! docker buildx inspect "$BUILDER_NAME" >/dev/null 2>&1; then
    echo "Creating docker buildx builder '$BUILDER_NAME'..."
    docker buildx create --name "$BUILDER_NAME" --driver docker-container >/dev/null
fi

docker buildx use "$BUILDER_NAME" >/dev/null
docker buildx inspect "$BUILDER_NAME" --bootstrap >/dev/null

echo "Building beaver:${VERSION} for ${PLATFORMS}..."

REMOTE_TAGS=( )
if [ -n "$REMOTE_IMAGE" ]; then
    REMOTE_TAGS=("${REMOTE_IMAGE}:${VERSION}" "${REMOTE_IMAGE}:latest" "${REMOTE_IMAGE}:0.1.39")
fi

BUILD_CMD=(docker buildx build
    --builder "$BUILDER_NAME"
    --platform "$PLATFORMS"
    -f docker/Dockerfile
)
for tag in "${REMOTE_TAGS[@]}"; do
    BUILD_CMD+=( -t "$tag" )
done

case "$OUTPUT_MODE" in
    push)
        if [ "${#REMOTE_TAGS[@]}" -eq 0 ]; then
            echo "REMOTE_IMAGE must be set when OUTPUT_MODE=push" >&2
            exit 1
        fi
        BUILD_CMD+=( --push )
        ;;
    oci)
        mkdir -p "$(dirname "$OUTPUT_DEST")"
        BUILD_CMD+=( --output "type=oci,dest=${OUTPUT_DEST}" )
        ;;
    *)
        echo "Unsupported OUTPUT_MODE: $OUTPUT_MODE (expected 'oci' or 'push')" >&2
        exit 1
        ;;
esac

BUILD_CMD+=( . )

"${BUILD_CMD[@]}"

LOAD_PLATFORM_RESOLVED=""
if [ "$LOAD_PLATFORM" = "auto" ]; then
    case "$(uname -m)" in
        x86_64)
            LOAD_PLATFORM_RESOLVED="linux/amd64"
            ;;
        arm64|aarch64)
            LOAD_PLATFORM_RESOLVED="linux/arm64"
            ;;
        *)
            echo "Skipping local load: unsupported host architecture '$(uname -m)'" >&2
            LOAD_PLATFORM_RESOLVED=""
            ;;
    esac
elif [ "$LOAD_PLATFORM" = "none" ]; then
    LOAD_PLATFORM_RESOLVED=""
else
    LOAD_PLATFORM_RESOLVED="$LOAD_PLATFORM"
fi

LOCAL_TAGS=("beaver:${VERSION}" "beaver:latest" "${REMOTE_IMAGE}:0.1.39")
LOCAL_LOADED=0
if [ -n "$LOAD_PLATFORM_RESOLVED" ]; then
    LOAD_CMD=(docker buildx build
        --builder "$BUILDER_NAME"
        --platform "$LOAD_PLATFORM_RESOLVED"
        -f docker/Dockerfile
    )
    for tag in "${LOCAL_TAGS[@]}"; do
        LOAD_CMD+=( -t "$tag" )
    done
    LOAD_CMD+=( --load . )
    "${LOAD_CMD[@]}"
    LOCAL_LOADED=1
fi

echo "✓ Built beaver:${VERSION}"
case "$OUTPUT_MODE" in
    push)
        echo "  Multi-arch pushed: ${REMOTE_TAGS[*]}"
        ;;
    oci)
        echo "  Multi-arch OCI archive: ${OUTPUT_DEST}"
        ;;
esac

if [ "$LOCAL_LOADED" = "1" ]; then
    echo "  Loaded local image for ${LOAD_PLATFORM_RESOLVED}: beaver:${VERSION}, beaver:latest"
else
    echo "  Local image not loaded"
fi

if [ "$VERIFY_MANIFEST" = "1" ]; then
    if [ "$OUTPUT_MODE" = "push" ]; then
        docker buildx imagetools inspect "${REMOTE_IMAGE}:${VERSION}"
    else
        echo "(Skipping manifest inspect: only available for pushed images)"
    fi
fi

echo ""
echo "Suggested next steps:"
if [ "$OUTPUT_MODE" = "push" ]; then
    echo "  docker buildx imagetools inspect ${REMOTE_IMAGE}:${VERSION}"
fi
if [ "$LOCAL_LOADED" = "1" ]; then
    echo "  docker run --platform=${LOAD_PLATFORM_RESOLVED} --rm beaver:${VERSION}"
fi
