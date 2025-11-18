#!/bin/bash
set -e

# Bump version across the entire repository
# Usage: ./bump_version.sh <major|minor|patch|version>
#
# Examples:
#   ./bump_version.sh patch     # 0.1.0 -> 0.1.1
#   ./bump_version.sh minor     # 0.1.0 -> 0.2.0
#   ./bump_version.sh major     # 0.1.0 -> 1.0.0
#   ./bump_version.sh 0.2.0     # Set to specific version

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INIT_FILE="$REPO_ROOT/python/src/beaver/__init__.py"
BUILD_SCRIPT="$REPO_ROOT/docker/build.sh"

if [ $# -ne 1 ]; then
    echo "Usage: $0 <major|minor|patch|version>"
    exit 1
fi

# Get current version
CURRENT_VERSION=$(grep '^__version__ = ' "$INIT_FILE" | cut -d'"' -f2)
echo "Current version: $CURRENT_VERSION"

# Parse current version
IFS='.' read -r MAJOR MINOR PATCH <<< "$CURRENT_VERSION"

# Calculate new version
case "$1" in
    major)
        NEW_VERSION="$((MAJOR + 1)).0.0"
        ;;
    minor)
        NEW_VERSION="${MAJOR}.$((MINOR + 1)).0"
        ;;
    patch)
        NEW_VERSION="${MAJOR}.${MINOR}.$((PATCH + 1))"
        ;;
    *)
        # Assume it's a specific version
        NEW_VERSION="$1"
        ;;
esac

echo "New version: $NEW_VERSION"

# Update __init__.py
sed -i.bak "s/__version__ = .*/__version__ = \"${NEW_VERSION}\"/" "$INIT_FILE"
rm "${INIT_FILE}.bak"
echo "✓ Updated $INIT_FILE"

# Update all workflow.nf files
echo "Updating workflow.nf files..."
find "$REPO_ROOT/examples" -name "workflow.nf" -type f | while read -r file; do
    sed -i.bak "s|ghcr.io/openmined/beaver:[0-9]\+\.[0-9]\+\.[0-9]\+|ghcr.io/openmined/beaver:${NEW_VERSION}|g" "$file"
    rm "${file}.bak"
    echo "  ✓ Updated $file"
done

# Update docker/build.sh
echo "Updating docker/build.sh..."
sed -i.bak "s|\${REMOTE_IMAGE}:[0-9]\+\.[0-9]\+\.[0-9]\+|\${REMOTE_IMAGE}:${NEW_VERSION}|g" "$BUILD_SCRIPT"
rm "${BUILD_SCRIPT}.bak"
echo "  ✓ Updated $BUILD_SCRIPT"

echo ""
echo "✓ Version bump complete!"
echo ""
echo "Updated files:"
echo "  - python/src/beaver/__init__.py"
echo "  - All examples/*/workflow.nf files"
echo "  - docker/build.sh"
echo ""
echo "Next steps:"
echo "  git add -A"
echo "  git commit -m \"Bump version to ${NEW_VERSION}\""
echo "  git tag v${NEW_VERSION}"
echo "  git push origin main --tags"
