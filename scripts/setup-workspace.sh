#!/bin/bash
set -euo pipefail

# Setup script for biovault-beaver workspace
# Clones dependencies to PARENT directory as siblings
#
# Dependencies:
#   - syftbox-sdk (for crypto/SDK features)
#   - syftbox (for Go server integration tests)
#
# In a repo-managed parent workspace (biovault-desktop), dependencies
# are already synced - this script detects that and exits early.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
PARENT_DIR="$(dirname "$REPO_ROOT")"

echo "Setting up biovault-beaver workspace..."
echo "  REPO_ROOT: $REPO_ROOT"
echo "  PARENT_DIR: $PARENT_DIR"

# Configure git to use HTTPS instead of SSH for GitHub (needed for CI)
git config --global url."https://github.com/".insteadOf "git@github.com:"

# Check if we're in a repo-managed workspace (parent has .repo)
if [[ -d "$PARENT_DIR/.repo" ]]; then
    echo "Detected repo-managed parent workspace - dependencies already synced"
    exit 0
fi

# Clone syftbox-sdk (required) to parent directory
if [[ -d "$PARENT_DIR/syftbox-sdk" ]]; then
    echo "syftbox-sdk already exists at $PARENT_DIR/syftbox-sdk"
elif [[ -L "$REPO_ROOT/syftbox-sdk" ]]; then
    echo "Removing stale syftbox-sdk symlink..."
    rm -f "$REPO_ROOT/syftbox-sdk"
    echo "Cloning syftbox-sdk to $PARENT_DIR/syftbox-sdk..."
    git clone https://github.com/OpenMined/syftbox-sdk.git "$PARENT_DIR/syftbox-sdk"
else
    echo "Cloning syftbox-sdk to $PARENT_DIR/syftbox-sdk..."
    git clone https://github.com/OpenMined/syftbox-sdk.git "$PARENT_DIR/syftbox-sdk"
fi

# Setup syftbox-sdk's own dependencies
if [[ -f "$PARENT_DIR/syftbox-sdk/scripts/setup-workspace.sh" ]]; then
    echo "Setting up syftbox-sdk dependencies..."
    chmod +x "$PARENT_DIR/syftbox-sdk/scripts/setup-workspace.sh"
    (cd "$PARENT_DIR/syftbox-sdk" && ./scripts/setup-workspace.sh)
fi

# Clone syftbox (for Go server) to parent directory
if [[ -d "$PARENT_DIR/syftbox" ]]; then
    echo "syftbox already exists at $PARENT_DIR/syftbox"
elif [[ -L "$REPO_ROOT/syftbox" ]]; then
    echo "Removing stale syftbox symlink..."
    rm -f "$REPO_ROOT/syftbox"
    echo "Cloning syftbox to $PARENT_DIR/syftbox..."
    git clone -b madhava/biovault https://github.com/OpenMined/syftbox.git "$PARENT_DIR/syftbox"
else
    echo "Cloning syftbox to $PARENT_DIR/syftbox..."
    git clone -b madhava/biovault https://github.com/OpenMined/syftbox.git "$PARENT_DIR/syftbox"
fi

# Create symlinks from repo root to parent deps (for code that expects ./syftbox-sdk)
if [[ ! -e "$REPO_ROOT/syftbox-sdk" ]]; then
    ln -s ../syftbox-sdk "$REPO_ROOT/syftbox-sdk"
    echo "Created symlink: syftbox-sdk -> ../syftbox-sdk"
fi

if [[ ! -e "$REPO_ROOT/syftbox" ]]; then
    ln -s ../syftbox "$REPO_ROOT/syftbox"
    echo "Created symlink: syftbox -> ../syftbox"
fi

echo ""
echo "Workspace setup complete!"
echo "Dependencies are at:"
echo "  $PARENT_DIR/syftbox-sdk"
echo "  $PARENT_DIR/syftbox"
