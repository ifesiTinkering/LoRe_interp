#!/bin/bash
# Setup script for APA project using uv.
# Source (don't execute) from the repo root:  source setup_uv.sh

set -e

# Use the repo containing this script as the working directory, so the venv
# lands next to the right pyproject.toml regardless of where this is sourced.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# Install uv if not present
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Create venv with Python 3.10/3.11 (torch wheels don't yet cover 3.14)
uv venv --python 3.11 2>/dev/null || uv venv --python 3.10 2>/dev/null || uv venv --python 3.9

# Sync dependencies
uv sync

echo ""
echo "Setup complete! Activate with:"
echo "  source $REPO_ROOT/.venv/bin/activate"
echo ""
echo "Or run directly with:"
echo "  uv run python tests/test_imports.py"
