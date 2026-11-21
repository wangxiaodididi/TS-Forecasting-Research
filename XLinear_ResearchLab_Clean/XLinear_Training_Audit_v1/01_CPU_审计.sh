#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT"
PYTHON_BIN="${PYTHON_BIN:-python}"
echo "Project: $PROJECT"
echo "Python: $(command -v "$PYTHON_BIN")"
"$PYTHON_BIN" "$SCRIPT_DIR/scripts/audit.py" environment
"$PYTHON_BIN" "$SCRIPT_DIR/scripts/audit.py" history
"$PYTHON_BIN" "$SCRIPT_DIR/scripts/audit.py" structure
