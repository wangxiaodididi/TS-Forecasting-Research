#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/../.venv/bin/python}"
[[ -x "$PYTHON" ]] || PYTHON=python
"$PYTHON" tests/test_local_structure.py
"$PYTHON" tests/test_official_integration.py
