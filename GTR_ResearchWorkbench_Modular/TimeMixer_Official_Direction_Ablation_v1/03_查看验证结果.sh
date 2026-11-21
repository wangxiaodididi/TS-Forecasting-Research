#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/../.venv/bin/python}"
[[ -x "$PYTHON" ]] || PYTHON=python
"$PYTHON" summary.py
