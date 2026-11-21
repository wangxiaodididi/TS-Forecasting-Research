#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${PYTHON_EXEC:-python}" "$ROOT/scripts/audit_source.py"
