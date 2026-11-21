#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${PYTHON_BIN:-python}" "$HERE/experiment.py" --mode smoke "$@"
