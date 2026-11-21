#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
if [[ ! -x "$PY" ]]; then PY=python; fi
"$PY" -m lab.cli install-official
"$PY" -m lab.cli verify-official
"$PY" -m pip install -r vendor/GTR/requirements.txt
printf '\nGTR source repair succeeded. You can start the dashboard now.\n'
