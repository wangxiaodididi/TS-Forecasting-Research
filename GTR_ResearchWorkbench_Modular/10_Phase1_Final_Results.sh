#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
echo 'WARNING: Use only after validation-based selection and model lock.'
./.venv/bin/python -m phase1.launch summary --include-test
