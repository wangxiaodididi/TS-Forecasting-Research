#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$(readlink -f "$0")")"
PYTHON="${PYTHON:-python}"
"$PYTHON" ops/install_hcf.py
"$PYTHON" -m tests.test_hcf_smoke
"$PYTHON" -m tests.test_hcf_official_integration
