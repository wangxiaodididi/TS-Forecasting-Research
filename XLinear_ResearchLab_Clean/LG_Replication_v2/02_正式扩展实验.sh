#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
"${PYTHON:-python}" LG_Replication_v2/scripts/replicate.py run "$@"
