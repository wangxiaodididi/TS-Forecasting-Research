#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python paloma_phase_replacement.py "$@"
