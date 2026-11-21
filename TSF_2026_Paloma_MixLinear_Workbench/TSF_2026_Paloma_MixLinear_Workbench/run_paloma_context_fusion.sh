#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec "$(command -v python)" paloma_context_fusion_v4.py "$@"
