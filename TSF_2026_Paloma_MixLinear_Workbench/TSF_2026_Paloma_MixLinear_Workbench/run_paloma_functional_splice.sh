#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec "$(command -v python)" paloma_functional_splice.py "$@"
