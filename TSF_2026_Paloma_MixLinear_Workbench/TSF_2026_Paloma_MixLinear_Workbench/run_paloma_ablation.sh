#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${PYTHON:-}" ]; then
  exec "$PYTHON" "$HERE/paloma_ablation.py" "$@"
fi
exec python "$HERE/paloma_ablation.py" "$@"
