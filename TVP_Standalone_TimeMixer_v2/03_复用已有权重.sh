#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
: "${CHECKPOINT:?Provide an exact existing checkpoint.pth path}"
: "${CONFIG_JSON:?Provide the matching training args JSON path}"
: "${OFFICIAL_ROOT:?Provide the matching official TimeMixer repository root}"
PYTHON="${PYTHON:-python}"
OUT="${OUT:-results/official_existing}"
"$PYTHON" -m tvp_lab.extract_official --family timemixer --official-root "$OFFICIAL_ROOT" \
  --config-json "$CONFIG_JSON" --checkpoint "$CHECKPOINT" \
  --data-root "$PWD/data" --data-file ETTm1.csv --batch-size 16 --device "${DEVICE:-cpu}" \
  --max-windows "${MAX_WINDOWS:-0}" --output-dir "$OUT/extracted"
"$PYTHON" -m tvp_lab.evaluate --val "$OUT/extracted/val.npz" \
  --test "$OUT/extracted/test.npz" --outdir "$OUT/compare"
