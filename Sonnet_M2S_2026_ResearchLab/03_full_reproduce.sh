#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$ROOT/official/Sonnet"
if [[ ! -f "$REPO/scripts/run_experiment.py" ]]; then
  echo 'Official source missing. Run bash 00_setup.sh first.' >&2; exit 1
fi
DATASET="${1:-energy}"
H="${2:-24}"
shift $(( $# >= 2 ? 2 : $# ))
case "$DATASET" in
  energy)
    case "$H" in 24|48|72|168) ;; *) echo 'Energy supports H=24,48,72,168'; exit 2;; esac
    DATA_ARGS=(dataset=exp_data_config/energy exp=energy)
    SEQ=24
    ;;
  etth1)
    case "$H" in 96|192|336|720) ;; *) echo 'ETTh1 supports H=96,192,336,720'; exit 2;; esac
    DATA_ARGS=(dataset=exp_data_config/etth1 exp=etth)
    SEQ=336
    ;;
  *) echo 'Usage: bash 03_full_reproduce.sh energy|etth1 H [Hydra overrides...]' >&2; exit 2 ;;
esac
PY="${PYTHON_EXEC:-python}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
RUN_ID="$(date '+%Y%m%d_%H%M%S')_${DATASET}_H${H}"
mkdir -p "$ROOT/runs/$RUN_ID"
OUT="$ROOT/runs/$RUN_ID"
printf '%q ' "$PY" scripts/run_experiment.py model=sonnet "${DATA_ARGS[@]}" "exp.seq_length=$SEQ" "exp.pred_length=$H" "$@" > "$OUT/command.txt"
{
  echo "Dataset $DATASET  H $H  SEQ $SEQ"
  echo "Repo revision: $(cat "$ROOT/official_revision.txt" 2>/dev/null || echo unknown)"
  "$PY" - <<'PY'
import torch, sys
print('Python',sys.version.split()[0], 'Torch',torch.__version__,'CUDA',torch.cuda.is_available())
PY
} | tee "$OUT/environment.txt"
cd "$REPO"
# Baseline entry directly invokes the unmodified official training script.
"$PY" scripts/run_experiment.py model=sonnet "${DATA_ARGS[@]}" "exp.seq_length=$SEQ" "exp.pred_length=$H" "$@" 2>&1 | tee "$OUT/train.log"
echo "Results/metadata: $OUT"
