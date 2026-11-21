#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$ROOT/official/Sonnet"
PY="${PYTHON_EXEC:-python}"
OFFICIAL_SHA="${OFFICIAL_SHA:-bf3d4801d34c5e7261718490f287c6fb15cadfdb}"
mkdir -p "$ROOT/official" "$ROOT/runs" "$ROOT/results"
echo "[1/4] Checking Python/PyTorch"
"$PY" - <<'PY'
import sys
print('Python:', sys.version.split()[0])
try:
 import torch
 print('Torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())
except Exception as exc:
 print('PyTorch not installed:', repr(exc))
PY
if [[ ! -f "$REPO/sonnet/mts_model/models/Sonnet.py" ]]; then
  if [[ -d "$REPO" ]]; then
    echo "[ERROR] $REPO exists but is incomplete. Remove it or provide official source." >&2
    exit 1
  fi
  echo "[2/4] Fetching official source; network required only on your remote machine"
  if command -v git >/dev/null 2>&1; then
    git clone --depth 1 https://github.com/ClaudiaShu/Sonnet.git "$REPO" || true
    if [[ -d "$REPO/.git" ]]; then
      if git -C "$REPO" fetch --depth 1 origin "$OFFICIAL_SHA" && git -C "$REPO" checkout --detach "$OFFICIAL_SHA"; then
        echo "Pinned source checkout OK"
      else
        echo "Git checkout failed; trying archive instead" >&2
        rm -rf "$REPO"
      fi
    fi
  fi
  if [[ ! -f "$REPO/sonnet/mts_model/models/Sonnet.py" ]]; then
    rm -rf "$REPO"
    ARCH="$ROOT/official/sonnet_main.zip"
    curl -fL --retry 3 --connect-timeout 15 "https://github.com/ClaudiaShu/Sonnet/archive/${OFFICIAL_SHA}.zip" -o "$ARCH"
    "$PY" - "$ARCH" "$ROOT/official" <<'PY'
import sys, zipfile, pathlib, shutil
p,root=sys.argv[1:]
with zipfile.ZipFile(p) as z: z.extractall(root)
matches=list(pathlib.Path(root).glob('Sonnet-*'))
if len(matches)!=1: raise RuntimeError(f'Unexpected archive roots: {matches}')
path=matches[0]
target=pathlib.Path(root)/'Sonnet'
if not path.exists(): raise RuntimeError('Expected Sonnet archive root')
shutil.move(path,target)
PY
  fi
else
  echo "[2/4] Official source already present (not modified)"
fi
if [[ ! -f "$REPO/sonnet/mts_model/models/Sonnet.py" ]]; then
 echo "[ERROR] Official model missing" >&2; exit 1
fi
if command -v git >/dev/null 2>&1 && [[ -d "$REPO/.git" ]]; then
  git -C "$REPO" rev-parse HEAD > "$ROOT/official_revision.txt"
  echo "Official commit: $(cat "$ROOT/official_revision.txt")"
else
  echo "$OFFICIAL_SHA (archive)" > "$ROOT/official_revision.txt"
  echo "Official source fetched from pinned archive $OFFICIAL_SHA"
fi
if [[ "${INSTALL_DEPS:-0}" == "1" ]]; then
  echo "[3/4] Installing supporting dependencies without forcing a different torch wheel"
  "$PY" -m pip install 'lightning>=2.4,<2.7' 'hydra-core>=1.3,<1.4' 'darts>=0.35' 'gluonts>=0.14' 'einops>=0.7' 'timm>=1.0' 'reformer-pytorch>=1.4' 'PyWavelets>=1.6' 'scikit-learn>=1.2' pandas scipy matplotlib rich
else
  echo "[3/4] Dependency installation skipped. Set INSTALL_DEPS=1 to install."
fi
printf '%s\n' '[4/4] Checking imports'
PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" "$PY" - <<'PY'
import importlib.util
for name in ['torch','lightning','darts','hydra','gluonts','einops','timm','reformer_pytorch','sklearn','scipy']:
 print(f'  {name:20}', 'OK' if importlib.util.find_spec(name) else 'MISSING')
PY
printf '\nRun: bash 01_smoke_cpu.sh\n'
