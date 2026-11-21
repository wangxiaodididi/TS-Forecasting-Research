#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${SOURCES_DIR:-$ROOT/sources}"
mkdir -p "$DEST"

fetch_repo() {
  local repo="$1" folder="$2" sha="$3" main_file="$4"
  if [ -f "$DEST/$folder/$main_file" ]; then
    echo "[OK] existing source $DEST/$folder"
    return
  fi
  local tmp
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' RETURN
  local url="https://codeload.github.com/$repo/zip/$sha"
  echo "[FETCH] $url"
  if ! curl --fail --location --retry 3 --connect-timeout 15 --max-time 240 "$url" -o "$tmp/source.zip"; then
    echo "[ERROR] GitHub archive unavailable. Re-run when network is available or extract the official source into $DEST/$folder" >&2
    return 1
  fi
  unzip -q "$tmp/source.zip" -d "$tmp/unpacked"
  local top
  top="$(find "$tmp/unpacked" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
  [ -n "$top" ] || { echo "[ERROR] empty GitHub archive" >&2; return 1; }
  rm -rf "$DEST/$folder"
  mv "$top" "$DEST/$folder"
  [ -f "$DEST/$folder/$main_file" ] || { echo "[ERROR] expected file missing: $main_file" >&2; return 1; }
  echo "[OK] pinned $repo @ $sha"
}

# These are read-only upstream snapshots. Do not edit them in place.
fetch_repo "TSTS13/Paloma" "Paloma" "def3dfc612bd6554be8f2ed7d5174848d6a37119" "models/Paloma.py"
fetch_repo "aitianma/MixLinear" "MixLinear" "42dbb98a5bbe64c13bc75b3cc07a9dc4acf20106" "models/MixLinear.py"

cat <<MSG
[OK] Sources prepared in $DEST
[NOTE] MixLinear repository has no declared license in its GitHub metadata. Keep downloaded code for local research; do not redistribute it without permission.
MSG
