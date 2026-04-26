#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${1:-$ROOT_DIR/build}"
WORKERS="${2:-4}"

cmake -S "$ROOT_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" --parallel --target dump_visuals

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/mimetic-mercator.XXXXXX")"
trap 'rm -rf "$TMP_DIR"' EXIT

echo "Preparing exact source field..."
"$BUILD_DIR/dump_visuals" --mercator-only --mercator-case=source > "$TMP_DIR/source.log"

cases=(
  cs_p1
  vor_p1
  cs_p2
  vor_p2
  cs_p3
  vor_p3
)

printf '%s\n' "${cases[@]}" | xargs -n 1 -P "$WORKERS" -I {} \
  /bin/zsh -lc "\"$BUILD_DIR/dump_visuals\" --mercator-only --mercator-case={} > \"$TMP_DIR/{}.log\""

for case_name in "${cases[@]}"; do
  echo "== ${case_name} =="
  tail -n 4 "$TMP_DIR/${case_name}.log"
done

echo "[SUCCESS] Parallel Mercator round-trip artifacts written to repository root."
