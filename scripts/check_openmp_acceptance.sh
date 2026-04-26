#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${1:-$ROOT_DIR/build-omp-release}"

cmake -S "$ROOT_DIR" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DMIMETIC_ENABLE_OPENMP=ON \
  -DOpenMP_ROOT=/opt/homebrew/Cellar/libomp/21.1.3

cmake --build "$BUILD_DIR" --parallel \
  --target spherical_high_order_moment_test spherical_high_order_hdiv_convergence_test

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/mimetic-omp-check.XXXXXX")"
trap 'rm -rf "$TMP_DIR"' EXIT

run_case() {
  local threads="$1"
  local label="$2"
  shift 2
  local exe="$1"
  shift
  local out="$TMP_DIR/${label}.t${threads}.out"
  local timing="$TMP_DIR/${label}.t${threads}.time"
  /usr/bin/time -p -o "$timing" \
    env OMP_NUM_THREADS="$threads" OMP_PROC_BIND=close OMP_PLACES=cores \
    "$exe" "$@" > "$out"
}

compare_case() {
  local label="$1"
  local base="$TMP_DIR/${label}.t1.out"
  for threads in 2 4; do
    local other="$TMP_DIR/${label}.t${threads}.out"
    if cmp -s "$base" "$other"; then
      continue
    fi
    if ! cmp -s "$base" "$other"; then
      echo "[FAILED] Non-deterministic output for ${label} between 1 and ${threads} threads" >&2
      diff -u "$base" "$other" || true
      exit 1
    fi
  done
}

print_timing() {
  local label="$1"
  echo "== ${label} =="
  for threads in 1 2 4; do
    local timing="$TMP_DIR/${label}.t${threads}.time"
    local real_time
    real_time="$(awk '/^real / {print $2}' "$timing")"
    echo "  threads=${threads} real=${real_time}s"
  done
}

tests=(
  "spherical_high_order_moment_test"
  "spherical_high_order_hdiv_convergence_test"
)

for test_name in "${tests[@]}"; do
  exe="$BUILD_DIR/$test_name"
  for threads in 1 2 4; do
    run_case "$threads" "$test_name" "$exe"
  done
  compare_case "$test_name"
  print_timing "$test_name"
done

echo "[SUCCESS] OpenMP outputs are deterministic for 1/2/4 threads."
