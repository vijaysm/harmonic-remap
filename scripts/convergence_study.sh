#!/bin/bash
set -euo pipefail

BUILD_DIR="${1:-/tmp/mimetic-sphpoly-build}"
BIN="$BUILD_DIR/spherical_quad_test"

if [ ! -x "$BIN" ]; then
    echo "ERROR: $BIN not found. Build first." >&2
    exit 1
fi

echo "source_n,target_n,edge_flux_l2_rel,edge_flux_linf,cell_field_l1,cell_field_linf"

for src_n in 3 4 6 8 12 16; do
    tgt_n=$((src_n + 2))
    output=$($BIN "$src_n" "$tgt_n" "/tmp/conv_${src_n}_${tgt_n}" 2>&1)
    l2_rel=$(printf '%s\n' "$output" | grep edge_flux_l2_rel | sed 's/.*edge_flux_l2_rel=//;s/ .*//')
    linf=$(printf '%s\n' "$output" | grep edge_flux_linf | sed 's/.*edge_flux_linf=//;s/ .*//')
    l1=$(printf '%s\n' "$output" | grep cell_field_l1 | sed 's/.*cell_field_l1=//;s/ .*//')
    clinf=$(printf '%s\n' "$output" | grep cell_field_linf | sed 's/.*cell_field_linf=//;s/ .*//')
    echo "$src_n,$tgt_n,$l2_rel,$linf,$l1,$clinf"
done
