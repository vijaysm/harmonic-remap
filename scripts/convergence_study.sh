#!/bin/bash
set -euo pipefail

BUILD_DIR="${1:-/tmp/mimetic-sphpoly-build}"
BIN="$BUILD_DIR/spherical_quad_test"

if [ ! -x "$BIN" ]; then
    echo "ERROR: $BIN not found. Build first." >&2
    exit 1
fi

echo "source_n,target_n,edge_flux_l2_rel,edge_flux_linf,direct_cell_avg_l1,direct_cell_avg_linf,recon_cell_avg_l1,recon_cell_avg_linf"

for src_n in 3 4 6 8 12 16; do
    tgt_n=$((src_n + 2))
    output=$($BIN "$src_n" "$tgt_n" "/tmp/conv_${src_n}_${tgt_n}" 2>&1)
    l2_rel=$(printf '%s\n' "$output" | grep edge_flux_l2_rel | sed 's/.*edge_flux_l2_rel=//;s/ .*//')
    linf=$(printf '%s\n' "$output" | grep edge_flux_linf | sed 's/.*edge_flux_linf=//;s/ .*//')
    direct_l1=$(printf '%s\n' "$output" | grep direct_cell_avg_l1 | sed 's/.*direct_cell_avg_l1=//;s/ .*//')
    direct_linf=$(printf '%s\n' "$output" | grep direct_cell_avg_linf | sed 's/.*direct_cell_avg_linf=//;s/ .*//')
    recon_l1=$(printf '%s\n' "$output" | grep recon_cell_avg_l1 | sed 's/.*recon_cell_avg_l1=//;s/ .*//')
    recon_linf=$(printf '%s\n' "$output" | grep recon_cell_avg_linf | sed 's/.*recon_cell_avg_linf=//;s/ .*//')
    echo "$src_n,$tgt_n,$l2_rel,$linf,$direct_l1,$direct_linf,$recon_l1,$recon_linf"
done
