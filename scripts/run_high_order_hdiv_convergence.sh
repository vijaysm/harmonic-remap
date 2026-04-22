#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 <build_dir> [csv_output]" >&2
  exit 1
fi

build_dir="$1"
csv_output="${2:-docs/high_order_hdiv_convergence.csv}"

"${build_dir}/high_order_hdiv_convergence_test" "${csv_output}"
