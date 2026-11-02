#!/usr/bin/env bash
# Same bounded acceptance entry. This file has NOT been executed on Linux/GPU.
set -euo pipefail
usage() {
  echo 'Usage: bash run_phase04T1_linux.sh PYTHON PACKAGE MANIFEST_SHA PRODUCTS TESTS SPLITS SPLIT_SHA TRAIN_CSV TRAIN_SHA NEW_OUTPUT [cpu|cuda]'
  echo 'Use independent, previously frozen hashes. Only grouped/train.csv is permitted. Output must be a new run directory.'
}
if [[ "${1:-}" == "--help" ]]; then usage; exit 0; fi
if [[ $# -lt 10 || $# -gt 11 ]]; then usage >&2; exit 2; fi
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONHASHSEED=20261101
# Required by deterministic CUDA matrix operations; no driver or environment installation.
export CUBLAS_WORKSPACE_CONFIG=:4096:8
exec "$1" -m cwm_fp.models.train_graph \
  --mode package_smoke --package "$2" --manifest-sha256 "$3" \
  --products "$4" --tests "$5" --splits "$6" --split-sha256 "$7" \
  --train-labels "$8" --train-label-sha256 "$9" --out "${10}" \
  --protocol grouped_bank_pilot_v1 --partition train --fold grouped \
  --kind rgcn --feature-schema semantic_identity_v2 --relation-mode full --ablation full \
  --device "${11:-cpu}" --seed 20261101 --max-updates 400 --max-model-seconds 1740 \
  --epochs 10 --micro-steps 100 --synthetic-steps 40 --batch-products 2
