#!/usr/bin/env bash
# Same fixed CPU experiment; Linux/GPU execution remains NOT_VERIFIED locally.
# Supply a protocol frozen on the target host with its actual absolute paths.
set -euo pipefail
if [[ $# -ne 3 ]]; then
  echo 'Usage: bash run_phase04R1_linux.sh GRAPH_PYTHON TARGET_PROTOCOL_JSON TRUSTED_PROTOCOL_SHA256' >&2
  exit 2
fi
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONHASHSEED=20261101
"$1" -m cwm_fp.models.train_graph --mode package_compare --experiment-protocol "$2" --experiment-sha256 "$3"
stats_python="$("$1" -c 'import json,sys; print(json.load(open(sys.argv[1]))["paths"]["stats_python"])' "$2")"
index="$("$1" -c 'import json,sys,pathlib; print(pathlib.Path(json.load(open(sys.argv[1]))["paths"]["out"])/"prediction_index.json")' "$2")"
"$stats_python" -m cwm_fp.models.graph_evaluator --mode package_validation \
  --experiment-protocol "$2" --experiment-sha256 "$3" --prediction-index "$index"
