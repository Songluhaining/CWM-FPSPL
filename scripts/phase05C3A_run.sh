#!/usr/bin/env bash
set -euo pipefail
root=/workspace/codes/wanghaining_cwm_fp/phase05C3A_v1
py=/workspace/codes/wanghaining_cwm_fp/phase05C1_v1/environment/venv/bin/python
code="$root/$2"
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$root/environment/tmp" PYTHONPATH="$code/src:$code/tests"
export CWM_FP_PREFIX_TEST_SOURCE=/workspace/codes/wanghaining_cwm_fp/phase05C1_v1/snapshot/data/05A
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
monitor=("$py" "$code/scripts/phase05C1_execute.py" --root "$root" --cwd "$code" --name "$1_$2" --max-rss-bytes 4294967296 --max-stage-bytes 2147483648 --)
case "$1" in
  runtime) "${monitor[@]}" "$py" "$code/scripts/phase05C3A_runtime.py" --root "$root" --out "runtime_$2" ;;
  tests) "${monitor[@]}" "$py" -m unittest discover -s tests -p 'test_field_evidence.py' -v ;;
  regressions) "${monitor[@]}" "$py" "$code/scripts/phase05C3A_regressions.py" ;;
  audit) "${monitor[@]}" "$py" "$code/scripts/phase05C3A_audit.py" --root "$root" --out "audit_$2" ;;
  report) "${monitor[@]}" "$py" "$code/scripts/phase05C3A_report.py" --root "$root" ;;
  *) exit 2 ;;
esac
