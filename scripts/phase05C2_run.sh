#!/usr/bin/env bash
set -euo pipefail
root=/workspace/codes/wanghaining_cwm_fp/phase05C2_v1
py=/workspace/codes/wanghaining_cwm_fp/phase05C1_v1/environment/venv/bin/python
code="$root/$2"
export CUDA_VISIBLE_DEVICES=""
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$root/environment/tmp"
export PYTHONPATH="$code/src:$code/tests"
export CWM_FP_PREFIX_TEST_SOURCE=/workspace/codes/wanghaining_cwm_fp/phase05C1_v1/snapshot/data/05A
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
case "$1" in
  tests)
    "$py" "$code/scripts/phase05C1_execute.py" --root "$root" --cwd "$code" --name "tests_$2" -- "$py" -m unittest discover -s tests -p 'test_local_dataset.py' -v ;;
  regressions)
    "$py" "$code/scripts/phase05C1_execute.py" --root "$root" --cwd "$code" --name "regressions_$2" -- "$py" "$code/scripts/phase05C2_regressions.py" ;;
  produce)
    "$py" "$code/scripts/phase05C1_execute.py" --root "$root" --cwd "$code" --name "production_$2" -- "$py" "$code/scripts/phase05C2_produce.py" --source "$root/inputs" --anchor 23e96a1bfc51a218b7ba50a1840c27a14e1d6cad9c1ad36aa2a45dbd498b0b01 --out "$root/production_$2" ;;
  verify)
    "$py" "$code/scripts/phase05C1_execute.py" --root "$root" --cwd "$code" --name "verification_$2" -- "$py" "$code/scripts/phase05C2_verify.py" --root "$root" --production production_code_v2 --out "verification_$2" ;;
  report)
    "$py" "$code/scripts/phase05C1_execute.py" --root "$root" --cwd "$code" --name "report_$2" -- "$py" "$code/scripts/phase05C2_report.py" --root "$root" ;;
  *) exit 2 ;;
esac
