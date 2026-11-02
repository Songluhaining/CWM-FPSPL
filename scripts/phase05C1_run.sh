#!/usr/bin/env bash
set -eu
r=/workspace/codes/wanghaining_cwm_fp/phase05C1_v1
p="$r/environment/venv/bin/python"
c="$r/${2:-code_v2}"
export CUDA_VISIBLE_DEVICES=GPU-e611ba0c-db46-7d8d-486c-3fd79962b2ac
export TMPDIR="$r/environment/tmp" XDG_CACHE_HOME="$r/environment/cache" PYTHONDONTWRITEBYTECODE=1
export CWM_FP_PREFIX_TEST_SOURCE="$r/snapshot/data/05A"
run() { "$p" "$r/phase05C1_execute.py" --root "$r" --cwd "$c" --name "$1" -- "$p" "${@:2}"; }
case "$1" in
  tests)
    for name in local_state prefix_observer prefix_reader transition; do
      run "${2:-code_v2}_test_$name" -m unittest discover -s tests -p "test_$name.py" -v
    done
    run "${2:-code_v2}_old_help" scripts/phase05B_linux_acceptance.py --help
    ;;
  remaining_tests)
    for name in prefix_reader transition; do
      run "${2:-code_v2}_test_$name" -m unittest discover -s tests -p "test_$name.py" -v
    done
    run "${2:-code_v2}_old_help" scripts/phase05B_linux_acceptance.py --help
    ;;
  diagnose)
    run "${2:-code_v2}_diagnose" scripts/phase05C1_diagnose.py --source "$r/snapshot/data/05A" --cache-root "$r/snapshot/data/05B" --out "$r/diagnostic_${2:-code_v2}"
    ;;
  old_gpu)
    run "${2:-code_v2}_old_gpu" scripts/phase05B_linux_acceptance.py --source "$r/snapshot/data/05A" --cache-root "$r/snapshot/data/05B" --out "$r/old_gpu_${2:-code_v2}" --device cuda:0
    ;;
  gpu)
    run "${2:-code_v2}_gpu" scripts/phase05C1_gpu_acceptance.py --source "$r/snapshot/data/05A" --cache-root "$r/snapshot/data/05B" --out "$r/gpu_${2:-code_v2}" --device cuda:0 --prior-updates "${3:-1}"
    ;;
  remaining_gpu)
    run "${2:-code_v2}_remaining_gpu" scripts/phase05C1_gpu_acceptance.py --source "$r/snapshot/data/05A" --cache-root "$r/snapshot/data/05B" --out "$r/gpu_${2:-code_v2}" --device cuda:0 --prior-updates 78 --models no_dep rgcn
    ;;
  state_test)
    run "${2:-code_v2}_test_local_state" -m unittest discover -s tests -p test_local_state.py -v
    ;;
  *) echo 'Unknown bounded action' >&2; exit 2;;
esac
