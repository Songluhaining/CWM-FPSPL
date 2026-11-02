"""Bounded phase04 checks, exact command/exit/log records, no source mutation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cwm_fp.data.common import sha256, write_json


def run(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--graph-python", type=Path, required=True)
    p.add_argument("--stage", choices=("tests", "smoke", "real"), required=True)
    a = p.parse_args(argv)
    env = os.environ.copy(); env["PYTHONPATH"] = str(ROOT / "src"); env["PYTHONUTF8"] = "1"
    graph = str(a.graph_python.resolve()); base = sys.executable
    commands = {
        "tests": [("base_tests", [base, "-m", "unittest", "discover", "-s", "tests", "-v"], 90, 0),
                  ("graph_tests", [graph, "-m", "unittest", "discover", "-s", "tests", "-p", "test_graph*.py", "-v"], 90, 0)],
        "smoke": [("synthetic", [graph, "-m", "cwm_fp.models.train_graph", "--mode", "smoke", "--config", "configs/graph_classifier.json"], 210, 0)],
        "real": [("readiness", [graph, "-m", "cwm_fp.models.train_graph", "--mode", "audit"], 45, 0)] +
                [("blocked_"+fold, [graph, "-m", "cwm_fp.models.train_graph", "--mode", "train", "--fold", fold], 45, 3)
                 for fold in ("grouped", "loso_BankAccountTP", "loso_Email")]
    }
    records = []
    report = ROOT / "outputs" / ("phase04_" + a.stage + "_run.json")
    for name, command, timeout, expected in commands[a.stage]:
        path = ROOT / "outputs" / ("phase04_" + name + ".log")
        start = time.monotonic()
        try:
            with path.open("w", encoding="utf-8") as f:
                r = subprocess.run(command, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
            code = r.returncode
        except subprocess.TimeoutExpired:
            code = "timeout"
        records.append({"command": command, "cwd": str(ROOT), "timeout_seconds": timeout,
                        "exit_code": code, "expected_exit_code": expected, "seconds": time.monotonic()-start,
                        "log": path.relative_to(ROOT).as_posix(), "log_sha256": sha256(path)})
        write_json(report, {"stage": a.stage, "records": records, "success": all(x["exit_code"] == x["expected_exit_code"] for x in records)})
        print(name, code, round(records[-1]["seconds"], 3), flush=True)
        if code != expected: return 1
    return 0


if __name__ == "__main__":
    sys.exit(run())
