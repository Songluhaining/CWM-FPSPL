"""Bounded phase-01 acceptance run; inventories files, never runs Java or models."""
from __future__ import annotations

import datetime
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cwm_fp.data.benchmark_adapter import load_sources
from cwm_fp.data.common import sha256, write_json


def main() -> int:
    out = ROOT / "outputs"
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1")
    config = ROOT / "configs/data_sources.local.json"
    sources = load_sources(config)
    # Fingerprint only evidence actually consumed: reports and label tables.
    evidence = []
    for source in sources:
        evidence.extend(source.root.glob("*/config.report.csv"))
        evidence.extend(source.root.glob("*/config.report.csv.done"))
        if source.label_root:
            evidence.extend(source.label_root.glob("*/config.report.csv"))
            evidence.extend(source.label_root.glob("*/config.report.csv.done"))
            evidence.extend(source.label_root.glob("*/variant_labels.csv"))
    before = {p.as_posix(): sha256(p) for p in sorted(evidence)}
    record = {"started_at": datetime.datetime.now().astimezone().isoformat(), "cwd": str(ROOT),
              "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
              "environment": {k: env[k] for k in ("PYTHONPATH", "PYTHONUTF8", "PYTHONDONTWRITEBYTECODE")},
              "scope": "phase01 only; no Java, feature fitting, baseline or model training",
              "commands": [], "input_evidence_sha256_before": before}
    def run(name, args, timeout=300):
        command = [sys.executable, "-X", "utf8", "-B", *args]
        start = time.perf_counter()
        completed = subprocess.run(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8", timeout=timeout)
        path = out / f"phase01_{name}.log"
        path.write_text(completed.stdout, encoding="utf-8")
        record["commands"].append({"name": name, "argv": command, "exit_code": completed.returncode,
                                   "seconds": round(time.perf_counter() - start, 3),
                                   "log": path.as_posix(), "log_sha256": sha256(path), "timeout_seconds": timeout})
        write_json(out / "phase01_run.json", record)
        print(f"{name}: exit={completed.returncode}; {record['commands'][-1]['seconds']}s; {path.name}", flush=True)
        if completed.returncode:
            print(completed.stdout[-3000:])
            raise RuntimeError(f"Acceptance command failed: {name}")

    adapter = ["-m", "cwm_fp.data.benchmark_adapter", "--sources", "configs/data_sources.local.json"]
    smoke = "data/processed/manifests/smoke"
    full = "data/processed/manifests"
    splitdir = "data/processed/splits"
    files = [f"{splitdir}/splits.csv", f"{splitdir}/loso_BankAccountTP.csv", f"{splitdir}/loso_Email.csv"]
    try:
        run("unit_tests", ["-m", "unittest", "discover", "-s", "tests", "-v"], 120)
        run("pip_freeze", ["-m", "pip", "freeze"], 60)
        run("smoke_manifest", [*adapter, "--out", smoke, "--smoke", "--max-versions", "1", "--max-products", "3"])
        run("smoke_labels", ["-m", "cwm_fp.data.labels", "--products", f"{smoke}/products.csv", "--sources", "configs/data_sources.local.json", "--out", "data/processed/labels/smoke"])
        run("smoke_validation", ["scripts/validate_contract.py", "--products", f"{smoke}/products.csv", "--tests", f"{smoke}/tests.csv", "--labels", "data/processed/labels/smoke/labels.csv", "--check-paths", "--out", "outputs/phase01_smoke_validation.json"])
        run("smoke_splits", ["-m", "cwm_fp.data.splits", "--products", f"{smoke}/products.csv", "--tests", f"{smoke}/tests.csv", "--out", "data/processed/splits/smoke", "--seed", "20261030"])
        run("full_manifest", [*adapter, "--out", full, "--max-versions", "109", "--max-products", "3454"])
        split_args = ["-m", "cwm_fp.data.splits", "--products", f"{full}/products.csv", "--tests", f"{full}/tests.csv", "--out", splitdir, "--seed", "20261030"]
        run("full_splits", split_args)
        hashes = {p: sha256(ROOT / p) for p in files}
        run("split_reproducibility", split_args)
        record["split_reproducible"] = hashes == {p: sha256(ROOT / p) for p in files}
        if not record["split_reproducible"]:
            raise RuntimeError("Split bytes changed with the same inputs/seed")
        run("full_labels", ["-m", "cwm_fp.data.labels", "--products", f"{full}/products.csv", "--sources", "configs/data_sources.local.json", "--out", "data/processed/labels/full", "--splits", *files])
        run("full_validation", ["scripts/validate_contract.py", "--products", f"{full}/products.csv", "--tests", f"{full}/tests.csv", "--labels", "data/processed/labels/full/labels.csv", "--splits", *files, "--check-paths", "--out", "outputs/phase01_validation.json"])
        record["input_evidence_sha256_after"] = {p.as_posix(): sha256(p) for p in sorted(evidence)}
        record["consumed_source_files_unchanged"] = record["input_evidence_sha256_after"] == before
        if not record["consumed_source_files_unchanged"]:
            raise RuntimeError("Consumed source evidence changed during the acceptance run")
        record["code_sha256"] = {p.relative_to(ROOT).as_posix(): sha256(p) for folder in ("src", "scripts", "tests") for p in sorted((ROOT / folder).rglob("*.py"))}
        record["completed_at"] = datetime.datetime.now().astimezone().isoformat()
        record["success"] = True
        write_json(out / "phase01_run.json", record)
        print("Phase 01 acceptance checks complete; source evidence unchanged; split bytes reproducible.")
        return 0
    except (RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
        record["success"] = False
        record["error"] = str(exc)
        write_json(out / "phase01_run.json", record)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
