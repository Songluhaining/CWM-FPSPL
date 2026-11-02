"""Read-only integrity checks; does not train or reevaluate a held-out score."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cwm_fp.data.common import sha256, write_json


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify():
    prior = read(ROOT / "outputs/phase03_artifacts.json")["files"]
    redirects = {"outputs/phase_status.md": "outputs/phase03_status.md", "DATA_CONTRACTS.md": "outputs/phase03_data_contracts.md"}
    unchanged = 0
    for relative, info in prior.items():
        path = ROOT / redirects.get(relative, relative)
        if sha256(path) != info["sha256"]: raise ValueError(f"Phase03 artifact changed: {relative}")
        unchanged += 1
    previous = read(ROOT / "outputs/phase02_artifacts.json")["files"]
    baseline = 0
    for relative, info in previous.items():
        if relative.startswith(("data/processed/baselines/", "outputs/metrics/phase02_v1/", "src/cwm_fp/baselines/")):
            if sha256(ROOT / relative) != info["sha256"]: raise ValueError(f"Phase02 baseline changed: {relative}")
            baseline += 1
    source_files = {}
    for r in read(ROOT / "data/processed/lddg/phase03_bank_v2/collection.json")["records"]:
        source_files.update(r["status"]["inputs"]["sources"])
    for path, expected in source_files.items():
        if sha256(Path(path)) != expected: raise ValueError(f"Original Java changed: {path}")
    producer = read(ROOT / "outputs/phase03_producer_provenance.json")
    for relative, expected in producer["files_sha256"].items():
        if sha256(Path(producer["source_root"]) / relative) != expected: raise ValueError("Original producer changed")
    config = read(ROOT / "configs/graph_classifier.json"); work = ROOT / config["work_dir"]
    for phase in ("tests", "smoke", "real"):
        result = read(ROOT / f"outputs/phase04_{phase}_run.json")
        if not result["success"]: raise ValueError("Phase04 command failed")
        for record in result["records"]:
            if record["log_sha256"] != sha256(ROOT / record["log"]): raise ValueError("Recorded command log changed")
    for relative, expected in read(work / "synthetic/source_fingerprint.json").items():
        if sha256(ROOT / relative) != expected: raise ValueError("Training source/config changed")
    training = read(work / "synthetic/training.json")
    evaluation = read(work / "synthetic/test_evaluation.json")
    pred_hash = sha256(work / "synthetic/frozen_test_predictions.json")
    if training["test_prediction_sha256"] != pred_hash or (work / "synthetic/test_evaluation.claim").read_text().strip() != pred_hash:
        raise ValueError("Synthetic prediction freeze changed")
    if len(training["runs"]) != 21 or len(evaluation["runs"]) != 21 or evaluation["test_evaluations_per_model_seed"] != 1:
        raise ValueError("Synthetic run count/once-only contract violated")
    for r in training["runs"]:
        if r["parameters"] > config["training"]["max_parameters"] or r["epochs"] > config["training"]["max_epochs"]:
            raise ValueError("Architecture/epoch budget exceeded")
        if r["kind"] != "coverage" and r["train_resubstitution"]["accuracy"] < .95:
            raise ValueError("Synthetic overfit check failed")
    readiness = read(work / "readiness.json")
    if readiness["status"] != "blocked" or readiness["real_training_runs"] or readiness["real_test_evaluations"]:
        raise ValueError("Unexpected empirical claim")
    for fold in config["folds"]:
        refusal = read(work / (fold + "_blocked.json"))
        if refusal["labels_read"] or refusal["ready"]: raise ValueError("Real training did not stop before targets")
    result = {"valid": True, "phase03_records_preserved": unchanged, "phase03_archive_redirects": redirects,
              "phase02_baseline_files_unchanged": baseline, "original_Java_files_unchanged": len(source_files),
              "original_instrumenter_files_unchanged": len(producer["files_sha256"]),
              "synthetic_model_seed_runs": len(training["runs"]), "separate_coverage_overfit": training["separate_coverage_overfit"]["train_resubstitution"]["accuracy"],
              "synthetic_test_evaluations_per_model_seed": 1, "real_training_runs": 0, "real_test_evaluations": 0,
              "new_Java_executions": 0, "source_fingerprints_match": True}
    write_json(ROOT / "outputs/phase04_verification.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    verify()
