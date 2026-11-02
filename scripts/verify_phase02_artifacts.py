"""Read-only evaluator-side consistency checks; no fitting or metric reselection."""
from __future__ import annotations

import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cwm_fp.baselines.boundaries import install_io_guard
from cwm_fp.baselines.evaluator import RESULT_FIELDS
from cwm_fp.baselines.pipeline import SCORE_FIELDS, configuration, context, load_features, split_file
from cwm_fp.data.common import KEY, indexed, read_csv, sha256, write_json


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check(condition, message):
    if not condition:
        raise ValueError(message)


def verify():
    install_io_guard("evaluator")
    c = configuration(ROOT / "configs/baselines.yaml")
    previous = load(ROOT / "outputs/phase01_artifacts.json")["files"]
    unchanged = [name for name in previous if name.startswith(("data/", "src/cwm_fp/data/", "configs/data_sources"))]
    for name in unchanged:
        check(sha256(ROOT / name) == previous[name]["sha256"], f"Phase01 input changed: {name}")
    vendors = {}
    for name, source in load(ROOT / "third_party/provenance.json").items():
        for relative, expected in source["files_sha256"].items():
            for parent in (ROOT / "third_party" / name, Path(source["source"])):
                check(sha256(parent / relative) == expected, f"Vendor changed: {parent / relative}")
        # Command-local exception for these already audited local repositories;
        # do not change global Git configuration or repository contents.
        git = ["git", "-c", "safe.directory=" + source["source"], "-C", source["source"]]
        commit = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
        status = subprocess.check_output([*git, "status", "--porcelain"], text=True).strip()
        check(commit == source["commit"] and not status, f"Vendor repository changed: {name}")
        vendors[name] = {"files": len(source["files_sha256"]), "commit": commit, "git_status_clean": True}
    evidence = load(c["work_dir"] / "observations_audit.json")
    check(sha256(c["work_dir"] / "observations.joblib") == evidence["snapshot_sha256"], "Observation snapshot changed")
    for k in ("products", "tests", "sources"):
        check(sha256(c[k]) == evidence[k + "_sha256"], f"Observation {k} mismatch")
    run = load(ROOT / "outputs/phase02_run.json")
    check(run["success"] and len(run["commands"]) == 5, "Incomplete pipeline run")
    for entry in run["commands"]:
        check(entry["exit_code"] == 0 and sha256(Path(entry["log"])) == entry["sha256"], "Stage log changed")
    for stage in run["commands"][1:]:
        for name, expected in stage["code_sha256"].items():
            check(sha256(ROOT / name) == expected, f"Post-prepare code changed: {name}")

    folds, files, total = {}, 0, 0
    for fold in c["folds"]:
        products, split, ordered = context(c, fold)
        split_id = fold + ":" + sha256(split_file(c, fold))
        by_sample = {p["sample_id"]: k for k, p in products.items()}
        expected_pass = {sid for sid, k in by_sample.items() if products[k]["observed_result"] == "PASS"}
        arrays = load_features(c, fold, ordered)
        dest = c["work_dir"] / fold
        feature_audit, scaler_audit = load(dest / "features_audit.json"), load(dest / "scaler_audit.json")
        training = {k for k in ordered if split[k]["split"] == "train"}
        for audit in (feature_audit, scaler_audit):
            check({tuple(k) for k in audit["training_keys"]} == training, f"Non-training contributor: {fold}")
        train_indices = [i for i, k in enumerate(ordered) if k in training]
        for family, values in arrays.items():
            stored = scaler_audit["scalers"][family]
            check(np.array_equal(values[train_indices].min(axis=0), stored["minimum"]), "Scaler minimum not train-only")
            check(np.array_equal(values[train_indices].max(axis=0), stored["maximum"]), "Scaler maximum not train-only")
        test_indices = [i for i, k in enumerate(ordered) if split[k]["split"] == "test"]
        all_zero = bool((arrays["pulp50"][test_indices] == 0).all())
        if fold.startswith("loso_"):
            check(all_zero, "Unexpected cross-system statement prior")
        labels = {}
        for partition in ("train", "val", "test"):
            labels.update(indexed(read_csv(c["labels"] / fold / (partition + ".csv"), KEY)))
        pass_counts = Counter(split[by_sample[sid]]["split"] for sid in expected_pass)
        label_counts = {part: dict(Counter(labels[by_sample[sid]]["label"] for sid in expected_pass
                                         if split[by_sample[sid]]["split"] == part)) for part in pass_counts}
        selected_c = {}
        for seed in c["seeds"]:
            source = dest / str(seed)
            selection = load(source / "selection.json")
            check(selection["test_labels_read"] is False, "Selection used test labels")
            selected_c[str(seed)] = {name: item["C"] for name, item in selection["models"].items()}
            for name in c["models"]:
                chosen = selection["models"][name]
                check(sha256(source / (chosen["trial_id"] + ".joblib")) == chosen["model_sha256"], "Selected model changed")
                result_path = c["metrics_dir"] / fold / str(seed) / (name + ".csv")
                rows = read_csv(result_path, RESULT_FIELDS)
                check(len(rows) == len(expected_pass) and {r["sample_id"] for r in rows} == expected_pass, "PASS coverage/duplicates")
                predictions_path = source / (name + "_predictions.csv")
                predictions = {r["sample_id"]: r for r in read_csv(predictions_path, SCORE_FIELDS)}
                metric = load(result_path.with_name(name + "_metrics.json"))
                check(sha256(predictions_path) == metric["predictions_sha256"], "Frozen predictions changed")
                for r in rows:
                    check(set(r) == set(RESULT_FIELDS), "Unexpected result field")
                    k = by_sample[r["sample_id"]]
                    check(r["system_id"] == k[0] and r["bug_version_id"] == k[1], "Identity mismatch")
                    check(r["partition"] == split[k]["split"] and r["split_id"] == split_id, "Split mismatch")
                    check(int(r["y_true"]) == int(labels[k]["label"] == "FP") and r["origin"] == labels[k]["origin"], "Truth/origin mismatch")
                    check(int(r["y_pred"]) in {0, 1} and math.isfinite(float(r["y_score"])), "Invalid predictions")
                    check(all(r[field] == predictions[r["sample_id"]][field] for field in SCORE_FIELDS), "Prediction changed when joining truth")
                for partition, n in pass_counts.items():
                    record = metric["partitions"][partition]
                    check(record["n"] == n, "Metric denominator mismatch")
                    check(record["exact_origin_strata"]["simulated"]["status"] == "unavailable", "Simulated result fabricated")
                files += 1
                total += len(rows)
        folds[fold] = {"PASS_counts": dict(pass_counts), "labels": label_counts, "training_references": len(training),
                       "test_pulp50_all_zero": all_zero, "selected_C": selected_c,
                       "feature_diagnostics": feature_audit["diagnostics"]}
    summary = load(c["metrics_dir"] / "summary.json")
    check(files == summary["runs"] == 45 and total == 99090, "Result count mismatch")
    return {"valid": True, "role": "evaluator_only_posthoc_consistency", "model_refits": 0, "test_metric_reselections": 0,
            "phase01_unchanged_files": len(unchanged), "vendors": vendors, "result_csv_files": files,
            "result_rows": total, "unique_original_PASS_products_per_run": 2202, "folds": folds,
            "raw_integrity_scope": "Source hashes were collected during preparation; this check does not rescan all 132758 raw XML files."}


if __name__ == "__main__":
    result = verify()
    write_json(ROOT / "outputs/phase02_verification.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
