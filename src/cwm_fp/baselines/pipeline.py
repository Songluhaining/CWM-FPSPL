"""Bounded, role-separated phase-02 command line."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from cwm_fp.data.common import KEY, ROOT, indexed, key, read_csv, sha256, write_csv, write_json
from cwm_fp.data.labels import load_targets
from cwm_fp.data.splits import require_training_keys, validate_split
from .boundaries import install_io_guard
from .features import CLAP6, COVERAGE2, PULP50, TrainReferenceFeatures
from .observations import collect
from .vendor import FrozenSourcePU, clap_fit, load_frozen_artifact

FAMILY = {"clap6_svm": "clap6", "pulp50_lr": "pulp50", "pulp50_svm": "pulp50",
          "pulp50_pu_source": "pulp50", "coverage2_lr": "coverage2"}
SCORE_FIELDS = ("sample_id", "partition", "model", "seed", "y_score", "y_pred", "score_kind")


def configuration(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config["suite_view"] != "original" or config["protocol"] != "strict_train_reference_v1":
        raise ValueError("Only explicitly named strict original-suite protocol is enabled")
    for name in ("products", "tests", "sources", "labels", "split_dir", "work_dir", "metrics_dir"):
        config[name] = (ROOT / config[name]).resolve()
    if not config["work_dir"].is_relative_to(ROOT / "data/processed/baselines") or not config["metrics_dir"].is_relative_to(ROOT / "outputs/metrics"):
        raise ValueError("Generated artifacts outside phase-02 output roots")
    if set(config["models"]) - FAMILY.keys():
        raise ValueError("Unknown baseline model")
    return config


def split_file(c, fold):
    return c["split_dir"] / ("splits.csv" if fold == "grouped" else fold + ".csv")


def context(c, fold):
    products = indexed(read_csv(c["products"], KEY))
    split = indexed(read_csv(split_file(c, fold), (*KEY, "split")))
    validate_split(products, split, holdout=None if fold == "grouped" else fold.removeprefix("loso_"))
    ordered = sorted(products)
    return products, split, ordered


def prepare(c):
    install_io_guard("features")
    work = c["work_dir"]
    work.mkdir(parents=True, exist_ok=True)
    snapshot = work / "observations.joblib"
    if c.get("reuse_observations", False):
        evidence = json.loads((work / "observations_audit.json").read_text(encoding="utf-8"))
        if (evidence["products_sha256"] != sha256(c["products"]) or evidence["tests_sha256"] != sha256(c["tests"])
                or evidence.get("snapshot_sha256") != sha256(snapshot) or evidence.get("sources_sha256") != sha256(c["sources"])):
            raise ValueError("Frozen observation snapshot identity mismatch")
        observations = joblib.load(snapshot)
        print("reusing explicitly requested frozen observation snapshot", flush=True)
    else:
        observations, evidence = collect(c["products"], c["tests"], c["sources"], c["bounds"])
        joblib.dump(observations, snapshot, compress=3)
        evidence["snapshot_sha256"] = sha256(snapshot)
        evidence["sources_sha256"] = sha256(c["sources"])
        write_json(work / "observations_audit.json", evidence)
    for fold in c["folds"]:
        _, split, ordered = context(c, fold)
        if [o.identity for o in observations] != ordered:
            raise ValueError("Observation row order differs from canonical product order")
        train = [o for o in observations if split[o.identity]["split"] == "train"]
        builder = TrainReferenceFeatures().fit(train, split_file(c, fold))
        arrays = builder.transform(observations)
        dest = work / fold
        dest.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dest / "features.npz", **arrays)
        write_json(dest / "features_audit.json", builder.audit() | {
            "products_sha256": sha256(c["products"]), "split_sha256": sha256(split_file(c, fold)),
            "features_sha256": sha256(dest / "features.npz"), "sample_ids": [o.sample_id for o in observations]})
        print(f"prepared {fold}: {len(observations)} rows; {len(train)} training references", flush=True)


def load_features(c, fold, ordered):
    dest = c["work_dir"] / fold
    audit = json.loads((dest / "features_audit.json").read_text(encoding="utf-8"))
    products = indexed(read_csv(c["products"], KEY))
    if (audit["products_sha256"] != sha256(c["products"]) or audit["split_sha256"] != sha256(split_file(c, fold))
            or audit["features_sha256"] != sha256(dest / "features.npz")
            or audit["sample_ids"] != [products[k]["sample_id"] for k in ordered]):
        raise ValueError("Feature input fingerprint or sample order mismatch")
    with np.load(dest / "features.npz", allow_pickle=False) as data:
        if set(data.files) != {"clap6", "pulp50", "coverage2"}:
            raise ValueError("Unexpected/forbidden feature family")
        arrays = {name: data[name] for name in data.files}
    for name, columns in (("clap6", CLAP6), ("pulp50", PULP50), ("coverage2", COVERAGE2)):
        if arrays[name].shape != (len(ordered), len(columns)) or not np.isfinite(arrays[name]).all():
            raise ValueError("Invalid feature matrix")
    return arrays


def score_model(model, x, name):
    if name == "pulp50_pu_source":
        scores = model.score(x)
        return scores, (scores > .5).astype(int), "probability_uncalibrated"
    if name.endswith("_lr"):
        scores = model.predict_proba(x)[:, list(model.classes_).index(1)]
        return scores, model.predict(x), "probability_uncalibrated"
    return model.decision_function(x), model.predict(x), "margin"


def score_rows(products, ordered, indexes, scores, predictions, kind, partition, name, seed):
    return [{"sample_id": products[ordered[i]]["sample_id"], "partition": partition, "model": name,
             "seed": seed, "y_score": float(score), "y_pred": int(pred), "score_kind": kind}
            for i, score, pred in zip(indexes, scores, predictions)]


def fit(c):
    allowed = install_io_guard("train")
    for fold in c["folds"]:
        allowed.clear()
        allowed.add((c["labels"] / fold / "train.csv").resolve())
        products, splits, ordered = context(c, fold)
        raw = load_features(c, fold, ordered)
        train = [i for i, k in enumerate(ordered) if splits[k]["split"] == "train"]
        passing = [i for i in train if products[ordered[i]]["observed_result"] == "PASS"]
        failing = [i for i in train if products[ordered[i]]["observed_result"] == "FAIL"]
        val = [i for i, k in enumerate(ordered) if splits[k]["split"] == "val" and products[k]["observed_result"] == "PASS"]
        require_training_keys([ordered[i] for i in train], split_file(c, fold), "normalization")
        scalers = {name: MinMaxScaler().fit(value[train]) for name, value in raw.items()}
        scaled = {name: scalers[name].transform(value) for name, value in raw.items()}
        targets = load_targets(c["labels"] / fold / "train.csv", c["products"], split_file(c, fold), partition="train", purpose="train")
        if any(targets.get(ordered[i]) not in {"FP", "TP"} for i in passing):
            raise ValueError("Unknown or inconsistent training PASS target")
        y = np.array([int(targets[ordered[i]] == "FP") for i in passing])
        if len(np.unique(y)) != 2:
            raise ValueError("Supervised training requires both FP and TP in TRAIN")
        fold_dir = c["work_dir"] / fold
        write_json(fold_dir / "scaler_audit.json", {"training_keys": [ordered[i] for i in train],
            "scalers": {n: {"minimum": s.data_min_.tolist(), "maximum": s.data_max_.tolist()} for n, s in scalers.items()},
            "split_sha256": sha256(split_file(c, fold))})
        for seed in c["seeds"]:
            dest = fold_dir / str(seed)
            dest.mkdir(exist_ok=True)
            trials = []
            for name in c["models"]:
                family = FAMILY[name]
                x = scaled[family]
                parameters = [1.0] if name in {"clap6_svm", "pulp50_pu_source"} else c["c_grid"]
                for number, regularization in enumerate(parameters):
                    if name == "clap6_svm":
                        model = clap_fit(x[passing], y)
                    elif name == "pulp50_pu_source":
                        model = FrozenSourcePU(seed, c["pu_estimators"], c["pu_sampling_ratio"]).fit(x[failing], x[passing])
                    elif name.endswith("_lr"):
                        model = LogisticRegression(C=regularization, solver="lbfgs", max_iter=3000, random_state=seed).fit(x[passing], y)
                    else:
                        model = SVC(C=regularization, kernel="linear", random_state=seed).fit(x[passing], y)
                    trial_id = f"{name}__c{number}"
                    joblib.dump({"model": model, "scaler": scalers[family], "family": family,
                        "split_sha256": sha256(split_file(c, fold)), "feature_sha256": sha256(fold_dir / "features.npz")}, dest / (trial_id + ".joblib"))
                    score, pred, kind = score_model(model, x[val], name)
                    write_csv(dest / (trial_id + "_val.csv"), score_rows(products, ordered, val, score, pred, kind, "val", name, seed), SCORE_FIELDS)
                    trials.append({"trial_id": trial_id, "model": name, "C": regularization,
                        "validation_prediction_sha256": sha256(dest / (trial_id + "_val.csv")),
                        "model_sha256": sha256(dest / (trial_id + ".joblib"))})
            write_json(dest / "trials.json", {"trials": trials, "training_PASS": len(passing), "training_F": len(failing), "seed": seed})
            print(f"fitted {fold}, seed={seed}: {len(trials)} candidates; no evaluation labels", flush=True)


def predict(c):
    install_io_guard("inference")
    for fold in c["folds"]:
        products, split, ordered = context(c, fold)
        raw = load_features(c, fold, ordered)
        for seed in c["seeds"]:
            dest = c["work_dir"] / fold / str(seed)
            selection = json.loads((dest / "selection.json").read_text(encoding="utf-8"))
            for name, trial in selection["models"].items():
                model_path = dest / (trial["trial_id"] + ".joblib")
                if sha256(model_path) != trial["model_sha256"]:
                    raise ValueError("Selected model changed after validation")
                artifact = load_frozen_artifact(model_path)
                if artifact["split_sha256"] != sha256(split_file(c, fold)) or artifact["feature_sha256"] != sha256(dest.parent / "features.npz"):
                    raise ValueError("Model/split/feature fingerprint mismatch")
                x = artifact["scaler"].transform(raw[artifact["family"]])
                rows = []
                for partition in ("train", "val", "test"):
                    selected = [i for i, k in enumerate(ordered) if split[k]["split"] == partition and products[k]["observed_result"] == "PASS"]
                    score, pred, kind = score_model(artifact["model"], x[selected], name)
                    rows.extend(score_rows(products, ordered, selected, score, pred, kind, partition, name, seed))
                write_csv(dest / (name + "_predictions.csv"), rows, SCORE_FIELDS)
            print(f"predicted {fold}, seed={seed}: frozen models, unlabeled outputs", flush=True)


def run_all(config_path: Path, c):
    import datetime
    if (c["metrics_dir"] / "summary.json").exists():
        raise ValueError("Final evaluation already exists; use a new declared run ID instead of silently retesting")
    record = {"started_at": datetime.datetime.now().astimezone().isoformat(), "config_sha256": sha256(config_path),
              "commands": [], "python": sys.version, "scope": "phase02 only", "success": False}
    write_json(ROOT / "outputs/phase02_run.json", record)
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    for stage in ("prepare", "fit", "select", "predict", "evaluate"):
        command = [sys.executable, "-X", "utf8", "-B", "-m", "cwm_fp.baselines.pipeline", stage, "--config", str(config_path)]
        log = ROOT / "outputs" / f"phase02_{stage}.log"
        start = time.perf_counter()
        stage_code = {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted((ROOT / "src/cwm_fp/baselines").glob("*.py"))}
        with log.open("w", encoding="utf-8") as output:
            try:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT,
                                        timeout=c["bounds"]["max_seconds_per_command"])
                exit_code, timeout_error = result.returncode, None
            except subprocess.TimeoutExpired as exc:
                exit_code, timeout_error = 124, str(exc)
        record["commands"].append({"argv": command, "exit_code": exit_code, "seconds": round(time.perf_counter()-start, 3),
                                   "log": log.as_posix(), "sha256": sha256(log), "timeout_error": timeout_error,
                                   "code_sha256": stage_code})
        write_json(ROOT / "outputs/phase02_run.json", record)
        print(f"{stage}: exit={exit_code}, seconds={record['commands'][-1]['seconds']}", flush=True)
        if exit_code:
            raise RuntimeError(f"Phase02 {stage} failed: {log}")
    record["completed_at"] = datetime.datetime.now().astimezone().isoformat()
    record["success"] = True
    write_json(ROOT / "outputs/phase02_run.json", record)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["run", "prepare", "fit", "select", "predict", "evaluate"])
    parser.add_argument("--config", type=Path, default=ROOT / "configs/baselines.yaml")
    a = parser.parse_args(argv)
    try:
        c = configuration(a.config)
        if a.action == "run":
            run_all(a.config.resolve(), c)
        elif a.action in {"select", "evaluate"}:
            from .evaluator import evaluate, select
            install_io_guard("evaluator")
            (select if a.action == "select" else evaluate)(c)
        else:
            {"prepare": prepare, "fit": fit, "predict": predict}[a.action](c)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
