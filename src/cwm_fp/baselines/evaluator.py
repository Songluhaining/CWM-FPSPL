"""The only component joining held-out truth/origin with frozen predictions."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn import metrics

from cwm_fp.data.common import KEY, indexed, read_csv, sha256, write_csv, write_json
from cwm_fp.data.labels import load_targets
from .pipeline import SCORE_FIELDS, context, split_file

RESULT_FIELDS = (*SCORE_FIELDS, "split_id", "y_true", "system_id", "bug_version_id", "product_id", "origin")


def binary_metrics(y, scores, pred, kind="margin") -> dict:
    y, scores, pred = np.asarray(y, dtype=int), np.asarray(scores, dtype=float), np.asarray(pred, dtype=int)
    if not len(y):
        return {"n": 0, "status": "unavailable", "reason": "no eligible PASS products"}
    if not np.isfinite(scores).all() or not set(y).issubset({0, 1}) or not set(pred).issubset({0, 1}):
        raise ValueError("Nonfinite score or invalid binary target/prediction")
    two = len(np.unique(y)) == 2
    tn, fp, fn, tp = metrics.confusion_matrix(y, pred, labels=[0, 1]).ravel()
    result = {"n": len(y), "FP": int(sum(y)), "TP": int(len(y)-sum(y)), "status": "ok",
        "fp_precision": float(tp / (tp+fp)) if tp+fp else 0.,
        "fp_recall": float(tp / (tp+fn)) if tp+fn else None,
        "fp_f1": float(2*tp / (2*tp+fp+fn)) if 2*tp+fp+fn else 0.,
        "accuracy": float(metrics.accuracy_score(y, pred)),
        "balanced_accuracy": float(metrics.balanced_accuracy_score(y, pred)) if two else None,
        "roc_auc": float(metrics.roc_auc_score(y, scores)) if two else None,
        "average_precision": float(metrics.average_precision_score(y, scores)) if two else None,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]]}
    if two:
        precision, recall, _ = metrics.precision_recall_curve(y, scores)
        result["pr_auc"] = float(metrics.auc(recall, precision))
    else:
        result["pr_auc"] = None
    probability = kind.startswith("probability")
    if probability and ((scores < 0).any() or (scores > 1).any()):
        raise ValueError("Probability outside [0,1]")
    result["brier"] = float(metrics.brier_score_loss(y, scores)) if probability else None
    ece = 0.
    if probability:
        for i in range(10):
            mask = (scores >= i/10) & ((scores <= 1) if i == 9 else (scores < (i+1)/10))
            if mask.any():
                ece += float(mask.mean() * abs(scores[mask].mean() - y[mask].mean()))
    result["ece_10_equal_width"] = ece if probability else None
    result["unavailable_reasons"] = ([] if two else ["AUC and balanced accuracy require both target classes"])
    if not probability:
        result["unavailable_reasons"].append("SVM margin is not a calibrated probability: Brier/ECE not computed")
    return result


def cluster_interval(rows: list[dict], replicates: int, seed: int) -> dict:
    groups = sorted({(r["system_id"], r["bug_version_id"]) for r in rows})
    if len(groups) < 2:
        return {"status": "unavailable", "reason": "fewer than two version clusters"}
    by_group = {g: [r for r in rows if (r["system_id"], r["bug_version_id"]) == g] for g in groups}
    rng = np.random.RandomState(seed)
    estimates = {name: [] for name in ("fp_f1", "average_precision", "roc_auc", "balanced_accuracy")}
    for _ in range(replicates):
        sample = [r for i in rng.randint(0, len(groups), len(groups)) for r in by_group[groups[i]]]
        y = np.array([r["y_true"] for r in sample])
        if len(np.unique(y)) < 2:
            continue
        p = np.array([r["y_pred"] for r in sample])
        score = np.array([r["y_score"] for r in sample])
        estimates["fp_f1"].append(float(metrics.f1_score(y, p, zero_division=0)))
        estimates["average_precision"].append(float(metrics.average_precision_score(y, score)))
        estimates["roc_auc"].append(float(metrics.roc_auc_score(y, score)))
        estimates["balanced_accuracy"].append(float(metrics.balanced_accuracy_score(y, p)))
    return {"method": "percentile bootstrap clustered by (system_id,bug_version_id)", "level": .95,
            "clusters": len(groups), "requested_replicates": replicates,
            "valid_two_class_replicates": len(estimates["fp_f1"]),
            "intervals": {n: np.percentile(v, [2.5, 97.5]).tolist() if v else None for n, v in estimates.items()}}


def select(c):
    for fold in c["folds"]:
        products, splits, ordered = context(c, fold)
        targets = load_targets(c["labels"] / fold / "val.csv", c["products"], split_file(c, fold), partition="val", purpose="evaluate")
        truth = {products[k]["sample_id"]: int(v == "FP") for k, v in targets.items() if products[k]["observed_result"] == "PASS"}
        for seed in c["seeds"]:
            dest = c["work_dir"] / fold / str(seed)
            trials = json.loads((dest / "trials.json").read_text(encoding="utf-8"))["trials"]
            chosen, values = {}, []
            for trial in trials:
                file = dest / (trial["trial_id"] + "_val.csv")
                if sha256(file) != trial["validation_prediction_sha256"]:
                    raise ValueError("Validation predictions changed before selection")
                rows = read_csv(file, SCORE_FIELDS)
                if len(rows) != len(truth) or {r["sample_id"] for r in rows} != set(truth) or any(r["partition"] != "val" for r in rows):
                    raise ValueError("Selection accepts exactly the validation PASS predictions")
                y = [truth[r["sample_id"]] for r in rows]
                score = float(metrics.f1_score(y, [int(r["y_pred"]) for r in rows], zero_division=0))
                values.append({"trial_id": trial["trial_id"], "C": trial["C"], "validation_F1": score})
                name = trial["model"]
                if name not in chosen or score > chosen[name]["validation_F1"]:
                    chosen[name] = trial | {"validation_F1": score}
            write_json(dest / "selection.json", {"models": chosen, "selection_rule": "maximum validation FP F1; ties retain first ascending C",
                "validation_label_sha256": sha256(c["labels"] / fold / "val.csv"), "test_labels_read": False})
            write_json(c["metrics_dir"] / fold / str(seed) / "validation_selection.json", values)
            print(f"selected on VAL only: {fold}, seed={seed}", flush=True)


def metric_rows(rows):
    if not rows:
        return binary_metrics([], [], [])
    return binary_metrics([r["y_true"] for r in rows], [r["y_score"] for r in rows],
                          [r["y_pred"] for r in rows], rows[0]["score_kind"])


def evaluate(c):
    if (c["metrics_dir"] / "summary.json").exists():
        raise ValueError("Test evaluation already recorded for this run ID")
    summaries = []
    for fold in c["folds"]:
        products, splits, ordered = context(c, fold)
        by_sample = {p["sample_id"]: k for k, p in products.items()}
        labels = {}
        for partition in ("train", "val", "test"):
            path = c["labels"] / fold / (partition + ".csv")
            allowed_targets = load_targets(path, c["products"], split_file(c, fold), partition=partition, purpose="evaluate")
            metadata = indexed(read_csv(path, (*KEY, "origin")))
            labels.update({k: {"y_true": int(value == "FP"), "origin": metadata[k]["origin"]}
                           for k, value in allowed_targets.items() if products[k]["observed_result"] == "PASS"})
        expected = {p["sample_id"] for p in products.values() if p["observed_result"] == "PASS"}
        split_id = fold + ":" + sha256(split_file(c, fold))
        for seed in c["seeds"]:
            source = c["work_dir"] / fold / str(seed)
            for name in c["models"]:
                prediction_file = source / (name + "_predictions.csv")
                predictions = read_csv(prediction_file, SCORE_FIELDS)
                if {r["sample_id"] for r in predictions} != expected or len(predictions) != len(expected):
                    raise ValueError("Evaluation needs each original PASS product exactly once; F is not a target class")
                rows = []
                for p in predictions:
                    k = by_sample[p["sample_id"]]
                    if p["partition"] != splits[k]["split"] or p["model"] != name or int(p["seed"]) != seed:
                        raise ValueError("Prediction partition/model identity mismatch")
                    rows.append(p | {"seed": seed, "y_score": float(p["y_score"]), "y_pred": int(p["y_pred"]),
                        "split_id": split_id, "system_id": k[0], "bug_version_id": k[1], "product_id": k[2], **labels[k]})
                dest = c["metrics_dir"] / fold / str(seed)
                write_csv(dest / (name + ".csv"), rows, RESULT_FIELDS)
                partitions = {}
                for partition in ("train", "val", "test"):
                    selected = [r for r in rows if r["partition"] == partition]
                    record = metric_rows(selected)
                    record["interpretation"] = "resubstitution, not generalization" if partition == "train" else "held-out prediction"
                    record["by_system"] = {s: metric_rows([r for r in selected if r["system_id"] == s]) for s in sorted({r["system_id"] for r in selected})}
                    record["macro_F1_available_systems"] = float(np.mean([v["fp_f1"] for v in record["by_system"].values()]))
                    record["origin_counts"] = dict(Counter(r["origin"] for r in selected))
                    record["exact_origin_strata"] = {o: metric_rows([r for r in selected if r["origin"] == o]) for o in ("natural", "simulated", "unknown")}
                    record["fp_origin_cohorts_with_shared_TP_controls"] = {}
                    for origin in ("natural", "simulated"):
                        positives = [r for r in selected if r["y_true"] == 1 and r["origin"] == origin]
                        cohort = positives + [r for r in selected if r["y_true"] == 0] if positives else []
                        record["fp_origin_cohorts_with_shared_TP_controls"][origin] = metric_rows(cohort)
                    if partition == "test":
                        record["cluster_bootstrap"] = cluster_interval(selected, c["bootstrap_replicates"], seed)
                    partitions[partition] = record
                summary = {"fold": fold, "seed": seed, "model": name, "partitions": partitions,
                           "predictions_sha256": sha256(prediction_file), "proxy_labels": True}
                write_json(dest / (name + "_metrics.json"), summary)
                summaries.append(summary)
            print(f"evaluated original PASS only: {fold}, seed={seed}", flush=True)
    aggregated = []
    for fold in c["folds"]:
        for name in c["models"]:
            values = [s["partitions"]["test"] for s in summaries if s["fold"] == fold and s["model"] == name]
            aggregated.append({"fold": fold, "model": name, "n_test": values[0]["n"],
                "seed_mean_std": {metric: seed_summary([v[metric] for v in values])
                                  for metric in ("fp_precision", "fp_recall", "fp_f1", "pr_auc", "average_precision", "roc_auc", "balanced_accuracy", "accuracy")}})
    write_json(c["metrics_dir"] / "summary.json", {"protocol": c["protocol"], "suite_view": "original", "seeds": c["seeds"],
        "test_evaluations_per_seed_fold_model": 1, "runs": len(summaries), "aggregate": aggregated,
        "paper_original": c["paper_original_protocol"],
        "missing_loso_systems": [s for s in ("BankAccountTP", "Email", "Elevator", "ExamDB", "GPL", "ZipMe") if "loso_"+s not in c["folds"]],
        "limitations": ["benchmark source membership oracle, not independent failure discovery", "not the paper 823-version subset", "original suites only; simulated suite disabled", "source-rule feature adapter, not exact paper evaluation"] +
            (["LOSO semantic priors cannot match unseen systems"] if c['protocol']=='strict_train_reference_v1' else
             ["target-version observable failing context allowed; not strict no-target-context induction"]),
        "base_feature_exclusion": False if c['protocol']=='version_context_v1' else None})


def seed_summary(values):
    available = [v for v in values if v is not None]
    return {"mean": float(np.mean(available)) if available else None,
            "std": float(np.std(available, ddof=1)) if len(available)>1 else None,
            "available_seeds": len(available), "requested_seeds": len(values)}
