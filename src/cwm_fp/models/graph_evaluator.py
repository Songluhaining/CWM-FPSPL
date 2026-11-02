"""Separate truth-reading evaluator; no torch or classifier imports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cwm_fp.data.common import KEY, ROOT, indexed, read_csv, sha256, write_json
from cwm_fp.data.labels import load_targets
from .graph_metrics import binary_metrics, clustered_interval


def evaluate_once(destination: Path, predictions: Path, expected_hash: str, evaluate):
    """Exclusive claim precedes truth access. Failed evaluations remain sealed."""
    if sha256(predictions) != expected_hash: raise ValueError("Frozen predictions changed")
    destination.parent.mkdir(parents=True, exist_ok=True)
    claim = destination.with_suffix(".claim")
    with claim.open("x", encoding="utf-8") as f:
        f.write(expected_hash + "\n")
    result = evaluate(json.loads(predictions.read_text(encoding="utf-8")))
    write_json(destination, result)
    return result


def score_rows(rows, truth, *, partition, expected_ids):
    if {r["sample_id"] for r in rows} != set(expected_ids) or len(rows) != len(expected_ids):
        raise ValueError("Prediction cohort differs from frozen eligible cohort")
    required = {"sample_id", "identity", "partition", "y_score", "y_pred", "score_kind"}
    for r in rows:
        if set(r) != required or r["partition"] != partition or r["sample_id"] not in truth:
            raise ValueError("Unknown prediction fields/partition/identity")
        entry = truth[r["sample_id"]]
        if r["identity"] != entry["identity"] or r["y_pred"] != int(r["y_score"] >= .5) or r["score_kind"] != "probability_uncalibrated":
            raise ValueError("Invalid frozen prediction")
    metrics = binary_metrics([truth[r["sample_id"]]["y_true"] for r in rows], [r["y_score"] for r in rows])
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('legacy','package_validation'),default='legacy')
    parser.add_argument('--experiment-protocol',type=Path)
    parser.add_argument('--experiment-sha256')
    parser.add_argument('--prediction-index',type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--fold")
    parser.add_argument("--partition", choices=("val", "test"))
    parser.add_argument("--predictions", type=Path)
    args = parser.parse_args(argv)
    if args.mode=='package_validation':
        from .package_comparison import evaluate_candidates
        return evaluate_candidates(args.experiment_protocol,args.experiment_sha256,args.prediction_index)
    if any(v is None for v in (args.config,args.fold,args.partition,args.predictions)):parser.error('Legacy evaluation requires config,fold,partition,predictions')
    from cwm_fp.baselines.boundaries import install_io_guard
    install_io_guard("evaluator")
    c = json.loads(args.config.read_text(encoding="utf-8"))
    if args.fold not in c["folds"] or c["suite_view"] != "original" or c["protocol"] != "strict_graph_v1":
        raise ValueError("Wrong graph protocol/fold")
    work = (ROOT / c["work_dir"]).resolve(); pred = args.predictions.resolve()
    if not work.is_relative_to((ROOT / "data/processed/graph_classifier").resolve()) or not pred.is_relative_to(work):
        raise ValueError("Unknown graph-run output")
    split = ROOT / c["split_dir"] / ("splits.csv" if args.fold == "grouped" else args.fold + ".csv")
    label_file = ROOT / c["labels"] / args.fold / (args.partition + ".csv")
    cohort = json.loads((pred.parent / "cohort.json").read_text(encoding="utf-8"))
    for name in ("products", "tests", "graphs"):
        if cohort["input_sha256"][name] != sha256(ROOT / c[name]): raise ValueError("Input changed after cohort freeze")
    if cohort["input_sha256"]["split"] != sha256(split): raise ValueError("Split changed")
    expected = cohort["sample_ids"][args.partition]
    def evaluate(rows):
        targets = load_targets(label_file, ROOT / c["products"], split, partition=args.partition, purpose="evaluate")
        products = indexed(read_csv(ROOT / c["products"], KEY))
        metadata = indexed(read_csv(label_file, (*KEY, "origin")))
        truth = {products[k]["sample_id"]: {"identity": list(k), "y_true": int(t == "FP"), "origin": metadata[k]["origin"]}
                 for k, t in targets.items() if products[k]["observed_result"] == "PASS" and t in {"FP", "TP"}}
        metrics = score_rows(rows, truth, partition=args.partition, expected_ids=expected)
        if args.partition == "val": return metrics
        joined = [r | truth[r["sample_id"]] for r in rows]
        metrics["cluster_bootstrap"] = clustered_interval(joined, frozen["seed"])
        metrics["by_system"] = {s: binary_metrics([r["y_true"] for r in joined if r["identity"][0] == s],
                                                     [r["y_score"] for r in joined if r["identity"][0] == s]) for s in sorted({r["identity"][0] for r in joined})}
        metrics["by_origin"] = {o: binary_metrics([r["y_true"] for r in joined if r["origin"] == o],
                                                     [r["y_score"] for r in joined if r["origin"] == o]) for o in ("natural", "simulated", "unknown")}
        return {"metrics": metrics, "rows": joined, "proxy_labels": True, "split_sha256": sha256(split),
                "predictions_sha256": sha256(pred), "test_evaluations": 1}
    if args.partition == "val":
        result = evaluate(json.loads(pred.read_text(encoding="utf-8")))
    else:
        frozen = json.loads((pred.parent / "frozen.json").read_text(encoding="utf-8"))
        if frozen["checkpoint_sha256"] != sha256(pred.parent / "model.pt") or frozen["split_sha256"] != sha256(split):
            raise ValueError("Frozen model/split changed")
        if (frozen["cohort_sha256"] != sha256(pred.parent / "cohort.json") or frozen["training_sha256"] != sha256(pred.parent / "training.json")
                or frozen["config_sha256"] != sha256(args.config) or frozen["input_sha256"] != cohort["input_sha256"]):
            raise ValueError("Frozen cohort/config/training evidence changed")
        destination = ROOT / "outputs/metrics" / work.name / pred.parent.relative_to(work) / "test_metrics.json"
        result = evaluate_once(destination, pred, frozen["predictions_sha256"], evaluate)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
