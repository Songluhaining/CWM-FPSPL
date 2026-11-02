"""Phase-04 bounded training. Real runs audit eligibility before opening labels."""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

from cwm_fp.data.common import ROOT, sha256, write_json
from .graph_classifier import GraphClassifier
from .graph_data import load_corpus, tensor_products
from .graph_encoder import Limits, collate, encode_test, evidence_errors, feature_names
from .graph_metrics import binary_metrics


@dataclass(frozen=True)
class TrainOptions:
    seed: int = 20261030
    max_epochs: int = 100
    max_seconds: float = 60.
    patience: int = 15
    batch_products: int = 8
    learning_rate: float = .01
    hidden: int = 16
    coverage_hidden: int = 32
    layers: int = 2
    max_parameters: int = 10000
    device: str = "cpu"

    def __post_init__(self):
        if not (1 <= self.max_epochs <= 500 and 0 < self.max_seconds <= 3600 and 1 <= self.patience <= 500
                and 1 <= self.batch_products <= 32 and 0 < self.learning_rate <= .1 and self.max_parameters <= 100000):
            raise ValueError("Training bounds exceeded")
        if self.device not in {"cpu", "cuda"}: raise ValueError("Explicit CPU/CUDA device required")


def check_partitions(products, assignments):
    groups = {}
    for k, part in assignments.items():
        if part not in {"train", "val", "test"}: raise ValueError("Unknown partition")
        previous = groups.setdefault(k[:2], part)
        if previous != part: raise ValueError("Version overlap across partitions")
    if set(assignments.values()) != {"train", "val", "test"}: raise ValueError("Three independent partitions required")
    seen = set()
    for p in products:
        if p.identity in seen or assignments.get(p.identity) != p.partition: raise ValueError("Duplicate/incorrect product partition")
        seen.add(p.identity)
        if p.observed_result != "PASS": raise ValueError("F products cannot enter FP/TP supervision or denominator")
        if p.audit.get("preview_only") or any(t.metadata.get("preview_only") for t in p.tests):
            raise ValueError("Exploratory tensors cannot enter training/evaluation")
        if any(p.audit.get(k,0)>0 for k in ('lost_test_fraction','lost_event_fraction','lost_edge_fraction','lost_node_fraction')):
            raise ValueError('Lossy product tensors cannot enter formal training/evaluation')
        if any(tuple(t.metadata["identity"][:3]) != p.identity for t in p.tests):
            raise ValueError("Test graph belongs to a different product/version")


@torch.no_grad()
def predict(model, products, batch_products=8):
    model.eval(); rows = []
    for start in range(0, len(products), batch_products):
        batch = products[start:start+batch_products]
        scores = torch.sigmoid(model(collate(batch).to(next(model.parameters()).device))).cpu().tolist()
        for p, s in zip(batch, scores):
            rows.append({"sample_id": p.sample_id, "identity": list(p.identity), "partition": p.partition,
                         "y_score": s, "y_pred": int(s >= .5), "score_kind": "probability_uncalibrated"})
    return rows


def fit(kind, train, targets, validation, assignments, evaluate_validation, options=TrainOptions()):
    """The callback returns validation metrics, never validation target records.

    The real CLI calls a separate evaluator process. Synthetic tests use an
    in-memory evaluator and label their output explicitly as synthetic.
    """
    check_partitions(train + validation, assignments)
    if not train or not validation or any(p.partition != "train" for p in train) or any(p.partition != "val" for p in validation):
        raise ValueError("Nonempty TRAIN and VAL inputs required")
    if set(targets) != {p.identity for p in train} or set(targets.values()) != {"FP", "TP"}:
        raise ValueError("Exactly both FP/TP classes for TRAIN PASS products required")
    torch.manual_seed(options.seed); random.seed(options.seed); torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if options.device == "cuda":
        if not torch.cuda.is_available(): raise ValueError("CUDA unavailable")
        torch.cuda.reset_peak_memory_stats()
    hidden = options.coverage_hidden if kind == "coverage" else options.hidden
    schemas={g.metadata.get('feature_schema','semantic_v1') for p in train+validation for g in p.tests}
    if len(schemas)!=1:raise ValueError('Cannot train with mixed feature schema versions')
    feature_schema=schemas.pop()
    model = GraphClassifier(kind, hidden, options.layers,feature_schema=feature_schema)
    n_parameters = sum(p.numel() for p in model.parameters())
    if n_parameters > options.max_parameters: raise ValueError("Parameter budget exceeded")
    model.fit_normalization(train, assignments); model.to(options.device)
    counts = {label: sum(t == label for t in targets.values()) for label in ("TP", "FP")}
    weights = torch.tensor([len(train) / (2 * counts[t]) for t in ("TP", "FP")], device=options.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=options.learning_rate)
    generator = random.Random(options.seed)
    start_time = time.monotonic(); deadline = start_time + options.max_seconds
    best, best_loss, best_epoch, stale = None, math.inf, 0, 0
    history, stop = [], "max_epochs"
    initial = predict(model, train, options.batch_products)
    initial_loss = binary_metrics([int(targets[tuple(r["identity"])] == "FP") for r in initial], [r["y_score"] for r in initial])["log_loss"]
    for epoch in range(1, options.max_epochs + 1):
        order = list(train); generator.shuffle(order); model.train(); losses = []
        for offset in range(0, len(order), options.batch_products):
            if time.monotonic() >= deadline:
                stop = "time_budget"; break
            batch = order[offset:offset+options.batch_products]
            y = torch.tensor([int(targets[p.identity] == "FP") for p in batch], dtype=torch.float32, device=options.device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(collate(batch).to(options.device)), y, weight=weights[y.long()])
            if not torch.isfinite(loss): raise ValueError("Nonfinite training loss")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); optimizer.step()
            losses.append((len(batch), float(loss.detach())))
        if stop == "time_budget": break
        remaining = deadline - time.monotonic()
        if remaining <= 0: stop = "time_budget"; break
        validation_metrics = evaluate_validation(predict(model, validation, options.batch_products), remaining)
        val_loss = validation_metrics["log_loss"]
        if not math.isfinite(val_loss): raise ValueError("Invalid validation selection score")
        history.append({"epoch": epoch, "train_weighted_loss": sum(n*l for n,l in losses)/len(train),
                        "validation_log_loss": val_loss, "validation_pr_auc": validation_metrics.get("pr_auc")})
        if val_loss < best_loss - 1e-7:
            best, best_loss, best_epoch, stale = copy.deepcopy(model.state_dict()), val_loss, epoch, 0
        else:
            stale += 1
        if time.monotonic() >= deadline: stop = "time_budget"; break
        if stale >= options.patience: stop = "validation_patience"; break
    if best is None: raise TimeoutError("No validated checkpoint within training budget")
    model.load_state_dict(best); final = predict(model, train, options.batch_products)
    train_metrics = binary_metrics([int(targets[tuple(r["identity"])] == "FP") for r in final], [r["y_score"] for r in final])
    report = {"kind": kind, "parameters": n_parameters, "hidden": hidden, 'feature_schema':feature_schema,
              'feature_names':list(feature_names(feature_schema)),"options": asdict(options), "epochs": len(history),
              "selected_epoch": best_epoch, "stop_reason": stop, "training_seconds": time.monotonic()-start_time,
              "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated() if options.device == "cuda" else 0,
              "gpu_memory_scope": "torch allocator" if options.device == "cuda" else "CPU run; no CUDA tensors allocated",
              "class_counts": counts, "class_weights": dict(zip(("TP", "FP"), weights.cpu().tolist())),
              "normalization": model.normalization_audit, "initial_train_log_loss": initial_loss,
              "train_resubstitution": train_metrics, "history": history,
              "selection": "minimum validation unweighted log loss; first epoch wins ties; fixed threshold 0.5",
              "test_labels_read_by_trainer": False}
    return model, report


def configuration(path):
    c = json.loads(Path(path).read_text(encoding="utf-8"))
    if c["protocol"] != "strict_graph_v1" or c["suite_view"] != "original": raise ValueError("Unsupported scientific protocol")
    for k in ("products", "tests", "graphs", "split_dir", "labels", "work_dir"):
        c[k] = (ROOT / c[k]).resolve()
    if not c["work_dir"].is_relative_to((ROOT / "data/processed/graph_classifier").resolve()):
        raise ValueError("Output must be inside phase-04 generated data")
    return c


def split_path(c, fold):
    if fold not in c["folds"]: raise ValueError("Unconfigured split")
    return c["split_dir"] / ("splits.csv" if fold == "grouped" else fold + ".csv")


def corpus(c, fold):
    return load_corpus(c["products"], c["tests"], c["graphs"], split_path(c, fold),
                       holdout=fold.removeprefix("loso_") if fold.startswith("loso_") else None)


def readiness(c):
    audits = {}
    for fold in c["folds"]:
        _, _, graphs, _, audit = corpus(c, fold)
        if fold == c["folds"][0]:
            previews = []
            for gs in graphs.values():
                for g in gs:
                    if evidence_errors(g, require_mapping=False): continue
                    t = encode_test(g, "both", Limits(**c["limits"]), preview=True)
                    previews.append({k: v for k, v in t.metadata.items() if k not in {"node_records", "edge_records"}})
            audit["structurally_supported_preview_tensors"] = previews
        audits[fold] = audit
    return {"protocol": c["protocol"], "status": "ready" if all(a["ready"] for a in audits.values()) else "blocked",
            "folds": audits, "real_training_runs": 0, "real_test_evaluations": 0,
            "same_cohort_tabular_comparison": {"status": "unavailable", "reason": "No eligible independent graph cohort; phase02 aggregate is not a paired graph comparison"},
            "state_transition_training": False, "intervention_training": False}


def real_train(c, config_path, fold, kind, ablation, seed):
    from cwm_fp.baselines.boundaries import install_io_guard
    from cwm_fp.data.labels import load_targets
    train_label = c["labels"] / fold / "train.csv"
    install_io_guard("graph_trainer", [train_label])
    products, splits, _, eligible, audit = corpus(c, fold)
    if not audit["ready"]:
        write_json(c["work_dir"] / (fold + "_blocked.json"), audit | {"real_training_runs": 0, "real_test_evaluations": 0})
        print(json.dumps({"status": "blocked_before_labels_or_fitting", "blockers": audit["blockers"]})); return 3
    if (kind == "coverage") != (ablation == "coverage"): raise ValueError("Coverage model and ablation must agree")
    dest = c["work_dir"] / fold / str(seed) / (kind + "_" + ablation)
    dest.mkdir(parents=True, exist_ok=False)  # no silent retraining on a tested run
    items = (tensor_products(products, splits, eligible) if ablation == 'full'
             else tensor_products(products, splits, eligible, ablation, Limits(**c["limits"])))
    assignment = {k: r["split"] for k, r in splits.items()}
    # Check every partition before any target is opened, including held-out data.
    check_partitions(items, assignment)
    train, val, test = ([p for p in items if p.partition == part] for part in ("train", "val", "test"))
    write_json(dest / "cohort.json", {"input_sha256": audit["input_sha256"],
               "sample_ids": {part: [p.sample_id for p in items if p.partition == part] for part in ("train", "val", "test")}})
    all_targets = load_targets(train_label, c["products"], split_path(c, fold), partition="train", purpose="train")
    targets = {p.identity: all_targets.get(p.identity) for p in train}
    def evaluate_val(rows, remaining):
        write_json(dest / "validation_predictions.json", rows)
        command = [sys.executable, "-m", "cwm_fp.models.graph_evaluator", "--config", str(config_path), "--fold", fold,
                   "--partition", "val", "--predictions", str(dest / "validation_predictions.json")]
        result = subprocess.run(command, capture_output=True, text=True, timeout=max(.01, remaining), check=True)
        return json.loads(result.stdout)
    options = TrainOptions(**(c["training"] | {"seed": seed}))
    model, report = fit(kind, train, targets, val, assignment, evaluate_val, options)
    checkpoint = dest / "model.pt"
    torch.save(model.state_dict(), checkpoint)
    write_json(dest / "training.json", report | {"input_sha256": audit["input_sha256"], "checkpoint_sha256": sha256(checkpoint),
               "ablation": ablation, "sampling": [{"identity": p.identity, **p.audit} for p in items]})
    predictions = predict(model, test, options.batch_products)
    write_json(dest / "test_predictions.json", predictions)
    write_json(dest / "frozen.json", {"predictions_sha256": sha256(dest / "test_predictions.json"),
                "checkpoint_sha256": sha256(checkpoint), "split_sha256": sha256(split_path(c, fold)),
                "cohort_sha256": sha256(dest / "cohort.json"), "training_sha256": sha256(dest / "training.json"),
                "config_sha256": sha256(config_path),
                "input_sha256": audit["input_sha256"], "kind": kind, "ablation": ablation, "seed": seed})
    print(json.dumps({"status": "frozen_for_separate_evaluation", "path": str(dest)})); return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/graph_classifier.json")
    parser.add_argument("--mode", choices=("audit", "train", "smoke", "package_smoke", "package_compare"), default="audit")
    parser.add_argument('--experiment-protocol',type=Path)
    parser.add_argument('--experiment-sha256')
    parser.add_argument("--fold", default="grouped")
    parser.add_argument("--kind", choices=("coverage", "agnostic", "rgcn"), default="rgcn")
    parser.add_argument("--ablation", choices=("coverage", "cdg", "ddg", "both", "full"), default="full")
    parser.add_argument("--seed", type=int, default=20261030)
    from .package_training import add_arguments,run_package_acceptance
    add_arguments(parser)
    args = parser.parse_args(argv)
    if args.mode=='package_compare':
        from .package_comparison import run_comparison
        return run_comparison(args.experiment_protocol,args.experiment_sha256)
    if args.mode=='package_smoke':return run_package_acceptance(args)
    c = configuration(args.config)
    if args.mode == "smoke":
        from .graph_smoke import run_smoke
        run_smoke(c); return 0
    if args.mode == "train": return real_train(c, args.config.resolve(), args.fold, args.kind, args.ablation, args.seed)
    from cwm_fp.baselines.boundaries import install_io_guard
    install_io_guard("graph_audit")
    report = readiness(c); write_json(c["work_dir"] / "readiness.json", report)
    print(json.dumps({"status": report["status"], "output": str(c["work_dir"] / "readiness.json")})); return 0


if __name__ == "__main__":
    sys.exit(main())
