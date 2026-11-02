"""Explicitly artificial graph tests; never empirical benchmark observations."""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import torch

from cwm_fp.data.common import ROOT, digest, sha256, write_json
from .graph_attribution import attribution
from .graph_data import tensor_products
from .graph_encoder import Limits
from .graph_evaluator import evaluate_once, score_rows
from .graph_metrics import binary_metrics, clustered_interval
from .train_graph import TrainOptions, fit, predict


def synthetic_graph(identity, test_id, label, *, coverage_signal=False):
    def node(i, t, **attrs): return {"id": i, "type": t, "attrs": attrs}
    nodes = [node("f:a", "feature", enabled=True, constraints_verified=True),
             node("s:b", "statement", available=True, executed=True, opcode=153, canonical_statement_id="toy|B|branch|1|0"),
             node("s:a", "statement", available=True, executed=bool(label) if coverage_signal else True, opcode=3, canonical_statement_id="toy|B|a|2|0"),
             node("s:c", "statement", available=True, executed=bool(label) if coverage_signal else True, opcode=21, canonical_statement_id="toy|B|c|3|0")]
    for i, opcode in enumerate((3, 21, 172)):
        nodes.append(node(f"e:{i}", "statement", event_index=i, opcode=opcode, executed=True, canonical_statement_id=f"toy|B|m|{i+4}|0"))
        if i < 2: nodes.append(node(f"v:{i}", "value", definition_event=i, observed=False))
    nodes += [node("a:0", "assertion", observed=True, passed=True), node("o:0", "output", observed=True, result="PASS", kind="original_test_result")]
    def edge(a, b, t, **kwargs):
        return {"source": a, "target": b, "type": t, "evidence": "static" if t in {"feature_controls", "control_dep"} else "dynamic", "origin": "explicit_synthetic_fixture", **kwargs}
    # Same features, counts and coverage across classes. Only CDG/DDG topology differs.
    source = int(label)
    edges = [edge("f:a", "s:b", "feature_controls"), edge("s:b", f"e:{source}", "control_dep"),
             edge(f"e:{source}", "e:2", "data_dep", definition_event=source, event_index=2),
             edge("e:0", "v:0", "produces", event_index=0), edge("e:1", "v:1", "produces", event_index=1),
             edge(f"v:{source}", "e:2", "consumes", event_index=2),
             edge("e:2", "a:0", "observes", event_index=2), edge("e:2", "o:0", "observes", event_index=2),
             edge("e:0", "e:1", "temporal", event_index=1), edge("e:1", "e:2", "temporal", event_index=2)]
    return {**dict(zip(("system_id", "bug_version_id", "product_id"), identity)), "test_id": test_id, "tier": "GRAPH", "nodes": nodes, "edges": edges,
            "quality": {"eligible_graph_structure": True, "verified_tier": "T1", "canonical_mapping_complete": True,
                        "constraints_verified": True, "events": 3},
            "provenance": {"producer": "phase04_synthetic_fixture_NOT_measured", "synthetic": True, "learnable": True, "coverage_only": False}}


def fixture(ablation, limits=Limits(), *, coverage_signal=False):
    products, splits, graphs, labels = {}, {}, {}, {}
    for part in ("train", "val", "test"):
        for version in range(2):
            for i in range(4):
                k = ("SyntheticOnly", f"{part}_version{version}", f"p{i}")
                labels[k] = "FP" if i % 2 else "TP"
                products[k] = {"sample_id": digest(k), "observed_result": "PASS"}
                splits[k] = {"split": part}
                graphs[k] = [synthetic_graph(k, f"test{j}", i % 2, coverage_signal=coverage_signal) for j in range(2)]
    return tensor_products(products, splits, graphs, ablation, limits), labels, {k: r["split"] for k, r in splits.items()}


def run_smoke(c):
    out = c["work_dir"] / "synthetic"
    out.mkdir(parents=True, exist_ok=False)
    source_paths = sorted(Path(__file__).parent.glob("*.py"))
    source_paths += [ROOT / "configs/graph_classifier.json"]
    write_json(out / "source_fingerprint.json", {p.relative_to(ROOT).as_posix(): sha256(p) for p in source_paths})
    start = time.monotonic()
    reports, predictions, evaluation_truth = [], [], {}
    combinations = [("coverage", "coverage")] + [(k, a) for k in ("agnostic", "rgcn") for a in ("cdg", "ddg", "both")]
    for seed in c["smoke_seeds"]:
        for kind, ablation in combinations:
            if time.monotonic() - start > c["smoke_max_seconds"]: raise TimeoutError("Synthetic experiment budget exhausted")
            items, labels, assignments = fixture(ablation, Limits(**c["limits"]))
            train, val, test = ([p for p in items if p.partition == part] for part in ("train", "val", "test"))
            train_targets = {p.identity: labels[p.identity] for p in train}
            def validation(rows, _remaining):
                if any(r["partition"] != "val" for r in rows): raise ValueError("Only validation predictions may guide early stopping")
                return binary_metrics([int(labels[tuple(r["identity"])] == "FP") for r in rows], [r["y_score"] for r in rows])
            options = TrainOptions(**(c["training"] | {"seed": seed, "max_seconds": min(c["training"]["max_seconds"], c["smoke_max_seconds"]-(time.monotonic()-start))}))
            model, report = fit(kind, train, train_targets, val, assignments, validation, options)
            if kind != "coverage" and report["train_resubstitution"]["accuracy"] < .95:
                raise AssertionError(f"Synthetic graph overfit failed: {kind}/{ablation}/{seed}")
            run_id = f"{kind}_{ablation}_{seed}"
            report.update(run_id=run_id, ablation=ablation, synthetic=True)
            torch.save(model.state_dict(), out / (run_id + ".pt"))
            reports.append(report)
            predictions.append({"run_id": run_id, "rows": predict(model, test, options.batch_products)})
            # Truth is held by evaluator and used only after all candidates freeze.
            evaluation_truth[run_id] = {p.sample_id: {"identity": list(p.identity), "y_true": int(labels[p.identity] == "FP")} for p in test}
            if kind == "rgcn" and ablation == "both" and seed == c["smoke_seeds"][0]:
                attribution(model, train[0], out / "attribution", synthetic=True)
            print(f"synthetic {run_id}: train_acc={report['train_resubstitution']['accuracy']:.3f}, epochs={report['epochs']}, seconds={report['training_seconds']:.3f}", flush=True)
    # Coverage-only has no signal in the structural control. Separately check
    # that its forward/backward path can overfit an explicit count-based fixture.
    remaining = c["smoke_max_seconds"] - (time.monotonic()-start)
    if remaining <= 0: raise TimeoutError("Synthetic experiment budget exhausted")
    items, labels, assignments = fixture("coverage", coverage_signal=True)
    train, val = ([p for p in items if p.partition == part] for part in ("train", "val"))
    model, coverage_check = fit("coverage", train, {p.identity: labels[p.identity] for p in train}, val, assignments,
        lambda rows, remaining: binary_metrics([int(labels[tuple(r["identity"])] == "FP") for r in rows], [r["y_score"] for r in rows]),
        TrainOptions(**(c["training"] | {"seed": c["smoke_seeds"][0], "max_seconds": min(remaining, c["training"]["max_seconds"])})))
    if coverage_check["train_resubstitution"]["accuracy"] < .95: raise AssertionError("Coverage MLP overfit failed")
    pred_path = out / "frozen_test_predictions.json"
    write_json(pred_path, predictions)
    def evaluate(bundle):
        records = []
        for run in bundle:
            truth = evaluation_truth[run["run_id"]]
            m = score_rows(run["rows"], truth, partition="test", expected_ids=list(truth))
            joined = [r | truth[r["sample_id"]] for r in run["rows"]]
            m["cluster_bootstrap"] = clustered_interval(joined, c["smoke_seeds"][0], replicates=50)
            records.append({"run_id": run["run_id"], "metrics": m,
                            "errors": [r for r in joined if r["y_true"] != r["y_pred"]]})
        return {"synthetic": True, "not_benchmark_performance": True, "test_evaluations_per_model_seed": 1, "runs": records}
    result = evaluate_once(out / "test_evaluation.json", pred_path, sha256(pred_path), evaluate)
    write_json(out / "training.json", {"synthetic": True, "runs": reports, "separate_coverage_overfit": coverage_check,
               "total_seconds": time.monotonic()-start, "test_prediction_sha256": sha256(pred_path),
               "environment": {"python": sys.version, "executable": sys.executable, "torch": torch.__version__,
                               "cuda_available": torch.cuda.is_available(), "device": c["training"]["device"], "platform": platform.platform()},
               "limits": c["limits"], "scope": "two reusable artificial topologies across six synthetic versions; engineering verification only"})
    print(json.dumps({"synthetic_runs": len(reports), "separate_coverage_overfit": True, "test_evaluations": len(result["runs"])}))
