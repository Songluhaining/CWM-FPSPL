"""Original-suite, full-key graph joins and fail-closed empirical eligibility."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from cwm_fp.data.common import KEY, indexed, key, read_csv, sha256
from cwm_fp.data.splits import validate_split
from .graph_encoder import Limits, ProductTensor, check_metadata, encode_test, encode_test_lossless, evidence_errors, dependency_errors, FEATURE_SCHEMA


def load_corpus(products_path: Path, tests_path: Path, graph_path: Path, split_path: Path, *, holdout=None):
    products = indexed(read_csv(products_path, (*KEY, "observed_result", "suite_view", "sample_id")))
    splits = indexed(read_csv(split_path, (*KEY, "split")))
    validate_split(products, splits, holdout=holdout)
    if any(p.get("label") or p.get("label_provenance") or p.get("origin") != "unknown" or p["suite_view"] != "original" for p in products.values()):
        raise ValueError("Use only sealed-label, original-suite public manifests")
    if len({p["sample_id"] for p in products.values()}) != len(products):
        raise ValueError("Duplicate public sample identity")
    tests, expected = {}, defaultdict(set)
    for t in read_csv(tests_path, (*KEY, "test_id", "result")):
        k = key(t); tk = (*k, t["test_id"])
        if tk in tests or k not in products or t.get("suite_id") != products[k].get("suite_id"):
            raise ValueError("Duplicate/unknown test or mismatched suite")
        if products[k]["observed_result"] == "PASS" and t["result"] != "PASS":
            raise ValueError("Contradictory product/test result")
        tests[tk] = t; expected[k].add(t["test_id"])
    graphs = defaultdict(list); seen = set()
    if graph_path.stat().st_size > 128 * 1024 * 1024:
        raise ValueError("Phase-04 bounded pilot accepts at most 128 MiB graphs")
    with graph_path.open(encoding="utf-8") as f:
        for line in f:
            g = json.loads(line); check_metadata(g); k = key(g); tk = (*k, g["test_id"])
            if tk in seen or tk not in tests: raise ValueError("Duplicate/unknown graph lineage")
            outputs = [n for n in g["nodes"] if n["type"] == "output" and n["attrs"].get("kind") == "original_test_result"]
            if len(outputs) != 1 or outputs[0]["attrs"].get("observed") is not True or outputs[0]["attrs"].get("result") != tests[tk]["result"]:
                raise ValueError("Graph output does not match original test evidence")
            seen.add(tk); graphs[k].append(g)
    audit = {"graphs": len(seen), "products_with_graphs": len(graphs),
             "version_groups": len({k[:2] for k in graphs}),
             "input_sha256": {"products": sha256(products_path), "tests": sha256(tests_path), "graphs": sha256(graph_path), "split": sha256(split_path)},
             "structure_qualified": sum(not dependency_errors(g) for gs in graphs.values() for g in gs),
             "mapping_complete": sum(g["quality"].get("canonical_mapping_complete") is True for gs in graphs.values() for g in gs),
             'program_identity_complete':sum(g['quality'].get('program_identity_complete') is True for gs in graphs.values() for g in gs),
             'identity_and_semantics_qualified':sum(not evidence_errors(g) for gs in graphs.values() for g in gs),
             "labels_read": False, "products": [], "eligible_by_partition": {p: 0 for p in ("train", "val", "test")}}
    eligible = {}
    for k, gs in sorted(graphs.items()):
        p = products[k]; partition = splits[k]["split"]
        reasons = []
        if p["observed_result"] != "PASS": reasons.append("original_product_failed_excluded_from_FP_TP")
        if {g["test_id"] for g in gs} != expected[k]: reasons.append("incomplete_original_test_collection")
        suite={'complete':{g['test_id'] for g in gs}==expected[k], 'expected_tests':len(expected[k]),'collected_tests':len(gs),
               'basis':'registered_original_suite_not_absolute_historical_completeness'}
        for g in gs:
            if g.get('provenance',{}).get('identity_schema') and g['quality'].get('suite_completeness')!=suite:
                reasons.append('suite_completeness_claim_disagrees_with_manifest')
        reasons.extend(sorted({e for g in gs for e in evidence_errors(g)}))
        record = {"identity": list(k), "sample_id": p["sample_id"], "partition": partition,
                  "observed_result": p["observed_result"], "manifest_tests": len(expected[k]),
                  "collected_tests": len(gs), "missing_tests": len(expected[k])-len(gs), "reasons": reasons}
        record['suite_completeness']=suite
        audit["products"].append(record)
        if not reasons:
            eligible[k] = sorted(gs, key=lambda g: g["test_id"])
            audit["eligible_by_partition"][partition] += 1
    audit["original_PASS_with_graphs"] = sum(products[k]["observed_result"] == "PASS" for k in graphs)
    audit["raw_PASS_by_partition"] = dict(Counter(splits[k]["split"] for k in graphs if products[k]["observed_result"] == "PASS"))
    audit["missing_graph_PASS_products"] = sum(p["observed_result"] == "PASS" and k not in graphs for k, p in products.items())
    audit["blockers"] = (["fewer_than_three_version_groups"] if audit["version_groups"] < 3 else [])
    audit["blockers"] += ["no_eligible_" + p + "_products" for p, n in audit["eligible_by_partition"].items() if not n]
    audit["ready"] = not audit["blockers"]
    return products, splits, graphs, eligible, audit


def tensor_products(products, splits, eligible, ablation='full', limits: Limits | None=None, *, feature_schema=FEATURE_SCHEMA):
    """Default is complete input. Explicit Limits retain legacy sampling audits.

    Formal fitting rejects a nonzero loss audit. Large products should use the
    per-test ProductPackage iterator instead of materializing this list.
    """
    out = []
    for k, graphs in sorted(eligible.items()):
        if products[k]['observed_result']!='PASS':raise ValueError('Only original PASS products may become classification tensors')
        if any(g.get('provenance',{}).get('identity_schema') for g in graphs):
            for g in graphs:
                suite=g['quality'].get('suite_completeness') or {}
                if not suite.get('complete') or suite.get('expected_tests')!=len(graphs) or suite.get('collected_tests')!=len(graphs):
                    raise ValueError('A qualified test graph is not a complete original product suite')
        selected = graphs if limits is None else graphs[:limits.max_tests]
        if limits is None:
            if ablation!='full' or feature_schema!=FEATURE_SCHEMA:raise ValueError('Default lossless path requires full relations and 22 features; choose legacy options explicitly')
            tests=[encode_test_lossless(g) for g in selected]
        else:tests = [encode_test(g, ablation, limits,feature_schema=feature_schema) for g in selected]
        total_edges = sum(len(g["edges"]) for g in graphs)
        total_events = sum(g["quality"]["events"] for g in graphs)
        audit = {"preview_only": False, "tests_before": len(graphs), "tests_after": len(tests),
                 'lost_node_fraction':1-sum(t.metadata['nodes_after_sampling'] for t in tests)/max(1,sum(len(g['nodes']) for g in graphs)),
                 "lost_test_fraction": 1-len(tests)/len(graphs),
                 "lost_event_fraction": 1-sum(t.metadata["events_after"] for t in tests)/max(1, total_events),
                 "lost_edge_fraction": 1-sum(t.metadata["edges_after_sampling"] for t in tests)/total_edges if total_edges else 0.}
        p = products[k]
        out.append(ProductTensor(k, p["sample_id"], splits[k]["split"], p["observed_result"], tests, audit))
    return out
