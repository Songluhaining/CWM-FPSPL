#!/usr/bin/env python3
"""Validate canonical CWM-FP input contracts without model dependencies."""
from __future__ import annotations
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cwm_fp.data.common import read_csv as strict_read_csv, write_json

PRODUCT_FIELDS = {"system_id", "bug_version_id", "product_id", "observed_result", "label", "origin", "source_variant_path", "feature_config_path", "test_suite_path", "label_provenance"}
TEST_FIELDS = {"system_id", "bug_version_id", "product_id", "test_id", "result", "coverage_path", "trace_path"}
NODE_TYPES = {"feature", "statement", "value", "assertion", "output"}
EDGE_TYPES = {"feature_controls", "control_dep", "data_dep", "produces", "consumes", "observes", "temporal"}
EDGE_EVIDENCE = {"static", "dynamic", "observed_intervention", "unknown"}
KEY_FIELDS = ("system_id", "bug_version_id", "product_id")


def read_csv(path, required):
    return strict_read_csv(Path(path), required)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for no, line in enumerate(f, 1):
            if line.strip():
                try:
                    yield no, json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"{path}:{no}: invalid JSON: {e}") from e


def validate_products(path):
    errors, products = [], {}
    for n, p in enumerate(read_csv(path, PRODUCT_FIELDS), 2):
        key = tuple(p[k] for k in KEY_FIELDS)
        if not all(key):
            errors.append(f"products row {n}: empty product key")
        if key in products:
            errors.append(f"products row {n}: duplicate key {key}")
        products[key] = p
        label, result = p["label"], p["observed_result"]
        if label not in {"F", "FP", "TP", ""}:
            errors.append(f"products row {n}: invalid label {label!r}")
        if result not in {"PASS", "FAIL"}:
            errors.append(f"products row {n}: invalid observed_result {result!r}")
        if (label == "F" and result != "FAIL") or (label in {"FP", "TP"} and result != "PASS"):
            errors.append(f"products row {n}: label/result contradiction")
        if p["origin"] not in {"natural", "simulated", "unknown"}:
            errors.append(f"products row {n}: invalid origin")
    return products, errors


def validate_tests(path, products):
    errors, tests = [], {}
    for n, t in enumerate(read_csv(path, TEST_FIELDS), 2):
        key = tuple(t[k] for k in KEY_FIELDS) + (t["test_id"],)
        if not all(key) or key in tests:
            errors.append(f"tests row {n}: incomplete/duplicate test key {key}")
        tests[key] = t
        if key[:3] not in products:
            errors.append(f"tests row {n}: unknown product {key[:3]}")
        if t["result"] not in {"PASS", "FAIL", "UNKNOWN"}:
            errors.append(f"tests row {n}: invalid result")
    per_product = defaultdict(list)
    for k, t in tests.items():
        per_product[k[:3]].append(t["result"])
    for k, results in per_product.items():
        if k not in products:
            continue
        observed = products[k]["observed_result"]
        if observed == "PASS" and "FAIL" in results:
            errors.append(f"tests for {k}: product PASS but original test FAIL")
        if observed == "FAIL" and "FAIL" not in results and "UNKNOWN" not in results:
            errors.append(f"tests for {k}: product FAIL but no failed original test")
    return tests, errors


def validate_graphs(path, tests):
    errors, count = [], 0
    for n, g in load_jsonl(path):
        count += 1
        key = tuple(g.get(k) for k in KEY_FIELDS) + (g.get("test_id"),)
        if tests is not None and key not in tests:
            errors.append(f"graph line {n}: graph without matching original test {key}")
        if g.get("tier") not in {"GRAPH", "VALUE"}:
            errors.append(f"graph line {n}: invalid tier; coverage alone is not a LDDG")
        nodes = g.get("nodes")
        edges = g.get("edges")
        if not isinstance(nodes, list) or not isinstance(edges, list):
            errors.append(f"graph line {n}: nodes/edges must be lists")
            continue
        ids = [node.get("id") for node in nodes]
        if len(ids) != len(set(ids)):
            errors.append(f"graph line {n}: duplicate node ID")
        for node in nodes:
            if node.get("type") not in NODE_TYPES:
                errors.append(f"graph line {n}: unsupported node type {node.get('type')}")
        nodeids = set(ids)
        for e in edges:
            if e.get("source") not in nodeids or e.get("target") not in nodeids:
                errors.append(f"graph line {n}: edge with missing endpoint")
            if e.get("type") not in EDGE_TYPES:
                errors.append(f"graph line {n}: unsupported edge type")
            if e.get("evidence") not in EDGE_EVIDENCE:
                errors.append(f"graph line {n}: edge lacks valid evidence")
            if e.get("event_index") is not None and (not isinstance(e["event_index"], int) or e["event_index"] < 0):
                errors.append(f"graph line {n}: invalid event_index")
        if g.get("provenance", {}).get("coverage_only", False):
            errors.append(f"graph line {n}: cannot call coverage-only data a dynamic graph")
    return count, errors


def validate_events(path):
    errors, count, last = [], 0, {}
    for n, e in load_jsonl(path):
        count += 1
        key = tuple(e.get(k) for k in KEY_FIELDS) + (e.get("test_id"),)
        idx = e.get("event_index")
        if not isinstance(idx, int) or idx < 0:
            errors.append(f"events line {n}: invalid event index")
        elif key in last and idx <= last[key]:
            errors.append(f"events line {n}: non-increasing event index for {key}")
        else:
            last[key] = idx
        if not e.get("event_type") or not e.get("statement_id"):
            errors.append(f"events line {n}: missing event_type/statement_id")
    return count, errors


def validate_interventions(path):
    errors, count = [], 0
    for n, row in load_jsonl(path):
        count += 1
        for key in ("intervention_id", "system_id", "bug_version_id", "product_id", "base_test_id", "action_type", "action_target", "valid", "execution_kind", "oracle_available", "observed_failure"):
            if key not in row:
                errors.append(f"intervention line {n}: missing {key}")
        kind = row.get("execution_kind")
        if kind not in {"REAL_RERUN", "MODEL_SIMULATION"}:
            errors.append(f"intervention line {n}: invalid execution_kind")
        if kind == "MODEL_SIMULATION" and row.get("observed_failure") is True:
            errors.append(f"intervention line {n}: simulated failure falsely claimed as observed")
        if row.get("observed_failure") is True and (not row.get("oracle_available") or not row.get("valid") or kind != "REAL_RERUN"):
            errors.append(f"intervention line {n}: unverifiable failure claim")
    return count, errors


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--products", type=Path, required=True)
    p.add_argument("--tests", type=Path)
    p.add_argument("--graphs", type=Path)
    p.add_argument("--events", type=Path)
    p.add_argument("--interventions", type=Path)
    p.add_argument("--labels", type=Path, help="Sealed labels; audit/evaluation only")
    p.add_argument("--splits", type=Path, nargs="*", default=[])
    p.add_argument("--check-paths", action="store_true")
    p.add_argument("--out", type=Path, help="Persist audit JSON")
    a = p.parse_args(argv)
    if a.labels or a.splits:
        from cwm_fp.data.validation import audit
        try:
            if not a.tests:
                raise ValueError("--tests required for real manifest audit")
            report = audit(a.products, a.tests, a.labels, a.splits, check_paths=a.check_paths)
        except (ValueError, OSError, KeyError) as exc:
            report = {"valid": False, "errors": [str(exc)]}
        if a.out:
            # Validation reports can be written only within this workspace's outputs/processed.
            root = Path(__file__).resolve().parents[1]
            output = a.out.resolve()
            if not any(output.is_relative_to((root / name).resolve()) for name in ("outputs", "data/processed")):
                p.error("audit output must be under outputs/ or data/processed/")
            write_json(output, report)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report["valid"] else 1
    products, errors = validate_products(a.products)
    tests = None
    counts = {"products": len(products)}
    if a.tests:
        tests, es = validate_tests(a.tests, products)
        errors += es
        counts["tests"] = len(tests)
    if a.graphs:
        counts["graphs"], es = validate_graphs(a.graphs, tests)
        errors += es
    if a.events:
        counts["events"], es = validate_events(a.events)
        errors += es
    if a.interventions:
        counts["interventions"], es = validate_interventions(a.interventions)
        errors += es
    print(json.dumps({"counts": counts, "valid": not errors, "errors": errors}, indent=2, ensure_ascii=False))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
