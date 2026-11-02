"""Build bounded manifests from the audited SPL report/coverage layout, read-only."""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .common import (ALIASES, KEY, digest, generated_dir, indexed, key, read_csv,
                     sha256, write_csv, write_json)

PRODUCT_FIELDS = (*KEY, "sample_id", "observed_result", "label", "origin",
                  "source_variant_path", "feature_config_path", "test_suite_path",
                  "label_provenance", "features_json", "suite_id", "suite_view",
                  "bug_count", "dataset_family", "eligible_for_paper_subset",
                  "source_report_sha256", "test_count", "failed_test_count",
                  "aggregate_passed_exists", "aggregate_failed_exists",
                  "constraints_verified")
TEST_FIELDS = (*KEY, "test_id", "result", "coverage_path", "trace_path", "suite_id",
               "coverage_size_bytes", "result_evidence")
VERSION = re.compile(r"_MultipleBugs_\.NOB_([123])\.ID_\d+")


@dataclass(frozen=True)
class Source:
    system_id: str
    alias: str
    root: Path
    label_root: Path | None
    label_source: str
    dataset_family: str


def report_path(version: Path) -> Path:
    """Completed reports are canonical; refuse conflicting intermediate copies."""
    complete, intermediate = version / "config.report.csv.done", version / "config.report.csv"
    if complete.is_file():
        if intermediate.is_file() and sha256(complete) != sha256(intermediate):
            raise ValueError(f"Conflicting completed/intermediate reports: {version}")
        return complete
    if intermediate.is_file():
        return intermediate
    raise ValueError(f"Missing config.report.csv.done or config.report.csv: {version}")


def load_sources(path: Path) -> list[Source]:
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    sources = []
    for spec in config["sources"]:
        alias = spec["system_alias"]
        if alias not in ALIASES:
            raise ValueError(f"Unknown system alias: {alias}")
        root = Path(spec["root"]).resolve()
        if not root.is_dir():
            raise ValueError(f"Missing source root: {root}")
        sources.append(Source(ALIASES[alias], alias, root,
                              Path(spec["label_root"]).resolve() if spec.get("label_root") else None,
                              spec.get("label_source", "unknown"), spec["dataset_family"]))
    if len({s.system_id for s in sources}) != len(sources):
        raise ValueError("Repeated canonical system; resolve duplicate source roots explicitly")
    return sources


def scan(sources: list[Source], *, max_versions: int, max_products: int,
         smoke: bool = False) -> tuple[list[dict], list[dict], dict]:
    if max_versions < 1 or max_products < 1:
        raise ValueError("Bounds must be positive")
    products, tests, provenance = [], [], []
    total_versions = 0
    for source in sorted(sources, key=lambda s: s.system_id):
        if smoke and source.system_id != "BankAccountTP":
            continue
        versions = [p for p in source.root.iterdir() if p.is_dir() and (p / "variants").is_dir()]
        if smoke:
            versions = [p for p in versions if p.name == "_MultipleBugs_.NOB_1.ID_1"]
        for version in sorted(versions):
            match = VERSION.fullmatch(version.name)
            if not match:
                raise ValueError(f"Unsupported version identity (explicit adapter required): {version}")
            total_versions += 1
            if total_versions > max_versions:
                raise ValueError(f"Version bound {max_versions} exceeded; no manifest written")
            report = report_path(version)
            report_hash = sha256(report)
            rows = read_csv(report, ("Product\\Feature", "__TEST_OUTPUT__"))
            by_id = {}
            for row in rows:
                pid = row["Product\\Feature"].strip()
                if not pid or pid in by_id:
                    raise ValueError(f"{report}: empty/duplicate product ID {pid!r}")
                by_id[pid] = row
            actual = {p.name: p for p in (version / "variants").iterdir() if p.is_dir()}
            if set(actual) != set(by_id):
                raise ValueError(f"{version}: product directories/report differ: {sorted(set(actual) ^ set(by_id))}")
            selected = {"model_m_ca4_0001", "model_m_ca4_0002", "model_m_ca4_0003"} if smoke else set(by_id)
            if not selected.issubset(actual):
                raise ValueError(f"Missing explicitly named smoke products: {version}")
            provenance.append({"system_id": source.system_id, "source_alias": source.alias,
                               "bug_version_id": version.name, "report_path": report.as_posix(),
                               "report_sha256": report_hash, "available_products": len(actual),
                               "selected_products": len(selected)})
            for pid in sorted(selected):
                if len(products) >= max_products:
                    raise ValueError(f"Product bound {max_products} exceeded; no manifest written")
                row, variant = by_id[pid], actual[pid]
                observed = {"__PASSED__": "PASS", "__FAILED__": "FAIL"}.get(row["__TEST_OUTPUT__"].strip())
                if observed is None:
                    raise ValueError(f"Unknown report result: {report}: {pid}")
                features = {}
                for name, value in row.items():
                    if name in {"Product\\Feature", "__TEST_OUTPUT__"}:
                        continue
                    if value.strip() not in {"T", "F"}:
                        raise ValueError(f"Unknown feature value: {report}: {name}={value!r}")
                    features[name] = value.strip() == "T"
                identity = dict(zip(KEY, (source.system_id, version.name, pid)))
                local_tests, seen = [], set()
                for folder, result in (("passed", "PASS"), ("failed", "FAIL")):
                    for coverage in sorted((variant / "coverage" / folder).glob("*.xml")):
                        if not coverage.name.endswith(".coverage.xml"):
                            raise ValueError(f"Unrecognized test filename: {coverage}")
                        tid = coverage.name.removesuffix(".coverage.xml")
                        if not tid or tid in seen:
                            raise ValueError(f"Repeated/inconsistent original test: {variant}: {tid}")
                        seen.add(tid)
                        local_tests.append(identity | {"test_id": tid, "result": result,
                            "coverage_path": coverage.as_posix(), "trace_path": "",
                            "coverage_size_bytes": coverage.stat().st_size,
                            "result_evidence": "coverage_directory_not_rerun"})
                failed = sum(t["result"] == "FAIL" for t in local_tests)
                if (observed == "PASS" and failed) or (observed == "FAIL" and local_tests and not failed):
                    raise ValueError(f"Original report/test contradiction: {variant}")
                suite_id = "original:" + digest([key(identity), sorted(
                    (t["test_id"], t["result"], t["coverage_size_bytes"]) for t in local_tests)])
                for test in local_tests:
                    test["suite_id"] = suite_id
                tests.extend(local_tests)
                products.append(identity | {
                    "sample_id": digest(key(identity)), "observed_result": observed,
                    "label": "", "origin": "unknown", "label_provenance": "",
                    "source_variant_path": variant.as_posix(), "feature_config_path": report.as_posix(),
                    "test_suite_path": (variant / "test").as_posix(),
                    "features_json": json.dumps(features, sort_keys=True, separators=(",", ":")),
                    "suite_id": suite_id, "suite_view": "original", "bug_count": int(match[1]),
                    "dataset_family": source.dataset_family, "eligible_for_paper_subset": "unknown",
                    "source_report_sha256": report_hash, "test_count": len(local_tests),
                    "failed_test_count": failed,
                    "aggregate_passed_exists": int((variant / "coverage/spectrum_passed_coverage.xml").is_file()),
                    "aggregate_failed_exists": int((variant / "coverage/spectrum_failed_coverage.xml").is_file()),
                    "constraints_verified": "false"})
    if not products:
        raise ValueError("No products matched requested scope")
    indexed(products)
    tests.sort(key=lambda t: (*key(t), t["test_id"]))
    audit = {"schema_version": "0.2", "suite_view": "original", "smoke": smoke,
             "bounds": {"max_versions": max_versions, "max_products": max_products},
             "versions": total_versions, "products": len(products), "tests": len(tests),
             "systems": dict(Counter(p["system_id"] for p in products)),
             "aliases": ALIASES, "provenance": provenance,
             "test_identity": "full coverage filename minus .coverage.xml; product-scoped",
             "content_scope": "configuration reports hashed; XML file identity/size inventoried, bodies not read",
             "source_test_completeness": "unknown: inventory of existing execution artifacts, not all declared Java methods",
             "capabilities": {"coverage_inventory": True, "graph": False, "values": False, "interventions": False},
             "source_write_operations": 0}
    return products, tests, audit


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sources", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--max-versions", type=int, required=True)
    p.add_argument("--max-products", type=int, required=True)
    a = p.parse_args(argv)
    try:
        sources = load_sources(a.sources)
        out = generated_dir(a.out, [s.root for s in sources] + [s.label_root for s in sources if s.label_root])
        products, tests, audit = scan(sources, max_versions=a.max_versions,
                                      max_products=a.max_products, smoke=a.smoke)
        write_csv(out / "products.csv", products, PRODUCT_FIELDS)
        write_csv(out / "tests.csv", tests, TEST_FIELDS)
        audit["source_config_sha256"] = sha256(a.sources)
        audit["manifest_sha256"] = {n: sha256(out / n) for n in ("products.csv", "tests.csv")}
        write_json(out / "manifest_audit.json", audit)
        print(json.dumps({k: audit[k] for k in ("versions", "products", "tests", "systems", "smoke")}, ensure_ascii=False))
    except (ValueError, OSError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
