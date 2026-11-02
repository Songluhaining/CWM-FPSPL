"""Allowlisted coverage and STATIC slice observations; no oracle/label loading."""
from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from cwm_fp.data.benchmark_adapter import load_sources
from cwm_fp.data.common import KEY, digest, indexed, key, read_csv, sha256


@dataclass
class Observation:
    identity: tuple[str, str, str]
    sample_id: str
    observed_result: str
    present: frozenset[str]
    covered: frozenset[str]
    passed: list[frozenset[str]]
    failed: list[frozenset[str]]
    slices: dict[str, frozenset[str]]
    ddu_complement: float
    coverage_ratio: float


def relative_java(path: str) -> str:
    path = path.replace("\\", "/")
    return path.rsplit("/src/", 1)[-1]


def hits(line: ET.Element) -> bool:
    # No arbitrary attribute dictionaries survive this boundary, including buggy.
    return any(int(line.get(name, "0")) > 0 for name in ("count", "truecount", "falsecount"))


def aggregate_mapping(paths: list[Path]) -> dict[tuple[str, str], str]:
    mapping = {}
    for path in paths:
        if not path.is_file():
            continue
        for file in ET.parse(path).getroot().findall("project/package/file"):
            name = relative_java(file.get("path", ""))
            for line in file.findall("line"):
                composed = (name, line.get("num", ""))
                feature, number = line.get("featureClass"), line.get("featureLineNum")
                if not all(composed) or not feature or not number:
                    raise ValueError(f"Incomplete statement mapping: {path}: {composed}")
                statement = feature + "." + number
                if composed in mapping and mapping[composed] != statement:
                    raise ValueError(f"Ambiguous statement mapping: {path}: {composed}")
                mapping[composed] = statement
    if not mapping:
        raise ValueError(f"No aggregate mapping: {paths}")
    return mapping


def test_coverage(path: Path, mapping: dict, *, hit_policy="all_counts", diagnostics=None) -> tuple[frozenset[str], frozenset[tuple[str, str]]]:
    if hit_policy not in {"all_counts", "source_count_only"}:
        raise ValueError("Unknown coverage hit policy")
    mapped, composed_hits = set(), set()
    for file in ET.parse(path).getroot().findall("project/package/file"):
        name = relative_java(file.get("path", ""))
        for line in file.findall("line"):
            count_hit = int(line.get("count", "0")) > 0
            if diagnostics is not None and hits(line) and not count_hit:
                diagnostics['branch_only_records'] += 1
                if (name, line.get('num', '')) not in mapping:
                    diagnostics['unmapped_branch_only_records'] += 1
            selected = count_hit if hit_policy == "source_count_only" else hits(line)
            if selected:
                composed = (name, line.get("num", ""))
                if composed not in mapping:
                    raise ValueError(f"Executed statement has no mapping: {path}: {composed}")
                mapped.add(mapping[composed])
                composed_hits.add(composed)
    return frozenset(mapped), frozenset(composed_hits)


def ddu_complement(paths: list[frozenset], present: set) -> float:
    n, m = len(paths), len(present)
    if not n or not m:
        raise ValueError("DDU needs observed tests and statements")
    density = 1 - abs(1 - 2 * sum(len(x) for x in paths) / (n * m))
    counts = Counter(paths)
    diversity = 1 - sum(c * (c - 1) for c in counts.values()) / (n * (n - 1)) if n > 1 else 0.0
    signatures = {tuple(s in path for path in paths) for s in present}
    uniqueness = len(signatures) / m
    return float(1 - density * diversity * uniqueness)


def read_slices(raw_variant: Path, derived_variant: Path) -> tuple[dict, dict]:
    raw_files = {p.relative_to(raw_variant / "src").as_posix(): sha256(p) for p in (raw_variant / "src").rglob("*.java")}
    derived_files = {p.relative_to(derived_variant / "src").as_posix(): sha256(p) for p in (derived_variant / "src").rglob("*.java")}
    if not raw_files or raw_files != derived_files:
        raise ValueError(f"Static slice source does not match original product: {derived_variant}")
    merged, hashes = defaultdict(set), {}
    for name in ("slice.all_statements.forward_slicing.log", "slice.all_statements.backward_slicing.log"):
        path = derived_variant / name
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(value, dict):
            raise ValueError(f"Invalid static slice table: {path}")
        for statement, dependencies in value.items():
            if not isinstance(dependencies, list) or not all(isinstance(s, str) for s in dependencies):
                raise ValueError(f"Invalid static slice targets: {path}")
            for token in [statement, *dependencies]:
                if not re.fullmatch(r"[\w.$]+\.\d+", token):
                    raise ValueError(f"Non-statement field in slice table: {path}: {token}")
            merged[statement].update(dependencies)
        hashes[path.as_posix()] = sha256(path)
    return {s: frozenset(v) for s, v in merged.items()}, {"source_tree_sha256": digest(raw_files), "slice_files": hashes}


def collect_product(p, local_tests, source, bounds):
    k = key(p)
    if p["suite_view"] != "original" or p.get("label") or p.get("label_provenance") or p.get("origin") != "unknown":
        raise ValueError("Only unlabelled original-suite manifests are accepted")
    variant = Path(p["source_variant_path"])
    if variant.resolve() != (source.root / k[1] / "variants" / k[2]).resolve():
        raise ValueError(f"Unexpected source/product identity: {variant}")
    aggregates = [variant / "coverage/spectrum_passed_coverage.xml"]
    if p["observed_result"] == "FAIL": aggregates.append(variant / "coverage/spectrum_failed_coverage.xml")
    mapping = aggregate_mapping(aggregates)
    passed, failed, activities, evidence = [], [], [], []
    coverage_diagnostics = Counter()
    coverage_root = (variant / 'coverage').resolve()
    from .boundaries import resolved
    for t in local_tests:
        file = Path(t["coverage_path"])
        if not resolved(str(file)).is_relative_to(coverage_root) or t["suite_id"] != p["suite_id"]:
            raise ValueError(f"Test source/suite mismatch: {file}")
        cov, composed = test_coverage(file, mapping,
            hit_policy=bounds.get('coverage_hit_policy', 'all_counts'), diagnostics=coverage_diagnostics)
        (passed if t["result"] == "PASS" else failed).append(cov)
        activities.append(composed)
        evidence.append((t["test_id"], t["result"], sha256(file)))
    if not activities or (bool(failed) != (p["observed_result"] == "FAIL")):
        raise ValueError(f"Missing/contradictory original tests: {k}")
    derived = source.label_root / k[1] / "variants" / k[2]
    slices, slice_evidence = read_slices(variant, derived)
    present = frozenset(mapping.values())
    covered = frozenset().union(*passed, *failed)
    observation = Observation(k, p["sample_id"], p["observed_result"], present, covered,
        passed, failed, slices, ddu_complement(activities, set(mapping)), len(covered) / len(present))
    provenance = {"sample_id": p["sample_id"], "test_content_sha256": digest(evidence),
        "coverage_diagnostics": dict(coverage_diagnostics),
        "aggregate_sha256": {path.as_posix(): sha256(path) for path in aggregates}, **slice_evidence}
    return observation, provenance


def cached_product(p, local_tests, source, bounds, cache):
    """Resume only after content hashes of every input have been rechecked."""
    import joblib
    from cwm_fp.data.common import write_json
    cache.mkdir(parents=True, exist_ok=True)
    data, receipt = cache/(p['sample_id']+'.joblib'), cache/(p['sample_id']+'.json')
    signature=digest([p,local_tests,str(source.root),str(source.label_root),bounds.get('coverage_hit_policy','all_counts')])
    if receipt.exists():
        meta=json.loads(receipt.read_text(encoding='utf-8'))
        if meta['input_records_sha256']!=signature or meta['data_sha256']!=sha256(data):
            raise ValueError('Observation checkpoint identity mismatch')
        observation, evidence = joblib.load(data)
        for path, expected in (evidence['aggregate_sha256'] | evidence['slice_files']).items():
            if sha256(Path(path))!=expected: raise ValueError('Cached source content changed')
        content=[(t['test_id'],t['result'],sha256(Path(t['coverage_path']))) for t in local_tests]
        if digest(content)!=evidence['test_content_sha256']: raise ValueError('Cached test coverage content changed')
        k=key(p);_, verified=read_slices(Path(p['source_variant_path']),source.label_root/k[1]/'variants'/k[2])
        if verified['source_tree_sha256']!=evidence['source_tree_sha256']: raise ValueError('Cached Java source changed')
        return observation,evidence
    if data.exists(): raise ValueError('Unsealed partial checkpoint retained; choose a new output directory')
    result=collect_product(p,local_tests,source,bounds)
    temporary=data.with_suffix('.partial.joblib')
    joblib.dump(result,temporary,compress=3)
    temporary.replace(data)
    write_json(receipt,{'input_records_sha256':signature,'data_sha256':sha256(data)})
    return result


def collect(products_path: Path, tests_path: Path, sources_path: Path, bounds: dict, *, cache_dir=None) -> tuple[list[Observation], dict]:
    products = indexed(read_csv(products_path, (*KEY, "sample_id", "suite_view")))
    tests = read_csv(tests_path, (*KEY, "test_id", "result", "coverage_path"))
    versions = {k[:2] for k in products}
    if len(products) > bounds["max_products"] or len(tests) > bounds["max_tests"] or len(versions) > bounds["max_versions"]:
        raise ValueError("Explicit observation collection bounds exceeded")
    sources = {s.system_id: s for s in load_sources(sources_path)}
    by_product = defaultdict(list)
    seen = set()
    for t in tests:
        k = key(t)
        tk = (*k, t["test_id"])
        if k not in products or tk in seen:
            raise ValueError("Unknown/duplicate test")
        seen.add(tk)
        by_product[k].append(t)
    workers=int(bounds.get('observation_workers',1))
    if not 1<=workers<=8: raise ValueError('Observation workers must be between 1 and 8')
    results={}
    def worker(k,p):
        args=(p,by_product[k],sources[k[0]],bounds)
        return cached_product(*args,Path(cache_dir)) if cache_dir is not None else collect_product(*args)
    if workers==1:
        for i,(k,p) in enumerate(sorted(products.items()),1):
            results[k]=worker(k,p)
            if i%50==0: print(f'coverage/static-slice observations: {i}/{len(products)}',flush=True)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            jobs={pool.submit(worker,k,p):k for k,p in sorted(products.items())}
            for future in as_completed(jobs):
                results[jobs[future]]=future.result()
                if len(results)%50==0: print(f'coverage/static-slice observations: {len(results)}/{len(products)}',flush=True)
    observations=[results[k][0] for k in sorted(products)]
    provenance=[results[k][1] for k in sorted(products)]
    return observations, {"products": len(products), "tests": len(tests), "versions": len(versions),
        "products_sha256": sha256(products_path), "tests_sha256": sha256(tests_path),
        "provenance": provenance, "mapping_confidence": "attribute consistency checked, not semantic proof",
        "slice_evidence": "static, original and derived Java sources byte-identical; slice semantics not independently verified",
        "forbidden_xml_attributes_exported": [], "xml_body_read": True,
        "coverage_hit_policy": bounds.get('coverage_hit_policy', 'all_counts')}
