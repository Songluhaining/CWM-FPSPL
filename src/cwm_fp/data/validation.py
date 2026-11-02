"""Read-only manifest, sealed-target and split audits; no feature fitting."""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from .common import KEY, SYSTEMS, indexed, key, read_csv, sha256
from .inputs import load_inputs
from .splits import test_assignments, validate_split

# Literature values, not measurements. Transcribed in outputs/audit.md, section 2.
PAPER = {
    "BankAccountTP": {"versions": 187, "F": 2055, "FP": 2328, "TP": 1975},
    "Elevator": {"versions": 41, "F": 217, "FP": 326, "TP": 195},
    "Email": {"versions": 69, "F": 553, "FP": 587, "TP": 723},
    "ExamDB": {"versions": 77, "F": 201, "FP": 127, "TP": 288},
    "GPL": {"versions": 355, "F": 6612, "FP": 9995, "TP": 18538},
    "ZipMe": {"versions": 94, "F": 686, "FP": 828, "TP": 836},
}


def validate_sealed(labels: dict, products: dict) -> list[str]:
    errors = []
    if set(labels) != set(products):
        errors.append("Label/product key sets differ")
    for k in set(labels) & set(products):
        label, product = labels[k], products[k]
        value = label.get("label", "")
        if value not in {"", "F", "FP", "TP"}:
            errors.append(f"Invalid sealed label: {k}")
        if label.get("origin") not in {"natural", "simulated", "unknown"}:
            errors.append(f"Invalid sealed origin: {k}")
        if value and not label.get("label_source"):
            errors.append(f"Missing label source: {k}")
        if value and ((value == "F") != (product["observed_result"] == "FAIL")):
            errors.append(f"Label/original result contradiction: {k}")
        if label.get("suite_id") != product.get("suite_id") or label.get("suite_view") != "original":
            errors.append(f"Label/original suite identity mismatch: {k}")
        if label.get("source_transformed_from_f") == "1" and (
                value != "F" or label.get("source_label") != "FP" or
                label.get("source_origin") != "simulated" or label.get("origin") != "unknown"):
            errors.append(f"Converted FP not isolated/restored to original F: {k}")
        if value in {"FP", "TP"} and not (label.get("label_source_path") and label.get("label_source_sha256")):
            errors.append(f"Missing FP/TP source evidence: {k}")
    return errors


def audit(products_path: Path, tests_path: Path, labels_path: Path | None,
          split_paths: list[Path], *, check_paths: bool = False) -> dict:
    products = indexed(read_csv(products_path, KEY))
    tests = read_csv(tests_path, (*KEY, "test_id", "result"))
    labels = indexed(read_csv(labels_path, (*KEY, "label", "origin"))) if labels_path else {}
    errors = validate_sealed(labels, products) if labels_path else []
    # Check the actual public input boundary, including sealed columns and config types.
    load_inputs(products_path)
    per_tests, seen = defaultdict(list), set()
    missing_xml, empty_xml = Counter(), Counter()
    for row in tests:
        k = key(row)
        tk = (*k, row["test_id"])
        if k not in products or tk in seen or not row["test_id"]:
            errors.append(f"Unknown/duplicate/empty test: {tk}")
            continue
        seen.add(tk)
        per_tests[k].append(row)
        if row.get("suite_id") != products[k]["suite_id"]:
            errors.append(f"Test suite mismatch: {tk}")
        if row["result"] not in {"PASS", "FAIL", "UNKNOWN"}:
            errors.append(f"Invalid test result: {tk}")
        path = row.get("coverage_path", "")
        if not path or (check_paths and not Path(path).is_file()):
            missing_xml[k[0]] += 1
        if row.get("coverage_size_bytes") == "0":
            empty_xml[k[0]] += 1
    systems = {}
    for system in SYSTEMS:
        keys = [k for k in products if k[0] == system]
        lab = [labels[k] for k in keys if k in labels]
        original_labels = Counter(r["label"] or "UNKNOWN" for r in lab)
        source_labels = Counter(r.get("source_label") or "UNKNOWN" for r in lab)
        counts = {"products": len(keys), "versions": len({k[1] for k in keys}),
                  "observed_result": dict(Counter(products[k]["observed_result"] for k in keys)),
                  "labels": {v: original_labels[v] for v in ("F", "FP", "TP", "UNKNOWN")},
                  "source_labels": {v: source_labels[v] for v in ("F", "FP", "TP", "UNKNOWN")},
                  "origin": {v: sum(r["origin"] == v for r in lab) for v in ("natural", "simulated", "unknown")},
                  "source_origin": {v: sum(r.get("source_origin") == v for r in lab) for v in ("natural", "simulated", "unknown")},
                  "tests": sum(len(per_tests[k]) for k in keys),
                  "products_without_test_artifacts": sum(not per_tests[k] for k in keys),
                  "missing_test_source_dirs": sum(not Path(products[k]["test_suite_path"]).is_dir() for k in keys) if check_paths else None,
                  "missing_coverage_files": missing_xml[system], "zero_byte_coverage_files": empty_xml[system],
                  "missing_passed_aggregate": sum(products[k].get("aggregate_passed_exists") == "0" for k in keys),
                  "missing_failed_aggregate_for_FAIL": sum(products[k]["observed_result"] == "FAIL" and products[k].get("aggregate_failed_exists") == "0" for k in keys),
                  "paper": PAPER[system]}
        counts["difference_from_paper"] = {"versions": counts["versions"] - PAPER[system]["versions"],
            "original_suite_labels": {v: original_labels[v] - PAPER[system][v] for v in ("F", "FP", "TP")},
            "source_labels": {v: source_labels[v] - PAPER[system][v] for v in ("F", "FP", "TP")}}
        counts["difference_reasons"] = (["本地只有单缺陷扩展版本，未取得823子集清单", "标签为本地代理，未独立验证相关可执行缺陷", "原始套件将转换FP还原为F，不能与论文转换套件直接比较"] if keys else ["该系统数据缺失"])
        systems[system] = counts
    for k, product in products.items():
        results = [r["result"] for r in per_tests[k]]
        if (product["observed_result"] == "PASS" and "FAIL" in results) or (product["observed_result"] == "FAIL" and results and "FAIL" not in results and "UNKNOWN" not in results):
            errors.append(f"Product/test contradiction: {k}")
        if int(product["test_count"]) != len(results) or int(product["failed_test_count"]) != results.count("FAIL"):
            errors.append(f"Product/test count mismatch: {k}")
    splits_summary = {}
    for path in split_paths:
        splits = indexed(read_csv(path, (*KEY, "split")))
        holdout = path.stem.removeprefix("loso_") if path.stem.startswith("loso_") else None
        validate_split(products, splits, holdout=holdout)
        counts = test_assignments(tests, products, splits)
        splits_summary[path.name] = {
            "sha256": sha256(path), "tests": counts,
            "partitions": {s: {
                "products": sum(r["split"] == s for r in splits.values()),
                "versions": len({k[:2] for k, r in splits.items() if r["split"] == s}),
                "systems": dict(Counter(k[0] for k, r in splits.items() if r["split"] == s)),
                "labels": dict(Counter(labels[k]["label"] or "UNKNOWN" for k, r in splits.items() if r["split"] == s and k in labels))
            } for s in ("train", "val", "test")}}
    return {"valid": not errors, "errors": errors, "systems": systems, "splits": splits_summary,
            "counts": {"products": len(products), "tests": len(tests), "versions": len({k[:2] for k in products})},
            "path_existence_checked": check_paths, "xml_content_validated": False,
            "paper_source": "provided CLAP PDF Table 3 and PULP PDF; phase00 audit section2",
            "paper_total": {"versions": 823, "F": 10433, "FP": 14191, "TP": 22555},
            "paper_row_sums": {v: sum(row[v] for row in PAPER.values()) for v in ("versions", "F", "FP", "TP")},
            "paper_internal_discrepancy": "两篇给定论文表格的F分系统行之和为10324，Total行写10433，相差109；两种原文值分别保留，未擅自修正",
            "limits": ["测试条目是已存在XML的清单，未重新运行或证明全部测试完整", "自然来源依据匹配的原始通过报告和非转换标记；不证明真实缺陷", "缺失四系统、多缺陷版本和正式823清单", "转换实验视图禁用；原始套件失败测试只属于原始F观测"]}
