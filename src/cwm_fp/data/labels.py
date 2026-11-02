"""Sealed source labels and ORIGINAL-suite targets; never imported by input loading."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from .benchmark_adapter import load_sources, report_path
from .common import KEY, generated_dir, indexed, key, read_csv, sha256, write_csv, write_json

FIELDS = (*KEY, "label", "origin", "label_source", "label_source_path", "label_source_sha256",
          "source_label", "source_transformed_from_f", "source_origin", "decision",
          "label_provenance", "suite_id", "suite_view", "quality")


def import_labels(products_path: Path, sources_path: Path) -> tuple[list[dict], dict]:
    products = indexed(read_csv(products_path, (*KEY, "observed_result", "suite_id", "suite_view")))
    sources = {s.system_id: s for s in load_sources(sources_path)}
    rows, source_audit = [], []
    groups = sorted({k[:2] for k in products})
    for system, version in groups:
        source = sources[system]
        group_products = {k: v for k, v in products.items() if k[:2] == (system, version)}
        source_rows, source_path, label_hash = {}, "", ""
        if source.label_root is not None:
            path = source.label_root / version / "variant_labels.csv"
            if path.is_file():
                if source.label_source == "unknown":
                    raise ValueError(f"Label source must be declared: {path}")
                companion = report_path(source.label_root / version)
                raw_report = report_path(source.root / version)
                if sha256(companion) != sha256(raw_report):
                    raise ValueError(f"Source label report differs from original: {companion}")
                expected_hashes = {p["source_report_sha256"] for p in group_products.values()}
                if expected_hashes != {sha256(raw_report)}:
                    raise ValueError(f"Report changed since manifest creation: {raw_report}")
                valid_ids = {r["Product\\Feature"].strip() for r in read_csv(raw_report, ("Product\\Feature",))}
                for row in read_csv(path, ("VARIANT", "LABEL", "FP transformed from F")):
                    pid = row["VARIANT"].strip()
                    if pid not in valid_ids or pid in source_rows:
                        raise ValueError(f"Unknown/duplicate source-label product: {path}: {pid}")
                    if row["LABEL"] not in {"F", "FP", "TP"} or row["FP transformed from F"] not in {"", "1"}:
                        raise ValueError(f"Invalid source label/transform flag: {path}: {pid}")
                    source_rows[pid] = row
                source_path, label_hash = path.as_posix(), sha256(path)
        source_audit.append({"system_id": system, "bug_version_id": version,
                             "path": source_path, "sha256": label_hash,
                             "source_rows": len(source_rows), "selected_products": len(group_products),
                             "source_report_match": bool(source_path)})
        for k, product in sorted(group_products.items()):
            if product["suite_view"] != "original":
                raise ValueError("Only original-suite targets are implemented; simulated view is unavailable")
            src = source_rows.get(k[2])
            source_label = src["LABEL"] if src else ""
            transformed = src["FP transformed from F"] if src else ""
            observed = product["observed_result"]
            if observed not in {"PASS", "FAIL"}:
                raise ValueError(f"Invalid original result: {k}")
            if transformed and (source_label != "FP" or observed != "FAIL"):
                raise ValueError(f"Invalid converted FP provenance: {k}")
            if source_label == "TP" and observed != "PASS" or source_label == "F" and observed != "FAIL":
                raise ValueError(f"Source label/original result contradiction: {k}")
            if source_label == "FP" and observed == "FAIL" and not transformed:
                raise ValueError(f"FAIL -> FP without explicit conversion record: {k}")
            effective = "F" if observed == "FAIL" else source_label
            decision = ("restore_original_failure" if transformed else
                        "retain_source_proxy" if source_label else
                        "observed_original_failure" if observed == "FAIL" else "missing_label")
            # Non-conversion + PASS supports a natural-suite designation in this
            # particular audited source; it does not verify the fault itself.
            origin = "natural" if (source.label_source == "local_pulp_fault_location_proxy" and
                source_label == "FP" and observed == "PASS" and not transformed) else "unknown"
            rows.append(dict(zip(KEY, k)) | {
                "label": effective, "origin": origin,
                "label_source": source.label_source if src else "original_report" if effective == "F" else "unknown",
                "label_source_path": source_path if src else "", "label_source_sha256": label_hash if src else "",
                "source_label": source_label, "source_transformed_from_f": transformed,
                "source_origin": "simulated" if transformed else origin,
                "decision": decision, "suite_id": product["suite_id"], "suite_view": "original",
                "label_provenance": "original_FAIL_overrides_converted_FP" if transformed else
                    "matched_report_plus_explicit_nonconversion" if origin == "natural" else
                    "matched_local_proxy" if src else "no_source_label",
                "quality": "observed_failure" if effective == "F" else "unverified_fault_proxy" if effective else "unknown"})
    audit = {"products_sha256": sha256(products_path), "sources": source_audit,
             "labels": dict(Counter(r["label"] or "UNKNOWN" for r in rows)),
             "source_labels": dict(Counter(r["source_label"] or "UNKNOWN" for r in rows)),
             "origin": dict(Counter(r["origin"] for r in rows)),
             "source_origin": dict(Counter(r["source_origin"] for r in rows)),
             "restored_original_failures": sum(r["decision"] == "restore_original_failure" for r in rows),
             "simulated_view_available": False,
             "natural_definition": "FP + original PASS + explicit blank conversion flag in matched audited PULP source; not a claim of real-world faults",
             "limits": "Fault-location proxies are not independently verified FP/TP truth; no mutant logs read"}
    return rows, audit


def load_targets(labels_path: Path, products_path: Path, splits_path: Path, *,
                 partition: str, purpose: str) -> dict[tuple[str, str, str], str]:
    """Explicit target access; callers receive labels only, never provenance.

    This API boundary is not an OS permission sandbox. Evaluation callers must
    not send returned labels back into preprocessing or model input.
    """
    from .splits import validate_split
    if purpose not in {"train", "evaluate"} or partition not in {"train", "val", "test"}:
        raise ValueError("Declare train/evaluate purpose and an existing partition")
    if purpose == "train" and partition != "train":
        raise ValueError("Training cannot access validation/test targets")
    products = indexed(read_csv(products_path, KEY))
    splits = indexed(read_csv(splits_path, (*KEY, "split")))
    validate_split(products, splits)
    selected = {k for k, row in splits.items() if row["split"] == partition}
    labels = indexed(read_csv(labels_path, (*KEY, "label", "suite_id")))
    if set(labels) != selected:
        raise ValueError("Use the sealed label file for this partition only (not the all-label audit file)")
    out = {}
    for k in selected:
        row, product = labels[k], products[k]
        if row["suite_id"] != product["suite_id"]:
            raise ValueError(f"Label suite mismatch: {k}")
        label = row["label"]
        if label not in {"", "F", "FP", "TP"}:
            raise ValueError(f"Invalid target: {k}")
        if (label == "F" and product["observed_result"] != "FAIL") or (label in {"FP", "TP"} and product["observed_result"] != "PASS"):
            raise ValueError(f"Target/original result contradiction: {k}")
        if label:
            out[k] = label
    return out


def export_partition_labels(rows: list[dict], products_path: Path, split_paths: list[Path], out: Path) -> dict:
    """Offline auditor creates separate files so training need not open test labels."""
    from .splits import validate_split
    products = indexed(read_csv(products_path, KEY))
    labels = indexed(rows)
    if set(products) != set(labels):
        raise ValueError("Labels and product identity sets differ")
    result = {}
    for split_path in split_paths:
        name = "grouped" if split_path.name == "splits.csv" else split_path.stem
        split = indexed(read_csv(split_path, (*KEY, "split")))
        validate_split(products, split, holdout=name.removeprefix("loso_") if name.startswith("loso_") else None)
        result[name] = {"splits_sha256": sha256(split_path), "partitions": {}}
        for partition in ("train", "val", "test"):
            selected = [labels[k] for k in sorted(labels) if split[k]["split"] == partition]
            path = out / name / f"{partition}.csv"
            write_csv(path, selected, FIELDS)
            result[name]["partitions"][partition] = {"rows": len(selected), "sha256": sha256(path)}
    return result


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--products", type=Path, required=True)
    p.add_argument("--sources", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--splits", type=Path, nargs="*", default=[])
    a = p.parse_args(argv)
    try:
        sources = load_sources(a.sources)
        out = generated_dir(a.out, [s.root for s in sources] + [s.label_root for s in sources if s.label_root])
        from .common import PROCESSED
        if not out.is_relative_to((PROCESSED / "labels").resolve()):
            raise ValueError("Sealed labels must live under data/processed/labels")
        rows, audit = import_labels(a.products, a.sources)
        write_csv(out / "labels.csv", rows, FIELDS)
        audit["labels_sha256"] = sha256(out / "labels.csv")
        audit["partition_files"] = export_partition_labels(rows, a.products, a.splits, out)
        write_json(out / "label_audit.json", audit)
        print(json.dumps({k: audit[k] for k in ("labels", "source_labels", "origin", "source_origin", "restored_original_failures")}, ensure_ascii=False))
    except (ValueError, OSError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
