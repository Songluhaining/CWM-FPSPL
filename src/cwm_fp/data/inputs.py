"""Allowlisted input boundary. Does not read labels or raw coverage/source files."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .common import KEY, digest, indexed, read_csv
from .splits import validate_split

FORBIDDEN = {"label", "origin", "label_provenance", "buggy", "oracle", "oracle_bug_line",
             "removed_failed_tests", "removed_tests", "source_label", "source_origin", "split",
             "source_transformed_from_f", "label_source", "label_source_path", "label_source_sha256",
             "true_fault_location", "true_fault_locations", "mutant", "mutant_log", "patch",
             "bug_count", "dataset_family", "eligible_for_paper_subset", "decision", "quality",
             "sample_id", "suite_id", "suite_view", "source_variant_path", "feature_config_path",
             "test_suite_path", "source_report_sha256", "test_count", "failed_test_count", *KEY}


@dataclass(frozen=True)
class ModelInput:
    # sample_id is a join handle, not a numerical/categorical feature.
    sample_id: str
    features: dict[str, bool]
    observed_result: str


def load_inputs(products_path: Path, *, splits_path: Path | None = None,
                partition: str | None = None) -> list[ModelInput]:
    products = indexed(read_csv(products_path, (*KEY, "sample_id", "features_json", "observed_result", "suite_view")))
    if (splits_path is None) != (partition is None):
        raise ValueError("Provide both split file and partition, or neither for inference")
    if partition is not None and partition not in {"train", "val", "test"}:
        raise ValueError("Unknown partition")
    splits = indexed(read_csv(splits_path, (*KEY, "split"))) if splits_path else None
    if splits is not None:
        validate_split(products, splits)
    out = []
    for k, row in sorted(products.items()):
        if splits is not None and splits[k]["split"] != partition:
            continue
        if row["suite_view"] != "original":
            raise ValueError("Simulated suite loading is disabled until hidden tests are isolated")
        if row.get("label") or row.get("label_provenance") or row.get("origin", "unknown") != "unknown":
            raise ValueError("Labels/origin must be sealed outside products.csv")
        if row["sample_id"] != digest(k) or row["observed_result"] not in {"PASS", "FAIL"}:
            raise ValueError("Invalid sample ID or original result")
        features = json.loads(row["features_json"])
        if not isinstance(features, dict) or any(name.lower() in FORBIDDEN or type(value) is not bool
                                                  for name, value in features.items()):
            raise ValueError("Input config must contain only named boolean features, without forbidden fields")
        out.append(ModelInput(row["sample_id"], features, row["observed_result"]))
    return out
