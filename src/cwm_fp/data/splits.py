"""Deterministic version grouping and six-system leave-one-system-out plans."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from .common import KEY, SYSTEMS, generated_dir, indexed, key, read_csv, sha256, write_csv, write_json


def stable_rank(value, seed):
    # Preserve the starter tool's grouping order for the same IDs/seed.
    return hashlib.sha256(f"{seed}|{value[0]}|{value[1]}".encode()).hexdigest()


def split_groups(groups, seed, train_frac, val_frac):
    if not 0 < train_frac < 1 or not 0 < val_frac < 1 or train_frac + val_frac >= 1:
        raise ValueError("Fractions must be positive and leave a test partition")
    ordered = sorted(groups, key=lambda g: (stable_rank(g, seed), g))
    n = len(ordered)
    if n < 3:
        raise ValueError(f"Need at least 3 buggy-version groups; have {n}")
    n_train = max(1, min(n - 2, round(n * train_frac)))
    n_val = max(1, min(n - n_train - 1, round(n * val_frac)))
    return {g: "train" if i < n_train else "val" if i < n_train + n_val else "test"
            for i, g in enumerate(ordered)}


def validate_split(products: dict, splits: dict, *, holdout: str | None = None) -> None:
    if set(products) != set(splits):
        raise ValueError("Split/product key sets differ")
    groups = defaultdict(set)
    for k, row in splits.items():
        if row["split"] not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split: {k}")
        groups[k[:2]].add(row["split"])
        if holdout is not None and ((k[0] == holdout) != (row["split"] == "test")):
            raise ValueError(f"LOSO holdout violated: {k}")
    if any(len(v) != 1 for v in groups.values()):
        raise ValueError("Version overlap across partitions")
    if {r["split"] for r in splits.values()} != {"train", "val", "test"}:
        raise ValueError("All three partitions must be nonempty")


def make_split(products_path, strategy="grouped", seed=42, holdout=None, train_frac=.70, val_frac=.15):
    products = indexed(read_csv(Path(products_path), KEY))
    if not products:
        raise ValueError("No products")
    groups = {k[:2] for k in products}
    if strategy == "grouped":
        mapping = split_groups(groups, seed, train_frac, val_frac)
    elif strategy == "loso":
        if holdout not in {k[0] for k in products}:
            raise ValueError(f"Holdout system not found: {holdout}")
        if not 0 < train_frac < 1 or not 0 < val_frac < 1 or train_frac + val_frac >= 1:
            raise ValueError("Invalid fractions")
        ordered = sorted((g for g in groups if g[0] != holdout), key=lambda g: (stable_rank(g, seed), g))
        if len(ordered) < 2:
            raise ValueError("Need >=2 non-holdout versions for train/val")
        n_val = max(1, min(len(ordered) - 1, round(len(ordered) * val_frac / (train_frac + val_frac))))
        mapping = {g: "val" if i < n_val else "train" for i, g in enumerate(ordered)}
        mapping.update({g: "test" for g in groups if g[0] == holdout})
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")
    rows = [dict(zip(KEY, k)) | {"split": mapping[k[:2]]} for k in sorted(products)]
    validate_split(products, indexed(rows), holdout=holdout if strategy == "loso" else None)
    return rows


def test_assignments(tests: list[dict], products: dict, splits: dict) -> dict:
    validate_split(products, splits)
    seen, counts = set(), Counter()
    for row in tests:
        k = key(row)
        tk = (*k, row["test_id"])
        if not row["test_id"] or tk in seen or k not in splits:
            raise ValueError(f"Duplicate/unknown/empty test identity: {tk}")
        seen.add(tk)
        if row.get("suite_id") and row["suite_id"] != products[k].get("suite_id"):
            raise ValueError(f"Test suite mismatch: {tk}")
        # A test has no independent partition: inherit exactly through product key.
        counts[splits[k]["split"]] += 1
    return dict(counts)


def require_training_keys(keys, splits_path: Path, operation: str) -> None:
    """Guard future fold-local fitting; does not fit or compute any features."""
    if operation not in {"normalization", "vocabulary", "suspiciousness", "pretraining"}:
        raise ValueError("Unknown training-only operation")
    splits = indexed(read_csv(splits_path, (*KEY, "split")))
    validate_split(splits, splits)
    selected = list(keys)
    if not selected or any(k not in splits or splits[k]["split"] != "train" for k in selected):
        raise ValueError(f"{operation}: all contributors must belong to TRAIN")


def build_all(products_path: Path, tests_path: Path, out: Path, seed: int, *, source_train_fraction=None) -> dict:
    products = indexed(read_csv(products_path, KEY))
    tests = read_csv(tests_path, (*KEY, "test_id"))
    present = {k[0] for k in products}
    result = {"seed": seed, "products_sha256": sha256(products_path), "tests_sha256": sha256(tests_path),
              "group_key": list(KEY[:2]), "test_split_rule": "inherit product split by full key",
              "fractions": {"train": .70, "val": .15, "test": .15},
              "complete_six_systems": set(SYSTEMS).issubset(present), "folds": {}}
    plans = [("grouped", None), *(("loso_" + s, s) for s in SYSTEMS)]
    for name, holdout in plans:
        file = out / ("splits.csv" if holdout is None else f"{name}.csv")
        if holdout is not None and holdout not in present:
            if file.exists():
                raise ValueError(f"Stale split for unavailable system; use a fresh output directory: {file}")
            result["folds"][name] = {"status": "unavailable", "reason": "holdout_system_missing", "holdout": holdout}
            continue
        # Insufficient group data must be explicit, never a fallback product split.
        try:
            fractions = {}
            if holdout and source_train_fraction is not None:
                if not 0 < source_train_fraction < 1:
                    raise ValueError("Invalid source train fraction")
                fractions = {'train_frac': .85 * source_train_fraction,
                             'val_frac': .85 * (1-source_train_fraction)}
            rows = make_split(products_path, "loso" if holdout else "grouped", seed, holdout, **fractions)
        except ValueError as exc:
            if file.exists():
                raise ValueError(f"Stale split for unavailable groups; use a fresh output directory: {file}") from exc
            result["folds"][name] = {"status": "unavailable", "reason": str(exc), "holdout": holdout}
            continue
        assignments = indexed(rows)
        test_counts = test_assignments(tests, products, assignments)
        write_csv(file, rows, (*KEY, "split"))
        groups = {(k[0], k[1], r["split"]) for k, r in assignments.items()}
        result["folds"][name] = {"status": "available", "holdout": holdout,
            "file": file.name, "sha256": sha256(file),
            "products": dict(Counter(r["split"] for r in rows)),
            "versions": dict(Counter(g[2] for g in groups)), "tests": test_counts,
            "split_id": f"{name}:seed={seed}:{sha256(file)}", "version_overlap": 0}
    result['source_train_fraction'] = source_train_fraction
    write_json(out / "split_audit.json", result)
    return result


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--products", type=Path, required=True)
    p.add_argument("--tests", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20261030)
    a = p.parse_args(argv)
    try:
        result = build_all(a.products, a.tests, generated_dir(a.out), a.seed)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, OSError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
