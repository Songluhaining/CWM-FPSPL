#!/usr/bin/env python3
"""Generate version-grouped or LOSO train/val/test splits from canonical products.csv."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path


# Re-export the tested canonical implementation; preserve the starter CLI/API.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cwm_fp.data.splits import stable_rank, split_groups, make_split


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--products", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--strategy", choices=["grouped", "loso"], default="grouped")
    p.add_argument("--holdout-system")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args(argv)
    try:
        rows = make_split(a.products, a.strategy, a.seed, a.holdout_system)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["system_id", "bug_version_id", "product_id", "split"])
            w.writeheader()
            w.writerows(rows)
        counts = {s: sum(r["split"] == s for r in rows) for s in ("train", "val", "test")}
        print(json.dumps({"path": str(a.out), "product_counts": counts, "seed": a.seed, "strategy": a.strategy}, ensure_ascii=False, indent=2))
    except (ValueError, FileNotFoundError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
