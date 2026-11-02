"""Process-role I/O restrictions; not a substitute for OS account separation."""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

from cwm_fp.data.common import ROOT


@lru_cache(maxsize=32768)
def resolved(path: str) -> Path:
    # Source trees are immutable during a worker run. Cache canonicalization,
    # not permission decisions; the role/allowlist is checked on every access.
    return Path(path).resolve()


def forbidden_read(path: Path, role: str, allowed_labels: set[Path]) -> bool:
    path = resolved(str(path))
    name = path.name.lower()
    sealed = ROOT / "data/processed/labels"
    metrics = ROOT / "outputs/metrics"
    if name.endswith(".mutant.log") or name in {"variant_labels.csv", "operational_labels.csv"}:
        return True
    if name.startswith("attributes") and path.suffix.lower() == ".csv":
        return True  # Existing whole-version normalized, label-bearing tables.
    if path.is_relative_to(metrics) and role != "evaluator":
        return True
    if path.is_relative_to(sealed) and role != "evaluator" and path not in allowed_labels:
        return True
    return False


def install_io_guard(role: str, allowed_labels=()) -> set[Path]:
    allowed = {Path(p).resolve() for p in allowed_labels}
    def hook(event, args):
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = resolved(os.fsdecode(args[0]))
        mode, flags = args[1], args[2]
        writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
            isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)))
        if writing:
            if path.is_relative_to(ROOT / "third_party") or path.is_relative_to(ROOT / "CLAP") or path.is_relative_to(ROOT / "PULP"):
                raise PermissionError(f"Immutable third-party source: {path}")
            if path.is_relative_to(ROOT / "outputs/metrics") and role != "evaluator":
                raise PermissionError("Only the evaluator may write labeled metrics")
        elif forbidden_read(path, role, allowed):
            raise PermissionError(f"{role} may not read sealed/oracle/legacy attributes: {path}")
    sys.addaudithook(hook)
    return allowed
