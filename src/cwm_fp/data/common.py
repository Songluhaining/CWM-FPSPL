"""Shared deterministic file contracts (standard library only)."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping

KEY = ("system_id", "bug_version_id", "product_id")
SYSTEMS = ("BankAccountTP", "Elevator", "Email", "ExamDB", "GPL", "ZipMe")
ALIASES = {s: s for s in SYSTEMS} | {
    "Elevator-FH-JML": "Elevator", "Email-FH-JML": "Email",
    "ExamDB-FH-JML": "ExamDB", "GPL-Test": "GPL",
}
ROOT = Path(__file__).resolve().parents[3]
PROCESSED = ROOT / "data" / "processed"


def key(row: Mapping[str, str]) -> tuple[str, str, str]:
    value = tuple(row.get(k, "") for k in KEY)
    if not all(value):
        raise ValueError(f"Incomplete product key: {value}")
    return value


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path, required: Iterable[str] = ()) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)):
            raise ValueError(f"{path}: duplicate column names")
        if missing := set(required) - set(fields):
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        rows = list(reader)
    if any(None in r or None in r.values() for r in rows):
        raise ValueError(f"{path}: malformed CSV row width")
    return rows


def indexed(rows: Iterable[dict[str, str]]) -> dict[tuple[str, str, str], dict[str, str]]:
    out = {}
    for row in rows:
        k = key(row)
        if k in out:
            raise ValueError(f"Duplicate product key: {k}")
        out[k] = row
    return out


def write_csv(path: Path, rows: Iterable[Mapping], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                    encoding="utf-8")


def generated_dir(path: Path, sources: Iterable[Path] = ()) -> Path:
    """CLI outputs are confined to processed data and cannot overlap source trees."""
    path = path.resolve()
    if not path.is_relative_to(PROCESSED.resolve()):
        raise ValueError(f"Output must be under {PROCESSED}")
    for source in sources:
        source = source.resolve()
        if path.is_relative_to(source) or source.is_relative_to(path):
            raise ValueError(f"Output/source overlap: {path}, {source}")
    return path
