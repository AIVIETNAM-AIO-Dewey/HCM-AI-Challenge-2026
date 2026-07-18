"""Canonical, service-independent result exports.

The competition's final submission schema is not public yet.  These helpers
therefore produce stable JSONL and CSV artifacts that can later be adapted by
a thin submission formatter without changing retrieval code.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


def to_primitive(value: Any) -> Any:
    """Convert Pydantic/dataclass-like values to JSON-serialisable values."""

    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "__dataclass_fields__"):
        from dataclasses import asdict

        return asdict(value)
    if isinstance(value, Mapping):
        return {str(key): to_primitive(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_primitive(item) for item in value]
    return value


def write_jsonl(path: str | Path, records: Iterable[Any]) -> Path:
    """Write canonical records atomically enough for notebook batch jobs."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(to_primitive(record), ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    temporary.replace(destination)
    return destination


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read non-empty JSONL records, rejecting malformed input naturally."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def flatten_result(record: Any) -> dict[str, Any]:
    """Flatten one canonical result into a CSV-friendly row.

    Nested score/evidence/event collections remain JSON strings so their
    provenance is preserved instead of being silently discarded.
    """

    data = to_primitive(record)
    if not isinstance(data, Mapping):
        return {"value": json.dumps(data, ensure_ascii=False)}

    row: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, (Mapping, list, tuple)):
            row[str(key)] = json.dumps(value, ensure_ascii=False, sort_keys=True)
        else:
            row[str(key)] = value
    return row


def write_csv(path: str | Path, records: Iterable[Any]) -> Path:
    """Export result records without imposing a contest-specific column set."""

    rows = [flatten_result(record) for record in records]
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(destination)
    return destination
