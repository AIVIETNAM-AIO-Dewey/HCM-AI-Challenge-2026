"""Deterministic artifact fingerprints and resumable JSON/JSONL storage."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from collections.abc import Iterable, Iterator, Mapping
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ArtifactRef(BaseModel):
    """Location and completion metadata for one content-addressed artifact."""

    model_config = ConfigDict(extra="forbid")

    kind: str
    fingerprint: str
    path: str
    record_count: int = Field(ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Enum):
        return _jsonable(value.value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    return value


def canonical_json(value: Any) -> str:
    """Serialize a JSON-compatible object in a stable form suitable for hashing."""

    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def fingerprint(value: Any) -> str:
    """Return a SHA-256 fingerprint of a canonical JSON value."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def fingerprint_inputs(*values: Any) -> str:
    """Fingerprint an ordered collection of data/configuration inputs."""

    return fingerprint({"inputs": list(values)})


def fingerprint_files(paths: Iterable[str | Path]) -> str:
    """Hash file contents deterministically without depending on absolute paths."""

    digest = hashlib.sha256()
    for path in sorted((Path(item) for item in paths), key=lambda item: item.as_posix()):
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _safe_kind(kind: str) -> Path:
    candidate = Path(kind)
    if not kind or candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        raise ValueError("artifact kind must be a non-empty relative path")
    return candidate


def _record_to_mapping(record: BaseModel | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(record, BaseModel):
        return record.model_dump(mode="json")
    if isinstance(record, Mapping):
        return _jsonable(record)
    raise TypeError("JSONL records must be Pydantic models or mappings")


class ArtifactStore:
    """Content-addressed local storage suitable for Google Drive-backed paths.

    Existing completed artifacts are returned untouched, so notebook restarts do
    not re-encode completed data.  Writes use a same-directory temporary file
    followed by an atomic replace.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def artifact_dir(self, kind: str, artifact_fingerprint: str) -> Path:
        if len(artifact_fingerprint) < 12 or any(char not in "0123456789abcdef" for char in artifact_fingerprint):
            raise ValueError("artifact fingerprint must be a hexadecimal digest")
        return self.root / _safe_kind(kind) / artifact_fingerprint

    def jsonl_path(
        self,
        kind: str,
        artifact_fingerprint: str,
        name: str = "records",
    ) -> Path:
        if not name or Path(name).name != name:
            raise ValueError("artifact file name must be a single path component")
        return self.artifact_dir(kind, artifact_fingerprint) / f"{name}.jsonl"

    def manifest_path(self, kind: str, artifact_fingerprint: str) -> Path:
        return self.artifact_dir(kind, artifact_fingerprint) / "manifest.json"

    def is_complete(self, kind: str, artifact_fingerprint: str, name: str = "records") -> bool:
        manifest_path = self.manifest_path(kind, artifact_fingerprint)
        data_path = self.jsonl_path(kind, artifact_fingerprint, name)
        if not manifest_path.is_file() or not data_path.is_file():
            return False
        try:
            manifest = self.read_json(manifest_path)
        except (OSError, ValueError, json.JSONDecodeError):
            return False
        return (
            manifest.get("fingerprint") == artifact_fingerprint
            and manifest.get("status") == "complete"
            and manifest.get("file") == data_path.name
        )

    def get_ref(self, kind: str, artifact_fingerprint: str, name: str = "records") -> ArtifactRef:
        manifest = self.read_json(self.manifest_path(kind, artifact_fingerprint))
        if not self.is_complete(kind, artifact_fingerprint, name):
            raise FileNotFoundError(f"incomplete artifact {kind}/{artifact_fingerprint}")
        return ArtifactRef(
            kind=kind,
            fingerprint=artifact_fingerprint,
            path=str(self.jsonl_path(kind, artifact_fingerprint, name)),
            record_count=int(manifest.get("record_count", 0)),
            metadata=dict(manifest.get("metadata", {})),
        )

    def write_jsonl(
        self,
        kind: str,
        artifact_fingerprint: str,
        records: Iterable[BaseModel | Mapping[str, Any]],
        *,
        name: str = "records",
        metadata: Mapping[str, Any] | None = None,
        overwrite: bool = False,
    ) -> ArtifactRef:
        """Write records and a completion manifest, or reuse an existing result."""

        if self.is_complete(kind, artifact_fingerprint, name) and not overwrite:
            return self.get_ref(kind, artifact_fingerprint, name)

        target = self.jsonl_path(kind, artifact_fingerprint, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and not overwrite:
            raise FileExistsError(f"artifact data exists but is not complete: {target}")

        count = 0
        temporary_path: Path | None = None
        try:
            with NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                dir=target.parent,
                prefix=f".{name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                for record in records:
                    handle.write(canonical_json(_record_to_mapping(record)))
                    handle.write("\n")
                    count += 1
            temporary_path.replace(target)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

        manifest = {
            "kind": kind,
            "fingerprint": artifact_fingerprint,
            "status": "complete",
            "file": target.name,
            "record_count": count,
            "metadata": _jsonable(dict(metadata or {})),
        }
        self.write_json(self.manifest_path(kind, artifact_fingerprint), manifest)
        return ArtifactRef(
            kind=kind,
            fingerprint=artifact_fingerprint,
            path=str(target),
            record_count=count,
            metadata=dict(metadata or {}),
        )

    def iter_jsonl(self, path: str | Path) -> Iterator[dict[str, Any]]:
        """Yield JSON objects from a JSONL file with useful line-number errors."""

        source = Path(path)
        with source.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL at {source}:{line_number}") from exc
                if not isinstance(payload, dict):
                    raise ValueError(f"JSONL records must be objects at {source}:{line_number}")
                yield payload

    @staticmethod
    def read_json(path: str | Path) -> dict[str, Any]:
        source = Path(path)
        with source.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"JSON artifact {source} must contain an object")
        return payload

    @staticmethod
    def write_json(path: str | Path, value: Mapping[str, Any]) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                dir=target.parent,
                prefix=f".{target.stem}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                handle.write(canonical_json(value))
                handle.write("\n")
            temporary_path.replace(target)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

    def write_checkpoint(
        self,
        name: str,
        checkpoint_fingerprint: str,
        value: Mapping[str, Any],
    ) -> Path:
        """Persist a small resumable checkpoint independently from JSONL records."""

        if not name or Path(name).name != name:
            raise ValueError("checkpoint name must be a single path component")
        target = self.artifact_dir("checkpoints", checkpoint_fingerprint) / f"{name}.json"
        self.write_json(target, value)
        return target


__all__ = [
    "ArtifactRef",
    "ArtifactStore",
    "canonical_json",
    "fingerprint",
    "fingerprint_files",
    "fingerprint_inputs",
]
