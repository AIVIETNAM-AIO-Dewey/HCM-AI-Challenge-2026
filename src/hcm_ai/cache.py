"""Small persistent JSON cache used for quota-limited Gemini calls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class JsonCache:
    """Content-addressed JSON cache safe for rerun-heavy Colab notebooks."""

    def __init__(self, root: str | Path, *, namespace: str) -> None:
        self.root = Path(root) / namespace
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(payload: Any) -> str:
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def get(self, payload: Any) -> Any | None:
        path = self.root / f"{self.key(payload)}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def set(self, payload: Any, value: Any) -> Path:
        path = self.root / f"{self.key(payload)}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        temporary.replace(path)
        return path
