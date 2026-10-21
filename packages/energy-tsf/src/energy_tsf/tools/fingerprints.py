"""Stable code and JSON fingerprints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def json_fingerprint(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def source_fingerprint(root: str | Path) -> str:
    digest = hashlib.sha256()
    base = Path(root)
    for path in sorted(base.rglob("*.py")):
        relative = path.relative_to(base).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()
