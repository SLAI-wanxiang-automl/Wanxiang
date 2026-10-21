"""Stable SolutionArtifact identity construction."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from energy_tsf.contracts.artifact import SolutionArtifact


def solution_fingerprint(value: SolutionArtifact | dict[str, Any]) -> str:
    """Return the canonical Solution ID while excluding its ID field."""
    payload = value.model_dump(mode="json") if isinstance(value, SolutionArtifact) else dict(value)
    payload.pop("solution_id", None)
    # New lineage fields are omitted when absent so pre-update Solution IDs remain valid.
    for optional in ("search_pool_id", "parent_solution_id", "update_evidence_ref"):
        if payload.get(optional) is None:
            payload.pop(optional, None)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        default=lambda item: item.model_dump(mode="json"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_solution(**values: Any) -> SolutionArtifact:
    """Validate Solution fields and assign their content-derived identity."""
    payload = dict(values)
    payload["solution_id"] = "0" * 64
    normalized = SolutionArtifact.model_validate(payload)
    return normalized.model_copy(update={"solution_id": solution_fingerprint(normalized)})
