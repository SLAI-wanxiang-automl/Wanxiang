"""Rebind one frozen SearchPool to a new update snapshot."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import cast

from energy_tsf.contracts.search_pool import SearchPoolArtifact
from energy_tsf.contracts.update import UpdatePlan
from energy_tsf.providers.registry import ProviderRegistry
from energy_tsf.providers.foundation.base import FoundationProvider


def _canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def update_plan_fingerprint(value: UpdatePlan | dict) -> str:
    """Return a stable fingerprint that excludes the plan's own ID field."""
    payload = value.model_dump(mode="json") if isinstance(value, UpdatePlan) else dict(value)
    payload.pop("plan_id", None)
    payload.pop("created_at", None)
    return hashlib.sha256(_canonical(payload)).hexdigest()


def build_update_plan(
    *,
    request_id: str,
    station_id: str,
    parent_solution_id: str,
    pool: SearchPoolArtifact,
    data_snapshot_id: str,
    split_id: str,
    input_fingerprint: str,
    train_end: int,
    validation_origins: tuple[int, ...],
    seed: int,
    registry: ProviderRegistry,
    required_candidates: int = 3,
) -> UpdatePlan:
    """Rebind only frozen SearchPool trials to the new update snapshot."""
    if pool.station_id != station_id:
        raise ValueError("search pool belongs to another station")
    availability = {}
    for name in dict.fromkeys(entry.spec.provider for entry in pool.entries):
        try:
            availability[name] = registry.create(name).available()
        except KeyError as exc:
            availability[name] = (False, str(exc))
    trials = []
    unavailable: dict[str, str] = {}
    for entry in pool.entries:
        old = entry.spec
        available, reason = availability.get(old.provider, (False, "availability not reported"))
        if not available:
            unavailable[old.trial_id] = reason or "provider unavailable"
            continue
        history = old.context_length or int(old.params.get("context_length", 0) or 0)
        if old.feature_spec is not None:
            history = max(history, old.feature_spec.required_history)
        if history >= train_end:
            unavailable[old.trial_id] = "candidate history requirement exceeds update train data"
            continue
        if old.provider in {"timesfm-2.5", "moirai-2", "chronos-2"}:
            provider = cast(FoundationProvider, registry.create(old.provider))
            ready, checkpoint_reason = provider.checkpoint_available(old.params.get("checkpoint_path"))
            if not ready:
                unavailable[old.trial_id] = checkpoint_reason or "foundation checkpoint unavailable"
                continue
        identity = {
            "provider": old.provider,
            "params": old.params,
            "feature_spec": old.feature_spec.model_dump(mode="json") if old.feature_spec else None,
            "input_mode": old.input_mode,
            "covariate_spec": old.covariate_spec.model_dump(mode="json"),
            "feature_selection": old.feature_selection.model_dump(mode="json"),
            "context_length": old.context_length,
            "seed": seed,
            "data_snapshot_id": data_snapshot_id,
            "split_id": split_id,
        }
        trial_id = hashlib.sha256(_canonical(identity)).hexdigest()[:16]
        trials.append(
            old.model_copy(
                update={
                    "trial_id": trial_id,
                    "seed": seed,
                    "data_snapshot_id": data_snapshot_id,
                    "split_id": split_id,
                }
            )
        )
    if len(trials) < required_candidates:
        raise ValueError(
            f"fewer than {required_candidates} executable candidates remain in the Active search pool"
        )
    payload = UpdatePlan(
        plan_id="0" * 64,
        request_id=request_id,
        station_id=station_id,
        parent_solution_id=parent_solution_id,
        source_pool_id=pool.pool_id,
        data_snapshot_id=data_snapshot_id,
        split_id=split_id,
        input_fingerprint=input_fingerprint,
        train_end=train_end,
        validation_origins=validation_origins,
        trials=tuple(trials),
        unavailable=unavailable,
        created_at=datetime.now(timezone.utc),
    )
    return payload.model_copy(update={"plan_id": update_plan_fingerprint(payload)})
