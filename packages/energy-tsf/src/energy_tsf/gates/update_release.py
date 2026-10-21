"""Release gate for a retrained station Solution."""

from __future__ import annotations

import math

from energy_tsf.assets.active import read_active
from energy_tsf.contracts.artifact import CandidateRef, ReleaseDecision
from energy_tsf.memory.repository import MemoryRepository


def evaluate_update_release(
    repository: MemoryRepository,
    candidate: CandidateRef,
    *,
    parent_solution_id: str,
    parent_rmse: float,
    updated_rmse: float,
    smoke_ok: bool,
) -> ReleaseDecision:
    """Gate an update on pool coverage, smoke checks, metrics, and parent state."""
    reasons: list[str] = []
    try:
        candidate = repository.load_candidate(candidate.path)
        updated = repository.load_solution(candidate.solution_id)
        parent = repository.load_solution(parent_solution_id)
    except (ValueError, FileNotFoundError) as exc:
        return ReleaseDecision(approved=False, reasons=(str(exc),))
    if updated.parent_solution_id != parent_solution_id:
        reasons.append("updated Solution does not identify the frozen parent")
    if updated.compatibility != parent.compatibility:
        reasons.append("update changed station compatibility")
    if not updated.search_pool_id:
        reasons.append("updated Solution lacks a next-generation search pool")
    else:
        try:
            pool = repository.load_search_pool(updated.search_pool_id)
            if pool.data_snapshot_id != updated.data_snapshot_id:
                reasons.append("next search pool and Solution use different data snapshots")
            if pool.window_count != 7:
                reasons.append("next search pool must cover all seven update windows")
        except (ValueError, FileNotFoundError) as exc:
            reasons.append(str(exc))
    if not updated.members or len(updated.members) != len(updated.ensemble.member_trial_ids):
        reasons.append("updated Solution must contain an aligned non-empty Top-K")
    for label, reference in (
        ("validation report", updated.validation_report_ref),
        ("validation predictions", updated.validation_predictions_ref),
        ("update evidence", updated.update_evidence_ref),
    ):
        if not reference:
            reasons.append(f"updated Solution lacks {label}")
            continue
        try:
            repository.resolve_reference(reference)
        except (ValueError, FileNotFoundError) as exc:
            reasons.append(str(exc))
    for member in updated.members:
        try:
            repository.resolve_reference(member.artifact_ref)
        except (ValueError, FileNotFoundError) as exc:
            reasons.append(str(exc))
    if not smoke_ok:
        reasons.append("updated Solution failed smoke inference")
    if not all(math.isfinite(value) and value >= 0 for value in (parent_rmse, updated_rmse)):
        reasons.append("parent and updated validation RMSE must be finite and non-negative")
    elif updated_rmse > parent_rmse + 1e-12:
        reasons.append("updated ensemble RMSE is worse than the parent on the same seven days")
    try:
        active = read_active(
            repository.root,
            parent.station_id,
            parent.compatibility.target,
            parent.compatibility.horizon,
        )
        if active.solution_id != parent_solution_id:
            reasons.append("Active solution changed after the update plan was frozen")
    except (ValueError, FileNotFoundError) as exc:
        reasons.append(str(exc))
    return ReleaseDecision(approved=not reasons, reasons=tuple(reasons))
