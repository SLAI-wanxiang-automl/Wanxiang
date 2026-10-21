"""Release checks for a weight-only backtest Solution update."""

from __future__ import annotations

import math

from energy_tsf.assets.active import read_active
from energy_tsf.contracts.artifact import CandidateRef, ReleaseDecision
from energy_tsf.memory.repository import MemoryRepository


def evaluate_backtest_release(
    repository: MemoryRepository,
    candidate: CandidateRef,
    *,
    parent_solution_id: str,
    completed_windows: int,
    expected_windows: int,
    before_rmse: float,
    after_rmse: float,
) -> ReleaseDecision:
    """Approve a weight-only update only when its evidence and metrics pass."""
    reasons: list[str] = []
    try:
        candidate = repository.load_candidate(candidate.path)
        updated = repository.load_solution(candidate.solution_id)
        parent = repository.load_solution(parent_solution_id)
    except (ValueError, FileNotFoundError) as exc:
        return ReleaseDecision(approved=False, reasons=(str(exc),))

    if updated.parent_solution_id != parent_solution_id:
        reasons.append("updated Solution does not identify the frozen parent")
    if updated.data_snapshot_id != parent.data_snapshot_id:
        reasons.append("weight update changed the training data snapshot")
    if updated.compatibility != parent.compatibility:
        reasons.append("weight update changed compatibility")
    if updated.members != parent.members:
        reasons.append("weight update changed model members")
    if updated.ensemble.member_trial_ids != parent.ensemble.member_trial_ids:
        reasons.append("weight update changed the ensemble member set")
    if updated.ensemble.method != "bayesian_update":
        reasons.append("updated ensemble is not marked as a Bayesian update")
    if not updated.update_evidence_ref:
        reasons.append("updated Solution lacks immutable backtest evidence")
    else:
        try:
            repository.resolve_reference(updated.update_evidence_ref)
        except (ValueError, FileNotFoundError) as exc:
            reasons.append(str(exc))
    if expected_windows < 2 or completed_windows != expected_windows:
        reasons.append("every one-day backtest window must complete before publication")
    if not all(math.isfinite(value) and value >= 0 for value in (before_rmse, after_rmse)):
        reasons.append("ensemble RMSE values must be finite and non-negative")
    elif after_rmse > before_rmse + 1e-12:
        reasons.append("updated ensemble RMSE is worse than the current ensemble")
    try:
        active = read_active(
            repository.root,
            parent.station_id,
            parent.compatibility.target,
            parent.compatibility.horizon,
        )
        if active.solution_id != parent_solution_id:
            reasons.append("Active solution changed after the backtest plan was frozen")
    except (ValueError, FileNotFoundError) as exc:
        reasons.append(str(exc))
    return ReleaseDecision(approved=not reasons, reasons=tuple(reasons))
