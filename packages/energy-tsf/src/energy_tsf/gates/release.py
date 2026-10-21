"""Release gate for solution integrity, reloadability, and smoke inference."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from energy_tsf.contracts.artifact import CandidateRef, ReleaseDecision, SolutionMember
from energy_tsf.memory.repository import MemoryRepository

SmokeCheck = Callable[[SolutionMember], np.ndarray]
_FOUNDATION = {"timesfm-2.5", "moirai-2", "chronos-2"}
_FEATURED = {"ridge", "lightgbm", "xgboost"}


def evaluate_release(
    repository: MemoryRepository,
    candidate: CandidateRef,
    smoke_check: SmokeCheck,
) -> ReleaseDecision:
    """Check artifact integrity, member metadata, reloadability, and smoke output."""
    reasons: list[str] = []
    try:
        candidate = repository.load_candidate(candidate.path)
        solution = repository.load_solution(candidate.solution_id)
    except (ValueError, FileNotFoundError) as exc:
        return ReleaseDecision(approved=False, reasons=(str(exc),))

    if not solution.members:
        reasons.append("a publishable solution must contain at least one member")
    if len(solution.members) != len(solution.ensemble.member_trial_ids):
        reasons.append("solution members and ensemble membership do not align")
    for reference_name in (
        "validation_report_ref",
        "test_report_ref",
        "validation_predictions_ref",
        "test_predictions_ref",
    ):
        try:
            repository.resolve_reference(getattr(solution, reference_name))
        except (FileNotFoundError, ValueError) as exc:
            reasons.append(str(exc))

    for member in solution.members:
        try:
            repository.resolve_reference(member.artifact_ref)
        except (FileNotFoundError, ValueError) as exc:
            reasons.append(str(exc))
            continue
        if member.test_prediction_ref is not None:
            try:
                repository.resolve_reference(member.test_prediction_ref)
            except (FileNotFoundError, ValueError) as exc:
                reasons.append(str(exc))
        if member.provider in _FEATURED and member.feature_spec is None:
            reasons.append(f"{member.provider} member lacks FeatureSpec")
        if member.provider in _FOUNDATION:
            if not member.params.get("revision") or not member.params.get("repo_id"):
                reasons.append(f"{member.provider} member lacks pinned source metadata")
            checkpoint = member.params.get("checkpoint_path")
            if not checkpoint or not Path(checkpoint).exists():
                reasons.append(f"{member.provider} checkpoint reference is not resolvable")
        try:
            output = np.asarray(smoke_check(member), dtype=float)
            if output.size == 0 or not np.isfinite(output).all():
                reasons.append(f"{member.provider} smoke prediction is empty or non-finite")
        except (ValueError, TypeError, RuntimeError, OSError, ImportError) as exc:
            reasons.append(f"{member.provider} smoke check failed: {exc}")
    return ReleaseDecision(approved=not reasons, reasons=tuple(reasons))
