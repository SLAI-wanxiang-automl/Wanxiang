"""Cheap validation before executing a frozen forecast plan."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from energy_tsf.contracts.forecast import (
    ForecastPlan,
    ForecastRequest,
    SimilarSolutionCandidate,
)
from energy_tsf.contracts.modeling import PreflightIssue, PreflightReport
from energy_tsf.data.schemas import ForecastContext
from energy_tsf.memory.retrieval import validate_plan_candidates


def forecast_precheck(
    request: ForecastRequest,
    plan: ForecastPlan,
    context: ForecastContext,
    candidates: tuple[SimilarSolutionCandidate, ...],
    *,
    now: datetime | None = None,
) -> PreflightReport:
    """Validate plan identity, source compatibility, issue time, and deadline."""
    errors: list[PreflightIssue] = []
    warnings: list[PreflightIssue] = []
    try:
        validate_plan_candidates(
            plan,
            request,
            context.profile.input_fingerprint,
            candidates,
        )
    except ValueError as exc:
        errors.append(PreflightIssue(code="invalid_forecast_plan", message=str(exc)))
    if request.issue_time is not None and context.profile.end_time is not None:
        last_observation = datetime.fromisoformat(context.profile.end_time)
        if request.issue_time < last_observation:
            errors.append(
                PreflightIssue(
                    code="issue_time_before_observation",
                    message="issue_time cannot be earlier than the last observation",
                )
            )
    if request.deadline_seconds is not None:
        current = now or datetime.now(timezone.utc)
        if current > plan.created_at + timedelta(seconds=request.deadline_seconds):
            errors.append(
                PreflightIssue(
                    code="deadline_exceeded",
                    message="forecast deadline elapsed before inference started",
                )
            )
    if context.history.quality.imputed:
        warnings.append(
            PreflightIssue(
                code="history_imputed",
                message=f"forecast history contains {context.history.quality.imputed} imputed values",
            )
        )
    return PreflightReport(valid=not errors, errors=tuple(errors), warnings=tuple(warnings))
