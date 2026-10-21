"""Deterministic Active-solution retrieval for direct and similar forecasts."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Literal

from energy_tsf.assets.active import list_active
from energy_tsf.contracts.artifact import SolutionArtifact, SolutionMember
from energy_tsf.contracts.forecast import (
    ForecastAssetInspection,
    ForecastPlan,
    ForecastRequest,
    ForecastSourcePlan,
    SimilarityScore,
    SimilarSolutionCandidate,
)
from energy_tsf.contracts.station import StationSpec, is_per_unit, power_unit_factor
from energy_tsf.memory.repository import MemoryRepository
from energy_tsf.memory.cache import ExecutionCache
from energy_tsf.memory.stations import StationRepository
from energy_tsf.tools.fingerprints import json_fingerprint

_FOUNDATION = {"timesfm-2.5", "moirai-2", "chronos-2"}


def _required_history(member: SolutionMember) -> int:
    values = [1]
    if member.feature_spec is not None:
        values.append(member.feature_spec.required_history)
    context_length = member.params.get("context_length")
    if isinstance(context_length, int):
        values.append(context_length)
    return max(values)


def _member_issues(
    repository: MemoryRepository,
    solution: SolutionArtifact,
    history_rows: int,
) -> tuple[tuple[str, ...], dict[str, str], int]:
    available: list[str] = []
    unavailable: dict[str, str] = {}
    required = 1
    for member in solution.members:
        member_required = _required_history(member)
        required = max(required, member_required)
        if history_rows < member_required:
            unavailable[member.trial_id] = f"requires {member_required} history rows"
            continue
        try:
            repository.resolve_reference(member.artifact_ref)
            if member.provider in _FOUNDATION:
                checkpoint = Path(str(member.params.get("checkpoint_path", ""))).expanduser()
                if not checkpoint.exists():
                    raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
        except (FileNotFoundError, ValueError) as exc:
            unavailable[member.trial_id] = str(exc)
            continue
        available.append(member.trial_id)
    return tuple(available), unavailable, required


def _base_compatibility(
    request: ForecastRequest,
    solution: SolutionArtifact,
    available: tuple[str, ...],
    unavailable: dict[str, str],
) -> list[str]:
    reasons: list[str] = []
    station = request.station
    compatibility = solution.compatibility
    if compatibility.target != station.target:
        reasons.append("target mismatch")
    if compatibility.frequency != station.frequency:
        reasons.append("frequency mismatch")
    if compatibility.horizon != request.horizon:
        reasons.append("horizon mismatch")
    if compatibility.prediction_mode != request.prediction_mode:
        reasons.append("prediction mode mismatch")
    if compatibility.covariates != request.covariates:
        reasons.append("covariate schema mismatch")
    requires_timestamp = any(
        member.feature_spec is not None and member.feature_spec.calendar
        for member in solution.members
    ) and compatibility.timestamp_column is not None
    if requires_timestamp and station.timestamp_column is None:
        reasons.append("source features require real timestamps")
    if len(available) < solution.ensemble.min_valid_members:
        reasons.append(
            f"only {len(available)} usable members; needs {solution.ensemble.min_valid_members}"
        )
    if unavailable and not solution.ensemble.renormalize_on_failure:
        reasons.append("solution forbids ensemble-member fallback")
    return reasons


def _transfer_compatibility(
    target: StationSpec, source: StationSpec
) -> tuple[list[str], Literal["none", "capacity_ratio"]]:
    reasons: list[str] = []
    if source.energy_type != target.energy_type:
        reasons.append("energy type mismatch")
    if source.target != target.target:
        reasons.append("station target semantics mismatch")
    if source.frequency != target.frequency:
        reasons.append("station frequency mismatch")

    target_power = power_unit_factor(target.target_unit)
    source_power = power_unit_factor(source.target_unit)
    target_pu = is_per_unit(target.target_unit)
    source_pu = is_per_unit(source.target_unit)
    if not (target_power or target_pu) or not (source_power or source_pu):
        reasons.append("target units are not convertible")
    if target_power and target.capacity is None:
        reasons.append("target capacity is required for absolute-power transfer")
    if source_power and source.capacity is None:
        reasons.append("source capacity is required for absolute-power transfer")
    scale_method: Literal["none", "capacity_ratio"] = (
        "none" if target_pu and source_pu else "capacity_ratio"
    )
    return reasons, scale_method


def _distance_km(first: StationSpec, second: StationSpec) -> float | None:
    if first.latitude is None or second.latitude is None:
        return None
    if first.longitude is None or second.longitude is None:
        return None
    lat1, lon1, lat2, lon2 = map(
        math.radians,
        (first.latitude, first.longitude, second.latitude, second.longitude),
    )
    a = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(
        (lon2 - lon1) / 2
    ) ** 2
    return 6371.0088 * 2 * math.asin(math.sqrt(a))


def _capacity_ratio(first: StationSpec, second: StationSpec) -> float | None:
    if first.capacity is None or second.capacity is None:
        return None
    first_factor = power_unit_factor(first.capacity_unit or "")
    second_factor = power_unit_factor(second.capacity_unit or "")
    if first_factor is None or second_factor is None:
        return None
    return (second.capacity * second_factor) / (first.capacity * first_factor)


def _source_month_score(source: StationSpec, reference_time: datetime | None) -> float:
    if reference_time is None:
        return 0.5
    raw = source.tags.get("data_months")
    if not raw:
        return 0.5
    try:
        months = {int(value) for value in raw.split(",")}
    except ValueError:
        return 0.5
    return 1.0 if reference_time.month in months else 0.0


def _quality(solution: SolutionArtifact) -> tuple[float, float | None]:
    values = []
    for member in solution.members:
        metric = member.validation_metrics.get("wape")
        if metric is not None:
            values.append(float(metric))
    if not values:
        return 0.5, None
    value = median(values)
    return 1.0 / (1.0 + max(value, 0.0)), value


def _recency(solution: SolutionArtifact, reference_time: datetime | None) -> float:
    if reference_time is None:
        return 0.5
    created = datetime.fromisoformat(solution.created_at.replace("Z", "+00:00"))
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    age_days = max((reference_time.astimezone(timezone.utc) - created.astimezone(timezone.utc)).days, 0)
    return max(0.0, 1.0 - age_days / 730.0)


def _score_candidate(
    target: StationSpec,
    source: StationSpec,
    solution: SolutionArtifact,
    reference_time: datetime | None,
) -> tuple[SimilarityScore, float | None, float | None, float | None]:
    distance = _distance_km(target, source)
    ratio = _capacity_ratio(target, source)
    technology = (
        0.5
        if target.technology is None or source.technology is None
        else float(target.technology == source.technology)
    )
    location = (
        max(0.0, 1.0 - distance / 1000.0)
        if distance is not None
        else (0.6 if target.timezone == source.timezone else 0.3)
    )
    capacity = min(ratio, 1.0 / ratio) if ratio is not None else 0.5
    quality, quality_metric = _quality(solution)
    score = SimilarityScore(
        technology=technology,
        location=location,
        capacity=capacity,
        season=_source_month_score(source, reference_time),
        quality=quality,
        recency=_recency(solution, reference_time),
    )
    return score, distance, ratio, quality_metric


class ForecastAssetRetriever:
    """Find direct or similar Active solutions for a forecast request.

    Compatibility is a hard filter evaluated before similarity ranking; scores
    cannot make a source with mismatched requirements eligible.
    """

    def __init__(self, memory_root: str | Path) -> None:
        """Connect retrieval to the asset repository and station index."""
        self.repository = MemoryRepository(memory_root)
        self.stations = StationRepository(memory_root)
        self.cache = ExecutionCache(memory_root)

    def direct(self, request: ForecastRequest, history_rows: int) -> SimilarSolutionCandidate | None:
        """Inspect the requested station's current Active solution, if present."""
        try:
            solution = self.cache.active_solution(
                self.repository,
                request.station.station_id,
                request.station.target,
                request.horizon,
            )
        except FileNotFoundError:
            return None
        available, unavailable, required = _member_issues(self.repository, solution, history_rows)
        reasons = _base_compatibility(request, solution, available, unavailable)
        return SimilarSolutionCandidate(
            station_id=solution.station_id,
            solution_id=solution.solution_id,
            compatible=not reasons,
            required_history=required,
            available_members=available,
            rejection_reasons=tuple(reasons),
            score=SimilarityScore(
                technology=1,
                location=1,
                capacity=1,
                season=1,
                quality=1,
                recency=1,
            ),
            total_score=1 if not reasons else 0,
            recommended_weight=1 if not reasons else 0,
            scale_method="none",
            timezone_match=True,
            technology_match=True,
            source_created_at=solution.created_at,
        )

    def similar(
        self,
        request: ForecastRequest,
        history_rows: int,
        *,
        reference_time: datetime | None = None,
        limit: int = 10,
    ) -> ForecastAssetInspection:
        """Rank compatible cross-station Active solutions and explain rejections."""
        compatible: list[SimilarSolutionCandidate] = []
        rejected: list[SimilarSolutionCandidate] = []
        for active in list_active(
            self.repository.root,
            target=request.station.target,
            horizon=request.horizon,
        ):
            if active.station_id == request.station.station_id:
                continue
            solution = self.repository.load_solution(active.solution_id)
            available, unavailable, required = _member_issues(self.repository, solution, history_rows)
            reasons = _base_compatibility(request, solution, available, unavailable)
            try:
                source = self.stations.load(active.station_id)
            except FileNotFoundError:
                source = None
                reasons.append("source station profile is missing")
            scale_method: Literal["none", "capacity_ratio"] = "none"
            score = SimilarityScore()
            distance = ratio = quality_metric = None
            if source is not None:
                transfer_reasons, scale_method = _transfer_compatibility(request.station, source)
                reasons.extend(transfer_reasons)
                score, distance, ratio, quality_metric = _score_candidate(
                    request.station,
                    source,
                    solution,
                    reference_time,
                )
            candidate = SimilarSolutionCandidate(
                station_id=active.station_id,
                solution_id=active.solution_id,
                compatible=not reasons,
                required_history=required,
                available_members=available,
                rejection_reasons=tuple(dict.fromkeys(reasons)),
                score=score,
                total_score=score.weighted_total if not reasons else 0,
                recommended_weight=0,
                scale_method=scale_method,
                distance_km=distance,
                capacity_ratio=ratio,
                timezone_match=source is not None and source.timezone == request.station.timezone,
                technology_match=(
                    None
                    if source is None
                    or source.technology is None
                    or request.station.technology is None
                    else source.technology == request.station.technology
                ),
                quality_metric=quality_metric,
                source_created_at=solution.created_at,
            )
            (compatible if candidate.compatible else rejected).append(candidate)

        compatible.sort(key=lambda item: (-item.total_score, item.station_id, item.solution_id))
        compatible = compatible[:limit]
        total = sum(item.total_score for item in compatible)
        if compatible:
            weights = (
                [item.total_score / total for item in compatible]
                if total > 0
                else [1 / len(compatible)] * len(compatible)
            )
            compatible = [
                item.model_copy(update={"recommended_weight": weight})
                for item, weight in zip(compatible, weights)
            ]
        rejected.sort(key=lambda item: (item.station_id, item.solution_id))
        mode: Literal["similar", "none"] = "similar" if compatible else "none"
        return ForecastAssetInspection(
            mode=mode,
            candidates=tuple(compatible),
            rejected=tuple(rejected),
        )

    def inspect(
        self,
        request: ForecastRequest,
        history_rows: int,
        *,
        reference_time: datetime | None = None,
    ) -> ForecastAssetInspection:
        """Prefer compatible same-station Active, then allowed similar sources."""
        direct = self.direct(request, history_rows)
        if direct is not None and direct.compatible:
            return ForecastAssetInspection(mode="direct", direct=direct)
        if not request.allow_similar or (direct is not None and not request.allow_similar_on_direct_failure):
            return ForecastAssetInspection(
                mode="none",
                rejected=(() if direct is None else (direct,)),
            )
        result = self.similar(request, history_rows, reference_time=reference_time)
        rejected = (() if direct is None else (direct,)) + result.rejected
        return result.model_copy(update={"rejected": rejected})


def forecast_plan_fingerprint(value: ForecastPlan | dict) -> str:
    """Return the stable content hash for a frozen forecast plan."""
    payload = value.model_dump(mode="json") if isinstance(value, ForecastPlan) else dict(value)
    payload.pop("plan_id", None)
    return json_fingerprint(payload)


def build_forecast_plan(
    *,
    request: ForecastRequest,
    input_fingerprint: str,
    mode: str,
    sources: tuple[ForecastSourcePlan, ...],
    created_at: datetime | None = None,
    renormalize_on_source_failure: bool = True,
) -> ForecastPlan:
    """Create an immutable plan pinned to explicit source Solution IDs."""
    payload = {
        "plan_id": "0" * 64,
        "request_id": request.request_id,
        "target_station_id": request.station.station_id,
        "mode": mode,
        "input_fingerprint": input_fingerprint,
        "sources": sources,
        "renormalize_on_source_failure": renormalize_on_source_failure,
        "created_at": created_at or datetime.now(timezone.utc),
    }
    provisional = ForecastPlan.model_validate(payload)
    return provisional.model_copy(update={"plan_id": forecast_plan_fingerprint(provisional)})


def validate_plan_candidates(
    plan: ForecastPlan,
    request: ForecastRequest,
    input_fingerprint: str,
    candidates: tuple[SimilarSolutionCandidate, ...],
) -> None:
    """Reject stale plans or plans selecting absent or incompatible candidates."""
    if plan.request_id != request.request_id or plan.target_station_id != request.station.station_id:
        raise ValueError("forecast plan does not belong to this request")
    if plan.input_fingerprint != input_fingerprint:
        raise ValueError("forecast plan input fingerprint is stale")
    allowed = {
        (candidate.station_id, candidate.solution_id): candidate
        for candidate in candidates
        if candidate.compatible
    }
    selected = {(source.station_id, source.solution_id) for source in plan.sources}
    if not selected <= set(allowed):
        raise ValueError("forecast plan selects an unavailable or incompatible solution")
    for source in plan.sources:
        candidate = allowed[(source.station_id, source.solution_id)]
        if source.scale_method != candidate.scale_method:
            raise ValueError("forecast plan changes the candidate scale method")
    if forecast_plan_fingerprint(plan) != plan.plan_id:
        raise ValueError("forecast plan fingerprint does not match its contents")
