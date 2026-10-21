"""Read-only inference for frozen Solution artifacts."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from energy_tsf.contracts.artifact import SolutionArtifact
from energy_tsf.contracts.forecast import ForecastPlan, ForecastSourcePlan
from energy_tsf.contracts.result import ForecastTable
from energy_tsf.contracts.station import StationSpec
from energy_tsf.data.schemas import ForecastContext
from energy_tsf.memory.repository import MemoryRepository
from energy_tsf.memory.cache import ExecutionCache
from energy_tsf.memory.stations import StationRepository
from energy_tsf.providers import modeling_registry
from energy_tsf.providers._common import forecast_table
from energy_tsf.providers.base import TrainableProvider
from energy_tsf.runtime.policy import FORECAST_POLICY
from energy_tsf.tools.constraints import constrain_forecast
from energy_tsf.tools.ensemble import combine_available_predictions, combine_forecast_sources
from energy_tsf.tools.transfer import adapt_history_to_source, adapt_predictions_to_target


@dataclass(frozen=True)
class SourceForecast:
    station_id: str
    solution_id: str
    predictions: np.ndarray
    member_predictions: dict[str, np.ndarray]
    member_details: dict[str, dict]
    failed_members: dict[str, str]
    degraded: bool


@dataclass(frozen=True)
class TrackForecast:
    table: ForecastTable
    sources: tuple[SourceForecast, ...]
    failed_sources: dict[str, str]
    degraded: bool


class ForecastTrack:
    """Execute a frozen plan without Active lookup, fitting, or publication."""

    policy = FORECAST_POLICY

    def __init__(self, memory_root: str | Path, *, registry=None) -> None:
        self.repository = MemoryRepository(memory_root)
        self.cache = ExecutionCache(memory_root)
        self.stations = StationRepository(memory_root)
        self.registry = registry or modeling_registry()

    def _member_future_index(
        self,
        solution: SolutionArtifact,
        context: ForecastContext,
    ) -> pd.Index:
        if solution.compatibility.timestamp_column is None:
            return pd.RangeIndex(1, context.horizon_steps + 1, name="horizon_step")
        if not isinstance(context.future_index, pd.DatetimeIndex):
            raise ValueError("solution requires timestamped future inputs")
        return context.future_index

    def execute_source(
        self,
        source_plan: ForecastSourcePlan,
        context: ForecastContext,
        target_station: StationSpec,
    ) -> SourceForecast:
        """Run a frozen source Solution and apply its member-level fallback rules."""
        solution = self.cache.get_by_solution_id(source_plan.solution_id)
        if solution is None:
            solution = self.repository.load_solution(source_plan.solution_id)
        if solution.station_id != source_plan.station_id:
            raise ValueError("source plan station does not match the frozen Solution")
        source_station = (
            target_station
            if source_plan.station_id == target_station.station_id
            else self.stations.load(source_plan.station_id)
        )
        history = context.history
        if source_plan.scale_method == "capacity_ratio":
            history = adapt_history_to_source(history, target_station, source_station)
        elif source_plan.scale_method != "none":
            raise ValueError(f"unsupported scale method: {source_plan.scale_method}")

        future_index = self._member_future_index(solution, context)
        if solution.compatibility.timestamp_column is None:
            history = replace(history, timestamp_column=None)
        member_predictions: dict[str, np.ndarray] = {}
        failures: dict[str, str] = {}
        for member in solution.members:
            try:
                provider = self.registry.create(member.provider)
                model = None
                if isinstance(provider, TrainableProvider):
                    model = provider.load(self.repository.resolve_reference(member.artifact_ref))
                table = provider.predict(
                    model,
                    history,
                    (len(history.frame),),
                    context.horizon_steps,
                    params=member.params,
                    feature_spec=member.feature_spec,
                    future_index=future_index,
                    **(
                        {
                            "covariate_spec": member.covariate_spec,
                            "future_covariates": context.future_covariates,
                        }
                        if member.covariate_spec.multivariate
                        else {}
                    ),
                )
                table.require_coverage((len(history.frame),), context.horizon_steps)
                minimum = solution.constraints.get("minimum", 0.0)
                maximum = solution.constraints.get("maximum")
                table = constrain_forecast(table, minimum=minimum, maximum=maximum)
                member_predictions[member.trial_id] = table.predictions
            except Exception as exc:  # member fallback is part of the published ensemble contract
                failures[member.trial_id] = f"{type(exc).__name__}: {exc}"

        combined = combine_available_predictions(solution.ensemble, member_predictions)
        if source_plan.scale_method == "capacity_ratio":
            combined = adapt_predictions_to_target(combined, source_station, target_station)
        return SourceForecast(
            station_id=source_plan.station_id,
            solution_id=source_plan.solution_id,
            predictions=combined,
            member_predictions=member_predictions,
            member_details={
                member.trial_id: {
                    "provider": member.provider,
                    "params": member.params,
                    "weight": solution.ensemble.weights[
                        solution.ensemble.member_trial_ids.index(member.trial_id)
                    ],
                }
                for member in solution.members
            },
            failed_members=failures,
            degraded=bool(failures),
        )

    def run(
        self,
        plan: ForecastPlan,
        context: ForecastContext,
        target_station: StationSpec,
    ) -> TrackForecast:
        """Execute each frozen source, combine results, and report degradation."""
        completed: list[SourceForecast] = []
        failures: dict[str, str] = {}
        for source in plan.sources:
            try:
                completed.append(self.execute_source(source, context, target_station))
            except Exception as exc:
                failures[source.solution_id] = f"{type(exc).__name__}: {exc}"
        predictions = combine_forecast_sources(
            plan,
            {source.solution_id: source.predictions for source in completed},
        )
        table = forecast_table(
            "forecast-ensemble",
            context.history,
            (context.origin,),
            [predictions],
            future_index=context.future_index,
        )
        return TrackForecast(
            table=table,
            sources=tuple(completed),
            failed_sources=failures,
            degraded=bool(failures) or any(source.degraded for source in completed),
        )
