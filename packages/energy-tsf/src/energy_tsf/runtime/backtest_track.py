"""Read-only rolling evaluation of every member in one frozen Active Solution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from energy_tsf.contracts.artifact import SolutionArtifact
from energy_tsf.contracts.forecast import ForecastSourcePlan
from energy_tsf.contracts.station import StationSpec
from energy_tsf.data.schemas import BacktestWindowData
from energy_tsf.runtime.forecast_track import ForecastTrack
from energy_tsf.runtime.policy import BACKTEST_POLICY


@dataclass(frozen=True)
class BacktestTrackResult:
    predictions: pd.DataFrame
    failures: dict[str, str]


class BacktestTrack:
    """Use published model objects without fitting or persisting models."""

    policy = BACKTEST_POLICY

    def __init__(self, memory_root: str | Path, *, registry=None) -> None:
        self.forecast_track = ForecastTrack(memory_root, registry=registry)

    def run(
        self,
        solution: SolutionArtifact,
        station: StationSpec,
        windows: tuple[BacktestWindowData, ...],
    ) -> BacktestTrackResult:
        """Evaluate every member over frozen windows without fitting models."""
        source_plan = ForecastSourcePlan(
            station_id=solution.station_id,
            solution_id=solution.solution_id,
            weight=1.0,
            scale_method="none",
            selection_reasons=("frozen direct Active backtest",),
        )
        member_by_id = {member.trial_id: member for member in solution.members}
        rows: list[dict] = []
        failures: dict[str, str] = {}
        for window in windows:
            try:
                result = self.forecast_track.execute_source(source_plan, window.context, station)
            except Exception as exc:
                failures[window.spec.window_id] = f"{type(exc).__name__}: {exc}"
                continue
            failures.update(
                {
                    f"{window.spec.window_id}:{trial_id}": reason
                    for trial_id, reason in result.failed_members.items()
                }
            )
            times = pd.DatetimeIndex(window.context.future_index)
            truth = window.truth.to_numpy(dtype=float)
            observed = window.observed_mask.to_numpy(dtype=bool)
            for trial_id, values in result.member_predictions.items():
                member = member_by_id[trial_id]
                rows.extend(
                    {
                        "window_id": window.spec.window_id,
                        "origin": window.spec.origin.isoformat(),
                        "valid_time": times[step - 1].isoformat(),
                        "horizon_step": step,
                        "trial_id": trial_id,
                        "provider": member.provider,
                        "kind": "member",
                        "y_true": float(truth[step - 1]) if observed[step - 1] else float("nan"),
                        "y_pred": float(value),
                        "observed": bool(observed[step - 1]),
                    }
                    for step, value in enumerate(values, 1)
                )
            rows.extend(
                {
                    "window_id": window.spec.window_id,
                    "origin": window.spec.origin.isoformat(),
                    "valid_time": times[step - 1].isoformat(),
                    "horizon_step": step,
                    "trial_id": "ensemble_before",
                    "provider": "ensemble",
                    "kind": "ensemble_before",
                    "y_true": float(truth[step - 1]) if observed[step - 1] else float("nan"),
                    "y_pred": float(value),
                    "observed": bool(observed[step - 1]),
                }
                for step, value in enumerate(result.predictions, 1)
            )
        if not rows:
            raise ValueError("no backtest window completed")
        return BacktestTrackResult(predictions=pd.DataFrame(rows), failures=failures)
