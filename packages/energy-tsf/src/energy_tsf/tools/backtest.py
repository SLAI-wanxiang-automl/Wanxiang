"""Time-consistent rolling-window construction for backtests."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from energy_tsf.contracts.artifact import SolutionArtifact, SolutionMember
from energy_tsf.contracts.backtest import BacktestPlan
from energy_tsf.data.cleaning import clean_forecast_context
from energy_tsf.data.schemas import BacktestInput, BacktestWindowData, PreparedSeries, RawSeries
from energy_tsf.tools.fingerprints import json_fingerprint
from energy_tsf.tools.metrics import point_metrics


def member_required_history(member: SolutionMember) -> int:
    """Return the largest history length required by a solution member."""
    required = 1
    if member.feature_spec is not None:
        required = max(required, member.feature_spec.required_history)
    context_length = member.params.get("context_length")
    if isinstance(context_length, int):
        required = max(required, context_length)
    return required


def solution_required_history(solution: SolutionArtifact) -> int:
    """Return the history length needed to run every member in a solution."""
    return max(member_required_history(member) for member in solution.members)


def backtest_plan_fingerprint(value: BacktestPlan | dict) -> str:
    """Hash plan content while excluding its self-referential plan ID."""
    payload = value.model_dump(mode="json") if isinstance(value, BacktestPlan) else dict(value)
    payload.pop("plan_id", None)
    return json_fingerprint(payload)


def build_backtest_plan(
    request_id: str,
    station_id: str,
    solution: SolutionArtifact,
    backtest: BacktestInput,
    *,
    created_at: datetime | None = None,
) -> BacktestPlan:
    """Freeze the windows and solution membership for a backtest run."""
    payload = {
        "plan_id": "0" * 64,
        "request_id": request_id,
        "station_id": station_id,
        "parent_solution_id": solution.solution_id,
        "data_snapshot_id": solution.data_snapshot_id,
        "input_fingerprint": backtest.profile.input_fingerprint,
        "member_trial_ids": tuple(member.trial_id for member in solution.members),
        "windows": backtest.windows,
        "created_at": created_at or datetime.now(timezone.utc),
    }
    plan = BacktestPlan.model_validate(payload)
    return plan.model_copy(update={"plan_id": backtest_plan_fingerprint(plan)})


def build_window_data(
    backtest: BacktestInput,
    snapshot: PreparedSeries,
    solution: SolutionArtifact,
    *,
    timezone_name: str,
) -> tuple[BacktestWindowData, ...]:
    """Create contexts using only records strictly before each window origin."""

    if snapshot.timestamp_column is None:
        raise ValueError("backtest requires a timestamped training snapshot")
    if snapshot.target != backtest.target or snapshot.frequency != backtest.frequency:
        raise ValueError("training snapshot is incompatible with backtest truth")
    timestamp = backtest.timestamp_column
    columns = [snapshot.timestamp_column, snapshot.target, *backtest.covariates.columns]
    missing = sorted(set(columns) - set(snapshot.frame.columns))
    if missing:
        raise ValueError(f"training snapshot lacks backtest covariates: {', '.join(missing)}")
    historical = snapshot.frame[columns].rename(
        columns={snapshot.timestamp_column: timestamp}
    )
    historical[timestamp] = pd.to_datetime(historical[timestamp], errors="raise")
    if historical[timestamp].dt.tz is None:
        historical[timestamp] = historical[timestamp].dt.tz_localize(
            timezone_name, ambiguous="raise", nonexistent="raise"
        )
    else:
        historical[timestamp] = historical[timestamp].dt.tz_convert(timezone_name)

    required = solution_required_history(solution)
    output: list[BacktestWindowData] = []
    for spec in backtest.windows:
        origin = pd.Timestamp(spec.origin)
        snapshot_past = historical.loc[historical[timestamp] < origin]
        input_past = backtest.frame.loc[
            backtest.frame[timestamp] < origin,
            [timestamp, backtest.target, *backtest.covariates.columns],
        ]
        combined = pd.concat([snapshot_past, input_past], ignore_index=True)
        combined = combined.drop_duplicates(timestamp, keep="last").sort_values(timestamp, kind="stable")
        combined = combined.reset_index(drop=True)
        if len(combined) < required:
            raise ValueError(
                f"window {spec.window_id} has {len(combined)} history rows; requires {required}"
            )
        raw = RawSeries(
            frame=combined,
            target=backtest.target,
            timestamp_column=timestamp,
            frequency=backtest.frequency,
            covariates=backtest.covariates,
        )
        current = backtest.frame.loc[
            (backtest.frame[timestamp] >= origin)
            & (backtest.frame[timestamp] < pd.Timestamp(spec.end_time))
        ]
        if len(current) != spec.horizon_steps:
            raise ValueError(f"window {spec.window_id} does not contain a complete horizon")
        future_covariates = (
            current[[timestamp, *backtest.covariates.known_future]].copy()
            if backtest.covariates.known_future
            else None
        )
        context = clean_forecast_context(
            raw,
            timezone=timezone_name,
            future_covariates=future_covariates,
        )
        actual_index = pd.DatetimeIndex(current[timestamp])
        if not context.future_index.equals(actual_index):
            raise ValueError(f"window {spec.window_id} is not contiguous with its historical context")
        mask = backtest.observed_mask.loc[current.index].reset_index(drop=True)
        truth = current[backtest.target].reset_index(drop=True)
        output.append(
            BacktestWindowData(
                spec=spec,
                context=context,
                truth=truth,
                observed_mask=mask,
            )
        )
    return tuple(output)


def backtest_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Calculate per-window and overall metrics from the long prediction table."""

    rows: list[dict] = []
    keys = ["trial_id", "provider", "kind"]
    for (window_id, *identity), group in predictions.groupby(["window_id", *keys], sort=False):
        mask = group["observed"].to_numpy(dtype=bool)
        values = point_metrics(group["y_true"], group["y_pred"], include_mask=mask)
        rows.append(
            {
                "scope": "window",
                "window_id": window_id,
                **dict(zip(keys, identity)),
                "observations": int(mask.sum()),
                **values,
            }
        )
    for identity, group in predictions.groupby(keys, sort=False):
        mask = group["observed"].to_numpy(dtype=bool)
        values = point_metrics(group["y_true"], group["y_pred"], include_mask=mask)
        rows.append(
            {
                "scope": "overall",
                "window_id": "all",
                **dict(zip(keys, identity)),
                "observations": int(mask.sum()),
                **values,
            }
        )
    return pd.DataFrame(rows)


def recombine_member_predictions(
    predictions: pd.DataFrame,
    member_trial_ids: tuple[str, ...],
    weights: tuple[float, ...],
) -> pd.DataFrame:
    """Recombine cached member predictions without running providers again."""

    if len(member_trial_ids) != len(weights):
        raise ValueError("members and weights must align")
    output: list[pd.DataFrame] = []
    member_rows = predictions.loc[predictions["kind"] == "member"]
    for window_id, group in member_rows.groupby("window_id", sort=False):
        pivot = group.pivot(index="horizon_step", columns="trial_id", values="y_pred")
        if set(pivot.columns) != set(member_trial_ids):
            raise ValueError(f"window {window_id} lacks a complete member prediction set")
        pivot = pivot.loc[:, list(member_trial_ids)]
        values = pivot.to_numpy(dtype=float) @ np.asarray(weights, dtype=float)
        template = group.loc[group["trial_id"] == member_trial_ids[0]].sort_values("horizon_step").copy()
        template["trial_id"] = "ensemble_after"
        template["provider"] = "ensemble"
        template["kind"] = "ensemble_after"
        template["y_pred"] = values
        output.append(template)
    return pd.concat(output, ignore_index=True)
