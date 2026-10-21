"""Explicit modeling workflow states and legal forward transitions."""

from __future__ import annotations

from enum import StrEnum


class ModelingState(StrEnum):
    CREATED = "created"
    DATA_PREPARED = "data_prepared"
    PLANNED = "planned"
    TRIALS_COMPLETED = "trials_completed"
    TOPK_FROZEN = "top_k_frozen"
    FINAL_EVALUATED = "final_evaluated"
    CANDIDATE = "candidate"
    RELEASED = "released"
    RELEASE_FAILED = "release_failed"
    INSUFFICIENT_CANDIDATES = "insufficient_candidates"


_TRANSITIONS = {
    ModelingState.CREATED: {ModelingState.DATA_PREPARED},
    ModelingState.DATA_PREPARED: {ModelingState.PLANNED},
    ModelingState.PLANNED: {ModelingState.TRIALS_COMPLETED},
    ModelingState.TRIALS_COMPLETED: {
        ModelingState.TOPK_FROZEN,
        ModelingState.INSUFFICIENT_CANDIDATES,
    },
    ModelingState.TOPK_FROZEN: {ModelingState.FINAL_EVALUATED},
    ModelingState.FINAL_EVALUATED: {ModelingState.CANDIDATE},
    ModelingState.CANDIDATE: {ModelingState.RELEASED, ModelingState.RELEASE_FAILED},
}


def transition(current: ModelingState, target: ModelingState) -> ModelingState:
    if target not in _TRANSITIONS.get(current, set()):
        raise ValueError(f"illegal modeling state transition: {current} -> {target}")
    return target


class ForecastState(StrEnum):
    CREATED = "created"
    INPUT_READY = "input_ready"
    PLAN_READY = "plan_ready"
    INFERRED = "inferred"
    DELIVERED = "delivered"
    REJECTED = "rejected"
    NO_COMPATIBLE_SOLUTION = "no_compatible_solution"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    FAILED = "failed"


_FORECAST_TRANSITIONS = {
    ForecastState.CREATED: {ForecastState.INPUT_READY, ForecastState.REJECTED},
    ForecastState.INPUT_READY: {
        ForecastState.PLAN_READY,
        ForecastState.NO_COMPATIBLE_SOLUTION,
        ForecastState.REJECTED,
    },
    ForecastState.PLAN_READY: {
        ForecastState.INFERRED,
        ForecastState.DEADLINE_EXCEEDED,
        ForecastState.FAILED,
    },
    ForecastState.INFERRED: {ForecastState.DELIVERED, ForecastState.FAILED},
}


def transition_forecast(current: ForecastState, target: ForecastState) -> ForecastState:
    if target not in _FORECAST_TRANSITIONS.get(current, set()):
        raise ValueError(f"illegal forecast state transition: {current} -> {target}")
    return target


class BacktestState(StrEnum):
    CREATED = "created"
    INPUT_READY = "input_ready"
    ASSET_PINNED = "asset_pinned"
    EVALUATED = "evaluated"
    WEIGHTS_UPDATED = "weights_updated"
    CANDIDATE = "candidate"
    RELEASED = "released"
    RELEASE_FAILED = "release_failed"


_BACKTEST_TRANSITIONS = {
    BacktestState.CREATED: {BacktestState.INPUT_READY},
    BacktestState.INPUT_READY: {BacktestState.ASSET_PINNED},
    BacktestState.ASSET_PINNED: {BacktestState.EVALUATED},
    BacktestState.EVALUATED: {BacktestState.WEIGHTS_UPDATED},
    BacktestState.WEIGHTS_UPDATED: {BacktestState.CANDIDATE, BacktestState.RELEASED},
    BacktestState.CANDIDATE: {BacktestState.RELEASED, BacktestState.RELEASE_FAILED},
}


def transition_backtest(current: BacktestState, target: BacktestState) -> BacktestState:
    if target not in _BACKTEST_TRANSITIONS.get(current, set()):
        raise ValueError(f"illegal backtest state transition: {current} -> {target}")
    return target
