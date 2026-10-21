"""Convenience functions for embedding the harness without the CLI."""

from __future__ import annotations

from pathlib import Path

from energy_tsf.api.service import EnergyTSFService
from energy_tsf.config.writer import (
    MaterializedBacktestRequest,
    MaterializedForecastRequest,
    MaterializedRequest,
    MaterializedUpdateRequest,
)
from energy_tsf.contracts.backtest import BacktestRequest, BacktestResult
from energy_tsf.contracts.forecast import ForecastPlan, ForecastRequest, ForecastResult
from energy_tsf.contracts.modeling import ModelingRequest
from energy_tsf.contracts.result import ModelingWorkflowResult
from energy_tsf.contracts.update import UpdateRequest, UpdateResult


def prepare_modeling(
    request: ModelingRequest,
    task_dir: str | Path,
    workspace_root: str | Path = ".",
) -> MaterializedRequest:
    """Validate a modeling request and write its canonical task configuration."""
    return EnergyTSFService(workspace_root).prepare_modeling(request, task_dir)


def inspect_modeling(config_path: str | Path, workspace_root: str | Path = ".") -> dict:
    """Inspect modeling data and build a trial plan without training models."""
    return EnergyTSFService(workspace_root).inspect_modeling(config_path)


def run_modeling(config_path: str | Path, workspace_root: str | Path = ".") -> ModelingWorkflowResult:
    """Run the frozen modeling configuration through candidate evaluation."""
    return EnergyTSFService(workspace_root).run_modeling(config_path)


def prepare_forecast(
    request: ForecastRequest,
    task_dir: str | Path,
    workspace_root: str | Path = ".",
) -> MaterializedForecastRequest:
    """Validate a forecast request and write its canonical task configuration."""
    return EnergyTSFService(workspace_root).prepare_forecast(request, task_dir)


def prepare_backtest(
    request: BacktestRequest,
    task_dir: str | Path,
    workspace_root: str | Path = ".",
) -> MaterializedBacktestRequest:
    """Validate a backtest request and write its canonical task configuration."""
    return EnergyTSFService(workspace_root).prepare_backtest(request, task_dir)


def inspect_backtest(config_path: str | Path, workspace_root: str | Path = ".") -> dict:
    """Pin the current Active solution and inspect rolling backtest windows."""
    return EnergyTSFService(workspace_root).inspect_backtest(config_path)


def run_backtest(
    config_path: str | Path,
    workspace_root: str | Path = ".",
    plan_path: str | Path | None = None,
) -> BacktestResult:
    """Evaluate the pinned Active solution and optionally publish new weights."""
    return EnergyTSFService(workspace_root).run_backtest(config_path, plan_path)


def inspect_forecast(config_path: str | Path, workspace_root: str | Path = ".") -> dict:
    """Inspect forecast input and compatible Active solutions."""
    return EnergyTSFService(workspace_root).inspect_forecast(config_path)


def plan_forecast(
    config_path: str | Path,
    selection_path: str | Path,
    workspace_root: str | Path = ".",
) -> ForecastPlan:
    """Validate a source selection and freeze a cross-station forecast plan."""
    return EnergyTSFService(workspace_root).plan_forecast(config_path, selection_path)


def run_forecast(
    config_path: str | Path,
    workspace_root: str | Path = ".",
    plan_path: str | Path | None = None,
) -> ForecastResult:
    """Run inference using a frozen forecast plan and write delivery reports."""
    return EnergyTSFService(workspace_root).run_forecast(config_path, plan_path)


def prepare_update(
    request: UpdateRequest,
    task_dir: str | Path,
    workspace_root: str | Path = ".",
) -> MaterializedUpdateRequest:
    """Validate an update request and write its canonical task configuration."""
    return EnergyTSFService(workspace_root).prepare_update(request, task_dir)


def inspect_update(config_path: str | Path, workspace_root: str | Path = ".") -> dict:
    """Pin the Active solution and its SearchPool, then freeze an update plan."""
    return EnergyTSFService(workspace_root).inspect_update(config_path)


def run_update(
    config_path: str | Path,
    workspace_root: str | Path = ".",
    plan_path: str | Path | None = None,
) -> UpdateResult:
    """Retrain candidates from the pinned SearchPool and evaluate publication."""
    return EnergyTSFService(workspace_root).run_update(config_path, plan_path)
