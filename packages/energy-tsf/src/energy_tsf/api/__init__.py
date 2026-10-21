"""Stable application interface for CLI and host Agent integrations."""

from energy_tsf.api.facade import (
    inspect_backtest,
    inspect_forecast,
    inspect_modeling,
    inspect_update,
    plan_forecast,
    prepare_backtest,
    prepare_forecast,
    prepare_modeling,
    prepare_update,
    run_backtest,
    run_forecast,
    run_modeling,
    run_update,
)
from energy_tsf.api.service import EnergyTSFService

__all__ = [
    "EnergyTSFService",
    "inspect_backtest",
    "inspect_forecast",
    "inspect_modeling",
    "inspect_update",
    "plan_forecast",
    "prepare_backtest",
    "prepare_forecast",
    "prepare_modeling",
    "prepare_update",
    "run_backtest",
    "run_forecast",
    "run_modeling",
    "run_update",
]
