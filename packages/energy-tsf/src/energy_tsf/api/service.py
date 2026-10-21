"""Application service shared by CLI and future host integrations."""

from __future__ import annotations

from pathlib import Path

from energy_tsf.automl.planner import PlanningInput, build_experiment_plan
from energy_tsf.automl.search_space import load_default_search_spaces
from energy_tsf.config.loader import (
    config_fingerprint,
    load_backtest_config,
    load_forecast_config,
    load_modeling_config,
    load_update_config,
)
from energy_tsf.config.writer import (
    MaterializedBacktestRequest,
    MaterializedForecastRequest,
    MaterializedRequest,
    MaterializedUpdateRequest,
    materialize_backtest_request,
    materialize_forecast_request,
    materialize_modeling_request,
    materialize_update_request,
)
from energy_tsf.contracts.backtest import BacktestRequest, BacktestResult
from energy_tsf.contracts.forecast import ForecastPlan, ForecastRequest, ForecastResult
from energy_tsf.contracts.modeling import ModelingRequest
from energy_tsf.contracts.result import ModelingWorkflowResult
from energy_tsf.contracts.update import UpdateRequest, UpdateResult
from energy_tsf.data import build_origin_sets, load_modeling_data, prepare_series
from energy_tsf.data.snapshots import snapshot_fingerprint
from energy_tsf.providers import modeling_registry
from energy_tsf.tools.fingerprints import json_fingerprint
from energy_tsf.workflows.backtest import BacktestWorkflow
from energy_tsf.workflows.forecast import ForecastWorkflow
from energy_tsf.workflows.modeling import ModelingWorkflow
from energy_tsf.workflows.update import UpdateWorkflow


class EnergyTSFService:
    """Coordinate request materialization and deterministic workflows.

    The service is the shared application boundary for the CLI and Python
    callers. It validates and loads task configuration, then delegates all
    domain calculations to the corresponding workflow.
    """

    def __init__(self, workspace_root: str | Path) -> None:
        """Create a service whose relative paths resolve from ``workspace_root``."""
        self.workspace_root = Path(workspace_root)

    def prepare_modeling(self, request: ModelingRequest, task_dir: str | Path) -> MaterializedRequest:
        """Persist a validated modeling request and its resolved TOML config."""
        return materialize_modeling_request(request, task_dir)

    def prepare_forecast(
        self, request: ForecastRequest, task_dir: str | Path
    ) -> MaterializedForecastRequest:
        """Persist a validated forecast request and its resolved TOML config."""
        return materialize_forecast_request(request, task_dir)

    def prepare_backtest(
        self, request: BacktestRequest, task_dir: str | Path
    ) -> MaterializedBacktestRequest:
        """Persist a validated backtest request and its resolved TOML config."""
        return materialize_backtest_request(request, task_dir)

    def prepare_update(
        self, request: UpdateRequest, task_dir: str | Path
    ) -> MaterializedUpdateRequest:
        """Persist a validated update request and its resolved TOML config."""
        return materialize_update_request(request, task_dir)

    def inspect_modeling(self, config_path: str | Path) -> dict:
        """Report data quality, chronological splits, and trials without fitting."""
        loaded = load_modeling_config(config_path)
        raw = load_modeling_data(loaded.config)
        prepared = prepare_series(raw)
        origins = build_origin_sets(prepared.split, prepared.frequency)
        registry = modeling_registry()
        planning = PlanningInput(
            data_snapshot_id=snapshot_fingerprint(prepared, {"config_hash": loaded.config_hash}),
            split_id=json_fingerprint(prepared.split.model_dump(mode="json")),
            train_end=prepared.split.train_end,
            validation_end=prepared.split.validation_end,
            frequency=prepared.frequency,
            providers=loaded.config.search.providers.enabled,
            max_trials=loaded.config.search.max_trials,
            seed=loaded.config.search.seed,
            foundation_checkpoints=loaded.config.search.providers.checkpoints,
            prediction_mode=loaded.config.request.prediction_mode,
            covariates=loaded.config.data.covariates,
            feature_options=loaded.config.features.model_dump(
                exclude={"statistical_selection", "tsfm_wrapper"}
            ),
            statistical_selection=loaded.config.features.statistical_selection,
            wrapper_selection=loaded.config.features.tsfm_wrapper,
        )
        plan = build_experiment_plan(planning, load_default_search_spaces(), registry)
        split = prepared.split.model_dump(mode="json")
        if split.get("partition_kind") == "modeling":
            split.pop("partition_kind")
        return {
            "config_hash": config_fingerprint(loaded.config),
            "rows": len(prepared.frame),
            "quality": prepared.quality.model_dump(mode="json"),
            "imputation": prepared.imputation.model_dump(mode="json"),
            "split": split,
            "horizon_steps": origins.horizon_steps,
            "origins": {
                "train": len(origins.train),
                "validation": len(origins.validation),
                "test": len(origins.test),
            },
            "plan": plan.model_dump(mode="json"),
        }

    def run_modeling(self, config_path: str | Path) -> ModelingWorkflowResult:
        """Execute model search, validation selection, final evaluation, and gates."""
        loaded = load_modeling_config(config_path)
        return ModelingWorkflow(self.workspace_root).run(loaded.config)

    def inspect_forecast(self, config_path: str | Path) -> dict:
        """Clean forecast history and inspect direct or similar Active sources."""
        loaded = load_forecast_config(config_path)
        return ForecastWorkflow(self.workspace_root).inspect(loaded.config)

    def plan_forecast(
        self,
        config_path: str | Path,
        selection_path: str | Path,
    ) -> ForecastPlan:
        """Freeze a forecast plan from a validated source selection."""
        loaded = load_forecast_config(config_path)
        return ForecastWorkflow(self.workspace_root).freeze_plan(loaded.config, selection_path)

    def run_forecast(
        self,
        config_path: str | Path,
        plan_path: str | Path | None = None,
    ) -> ForecastResult:
        """Execute inference from the frozen plan and return delivery metadata."""
        loaded = load_forecast_config(config_path)
        return ForecastWorkflow(self.workspace_root).run(loaded.config, plan_path)

    def inspect_backtest(self, config_path: str | Path) -> dict:
        """Pin the Active parent and prepare point-in-time evaluation windows."""
        loaded = load_backtest_config(config_path)
        return BacktestWorkflow(self.workspace_root).inspect(loaded.config)

    def run_backtest(
        self,
        config_path: str | Path,
        plan_path: str | Path | None = None,
    ) -> BacktestResult:
        """Replay the pinned solution and optionally publish updated weights."""
        loaded = load_backtest_config(config_path)
        return BacktestWorkflow(self.workspace_root).run(loaded.config, plan_path)

    def inspect_update(self, config_path: str | Path) -> dict:
        """Pin the Active SearchPool and create a deterministic update plan."""
        loaded = load_update_config(config_path)
        return UpdateWorkflow(self.workspace_root).inspect(loaded.config)

    def run_update(
        self,
        config_path: str | Path,
        plan_path: str | Path | None = None,
    ) -> UpdateResult:
        """Retrain only pinned pool candidates and evaluate the release gate."""
        loaded = load_update_config(config_path)
        return UpdateWorkflow(self.workspace_root).run(loaded.config, plan_path)
