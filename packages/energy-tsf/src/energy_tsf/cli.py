"""Small deterministic command-line boundary used by the external Agent."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from typing import Any

from energy_tsf.api.service import EnergyTSFService
from energy_tsf.contracts.backtest import BacktestRequest
from energy_tsf.contracts.features import CovariateSpec
from energy_tsf.contracts.forecast import ForecastRequest
from energy_tsf.contracts.modeling import DEFAULT_PROVIDERS, ModelingRequest, SearchBudget
from energy_tsf.contracts.station import StationSpec
from energy_tsf.contracts.update import UpdateRequest
from energy_tsf.config.writer import (
    materialize_backtest_request,
    materialize_forecast_request,
    materialize_modeling_request,
)
from energy_tsf.data.solar import export_legacy_modern_tsf_solar
from energy_tsf.memory import StationRepository, load_station_toml


def _parser() -> argparse.ArgumentParser:
    """Define command names, arguments, and defaults exposed by the CLI."""
    parser = argparse.ArgumentParser(prog="energy-tsf")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser(
        "prepare-modeling",
        help="validate a structured modeling request and create TOML snapshots",
    )
    prepare.add_argument("--request-id", required=True)
    prepare.add_argument("--station-id", required=True)
    prepare.add_argument("--data", required=True)
    prepare.add_argument("--target", required=True)
    prepare.add_argument("--timestamp-column")
    prepare.add_argument("--frequency", required=True)
    prepare.add_argument("--prediction-mode", choices=("univariate", "multivariate"), default="univariate")
    prepare.add_argument("--past-covariate", action="append", default=[])
    prepare.add_argument("--known-future-covariate", action="append", default=[])
    prepare.add_argument("--top-k", type=int, default=5)
    prepare.add_argument("--provider", action="append", dest="providers")
    prepare.add_argument(
        "--foundation-checkpoint",
        action="append",
        default=[],
        metavar="PROVIDER=PATH",
    )
    prepare.add_argument("--max-trials", type=int, default=64)
    prepare.add_argument("--max-parallel-trials", type=int, default=2)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--no-publish", action="store_true")
    prepare.add_argument("--output-dir", required=True)
    inspect = commands.add_parser("inspect-modeling", help="inspect data and expand the trial plan without training")
    inspect.add_argument("--config", required=True)
    inspect.add_argument("--workspace-root", default=".")
    run = commands.add_parser("run-modeling", help="run modeling through Candidate and optional Active publication")
    run.add_argument("--config", required=True)
    run.add_argument("--workspace-root", default=".")
    solar = commands.add_parser("import-solar", help="convert one ModernTSF Solar column to generic CSV")
    solar.add_argument("--source", required=True)
    solar.add_argument("--output", required=True)
    solar.add_argument("--column", type=int, default=0)
    solar.add_argument("--target", default="power")
    solar.add_argument(
        "--tail-fraction",
        type=float,
        default=1.0,
        help="retain the final fraction of rows after selecting the target channel",
    )
    forecast = commands.add_parser(
        "prepare-forecast",
        help="validate a structured forecast request and create TOML snapshots",
    )
    forecast.add_argument("--request-id", required=True)
    forecast.add_argument("--station-id", required=True)
    forecast.add_argument("--energy-type", required=True)
    forecast.add_argument("--timezone", required=True)
    forecast.add_argument("--data", required=True)
    forecast.add_argument("--target", required=True)
    forecast.add_argument("--target-unit", required=True)
    forecast.add_argument("--timestamp-column")
    forecast.add_argument("--frequency", required=True)
    forecast.add_argument("--prediction-mode", choices=("univariate", "multivariate"), default="univariate")
    forecast.add_argument("--past-covariate", action="append", default=[])
    forecast.add_argument("--known-future-covariate", action="append", default=[])
    forecast.add_argument("--future-covariates")
    forecast.add_argument("--latitude", type=float)
    forecast.add_argument("--longitude", type=float)
    forecast.add_argument("--technology")
    forecast.add_argument("--capacity", type=float)
    forecast.add_argument("--capacity-unit")
    forecast.add_argument("--issue-time")
    forecast.add_argument("--no-similar", action="store_true")
    forecast.add_argument("--allow-similar-on-direct-failure", action="store_true")
    forecast.add_argument("--max-source-solutions", type=int, default=3)
    forecast.add_argument("--deadline-seconds", type=int)
    forecast.add_argument("--output-dir", required=True)
    inspect_forecast = commands.add_parser(
        "inspect-forecast",
        help="clean recent history and inspect direct or similar Active solutions",
    )
    inspect_forecast.add_argument("--config", required=True)
    inspect_forecast.add_argument("--workspace-root", default=".")
    plan_forecast = commands.add_parser(
        "plan-forecast",
        help="validate an Agent selection and freeze a similar-station plan",
    )
    plan_forecast.add_argument("--config", required=True)
    plan_forecast.add_argument("--selection", required=True)
    plan_forecast.add_argument("--workspace-root", default=".")
    run_forecast = commands.add_parser(
        "run-forecast",
        help="execute a frozen forecast plan and deliver reports",
    )
    run_forecast.add_argument("--config", required=True)
    run_forecast.add_argument("--plan")
    run_forecast.add_argument("--workspace-root", default=".")
    register_station = commands.add_parser(
        "register-station",
        help="register explicit station metadata for similar-station retrieval",
    )
    register_station.add_argument("--config", required=True)
    register_station.add_argument("--memory-root", default="asset_store")
    backtest = commands.add_parser(
        "prepare-backtest",
        help="validate a timestamped backtest request and create TOML snapshots",
    )
    backtest.add_argument("--request-id", required=True)
    backtest.add_argument("--station-id", required=True)
    backtest.add_argument("--data", required=True)
    backtest.add_argument("--target", required=True)
    backtest.add_argument("--timestamp-column", required=True)
    backtest.add_argument("--frequency", required=True)
    backtest.add_argument("--timezone", required=True)
    backtest.add_argument("--prediction-mode", choices=("univariate", "multivariate"), default="univariate")
    backtest.add_argument("--past-covariate", action="append", default=[])
    backtest.add_argument("--known-future-covariate", action="append", default=[])
    backtest.add_argument("--epsilon", type=float, default=1e-3)
    backtest.add_argument("--temperature", type=float, default=0.10)
    backtest.add_argument("--prior-strength", type=float)
    backtest.add_argument("--no-publish", action="store_true")
    backtest.add_argument("--output-dir", required=True)
    inspect_backtest = commands.add_parser(
        "inspect-backtest",
        help="pin the exact Active solution and inspect rolling windows",
    )
    inspect_backtest.add_argument("--config", required=True)
    inspect_backtest.add_argument("--workspace-root", default=".")
    run_backtest = commands.add_parser(
        "run-backtest",
        help="evaluate the frozen solution, update weights, and optionally publish",
    )
    run_backtest.add_argument("--config", required=True)
    run_backtest.add_argument("--plan")
    run_backtest.add_argument("--workspace-root", default=".")
    update = commands.add_parser(
        "prepare-update",
        help="validate an existing-station model update request and create TOML snapshots",
    )
    update.add_argument("--request-id", required=True)
    update.add_argument("--station-id", required=True)
    update.add_argument("--data", required=True)
    update.add_argument("--target", required=True)
    update.add_argument("--timestamp-column", required=True)
    update.add_argument("--frequency", required=True)
    update.add_argument("--timezone", required=True)
    update.add_argument("--prediction-mode", choices=("univariate", "multivariate"), default="univariate")
    update.add_argument("--past-covariate", action="append", default=[])
    update.add_argument("--known-future-covariate", action="append", default=[])
    update.add_argument("--top-k", type=int, default=5)
    update.add_argument("--max-parallel-trials", type=int, default=2)
    update.add_argument("--seed", type=int, default=42)
    update.add_argument("--no-publish", action="store_true")
    update.add_argument("--output-dir", required=True)
    inspect_update = commands.add_parser(
        "inspect-update",
        help="pin the Active search pool and freeze the retraining plan",
    )
    inspect_update.add_argument("--config", required=True)
    inspect_update.add_argument("--workspace-root", default=".")
    run_update = commands.add_parser(
        "run-update",
        help="retrain the Active search pool and optionally publish the new Solution",
    )
    run_update.add_argument("--config", required=True)
    run_update.add_argument("--plan")
    run_update.add_argument("--workspace-root", default=".")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch one CLI command and print its machine-readable result.

    The command layer only parses inputs and formats outputs; validation and
    task execution are delegated to the shared service and workflows.
    """

    args = _parser().parse_args(argv)
    request: Any
    result: Any
    if args.command == "prepare-modeling":
        checkpoints = {}
        for item in args.foundation_checkpoint:
            if "=" not in item:
                raise ValueError("--foundation-checkpoint must use PROVIDER=PATH")
            provider_name, checkpoint_path = item.split("=", 1)
            checkpoints[provider_name] = checkpoint_path
        request = ModelingRequest(
            request_id=args.request_id,
            station_id=args.station_id,
            data_uri=args.data,
            target=args.target,
            timestamp_column=args.timestamp_column,
            frequency=args.frequency,
            prediction_mode=args.prediction_mode,
            covariates=CovariateSpec(
                past_only=tuple(args.past_covariate),
                known_future=tuple(args.known_future_covariate),
            ),
            top_k=args.top_k,
            providers=tuple(args.providers or DEFAULT_PROVIDERS),
            foundation_checkpoints=checkpoints,
            budget=SearchBudget(
                max_trials=args.max_trials,
                max_parallel_trials=args.max_parallel_trials,
            ),
            seed=args.seed,
            publish=not args.no_publish,
        )
        result = materialize_modeling_request(request, args.output_dir)
        print(
            json.dumps(
                {
                    "request_id": request.request_id,
                    "task_dir": str(result.task_dir),
                    "config_hash": result.config_hash,
                    "resolved_config": str(result.resolved_path),
                    "warnings": [issue.model_dump() for issue in result.preflight.warnings],
                },
                ensure_ascii=False,
            )
        )
        return 0
    if args.command == "prepare-forecast":
        station = StationSpec(
            station_id=args.station_id,
            energy_type=args.energy_type,
            timezone=args.timezone,
            latitude=args.latitude,
            longitude=args.longitude,
            technology=args.technology,
            capacity=args.capacity,
            capacity_unit=args.capacity_unit,
            target=args.target,
            target_unit=args.target_unit,
            frequency=args.frequency,
            timestamp_column=args.timestamp_column,
        )
        request = ForecastRequest(
            request_id=args.request_id,
            station=station,
            data_uri=args.data,
            future_covariates_uri=args.future_covariates,
            prediction_mode=args.prediction_mode,
            covariates=CovariateSpec(
                past_only=tuple(args.past_covariate),
                known_future=tuple(args.known_future_covariate),
            ),
            issue_time=args.issue_time,
            allow_similar=not args.no_similar,
            allow_similar_on_direct_failure=args.allow_similar_on_direct_failure,
            max_source_solutions=args.max_source_solutions,
            deadline_seconds=args.deadline_seconds,
        )
        result = materialize_forecast_request(request, args.output_dir)
        print(
            json.dumps(
                {
                    "request_id": request.request_id,
                    "task_dir": str(result.task_dir),
                    "config_hash": result.config_hash,
                    "resolved_config": str(result.resolved_path),
                    "warnings": [issue.model_dump() for issue in result.preflight.warnings],
                },
                ensure_ascii=False,
            )
        )
        return 0
    if args.command == "prepare-backtest":
        request = BacktestRequest(
            request_id=args.request_id,
            station_id=args.station_id,
            data_uri=args.data,
            target=args.target,
            timestamp_column=args.timestamp_column,
            frequency=args.frequency,
            timezone=args.timezone,
            prediction_mode=args.prediction_mode,
            covariates=CovariateSpec(
                past_only=tuple(args.past_covariate),
                known_future=tuple(args.known_future_covariate),
            ),
            publish=not args.no_publish,
            epsilon=args.epsilon,
            temperature=args.temperature,
            prior_strength=args.prior_strength,
        )
        result = materialize_backtest_request(request, args.output_dir)
        print(
            json.dumps(
                {
                    "request_id": request.request_id,
                    "task_dir": str(result.task_dir),
                    "config_hash": result.config_hash,
                    "resolved_config": str(result.resolved_path),
                    "warnings": [issue.model_dump() for issue in result.preflight.warnings],
                },
                ensure_ascii=False,
            )
        )
        return 0
    if args.command == "prepare-update":
        request = UpdateRequest(
            request_id=args.request_id,
            station_id=args.station_id,
            data_uri=args.data,
            target=args.target,
            timestamp_column=args.timestamp_column,
            frequency=args.frequency,
            timezone=args.timezone,
            prediction_mode=args.prediction_mode,
            covariates=CovariateSpec(
                past_only=tuple(args.past_covariate),
                known_future=tuple(args.known_future_covariate),
            ),
            top_k=args.top_k,
            max_parallel_trials=args.max_parallel_trials,
            seed=args.seed,
            publish=not args.no_publish,
        )
        result = EnergyTSFService(".").prepare_update(request, args.output_dir)
        print(
            json.dumps(
                {
                    "request_id": request.request_id,
                    "task_dir": str(result.task_dir),
                    "config_hash": result.config_hash,
                    "resolved_config": str(result.resolved_path),
                    "warnings": [issue.model_dump() for issue in result.preflight.warnings],
                },
                ensure_ascii=False,
            )
        )
        return 0
    if args.command == "inspect-update":
        result = EnergyTSFService(args.workspace_root).inspect_update(args.config)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    if args.command == "run-update":
        result = EnergyTSFService(args.workspace_root).run_update(args.config, args.plan)
        print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False))
        return 0
    if args.command == "inspect-backtest":
        result = EnergyTSFService(args.workspace_root).inspect_backtest(args.config)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    if args.command == "run-backtest":
        result = EnergyTSFService(args.workspace_root).run_backtest(args.config, args.plan)
        print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False))
        return 0
    if args.command == "inspect-forecast":
        result = EnergyTSFService(args.workspace_root).inspect_forecast(args.config)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    if args.command == "plan-forecast":
        result = EnergyTSFService(args.workspace_root).plan_forecast(
            args.config,
            args.selection,
        )
        print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False))
        return 0
    if args.command == "run-forecast":
        result = EnergyTSFService(args.workspace_root).run_forecast(args.config, args.plan)
        print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False))
        return 0
    if args.command == "register-station":
        station = load_station_toml(args.config)
        path = StationRepository(args.memory_root).save(station)
        print(
            json.dumps(
                {"station_id": station.station_id, "profile_path": str(path)},
                ensure_ascii=False,
            )
        )
        return 0
    if args.command == "inspect-modeling":
        result = EnergyTSFService(args.workspace_root).inspect_modeling(args.config)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    if args.command == "run-modeling":
        result = EnergyTSFService(args.workspace_root).run_modeling(args.config)
        print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False))
        return 0
    if args.command == "import-solar":
        output = export_legacy_modern_tsf_solar(
            args.source,
            args.output,
            column=args.column,
            target=args.target,
            tail_fraction=args.tail_fraction,
        )
        import pandas as pd

        output_rows = len(pd.read_csv(output, usecols=[args.target]))
        print(
            json.dumps(
                {
                    "output": str(output),
                    "target": args.target,
                    "column": args.column,
                    "tail_fraction": args.tail_fraction,
                    "output_rows": output_rows,
                },
                ensure_ascii=False,
            )
        )
        return 0
    raise AssertionError(f"unhandled command: {args.command}")
