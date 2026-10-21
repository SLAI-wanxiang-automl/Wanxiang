"""Create deterministic task TOML and immutable request snapshots."""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from energy_tsf.config.loader import (
    LoadedBacktestConfig,
    LoadedForecastConfig,
    LoadedModelingConfig,
    LoadedUpdateConfig,
    load_backtest_config,
    load_forecast_config,
    load_modeling_config,
    load_update_config,
)
from energy_tsf.config.preflight import preflight_request
from energy_tsf.config.schema import (
    BacktestConfig,
    BacktestDataConfig,
    BacktestRequestConfig,
    BayesianUpdateConfig,
    DataConfig,
    ForecastConfig,
    ForecastDataConfig,
    ForecastRequestConfig,
    ForecastRetrievalConfig,
    ModelingConfig,
    RequestConfig,
    SearchConfig,
    SearchProvidersConfig,
    UpdateConfig,
    UpdateDataConfig,
    UpdateRequestConfig,
    UpdateSearchConfig,
)
from energy_tsf.config.preflight import preflight_backtest_request, preflight_forecast_request, preflight_update_request
from energy_tsf.contracts.backtest import BacktestRequest
from energy_tsf.contracts.forecast import ForecastRequest
from energy_tsf.contracts.modeling import ModelingRequest, PreflightReport
from energy_tsf.contracts.update import UpdateRequest


@dataclass(frozen=True)
class MaterializedRequest:
    task_dir: Path
    request_path: Path
    modeling_path: Path
    resolved_path: Path
    config_hash: str
    preflight: PreflightReport


@dataclass(frozen=True)
class MaterializedForecastRequest:
    task_dir: Path
    request_path: Path
    forecast_path: Path
    resolved_path: Path
    config_hash: str
    preflight: PreflightReport


@dataclass(frozen=True)
class MaterializedBacktestRequest:
    task_dir: Path
    request_path: Path
    backtest_path: Path
    resolved_path: Path
    config_hash: str
    preflight: PreflightReport


@dataclass(frozen=True)
class MaterializedUpdateRequest:
    task_dir: Path
    request_path: Path
    update_path: Path
    resolved_path: Path
    config_hash: str
    preflight: PreflightReport


_BARE_TOML_KEY = re.compile(r"^[A-Za-z0-9_-]+$")


def _toml_key(value: str) -> str:
    """Render one TOML key without changing dots into table separators."""

    return value if _BARE_TOML_KEY.fullmatch(value) else json.dumps(value, ensure_ascii=False)


def config_from_request(request: ModelingRequest) -> ModelingConfig:
    """Map the public request into the complete first-release config."""

    return ModelingConfig(
        request=RequestConfig(
            task_type=request.task_type,
            request_id=request.request_id,
            station_id=request.station_id,
            target=request.target,
            horizon=request.horizon,
            prediction_mode=request.prediction_mode,
            top_k=request.top_k,
            publish=request.publish,
        ),
        data=DataConfig(
            path=request.data_uri,
            timestamp_column=request.timestamp_column,
            frequency=request.frequency,
            covariates=request.covariates,
        ),
        split=request.split,
        search=SearchConfig(
            seed=request.seed,
            max_trials=request.budget.max_trials,
            max_parallel_trials=request.budget.max_parallel_trials,
            providers=SearchProvidersConfig(
                enabled=request.providers,
                checkpoints=request.foundation_checkpoints,
            ),
        ),
    )


def forecast_config_from_request(request: ForecastRequest) -> ForecastConfig:
    """Map a public forecast request to the resolved first-release config."""

    return ForecastConfig(
        request=ForecastRequestConfig(
            request_id=request.request_id,
            horizon=request.horizon,
            prediction_mode=request.prediction_mode,
            issue_time=request.issue_time,
        ),
        station=request.station,
        data=ForecastDataConfig(
            path=request.data_uri,
            future_covariates_path=request.future_covariates_uri,
            covariates=request.covariates,
        ),
        retrieval=ForecastRetrievalConfig(
            allow_similar=request.allow_similar,
            allow_similar_on_direct_failure=request.allow_similar_on_direct_failure,
            max_source_solutions=request.max_source_solutions,
            deadline_seconds=request.deadline_seconds,
        ),
    )


def backtest_config_from_request(request: BacktestRequest) -> BacktestConfig:
    """Map a validated BacktestRequest into its executable config model."""
    return BacktestConfig(
        request=BacktestRequestConfig(
            request_id=request.request_id,
            station_id=request.station_id,
            target=request.target,
            horizon=request.horizon,
            stride=request.stride,
            prediction_mode=request.prediction_mode,
            timezone=request.timezone,
            publish=request.publish,
        ),
        data=BacktestDataConfig(
            path=request.data_uri,
            timestamp_column=request.timestamp_column,
            frequency=request.frequency,
            covariates=request.covariates,
        ),
        bayesian=BayesianUpdateConfig(
            epsilon=request.epsilon,
            temperature=request.temperature,
            prior_strength=request.prior_strength,
        ),
    )


def update_config_from_request(request: UpdateRequest) -> UpdateConfig:
    """Map a validated UpdateRequest into its executable config model."""
    return UpdateConfig(
        request=UpdateRequestConfig(
            request_id=request.request_id,
            station_id=request.station_id,
            target=request.target,
            horizon=request.horizon,
            stride=request.stride,
            timezone=request.timezone,
            validation_days=request.validation_days,
            prediction_mode=request.prediction_mode,
            top_k=request.top_k,
            publish=request.publish,
            seed=request.seed,
        ),
        data=UpdateDataConfig(
            path=request.data_uri,
            timestamp_column=request.timestamp_column,
            frequency=request.frequency,
            covariates=request.covariates,
        ),
        search=UpdateSearchConfig(max_parallel_trials=request.max_parallel_trials),
    )


def _toml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_scalar(item) for item in value) + "]"
    raise TypeError(f"unsupported TOML value: {type(value).__name__}")


def render_toml(value: ModelingConfig | ForecastConfig | BacktestConfig | UpdateConfig | Mapping[str, Any]) -> str:
    """Render the small supported config shape in a deterministic order."""

    data = (
        value.model_dump(mode="json", exclude_none=True)
        if isinstance(value, (ModelingConfig, ForecastConfig, BacktestConfig, UpdateConfig))
        else dict(value)
    )
    lines: list[str] = []

    def emit_table(table: Mapping[str, Any], path: tuple[str, ...]) -> None:
        if path:
            lines.append(f"[{'.'.join(_toml_key(item) for item in path)}]")
        for key, item in table.items():
            if not isinstance(item, Mapping):
                lines.append(f"{_toml_key(key)} = {_toml_scalar(item)}")
        if path:
            lines.append("")
        for key, item in table.items():
            if isinstance(item, Mapping):
                emit_table(item, (*path, key))

    emit_table(data, ())
    return "\n".join(lines).rstrip() + "\n"


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_modeling_config(config: ModelingConfig, path: str | os.PathLike[str]) -> Path:
    """Serialize and atomically write a modeling TOML configuration."""
    target = Path(path)
    _atomic_write(target, render_toml(config))
    return target


def write_forecast_config(config: ForecastConfig, path: str | os.PathLike[str]) -> Path:
    """Serialize and atomically write a forecast TOML configuration."""
    target = Path(path)
    _atomic_write(target, render_toml(config))
    return target


def write_backtest_config(config: BacktestConfig, path: str | os.PathLike[str]) -> Path:
    """Serialize and atomically write a backtest TOML configuration."""
    target = Path(path)
    _atomic_write(target, render_toml(config))
    return target


def write_update_config(config: UpdateConfig, path: str | os.PathLike[str]) -> Path:
    """Serialize and atomically write an update TOML configuration."""
    target = Path(path)
    _atomic_write(target, render_toml(config))
    return target


def materialize_modeling_request(
    request: ModelingRequest,
    task_dir: str | os.PathLike[str],
    defaults: Iterable[str | os.PathLike[str]] = (),
) -> MaterializedRequest:
    """Validate, preflight, and persist request/modeling/resolved snapshots."""

    preflight = preflight_request(request)
    if not preflight.valid:
        details = "; ".join(issue.message for issue in preflight.errors)
        raise ValueError(f"modeling preflight failed: {details}")

    target_dir = Path(task_dir)
    request_path = target_dir / "request.json"
    modeling_path = target_dir / "modeling.toml"
    resolved_path = target_dir / "resolved.toml"
    _atomic_write(
        request_path,
        json.dumps(request.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    write_modeling_config(config_from_request(request), modeling_path)
    loaded: LoadedModelingConfig = load_modeling_config(modeling_path, defaults=defaults)
    write_modeling_config(loaded.config, resolved_path)
    reloaded = load_modeling_config(resolved_path)
    if reloaded.config_hash != loaded.config_hash:
        raise RuntimeError("resolved config changed during serialization")

    return MaterializedRequest(
        task_dir=target_dir,
        request_path=request_path,
        modeling_path=modeling_path,
        resolved_path=resolved_path,
        config_hash=loaded.config_hash,
        preflight=preflight,
    )


def materialize_forecast_request(
    request: ForecastRequest,
    task_dir: str | os.PathLike[str],
    defaults: Iterable[str | os.PathLike[str]] = (),
) -> MaterializedForecastRequest:
    """Validate and persist request, editable, and resolved forecast snapshots."""

    preflight = preflight_forecast_request(request)
    if not preflight.valid:
        details = "; ".join(issue.message for issue in preflight.errors)
        raise ValueError(f"forecast preflight failed: {details}")

    target_dir = Path(task_dir)
    request_path = target_dir / "request.json"
    forecast_path = target_dir / "forecast.toml"
    resolved_path = target_dir / "resolved.toml"
    _atomic_write(
        request_path,
        json.dumps(request.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    write_forecast_config(forecast_config_from_request(request), forecast_path)
    loaded: LoadedForecastConfig = load_forecast_config(forecast_path, defaults=defaults)
    write_forecast_config(loaded.config, resolved_path)
    reloaded = load_forecast_config(resolved_path)
    if reloaded.config_hash != loaded.config_hash:
        raise RuntimeError("resolved forecast config changed during serialization")
    return MaterializedForecastRequest(
        task_dir=target_dir,
        request_path=request_path,
        forecast_path=forecast_path,
        resolved_path=resolved_path,
        config_hash=loaded.config_hash,
        preflight=preflight,
    )


def materialize_backtest_request(
    request: BacktestRequest,
    task_dir: str | os.PathLike[str],
    defaults: Iterable[str | os.PathLike[str]] = (),
) -> MaterializedBacktestRequest:
    """Validate and persist request, editable, and resolved backtest snapshots."""

    preflight = preflight_backtest_request(request)
    if not preflight.valid:
        details = "; ".join(issue.message for issue in preflight.errors)
        raise ValueError(f"backtest preflight failed: {details}")
    target_dir = Path(task_dir)
    request_path = target_dir / "request.json"
    backtest_path = target_dir / "backtest.toml"
    resolved_path = target_dir / "resolved.toml"
    _atomic_write(
        request_path,
        json.dumps(request.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    write_backtest_config(backtest_config_from_request(request), backtest_path)
    loaded: LoadedBacktestConfig = load_backtest_config(backtest_path, defaults=defaults)
    write_backtest_config(loaded.config, resolved_path)
    reloaded = load_backtest_config(resolved_path)
    if reloaded.config_hash != loaded.config_hash:
        raise RuntimeError("resolved backtest config changed during serialization")
    return MaterializedBacktestRequest(
        task_dir=target_dir,
        request_path=request_path,
        backtest_path=backtest_path,
        resolved_path=resolved_path,
        config_hash=loaded.config_hash,
        preflight=preflight,
    )


def materialize_update_request(
    request: UpdateRequest,
    task_dir: str | os.PathLike[str],
    defaults: Iterable[str | os.PathLike[str]] = (),
) -> MaterializedUpdateRequest:
    """Validate an update request and persist request, task, and resolved files."""
    preflight = preflight_update_request(request)
    if not preflight.valid:
        details = "; ".join(issue.message for issue in preflight.errors)
        raise ValueError(f"update preflight failed: {details}")
    target_dir = Path(task_dir)
    request_path = target_dir / "request.json"
    update_path = target_dir / "update.toml"
    resolved_path = target_dir / "resolved.toml"
    _atomic_write(
        request_path,
        json.dumps(request.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    write_update_config(update_config_from_request(request), update_path)
    loaded: LoadedUpdateConfig = load_update_config(update_path, defaults=defaults)
    write_update_config(loaded.config, resolved_path)
    reloaded = load_update_config(resolved_path)
    if reloaded.config_hash != loaded.config_hash:
        raise RuntimeError("resolved update config changed during serialization")
    return MaterializedUpdateRequest(
        task_dir=target_dir,
        request_path=request_path,
        update_path=update_path,
        resolved_path=resolved_path,
        config_hash=loaded.config_hash,
        preflight=preflight,
    )
