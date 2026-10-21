"""Load, inherit, validate, and fingerprint modeling TOML files."""

from __future__ import annotations

import hashlib
import json
import os
import tomllib
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel

from energy_tsf.config.schema import BacktestConfig, ForecastConfig, ModelingConfig, UpdateConfig


@dataclass(frozen=True)
class LoadedModelingConfig:
    raw: dict[str, Any]
    config: ModelingConfig
    config_hash: str
    source_paths: tuple[Path, ...]


@dataclass(frozen=True)
class LoadedForecastConfig:
    raw: dict[str, Any]
    config: ForecastConfig
    config_hash: str
    source_paths: tuple[Path, ...]


@dataclass(frozen=True)
class LoadedBacktestConfig:
    raw: dict[str, Any]
    config: BacktestConfig
    config_hash: str
    source_paths: tuple[Path, ...]


@dataclass(frozen=True)
class LoadedUpdateConfig:
    raw: dict[str, Any]
    config: UpdateConfig
    config_hash: str
    source_paths: tuple[Path, ...]


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge nested tables while replacing lists and scalar values."""

    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _read_with_extends(path: Path, stack: tuple[Path, ...] = ()) -> tuple[dict[str, Any], tuple[Path, ...]]:
    path = path.resolve()
    if path in stack:
        chain = " -> ".join(str(item) for item in (*stack, path))
        raise ValueError(f"cyclic config extends: {chain}")

    with path.open("rb") as stream:
        raw = tomllib.load(stream)

    extends = raw.pop("extends", ())
    if isinstance(extends, str):
        extends = (extends,)
    if not isinstance(extends, (list, tuple)):
        raise ValueError("extends must be a path or a list of paths")

    merged: dict[str, Any] = {}
    sources: tuple[Path, ...] = ()
    for item in extends:
        parent = Path(item)
        if not parent.is_absolute():
            parent = path.parent / parent
        inherited, inherited_sources = _read_with_extends(parent, (*stack, path))
        merged = deep_merge(merged, inherited)
        sources += inherited_sources
    return deep_merge(merged, raw), (*sources, path)


def _config_fingerprint(config: BaseModel) -> str:
    """Hash the canonical JSON representation of a validated task config."""
    payload = json.dumps(
        config.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def config_fingerprint(config: ModelingConfig) -> str:
    """Return the stable identity hash for a modeling configuration."""
    return _config_fingerprint(config)


def forecast_config_fingerprint(config: ForecastConfig) -> str:
    """Return the stable identity hash for a forecast configuration."""
    return _config_fingerprint(config)


def backtest_config_fingerprint(config: BacktestConfig) -> str:
    """Return the stable identity hash for a backtest configuration."""
    return _config_fingerprint(config)


def update_config_fingerprint(config: UpdateConfig) -> str:
    """Return the stable identity hash for an update configuration."""
    return _config_fingerprint(config)


def load_modeling_config(
    path: str | os.PathLike[str],
    defaults: Iterable[str | os.PathLike[str]] = (),
) -> LoadedModelingConfig:
    """Load defaults then the task file, validate, and return a stable hash."""

    merged: dict[str, Any] = {}
    sources: tuple[Path, ...] = ()
    for default_path in defaults:
        raw, loaded_paths = _read_with_extends(Path(default_path))
        merged = deep_merge(merged, raw)
        sources += loaded_paths

    raw, loaded_paths = _read_with_extends(Path(path))
    merged = deep_merge(merged, raw)
    sources += loaded_paths
    config = ModelingConfig.model_validate(merged)
    return LoadedModelingConfig(
        raw=merged,
        config=config,
        config_hash=config_fingerprint(config),
        source_paths=tuple(dict.fromkeys(sources)),
    )


def load_forecast_config(
    path: str | os.PathLike[str],
    defaults: Iterable[str | os.PathLike[str]] = (),
) -> LoadedForecastConfig:
    """Load, validate, and fingerprint a forecast TOML file."""

    merged: dict[str, Any] = {}
    sources: tuple[Path, ...] = ()
    for default_path in defaults:
        raw, loaded_paths = _read_with_extends(Path(default_path))
        merged = deep_merge(merged, raw)
        sources += loaded_paths

    raw, loaded_paths = _read_with_extends(Path(path))
    merged = deep_merge(merged, raw)
    sources += loaded_paths
    config = ForecastConfig.model_validate(merged)
    return LoadedForecastConfig(
        raw=merged,
        config=config,
        config_hash=forecast_config_fingerprint(config),
        source_paths=tuple(dict.fromkeys(sources)),
    )


def load_backtest_config(
    path: str | os.PathLike[str],
    defaults: Iterable[str | os.PathLike[str]] = (),
) -> LoadedBacktestConfig:
    """Load, validate, and fingerprint a backtest TOML file."""

    merged: dict[str, Any] = {}
    sources: tuple[Path, ...] = ()
    for default_path in defaults:
        raw, loaded_paths = _read_with_extends(Path(default_path))
        merged = deep_merge(merged, raw)
        sources += loaded_paths
    raw, loaded_paths = _read_with_extends(Path(path))
    merged = deep_merge(merged, raw)
    sources += loaded_paths
    config = BacktestConfig.model_validate(merged)
    return LoadedBacktestConfig(
        raw=merged,
        config=config,
        config_hash=backtest_config_fingerprint(config),
        source_paths=tuple(dict.fromkeys(sources)),
    )


def load_update_config(
    path: str | os.PathLike[str],
    defaults: Iterable[str | os.PathLike[str]] = (),
) -> LoadedUpdateConfig:
    """Load defaults and an update TOML, then validate and fingerprint it."""
    merged: dict[str, Any] = {}
    sources: tuple[Path, ...] = ()
    for default_path in defaults:
        raw, loaded_paths = _read_with_extends(Path(default_path))
        merged = deep_merge(merged, raw)
        sources += loaded_paths
    raw, loaded_paths = _read_with_extends(Path(path))
    merged = deep_merge(merged, raw)
    sources += loaded_paths
    config = UpdateConfig.model_validate(merged)
    return LoadedUpdateConfig(
        raw=merged,
        config=config,
        config_hash=update_config_fingerprint(config),
        source_paths=tuple(dict.fromkeys(sources)),
    )
