"""Save model capabilities, selected settings, metrics, and optional checkpoints."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path

import pandas as pd

from .registry import model_catalog


def _jsonable(value):
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {
            str(k): (
                "[redacted]"
                if any(
                    term in str(k).lower()
                    for term in ("api_key", "password", "token", "secret")
                )
                else _jsonable(v)
            )
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is None or isinstance(value, (str, float, int, bool)):
        return value
    if hasattr(value, "item"):
        return value.item()
    return repr(value)


def save_pipeline_result(result, directory):
    """Write inspectable reports. Model weights remain in result.*.fitted_model."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model_catalog().to_csv(directory / "model_catalog.csv", index=False)
    result.leaderboard.to_csv(directory / "validation_scores.csv", index=False)
    result.test.summary.to_csv(directory / "test_scores.csv", index=False)
    result.test.forecasts.to_csv(directory / "test_forecasts.csv", index=False)
    result.test.by_lead.to_csv(directory / "test_by_lead.csv", index=False)
    for alias, validation in result.validation.items():
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in alias)
        validation.forecasts.to_csv(directory / f"validation_{safe}.csv", index=False)
    if result.test_results:
        pd.concat(
            [r.summary for r in result.test_results.values()], ignore_index=True
        ).to_csv(directory / "test_comparison_scores.csv", index=False)
        pd.concat(
            [r.forecasts for r in result.test_results.values()], ignore_index=True
        ).to_csv(directory / "test_comparison_forecasts.csv", index=False)
    metadata = dict(
        boundaries=result.boundaries,
        inner_n_windows=result.inner_n_windows,
        selected_model=result.selected_model,
        selected_models=result.selected_models,
        ensemble_weights=result.ensemble_weights,
        tuning_logs=result.tuning_logs,
        models={
            name: dict(
                params=model.get_params() if hasattr(model, "get_params") else {},
                capabilities=asdict(model.capabilities),
            )
            for name, model in result.fixed_models.items()
        },
    )
    (directory / "selection.json").write_text(
        json.dumps(_jsonable(metadata), ensure_ascii=False, indent=2)
    )
    return directory


def save_fitted_model(model, path):
    """Persist a local fitted object including its current rolling state.

    Load only trusted checkpoints with joblib.load. Foundation/service clients
    are excluded: their checkpoint reference is saved in selection.json.
    """
    from ..tuning import FixedAliasWindowModel
    from .ensemble import FittedMedianEnsemble
    from .foundation import FittedFoundation

    def contains_foundation(source):
        if isinstance(source, FixedAliasWindowModel):
            return contains_foundation(source.selected)
        if isinstance(source, FittedMedianEnsemble):
            return any(contains_foundation(member) for member in source.models)
        return isinstance(source, FittedFoundation)

    if contains_foundation(model):
        raise ValueError(
            "Save foundation checkpoint references instead of model clients"
        )
    import joblib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path
