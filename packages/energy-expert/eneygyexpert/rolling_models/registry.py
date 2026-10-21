"""Lazy factories and an explicit first-version model capability catalog."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from .foundation import FOUNDATION_MODELS, FoundationAdapter
from .ml import ML_MODELS, create_ml_model
from .neural import NEURAL_MODELS, AutoNeuralForecastAdapter, NeuralForecastAdapter
from .prophet import ProphetAdapter
from .stats import STATE_MODELS, STATS_MODELS, NativeStatsAdapter, StatsForecastAdapter


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    historical_covariates: bool
    future_covariates: bool
    update_mode: str
    tuning: str
    important_parameters: tuple[str, ...]


_SPECS = [
    ModelSpec(n, "MLForecast", False, False, "frozen", "native", p)
    for n, p in {
        "AutoLinearRegression": ("lags", "lag_transforms", "fit_intercept"),
        "AutoRidge": ("lags", "lag_transforms", "alpha"),
        "AutoLasso": ("lags", "lag_transforms", "alpha"),
        "AutoElasticNet": ("lags", "lag_transforms", "alpha", "l1_ratio"),
        "AutoRandomForest": ("lags", "n_estimators", "max_depth", "min_samples_leaf"),
        "AutoLGBM": (
            "lags",
            "learning_rate",
            "num_leaves",
            "min_child_samples",
            "n_estimators",
        ),
        "AutoXGBoost": (
            "lags",
            "learning_rate",
            "max_depth",
            "n_estimators",
            "reg_lambda",
        ),
        "AutoCatboost": ("lags", "learning_rate", "depth", "iterations", "l2_leaf_reg"),
    }.items()
]
_SPECS += [
    ModelSpec(
        n,
        "NeuralForecast",
        n in ("AutoNHITS", "AutoTFT"),
        n in ("AutoNHITS", "AutoTFT", "AutoDeepAR"),
        "frozen",
        "native",
        p,
    )
    for n, p in {
        "AutoPatchTST": (
            "input_size",
            "patch_len",
            "stride",
            "hidden_size",
            "n_heads",
            "encoder_layers",
            "dropout",
            "learning_rate",
            "max_steps",
        ),
        "AutoNBEATS": (
            "input_size",
            "n_blocks",
            "mlp_units",
            "learning_rate",
            "max_steps",
        ),
        "AutoNHITS": (
            "input_size",
            "n_pool_kernel_size",
            "n_freq_downsample",
            "mlp_units",
            "learning_rate",
            "max_steps",
        ),
        "AutoTFT": (
            "input_size",
            "hidden_size",
            "n_head",
            "dropout",
            "learning_rate",
            "max_steps",
        ),
        "AutoDeepAR": (
            "input_size",
            "lstm_hidden_size",
            "lstm_n_layers",
            "lstm_dropout",
            "learning_rate",
            "max_steps",
        ),
    }.items()
]
_SPECS += [
    ModelSpec(
        n,
        "StatsForecast",
        False,
        False,
        "state_update" if n in STATE_MODELS else "frozen",
        "native"
        if n.startswith("Auto")
        else ("framework" if "Theta" in n else "fixed"),
        p,
    )
    for n, p in {
        "AutoARIMA": (
            "season_length",
            "order",
            "seasonal_order",
            "include_mean",
            "include_drift",
        ),
        "AutoETS": ("season_length", "model", "damped"),
        "AutoCES": ("season_length", "model"),
        "Theta": ("season_length", "decomposition_type"),
        "DynamicOptimizedTheta": ("season_length", "decomposition_type"),
        "SeasonalNaive": ("season_length",),
        "HistoricAverage": (),
        "ZeroModel": (),
        "CrostonClassic": (),
        "ADIDA": (),
        "IMAPA": (),
    }.items()
]
_SPECS += [
    ModelSpec(
        "Prophet",
        "Prophet",
        False,
        True,
        "frozen",
        "framework",
        ("changepoint_prior_scale", "seasonality_prior_scale", "seasonality_mode"),
    )
]
_SPECS += [
    ModelSpec(
        n,
        "Foundation",
        False,
        False,
        "frozen",
        "fixed",
        ("checkpoint/model", "context_length", "inference configuration"),
    )
    for n in FOUNDATION_MODELS
]
MODEL_SPECS = {s.name: s for s in _SPECS}


def model_catalog():
    from dataclasses import asdict

    import pandas as pd

    return pd.DataFrame([asdict(s) for s in _SPECS])


def create_model(
    name,
    *,
    h=None,
    season_length=96,
    tuning=True,
    params=None,
    alias=None,
    historical_covariate_cols=(),
    future_covariate_cols=(),
    **kwargs,
):
    """Create adapters without importing optional backends until needed.

    Variable roles are explicit. Unsupported covariates raise rather than
    silently becoming available to a model. Ignoring extra dataset variables
    is achieved by leaving these tuples empty.
    """
    if name not in MODEL_SPECS:
        raise ValueError(f"Unknown model {name}; available: {sorted(MODEL_SPECS)}")
    spec = MODEL_SPECS[name]
    if historical_covariate_cols and not spec.historical_covariates:
        raise ValueError(f"{name} cannot consume historical covariates in this adapter")
    if future_covariate_cols and not spec.future_covariates:
        raise ValueError(f"{name} cannot consume future covariates in this adapter")
    alias = alias or name
    if name in ML_MODELS:
        if not tuning:
            raise ValueError(
                "Use the fixed ML adapter returned by tune() with its lag settings"
            )
        if params:
            raise ValueError("For custom ML spaces, configure AutoModel directly")
        return create_ml_model(name, season_length=season_length, alias=alias, **kwargs)
    if name in NEURAL_MODELS:
        if h is None:
            raise ValueError("Neural models require h when constructed")
        cls = AutoNeuralForecastAdapter if tuning else NeuralForecastAdapter
        return cls(
            name,
            h=h,
            config=params if params is not None else (None if tuning else {}),
            alias=alias,
            historical_covariate_cols=historical_covariate_cols,
            future_covariate_cols=future_covariate_cols,
            **kwargs,
        )
    if name in STATS_MODELS:
        settings = dict(params or {})
        if name in STATE_MODELS or name == "SeasonalNaive":
            settings.setdefault("season_length", season_length)
        if name.startswith("Auto") and tuning:
            return NativeStatsAdapter(name, params=settings, alias=alias, **kwargs)
        if name.startswith("Auto") and not tuning:
            if name == "AutoARIMA" and "order" not in settings:
                raise ValueError("Fixed AutoARIMA requires order and seasonal_order")
            if name in ("AutoETS", "AutoCES") and "Z" in settings.get("model", "Z"):
                raise ValueError(
                    "Fixed ETS/CES requires a selected structure without Z"
                )
        if "Theta" in name and tuning:
            return tune_grid(
                name,
                h=h,
                season_length=season_length,
                alias=alias,
                grid={"decomposition_type": ("additive", "multiplicative")},
                base_params=settings,
            )
        return StatsForecastAdapter(name, params=settings, alias=alias, **kwargs)
    if name == "Prophet":
        if tuning:
            return tune_grid(
                name,
                h=h,
                alias=alias,
                grid={
                    "changepoint_prior_scale": (0.01, 0.1),
                    "seasonality_mode": ("additive", "multiplicative"),
                },
                base_params=params,
                future_covariate_cols=future_covariate_cols,
            )
        return ProphetAdapter(
            params=params,
            alias=alias,
            future_covariate_cols=future_covariate_cols,
            **kwargs,
        )
    return FoundationAdapter(name, alias=alias, model_kwargs=params, **kwargs)


def tune_grid(name, *, h, grid, alias=None, base_params=None, **kwargs):
    """Declare finite parameter spaces for models without native Auto search."""
    from ..tuning import GridTunedWindowModel

    if h is None:
        raise ValueError("Framework tuning requires h")
    if not grid or any(not choices for choices in grid.values()):
        raise ValueError("Parameter grid must contain nonempty axes")
    candidates = []
    for i, values in enumerate(product(*grid.values())):
        params = {**(base_params or {}), **dict(zip(grid, values, strict=True))}
        candidates.append(
            create_model(
                name,
                h=h,
                tuning=False,
                params=params,
                alias=f"{alias or name}_option_{i}",
                **kwargs,
            )
        )
    return GridTunedWindowModel(candidates, alias=alias or name, inner_h=h)
