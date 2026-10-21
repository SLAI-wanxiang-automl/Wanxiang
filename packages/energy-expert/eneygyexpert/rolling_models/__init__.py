"""Rolling model adapters for energy output forecasting."""

from .ensemble import MedianEnsembleAdapter
from .foundation import FoundationAdapter
from .neural import AutoNeuralForecastAdapter, NeuralForecastAdapter
from .prophet import ProphetAdapter
from .registry import (
    MODEL_SPECS,
    create_model,
    model_catalog,
    tune_grid,
)
from .sktime import SKTimeRollingAdapter
from .stats import NativeStatsAdapter, StatsForecastAdapter

__all__ = [
    "create_model",
    "tune_grid",
    "model_catalog",
    "MODEL_SPECS",
    "StatsForecastAdapter",
    "NativeStatsAdapter",
    "NeuralForecastAdapter",
    "AutoNeuralForecastAdapter",
    "ProphetAdapter",
    "FoundationAdapter",
    "MedianEnsembleAdapter",
    "SKTimeRollingAdapter",
]
