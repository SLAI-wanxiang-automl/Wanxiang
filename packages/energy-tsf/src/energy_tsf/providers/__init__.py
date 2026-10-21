"""Finite, capability-driven model provider collection."""

from energy_tsf.providers.base import (
    ForecastProvider,
    ProviderCapabilities,
    ProviderParams,
    TrainableProvider,
)
from energy_tsf.providers.baseline import SeasonalNaiveProvider
from energy_tsf.providers.registry import ProviderRegistry


def reference_registry() -> ProviderRegistry:
    registry = ProviderRegistry()
    registry.register("seasonal-naive", SeasonalNaiveProvider)
    return registry


def modeling_registry() -> ProviderRegistry:
    """Register the eight supported modeling providers."""

    from energy_tsf.providers.classical import ARIMAProvider, RidgeProvider, SARIMAProvider
    from energy_tsf.providers.foundation import ChronosProvider, MoiraiProvider, TimesFMProvider
    from energy_tsf.providers.ml import LightGBMProvider, XGBoostProvider

    registry = ProviderRegistry()
    for provider in (
        ARIMAProvider,
        SARIMAProvider,
        RidgeProvider,
        LightGBMProvider,
        XGBoostProvider,
        TimesFMProvider,
        MoiraiProvider,
        ChronosProvider,
    ):
        registry.register(provider.name, provider)
    return registry


__all__ = [
    "ForecastProvider",
    "ProviderCapabilities",
    "ProviderParams",
    "ProviderRegistry",
    "SeasonalNaiveProvider",
    "TrainableProvider",
    "reference_registry",
    "modeling_registry",
]
