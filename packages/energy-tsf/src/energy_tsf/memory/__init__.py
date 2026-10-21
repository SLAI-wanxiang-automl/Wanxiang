"""Explicit repository facade for data, model objects, trials, and solutions."""

from energy_tsf.memory.repository import MemoryRepository
from energy_tsf.memory.cache import CachedExecutionRecipe, ExecutionCache, ExecutionShortcut
from energy_tsf.memory.retrieval import (
    ForecastAssetRetriever,
    build_forecast_plan,
    forecast_plan_fingerprint,
    validate_plan_candidates,
)
from energy_tsf.memory.solution import build_solution, solution_fingerprint
from energy_tsf.memory.search_pool import load_search_pool, save_search_pool
from energy_tsf.memory.stations import StationRepository, load_station_toml

__all__ = [
    "ForecastAssetRetriever",
    "CachedExecutionRecipe",
    "ExecutionCache",
    "ExecutionShortcut",
    "MemoryRepository",
    "StationRepository",
    "build_forecast_plan",
    "build_solution",
    "forecast_plan_fingerprint",
    "load_station_toml",
    "load_search_pool",
    "save_search_pool",
    "solution_fingerprint",
    "validate_plan_candidates",
]
