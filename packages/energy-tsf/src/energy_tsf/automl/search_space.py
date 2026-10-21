"""Load small explicit provider search spaces from TOML."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, Iterable

from pydantic import Field

from energy_tsf.contracts.common import ContractModel


class SearchCandidate(ContractModel):
    params: dict[str, Any] = Field(default_factory=dict)
    feature: dict[str, Any] = Field(default_factory=dict)
    context_days: int | None = Field(default=None, gt=0)


class SearchSpace(ContractModel):
    provider: str
    max_trials: int = Field(gt=0)
    candidates: tuple[SearchCandidate, ...]


def load_search_spaces(paths: Iterable[str | Path]) -> dict[str, SearchSpace]:
    """Load and validate Provider search spaces from TOML files."""
    spaces: dict[str, SearchSpace] = {}
    for path in paths:
        with Path(path).open("rb") as stream:
            raw = tomllib.load(stream)
        for name, definition in raw.get("providers", {}).items():
            if name in spaces:
                raise ValueError(f"duplicate provider search space: {name}")
            spaces[name] = SearchSpace(provider=name, **definition)
    return spaces


def load_default_search_spaces(config_dir: str | Path | None = None) -> dict[str, SearchSpace]:
    """Load the repository's classical, ML, and foundation search spaces."""
    root = Path(config_dir) if config_dir is not None else Path(__file__).parents[3] / "configs" / "providers"
    return load_search_spaces(root / name for name in ("classical.toml", "ml.toml", "foundation.toml"))
