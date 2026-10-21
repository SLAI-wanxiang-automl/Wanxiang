"""Small capability-driven model registry; it never registers Agents."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from energy_tsf.providers.base import ForecastProvider

ProviderFactory = Callable[[], ForecastProvider]


class ProviderRegistry:
    """Create named Provider instances and report their runtime availability."""

    def __init__(self) -> None:
        """Initialize an empty registry; providers are added explicitly."""
        self._factories: dict[str, ProviderFactory] = {}

    def register(self, name: str, factory: ProviderFactory) -> None:
        """Register a unique name after checking the factory's declared name."""
        key = name.strip().lower()
        if key in self._factories:
            raise ValueError(f"provider already registered: {key}")
        provider = factory()
        if provider.name != key:
            raise ValueError("provider factory name does not match registry key")
        self._factories[key] = factory

    def create(self, name: str) -> ForecastProvider:
        """Create a fresh Provider instance by its normalized registry name."""
        key = name.strip().lower()
        try:
            return self._factories[key]()
        except KeyError as exc:
            raise KeyError(f"unknown provider: {key}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(self._factories)

    def availability(self, names: Iterable[str] | None = None) -> dict[str, tuple[bool, str | None]]:
        """Return availability and optional failure reasons for selected providers."""
        selected = self.names() if names is None else tuple(names)
        return {name: self.create(name).available() for name in selected}
