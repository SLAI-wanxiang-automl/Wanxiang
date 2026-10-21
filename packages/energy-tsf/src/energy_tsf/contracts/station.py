"""Station metadata required by deterministic retrieval and transfer."""

from __future__ import annotations

from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field, field_validator, model_validator

from energy_tsf.contracts.common import ContractModel
from energy_tsf.contracts.modeling import frequency_seconds

_CAPACITY_UNITS = {"w": "W", "kw": "kW", "mw": "MW", "gw": "GW"}
_TARGET_UNITS = {**_CAPACITY_UNITS, "pu": "p.u.", "p.u.": "p.u."}
_POWER_FACTORS = {"W": 1.0, "kW": 1_000.0, "MW": 1_000_000.0, "GW": 1_000_000_000.0}


def power_unit_factor(unit: str) -> float | None:
    """Return a conversion factor to watts for supported power units."""

    canonical = _CAPACITY_UNITS.get(unit.strip().lower(), unit.strip())
    return _POWER_FACTORS.get(canonical)


def is_per_unit(unit: str) -> bool:
    return unit.strip().lower() in {"pu", "p.u."}


def capacity_in_target_unit(station: "StationSpec") -> float | None:
    if station.capacity is None or is_per_unit(station.target_unit):
        return 1.0 if is_per_unit(station.target_unit) else None
    capacity_factor = power_unit_factor(station.capacity_unit or "")
    target_factor = power_unit_factor(station.target_unit)
    if capacity_factor is None or target_factor is None:
        return None
    return station.capacity * capacity_factor / target_factor


class StationSpec(ContractModel):
    """Stable facts about the station receiving or supplying a forecast."""

    station_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    energy_type: str = Field(min_length=1)
    timezone: str = Field(min_length=1)
    target: str = Field(min_length=1)
    target_unit: str = Field(min_length=1)
    frequency: str
    timestamp_column: str | None = None
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    technology: str | None = None
    capacity: float | None = Field(default=None, gt=0)
    capacity_unit: str | None = None
    tags: dict[str, str] = Field(default_factory=dict)

    @field_validator("energy_type", "target")
    @classmethod
    def normalize_labels(cls, value: str) -> str:
        return value.strip().lower()

    @field_validator("frequency")
    @classmethod
    def validate_frequency(cls, value: str) -> str:
        value = value.strip().lower()
        frequency_seconds(value)
        return value

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"unknown IANA timezone: {value}") from exc
        return value

    @field_validator("timestamp_column", "technology")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        return value or None

    @field_validator("capacity_unit")
    @classmethod
    def normalize_capacity_unit(cls, value: str | None) -> str | None:
        if value is None:
            return None
        canonical = _CAPACITY_UNITS.get(value.strip().lower())
        if canonical is None:
            raise ValueError("capacity_unit must be W, kW, MW, or GW")
        return canonical

    @field_validator("target_unit")
    @classmethod
    def normalize_target_unit(cls, value: str) -> str:
        stripped = value.strip()
        return _TARGET_UNITS.get(stripped.lower(), stripped)

    @model_validator(mode="after")
    def validate_paired_metadata(self) -> "StationSpec":
        if (self.latitude is None) != (self.longitude is None):
            raise ValueError("latitude and longitude must be provided together")
        if (self.capacity is None) != (self.capacity_unit is None):
            raise ValueError("capacity and capacity_unit must be provided together")
        return self
