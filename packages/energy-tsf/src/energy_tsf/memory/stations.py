"""Small explicit station-profile repository used by cross-station retrieval."""

from __future__ import annotations

import os
import tempfile
import tomllib
from pathlib import Path

from energy_tsf.contracts.station import StationSpec


class StationRepository:
    def __init__(self, memory_root: str | os.PathLike[str]) -> None:
        """Store station profiles below the repository's ``stations`` folder."""
        self.root = Path(memory_root) / "stations"

    def _path(self, station_id: str) -> Path:
        return self.root / station_id / "profile.json"

    def save(self, station: StationSpec) -> Path:
        """Atomically write a validated station profile and completion marker."""
        path = self._path(station.station_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(prefix=".profile.", dir=path.parent)
        try:
            with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(station.model_dump_json(indent=2) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        (path.parent / ".complete").write_text(station.station_id + "\n", encoding="utf-8")
        return path

    def load(self, station_id: str) -> StationSpec:
        """Load one station profile and verify its ID matches its directory."""
        path = self._path(station_id)
        if not (path.parent / ".complete").is_file() or not path.is_file():
            raise FileNotFoundError(f"station profile not found: {station_id}")
        station = StationSpec.model_validate_json(path.read_text(encoding="utf-8"))
        if station.station_id != station_id:
            raise ValueError("station profile id does not match its directory")
        return station

    def list(self) -> tuple[StationSpec, ...]:
        """Return all registered station profiles in stable path order."""
        if not self.root.is_dir():
            return ()
        return tuple(self.load(path.parent.name) for path in sorted(self.root.glob("*/profile.json")))


def load_station_toml(path: str | os.PathLike[str]) -> StationSpec:
    """Parse a station TOML file and validate its ``[station]`` table."""
    with Path(path).open("rb") as stream:
        raw = tomllib.load(stream)
    if "station" not in raw:
        raise ValueError("station TOML must contain a [station] table")
    return StationSpec.model_validate(raw["station"])
