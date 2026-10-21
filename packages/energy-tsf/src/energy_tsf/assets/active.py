"""Read-only Active pointer resolution for the forecast track."""

from __future__ import annotations

import os
from pathlib import Path

from energy_tsf.contracts.artifact import ActiveRef


def active_pointer_path(root: str | os.PathLike[str], station_id: str, target: str, horizon: str) -> Path:
    """Return the Active pointer path for a station/target/horizon key."""
    return Path(root) / "active" / station_id / target / horizon / "pointer.json"


def read_active(root: str | os.PathLike[str], station_id: str, target: str, horizon: str) -> ActiveRef:
    """Load and validate the Active pointer and its referenced Solution."""
    path = active_pointer_path(root, station_id, target, horizon)
    if not path.is_file():
        raise FileNotFoundError(f"no Active solution for {station_id}/{target}/{horizon}")
    active = ActiveRef.model_validate_json(path.read_text(encoding="utf-8"))
    if (active.station_id, active.target, active.horizon) != (station_id, target, horizon):
        raise ValueError("Active pointer contents do not match its path")
    solution_path = Path(active.solution_path)
    if not solution_path.is_absolute():
        solution_path = (Path(root).resolve().parent / solution_path).resolve()
    if not (solution_path / ".complete").is_file():
        raise ValueError("Active pointer targets a missing or incomplete solution")
    marker = (solution_path / ".complete").read_text(encoding="utf-8").splitlines()
    if not marker or marker[0] != active.solution_id:
        raise ValueError("Active pointer solution id does not match its completion marker")
    return active


def list_active(
    root: str | os.PathLike[str],
    *,
    target: str | None = None,
    horizon: str | None = None,
) -> tuple[ActiveRef, ...]:
    """List complete Active pointers without consulting Candidate storage."""

    active_root = Path(root) / "active"
    if not active_root.is_dir():
        return ()
    found: list[ActiveRef] = []
    for pointer in sorted(active_root.glob("*/*/*/pointer.json")):
        station_id, pointer_target, pointer_horizon, _ = pointer.relative_to(active_root).parts
        if target is not None and pointer_target != target:
            continue
        if horizon is not None and pointer_horizon != horizon:
            continue
        found.append(read_active(root, station_id, pointer_target, pointer_horizon))
    return tuple(found)
