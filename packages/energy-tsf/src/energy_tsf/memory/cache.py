"""Recoverable short-term cache for the latest executable forecast recipe."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from pydantic import Field, model_validator

from energy_tsf.assets.active import read_active
from energy_tsf.contracts.artifact import SolutionArtifact
from energy_tsf.contracts.common import ContractModel
from energy_tsf.memory.repository import MemoryRepository
from energy_tsf.memory.solution import solution_fingerprint


class ExecutionShortcut(ContractModel):
    member_trial_ids: tuple[str, ...]
    providers: tuple[str, ...]
    artifact_refs: tuple[str, ...]
    weights: tuple[float, ...]
    constraints: dict


class CachedExecutionRecipe(ContractModel):
    station_id: str
    target: str
    horizon: str
    solution_id: str
    cached_at: str
    solution: SolutionArtifact
    shortcut: ExecutionShortcut
    last_issue_time: str | None = None
    last_input_fingerprint: str | None = None
    format_version: int = Field(default=2, ge=1)

    @model_validator(mode="after")
    def validate_shortcut(self) -> "CachedExecutionRecipe":
        expected = tuple(member.trial_id for member in self.solution.members)
        if self.shortcut.member_trial_ids != expected:
            raise ValueError("cached shortcut does not match Solution members")
        if self.shortcut.weights != self.solution.ensemble.weights:
            raise ValueError("cached shortcut does not match Solution weights")
        return self


class ExecutionCache:
    """Cache full immutable Solution content; long-term memory remains authoritative."""

    def __init__(self, memory_root: str | os.PathLike[str]) -> None:
        self.root = Path(memory_root)

    def _path(self, station_id: str, target: str, horizon: str) -> Path:
        return self.root / "short_term_cache" / station_id / target / horizon / "recipe.json"

    @staticmethod
    def _shortcut(solution: SolutionArtifact) -> ExecutionShortcut:
        return ExecutionShortcut(
            member_trial_ids=tuple(member.trial_id for member in solution.members),
            providers=tuple(member.provider for member in solution.members),
            artifact_refs=tuple(member.artifact_ref for member in solution.members),
            weights=solution.ensemble.weights,
            constraints=solution.constraints,
        )

    def _load(self, path: Path) -> CachedExecutionRecipe:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if "shortcut" not in raw:
            solution = SolutionArtifact.model_validate(raw["solution"])
            raw["shortcut"] = self._shortcut(solution).model_dump(mode="json")
            raw["format_version"] = 2
        recipe = CachedExecutionRecipe.model_validate(raw)
        if solution_fingerprint(recipe.solution) != recipe.solution_id:
            raise ValueError("cached Solution fingerprint does not match its contents")
        return recipe

    def _write(self, path: Path, recipe: CachedExecutionRecipe) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(prefix=".recipe.", dir=path.parent)
        try:
            with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(recipe.model_dump(mode="json"), ensure_ascii=False, indent=2))
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return path

    def get(self, station_id: str, target: str, horizon: str) -> SolutionArtifact | None:
        """Load a matching cached Solution, ignoring corrupt shortcuts safely."""
        path = self._path(station_id, target, horizon)
        if not path.is_file():
            return None
        try:
            recipe = self._load(path)
        except (OSError, ValueError, json.JSONDecodeError):
            return None
        expected = (station_id, target, horizon)
        if (recipe.station_id, recipe.target, recipe.horizon) != expected:
            raise ValueError("cached execution recipe does not match its path")
        if recipe.solution.solution_id != recipe.solution_id:
            raise ValueError("cached execution recipe has mismatched Solution content")
        return recipe.solution

    def get_by_solution_id(self, solution_id: str) -> SolutionArtifact | None:
        """Find a cached Solution by its immutable ID across station keys."""
        cache_root = self.root / "short_term_cache"
        if not cache_root.is_dir():
            return None
        for path in cache_root.glob("*/*/*/recipe.json"):
            try:
                recipe = self._load(path)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if recipe.solution_id == solution_id:
                return recipe.solution
        return None

    def put(
        self,
        solution: SolutionArtifact,
        *,
        issue_time: str | None = None,
        input_fingerprint: str | None = None,
    ) -> Path:
        """Cache a Solution and its validated inference shortcut recipe."""
        path = self._path(
            solution.station_id,
            solution.compatibility.target,
            solution.compatibility.horizon,
        )
        recipe = CachedExecutionRecipe(
            station_id=solution.station_id,
            target=solution.compatibility.target,
            horizon=solution.compatibility.horizon,
            solution_id=solution.solution_id,
            cached_at=datetime.now(timezone.utc).isoformat(),
            solution=solution,
            shortcut=self._shortcut(solution),
            last_issue_time=issue_time,
            last_input_fingerprint=input_fingerprint,
        )
        return self._write(path, recipe)

    def touch(
        self,
        station_id: str,
        target: str,
        horizon: str,
        *,
        issue_time: str | None,
        input_fingerprint: str,
    ) -> None:
        """Refresh cache recency and the latest input metadata when present."""
        path = self._path(station_id, target, horizon)
        if not path.is_file():
            return
        try:
            current = self._load(path)
        except (OSError, ValueError, json.JSONDecodeError):
            return
        recipe = current.model_copy(
            update={
                "cached_at": datetime.now(timezone.utc).isoformat(),
                "last_issue_time": issue_time,
                "last_input_fingerprint": input_fingerprint,
            }
        )
        self._write(path, recipe)

    def invalidate(self, station_id: str, target: str, horizon: str) -> None:
        """Remove the shortcut for a published station/target/horizon key."""
        path = self._path(station_id, target, horizon)
        if path.is_file():
            path.unlink()

    def active_solution(
        self,
        repository: MemoryRepository,
        station_id: str,
        target: str,
        horizon: str,
    ) -> SolutionArtifact:
        """Return cached data only when it matches the authoritative Active ID."""
        active = read_active(repository.root, station_id, target, horizon)
        cached = self.get(station_id, target, horizon)
        if cached is not None and cached.solution_id == active.solution_id:
            return cached
        solution = repository.load_solution(active.solution_id)
        self.put(solution)
        return solution
