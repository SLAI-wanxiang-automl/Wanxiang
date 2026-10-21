"""Atomic Candidate promotion and one-step rollback."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from energy_tsf.assets.active import active_pointer_path, read_active
from energy_tsf.contracts.artifact import ActiveRef, CandidateRef, ReleaseDecision
from energy_tsf.memory.repository import MemoryRepository

_PROMOTION_LOCK = threading.Lock()


@contextmanager
def _cross_process_lock(pointer: Path) -> Iterator[None]:
    """Serialize pointer read/compare/write across threads and processes."""
    lock_path = pointer.parent / ".promotion.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with _PROMOTION_LOCK, lock_path.open("a+b") as stream:
        if os.name == "nt":
            import msvcrt

            stream.seek(0, os.SEEK_END)
            if stream.tell() == 0:
                # Initialize before locking: reading a byte already locked by
                # another process can itself raise PermissionError on Windows.
                stream.write(b"0")
                stream.flush()
            while True:
                stream.seek(0)
                try:
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(0.05)
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            # POSIX uses flock; Windows takes the msvcrt branch above.
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)  # type: ignore[attr-defined]
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)  # type: ignore[attr-defined]


def _atomic_pointer(path: Path, active: ActiveRef) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=".pointer.", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(active.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def promote_candidate(
    repository: MemoryRepository,
    candidate: CandidateRef,
    decision: ReleaseDecision,
    *,
    expected_active_solution_id: str | None = None,
) -> ActiveRef:
    """Atomically promote an approved Candidate to the Active pointer.

    ``expected_active_solution_id`` provides a compare-and-swap guard so a
    stale backtest or update cannot overwrite a newer publication.
    """
    if not decision.approved:
        raise ValueError("Candidate did not pass the release gate: " + "; ".join(decision.reasons))
    candidate = repository.load_candidate(candidate.path)
    solution = repository.load_solution(candidate.solution_id)
    pointer = active_pointer_path(
        repository.root,
        solution.station_id,
        solution.compatibility.target,
        solution.compatibility.horizon,
    )
    with _cross_process_lock(pointer):
        previous = None
        if pointer.is_file():
            previous = read_active(
                repository.root,
                solution.station_id,
                solution.compatibility.target,
                solution.compatibility.horizon,
            ).solution_id
        if expected_active_solution_id is not None and previous != expected_active_solution_id:
            raise ValueError("Active solution changed after the update plan was frozen")
        active = ActiveRef(
            station_id=solution.station_id,
            target=solution.compatibility.target,
            horizon=solution.compatibility.horizon,
            solution_id=solution.solution_id,
            solution_path=str((repository.root / "solutions" / solution.solution_id).resolve()),
            previous_solution_id=previous,
        )
        _atomic_pointer(pointer, active)
    history = pointer.parent / "history"
    history.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    (history / f"{stamp}-{solution.solution_id[:12]}.json").write_text(
        json.dumps(active.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return active


def rollback_active(repository: MemoryRepository, station_id: str, target: str, horizon: str) -> ActiveRef:
    """Restore the previous Active Solution while holding the pointer lock."""
    pointer = active_pointer_path(repository.root, station_id, target, horizon)
    with _cross_process_lock(pointer):
        current = read_active(repository.root, station_id, target, horizon)
        if current.previous_solution_id is None:
            raise ValueError("Active solution has no rollback target")
        previous = repository.load_solution(current.previous_solution_id)
        restored = ActiveRef(
            station_id=station_id,
            target=target,
            horizon=horizon,
            solution_id=previous.solution_id,
            solution_path=str((repository.root / "solutions" / previous.solution_id).resolve()),
            previous_solution_id=current.solution_id,
        )
        _atomic_pointer(pointer, restored)
        return restored
