"""Small deterministic projections rebuilt from ledger events."""

from __future__ import annotations

from collections.abc import Iterable

from energy_tsf.ledger.events import LedgerEvent


def latest_task_states(events: Iterable[LedgerEvent]) -> dict[str, str]:
    """Project the most recent status event for each task ID."""
    states: dict[str, str] = {}
    for event in events:
        if event.task_id is not None and "status" in event.payload:
            states[event.task_id] = str(event.payload["status"])
    return states


def related_events(
    events: Iterable[LedgerEvent],
    *,
    task_id: str | None = None,
    trial_id: str | None = None,
    solution_id: str | None = None,
) -> tuple[LedgerEvent, ...]:
    """Filter ledger events by any supplied task, trial, or Solution ID."""
    return tuple(
        event
        for event in events
        if (task_id is None or event.task_id == task_id)
        and (trial_id is None or event.trial_id == trial_id)
        and (solution_id is None or event.solution_id == solution_id)
    )
