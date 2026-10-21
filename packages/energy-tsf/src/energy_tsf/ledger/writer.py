"""Process-local synchronized, append-only JSONL ledger."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from energy_tsf.ledger.events import LedgerEvent

_LOCK = threading.Lock()


class LedgerWriter:
    def __init__(self, memory_root: str | os.PathLike[str]) -> None:
        self.path = Path(memory_root) / "ledger" / "events.jsonl"

    def append(self, event: LedgerEvent) -> bool:
        """Append once; return False when event_id was already recorded."""

        with _LOCK:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.is_file():
                for existing in self.read():
                    if existing.event_id == event.event_id:
                        return False
            with self.path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(event.model_dump(mode="json"), ensure_ascii=False, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            return True

    def read(self) -> tuple[LedgerEvent, ...]:
        """Read and validate the append-only event stream in file order."""
        if not self.path.is_file():
            return ()
        return tuple(
            LedgerEvent.model_validate_json(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
