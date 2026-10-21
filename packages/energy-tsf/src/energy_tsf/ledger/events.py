"""Public append-only ledger event contract."""

from __future__ import annotations

from typing import Any

from pydantic import Field, model_validator

from energy_tsf.contracts.common import ContractModel

_PRIVATE_KEYS = {"chain_of_thought", "private_thought", "reasoning_trace", "agent_scratchpad"}


class LedgerEvent(ContractModel):
    event_id: str = Field(min_length=1)
    event_type: str = Field(min_length=1)
    timestamp: str
    task_id: str | None = None
    trial_id: str | None = None
    solution_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def reject_private_reasoning(self) -> "LedgerEvent":
        def visit(value: Any) -> None:
            if isinstance(value, dict):
                for key, item in value.items():
                    if key.lower() in _PRIVATE_KEYS:
                        raise ValueError("ledger cannot store Agent private reasoning")
                    visit(item)
            elif isinstance(value, list):
                for item in value:
                    visit(item)

        visit(self.payload)
        return self
