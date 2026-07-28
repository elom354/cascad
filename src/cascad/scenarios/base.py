"""Scenario contracts shared by the reference simulator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ScenarioInstance:
    """Seeded surface realization of one fixed causal scenario."""

    instance_id: str
    instance_seed: int
    task: str
    tool_values: dict[str, dict[str, Any]]
    memory_default: Any
    clean_state: str | None = None
    memory_field: str = "authorization_state"
    notification_error: str = "the downstream action could not be completed with the persisted state"
    success_answer: str = "Document delivered."
    failure_answer: str = "Delivery failed."


@dataclass(frozen=True)
class Scenario:
    """A reproducible task topology and nominal values."""

    name: str
    nodes: tuple[str, ...]
    task: str
    tool_values: dict[str, dict[str, Any]]
    memory_default: Any = "recipient_prefers_pdf"
    attribution_variant: str | None = None
    distant_symptom: bool = False
    instance_generator: Callable[[int], ScenarioInstance] | None = None

    def instantiate(self, seed: int) -> ScenarioInstance:
        """Return a stable instance; fixed scenarios ignore surface variation."""
        if self.instance_generator:
            return self.instance_generator(seed)
        return ScenarioInstance(
            instance_id=f"{self.name}-fixed",
            instance_seed=seed,
            task=self.task,
            tool_values=self.tool_values,
            memory_default=self.memory_default,
        )
