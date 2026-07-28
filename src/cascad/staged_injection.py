"""Stage-oriented injection baseline inspired by AgentProp-Bench."""

from __future__ import annotations

from dataclasses import dataclass

from cascad.models import EventKind, RunTrace


@dataclass(frozen=True)
class StagedPropagation:
    injected: bool
    executed_with_corruption: bool
    final_response_incorrect: bool

    @property
    def propagation_probability(self) -> float:
        return float(self.injected and self.executed_with_corruption and self.final_response_incorrect)


def stage_indicators(trace: RunTrace) -> StagedPropagation:
    """Extract S1 -> S2 -> S3 indicators from a staged-injection run."""
    injected = any(event.kind == EventKind.FAULT_INJECTED and "args" in event.node_id for event in trace.events)
    executed = any(
        event.kind == EventKind.TOOL_CALL
        and (event.payload.get("ok") is False or not isinstance(event.payload.get("arguments", {}).get("task"), str))
        for event in trace.events
    )
    answers = [event.payload.get("answer", "") for event in trace.events if event.node_id == "responder"]
    incorrect = any("failed" in str(answer).lower() or "blocked" in str(answer).lower() for answer in answers)
    return StagedPropagation(injected, executed, incorrect)
