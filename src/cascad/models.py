"""Core data models for Cascad."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from time import perf_counter
from typing import Any, Literal
from uuid import uuid4


class EventKind(str, Enum):
    """Kinds of events emitted by an agent instance."""

    NODE_START = "node_start"
    NODE_END = "node_end"
    TOOL_CALL = "tool_call"
    MEMORY_READ = "memory_read"
    MEMORY_WRITE = "memory_write"
    FAULT_INJECTED = "fault_injected"
    ERROR_OBSERVED = "error_observed"
    INTERVENTION = "intervention"


class FaultKind(str, Enum):
    """Controlled fault categories."""

    CORRUPT_VALUE = "corrupt_value"
    DROP_FIELD = "drop_field"
    DELAY = "delay"
    EXCEPTION = "exception"
    MEMORY_POISON = "memory_poison"
    TOOL_MISRESULT = "tool_misresult"
    STAGED_TOOL_ARGUMENT = "staged_tool_argument"


@dataclass(frozen=True)
class FaultSpec:
    """Description of a controlled fault injection."""

    target_node: str
    kind: FaultKind | str
    payload: dict[str, Any] = field(default_factory=dict)
    probability: float = 1.0
    seed: int | None = None
    label: str = "fault"

    @property
    def injection_node(self) -> str:
        """Ground-truth intervention source fixed before reconstruction."""
        return self.target_node


@dataclass
class NodeEvent:
    """A timestamped observation from one agent node/component."""

    node_id: str
    kind: EventKind | str
    run_id: str
    event_id: str = field(default_factory=lambda: str(uuid4()))
    parent_event_id: str | None = None
    timestamp: float = field(default_factory=perf_counter)
    payload: dict[str, Any] = field(default_factory=dict)
    tags: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class ErrorObservation:
    """Detected evidence that an error exists at a node."""

    node_id: str
    event_id: str
    detector: str
    confidence: float
    reason: str
    timestamp: float
    source_fault_id: str | None = None


@dataclass(frozen=True)
class CausalEdge:
    """Directed causal relation between two affected events or nodes."""

    source: str
    target: str
    relation: Literal[
        "data_dependency",
        "control_dependency",
        "memory_dependency",
        "tool_dependency",
        "temporal_dependency",
    ]
    weight: float = 1.0
    evidence: str = ""


@dataclass(frozen=True)
class InterventionDecision:
    """Decision produced by an intervention policy."""

    action: Literal[
        "continue",
        "quarantine",
        "rollback",
        "retry",
        "halt",
        "quarantine_memory_write",
        "discard_retrieved_memory",
        "block_final_action",
    ]
    node_id: str
    reason: str
    confidence: float = 1.0


@dataclass
class RunTrace:
    """Complete trace for one agent instance."""

    run_id: str = field(default_factory=lambda: str(uuid4()))
    session_id: str | None = None
    episode_id: int | None = None
    events: list[NodeEvent] = field(default_factory=list)
    observations: list[ErrorObservation] = field(default_factory=list)
    causal_edges: list[CausalEdge] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_event(self, event: NodeEvent) -> NodeEvent:
        """Append an event and return it."""
        self.events.append(event)
        return event

    def add_observation(self, observation: ErrorObservation) -> ErrorObservation:
        """Append an error observation and return it."""
        self.observations.append(observation)
        return observation

    def add_edge(self, edge: CausalEdge) -> CausalEdge:
        """Append a causal edge and return it."""
        self.causal_edges.append(edge)
        return edge

    def affected_nodes(self) -> set[str]:
        """Return nodes with detected errors."""
        return {observation.node_id for observation in self.observations}
