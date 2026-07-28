"""Tracing primitives for instrumenting agent nodes."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from cascad.models import CausalEdge, EventKind, NodeEvent, RunTrace


class CascadTracer:
    """Collect node events and causal relations for one agent run."""

    def __init__(self, run_id: str | None = None, **metadata: Any) -> None:
        self.trace = RunTrace(metadata=metadata)
        if run_id:
            self.trace.run_id = run_id
        self._stack: list[NodeEvent] = []

    @property
    def run_id(self) -> str:
        """Return the current run id."""
        return self.trace.run_id

    def emit(
        self,
        node_id: str,
        kind: EventKind | str,
        payload: dict[str, Any] | None = None,
        tags: set[str] | None = None,
    ) -> NodeEvent:
        """Emit one event."""
        parent_event_id = self._stack[-1].event_id if self._stack else None
        event = NodeEvent(
            node_id=node_id,
            kind=kind,
            run_id=self.run_id,
            parent_event_id=parent_event_id,
            payload=payload or {},
            tags=tags or set(),
        )
        return self.trace.add_event(event)

    @contextmanager
    def span(self, node_id: str, **payload: Any) -> Iterator[NodeEvent]:
        """Instrument a node execution span."""
        start = self.emit(node_id, EventKind.NODE_START, payload)
        self._stack.append(start)
        try:
            yield start
        except Exception as exc:
            self.emit(
                node_id,
                EventKind.ERROR_OBSERVED,
                {"exception_type": exc.__class__.__name__, "message": str(exc)},
                {"exception"},
            )
            raise
        finally:
            self._stack.pop()
            self.emit(node_id, EventKind.NODE_END, {"started_by": start.event_id})

    def link(
        self,
        source: str,
        target: str,
        relation: str = "data_dependency",
        weight: float = 1.0,
        evidence: str = "",
    ) -> CausalEdge:
        """Add a causal link."""
        edge = CausalEdge(
            source=source,
            target=target,
            relation=relation,  # type: ignore[arg-type]
            weight=weight,
            evidence=evidence,
        )
        return self.trace.add_edge(edge)

