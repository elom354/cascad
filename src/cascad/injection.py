"""Controlled fault injection for agent simulations."""

from __future__ import annotations

import random
from copy import deepcopy
from typing import Any

from cascad.models import EventKind, FaultKind, FaultSpec
from cascad.tracer import CascadTracer


class FaultInjector:
    """Apply controlled faults at named agent nodes."""

    def __init__(self, specs: list[FaultSpec] | None = None) -> None:
        self.specs = specs or []

    def should_inject(self, spec: FaultSpec) -> bool:
        """Return whether this fault should fire."""
        rng = random.Random(spec.seed)
        return rng.random() <= spec.probability

    def apply(self, node_id: str, value: Any, tracer: CascadTracer) -> Any:
        """Apply all faults targeting node_id to value."""
        current = value
        for spec in self.specs:
            if spec.target_node != node_id or not self.should_inject(spec):
                continue
            current = self._apply_one(spec, current)
            tracer.emit(
                node_id,
                EventKind.FAULT_INJECTED,
                {
                    "fault_label": spec.label,
                    "fault_kind": str(spec.kind),
                    "payload": spec.payload,
                    "result": current,
                },
                {"fault"},
            )
        return current

    def _apply_one(self, spec: FaultSpec, value: Any) -> Any:
        kind = FaultKind(spec.kind)
        if kind == FaultKind.EXCEPTION:
            raise RuntimeError(spec.payload.get("message", "Injected exception"))
        if kind == FaultKind.DELAY:
            return value
        if kind == FaultKind.CORRUPT_VALUE:
            return spec.payload.get("value", "__CORRUPTED__")
        if kind == FaultKind.TOOL_MISRESULT:
            return spec.payload.get("value", {"ok": False, "error": "wrong tool result"})
        if kind == FaultKind.STAGED_TOOL_ARGUMENT:
            return spec.payload.get("value", {"task": "__CORRUPTED__"})
        if kind == FaultKind.MEMORY_POISON:
            poisoned = deepcopy(value)
            if isinstance(poisoned, list):
                poisoned.append(spec.payload.get("memory", "poisoned memory"))
            return poisoned
        if kind == FaultKind.DROP_FIELD and isinstance(value, dict):
            dropped = deepcopy(value)
            dropped.pop(spec.payload.get("field"), None)
            return dropped
        return value
