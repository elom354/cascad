"""Intervention policies for limiting propagation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Callable

from cascad.divergence import value_distance
from cascad.metrics import PropagationMetrics
from cascad.models import InterventionDecision


class ThresholdInterventionPolicy:
    """Simple threshold policy for early containment."""

    def __init__(self, max_depth: int = 2, max_breadth: int = 3) -> None:
        self.max_depth = max_depth
        self.max_breadth = max_breadth

    def decide(self, metrics: PropagationMetrics) -> InterventionDecision:
        """Return an intervention decision."""
        node = metrics.first_visible_error_node or "unknown"
        if metrics.propagation_depth > self.max_depth:
            return InterventionDecision(
                action="rollback",
                node_id=node,
                reason="propagation depth exceeded threshold",
            )
        if metrics.propagation_breadth > self.max_breadth:
            return InterventionDecision(
                action="quarantine",
                node_id=node,
                reason="propagation breadth exceeded threshold",
            )
        return InterventionDecision(
            action="continue",
            node_id=node,
            reason="propagation within thresholds",
        )


class GenericInterceptorPolicy:
    """Baseline interceptor: reject malformed tool arguments only."""

    def validate(self, node_id: str, value: Any, schema: dict[str, type] | None = None) -> InterventionDecision:
        if schema is None or not isinstance(value, dict):
            return InterventionDecision("continue", node_id, "no schema violation")
        invalid = [key for key, expected in schema.items() if key not in value or not isinstance(value[key], expected)]
        if invalid:
            return InterventionDecision("quarantine", node_id, f"invalid tool arguments: {', '.join(invalid)}")
        return InterventionDecision("continue", node_id, "tool arguments match schema")


@dataclass(frozen=True)
class CalibrationProfile:
    """Reference distribution for online control points."""

    scenario: str
    controls: dict[str, dict[str, Any]]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationProfile":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def calibrate_intervention(
    scenario: Callable[..., Any], M: int = 8, scenario_name: str = "default"
) -> CalibrationProfile:
    """Create a clean reference profile for memory, tool and final-action checks."""
    if M < 1:
        raise ValueError("M must be at least 1")
    samples: dict[str, list[Any]] = {"memory": [], "tool": [], "responder": []}
    for seed in range(M):
        try:
            result = scenario(seed=seed)
        except TypeError:
            result = scenario()
        trace = result.trace if hasattr(result, "trace") else result
        for event in trace.events:
            if event.node_id in samples and str(getattr(event.kind, "value", event.kind)) not in {"node_start", "node_end"}:
                samples[event.node_id].append(event.payload)
    controls: dict[str, dict[str, Any]] = {}
    for node, values in samples.items():
        if not values:
            continue
        reference = values[0]
        distances = [value_distance(reference, value) for value in values]
        controls[node] = {"mean": mean(distances), "stddev": stdev(distances) if len(distances) > 1 else 0.0, "reference": reference}
    return CalibrationProfile(scenario_name, controls)


class CalibratedInterventionPolicy:
    """Active online policy at memory write/read and final response checkpoints."""

    def __init__(self, profile: CalibrationProfile, epsilon: float = 0.05) -> None:
        self.profile = profile
        self.epsilon = epsilon

    def decide_value(self, node_id: str, value: Any) -> InterventionDecision:
        control = self.profile.controls.get(node_id)
        if not control:
            return InterventionDecision("continue", node_id, "no calibration profile for node")
        distance = value_distance(control["reference"], value)
        if distance <= control["mean"] + control["stddev"] + self.epsilon:
            return InterventionDecision("continue", node_id, "within calibrated natural divergence")
        action = {
            "memory": "quarantine_memory_write",
            "tool": "discard_retrieved_memory",
            "responder": "block_final_action",
        }.get(node_id, "quarantine")
        return InterventionDecision(action, node_id, f"distance {distance:.3f} exceeded calibrated threshold")  # type: ignore[arg-type]
