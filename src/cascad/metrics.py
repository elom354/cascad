"""Propagation metrics for Cascad."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from statistics import mean, stdev
from typing import Iterable

from cascad.causal import CausalGraph
from cascad.divergence import DivergenceDistribution, encoder_status
from cascad.models import EventKind, RunTrace


@dataclass(frozen=True)
class PropagationMetrics:
    """Quantitative propagation metrics."""

    propagation_depth: int
    propagation_delay: float
    propagation_breadth: int
    memory_amplification_factor: float
    affected_nodes: list[str]
    first_fault_node: str | None
    first_visible_error_node: str | None
    contamination_breadth_curve: tuple[tuple[float, float], ...] = ()
    memory_persistence: float = 0.0
    encoder_used: str = "not_applicable"
    encoder_reason: str = "no textual distance computed"


def compute_metrics(
    trace: RunTrace,
    clean_trace: RunTrace | None = None,
    natural_divergence: dict[str, DivergenceDistribution] | None = None,
    epsilon: float = 0.05,
    construction_method: str = "counterfactual",
) -> PropagationMetrics:
    """Compute propagation metrics from a trace."""
    fault_events = [event for event in trace.events if event.kind == EventKind.FAULT_INJECTED]
    observations = sorted(trace.observations, key=lambda item: item.timestamp)
    affected_nodes = sorted({item.node_id for item in observations})

    first_fault = fault_events[0] if fault_events else None
    first_visible = observations[0] if observations else None
    delay = 0.0
    if first_fault and first_visible:
        delay = max(0.0, first_visible.timestamp - first_fault.timestamp)

    depth = 0
    if first_fault:
        graph = CausalGraph.from_trace(
            trace,
            clean_trace=clean_trace,
            natural_divergence=natural_divergence,
            epsilon=epsilon,
            construction_method=construction_method,
        )
        depths = graph.shortest_depths(first_fault.node_id)
        depth = max((depths.get(node, 0) for node in affected_nodes), default=0)

    memory_observations = [
        item
        for item in observations
        if _event_kind(trace, item.event_id) in {EventKind.MEMORY_READ, EventKind.MEMORY_WRITE}
    ]
    memory_amplification = len(memory_observations) / max(1, len(observations))

    curve = contamination_breadth_curve(trace, graph=graph if first_fault else None)
    persistence = memory_persistence_rate(trace, k_turns=1)
    return PropagationMetrics(
        propagation_depth=depth,
        propagation_delay=delay,
        # Compatibility field. New analyses should consume the curve below.
        propagation_breadth=len(affected_nodes),
        memory_amplification_factor=memory_amplification,
        affected_nodes=affected_nodes,
        first_fault_node=first_fault.node_id if first_fault else None,
        first_visible_error_node=first_visible.node_id if first_visible else None,
        contamination_breadth_curve=tuple(curve),
        memory_persistence=persistence,
        encoder_used=encoder_status()["encoder_used"],
        encoder_reason=encoder_status()["reason"],
    )


def contamination_breadth_curve(
    trace: RunTrace, graph: CausalGraph | None = None
) -> list[tuple[float, float]]:
    """Return the monotonic normalized contamination breadth curve ``CB(t)``."""
    fault_events = [event for event in trace.events if event.kind == EventKind.FAULT_INJECTED]
    if not fault_events:
        return []
    origin = fault_events[0].node_id
    graph = graph or CausalGraph.from_trace(trace)
    reachable = graph.reachable(origin)
    denominator = max(1, len(reachable))
    observations = sorted(trace.observations, key=lambda item: item.timestamp)
    seen: set[str] = set()
    curve: list[tuple[float, float]] = []
    start = fault_events[0].timestamp
    for observation in observations:
        if observation.node_id == origin or observation.node_id in reachable:
            seen.add(observation.node_id)
        time = max(0.0, observation.timestamp - start)
        curve.append((time, min(1.0, len(seen - {origin}) / denominator)))
    return curve


def contamination_breadth_final(curve: Iterable[tuple[float, float]]) -> float:
    """Return the final scalar CB value as a percentage for legacy callers.

    ``PropagationMetrics.propagation_breadth`` historically was a count.  It
    stays a count in current results through ``compute_metrics``' affected-node
    set, while this helper exposes the methodological scalar in ``[0, 1]``.
    """
    values = list(curve)
    return values[-1][1] if values else 0.0


def memory_persistence_rate(run_trace: RunTrace | Iterable[RunTrace], k_turns: int) -> float:
    """Fraction of the next ``k_turns`` whose memory still contains corruption."""
    traces = [run_trace] if isinstance(run_trace, RunTrace) else list(run_trace)
    if not traces or k_turns < 1:
        return 0.0
    ordered = sorted(traces, key=lambda trace: trace.episode_id if trace.episode_id is not None else 0)
    poisoned = any(event.kind == EventKind.FAULT_INJECTED and "memory" in event.node_id for event in ordered[0].events)
    if not poisoned:
        return 0.0
    following = ordered[1 : k_turns + 1]
    if not following:
        # A single trace can still demonstrate a poisonous write/read within the
        # same episode; it is not evidence of cross-episode persistence.
        return 0.0
    persistent = 0
    for trace in following:
        text = " ".join(str(event.payload).lower() for event in trace.events if "memory" in event.node_id)
        persistent += int("poison" in text or "corrupt" in text)
    return persistent / len(following)


def variance_report(metric_values: list[float]) -> dict[str, float]:
    """Return mean, sample deviation and normal 95% confidence interval."""
    if not metric_values:
        return {"mean": 0.0, "stddev": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    average = mean(metric_values)
    deviation = stdev(metric_values) if len(metric_values) > 1 else 0.0
    margin = 1.96 * deviation / sqrt(len(metric_values))
    return {"mean": average, "stddev": deviation, "ci95_low": average - margin, "ci95_high": average + margin}


def _event_kind(trace: RunTrace, event_id: str) -> EventKind | str | None:
    for event in trace.events:
        if event.event_id == event_id:
            return event.kind
    return None
