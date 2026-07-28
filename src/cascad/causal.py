"""Causal graph reconstruction."""

from __future__ import annotations

from collections import defaultdict, deque

from cascad.divergence import DivergenceDistribution, corresponding_events, event_distance
from cascad.models import CausalEdge, EventKind, RunTrace


class CausalGraph:
    """Legacy name for a counterfactual propagation graph.

    The edges are observed native dependencies, not a learned complete
    structural causal model. ``CounterfactualPropagationGraph`` is the
    preferred public terminology for new code.
    """

    def __init__(
        self,
        edges: list[CausalEdge],
        construction_method: str = "counterfactual",
        contaminated_nodes: set[str] | None = None,
    ) -> None:
        self.edges = edges
        self.construction_method = construction_method
        self.contaminated_nodes = contaminated_nodes or set()
        self._adj: dict[str, list[CausalEdge]] = defaultdict(list)
        for edge in edges:
            self._adj[edge.source].append(edge)

    @classmethod
    def from_trace(
        cls,
        trace: RunTrace,
        clean_trace: RunTrace | None = None,
        natural_divergence: dict[str, DivergenceDistribution] | None = None,
        epsilon: float = 0.05,
        construction_method: str = "counterfactual",
    ) -> "CausalGraph":
        """Build a causal graph with calibrated counterfactual reconstruction.

        ``temporal`` preserves the MVP behaviour solely as an explicit ablation.
        Without a clean pair, counterfactual mode retains native trace edges but
        never invents temporal causality.
        """
        if construction_method not in {"counterfactual", "temporal"}:
            raise ValueError("construction_method must be 'counterfactual' or 'temporal'")
        if construction_method == "counterfactual" and clean_trace is not None:
            return cls._from_counterfactual(trace, clean_trace, natural_divergence or {}, epsilon)
        edges = list(trace.causal_edges)
        if construction_method == "temporal" and not edges:
            previous = None
            for event in trace.events:
                if previous and previous.node_id != event.node_id:
                    edges.append(
                        CausalEdge(
                            source=previous.node_id,
                            target=event.node_id,
                            relation="temporal_dependency",
                            weight=0.25,
                            evidence="temporal fallback",
                        )
                    )
                previous = event
        return cls(edges, construction_method=construction_method)

    @classmethod
    def aggregate(cls, graphs: list["CausalGraph"], majority: float = 0.5) -> "CausalGraph":
        """Aggregate repeated counterfactual runs; edge weight is confidence."""
        if not graphs:
            return cls([])
        occurrences: dict[tuple[str, str, str], list[CausalEdge]] = defaultdict(list)
        for graph in graphs:
            for edge in graph.edges:
                occurrences[(edge.source, edge.target, edge.relation)].append(edge)
        edges = []
        for _, matches in occurrences.items():
            confidence = len(matches) / len(graphs)
            if confidence > majority:
                exemplar = matches[0]
                edges.append(CausalEdge(exemplar.source, exemplar.target, exemplar.relation, confidence,
                                        f"counterfactual confidence={confidence:.3f}"))
        return cls(edges, construction_method="counterfactual")

    @classmethod
    def _from_counterfactual(
        cls,
        corrupt: RunTrace,
        clean: RunTrace,
        natural: dict[str, DivergenceDistribution],
        epsilon: float,
    ) -> "CausalGraph":
        pairs = corresponding_events(corrupt, clean)
        faults = [event for event in corrupt.events if event.kind == EventKind.FAULT_INJECTED]
        if not faults:
            return cls([], construction_method="counterfactual", contaminated_nodes=set())
        origin = faults[0].node_id
        event_positions = {event.node_id: pos for pos, event in enumerate(corrupt.events)}
        injection_position = next(pos for pos, event in enumerate(corrupt.events) if event.kind == EventKind.FAULT_INJECTED)
        contaminated = {origin}
        evidence: dict[str, float] = {}
        for node, (corrupt_event, clean_event) in pairs.items():
            if event_positions.get(node, -1) < injection_position:
                continue
            distance = event_distance(corrupt_event, clean_event)
            baseline = natural.get(node, DivergenceDistribution(0.0, 0.0))
            if distance > baseline.mean + baseline.stddev + epsilon:
                contaminated.add(node)
                evidence[node] = distance
        native = list(corrupt.causal_edges)
        edges: list[CausalEdge] = []
        pending = set(contaminated) - {origin}
        # Only retain dependency edges reachable through already contaminated nodes.
        while pending:
            progressed = False
            for edge in native:
                if edge.source in contaminated and edge.target in pending:
                    edges.append(
                        CausalEdge(edge.source, edge.target, edge.relation, edge.weight,
                                   f"counterfactual divergence={evidence.get(edge.target, 0.0):.3f}"))
                    pending.remove(edge.target)
                    progressed = True
                    break
            if not progressed:
                break
        return cls(edges, construction_method="counterfactual", contaminated_nodes=contaminated)

    def reachable(self, start: str) -> set[str]:
        """Return nodes reachable from start."""
        seen: set[str] = set()
        queue: deque[str] = deque([start])
        while queue:
            node = queue.popleft()
            for edge in self._adj.get(node, []):
                if edge.target not in seen:
                    seen.add(edge.target)
                    queue.append(edge.target)
        return seen

    def shortest_depths(self, start: str) -> dict[str, int]:
        """Return shortest propagation depth from start."""
        depths = {start: 0}
        queue: deque[str] = deque([start])
        while queue:
            node = queue.popleft()
            for edge in self._adj.get(node, []):
                if edge.target not in depths:
                    depths[edge.target] = depths[node] + 1
                    queue.append(edge.target)
        return depths

    def to_dot(self, affected_nodes: set[str] | None = None) -> str:
        """Export graph as Graphviz DOT."""
        affected_nodes = affected_nodes or set()
        lines = ["digraph CascadPropagation {"]
        for edge in self.edges:
            color = "red" if edge.source in affected_nodes or edge.target in affected_nodes else "gray"
            lines.append(
                f'  "{edge.source}" -> "{edge.target}" '
                f'[label="{edge.relation}", color="{color}"];'
            )
        lines.append("}")
        return "\n".join(lines)


CounterfactualPropagationGraph = CausalGraph
