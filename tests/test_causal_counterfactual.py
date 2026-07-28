from cascad.causal import CausalGraph
from cascad.divergence import estimate_natural_divergence
from cascad.injection import FaultInjector
from cascad.simulator import ReActPropagationSimulator, default_fault


def test_clean_clean_counterfactual_creates_no_contamination_edges() -> None:
    simulator = ReActPropagationSimulator()
    first, second = simulator.run(seed=1).trace, simulator.run(seed=2).trace
    graph = CausalGraph.from_trace(first, clean_trace=second, natural_divergence={})
    assert graph.edges == []


def test_counterfactual_recovers_only_dependent_corrupted_path() -> None:
    natural = estimate_natural_divergence(lambda seed=0: ReActPropagationSimulator().run(seed=seed), M=2)
    clean = ReActPropagationSimulator().run().trace
    corrupt = ReActPropagationSimulator(FaultInjector([default_fault()])).run().trace
    graph = CausalGraph.from_trace(corrupt, clean_trace=clean, natural_divergence=natural)
    assert any(edge.source == "tool" and edge.target == "verifier" for edge in graph.edges)
    assert all(edge.relation != "temporal_dependency" for edge in graph.edges)
