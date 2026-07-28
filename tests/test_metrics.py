from cascad.injection import FaultInjector
from cascad.simulator import ReActPropagationSimulator, default_fault


def test_tool_fault_has_propagation_metrics() -> None:
    fault = default_fault(kind="tool_misresult", target_node="tool")
    result = ReActPropagationSimulator(FaultInjector([fault])).run()

    assert result.metrics.propagation_breadth >= 1
    assert "tool" in result.metrics.affected_nodes
    assert result.metrics.first_fault_node == "tool"


def test_memory_poison_contributes_to_memory_amplification() -> None:
    fault = default_fault(kind="memory_poison", target_node="memory")
    result = ReActPropagationSimulator(FaultInjector([fault])).run()

    assert "memory" in result.metrics.affected_nodes
    assert result.metrics.memory_amplification_factor > 0

