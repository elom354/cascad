from pathlib import Path

from cascad.export import write_trace_bundle
from cascad.injection import FaultInjector
from cascad.simulator import ReActPropagationSimulator, default_fault


def test_export_bundle(tmp_path: Path) -> None:
    fault = default_fault(kind="tool_misresult", target_node="tool")
    result = ReActPropagationSimulator(FaultInjector([fault])).run()

    write_trace_bundle(tmp_path, result.trace, result.metrics)

    assert (tmp_path / "trace.json").exists()
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "causal_graph.dot").exists()

