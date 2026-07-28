from pathlib import Path

from cascad.export import write_trace_bundle
from cascad.injection import FaultInjector
from cascad.simulator import ReActPropagationSimulator, default_fault
from cascad.viewer import list_runs, load_run


def test_viewer_loads_run_bundle(tmp_path: Path) -> None:
    fault = default_fault(kind="tool_misresult", target_node="tool")
    result = ReActPropagationSimulator(FaultInjector([fault])).run()
    write_trace_bundle(tmp_path / "run-1", result.trace, result.metrics)

    runs = list_runs(tmp_path)
    loaded = load_run(tmp_path, "run-1")

    assert runs[0]["id"] == "run-1"
    assert loaded["graph"]["nodes"]
    assert loaded["graph"]["edges"]
