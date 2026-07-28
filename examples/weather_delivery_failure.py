"""Example: trace a failed weather report delivery."""

from cascad.export import write_trace_bundle
from cascad.injection import FaultInjector
from cascad.simulator import ReActPropagationSimulator, default_fault


if __name__ == "__main__":
    fault = default_fault(kind="tool_misresult", target_node="tool")
    result = ReActPropagationSimulator(FaultInjector([fault])).run(
        "Generate a PDF weather report and email it."
    )
    write_trace_bundle("runs/weather_delivery_failure", result.trace, result.metrics)
    print(result.metrics)

