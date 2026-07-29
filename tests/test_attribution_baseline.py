from cascad.attribution_baseline import (
    attribute_failure,
    attribute_failure_detailed,
    build_attribution_prompt,
    expand_compact_observable_trace,
    localization_accuracy,
    serialize_trace_for_attribution,
)
from cascad.injection import FaultInjector
from cascad.models import NodeEvent, RunTrace
from cascad.simulator import ReActPropagationSimulator, default_fault


def test_mocked_attribution_accuracy() -> None:
    trace = ReActPropagationSimulator(FaultInjector([default_fault()])).run().trace
    assert attribute_failure(trace, lambda _: "tool") == "tool"
    assert localization_accuracy(["tool", "memory"], ["tool", "tool"]) == 0.5


def test_attribution_keeps_raw_evidence_and_normalizes_safe_formatting() -> None:
    trace = ReActPropagationSimulator(FaultInjector([default_fault()])).run().trace
    result = attribute_failure_detailed(trace, lambda _: "**TOOL**")
    assert result.predicted_node == "tool"
    assert result.raw_response == "**TOOL**"
    assert '"node_id": "tool"' in result.prompt


def test_attribution_rejects_ambiguous_multi_node_answer() -> None:
    trace = ReActPropagationSimulator(FaultInjector([default_fault()])).run().trace
    result = attribute_failure_detailed(trace, lambda _: "tool then responder")
    assert result.predicted_node is None


def test_paired_prompt_accepts_structurally_different_candidate_sets() -> None:
    clean = RunTrace(run_id="clean")
    clean.add_event(NodeEvent("planner", "model_response", "clean"))
    observed = RunTrace(run_id="observed")
    observed.add_event(NodeEvent("planner", "model_response", "observed"))
    observed.add_event(NodeEvent("recovery", "tool_call", "observed"))

    bundle = build_attribution_prompt(
        observed,
        clean_trace=clean,
        mode="paired",
    )

    assert bundle.candidates == ("planner", "recovery")
    assert "Candidate node_ids: ['planner', 'recovery']" in bundle.prompt


def test_compact_trace_round_trips_cumulative_model_requests() -> None:
    trace = RunTrace(run_id="compact")
    static = {
        "runtime": {"model": "test"},
        "system_message": "system",
        "available_tools": ["calculate"],
    }
    first_message = {"type": "human", "content": "calculate"}
    trace.add_event(
        NodeEvent(
            "call_model",
            "model_request",
            trace.run_id,
            payload={**static, "messages": [first_message]},
        )
    )
    trace.add_event(
        NodeEvent(
            "call_model",
            "model_request",
            trace.run_id,
            payload={
                **static,
                "messages": [
                    first_message,
                    {"type": "ai", "content": "done"},
                ],
            },
        )
    )

    full = serialize_trace_for_attribution(trace)
    compact = serialize_trace_for_attribution(
        trace,
        serialization_version="compact-v1",
    )

    assert expand_compact_observable_trace(compact) == full
    assert len(str(compact)) < len(str(full))
    assert compact[1]["payload"]["inherits"] == [
        "runtime",
        "system_message",
        "available_tools",
    ]
