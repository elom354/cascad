from cascad.attribution_baseline import serialize_trace_for_attribution
from cascad.experiment import ExperimentCase, run_experiment
from cascad.injection import FaultInjector
from cascad.models import EventKind
from cascad.scenarios import ATTRIBUTION_SCENARIOS, attribution_fault
from cascad.simulator import ReActPropagationSimulator


def test_neutral_fault_messages_do_not_name_their_target() -> None:
    for scenario in ("support_neutral", "document_neutral", "cloud_neutral"):
        fault = attribution_fault(scenario)
        message = str(fault.payload["value"].get("error", "")).casefold()
        assert fault.target_node.casefold() not in message
        assert all(token not in message for token in ("refund", "report", "sharing", "upload", "notify"))


def test_attribution_view_hides_privileged_injection_event() -> None:
    scenario = "cloud_neutral"
    result = ReActPropagationSimulator(
        FaultInjector([attribution_fault(scenario)]), scenario=scenario
    ).run()
    observable = serialize_trace_for_attribution(result.trace)
    assert not any(event["event_kind"] == EventKind.FAULT_INJECTED.value for event in observable)
    assert "fault_label" not in str(observable)


def test_distant_symptom_separates_root_from_visible_failure() -> None:
    scenario = "cloud_distant_symptom"
    summary = run_experiment([
        ExperimentCase(
            name="distant",
            scenario=scenario,
            method="attribution",
            faults=[attribution_fault(scenario)],
            trials=1,
            attribution_llm=lambda _: "notify",
        )
    ])[0]
    record = summary.attribution_records[0]
    assert record["injection_node"] == "share"
    assert record["visible_failure_node"] == "notify"
    assert record["deepseek_prediction_type"] == "visible_symptom"
    assert record["graph_predicted_node"] == "share"
    assert record["graph_prediction_type"] == "root_cause"
    assert '"node_id": "share"' in record["prompt"]


def test_all_attribution_variants_are_registered() -> None:
    assert set(ATTRIBUTION_SCENARIOS) == {
        "support_neutral", "document_neutral", "cloud_neutral",
        "cloud_distant_symptom", "cloud_distant_symptom_natural_noise",
    }
