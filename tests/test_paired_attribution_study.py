from cascad.attribution_baseline import (
    attribute_failure_detailed,
    build_attribution_prompt,
)
from cascad.experiment import ExperimentCase, run_experiment
from cascad.injection import FaultInjector
from cascad.scenarios import attribution_fault
from cascad.simulator import ReActPropagationSimulator
import pytest


SCENARIO = "cloud_distant_symptom"


def paired_traces(seed: int = 0):
    clean = ReActPropagationSimulator(scenario=SCENARIO).run(seed=seed).trace
    corrupt = ReActPropagationSimulator(
        FaultInjector([attribution_fault(SCENARIO, seed)]), scenario=SCENARIO
    ).run(seed=seed).trace
    return clean, corrupt


def test_single_neutral_contains_one_trace_without_coaching() -> None:
    clean, corrupt = paired_traces()
    bundle = build_attribution_prompt(corrupt, mode="single-neutral", clean_trace=clean)
    assert bundle.clean_observable_trace is None
    assert bundle.corrupt_trace_present
    lowered = bundle.prompt.lower()
    assert "root cause" not in lowered and "root-cause" not in lowered
    assert "downstream symptom" not in lowered
    assert "clean reference trace" not in lowered


def test_single_guided_preserves_existing_instruction() -> None:
    clean, corrupt = paired_traces()
    bundle = build_attribution_prompt(corrupt, mode="single-guided", clean_trace=clean)
    assert "identify the root-cause node, not merely the node where a downstream symptom became visible" in bundle.prompt
    assert bundle.clean_observable_trace is None


def test_paired_prompt_has_both_complete_observable_traces_without_privilege() -> None:
    clean, corrupt = paired_traces()
    bundle = build_attribution_prompt(corrupt, mode="paired", clean_trace=clean)
    assert bundle.clean_trace_present and bundle.corrupt_trace_present
    assert bundle.clean_observable_trace
    assert bundle.corrupt_observable_trace
    assert not bundle.privileged_metadata_present
    assert bundle.leaked_terms == ()
    assert "fault_injected" not in bundle.prompt.lower()


def test_all_modes_share_candidates_and_parser() -> None:
    clean, corrupt = paired_traces()
    modes = ("single-neutral", "single-guided", "paired")
    bundles = [build_attribution_prompt(corrupt, mode=mode, clean_trace=clean) for mode in modes]
    assert len({bundle.candidates for bundle in bundles}) == 1
    results = [
        attribute_failure_detailed(corrupt, lambda _: "**SHARE**", clean_trace=clean, mode=mode)
        for mode in modes
    ]
    assert [result.predicted_node for result in results] == ["share"] * 3


def test_root_mediator_and_symptom_contract() -> None:
    summary = run_experiment([
        ExperimentCase(
            "contract",
            [attribution_fault(SCENARIO, 0)],
            fault_factory=lambda seed: [attribution_fault(SCENARIO, seed)],
            scenario=SCENARIO,
            method="attribution",
            trials=1,
            attribution_mode="single-neutral",
            attribution_llm=lambda _: "memory",
        )
    ])[0]
    record = summary.attribution_records[0]
    assert record["ground_truth_root"] == "share"
    assert record["propagation_mediator"] == "memory"
    assert record["visible_symptom"] == "notify"
    assert record["prediction_type"] == "mediator"
    assert record["cascad_prediction"] == "share"


def test_varied_instances_have_unique_trace_and_prompt_hashes() -> None:
    prompt_hashes, trace_hashes = set(), set()
    for seed in range(20):
        clean, corrupt = paired_traces(seed)
        bundle = build_attribution_prompt(corrupt, mode="paired", clean_trace=clean)
        prompt_hashes.add(bundle.prompt_sha256)
        trace_hashes.add(bundle.corrupt_trace_sha256)
    assert len(prompt_hashes) == 20
    assert len(trace_hashes) == 20


def test_identical_input_calls_are_labeled_stability_repetitions() -> None:
    summary = run_experiment([
        ExperimentCase(
            "stability",
            [attribution_fault(SCENARIO, 3)],
            fault_factory=lambda seed: [attribution_fault(SCENARIO, seed)],
            scenario=SCENARIO,
            method="attribution",
            trials=2,
            fixed_instance_seed=3,
            attribution_mode="single-guided",
            attribution_llm=lambda _: "share",
        )
    ])[0]
    assert summary.unique_trace_count == 1
    assert summary.unique_prompt_count == 1
    assert summary.api_call_count == 2
    assert summary.experimental_role == "repeated_call_stability"
    assert {record["experimental_role"] for record in summary.attribution_records} == {"repeated_call_stability"}


def test_varied_study_fails_when_instances_are_not_unique() -> None:
    with pytest.raises(ValueError, match="varied-instance requirement failed"):
        run_experiment([
            ExperimentCase(
                "invalid-varied",
                [attribution_fault(SCENARIO, 0)],
                fault_factory=lambda seed: [attribution_fault(SCENARIO, seed)],
                scenario=SCENARIO,
                method="attribution",
                trials=2,
                fixed_instance_seed=0,
                require_unique_instances=True,
                attribution_mode="paired",
                attribution_llm=lambda _: "share",
            )
        ])
