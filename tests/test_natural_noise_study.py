import json

import pytest

from cascad.attribution_baseline import build_attribution_prompt, serialize_trace_for_attribution
from cascad.models import EventKind
from cascad.natural_noise import (
    CALIBRATION_TEMPLATES,
    EVALUATION_TEMPLATES,
    LEVEL_COUNTS,
    MEDIATOR,
    NON_CAUSAL_FIELDS,
    ROOT,
    SYMPTOM,
    build_calibrated_deepseek_prompt,
    build_calibration,
    canonicalized_first_difference,
    diagnose_pair,
    generate_pair,
    run_natural_noise_study,
)


def calibration_profile():
    return build_calibration([
        generate_pair(100_000 + index, tuple(LEVEL_COUNTS)[index % 4], "calibration")
        for index in range(12)
    ])


def test_role_contract_is_root_mediator_and_visible_symptom() -> None:
    assert (ROOT, MEDIATOR, SYMPTOM) == ("share", "memory", "notify")
    pair = generate_pair(0, "N3", "evaluation")
    categories = {item["category"]: item["node"] for item in pair.annotations if item["category"] != "benign_natural_divergence"}
    assert categories["injected_root_divergence"] == ROOT
    assert categories["visible_failure"] == SYMPTOM


def test_benign_noise_does_not_change_task_outcome() -> None:
    pair = generate_pair(7, "N3", "calibration")
    clean_answer = next(event.payload["answer"] for event in pair.clean.events if event.node_id == "responder" and "answer" in event.payload)
    noisy_answer = next(event.payload["answer"] for event in pair.comparison.events if event.node_id == "responder" and "answer" in event.payload)
    assert clean_answer == noisy_answer


def test_calibration_pairs_are_fault_free_and_evaluation_is_faulted() -> None:
    calibration = generate_pair(1, "N2", "calibration")
    evaluation = generate_pair(1, "N2", "evaluation")
    assert not calibration.contains_fault
    assert not any(event.kind == EventKind.FAULT_INJECTED for event in calibration.comparison.events)
    assert evaluation.contains_fault
    assert any(event.kind == EventKind.FAULT_INJECTED for event in evaluation.comparison.events)


def test_calibration_and_evaluation_templates_are_disjoint() -> None:
    calibration_ids = {item.template_id for values in CALIBRATION_TEMPLATES.values() for item in values}
    evaluation_ids = {item.template_id for values in EVALUATION_TEMPLATES.values() for item in values}
    assert calibration_ids.isdisjoint(evaluation_ids)


@pytest.mark.parametrize("level,count", LEVEL_COUNTS.items())
def test_noise_levels_have_requested_divergence_count(level: str, count: int) -> None:
    pair = generate_pair(3, level, "evaluation")
    benign = [item for item in pair.annotations if item["category"] == "benign_natural_divergence"]
    assert len(benign) == count
    assert len({item["node"] for item in benign}) == count


def test_evaluator_metadata_never_enters_prompts() -> None:
    pair = generate_pair(2, "N3", "evaluation")
    profile = calibration_profile()
    raw = build_attribution_prompt(pair.comparison, mode="paired", clean_trace=pair.clean)
    calibrated, _ = build_calibrated_deepseek_prompt(pair, profile)
    serialized = (raw.prompt + calibrated).casefold()
    for forbidden in (
        "benign_natural_divergence", "injected_root_divergence",
        "downstream_contamination", "visible_failure", "fault_injected",
    ):
        assert forbidden not in serialized


def test_paired_methods_receive_identical_candidates() -> None:
    pair = generate_pair(4, "N2", "evaluation")
    raw = build_attribution_prompt(pair.comparison, mode="paired", clean_trace=pair.clean)
    _, calibrated = build_calibrated_deepseek_prompt(pair, calibration_profile())
    assert raw.candidates == calibrated


def test_canonicalization_ignores_only_declared_noncausal_fields() -> None:
    structured = generate_pair(8, "N2", "evaluation")
    assert canonicalized_first_difference(structured) == "planner"
    assert "resource_id" in NON_CAUSAL_FIELDS
    semantic_only = generate_pair(8, "N1", "calibration")
    assert canonicalized_first_difference(semantic_only) == "planner"


def test_calibration_switch_changes_only_thresholding_behavior() -> None:
    pair = generate_pair(6, "N1", "evaluation")
    profile = calibration_profile()
    raw_prediction, _, raw_distances = diagnose_pair(
        pair, profile, use_calibration=False, use_dependencies=True
    )
    calibrated_prediction, _, calibrated_distances = diagnose_pair(
        pair, profile, use_calibration=True, use_dependencies=True
    )
    assert raw_prediction == "planner"
    assert calibrated_prediction == ROOT
    assert raw_distances.keys() == calibrated_distances.keys()


def test_dependency_switch_changes_only_reachability_filtering() -> None:
    pair = generate_pair(5, "N3", "evaluation")
    profile = calibration_profile()
    full_prediction, full_nodes, full_distances = diagnose_pair(
        pair, profile, use_calibration=True, use_dependencies=True
    )
    free_prediction, free_nodes, free_distances = diagnose_pair(
        pair, profile, use_calibration=True, use_dependencies=False
    )
    assert full_prediction == free_prediction == ROOT
    assert full_distances == free_distances
    assert full_nodes <= free_nodes


def test_unique_prompts_and_traces() -> None:
    bundles = []
    for seed in range(5):
        pair = generate_pair(seed, "N0", "evaluation")
        bundles.append(build_attribution_prompt(pair.comparison, mode="paired", clean_trace=pair.clean))
    assert len({bundle.prompt_sha256 for bundle in bundles}) == 5
    assert len({bundle.clean_trace_sha256 for bundle in bundles}) == 5
    assert len({bundle.corrupt_trace_sha256 for bundle in bundles}) == 5


def test_same_seed_regenerates_same_observable_pair() -> None:
    first = generate_pair(11, "N3", "evaluation")
    second = generate_pair(11, "N3", "evaluation")
    assert serialize_trace_for_attribution(first.clean) == serialize_trace_for_attribution(second.clean)
    assert serialize_trace_for_attribution(first.comparison) == serialize_trace_for_attribution(second.comparison)
    assert first.noise_template_ids == second.noise_template_ids


def test_study_runner_exports_manifests_audit_and_summaries(tmp_path) -> None:
    result = run_natural_noise_study(
        tmp_path, instances_per_level=2, calibration_pairs=8
    )
    assert result["api_call_count"] == 0
    assert result["uniqueness"]["passed"]
    for filename in (
        "calibration_manifest.json", "evaluation_manifest.json",
        "fairness_audit.json", "raw_results.jsonl", "summary.csv",
        "bootstrap_confidence_intervals.json", "mcnemar_tables.json",
    ):
        assert (tmp_path / filename).exists()
    audit = json.loads((tmp_path / "fairness_audit.json").read_text())
    assert audit["passed"]
