from cascad.attribution_baseline import serialize_trace_for_attribution
from cascad.branched_dependency import (
    INDEPENDENT_NODES,
    MAIN_NODES,
    MEDIATOR,
    NATIVE_EDGES,
    ROOT,
    SYMPTOM,
    _threshold_audit,
    build_branched_calibration,
    diagnose_branched,
    generate_branched_pair,
    graph_topology_audit,
    main_outcome_without_independent_branch,
    run_branched_dependency_study,
)


def profile():
    return build_branched_calibration([
        generate_branched_pair(200_000 + seed, "B1", "calibration")
        for seed in range(8)
    ])


def test_exact_native_graph_topology() -> None:
    trace = generate_branched_pair(0, "B3", "evaluation").comparison
    actual = {(edge.source, edge.target) for edge in trace.causal_edges}
    assert actual == set(NATIVE_EDGES)
    assert graph_topology_audit(trace)["passed"]


def test_no_cross_branch_dependency_path() -> None:
    audit = graph_topology_audit(
        generate_branched_pair(1, "B3", "evaluation").comparison
    )
    assert not audit["share_reaches_audit_log"]
    assert not audit["share_reaches_metrics_export"]
    assert audit["forbidden_edges_present"] == []


def test_role_and_true_contamination_contract() -> None:
    assert (ROOT, MEDIATOR, SYMPTOM) == ("share", "memory", "notify")
    assert MAIN_NODES == ("share", "memory", "notify", "responder")
    assert set(MAIN_NODES).isdisjoint(INDEPENDENT_NODES)


def test_independent_branch_never_changes_main_result() -> None:
    for seed in range(4):
        for corrupt in (False, True):
            with_branch, without_branch = main_outcome_without_independent_branch(
                seed, corrupt
            )
            assert with_branch == without_branch


def test_class_1_is_calibrated_and_class_2_exceeds_threshold() -> None:
    calibration = profile()
    pairs = [
        generate_branched_pair(1, level, "evaluation")
        for level in ("B1", "B2", "B3")
    ]
    audit = _threshold_audit(pairs, calibration, 0.05)
    assert audit["class_1_within_range"]
    assert audit["class_2_above_threshold"]


def test_no_evaluator_metadata_in_observable_traces() -> None:
    pair = generate_branched_pair(3, "B3", "evaluation")
    serialized = str(serialize_trace_for_attribution(pair.comparison)).lower()
    for forbidden in (
        "ground_truth", "root_cause", "visible_symptom",
        "salient_non_causal", "true_contaminated",
    ):
        assert forbidden not in serialized


def test_dependency_ablation_changes_only_filtering() -> None:
    pair = generate_branched_pair(2, "B3", "evaluation")
    calibration = profile()
    full_prediction, full_nodes, full_distances = diagnose_branched(
        pair, calibration, use_calibration=True, use_dependencies=True
    )
    free_prediction, free_nodes, free_distances = diagnose_branched(
        pair, calibration, use_calibration=True, use_dependencies=False
    )
    assert full_distances == free_distances
    assert full_prediction == ROOT
    assert free_prediction in {"audit_log", ROOT}
    assert full_nodes <= free_nodes


def test_full_cascad_excludes_disconnected_salient_nodes() -> None:
    pair = generate_branched_pair(9, "B3", "evaluation")
    prediction, nodes, _ = diagnose_branched(
        pair, profile(), use_calibration=True, use_dependencies=True
    )
    assert prediction == ROOT
    assert nodes == set(MAIN_NODES)
    assert set(INDEPENDENT_NODES).isdisjoint(nodes)


def test_same_seed_regenerates_same_observable_traces() -> None:
    first = generate_branched_pair(11, "B3", "evaluation")
    second = generate_branched_pair(11, "B3", "evaluation")
    assert serialize_trace_for_attribution(first.clean) == serialize_trace_for_attribution(second.clean)
    assert serialize_trace_for_attribution(first.comparison) == serialize_trace_for_attribution(second.comparison)
    assert first.template_ids == second.template_ids


def test_held_out_split_integrity() -> None:
    calibration = {
        template
        for seed in range(8)
        for template in generate_branched_pair(200_000 + seed, "B1", "calibration").template_ids
    }
    evaluation = {
        template
        for level in ("B0", "B1", "B2", "B3")
        for template in generate_branched_pair(0, level, "evaluation").template_ids
    }
    assert calibration.isdisjoint(evaluation)


def test_runner_exports_complete_structural_study(tmp_path) -> None:
    result = run_branched_dependency_study(
        tmp_path, instances_per_level=2, calibration_pairs=8
    )
    assert result["uniqueness"]["passed"]
    assert result["api_calls"] == 0
    for filename in (
        "calibration_manifest.json", "evaluation_manifest.json",
        "fairness_audit.json", "graph_topology_audit.json",
        "raw_results.jsonl", "summary.csv",
        "paired_correctness.json", "bootstrap_confidence_intervals.json",
    ):
        assert (tmp_path / filename).is_file()
