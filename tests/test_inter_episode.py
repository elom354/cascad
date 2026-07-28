from cascad.inter_episode import (
    CORRECTION_EPISODES,
    DIFFICULTIES,
    FAILED_SAFEGUARD,
    HORIZONS,
    PERSISTENCE_MEDIATOR,
    SOURCE,
    VISIBLE_SYMPTOM,
    _audits,
    _attribution_record,
    _enrich_role_estimates,
    build_history_prompt,
    build_inter_episode_calibration,
    diagnose_history,
    generate_multi_episode_pair,
    serialize_history,
    simulate_policy,
    verifier_decision,
)


def calibration():
    return build_inter_episode_calibration([
        generate_multi_episode_pair(300_000 + seed, "E3", 10, "calibration")
        for seed in range(8)
    ])


def test_preregistered_roles_and_horizons_are_distinct() -> None:
    assert (SOURCE, FAILED_SAFEGUARD, PERSISTENCE_MEDIATOR, VISIBLE_SYMPTOM) == (
        "tool", "verifier", "memory", "action"
    )
    assert HORIZONS == (1, 3, 5, 10)
    assert CORRECTION_EPISODES == (2, 3, 5)


def test_verifier_has_real_blocking_opportunity_and_separate_fail_open() -> None:
    blocked = verifier_decision("restricted", fail_open=False)
    allowed = verifier_decision("restricted", fail_open=True)
    assert not blocked["allowed"]
    assert allowed["allowed"]
    assert not allowed["state_usable"]


def test_episode_records_expose_required_persistence_facts() -> None:
    pair = generate_multi_episode_pair(1, "E3", 5, "evaluation")
    rows = simulate_policy(pair, "none")
    assert len(rows) == 5
    assert not rows[0]["memory_read_attempted"]
    assert rows[1]["memory_read_attempted"]
    for row in rows:
        for field in (
            "injection_episode", "current_episode", "memory_read_attempted",
            "contaminated_memory_readable",
            "contaminated_memory_changed_behavior", "explicit_failure",
        ):
            assert field in row


def test_prewrite_quarantine_changes_future_behavior_without_clean_false_positive() -> None:
    pair = generate_multi_episode_pair(2, "E0", 5, "evaluation")
    untreated = simulate_policy(pair, "none")
    quarantined = simulate_policy(pair, "prewrite_quarantine")
    clean = simulate_policy(
        pair, "prewrite_quarantine", clean_control=True
    )
    assert any(row["contaminated_memory_readable"] for row in untreated[1:])
    assert not any(row["contaminated_memory_readable"] for row in quarantined)
    assert quarantined[0]["prewrite_blocked"]
    assert not any(row["prewrite_triggered"] for row in clean)


def test_delayed_correction_uses_predeclared_episode() -> None:
    pair = generate_multi_episode_pair(3, "E0", 5, "evaluation")
    rows = simulate_policy(
        pair, "delayed_correction", correction_episode=3
    )
    assert rows[1]["contaminated_memory_readable"]
    assert not rows[2]["contaminated_memory_readable"]
    assert not any(row["contaminated_memory_readable"] for row in rows[2:])


def test_difficulty_controls_delayed_explicit_symptom() -> None:
    pair = generate_multi_episode_pair(4, "E3", 10, "evaluation")
    rows = simulate_policy(pair, "none")
    assert all(not row["explicit_failure"] for row in rows[:4])
    assert rows[4]["explicit_failure"]


def test_calibration_and_evaluation_templates_are_disjoint() -> None:
    calibration_pair = generate_multi_episode_pair(
        300_001, "E3", 10, "calibration"
    )
    evaluation = [
        generate_multi_episode_pair(1, difficulty, 10, "evaluation")
        for difficulty in DIFFICULTIES
    ]
    cal = set(calibration_pair.template_ids)
    evaluate = {item for pair in evaluation for item in pair.template_ids}
    assert cal.isdisjoint(evaluate)


def test_information_windows_and_prompts_do_not_leak_roles() -> None:
    pair = generate_multi_episode_pair(5, "E3", 10, "evaluation")
    symptom_prompt, symptom_candidates, _, symptom_clean = build_history_prompt(
        pair, "symptom_episode_only"
    )
    full_prompt, full_candidates, _, full_clean = build_history_prompt(
        pair, "corrupt_full_history"
    )
    paired_prompt, paired_candidates, _, paired_clean = build_history_prompt(
        pair, "paired_multi_episode"
    )
    assert symptom_clean is None and full_clean is None
    assert paired_clean is not None
    assert symptom_candidates == full_candidates == paired_candidates
    for prompt in (symptom_prompt, full_prompt, paired_prompt):
        lowered = prompt.lower()
        assert "fault_injected" not in lowered
        assert "failed_safeguard" not in lowered
        assert "persistence_mediator" not in lowered


def test_full_history_recovers_source_independently_of_fault_metadata() -> None:
    pair = generate_multi_episode_pair(6, "E3", 10, "evaluation")
    estimate, nodes, _ = diagnose_history(
        pair, calibration(), use_calibration=True, use_dependencies=True
    )
    assert estimate == SOURCE
    assert SOURCE in nodes
    assert FAILED_SAFEGUARD in nodes
    assert PERSISTENCE_MEDIATOR in nodes


def test_cascad_role_estimates_are_separate_from_source_estimate() -> None:
    pair = generate_multi_episode_pair(6, "E3", 10, "evaluation")
    row = _attribution_record(
        pair, "cascad_full", "paired_multi_episode", SOURCE,
        {SOURCE, FAILED_SAFEGUARD, PERSISTENCE_MEDIATOR, "action"},
        0.0, prompt=None, raw=None, usage=None,
    )
    enriched = _enrich_role_estimates(row, pair)
    assert enriched["estimated_source"] == SOURCE
    assert enriched["estimated_failed_safeguard"] == FAILED_SAFEGUARD
    assert enriched["estimated_persistence_mediator"] == PERSISTENCE_MEDIATOR
    assert enriched["estimated_visible_symptom"] == VISIBLE_SYMPTOM


def test_observable_history_hides_fault_event() -> None:
    pair = generate_multi_episode_pair(7, "E1", 3, "evaluation")
    serialized = serialize_history(pair.corrupt)
    assert serialized
    assert all(event["event_kind"] != "fault_injected" for event in serialized)


def test_generation_is_deterministic_from_seed() -> None:
    first = generate_multi_episode_pair(8, "E2", 5, "evaluation")
    second = generate_multi_episode_pair(8, "E2", 5, "evaluation")
    assert serialize_history(first.clean) == serialize_history(second.clean)
    assert serialize_history(first.corrupt) == serialize_history(second.corrupt)
    assert first.memory_object_id == second.memory_object_id


def test_full_study_audits_pass_before_api() -> None:
    calibration_pairs = [
        generate_multi_episode_pair(300_000 + seed, "E3", 10, "calibration")
        for seed in range(8)
    ]
    evaluation = [
        generate_multi_episode_pair(
            level_index * 100_000 + horizon * 1_000 + seed,
            difficulty, horizon, "evaluation",
        )
        for level_index, difficulty in enumerate(DIFFICULTIES)
        for horizon in HORIZONS
        for seed in range(5)
    ]
    audit = _audits(calibration_pairs, evaluation)
    assert audit["passed"]
    assert audit["identical_candidate_sets"]
    assert audit["verifier_blocking_opportunity"]
