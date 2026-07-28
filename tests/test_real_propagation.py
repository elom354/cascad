from cascad.real_propagation import classify_propagation


def test_memory_propagation_is_retained_even_with_final_recovery() -> None:
    outcome = classify_propagation(
        structurally_multi_step=True,
        persistence_opportunity_realized=True,
        memory_contaminated=True,
        downstream_contaminated=True,
        explicit_detection=False,
        final_failure=False,
    )
    assert outcome.primary_class == "propagated_to_memory"
    assert outcome.propagated_to_memory is True
    assert outcome.successful_recovery is True


def test_unrealized_opportunity_is_not_excluded_or_called_absorption() -> None:
    outcome = classify_propagation(
        structurally_multi_step=True,
        persistence_opportunity_realized=False,
        memory_contaminated=False,
        downstream_contaminated=False,
        explicit_detection=False,
        final_failure=False,
    )
    assert outcome.primary_class == "no_persistence_opportunity_realized"
    assert outcome.no_persistence_opportunity_realized is True
