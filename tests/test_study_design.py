import pytest

from cascad.study_design import (
    balanced_target,
    paired_mcnemar_normal_approx_n,
    worst_case_proportion_n,
)


def test_default_v3_power_assumptions_require_459_pairs() -> None:
    assert (
        paired_mcnemar_normal_approx_n(
            p_method_a_only_correct=0.08,
            p_method_b_only_correct=0.03,
        )
        == 459
    )


def test_precision_and_balancing_round_up() -> None:
    assert worst_case_proportion_n(half_width=0.04) == 601
    assert balanced_target(601, strata=54) == (648, 12)


def test_equal_discordant_probabilities_are_not_a_power_alternative() -> None:
    with pytest.raises(ValueError, match="non-zero"):
        paired_mcnemar_normal_approx_n(
            p_method_a_only_correct=0.05,
            p_method_b_only_correct=0.05,
        )
