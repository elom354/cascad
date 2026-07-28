from cascad.models import FaultKind, FaultSpec
from cascad.statistics import exact_mcnemar, paired_correctness, wilson_interval


def test_fault_spec_exposes_independent_injection_node() -> None:
    fault = FaultSpec("tool", FaultKind.TOOL_MISRESULT)
    assert fault.injection_node == "tool"


def test_wilson_interval_known_extremes_are_not_degenerate() -> None:
    low, high = wilson_interval(20, 20)
    assert 0.83 < low < 1.0
    assert high == 1.0
    low, high = wilson_interval(0, 20)
    assert low == 0.0
    assert 0.0 < high < 0.17


def test_exact_mcnemar_exports_table_p_value_and_effect() -> None:
    result = exact_mcnemar(76, 2, 2, 0)
    assert result["discordant_count"] == 4
    assert result["exact_two_sided_p_value"] == 1.0
    assert result["accuracy_difference_a_minus_b"] == 0.0


def test_paired_correctness_constructs_complete_table() -> None:
    result = paired_correctness([True, True, False], [True, False, True])
    assert result["both_correct"] == 1
    assert result["a_correct_b_wrong"] == 1
    assert result["a_wrong_b_correct"] == 1
    assert result["both_wrong"] == 0
