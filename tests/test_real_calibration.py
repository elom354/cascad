from cascad.real_calibration import freeze_threshold


def test_frozen_threshold_exceeds_calibration_maximum() -> None:
    result = freeze_threshold(
        "config-a",
        "task-a",
        "node|kind|0",
        [0.0, 0.1, 0.2, 0.3],
    )
    assert result.threshold > 0.3
    assert result.calibration_exceedances == 0
    assert result.sample_count == 4


def test_frozen_threshold_handles_constant_zero_distance() -> None:
    result = freeze_threshold(
        "config-a",
        "task-a",
        "node|kind|0",
        [0.0, 0.0],
    )
    assert result.threshold == 1e-9
    assert result.epsilon == 1e-9
