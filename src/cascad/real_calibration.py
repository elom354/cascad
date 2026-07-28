"""Held-out-safe calibration for paired real-agent event divergences."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, stdev
from typing import Iterable


@dataclass(frozen=True)
class FrozenThreshold:
    """One event-family threshold derived only from clean/clean pairs."""

    configuration_id: str
    task_family: str
    event_key: str
    sample_count: int
    mean: float
    sample_stddev: float
    maximum: float
    epsilon: float
    threshold: float
    calibration_exceedances: int


def freeze_threshold(
    configuration_id: str,
    task_family: str,
    event_key: str,
    distances: Iterable[float],
    *,
    numerical_margin: float = 1e-9,
) -> FrozenThreshold:
    """Freeze a conservative clean/clean maximum threshold.

    The public Cascad formula remains ``mean + stddev + epsilon``. Epsilon is
    selected only from calibration so the resulting threshold lies a numerical
    margin above the largest calibration distance.
    """
    values = tuple(float(value) for value in distances)
    if not values:
        raise ValueError("at least one calibration distance is required")
    average = mean(values)
    deviation = stdev(values) if len(values) > 1 else 0.0
    maximum = max(values)
    epsilon = max(0.0, maximum - average - deviation) + numerical_margin
    threshold = average + deviation + epsilon
    return FrozenThreshold(
        configuration_id=configuration_id,
        task_family=task_family,
        event_key=event_key,
        sample_count=len(values),
        mean=average,
        sample_stddev=deviation,
        maximum=maximum,
        epsilon=epsilon,
        threshold=threshold,
        calibration_exceedances=sum(value > threshold for value in values),
    )
