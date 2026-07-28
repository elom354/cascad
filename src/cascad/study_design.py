"""Transparent sample-size planning helpers for paired validation studies."""

from __future__ import annotations

from math import ceil, sqrt
from statistics import NormalDist


def paired_mcnemar_normal_approx_n(
    *,
    p_method_a_only_correct: float,
    p_method_b_only_correct: float,
    alpha: float = 0.05,
    power: float = 0.90,
) -> int:
    """Approximate paired sample size for a two-sided McNemar comparison.

    The two discordant probabilities must be specified before opening the
    confirmatory split. The approximation is intended for planning; final
    inference still uses Cascad's exact McNemar test.
    """
    p10 = p_method_a_only_correct
    p01 = p_method_b_only_correct
    if not 0 <= p10 <= 1 or not 0 <= p01 <= 1:
        raise ValueError("discordant probabilities must be between zero and one")
    if p10 + p01 > 1:
        raise ValueError("discordant probabilities cannot sum above one")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    if not 0.5 < power < 1:
        raise ValueError("power must be between 0.5 and one")
    difference = abs(p10 - p01)
    if difference == 0:
        raise ValueError("a non-zero discordant difference is required")
    discordance = p10 + p01
    z_alpha = NormalDist().inv_cdf(1 - alpha / 2)
    z_power = NormalDist().inv_cdf(power)
    numerator = (
        z_alpha * sqrt(discordance)
        + z_power * sqrt(discordance - difference * difference)
    )
    return ceil((numerator / difference) ** 2)


def worst_case_proportion_n(
    *,
    half_width: float = 0.04,
    confidence: float = 0.95,
) -> int:
    """Conservative binomial sample size using the p=0.5 variance bound."""
    if not 0 < half_width < 0.5:
        raise ValueError("half_width must be between zero and 0.5")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    z_value = NormalDist().inv_cdf(0.5 + confidence / 2)
    return ceil(z_value * z_value / (4 * half_width * half_width))


def balanced_target(minimum: int, *, strata: int) -> tuple[int, int]:
    """Round a required sample upward to equal-sized preregistered strata."""
    if minimum < 1:
        raise ValueError("minimum must be positive")
    if strata < 1:
        raise ValueError("strata must be positive")
    per_stratum = ceil(minimum / strata)
    return per_stratum * strata, per_stratum
