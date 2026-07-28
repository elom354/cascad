"""Predeclared statistical policy for Cascad experimental reports."""

from __future__ import annotations

from hashlib import sha256
from math import comb, sqrt
from random import Random
from statistics import mean
from typing import Iterable


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a binary proportion."""
    if total < 1 or successes < 0 or successes > total:
        raise ValueError("require 0 <= successes <= total and total >= 1")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def exact_mcnemar(
    both_correct: int,
    a_correct_b_wrong: int,
    a_wrong_b_correct: int,
    both_wrong: int,
) -> dict[str, float | int]:
    """Complete paired 2x2 table with the exact two-sided binomial p-value."""
    values = (both_correct, a_correct_b_wrong, a_wrong_b_correct, both_wrong)
    if any(value < 0 for value in values):
        raise ValueError("McNemar counts cannot be negative")
    discordant = a_correct_b_wrong + a_wrong_b_correct
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(comb(discordant, index) for index in range(min(
            a_correct_b_wrong, a_wrong_b_correct
        ) + 1)) / (2 ** discordant)
        p_value = min(1.0, 2.0 * tail)
    total = sum(values)
    return {
        "both_correct": both_correct,
        "a_correct_b_wrong": a_correct_b_wrong,
        "a_wrong_b_correct": a_wrong_b_correct,
        "both_wrong": both_wrong,
        "discordant_count": discordant,
        "paired_total": total,
        "accuracy_difference_a_minus_b": (
            (a_correct_b_wrong - a_wrong_b_correct) / total if total else 0.0
        ),
        "exact_two_sided_p_value": p_value,
    }


def paired_correctness(a: Iterable[bool], b: Iterable[bool]) -> dict[str, float | int]:
    """Construct an exact McNemar result from paired correctness vectors."""
    left, right = list(a), list(b)
    if len(left) != len(right) or not left:
        raise ValueError("paired correctness vectors must be non-empty and equal")
    return exact_mcnemar(
        sum(x and y for x, y in zip(left, right)),
        sum(x and not y for x, y in zip(left, right)),
        sum(not x and y for x, y in zip(left, right)),
        sum(not x and not y for x, y in zip(left, right)),
    )


def bootstrap_interval(
    values: Iterable[float],
    *,
    seed: str,
    samples: int = 2000,
) -> tuple[float, float]:
    """Deterministic percentile bootstrap interval for continuous/composite metrics."""
    values = list(values)
    if not values:
        raise ValueError("bootstrap values cannot be empty")
    random = Random(int.from_bytes(sha256(seed.encode()).digest()[:8], "big"))
    estimates = sorted(
        mean(random.choice(values) for _ in values) for _ in range(samples)
    )
    low = max(0, int(0.025 * samples) - 1)
    high = min(samples - 1, int(0.975 * samples) - 1)
    return estimates[low], estimates[high]
