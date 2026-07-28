"""Specificity metrics for held-out clean/clean real-agent pairs."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any

from cascad.statistics import wilson_interval


def specificity_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate clean-pair false positives globally and by frozen strata."""
    if not records:
        raise ValueError("specificity records must not be empty")
    return {
        "root_accuracy": "not_applicable_clean_clean",
        "global": _pair_rate(records),
        "by_configuration": _group_pair_rates(records, "configuration_id"),
        "by_application_family": _group_pair_rates(records, "task_family"),
        "by_event_key": _event_rates(records),
        "false_subgraph_pair_rate": _pair_rate(records),
        "mean_falsely_contaminated_nodes": mean(
            len(row["falsely_contaminated_nodes"]) for row in records
        ),
    }


def _pair_rate(records: list[dict[str, Any]]) -> dict[str, Any]:
    positives = sum(bool(row["clean_pair_false_positive"]) for row in records)
    total = len(records)
    return {
        "false_positive_pairs": positives,
        "denominator_pairs": total,
        "rate": positives / total,
        "wilson_95": wilson_interval(positives, total),
    }


def _group_pair_rates(
    records: list[dict[str, Any]],
    field: str,
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[str(row[field])].append(row)
    return [
        {field: key, **_pair_rate(items)}
        for key, items in sorted(groups.items())
    ]


def _event_rates(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[bool]] = defaultdict(list)
    for row in records:
        for event in row["event_threshold_results"]:
            groups[event["event_key"]].append(bool(event["exceeds_threshold"]))
    output = []
    for key, values in sorted(groups.items()):
        positives = sum(values)
        total = len(values)
        output.append(
            {
                "event_key": key,
                "false_positive_events": positives,
                "denominator_events": total,
                "rate": positives / total,
                "wilson_95": wilson_interval(positives, total),
            }
        )
    return output
