"""Metrics for the corrected real-agent controlled confirmatory split."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any

from cascad.statistics import wilson_interval


def controlled_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize source localization and retained propagation outcomes."""
    if not records:
        raise ValueError("controlled records cannot be empty")
    return {
        "global": _stratum(records),
        "by_configuration": _group(records, "configuration_id"),
        "by_task_family": _group(records, "task_family"),
        "by_fault_family": _group(records, "fault_family"),
        "by_propagation_opportunity": _group(
            records,
            "propagation_opportunity",
        ),
        "by_realized_path_length": _group(records, "realized_path_length"),
        "outcome_counts": _counts(records, "primary_class"),
        "alignment": {
            "aligned_pairs": sum(row["trajectory_aligned"] for row in records),
            "structurally_divergent_pairs": sum(
                not row["trajectory_aligned"] for row in records
            ),
            "pairs_with_uncalibrated_realized_extensions": sum(
                row["uncalibrated_realized_extension_count"] > 0
                for row in records
            ),
        },
    }


def _group(
    records: list[dict[str, Any]],
    field: str,
) -> list[dict[str, Any]]:
    grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[row[field]].append(row)
    return [
        {field: value, **_stratum(rows)}
        for value, rows in sorted(grouped.items(), key=lambda item: str(item[0]))
    ]


def _stratum(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    correct = sum(row["graph_correct"] for row in records)
    low, high = wilson_interval(correct, total)
    return {
        "n": total,
        "graph_root_correct": correct,
        "graph_root_accuracy": correct / total,
        "graph_root_accuracy_wilson_95": [low, high],
        "realized_propagation_rate": sum(
            row["realized_propagation"] for row in records
        )
        / total,
        "absorption_rate": sum(row["absorbed_by_agent"] for row in records)
        / total,
        "persistence_rate": sum(row["propagated_to_memory"] for row in records)
        / total,
        "final_failure_rate": sum(row["final_failure"] for row in records)
        / total,
        "successful_recovery_rate": sum(
            row["successful_recovery"] for row in records
        )
        / total,
        "mean_observed_depth": mean(row["observed_depth"] for row in records),
        "mean_subgraph_precision": mean(
            row["subgraph_precision"] for row in records
        ),
        "mean_subgraph_recall": mean(
            row["subgraph_recall"] for row in records
        ),
        "repeated_exposure_pairs": sum(
            row["fault_event_count"] > 1 for row in records
        ),
    }


def _counts(
    records: list[dict[str, Any]],
    field: str,
) -> dict[str, int]:
    output: dict[str, int] = defaultdict(int)
    for row in records:
        output[str(row[field])] += 1
    return dict(sorted(output.items()))
