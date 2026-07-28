"""Analyze all retained v2 controlled real-agent pairs without exclusions."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from statistics import mean
from typing import Any

from cascad.adapters.react_agent import (
    aligned_divergences_with_structure,
    canonical_trace_sha256,
    load_react_agent_trace,
)
from cascad.attribution_baseline import (
    DeepSeekAttributor,
    OpenAIAttributor,
    attribute_failure_detailed,
    configured_attributors,
)
from cascad.export import write_csv, write_json
from cascad.real_controlled import controlled_summary
from cascad.real_propagation import classify_propagation
from cascad.statistics import paired_correctness, wilson_interval


def main() -> None:
    """Validate, snapshot, analyze, and optionally attribute the controlled split."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="../react-agent/agent_data/real_agent_confirmatory_v2/controlled",
    )
    parser.add_argument(
        "--protocol",
        default="../react-agent/experiments/real_agent_confirmatory_v2/protocol.json",
    )
    parser.add_argument(
        "--thresholds",
        default="../react-agent/experiments/real_agent_confirmatory_v2/threshold_freeze.json",
    )
    parser.add_argument("--env-file", default=".env")
    parser.add_argument(
        "--attribution",
        choices=["none", "deepseek", "openai", "auto", "both"],
        default="none",
    )
    parser.add_argument(
        "--out",
        default="runs/real-agent-confirmatory-v2-controlled",
    )
    args = parser.parse_args()

    source = Path(args.source).resolve()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    protocol = _read_json(Path(args.protocol))
    freeze = _read_json(Path(args.thresholds))
    _validate_freeze(protocol, freeze)
    attempts = _jsonl(source / "execution_manifest.jsonl")
    selected, attempt_audit = _select_successful_pairs(
        attempts,
        source,
        protocol,
    )
    snapshots = _snapshot(selected, out / "raw")
    thresholds = {
        (
            row["configuration_id"],
            row["task_family"],
            row["event_key"],
        ): row["threshold"]
        for row in freeze["thresholds"]
    }

    records = []
    divergence_rows = []
    integrity_rows = []
    trace_pairs = {}
    protocol_instances = {
        row["instance_id"]: row for row in protocol["controlled"]["instances"]
    }
    for instance_id, pair in sorted(snapshots.items()):
        clean = load_react_agent_trace(pair["clean"]["snapshot_path"])
        perturbed = load_react_agent_trace(pair["perturbed"]["snapshot_path"])
        trace_pairs[instance_id] = (clean, perturbed)
        instance = protocol_instances[instance_id]
        record, divergences = _analyze_pair(
            instance,
            pair,
            clean,
            perturbed,
            thresholds,
        )
        records.append(record)
        divergence_rows.extend(divergences)
        integrity_rows.append(
            {
                "instance_id": instance_id,
                "instance_spec_sha256": instance["instance_spec_sha256"],
                "raw_clean_trace_sha256": pair["clean"]["snapshot_sha256"],
                "raw_perturbed_trace_sha256": pair["perturbed"][
                    "snapshot_sha256"
                ],
                "canonical_clean_trace_sha256": canonical_trace_sha256(clean),
                "canonical_perturbed_trace_sha256": canonical_trace_sha256(
                    perturbed
                ),
            }
        )

    summary = controlled_summary(records)
    write_json(out / "attempt_audit.json", attempt_audit)
    write_json(out / "controlled_records.json", records)
    write_csv(out / "controlled_records.csv", [_flat_record(row) for row in records])
    write_json(out / "divergences.json", divergence_rows)
    write_csv(out / "divergences.csv", divergence_rows)
    write_json(out / "controlled_summary.json", summary)
    write_json(out / "post_collection_integrity.json", integrity_rows)
    write_json(
        out / "analysis_audit.json",
        {
            "verdict": "PASS",
            "protocol_sha256": protocol["integrity"]["protocol_sha256"],
            "controlled_pairs": len(records),
            "excluded_pairs": 0,
            "failed_attempts_retained": attempt_audit["failed_attempts"],
            "repeated_exposure_pairs_retained": attempt_audit[
                "repeated_exposure_pairs"
            ],
            "structurally_divergent_pairs_retained": summary["alignment"][
                "structurally_divergent_pairs"
            ],
            "threshold_source": "v2_calibration_only",
            "structural_divergence_policy": (
                "presence/absence of an observable event is a categorical "
                "trajectory divergence; matched extensions without a frozen "
                "key are flagged as uncalibrated rather than silently thresholded"
            ),
        },
    )
    if args.attribution != "none":
        clients = _attribution_clients(args.attribution, args.env_file)
        llm_records = _run_attribution(
            out,
            clients,
            trace_pairs,
            records,
        )
        write_json(out / "llm_attribution_records.json", llm_records)
        write_csv(
            out / "llm_attribution_records.csv",
            [_flat_llm(row) for row in llm_records if row["status"] == "completed"],
        )
        write_json(
            out / "llm_attribution_summary.json",
            _llm_summary(llm_records),
        )
    _integrity(out)


def _analyze_pair(
    instance: dict[str, Any],
    pair: dict[str, dict[str, Any]],
    clean: Any,
    perturbed: Any,
    thresholds: dict[tuple[str, str, str], float],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    divergences = aligned_divergences_with_structure(clean, perturbed)
    event_rows = []
    predicted_nodes = []
    reference_nodes = []
    uncalibrated_extensions = 0
    for item in divergences:
        threshold_key = (
            instance["configuration_id"],
            instance["task_family"],
            item.event_key,
        )
        threshold = thresholds.get(threshold_key)
        if item.alignment_status != "matched":
            affected = True
            threshold_source = "structural_presence_absence"
        elif threshold is None:
            affected = item.distance > 0
            threshold_source = "uncalibrated_realized_extension"
            uncalibrated_extensions += 1
        else:
            affected = item.distance > threshold
            threshold_source = "frozen_v2_calibration"
        if item.distance > 0:
            reference_nodes.append(item.node_id)
        if affected:
            predicted_nodes.append(item.node_id)
        event_rows.append(
            {
                "instance_id": instance["instance_id"],
                **asdict(item),
                "threshold": threshold,
                "threshold_source": threshold_source,
                "declared_contaminated": affected,
            }
        )

    contaminated = list(dict.fromkeys(predicted_nodes))
    reference = set(reference_nodes)
    predicted = set(contaminated)
    injection_node = pair["perturbed"]["injection_node"]
    graph_prediction = contaminated[0] if contaminated else None
    downstream = [node for node in contaminated if node != injection_node]
    memory_nodes = {
        row["node_id"]
        for row in event_rows
        if row["declared_contaminated"]
        and (
            row["event_kind"] in {"memory_read", "memory_write"}
            or "memory" in row["node_id"]
            or row["node_id"] in {"tool::remember", "tool::search_memory"}
        )
    }
    persistence_realized = bool(
        pair["perturbed"]["memory_write_realized"]
        and pair["perturbed"]["memory_tool_read_realized"]
    )
    final_response_affected = _last_model_response_affected(event_rows)
    explicit_detection = _explicit_detection(
        pair["perturbed"].get("final_response", "")
    )
    outcome = classify_propagation(
        structurally_multi_step=(
            instance["propagation_opportunity"] == "multi_step"
        ),
        persistence_opportunity_realized=persistence_realized,
        memory_contaminated=bool(memory_nodes),
        downstream_contaminated=bool(downstream),
        explicit_detection=explicit_detection,
        final_failure=final_response_affected,
    )
    true_positive = len(predicted & reference)
    observed_depth = max(0, len(contaminated) - 1)
    record = {
        "instance_id": instance["instance_id"],
        "configuration_id": instance["configuration_id"],
        "task_family": instance["task_family"],
        "fault_family": instance["fault_family"],
        "propagation_opportunity": instance["propagation_opportunity"],
        "injection_node": injection_node,
        "fault_event_count": pair["perturbed"]["fault_event_count"],
        "repeated_fault_exposure": pair["perturbed"]["fault_event_count"] > 1,
        "trajectory_aligned": all(
            item.alignment_status == "matched" for item in divergences
        ),
        "structural_divergence_event_count": sum(
            item.alignment_status != "matched" for item in divergences
        ),
        "uncalibrated_realized_extension_count": uncalibrated_extensions,
        "graph_predicted_node": graph_prediction,
        "graph_correct": graph_prediction == injection_node,
        "contaminated_nodes": contaminated,
        "reference_observably_changed_nodes": sorted(reference),
        "subgraph_precision": true_positive / max(1, len(predicted)),
        "subgraph_recall": true_positive / max(1, len(reference)),
        "realized_propagation": bool(downstream),
        "observed_depth": observed_depth,
        "realized_path_length": observed_depth + 1 if contaminated else 0,
        "memory_contaminated_nodes": sorted(memory_nodes),
        "persistence_opportunity_realized": persistence_realized,
        "explicit_detection_heuristic": explicit_detection,
        **asdict(outcome),
    }
    return record, event_rows


def _select_successful_pairs(
    attempts: list[dict[str, Any]],
    source: Path,
    protocol: dict[str, Any],
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
    selected: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    failed_attempts = []
    repeated = set()
    for index, row in enumerate(attempts):
        acceptable = _acceptable(row)
        if not acceptable:
            failed_attempts.append(
                {
                    "attempt_index": index,
                    "instance_id": row["instance_id"],
                    "condition": row["condition"],
                    "status": row["status"],
                    "fault_event_count": row["fault_event_count"],
                    "error": row["error"],
                }
            )
            continue
        key = row["condition"]
        if key in selected[row["instance_id"]]:
            raise RuntimeError(
                f"multiple successful attempts for {row['instance_id']} {key}"
            )
        path = Path(row["raw_trace_path"])
        if sha256(path.read_bytes()).hexdigest() != row["raw_trace_sha256"]:
            raise RuntimeError(f"raw hash mismatch: {path}")
        selected[row["instance_id"]][key] = row
        if key == "perturbed" and row["fault_event_count"] > 1:
            repeated.add(row["instance_id"])
    expected_ids = {
        row["instance_id"] for row in protocol["controlled"]["instances"]
    }
    complete = {
        key
        for key, value in selected.items()
        if set(value) == {"clean", "perturbed"}
    }
    collection = _read_json(source / "collection_manifest.json")
    if complete != expected_ids or not collection.get("collection_complete"):
        raise RuntimeError(
            f"controlled collection incomplete: {len(complete)}/{len(expected_ids)}"
        )
    return dict(selected), {
        "verdict": "PASS",
        "attempt_records": len(attempts),
        "successful_execution_keys": sum(
            len(value) for value in selected.values()
        ),
        "complete_pairs": len(complete),
        "failed_attempts": len(failed_attempts),
        "failed_attempt_records": failed_attempts,
        "repeated_exposure_pairs": len(repeated),
        "repeated_exposure_instance_ids": sorted(repeated),
        "excluded_pairs": 0,
        "policy": (
            "Use one successful clean and perturbed execution per preregistered "
            "instance; retain failed attempts and repeated target exposure."
        ),
    }


def _snapshot(
    selected: dict[str, dict[str, dict[str, Any]]],
    destination: Path,
) -> dict[str, dict[str, dict[str, Any]]]:
    output: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for instance_id, pair in selected.items():
        for condition, row in pair.items():
            source = Path(row["raw_trace_path"])
            target = destination / instance_id / f"{condition}.events.jsonl"
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and sha256(target.read_bytes()).hexdigest() != row[
                "raw_trace_sha256"
            ]:
                raise RuntimeError(f"changed snapshot exists: {target}")
            if not target.exists():
                shutil.copy2(source, target)
            output[instance_id][condition] = {
                **row,
                "snapshot_path": str(target.resolve()),
                "snapshot_sha256": sha256(target.read_bytes()).hexdigest(),
            }
    return dict(output)


def _run_attribution(
    out: Path,
    clients: tuple[DeepSeekAttributor, ...],
    trace_pairs: dict[str, tuple[Any, Any]],
    graph_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    checkpoint_path = out / "llm_attribution_raw.jsonl"
    existing = _jsonl(checkpoint_path) if checkpoint_path.exists() else []
    completed = {
        (row["provider"], row["instance_id"])
        for row in existing
        if row["status"] == "completed"
    }
    records = list(existing)
    graph_by_id = {row["instance_id"]: row for row in graph_records}
    for client in clients:
        for instance_id, (clean, perturbed) in sorted(trace_pairs.items()):
            if (client.provider, instance_id) in completed:
                continue
            graph = graph_by_id[instance_id]
            try:
                result = attribute_failure_detailed(
                    perturbed,
                    client,
                    clean_trace=clean,
                    mode="paired",
                )
                row = {
                    "status": "completed",
                    "provider": client.provider,
                    "model": client.model,
                    "instance_id": instance_id,
                    "ground_truth_root": graph["injection_node"],
                    "visible_symptom": (
                        "call_model" if graph["final_failure"] else None
                    ),
                    "raw_response": result.raw_response,
                    "parsed_node": result.predicted_node,
                    "correct": result.predicted_node == graph["injection_node"],
                    "selected_symptom": (
                        result.predicted_node == "call_model"
                        and graph["final_failure"]
                    ),
                    "graph_prediction": graph["graph_predicted_node"],
                    "graph_correct": graph["graph_correct"],
                    "prompt_sha256": result.prompt_bundle.prompt_sha256,
                    "clean_trace_sha256": result.prompt_bundle.clean_trace_sha256,
                    "perturbed_trace_sha256": (
                        result.prompt_bundle.corrupt_trace_sha256
                    ),
                    "candidate_nodes": list(result.candidates),
                    **(client.last_call_metadata or {}),
                }
                completed.add((client.provider, instance_id))
            except Exception as exc:
                row = {
                    "status": "error",
                    "provider": client.provider,
                    "model": client.model,
                    "instance_id": instance_id,
                    "error": {
                        "type": exc.__class__.__name__,
                        "message": str(exc),
                    },
                    "call_metadata": client.last_call_metadata,
                }
            records.append(row)
            with checkpoint_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return records


def _llm_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in records if row["status"] == "completed"]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in completed:
        grouped[row["provider"]].append(row)
    methods = {}
    for provider, rows in sorted(grouped.items()):
        correct = sum(row["correct"] for row in rows)
        methods[provider] = {
            "model": rows[0]["model"],
            "n": len(rows),
            "correct": correct,
            "root_accuracy": correct / len(rows),
            "wilson_95": list(wilson_interval(correct, len(rows))),
            "symptom_selection_rate": sum(
                row["selected_symptom"] for row in rows
            )
            / len(rows),
            "mean_latency_ms": mean(
                row["latency_ms"]
                for row in rows
                if row.get("latency_ms") is not None
            ),
            "graph_vs_llm_paired": paired_correctness(
                [row["graph_correct"] for row in rows],
                [row["correct"] for row in rows],
            ),
        }
    indexed = {
        (row["provider"], row["instance_id"]): row for row in completed
    }
    paired_ids = sorted(
        instance_id
        for instance_id in {row["instance_id"] for row in completed}
        if ("deepseek", instance_id) in indexed
        and ("openai", instance_id) in indexed
    )
    paired = None
    if paired_ids:
        paired = paired_correctness(
            [indexed[("deepseek", key)]["correct"] for key in paired_ids],
            [indexed[("openai", key)]["correct"] for key in paired_ids],
        )
    return {
        "methods": methods,
        "error_calls": sum(row["status"] == "error" for row in records),
        "deepseek_vs_openai_paired": paired,
        "paired_instance_count": len(paired_ids),
    }


def _attribution_clients(
    mode: str,
    env_file: str,
) -> tuple[DeepSeekAttributor, ...]:
    if mode == "auto":
        return configured_attributors(env_file)
    if mode == "deepseek":
        return (DeepSeekAttributor.from_environment(env_file),)
    if mode == "openai":
        return (OpenAIAttributor.from_environment(env_file),)
    if mode == "both":
        return (
            DeepSeekAttributor.from_environment(env_file),
            OpenAIAttributor.from_environment(env_file),
        )
    raise ValueError(f"unsupported attribution mode: {mode}")


def _last_model_response_affected(rows: list[dict[str, Any]]) -> bool:
    responses = [
        row
        for row in rows
        if row["node_id"] == "call_model"
        and row["event_kind"] == "model_response"
    ]
    return bool(responses and responses[-1]["declared_contaminated"])


def _explicit_detection(response: str) -> bool:
    normalized = response.casefold()
    return any(
        token in normalized
        for token in (
            "error",
            "failed",
            "failure",
            "could not",
            "cannot",
            "unable",
            "invalid",
            "malformed",
        )
    )


def _acceptable(row: dict[str, Any]) -> bool:
    if row["status"] != "completed" or not row["target_tool_observed"]:
        return False
    if row["condition"] == "perturbed":
        return row["fault_event_count"] >= 1
    return row["fault_event_count"] == 0


def _flat_record(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "instance_id",
        "configuration_id",
        "task_family",
        "fault_family",
        "propagation_opportunity",
        "injection_node",
        "fault_event_count",
        "repeated_fault_exposure",
        "trajectory_aligned",
        "graph_predicted_node",
        "graph_correct",
        "primary_class",
        "realized_propagation",
        "propagated_to_memory",
        "absorbed_by_agent",
        "detected_and_blocked",
        "no_persistence_opportunity_realized",
        "final_failure",
        "successful_recovery",
        "observed_depth",
        "realized_path_length",
        "subgraph_precision",
        "subgraph_recall",
    )
    return {key: row[key] for key in keys}


def _flat_llm(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: row.get(key)
        for key in (
            "provider",
            "model",
            "instance_id",
            "ground_truth_root",
            "visible_symptom",
            "parsed_node",
            "correct",
            "selected_symptom",
            "graph_prediction",
            "graph_correct",
            "prompt_sha256",
            "latency_ms",
            "input_tokens",
            "output_tokens",
            "total_tokens",
        )
    }


def _validate_freeze(
    protocol: dict[str, Any],
    freeze: dict[str, Any],
) -> None:
    if not (
        freeze.get("verdict") == "PASS"
        and freeze.get("thresholds_frozen") is True
        and freeze.get("protocol_sha256")
        == protocol["integrity"]["protocol_sha256"]
        and freeze.get("heldout_data_accessed") is False
    ):
        raise RuntimeError("invalid v2 threshold freeze")


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _integrity(out: Path) -> None:
    destination = out / "integrity_manifest.json"
    artifacts = {}
    for path in sorted(out.rglob("*")):
        if path.is_file() and path != destination:
            artifacts[str(path.relative_to(out))] = {
                "sha256": sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            }
    write_json(
        destination,
        {"hash_algorithm": "SHA-256", "artifacts": artifacts},
    )


if __name__ == "__main__":
    main()
