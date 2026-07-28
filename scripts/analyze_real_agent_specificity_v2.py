"""Evaluate held-out clean/clean false positives without root accuracy."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from typing import Any

from cascad.adapters.react_agent import (
    aligned_divergences,
    canonical_trace_sha256,
    load_react_agent_trace,
)
from cascad.export import write_csv, write_json
from cascad.real_specificity import specificity_summary


def main() -> None:
    """Apply frozen v2 thresholds to the disjoint specificity split."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="../react-agent/agent_data/real_agent_confirmatory_v2/specificity",
    )
    parser.add_argument(
        "--protocol",
        default="../react-agent/experiments/real_agent_confirmatory_v2/protocol.json",
    )
    parser.add_argument(
        "--thresholds",
        default="../react-agent/experiments/real_agent_confirmatory_v2/threshold_freeze.json",
    )
    parser.add_argument(
        "--out",
        default="runs/real-agent-confirmatory-v2-specificity",
    )
    args = parser.parse_args()
    source = Path(args.source)
    protocol = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    freeze = json.loads(Path(args.thresholds).read_text(encoding="utf-8"))
    _validate_freeze(protocol, freeze)
    rows = _jsonl(source / "execution_manifest.jsonl")
    _validate_collection(rows, source, protocol)
    thresholds = {
        (
            row["configuration_id"],
            row["task_family"],
            row["event_key"],
        ): row["threshold"]
        for row in freeze["thresholds"]
    }
    records = []
    integrity = []
    for instance_id, pair in sorted(_pairs(rows).items()):
        left = load_react_agent_trace(pair["clean_a"]["raw_trace_path"])
        right = load_react_agent_trace(pair["clean_b"]["raw_trace_path"])
        event_results = []
        false_nodes = set()
        for item in aligned_divergences(left, right):
            key = (
                pair["clean_a"]["configuration_id"],
                pair["clean_a"]["task_family"],
                item.event_key,
            )
            if key not in thresholds:
                raise RuntimeError(f"missing frozen threshold: {key}")
            exceeds = item.distance > thresholds[key]
            if exceeds:
                false_nodes.add(item.node_id)
            event_results.append(
                {
                    **asdict(item),
                    "threshold": thresholds[key],
                    "exceeds_threshold": exceeds,
                }
            )
        records.append(
            {
                "instance_id": instance_id,
                "configuration_id": pair["clean_a"]["configuration_id"],
                "task_family": pair["clean_a"]["task_family"],
                "root_accuracy": None,
                "root_accuracy_reason": "not_applicable_clean_clean",
                "clean_pair_false_positive": bool(false_nodes),
                "falsely_contaminated_nodes": sorted(false_nodes),
                "event_threshold_results": event_results,
            }
        )
        integrity.append(
            {
                "instance_id": instance_id,
                "instance_spec_sha256": pair["clean_a"]["instance_spec_sha256"],
                "raw_clean_a_trace_sha256": pair["clean_a"]["raw_trace_sha256"],
                "raw_clean_b_trace_sha256": pair["clean_b"]["raw_trace_sha256"],
                "canonical_clean_a_trace_sha256": canonical_trace_sha256(left),
                "canonical_clean_b_trace_sha256": canonical_trace_sha256(right),
            }
        )
    summary = specificity_summary(records)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "specificity_records.json", records)
    write_json(out / "specificity_summary.json", summary)
    write_csv(out / "specificity_pair_summary.csv", _flat_pairs(records))
    write_json(out / "post_collection_integrity.json", integrity)
    write_json(
        out / "audit.json",
        {
            "verdict": "PASS",
            "protocol_sha256": protocol["integrity"]["protocol_sha256"],
            "threshold_source": "v2_calibration_only",
            "root_accuracy_computed": False,
            "specificity_pairs": len(records),
        },
    )
    _integrity(out)


def _validate_freeze(
    protocol: dict[str, Any],
    freeze: dict[str, Any],
) -> None:
    if not (
        freeze.get("verdict") == "PASS"
        and freeze.get("thresholds_frozen") is True
        and freeze.get("heldout_data_accessed") is False
        and freeze.get("protocol_sha256")
        == protocol["integrity"]["protocol_sha256"]
    ):
        raise RuntimeError("invalid v2 threshold freeze")


def _validate_collection(
    rows: list[dict[str, Any]],
    source: Path,
    protocol: dict[str, Any],
) -> None:
    if len(rows) != 120:
        raise RuntimeError(f"expected 120 specificity records, observed {len(rows)}")
    if any(row["split"] != "specificity" for row in rows):
        raise RuntimeError("non-specificity record present")
    if any(row["fault_event_count"] for row in rows):
        raise RuntimeError("specificity split contains an injected fault")
    if any(row["status"] != "completed" for row in rows):
        raise RuntimeError("specificity split contains incomplete execution")
    for row in rows:
        path = Path(row["raw_trace_path"])
        if sha256(path.read_bytes()).hexdigest() != row["raw_trace_sha256"]:
            raise RuntimeError(f"raw specificity hash mismatch: {path}")
    manifest = json.loads(
        (source / "collection_manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("complete_pairs") != protocol["specificity"]["total_pairs"]:
        raise RuntimeError("specificity collection incomplete")


def _pairs(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    output: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        output.setdefault(row["instance_id"], {})[row["condition"]] = row
    return output


def _flat_pairs(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "instance_id": row["instance_id"],
            "configuration_id": row["configuration_id"],
            "task_family": row["task_family"],
            "clean_pair_false_positive": row["clean_pair_false_positive"],
            "false_node_count": len(row["falsely_contaminated_nodes"]),
        }
        for row in records
    ]


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _integrity(out: Path) -> None:
    destination = out / "integrity_manifest.json"
    artifacts = {}
    for path in sorted(out.rglob("*")):
        if path.is_file() and path != destination:
            artifacts[str(path.relative_to(out))] = {
                "sha256": sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            }
    write_json(destination, {"hash_algorithm": "SHA-256", "artifacts": artifacts})


if __name__ == "__main__":
    main()
