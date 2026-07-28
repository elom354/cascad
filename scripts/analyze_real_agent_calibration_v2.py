"""Audit v2 clean/clean calibration and freeze all covered event thresholds."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from typing import Any

from cascad.adapters.react_agent import (
    aligned_divergences,
    canonical_trace_sha256,
    load_react_agent_trace,
)
from cascad.divergence import encoder_status
from cascad.export import write_csv, write_json
from cascad.real_calibration import freeze_threshold


def main() -> None:
    """Use only v2 calibration data, verify coverage, and freeze thresholds."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="../react-agent/agent_data/real_agent_confirmatory_v2/calibration",
    )
    parser.add_argument(
        "--protocol",
        default="../react-agent/experiments/real_agent_confirmatory_v2/protocol.json",
    )
    parser.add_argument(
        "--out",
        default="runs/real-agent-confirmatory-v2-calibration",
    )
    parser.add_argument(
        "--freeze-output",
        default="../react-agent/experiments/real_agent_confirmatory_v2/threshold_freeze.json",
    )
    args = parser.parse_args()
    source = Path(args.source).resolve()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    rows = _jsonl(source / "execution_manifest.jsonl")
    audit = _audit(rows, source, protocol)
    if audit["verdict"] != "PASS":
        write_json(out / "calibration_audit.json", audit)
        raise RuntimeError(f"v2 calibration audit FAIL: {audit['errors']}")
    snapshot = _snapshot(rows, out / "raw")
    paired = _pairs(snapshot)
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    divergence_rows = []
    canonical_rows = []
    for instance_id, conditions in sorted(paired.items()):
        left = load_react_agent_trace(conditions["clean_a"]["snapshot_path"])
        right = load_react_agent_trace(conditions["clean_b"]["snapshot_path"])
        canonical_rows.append(
            {
                "instance_id": instance_id,
                "instance_spec_sha256": conditions["clean_a"][
                    "instance_spec_sha256"
                ],
                "raw_clean_a_trace_sha256": conditions["clean_a"][
                    "snapshot_sha256"
                ],
                "raw_clean_b_trace_sha256": conditions["clean_b"][
                    "snapshot_sha256"
                ],
                "canonical_clean_a_trace_sha256": canonical_trace_sha256(left),
                "canonical_clean_b_trace_sha256": canonical_trace_sha256(right),
            }
        )
        for item in aligned_divergences(left, right):
            group = (
                conditions["clean_a"]["configuration_id"],
                conditions["clean_a"]["task_family"],
                item.event_key,
            )
            grouped[group].append(item.distance)
            divergence_rows.append(
                {
                    "instance_id": instance_id,
                    "configuration_id": group[0],
                    "task_family": group[1],
                    **asdict(item),
                }
            )
    thresholds = [
        asdict(freeze_threshold(*group, values))
        for group, values in sorted(grouped.items())
    ]
    coverage = _coverage(protocol, thresholds)
    if coverage["verdict"] != "PASS":
        write_json(out / "coverage_audit.json", coverage)
        raise RuntimeError(f"v2 threshold coverage FAIL: {coverage['missing']}")
    encoder = encoder_status()
    freeze = {
        "verdict": "PASS",
        "thresholds_frozen": True,
        "protocol_sha256": protocol["integrity"]["protocol_sha256"],
        "source_split": "v2_calibration_clean_clean_only",
        "heldout_data_accessed": False,
        "formula": (
            "mean(clean_clean_distance) + sample_stddev(clean_clean_distance) "
            "+ epsilon"
        ),
        "epsilon_policy": (
            "max(0, calibration_max - mean - sample_stddev) + 1e-9"
        ),
        "grouping": ["configuration_id", "task_family", "event_key"],
        "threshold_count": len(thresholds),
        "encoder_used": encoder["encoder_used"],
        "encoder_reason": encoder["reason"],
        "coverage_verdict": coverage["verdict"],
        "thresholds": thresholds,
    }
    write_json(out / "calibration_audit.json", audit)
    write_json(out / "coverage_audit.json", coverage)
    write_json(out / "calibration_manifest.json", snapshot)
    write_json(out / "post_collection_integrity.json", canonical_rows)
    write_json(out / "divergences.json", divergence_rows)
    write_csv(out / "divergences.csv", divergence_rows)
    write_json(out / "thresholds.json", thresholds)
    write_csv(out / "thresholds.csv", thresholds)
    write_json(out / "threshold_freeze.json", freeze)
    _write_reproducible(Path(args.freeze_output), freeze)
    _integrity(out)


def _audit(
    rows: list[dict[str, Any]],
    source: Path,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    errors = []
    expected = protocol["integrity"]["protocol_sha256"]
    if len(rows) != 120:
        errors.append(f"expected 120 records, observed {len(rows)}")
    if any(row["protocol_sha256"] != expected for row in rows):
        errors.append("protocol hash mismatch")
    if any(row["split"] != "calibration" for row in rows):
        errors.append("non-calibration record present")
    if any(row["fault_event_count"] != 0 for row in rows):
        errors.append("fault event present")
    if any(row["status"] != "completed" for row in rows):
        errors.append("incomplete execution present")
    hash_failures = []
    conditions: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        conditions[row["instance_id"]].add(row["condition"])
        path = Path(row["raw_trace_path"])
        if not path.is_file() or sha256(path.read_bytes()).hexdigest() != row[
            "raw_trace_sha256"
        ]:
            hash_failures.append(str(path))
    if hash_failures:
        errors.append(f"{len(hash_failures)} raw hash failures")
    if len(conditions) != 60 or any(
        value != {"clean_a", "clean_b"} for value in conditions.values()
    ):
        errors.append("incomplete clean/clean pairs")
    collection = json.loads(
        (source / "collection_manifest.json").read_text(encoding="utf-8")
    )
    if not collection.get("collection_complete"):
        errors.append("collection manifest incomplete")
    return {
        "verdict": "PASS" if not errors else "FAIL",
        "errors": errors,
        "protocol_sha256": expected,
        "execution_records": len(rows),
        "complete_pairs": len(conditions),
        "hash_failures": hash_failures,
        "fault_events": sum(row["fault_event_count"] for row in rows),
        "heldout_data_accessed": False,
    }


def _coverage(
    protocol: dict[str, Any],
    thresholds: list[dict[str, Any]],
) -> dict[str, Any]:
    available: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in thresholds:
        available[(row["configuration_id"], row["task_family"])].add(
            row["event_key"]
        )
    required: dict[tuple[str, str], set[str]] = defaultdict(set)
    for split in ("specificity", "controlled"):
        for row in protocol[split]["instances"]:
            required[(row["configuration_id"], row["task_family"])].update(
                row["expected_event_keys"]
            )
    missing = []
    for stratum, keys in sorted(required.items()):
        for key in sorted(keys - available.get(stratum, set())):
            missing.append(
                {
                    "configuration_id": stratum[0],
                    "task_family": stratum[1],
                    "event_key": key,
                }
            )
    return {
        "verdict": "PASS" if not missing else "FAIL",
        "required_strata": len(required),
        "available_thresholds": len(thresholds),
        "missing": missing,
    }


def _snapshot(
    rows: list[dict[str, Any]],
    destination: Path,
) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        source = Path(row["raw_trace_path"])
        target = destination / row["instance_id"] / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and sha256(target.read_bytes()).hexdigest() != row[
            "raw_trace_sha256"
        ]:
            raise RuntimeError(f"refusing to overwrite changed snapshot: {target}")
        if not target.exists():
            shutil.copy2(source, target)
        output.append(
            {
                **row,
                "source_path": str(source),
                "snapshot_path": str(target.resolve()),
                "snapshot_sha256": sha256(target.read_bytes()).hexdigest(),
            }
        )
    return output


def _pairs(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    output: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        output[row["instance_id"]][row["condition"]] = row
    return dict(output)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_reproducible(path: Path, value: dict[str, Any]) -> None:
    content = json.dumps(value, indent=2, ensure_ascii=False) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != content:
        raise RuntimeError(f"refusing to overwrite changed threshold freeze: {path}")
    path.write_text(content, encoding="utf-8")


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
