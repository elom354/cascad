#!/usr/bin/env python3
"""Reparse immutable Hugging Face raw responses after a parser correction."""

from __future__ import annotations

import argparse
import csv
import json
from hashlib import sha256
from pathlib import Path
from statistics import mean
from typing import Any

from cascad.attribution_baseline import parse_attribution_node
from cascad.statistics import paired_correctness, wilson_interval


PARSER_VERSION = "candidate-literal-v2"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    source = Path(args.source)
    out = Path(args.out)
    integrity = verify_integrity(source)
    records = [
        json.loads(line)
        for line in (source / "raw_results.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    reparsed = []
    for record in records:
        if record.get("status") != "completed":
            reparsed.append(record)
            continue
        predicted = parse_attribution_node(
            record["raw_response"],
            tuple(record["candidate_nodes"]),
        )
        reparsed.append(
            {
                **record,
                "original_parsed_node": record.get("parsed_node"),
                "original_parse_valid": record.get("parse_valid"),
                "original_correct": record.get("correct"),
                "parsed_node": predicted,
                "parse_valid": predicted is not None,
                "correct": predicted == record["ground_truth_root"],
                "parser_version": PARSER_VERSION,
            }
        )

    completed = [
        record for record in reparsed if record.get("status") == "completed"
    ]
    correct = sum(bool(record["correct"]) for record in completed)
    low, high = wilson_interval(correct, len(completed))
    summary = {
        "analysis_role": "post_hoc_parser_bug_correction",
        "source_directory": str(source),
        "source_raw_results_sha256": sha256(
            (source / "raw_results.jsonl").read_bytes()
        ).hexdigest(),
        "source_integrity": integrity,
        "parser_version": PARSER_VERSION,
        "n": len(completed),
        "correct": correct,
        "root_accuracy": correct / len(completed),
        "wilson_95": [low, high],
        "invalid_parse_count": sum(
            not record["parse_valid"] for record in completed
        ),
        "original_invalid_parse_count": sum(
            not record["original_parse_valid"] for record in completed
        ),
        "predictions_changed": sum(
            record["parsed_node"] != record["original_parsed_node"]
            for record in completed
        ),
        "graph_vs_model_paired": paired_correctness(
            [bool(record["graph_correct"]) for record in completed],
            [bool(record["correct"]) for record in completed],
        ),
        "mean_latency_ms": mean(
            record["latency_ms"] for record in completed
        ),
        "interpretation": (
            "The raw model outputs are unchanged. This is a post-hoc "
            "correction of a parser that failed to recognize candidate IDs "
            "containing '::' inside otherwise accepted formatting."
        ),
    }
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / "reparsed_records.json", reparsed)
    _write_csv(out / "reparsed_records.csv", reparsed)
    _write_json(out / "reparsed_summary.json", summary)
    _write_integrity(out)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def verify_integrity(source: Path) -> str:
    manifest = json.loads(
        (source / "integrity_manifest.json").read_text(encoding="utf-8")
    )
    failures = []
    for relative, expected in manifest["artifacts"].items():
        path = source / relative
        if (
            not path.is_file()
            or sha256(path.read_bytes()).hexdigest() != expected["sha256"]
            or path.stat().st_size != expected["bytes"]
        ):
            failures.append(relative)
    if failures:
        raise RuntimeError(
            f"source integrity verification failed: {failures}"
        )
    return "PASS"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flat = []
    for record in rows:
        flat.append(
            {
                key: (
                    json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (dict, list))
                    else value
                )
                for key, value in record.items()
            }
        )
    columns = list(
        dict.fromkeys(key for record in flat for key in record)
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(flat)


def _write_integrity(out: Path) -> None:
    destination = out / "integrity_manifest.json"
    artifacts = {}
    for path in sorted(out.iterdir()):
        if path.is_file() and path != destination:
            artifacts[path.name] = {
                "sha256": sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            }
    _write_json(
        destination,
        {"hash_algorithm": "SHA-256", "artifacts": artifacts},
    )


if __name__ == "__main__":
    main()
