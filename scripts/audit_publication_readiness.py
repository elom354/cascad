#!/usr/bin/env python3
"""Audit the evidence boundary relevant to a Cascad paper submission."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from hashlib import sha256
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--controlled-records",
        default=(
            "runs/real-agent-confirmatory-v2-controlled/"
            "controlled_records.json"
        ),
    )
    parser.add_argument(
        "--llm-summary",
        default=(
            "runs/real-agent-confirmatory-v2-controlled/"
            "llm_attribution_summary.json"
        ),
    )
    parser.add_argument("--hf-results")
    parser.add_argument("--out")
    args = parser.parse_args()

    controlled = _read_json(Path(args.controlled_records))
    llm_summary = _read_json(Path(args.llm_summary))
    report = build_report(controlled, llm_summary)
    if args.hf_results:
        report["huggingface_run"] = audit_huggingface_run(
            Path(args.hf_results)
        )
    encoded = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        destination = Path(args.out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


def build_report(
    controlled: list[dict[str, Any]],
    llm_summary: dict[str, Any],
) -> dict[str, Any]:
    roots = Counter(row["injection_node"] for row in controlled)
    tasks = Counter(row["task_family"] for row in controlled)
    configurations = Counter(row["configuration_id"] for row in controlled)
    faults = Counter(row["fault_family"] for row in controlled)
    deepseek = (llm_summary.get("methods") or {}).get("deepseek")
    return {
        "verdict": "NOT_READY_FOR_BROAD_CONFIRMATORY_CLAIMS",
        "controlled_pairs": len(controlled),
        "independent_pair_definition": (
            "one distinct preregistered task instance with one matched clean/"
            "perturbed trajectory pair"
        ),
        "coverage": {
            "agent_model_families": 1,
            "configurations": dict(configurations),
            "task_families": dict(tasks),
            "fault_families": dict(faults),
            "injection_nodes": dict(roots),
        },
        "difficulty_signal": {
            "deepseek_root_accuracy": (
                deepseek.get("root_accuracy") if deepseek else None
            ),
            "interpretation": (
                "A high paired attribution score and only two injected roots "
                "indicate a narrow/easy attribution distribution; increasing "
                "repetitions of the same cells does not fix this."
            ),
        },
        "confirmed_strengths": [
            "matched clean/perturbed intervention-defined ground truth",
            "no exclusion of absorbed, recovered, or divergent pairs",
            "exact paired test and Wilson intervals",
            "frozen raw traces and integrity manifests",
            "held-out specificity split",
        ],
        "blocking_items": [
            "V2 clock normalization was amended after outcome inspection",
            "one agent-model family only",
            "two injected root nodes and incomplete factorial crossing",
            "no prospective V2 power analysis",
            "matched compact-v2 API baselines are not yet complete",
            "fresh held-out V3 collection has not been executed",
        ],
    }


def audit_huggingface_run(directory: Path) -> dict[str, Any]:
    manifest = _read_json(directory / "integrity_manifest.json")
    mismatches = []
    for relative, expected in manifest["artifacts"].items():
        path = directory / relative
        if (
            not path.is_file()
            or sha256(path.read_bytes()).hexdigest() != expected["sha256"]
            or path.stat().st_size != expected["bytes"]
        ):
            mismatches.append(relative)
    summary = _read_json(directory / "summary.json")
    raw = [
        json.loads(line)
        for line in (directory / "raw_results.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    statuses = Counter(row["status"] for row in raw)
    completed = [row for row in raw if row["status"] == "completed"]
    errors = [row for row in raw if row["status"] == "error"]
    return {
        "integrity": "PASS" if not mismatches else "FAIL",
        "integrity_mismatches": mismatches,
        "study_complete": summary.get("study_complete"),
        "status_counts": dict(statuses),
        "invalid_parse_count": sum(
            not row.get("parse_valid", False) for row in completed
        ),
        "error_types": dict(
            Counter(row.get("error", {}).get("type") for row in errors)
        ),
        "acceptance_decision": "EXCLUDE_INVALID_RUNTIME",
        "reason": (
            "Every completed response was unparsable and every multi-step "
            "instance failed with an out-of-memory error."
        ),
    }


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
