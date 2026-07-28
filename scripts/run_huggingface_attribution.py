#!/usr/bin/env python3
"""Run checkpointed Hugging Face attribution on frozen real-agent traces."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from statistics import mean
from typing import Any

from cascad.adapters.react_agent import load_react_agent_trace
from cascad.attribution_baseline import attribute_failure_detailed
from cascad.huggingface_attribution import (
    DEFAULT_MODEL_ALIASES,
    HuggingFaceAttributor,
    model_spec,
    resolve_model_revision,
)
from cascad.statistics import paired_correctness, wilson_interval


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="runs/real-agent-confirmatory-v2-controlled/raw",
    )
    parser.add_argument(
        "--graph-records",
        default="runs/real-agent-confirmatory-v2-controlled/controlled_records.json",
    )
    parser.add_argument(
        "--deepseek-records",
        default=(
            "runs/real-agent-confirmatory-v2-controlled/"
            "llm_attribution_records.json"
        ),
    )
    parser.add_argument(
        "--out",
        default="runs/real-agent-confirmatory-v2-huggingface",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODEL_ALIASES),
    )
    parser.add_argument(
        "--quantization",
        choices=["4bit", "8bit", "none"],
        default="4bit",
    )
    parser.add_argument(
        "--attention-backend",
        choices=["sdpa", "eager"],
        default="sdpa",
    )
    parser.add_argument(
        "--cache-implementation",
        choices=["dynamic", "offloaded"],
        default="offloaded",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--limit",
        type=int,
        help="Deterministic smoke-test limit; never use for final reporting.",
    )
    args = parser.parse_args()

    source = Path(args.source)
    out = Path(args.out)
    if not source.is_dir():
        raise RuntimeError(f"trace directory not found: {source}")
    graph_records = _read_json(Path(args.graph_records))
    graph_by_id = {row["instance_id"]: row for row in graph_records}
    pairs = _trace_pairs(source)
    if set(pairs) != set(graph_by_id):
        raise RuntimeError(
            "trace/ground-truth instance sets differ: "
            f"{len(pairs)} traces versus {len(graph_by_id)} records"
        )
    selected_ids = sorted(pairs)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        selected_ids = selected_ids[: args.limit]

    execution_config_id = _execution_config_id(
        quantization=args.quantization,
        attention_backend=args.attention_backend,
        cache_implementation=args.cache_implementation,
        max_new_tokens=args.max_new_tokens,
    )
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "raw_results.jsonl"
    existing = _read_jsonl(checkpoint)
    completed = {
        (
            row["model_id"],
            row["resolved_revision"],
            row.get("execution_config_id"),
            row["instance_id"],
        )
        for row in existing
        if row.get("status") == "completed"
    }
    records = list(existing)
    token = os.getenv("HF_TOKEN")
    resolved_models = []
    manifest = {
        "study": "real-agent-v2-secondary-huggingface-baseline",
        "started_utc": datetime.now(tz=UTC).isoformat(),
        "completed_utc": None,
        "status": "running",
        "selection": (
            "all frozen controlled instances in lexicographic instance_id order"
            if args.limit is None
            else f"smoke-only first {args.limit} lexicographic instance IDs"
        ),
        "expected_instance_count": len(selected_ids),
        "execution_config_id": execution_config_id,
        "models": resolved_models,
        "decoding": {
            "temperature": 0.0,
            "do_sample": False,
            "max_new_tokens": args.max_new_tokens,
            "truncation": "forbidden",
            "attention_backend": args.attention_backend,
            "cache_implementation": args.cache_implementation,
        },
        "prompt_contract": "same paired observable prompt and strict parser",
        "checkpoint": "append-only raw_results.jsonl",
    }
    _write_json(out / "run_manifest.json", manifest)

    for alias in args.models:
        spec = model_spec(alias)
        resolved = resolve_model_revision(spec, token=token)
        resolved_models.append(
            {
                "alias": alias,
                "model_id": spec.model_id,
                "requested_revision": spec.requested_revision,
                "resolved_revision": resolved,
                "context_tokens": spec.context_tokens,
                "quantization": args.quantization,
                "attention_backend": args.attention_backend,
                "cache_implementation": args.cache_implementation,
                "execution_config_id": execution_config_id,
            }
        )
        _write_json(out / "run_manifest.json", manifest)
        client = HuggingFaceAttributor(
            spec,
            resolved_revision=resolved,
            quantization=args.quantization,
            attention_backend=args.attention_backend,
            cache_implementation=args.cache_implementation,
            max_new_tokens=args.max_new_tokens,
            token=token,
        )
        pending = [
            instance_id
            for instance_id in selected_ids
            if (
                spec.model_id,
                resolved,
                execution_config_id,
                instance_id,
            )
            not in completed
        ]
        if pending:
            client.load()
        for index, instance_id in enumerate(pending, start=1):
            clean_path, perturbed_path = pairs[instance_id]
            graph = graph_by_id[instance_id]
            try:
                clean = load_react_agent_trace(clean_path)
                perturbed = load_react_agent_trace(perturbed_path)
                result = attribute_failure_detailed(
                    perturbed,
                    client,
                    clean_trace=clean,
                    mode="paired",
                )
                row = {
                    "status": "completed",
                    "provider": client.provider,
                    "model_alias": alias,
                    "model_id": spec.model_id,
                    "requested_revision": spec.requested_revision,
                    "resolved_revision": resolved,
                    "instance_id": instance_id,
                    "ground_truth_root": graph["injection_node"],
                    "raw_response": result.raw_response,
                    "parsed_node": result.predicted_node,
                    "parse_valid": result.predicted_node is not None,
                    "correct": result.predicted_node == graph["injection_node"],
                    "graph_prediction": graph["graph_predicted_node"],
                    "graph_correct": graph["graph_correct"],
                    "candidate_nodes": list(result.candidates),
                    "prompt_sha256": result.prompt_bundle.prompt_sha256,
                    "clean_trace_sha256": (
                        result.prompt_bundle.clean_trace_sha256
                    ),
                    "perturbed_trace_sha256": (
                        result.prompt_bundle.corrupt_trace_sha256
                    ),
                    **(client.last_call_metadata or {}),
                }
                row["execution_config_id"] = execution_config_id
                completed.add(
                    (
                        spec.model_id,
                        resolved,
                        execution_config_id,
                        instance_id,
                    )
                )
            except Exception as exc:
                row = {
                    "status": "error",
                    "provider": client.provider,
                    "model_alias": alias,
                    "model_id": spec.model_id,
                    "requested_revision": spec.requested_revision,
                    "resolved_revision": resolved,
                    "instance_id": instance_id,
                    "execution_config_id": execution_config_id,
                    "error": {
                        "type": exc.__class__.__name__,
                        "message": str(exc),
                    },
                    "call_metadata": client.last_call_metadata,
                }
            records.append(row)
            _append_jsonl(checkpoint, row)
            print(
                f"[{alias}] {index}/{len(pending)} {instance_id} "
                f"{row['status']}"
                + (
                    ""
                    if row["status"] == "completed"
                    else (
                        f" {row['error']['type']}: "
                        f"{row['error']['message']}"
                    )
                ),
                flush=True,
            )

    _write_json(out / "records.json", records)
    _write_csv(
        out / "records.csv",
        [_flat_record(row) for row in records],
    )
    deepseek = _deepseek_by_id(Path(args.deepseek_records))
    summary = summarize(
        records,
        deepseek,
        execution_config_id=execution_config_id,
        expected_instance_ids=set(selected_ids),
    )
    _write_json(out / "summary.json", summary)
    manifest["status"] = (
        "completed" if summary["study_complete"] else "finished_with_errors"
    )
    manifest["completed_utc"] = datetime.now(tz=UTC).isoformat()
    _write_json(out / "run_manifest.json", manifest)
    _write_integrity(out)


def summarize(
    records: list[dict[str, Any]],
    deepseek_by_id: dict[str, dict[str, Any]] | None = None,
    *,
    execution_config_id: str | None = None,
    expected_instance_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Summarize completed calls by immutable model revision."""
    grouped: dict[
        tuple[str, str],
        dict[str, dict[str, Any]],
    ] = defaultdict(dict)
    errors = 0
    for row in records:
        if (
            execution_config_id is not None
            and row.get("execution_config_id") != execution_config_id
        ):
            continue
        if row["status"] == "completed":
            grouped[(row["model_id"], row["resolved_revision"])][
                row["instance_id"]
            ] = row
        else:
            errors += 1
    models = []
    for (model_id, revision), rows_by_id in sorted(grouped.items()):
        rows = list(rows_by_id.values())
        correct = sum(bool(row["correct"]) for row in rows)
        low, high = wilson_interval(correct, len(rows))
        entry = {
            "model_id": model_id,
            "resolved_revision": revision,
            "n": len(rows),
            "correct": correct,
            "root_accuracy": correct / len(rows),
            "wilson_95": [low, high],
            "invalid_parse_count": sum(not row["parse_valid"] for row in rows),
            "mean_latency_ms": mean(
                row["latency_ms"]
                for row in rows
                if row.get("latency_ms") is not None
            ),
            "total_tokens": sum(row.get("total_tokens") or 0 for row in rows),
            "graph_vs_model_paired": paired_correctness(
                [bool(row["graph_correct"]) for row in rows],
                [bool(row["correct"]) for row in rows],
            ),
        }
        if expected_instance_ids is not None:
            completed_ids = {row["instance_id"] for row in rows}
            missing = sorted(expected_instance_ids - completed_ids)
            entry.update(
                {
                    "expected_n": len(expected_instance_ids),
                    "unique_completed_n": len(completed_ids),
                    "missing_instance_ids": missing,
                    "complete": not missing,
                }
            )
        if deepseek_by_id:
            paired_rows = [
                row for row in rows if row["instance_id"] in deepseek_by_id
            ]
            if paired_rows:
                entry["model_vs_deepseek_paired"] = paired_correctness(
                    [bool(row["correct"]) for row in paired_rows],
                    [
                        bool(deepseek_by_id[row["instance_id"]]["correct"])
                        for row in paired_rows
                    ],
                )
        models.append(entry)
    return {
        "execution_config_id": execution_config_id,
        "models": models,
        "completed_calls": sum(len(rows) for rows in grouped.values()),
        "error_attempts": errors,
        "study_complete": bool(models)
        and all(model.get("complete", True) for model in models),
    }


def _execution_config_id(
    *,
    quantization: str,
    attention_backend: str,
    cache_implementation: str,
    max_new_tokens: int,
) -> str:
    payload = json.dumps(
        {
            "quantization": quantization,
            "attention_backend": attention_backend,
            "cache_implementation": cache_implementation,
            "max_new_tokens": max_new_tokens,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(payload.encode("utf-8")).hexdigest()[:16]


def _trace_pairs(source: Path) -> dict[str, tuple[Path, Path]]:
    pairs = {}
    for directory in sorted(path for path in source.iterdir() if path.is_dir()):
        clean = directory / "clean.events.jsonl"
        perturbed = directory / "perturbed.events.jsonl"
        if clean.is_file() and perturbed.is_file():
            pairs[directory.name] = (clean, perturbed)
    return pairs


def _deepseek_by_id(path: Path) -> dict[str, dict[str, Any]] | None:
    if not path.is_file():
        return None
    return {
        row["instance_id"]: row
        for row in _read_json(path)
        if row.get("provider") == "deepseek" and row.get("status") == "completed"
    }


def _flat_record(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "status",
        "model_alias",
        "model_id",
        "resolved_revision",
        "instance_id",
        "ground_truth_root",
        "parsed_node",
        "parse_valid",
        "correct",
        "graph_prediction",
        "graph_correct",
        "prompt_sha256",
        "latency_ms",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "hardware",
    )
    flat = {key: row.get(key) for key in keys}
    if row["status"] == "error":
        flat["error"] = json.dumps(row.get("error"), ensure_ascii=False)
    return flat


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = list(
        dict.fromkeys(
            key
            for row in rows
            for key in row
        )
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_integrity(out: Path) -> None:
    destination = out / "integrity_manifest.json"
    artifacts = {}
    for path in sorted(out.rglob("*")):
        if path.is_file() and path != destination:
            artifacts[str(path.relative_to(out))] = {
                "sha256": sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            }
    _write_json(
        destination,
        {"hash_algorithm": "SHA-256", "artifacts": artifacts},
    )


if __name__ == "__main__":
    main()
