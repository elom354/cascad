"""Matched DeepSeek/OpenAI attribution comparison on identical simulator traces."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from cascad.attribution_baseline import configured_attributors
from cascad.experiment import ExperimentCase, run_experiment
from cascad.export import write_csv, write_json
from cascad.scenarios import attribution_fault
from cascad.statistics import paired_correctness, wilson_interval

DEFAULT_COMPARISON_SCENARIOS = (
    "support_neutral",
    "document_neutral",
    "cloud_neutral",
    "cloud_distant_symptom",
)


def run_llm_attribution_comparison(
    out_dir: str | Path,
    *,
    env_file: str | None = ".env",
    scenarios: Iterable[str] = DEFAULT_COMPARISON_SCENARIOS,
    n_repeats: int = 20,
    seed_start: int = 0,
    attribution_mode: str = "paired",
) -> dict[str, Any]:
    """Compare configured LLM baselines with matched prompts and trace pairs."""
    if n_repeats < 1:
        raise ValueError("n_repeats must be positive")
    selected_scenarios = tuple(scenarios)
    unknown = set(selected_scenarios) - set(DEFAULT_COMPARISON_SCENARIOS)
    if unknown:
        raise ValueError(f"unsupported comparison scenarios: {sorted(unknown)}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw_path = out / "attribution_raw.jsonl"
    existing = _load_jsonl(raw_path)
    completed = {
        (row["provider"], row["scenario"], row["seed"])
        for row in existing
        if row.get("status") == "completed"
    }
    records = list(existing)
    clients = configured_attributors(env_file)

    for client in clients:
        for scenario in selected_scenarios:
            for seed in range(seed_start, seed_start + n_repeats):
                key = (client.provider, scenario, seed)
                if key in completed:
                    continue
                try:
                    summary = run_experiment(
                        [
                            ExperimentCase(
                                name=f"{client.provider}_{scenario}_{seed}",
                                scenario=scenario,
                                method="attribution",
                                faults=[attribution_fault(scenario, seed)],
                                fault_factory=lambda current, name=scenario: [
                                    attribution_fault(name, current)
                                ],
                                trials=1,
                                seed_start=seed,
                                attribution_llm=client,
                                attribution_mode=attribution_mode,
                                fixed_instance_seed=seed,
                            )
                        ]
                    )[0]
                    record = dict((summary.attribution_records or [])[0])
                    row = {
                        "status": "completed",
                        "provider": client.provider,
                        "scenario": scenario,
                        "seed": seed,
                        **record,
                    }
                    completed.add(key)
                except Exception as exc:
                    row = {
                        "status": "error",
                        "provider": client.provider,
                        "scenario": scenario,
                        "seed": seed,
                        "model": client.model,
                        "error": {
                            "type": exc.__class__.__name__,
                            "message": str(exc),
                        },
                        "call_metadata": client.last_call_metadata,
                    }
                records.append(row)
                with raw_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    completed_records = [
        row for row in records if row.get("status") == "completed"
    ]
    summary = _summary(
        completed_records,
        all_records=records,
        configured_providers=[client.provider for client in clients],
        scenarios=selected_scenarios,
        n_repeats=n_repeats,
    )
    pairing = _pairing_audit(completed_records)
    flat = [_flat_record(row) for row in completed_records]
    write_json(out / "comparison_records.json", completed_records)
    write_csv(out / "comparison_records.csv", flat)
    write_json(out / "summary.json", summary)
    write_csv(out / "summary.csv", summary["by_provider_scenario"])
    write_json(out / "pairing_audit.json", pairing)
    return {
        "out_dir": str(out),
        "configured_providers": summary["configured_providers"],
        "completed_calls": len(completed_records),
        "error_calls": summary["error_calls"],
        "pairing_verdict": pairing["verdict"],
    }


def _summary(
    records: list[dict[str, Any]],
    *,
    all_records: list[dict[str, Any]],
    configured_providers: list[str],
    scenarios: tuple[str, ...],
    n_repeats: int,
) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[(row["provider"], row["scenario"])].append(row)
    strata = []
    for (provider, scenario), rows in sorted(grouped.items()):
        correct = sum(row["prediction_type"] == "root" for row in rows)
        strata.append(
            {
                "provider": provider,
                "scenario": scenario,
                "model": rows[0]["model"],
                "n": len(rows),
                "root_correct": correct,
                "root_accuracy": correct / len(rows),
                "wilson_95_low": wilson_interval(correct, len(rows))[0],
                "wilson_95_high": wilson_interval(correct, len(rows))[1],
                "symptom_selection_rate": sum(
                    row["prediction_type"] == "symptom" for row in rows
                )
                / len(rows),
                "invalid_output_rate": sum(
                    row["prediction_type"] == "invalid" for row in rows
                )
                / len(rows),
                "cascad_root_accuracy": sum(
                    row["cascad_prediction_type"] == "root" for row in rows
                )
                / len(rows),
            }
        )
    paired = _paired_provider_statistics(records)
    errors = [row for row in all_records if row.get("status") == "error"]
    return {
        "study_role": "matched_llm_attribution_baseline_comparison",
        "agent_model_changed": False,
        "configured_providers": configured_providers,
        "openai_optional": True,
        "openai_executed": "openai" in configured_providers,
        "scenarios": list(scenarios),
        "planned_calls_per_provider": len(scenarios) * n_repeats,
        "completed_calls": len(records),
        "error_calls": len(errors),
        "by_provider_scenario": strata,
        "paired_provider_statistics": paired,
        "interpretation_boundary": (
            "This compares attribution baselines on matched observable traces; "
            "it does not constitute a second agent-model validation."
        ),
    }


def _paired_provider_statistics(
    records: list[dict[str, Any]],
) -> dict[str, Any] | None:
    indexed = {
        (row["provider"], row["scenario"], row["seed"]): row for row in records
    }
    keys = sorted(
        {
            (row["scenario"], row["seed"])
            for row in records
            if ("deepseek", row["scenario"], row["seed"]) in indexed
            and ("openai", row["scenario"], row["seed"]) in indexed
        }
    )
    if not keys:
        return None
    deepseek = [
        indexed[("deepseek", scenario, seed)]["prediction_type"] == "root"
        for scenario, seed in keys
    ]
    openai = [
        indexed[("openai", scenario, seed)]["prediction_type"] == "root"
        for scenario, seed in keys
    ]
    return {
        "pair_count": len(keys),
        "deepseek_vs_openai": paired_correctness(deepseek, openai),
    }


def _pairing_audit(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[(row["scenario"], row["seed"])].append(row)
    comparisons = []
    for (scenario, seed), rows in sorted(groups.items()):
        if {row["provider"] for row in rows} != {"deepseek", "openai"}:
            continue
        by_provider = {row["provider"]: row for row in rows}
        left, right = by_provider["deepseek"], by_provider["openai"]
        comparison = {
            "scenario": scenario,
            "seed": seed,
            "same_prompt_sha256": left["prompt_sha256"] == right["prompt_sha256"],
            "same_clean_trace_sha256": (
                left["clean_trace_sha256"] == right["clean_trace_sha256"]
            ),
            "same_corrupt_trace_sha256": (
                left["corrupt_trace_sha256"] == right["corrupt_trace_sha256"]
            ),
            "same_candidate_nodes": (
                left["candidate_nodes"] == right["candidate_nodes"]
            ),
        }
        comparison["pass"] = all(
            value
            for key, value in comparison.items()
            if key not in {"scenario", "seed"}
        )
        comparisons.append(comparison)
    return {
        "verdict": (
            "NOT_APPLICABLE_ONE_PROVIDER"
            if not comparisons
            else "PASS"
            if all(row["pass"] for row in comparisons)
            else "FAIL"
        ),
        "matched_provider_pairs": len(comparisons),
        "comparisons": comparisons,
    }


def _flat_record(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "provider": row["provider"],
        "model": row["model"],
        "scenario": row["scenario"],
        "seed": row["seed"],
        "ground_truth_root": row["ground_truth_root"],
        "visible_symptom": row["visible_symptom"],
        "prediction": row["attribution_parsed_node"],
        "prediction_type": row["prediction_type"],
        "cascad_prediction": row["cascad_prediction"],
        "cascad_prediction_type": row["cascad_prediction_type"],
        "prompt_sha256": row["prompt_sha256"],
        "latency_ms": row.get("latency_ms"),
        "input_tokens": row.get("input_tokens"),
        "output_tokens": row.get("output_tokens"),
        "total_tokens": row.get("total_tokens"),
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
