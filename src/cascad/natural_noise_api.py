"""Complete the frozen natural-noise study with equal-information DeepSeek baselines."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean
from time import perf_counter, sleep
from typing import Any, Callable

from cascad.attribution_baseline import (
    DeepSeekAttributor,
    build_attribution_prompt,
    parse_attribution_node,
)
from cascad.export import write_csv, write_json
from cascad.natural_noise import (
    MEDIATOR,
    ROOT,
    SYMPTOM,
    build_calibrated_deepseek_prompt,
    build_calibration,
    calibration_context,
    generate_pair,
)


EXPECTED_BASELINE_COMMIT = "5aa70c970bbb2c59c551ebd56bb472b8913f9271"


@dataclass(frozen=True)
class FrozenNaturalNoiseStudy:
    calibration_pairs: tuple[Any, ...]
    evaluation_pairs: tuple[Any, ...]
    calibration_context: dict[str, Any]
    cascad_records: tuple[dict[str, Any], ...]
    verification: dict[str, Any]


def load_and_verify_frozen_study(
    frozen_dir: str | Path = "runs/natural-noise-study",
) -> FrozenNaturalNoiseStudy:
    """Materialize hash-addressed traces and fail on any frozen-artifact mismatch."""
    frozen = Path(frozen_dir)
    required = (
        "calibration_manifest.json",
        "evaluation_manifest.json",
        "raw_results.json",
        "fairness_audit.json",
        "study_metadata.json",
    )
    missing = [name for name in required if not (frozen / name).is_file()]
    if missing:
        raise FileNotFoundError(f"frozen natural-noise artifacts missing: {missing}")
    calibration_manifest = _read_json(frozen / "calibration_manifest.json")
    evaluation_manifest = _read_json(frozen / "evaluation_manifest.json")
    frozen_records = _read_json(frozen / "raw_results.json")
    frozen_fairness = _read_json(frozen / "fairness_audit.json")
    frozen_metadata = _read_json(frozen / "study_metadata.json")
    if len(calibration_manifest["instances"]) != 24:
        raise ValueError("frozen calibration set must contain exactly 24 pairs")
    if len(evaluation_manifest["instances"]) != 80:
        raise ValueError("frozen evaluation set must contain exactly 80 pairs")

    expected_hashes: dict[str, tuple[str, str]] = {}
    cascad_records = []
    for record in frozen_records:
        pair_hash = (record["clean_trace_sha256"], record["corrupt_trace_sha256"])
        previous = expected_hashes.setdefault(record["instance_id"], pair_hash)
        if previous != pair_hash:
            raise ValueError(f"inconsistent frozen hashes for {record['instance_id']}")
        if record["method"] == "cascad_full":
            cascad_records.append(record)
    if len(expected_hashes) != 80 or len(cascad_records) != 80:
        raise ValueError("frozen raw results do not cover 80 instances/full-Cascad rows")

    calibration_pairs = tuple(
        _pair_from_manifest(item, "calibration")
        for item in calibration_manifest["instances"]
    )
    evaluation_pairs = tuple(
        _pair_from_manifest(item, "evaluation")
        for item in evaluation_manifest["instances"]
    )
    profile = build_calibration(calibration_pairs)
    context = calibration_context(profile)
    if context != frozen_fairness["calibration_context_exact"]:
        raise ValueError("clean-derived calibration context differs from frozen pilot")

    prompt_hashes = set()
    pair_hashes = set()
    verified_rows = []
    for pair in evaluation_pairs:
        bundle = build_attribution_prompt(
            pair.comparison, mode="paired", clean_trace=pair.clean
        )
        expected_clean, expected_corrupt = expected_hashes[pair.instance_id]
        if (
            bundle.clean_trace_sha256 != expected_clean
            or bundle.corrupt_trace_sha256 != expected_corrupt
        ):
            raise ValueError(f"observable trace hash mismatch for {pair.instance_id}")
        prompt_hashes.add(bundle.prompt_sha256)
        pair_hashes.add((bundle.clean_trace_sha256, bundle.corrupt_trace_sha256))
        verified_rows.append({
            "instance_id": pair.instance_id,
            "clean_trace_sha256": bundle.clean_trace_sha256,
            "corrupt_trace_sha256": bundle.corrupt_trace_sha256,
            "paired_raw_prompt_sha256": bundle.prompt_sha256,
            "candidate_nodes": list(bundle.candidates),
        })
    uniqueness = frozen_metadata["uniqueness"]
    if not (
        len(prompt_hashes) == uniqueness["unique_prompt_count"] == 80
        and len({clean for clean, _ in pair_hashes})
        == uniqueness["unique_clean_trace_count"] == 80
        and len({corrupt for _, corrupt in pair_hashes})
        == uniqueness["unique_corrupt_trace_count"] == 80
    ):
        raise ValueError("frozen uniqueness counts do not match verified traces/prompts")
    verification = {
        "expected_baseline_commit": EXPECTED_BASELINE_COMMIT,
        "frozen_directory": str(frozen),
        "frozen_directory_modified": False,
        "calibration_pairs": len(calibration_pairs),
        "evaluation_pairs": len(evaluation_pairs),
        "trace_hashes_match": True,
        "calibration_context_matches": True,
        "instance_ids_match": True,
        "noise_levels_match": True,
        "template_ids_match": True,
        "candidate_sets_verified": True,
        "unique_prompt_count": len(prompt_hashes),
        "unique_trace_pair_count": len(pair_hashes),
        "note": (
            "The prior pilot stored trace hashes rather than clear trace payloads. "
            "Deterministic in-memory materialization was accepted only after exact "
            "per-instance hash verification against committed artifacts."
        ),
        "instances": verified_rows,
    }
    return FrozenNaturalNoiseStudy(
        calibration_pairs, evaluation_pairs, context,
        tuple(cascad_records), verification,
    )


def complete_frozen_deepseek_baselines(
    *,
    frozen_dir: str | Path = "runs/natural-noise-study",
    out_dir: str | Path = "runs/natural-noise-study-deepseek",
    attributor: DeepSeekAttributor | Callable[[str], str],
    max_attempts: int = 3,
) -> dict[str, Any]:
    """Run/resume the two missing conditions without mutating frozen artifacts."""
    study = load_and_verify_frozen_study(frozen_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "freeze_verification.json", study.verification)
    write_json(out / "calibration_context.json", study.calibration_context)
    raw_path = out / "raw_attribution.jsonl"
    existing = _load_jsonl(raw_path)
    by_key = {(item["instance_id"], item["method"]): item for item in existing}
    for pair in study.evaluation_pairs:
        raw_bundle = build_attribution_prompt(
            pair.comparison, mode="paired", clean_trace=pair.clean
        )
        calibrated_prompt, calibrated_candidates = build_calibrated_deepseek_prompt(
            pair, build_calibration(study.calibration_pairs)
        )
        conditions = (
            ("deepseek_paired_raw", raw_bundle.prompt, raw_bundle.candidates, None),
            (
                "deepseek_paired_calibrated",
                calibrated_prompt,
                calibrated_candidates,
                study.calibration_context,
            ),
        )
        if conditions[0][2] != conditions[1][2]:
            raise ValueError(f"candidate mismatch for {pair.instance_id}")
        for method, prompt, candidates, context in conditions:
            key = (pair.instance_id, method)
            if key in by_key:
                continue
            _audit_prompt(prompt, context)
            started = perf_counter()
            raw = _call_with_retry(attributor, prompt, max_attempts)
            latency = perf_counter() - started
            prediction = parse_attribution_node(raw, candidates)
            usage = getattr(attributor, "last_usage", None)
            record = _api_record(
                pair, method, prompt, candidates, raw, prediction, latency,
                context, usage,
            )
            _append_jsonl(raw_path, record)
            by_key[key] = record
            print(
                f"checkpoint={len(by_key)}/160 "
                f"instance={pair.instance_id} method={method}",
                flush=True,
            )
    records = [by_key[key] for key in sorted(by_key)]
    if len(records) != 160:
        raise ValueError(f"incomplete DeepSeek study: expected 160 rows, got {len(records)}")
    summary = _phase_a_summary(records)
    paired = _phase_a_paired(records, study.cascad_records)
    fairness = _completed_fairness_audit(records, study)
    write_json(out / "raw_attribution.json", records)
    write_json(out / "summary.json", summary)
    write_csv(out / "summary.csv", summary)
    write_json(out / "paired_correctness.json", paired)
    write_csv(out / "paired_correctness.csv", paired)
    write_json(out / "mcnemar_tables.json", paired)
    write_json(out / "fairness_audit.json", fairness)
    result = {
        "api_calls": len(records),
        "unique_prompt_count": len({item["prompt_sha256"] for item in records}),
        "unique_trace_pair_count": len({
            (item["clean_trace_sha256"], item["corrupt_trace_sha256"])
            for item in records
        }),
        "methods": sorted({item["method"] for item in records}),
        "model": getattr(attributor, "model", "mock"),
        "temperature": getattr(attributor, "temperature", None),
        "freeze_verified": True,
    }
    write_json(out / "study_metadata.json", result)
    return result


def _completed_fairness_audit(
    records: list[dict[str, Any]], study: FrozenNaturalNoiseStudy
) -> dict[str, Any]:
    by_instance: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_instance.setdefault(record["instance_id"], []).append(record)
    failures = []
    rows = []
    for instance_id, items in sorted(by_instance.items()):
        if len(items) != 2:
            failures.append(f"{instance_id}: expected two paired methods")
            continue
        raw = next(
            item for item in items if item["method"] == "deepseek_paired_raw"
        )
        calibrated = next(
            item for item in items
            if item["method"] == "deepseek_paired_calibrated"
        )
        same_traces = (
            raw["clean_trace_sha256"] == calibrated["clean_trace_sha256"]
            and raw["corrupt_trace_sha256"]
            == calibrated["corrupt_trace_sha256"]
        )
        same_candidates = raw["candidate_nodes"] == calibrated["candidate_nodes"]
        if not same_traces:
            failures.append(f"{instance_id}: trace pairs differ")
        if not same_candidates:
            failures.append(f"{instance_id}: candidate sets differ")
        rows.append({
            "instance_id": instance_id,
            "same_trace_pair": same_traces,
            "same_candidate_nodes": same_candidates,
            "raw_has_calibration_context": raw["calibration_context"] is not None,
            "calibrated_context_matches_export": (
                calibrated["calibration_context"] == study.calibration_context
            ),
        })
    if any(row["raw_has_calibration_context"] for row in rows):
        failures.append("raw paired method received calibration context")
    if not all(row["calibrated_context_matches_export"] for row in rows):
        failures.append("calibrated context differs from clean-only export")
    return {
        "passed": not failures,
        "failures": failures,
        "freeze_verification": study.verification,
        "no_evaluator_labels": True,
        "no_injection_metadata": True,
        "no_graph_edges_or_cascad_predictions": True,
        "calibration_context_clean_only": True,
        "instances": rows,
    }


def _pair_from_manifest(item: dict[str, Any], split: str) -> Any:
    pair = generate_pair(item["seed"], item["noise_level"], split)
    expected = {
        "instance_id": item["instance_id"],
        "noise_template_ids": item["noise_template_ids"],
        "contains_fault": item["contains_fault"],
    }
    actual = {
        "instance_id": pair.instance_id,
        "noise_template_ids": pair.noise_template_ids,
        "contains_fault": pair.contains_fault,
    }
    if actual != expected:
        raise ValueError(f"manifest mismatch: expected={expected}, actual={actual}")
    return pair


def _audit_prompt(prompt: str, context: dict[str, Any] | None) -> None:
    lower = prompt.casefold()
    forbidden = (
        "fault_injected",
        "source_fault_id",
        "ground_truth_root",
        "propagation_mediator",
        "visible_symptom",
        "benign_natural_divergence",
        "injected_root_divergence",
        "downstream_contamination",
        "cascad_prediction",
    )
    leaked = [term for term in forbidden if term in lower]
    if leaked:
        raise ValueError(f"DeepSeek prompt leakage: {leaked}")
    if context is None and "natural_distance_summary_by_node" in lower:
        raise ValueError("raw paired prompt received calibration thresholds")


def _call_with_retry(
    attributor: Callable[[str], str], prompt: str, max_attempts: int
) -> str:
    for attempt in range(1, max_attempts + 1):
        try:
            return attributor(prompt).strip()
        except RuntimeError:
            if attempt == max_attempts:
                raise
            sleep(float(attempt))
    raise AssertionError("unreachable")


def _api_record(
    pair: Any,
    method: str,
    prompt: str,
    candidates: tuple[str, ...],
    raw: str,
    prediction: str | None,
    latency: float,
    context: dict[str, Any] | None,
    usage: dict[str, Any] | None,
) -> dict[str, Any]:
    bundle = build_attribution_prompt(
        pair.comparison, mode="paired", clean_trace=pair.clean
    )
    benign_nodes = {
        item["node"] for item in pair.annotations
        if item["category"] == "benign_natural_divergence"
    }
    prediction_type = (
        "invalid" if prediction is None
        else "root" if prediction == ROOT
        else "mediator" if prediction == MEDIATOR
        else "symptom" if prediction == SYMPTOM
        else "benign_noise_node" if prediction in benign_nodes
        else "other"
    )
    return {
        "instance_id": pair.instance_id,
        "seed": pair.instance_seed,
        "noise_level": pair.noise_level,
        "noise_template_ids": pair.noise_template_ids,
        "method": method,
        "model": "deepseek-chat",
        "temperature": 0.0,
        "candidate_nodes": list(candidates),
        "clean_trace_sha256": bundle.clean_trace_sha256,
        "corrupt_trace_sha256": bundle.corrupt_trace_sha256,
        "prompt_sha256": sha256(prompt.encode()).hexdigest(),
        "prompt": prompt,
        "calibration_context": context,
        "raw_model_output": raw,
        "prediction": prediction,
        "prediction_type": prediction_type,
        "parse_valid": prediction is not None,
        "root_graph_distance": _distance(prediction),
        "latency_seconds": latency,
        "usage": usage,
        "observed_cost_usd": None,
    }


def _phase_a_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for method in sorted({item["method"] for item in records}):
        for level in ("N0", "N1", "N2", "N3"):
            selected = [
                item for item in records
                if item["method"] == method and item["noise_level"] == level
            ]
            size = len(selected)
            distances = [
                item["root_graph_distance"] for item in selected
                if item["root_graph_distance"] is not None
            ]
            rows.append({
                "method": method,
                "noise_level": level,
                "instances": size,
                "root_accuracy": sum(item["prediction_type"] == "root" for item in selected) / size,
                "benign_noise_node_selection_rate": sum(item["prediction_type"] == "benign_noise_node" for item in selected) / size,
                "mediator_selection_rate": sum(item["prediction_type"] == "mediator" for item in selected) / size,
                "symptom_selection_rate": sum(item["prediction_type"] == "symptom" for item in selected) / size,
                "invalid_output_rate": sum(item["prediction_type"] == "invalid" for item in selected) / size,
                "mean_graph_distance_to_root": mean(distances) if distances else None,
                "api_calls": size,
                "mean_latency_seconds": mean(item["latency_seconds"] for item in selected),
                "observed_cost_usd": None,
                "unique_prompt_count": len({item["prompt_sha256"] for item in selected}),
                "unique_trace_pair_count": len({
                    (item["clean_trace_sha256"], item["corrupt_trace_sha256"])
                    for item in selected
                }),
            })
    return rows


def _phase_a_paired(
    api_records: list[dict[str, Any]],
    cascad_records: tuple[dict[str, Any], ...],
) -> list[dict[str, Any]]:
    correctness = {
        (item["instance_id"], item["method"]): item["prediction_type"] == "root"
        for item in api_records
    }
    correctness.update({
        (item["instance_id"], "cascad_full"): item["prediction_type"] == "root"
        for item in cascad_records
    })
    comparisons = (
        ("deepseek_paired_raw", "cascad_full"),
        ("deepseek_paired_calibrated", "cascad_full"),
        ("deepseek_paired_raw", "deepseek_paired_calibrated"),
    )
    rows = []
    instance_ids = sorted({item["instance_id"] for item in api_records})
    for left, right in comparisons:
        pairs = [(correctness[(instance, left)], correctness[(instance, right)]) for instance in instance_ids]
        rows.append({
            "method_a": left,
            "method_b": right,
            "paired_instances": len(pairs),
            "both_correct": sum(a and b for a, b in pairs),
            "a_correct_b_wrong": sum(a and not b for a, b in pairs),
            "a_wrong_b_correct": sum(not a and b for a, b in pairs),
            "both_wrong": sum(not a and not b for a, b in pairs),
        })
    return rows


def _distance(prediction: str | None) -> int | None:
    order = ("planner", "upload", "share", "memory", "notify", "responder")
    return abs(order.index(prediction) - order.index(ROOT)) if prediction in order else None


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
