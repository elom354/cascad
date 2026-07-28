"""Controlled natural-divergence robustness study.

This module is intentionally separate from the frozen paired-attribution study.
Evaluator annotations live in manifests/records, never in execution payloads.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from random import Random
from statistics import mean, stdev
from time import perf_counter
from typing import Any, Callable, Iterable

from cascad.attribution_baseline import (
    DeepSeekAttributor,
    build_attribution_prompt,
    parse_attribution_node,
    serialize_trace_for_attribution,
)
from cascad.divergence import DivergenceDistribution, corresponding_events, encoder_status, event_distance
from cascad.export import write_csv, write_json
from cascad.injection import FaultInjector
from cascad.models import NodeEvent, RunTrace
from cascad.scenarios import attribution_fault
from cascad.simulator import ReActPropagationSimulator


SCENARIO = "cloud_distant_symptom_natural_noise"
SKELETON = ("planner", "upload", "share", "memory", "notify", "responder")
ROOT = "share"
MEDIATOR = "memory"
SYMPTOM = "notify"
TRUE_CONTAMINATED = {ROOT, MEDIATOR, SYMPTOM, "responder"}
LEVEL_COUNTS = {"N0": 0, "N1": 1, "N2": 3, "N3": 5}
NON_CAUSAL_FIELDS = {
    "correlation",
    "correlation_id",
    "receipt",
    "receipt_id",
    "timestamp",
    "resource_id",
    "file_id",
    "harmless_metadata",
}
EVALUATOR_TERMS = {
    "benign_natural_divergence",
    "injected_root_divergence",
    "downstream_contamination",
    "visible_failure",
    "ground_truth_root",
    "propagation_mediator",
    "visible_symptom",
    "source_fault_id",
    "fault_injected",
}


@dataclass(frozen=True)
class NoiseTemplate:
    template_id: str
    node: str
    category: str
    field: str
    value: Any
    rationale: str


@dataclass
class NaturalNoisePair:
    instance_id: str
    instance_seed: int
    split: str
    noise_level: str
    noise_template_ids: list[str]
    annotations: list[dict[str, str]]
    clean: RunTrace
    comparison: RunTrace
    contains_fault: bool


@dataclass(frozen=True)
class CalibrationProfile:
    split_id: str
    by_node: dict[str, DivergenceDistribution]
    variable_fields_by_node: dict[str, tuple[str, ...]]
    event_kinds_by_node: dict[str, tuple[str, ...]]
    instance_ids: tuple[str, ...]
    template_ids: tuple[str, ...]


CALIBRATION_TEMPLATES: dict[str, tuple[NoiseTemplate, ...]] = {
    "planner": (
        NoiseTemplate("cal-sem-plan-01", "planner", "semantic_paraphrase", "plan_summary",
                      "Carry out each workflow stage and report completion.",
                      "This restates the fixed plan without changing its nodes or order."),
        NoiseTemplate("cal-sem-plan-02", "planner", "semantic_paraphrase", "plan_summary",
                      "Execute the prescribed stages, then communicate the outcome.",
                      "The wording is descriptive only and is not consumed by execution."),
    ),
    "upload": (
        NoiseTemplate("cal-struct-upload-01", "upload", "structured_variation", "resource_id",
                      "cal-resource-A", "The simulator never reads this post-execution display identifier."),
        NoiseTemplate("cal-struct-upload-02", "upload", "structured_variation", "correlation_id",
                      "CAL-0002", "The correlation value is observability metadata only."),
    ),
    "memory": (
        NoiseTemplate("cal-sem-memory-01", "memory", "semantic_paraphrase", "memory_summary",
                      "The retained authorization state is available for the next step.",
                      "The summary is not read by workflow logic; the original state remains unchanged."),
        NoiseTemplate("cal-sem-memory-02", "memory", "semantic_paraphrase", "memory_summary",
                      "The access state was retained for subsequent processing.",
                      "This is a semantically equivalent display summary."),
    ),
    "share": (
        NoiseTemplate("cal-struct-share-01", "share", "structured_variation", "harmless_metadata",
                      {"format": "compact"}, "Formatting metadata is not used to compute authorization."),
        NoiseTemplate("cal-struct-share-02", "share", "structured_variation", "correlation_id",
                      "CAL-S-02", "The correlation value has no control or data-flow effect."),
    ),
    "notify": (
        NoiseTemplate("cal-struct-notify-01", "notify", "structured_variation", "timestamp",
                      "2030-01-01T08:00:00Z", "Equivalent timestamp formatting is not read by the workflow."),
        NoiseTemplate("cal-struct-notify-02", "notify", "structured_variation", "receipt_id",
                      "CAL-R-02", "Receipt formatting has no effect on notification success."),
    ),
}

EVALUATION_TEMPLATES: dict[str, tuple[NoiseTemplate, ...]] = {
    "planner": (
        NoiseTemplate("eval-sem-plan-01", "planner", "semantic_paraphrase", "plan_summary",
                      "Complete the established sequence and provide the result.",
                      "The phrase describes but does not alter the executable plan."),
        NoiseTemplate("eval-sem-plan-02", "planner", "semantic_paraphrase", "plan_summary",
                      "Follow the configured processing path before reporting its outcome.",
                      "No node, argument, or control decision reads this prose."),
        NoiseTemplate("eval-sem-plan-03", "planner", "semantic_paraphrase", "plan_summary",
                      "Proceed through the defined operations and communicate completion.",
                      "Only the observable surface changes."),
    ),
    "upload": (
        NoiseTemplate("eval-struct-upload-01", "upload", "structured_variation", "resource_id",
                      "eval-resource-X", "The identifier is declared non-causal and not read downstream."),
        NoiseTemplate("eval-struct-upload-02", "upload", "structured_variation", "correlation_id",
                      "EVAL-U-02", "This tracing token is ignored by execution logic."),
        NoiseTemplate("eval-struct-upload-03", "upload", "structured_variation", "harmless_metadata",
                      {"transport": "standard"}, "Optional transport metadata is never consumed."),
    ),
    "memory": (
        NoiseTemplate("eval-sem-memory-01", "memory", "semantic_paraphrase", "memory_summary",
                      "The recorded access setting can be used by the following operation.",
                      "The executable authorization field is left untouched."),
        NoiseTemplate("eval-sem-memory-02", "memory", "semantic_paraphrase", "memory_summary",
                      "A retained entitlement setting is available to the subsequent stage.",
                      "The field is an unconsumed summary."),
        NoiseTemplate("eval-sem-memory-03", "memory", "semantic_paraphrase", "memory_summary",
                      "The workflow retained its access setting for later use.",
                      "This display-only sentence cannot affect the outcome."),
    ),
    "share": (
        NoiseTemplate("eval-struct-share-01", "share", "structured_variation", "harmless_metadata",
                      {"encoding": "verbose"}, "Encoding metadata is independent of permission logic."),
        NoiseTemplate("eval-struct-share-02", "share", "structured_variation", "correlation_id",
                      "EVAL-S-02", "This value is observability-only."),
        NoiseTemplate("eval-struct-share-03", "share", "structured_variation", "timestamp",
                      "2030-01-01 08:00 UTC", "The timestamp representation is not consumed."),
    ),
    "notify": (
        NoiseTemplate("eval-struct-notify-01", "notify", "structured_variation", "receipt_id",
                      "EVAL-R-X", "Receipt text cannot change success or failure."),
        NoiseTemplate("eval-struct-notify-02", "notify", "structured_variation", "timestamp",
                      "2030-01-01T08:00:00+00:00", "Only an equivalent display format changes."),
        NoiseTemplate("eval-struct-notify-03", "notify", "structured_variation", "harmless_metadata",
                      {"locale": "en"}, "The locale metadata is never read by the simulator."),
    ),
}


def generate_pair(seed: int, noise_level: str, split: str) -> NaturalNoisePair:
    """Generate a deterministic clean/clean or held-out clean/corrupt pair."""
    if noise_level not in LEVEL_COUNTS:
        raise ValueError(f"unknown noise level: {noise_level}")
    if split not in {"calibration", "evaluation"}:
        raise ValueError("split must be calibration or evaluation")
    clean = ReActPropagationSimulator(scenario=SCENARIO).run(seed=seed).trace
    if split == "evaluation":
        comparison = ReActPropagationSimulator(
            FaultInjector([attribution_fault(SCENARIO, seed)]), scenario=SCENARIO
        ).run(seed=seed).trace
    else:
        comparison = ReActPropagationSimulator(scenario=SCENARIO).run(seed=seed).trace
    templates = _select_templates(seed, noise_level, split)
    annotations = []
    for template in templates:
        _apply_template(comparison, template, seed)
        annotations.append({
            "node": template.node,
            "field": template.field,
            "category": "benign_natural_divergence",
            "template_id": template.template_id,
            "rationale": template.rationale,
        })
    if split == "evaluation":
        annotations.extend([
            {"node": ROOT, "field": "permission", "category": "injected_root_divergence",
             "template_id": "causal-root-v1", "rationale": "First changed executable authorization state."},
            {"node": MEDIATOR, "field": "memories", "category": "downstream_contamination",
             "template_id": "causal-memory-v1", "rationale": "Persists the changed authorization state."},
            {"node": SYMPTOM, "field": "ok", "category": "visible_failure",
             "template_id": "causal-symptom-v1", "rationale": "First explicit externally observable failure."},
            {"node": "responder", "field": "answer", "category": "downstream_contamination",
             "template_id": "causal-response-v1", "rationale": "Reports the failed downstream outcome."},
        ])
    instance_id = f"{split}-{noise_level.lower()}-{seed:05d}"
    clean.metadata.update({"study_instance_id": instance_id, "study_split": split})
    comparison.metadata.update({"study_instance_id": instance_id, "study_split": split})
    return NaturalNoisePair(
        instance_id=instance_id,
        instance_seed=seed,
        split=split,
        noise_level=noise_level,
        noise_template_ids=[item.template_id for item in templates],
        annotations=annotations,
        clean=clean,
        comparison=comparison,
        contains_fault=split == "evaluation",
    )


def build_calibration(pairs: Iterable[NaturalNoisePair], epsilon: float = 0.05) -> CalibrationProfile:
    """Estimate node distributions and naturally variable fields from clean pairs."""
    del epsilon  # retained in the stable public signature; thresholds apply at diagnosis.
    pairs = list(pairs)
    if not pairs or any(pair.contains_fault or pair.split != "calibration" for pair in pairs):
        raise ValueError("calibration requires fault-free clean/clean pairs")
    samples: dict[str, list[float]] = {}
    variable_fields: dict[str, set[str]] = {}
    event_kinds: dict[str, set[str]] = {}
    for pair in pairs:
        for annotation in pair.annotations:
            variable_fields.setdefault(annotation["node"], set()).add(annotation["field"])
        for event in pair.comparison.events:
            event_kinds.setdefault(event.node_id, set()).add(str(getattr(event.kind, "value", event.kind)))
    # Calibration and diagnosis must use the same representation. Otherwise a
    # harmless field can inflate the threshold and hide a causal field after
    # that harmless field has been canonicalized away at evaluation time.
    for pair in pairs:
        for node, (left, right) in corresponding_events(pair.comparison, pair.clean).items():
            fields = variable_fields.get(node, set()) | NON_CAUSAL_FIELDS
            samples.setdefault(node, []).append(event_distance(
                _event_without_fields(left, fields),
                _event_without_fields(right, fields),
            ))
    distributions = {
        node: DivergenceDistribution(
            mean=mean(values),
            stddev=stdev(values) if len(values) > 1 else 0.0,
            samples=tuple(values),
        )
        for node, values in samples.items()
    }
    return CalibrationProfile(
        split_id="natural-noise-calibration-v1",
        by_node=distributions,
        variable_fields_by_node={node: tuple(sorted(fields)) for node, fields in variable_fields.items()},
        event_kinds_by_node={node: tuple(sorted(kinds)) for node, kinds in event_kinds.items()},
        instance_ids=tuple(pair.instance_id for pair in pairs),
        template_ids=tuple(sorted({item for pair in pairs for item in pair.noise_template_ids})),
    )


def diagnose_pair(
    pair: NaturalNoisePair,
    calibration: CalibrationProfile,
    *,
    use_calibration: bool,
    use_dependencies: bool,
    epsilon: float = 0.05,
) -> tuple[str | None, set[str], dict[str, float]]:
    """Run one controlled Cascad ablation with orthogonal feature switches."""
    pairs = corresponding_events(pair.comparison, pair.clean)
    distances: dict[str, float] = {}
    divergent: set[str] = set()
    for node, (observed, reference) in pairs.items():
        left, right = observed, reference
        if use_calibration:
            variable = set(calibration.variable_fields_by_node.get(node, ())) | NON_CAUSAL_FIELDS
            left = _event_without_fields(observed, variable)
            right = _event_without_fields(reference, variable)
        distance = event_distance(left, right)
        distances[node] = distance
        baseline = calibration.by_node.get(node, DivergenceDistribution(0.0, 0.0)) if use_calibration else DivergenceDistribution(0.0, 0.0)
        if distance > baseline.mean + baseline.stddev + epsilon:
            divergent.add(node)
    ordered = [node for node in SKELETON if node in divergent]
    if not ordered:
        return None, set(), distances
    prediction = ordered[0]
    if not use_dependencies:
        return prediction, divergent, distances
    reachable = _reachable(pair.comparison, prediction) | {prediction}
    constrained = divergent & reachable
    return prediction, constrained, distances


def naive_first_raw_difference(pair: NaturalNoisePair) -> str | None:
    """Return the first node with any exact serialized-event difference."""
    left = _events_by_node(serialize_trace_for_attribution(pair.clean))
    right = _events_by_node(serialize_trace_for_attribution(pair.comparison))
    return next((node for node in SKELETON if left.get(node) != right.get(node)), None)


def canonicalized_first_difference(pair: NaturalNoisePair) -> str | None:
    """Ignore only declared non-causal fields, then return the first difference."""
    left = _canonicalize(_events_by_node(serialize_trace_for_attribution(pair.clean)))
    right = _canonicalize(_events_by_node(serialize_trace_for_attribution(pair.comparison)))
    return next((node for node in SKELETON if left.get(node) != right.get(node)), None)


def maximum_raw_divergence(pair: NaturalNoisePair) -> str | None:
    """Return the greatest uncalibrated aligned terminal-event divergence."""
    values = {
        node: event_distance(observed, reference)
        for node, (observed, reference) in corresponding_events(pair.comparison, pair.clean).items()
        if node in SKELETON
    }
    return max(values, key=lambda node: (values[node], -SKELETON.index(node))) if values else None


def calibration_context(profile: CalibrationProfile) -> dict[str, Any]:
    """Transparent context available equally from the clean/clean split."""
    return {
        "calibration_split_id": profile.split_id,
        "naturally_variable_fields_by_node": {
            node: list(fields) for node, fields in sorted(profile.variable_fields_by_node.items())
        },
        "event_kinds_observed_by_node": {
            node: list(kinds) for node, kinds in sorted(profile.event_kinds_by_node.items())
        },
        "natural_distance_summary_by_node": {
            node: {"mean": value.mean, "stddev": value.stddev, "sample_count": len(value.samples)}
            for node, value in sorted(profile.by_node.items())
        },
    }


def build_calibrated_deepseek_prompt(pair: NaturalNoisePair, profile: CalibrationProfile) -> tuple[str, tuple[str, ...]]:
    """Build paired prompt plus clean-only calibration information."""
    base = build_attribution_prompt(pair.comparison, mode="paired", clean_trace=pair.clean)
    context = calibration_context(profile)
    prompt = (
        base.prompt
        + "\n\nClean/clean calibration context (derived only from separate reference executions):\n"
        + json.dumps(context, sort_keys=True)
        + "\nTreat listed natural variations as potentially benign. Return only one exact node_id."
    )
    return prompt, base.candidates


def run_natural_noise_study(
    out_dir: str | Path,
    *,
    instances_per_level: int = 20,
    calibration_pairs: int = 24,
    epsilon: float = 0.05,
    attributor: DeepSeekAttributor | Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """Execute the fixed pilot, checkpointing every optional API response."""
    if instances_per_level < 1 or calibration_pairs < 4:
        raise ValueError("instances_per_level >= 1 and calibration_pairs >= 4 are required")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    calibration_set = [
        generate_pair(100_000 + index, tuple(LEVEL_COUNTS)[index % 4], "calibration")
        for index in range(calibration_pairs)
    ]
    evaluation_set = [
        generate_pair(level_index * 10_000 + index, level, "evaluation")
        for level_index, level in enumerate(LEVEL_COUNTS)
        for index in range(instances_per_level)
    ]
    profile = build_calibration(calibration_set, epsilon)
    split_audit = _split_audit(calibration_set, evaluation_set)
    if not split_audit["passed"]:
        raise ValueError(f"calibration/evaluation split audit failed: {split_audit}")
    manifest = _manifest(calibration_set, evaluation_set, split_audit)
    write_json(out / "calibration_manifest.json", manifest["calibration"])
    write_json(out / "evaluation_manifest.json", manifest["evaluation"])

    fairness = _fairness_audit(evaluation_set, profile)
    write_json(out / "fairness_audit.json", fairness)
    if not fairness["passed"]:
        raise ValueError(f"fairness/leakage audit failed: {fairness['failures']}")

    records: list[dict[str, Any]] = []
    raw_path = out / "raw_results.jsonl"
    raw_path.write_text("", encoding="utf-8")
    for pair in evaluation_set:
        local_methods: list[tuple[str, str | None, set[str], float]] = []
        started = perf_counter()
        prediction = naive_first_raw_difference(pair)
        local_methods.append(("naive_first_raw_difference", prediction, _singleton(prediction), perf_counter() - started))
        started = perf_counter()
        prediction = canonicalized_first_difference(pair)
        local_methods.append(("canonicalized_first_difference", prediction, _singleton(prediction), perf_counter() - started))
        started = perf_counter()
        prediction = maximum_raw_divergence(pair)
        local_methods.append(("maximum_raw_divergence", prediction, _singleton(prediction), perf_counter() - started))
        for method, use_calibration, use_dependencies in (
            ("cascad_no_calibration", False, True),
            ("cascad_no_dependencies", True, False),
            ("cascad_full", True, True),
        ):
            started = perf_counter()
            prediction, contaminated, _ = diagnose_pair(
                pair, profile, use_calibration=use_calibration,
                use_dependencies=use_dependencies, epsilon=epsilon,
            )
            local_methods.append((method, prediction, contaminated, perf_counter() - started))
        for method, prediction, contaminated, latency in local_methods:
            record = _record(pair, method, prediction, contaminated, latency_seconds=latency)
            records.append(record)
            _append_jsonl(raw_path, record)

        if attributor is not None:
            base = build_attribution_prompt(pair.comparison, mode="paired", clean_trace=pair.clean)
            for method, prompt, candidates, context in (
                ("deepseek_paired_raw", base.prompt, base.candidates, None),
                ("deepseek_paired_calibrated", *build_calibrated_deepseek_prompt(pair, profile), calibration_context(profile)),
            ):
                _audit_external_prompt(prompt, pair, candidates)
                started = perf_counter()
                raw = attributor(prompt).strip()
                latency = perf_counter() - started
                prediction = parse_attribution_node(raw, candidates)
                record = _record(
                    pair, method, prediction, _singleton(prediction),
                    latency_seconds=latency,
                    prompt=prompt, raw_model_output=raw,
                    calibration_context_value=context,
                )
                records.append(record)
                _append_jsonl(raw_path, record)

    uniqueness = _uniqueness_audit(evaluation_set)
    if not uniqueness["passed"]:
        raise ValueError(f"uniqueness audit failed: {uniqueness}")
    methods = sorted({record["method"] for record in records})
    summaries = _summaries(records, methods)
    paired = _paired_correctness(records, methods)
    stats = _bootstrap_artifacts(records, methods)
    mcnemar = _mcnemar_tables(records, methods)
    write_json(out / "raw_results.json", records)
    write_json(out / "summary.json", summaries)
    write_csv(out / "summary.csv", summaries)
    ablation_rows = [
        row for row in summaries
        if row["method"] in {"cascad_no_calibration", "cascad_no_dependencies", "cascad_full"}
    ]
    write_json(out / "ablation_table.json", ablation_rows)
    write_csv(out / "ablation_table.csv", ablation_rows)
    latency_cost_rows = [{
        "method": row["method"],
        "noise_level": row["noise_level"],
        "mean_latency_seconds": row["mean_latency_seconds"],
        "api_call_count": row["api_call_count"],
        "api_cost_usd": row["api_cost_usd"],
    } for row in summaries]
    write_json(out / "latency_cost_comparison.json", latency_cost_rows)
    write_csv(out / "latency_cost_comparison.csv", latency_cost_rows)
    write_json(out / "paired_correctness.json", paired)
    write_csv(out / "paired_correctness.csv", paired)
    write_json(out / "bootstrap_confidence_intervals.json", stats)
    write_json(out / "mcnemar_tables.json", mcnemar)
    _write_plots(out / "plots", summaries)
    result = {
        "scenario": SCENARIO,
        "encoder": encoder_status(),
        "calibration_instance_ids": list(profile.instance_ids),
        "evaluation_instance_ids": [pair.instance_id for pair in evaluation_set],
        "noise_template_ids": {
            "calibration": list(profile.template_ids),
            "evaluation": sorted({item for pair in evaluation_set for item in pair.noise_template_ids}),
        },
        "split_overlap_check": split_audit,
        "uniqueness": uniqueness,
        "methods": methods,
        "api_call_count": sum(record["method"].startswith("deepseek") for record in records),
        "summary_rows": len(summaries),
    }
    write_json(out / "study_metadata.json", result)
    return result


def _select_templates(seed: int, level: str, split: str) -> list[NoiseTemplate]:
    count = LEVEL_COUNTS[level]
    if count == 0:
        return []
    templates = CALIBRATION_TEMPLATES if split == "calibration" else EVALUATION_TEMPLATES
    nodes = ("planner", "upload", "memory", "share", "notify")
    selected = []
    for offset, node in enumerate(nodes[:count]):
        choices = templates[node]
        selected.append(choices[(seed + offset) % len(choices)])
    return selected


def _apply_template(trace: RunTrace, template: NoiseTemplate, seed: int) -> None:
    events = [
        event for event in trace.events
        if event.node_id == template.node
        and str(getattr(event.kind, "value", event.kind)) not in {"node_start", "node_end", "fault_injected"}
    ]
    if not events:
        raise ValueError(f"no observable event for noise template {template.template_id}")
    value = deepcopy(template.value)
    if isinstance(value, str) and template.category == "structured_variation":
        value = f"{value}-{seed:05d}"
    for event in events:
        event.payload[template.field] = deepcopy(value)


def _event_without_fields(event: NodeEvent, fields: set[str]) -> NodeEvent:
    copy = deepcopy(event)
    copy.payload = _remove_fields(copy.payload, fields)
    return copy


def _remove_fields(value: Any, fields: set[str]) -> Any:
    if isinstance(value, dict):
        return {key: _remove_fields(item, fields) for key, item in value.items() if key not in fields}
    if isinstance(value, list):
        return [_remove_fields(item, fields) for item in value]
    return value


def _canonicalize(value: Any) -> Any:
    return _remove_fields(value, NON_CAUSAL_FIELDS)


def _events_by_node(events: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        result.setdefault(event["node_id"], []).append(event)
    return result


def _reachable(trace: RunTrace, origin: str) -> set[str]:
    adjacency: dict[str, set[str]] = {}
    for edge in trace.causal_edges:
        adjacency.setdefault(edge.source, set()).add(edge.target)
    seen: set[str] = set()
    pending = [origin]
    while pending:
        node = pending.pop()
        for target in adjacency.get(node, set()):
            if target not in seen:
                seen.add(target)
                pending.append(target)
    return seen


def _record(
    pair: NaturalNoisePair,
    method: str,
    prediction: str | None,
    predicted_contaminated: set[str],
    *,
    latency_seconds: float,
    prompt: str | None = None,
    raw_model_output: str | None = None,
    calibration_context_value: dict[str, Any] | None = None,
) -> dict[str, Any]:
    benign_nodes = {item["node"] for item in pair.annotations if item["category"] == "benign_natural_divergence"}
    false = predicted_contaminated - TRUE_CONTAMINATED
    missed = TRUE_CONTAMINATED - predicted_contaminated
    clean_observable = serialize_trace_for_attribution(pair.clean)
    corrupt_observable = serialize_trace_for_attribution(pair.comparison)
    return {
        "instance_id": pair.instance_id,
        "seed": pair.instance_seed,
        "scenario": SCENARIO,
        "noise_level": pair.noise_level,
        "noise_template_ids": pair.noise_template_ids,
        "calibration_split_id": "natural-noise-calibration-v1",
        "method": method,
        "ground_truth_root": ROOT,
        "injection_node": ROOT,
        "estimated_source": prediction,
        "propagation_mediator": MEDIATOR,
        "visible_symptom": SYMPTOM,
        "prediction": prediction,
        "prediction_type": _prediction_type(prediction, benign_nodes),
        "root_graph_distance": _graph_distance(ROOT, prediction),
        "predicted_contaminated_nodes": sorted(predicted_contaminated),
        "true_contaminated_nodes": sorted(TRUE_CONTAMINATED),
        "false_contaminated_nodes": sorted(false),
        "missed_contaminated_nodes": sorted(missed),
        "clean_trace_sha256": _json_hash(clean_observable),
        "corrupt_trace_sha256": _json_hash(corrupt_observable),
        "prompt_sha256": sha256(prompt.encode()).hexdigest() if prompt else None,
        "raw_model_output": raw_model_output,
        "parse_valid": prediction is not None,
        "latency_seconds": latency_seconds,
        "api_cost_usd": None,
        "calibration_context": calibration_context_value,
    }


def _prediction_type(prediction: str | None, benign_nodes: set[str]) -> str:
    if prediction is None:
        return "invalid"
    if prediction == ROOT:
        return "root"
    if prediction == MEDIATOR:
        return "mediator"
    if prediction == SYMPTOM:
        return "symptom"
    if prediction in benign_nodes:
        return "benign_noise_node"
    return "other"


def _singleton(prediction: str | None) -> set[str]:
    return {prediction} if prediction else set()


def _graph_distance(root: str, prediction: str | None) -> int | None:
    if prediction is None:
        return None
    adjacency = {node: set() for node in SKELETON}
    for left, right in zip(SKELETON, SKELETON[1:]):
        adjacency[left].add(right)
        adjacency[right].add(left)
    pending = [(root, 0)]
    seen = {root}
    while pending:
        node, distance = pending.pop(0)
        if node == prediction:
            return distance
        for target in adjacency.get(node, ()):
            if target not in seen:
                seen.add(target)
                pending.append((target, distance + 1))
    return None


def _split_audit(calibration: list[NaturalNoisePair], evaluation: list[NaturalNoisePair]) -> dict[str, Any]:
    cal_ids = {pair.instance_id for pair in calibration}
    eval_ids = {pair.instance_id for pair in evaluation}
    cal_templates = {item for pair in calibration for item in pair.noise_template_ids}
    eval_templates = {item for pair in evaluation for item in pair.noise_template_ids}
    return {
        "instance_id_overlap": sorted(cal_ids & eval_ids),
        "noise_template_overlap": sorted(cal_templates & eval_templates),
        "calibration_contains_fault": any(pair.contains_fault for pair in calibration),
        "evaluation_missing_fault": any(not pair.contains_fault for pair in evaluation),
        "passed": not (cal_ids & eval_ids or cal_templates & eval_templates)
        and not any(pair.contains_fault for pair in calibration)
        and not any(not pair.contains_fault for pair in evaluation),
    }


def _manifest(
    calibration: list[NaturalNoisePair],
    evaluation: list[NaturalNoisePair],
    audit: dict[str, Any],
) -> dict[str, Any]:
    def item(pair: NaturalNoisePair) -> dict[str, Any]:
        return {
            "instance_id": pair.instance_id,
            "seed": pair.instance_seed,
            "split": pair.split,
            "noise_level": pair.noise_level,
            "noise_template_ids": pair.noise_template_ids,
            "contains_fault": pair.contains_fault,
            "annotations": pair.annotations,
        }
    return {
        "calibration": {"instances": [item(pair) for pair in calibration], "split_overlap_check": audit},
        "evaluation": {"instances": [item(pair) for pair in evaluation], "split_overlap_check": audit},
    }


def _fairness_audit(evaluation: list[NaturalNoisePair], profile: CalibrationProfile) -> dict[str, Any]:
    failures: list[str] = []
    rows = []
    for pair in evaluation:
        raw = build_attribution_prompt(pair.comparison, mode="paired", clean_trace=pair.clean)
        calibrated_prompt, calibrated_candidates = build_calibrated_deepseek_prompt(pair, profile)
        raw_lower = raw.prompt.casefold()
        calibrated_lower = calibrated_prompt.casefold()
        leaked = sorted(term for term in EVALUATOR_TERMS if term in raw_lower or term in calibrated_lower)
        if leaked:
            failures.append(f"{pair.instance_id}: evaluator terms {leaked}")
        if raw.candidates != calibrated_candidates:
            failures.append(f"{pair.instance_id}: candidate sets differ")
        rows.append({
            "instance_id": pair.instance_id,
            "candidate_nodes_raw": list(raw.candidates),
            "candidate_nodes_calibrated": list(calibrated_candidates),
            "same_evaluation_traces": True,
            "evaluator_terms_found": leaked,
            "fault_event_present": "fault_injected" in raw_lower or "fault_injected" in calibrated_lower,
            "calibration_source": profile.split_id,
        })
    return {
        "passed": not failures,
        "failures": failures,
        "checks": {
            "same_traces_for_paired_methods": True,
            "calibrated_context_clean_only": True,
            "identical_candidate_sets": all(
                row["candidate_nodes_raw"] == row["candidate_nodes_calibrated"] for row in rows
            ),
            "no_evaluator_metadata": all(not row["evaluator_terms_found"] for row in rows),
            "no_fault_events": all(not row["fault_event_present"] for row in rows),
        },
        "instances": rows,
        "calibration_context_exact": calibration_context(profile),
    }


def _audit_external_prompt(prompt: str, pair: NaturalNoisePair, candidates: tuple[str, ...]) -> None:
    leaked = sorted(term for term in EVALUATOR_TERMS if term in prompt.casefold())
    actual = tuple(sorted({event["node_id"] for event in serialize_trace_for_attribution(pair.clean)}))
    if leaked or candidates != actual:
        raise ValueError(f"external attribution audit failed: leaked={leaked}, candidates={candidates}")


def _uniqueness_audit(evaluation: list[NaturalNoisePair]) -> dict[str, Any]:
    prompts, clean, corrupt = set(), set(), set()
    for pair in evaluation:
        bundle = build_attribution_prompt(pair.comparison, mode="paired", clean_trace=pair.clean)
        prompts.add(bundle.prompt_sha256)
        clean.add(bundle.clean_trace_sha256)
        corrupt.add(bundle.corrupt_trace_sha256)
    expected = len(evaluation)
    return {
        "number_of_instances": expected,
        "unique_prompt_count": len(prompts),
        "unique_clean_trace_count": len(clean),
        "unique_corrupt_trace_count": len(corrupt),
        "passed": len(prompts) == expected and len(clean) == expected and len(corrupt) == expected,
    }


def _summaries(records: list[dict[str, Any]], methods: list[str]) -> list[dict[str, Any]]:
    rows = []
    for method in methods:
        method_records = [record for record in records if record["method"] == method]
        by_level = []
        for level in LEVEL_COUNTS:
            selected = [record for record in method_records if record["noise_level"] == level]
            if selected:
                by_level.append(_summary_row(method, level, selected))
        rows.extend(by_level)
        if by_level:
            n0 = next(row for row in by_level if row["noise_level"] == "N0")
            n3 = next(row for row in by_level if row["noise_level"] == "N3")
            for row in by_level:
                row["root_accuracy_degradation_n0_to_n3"] = n0["root_accuracy"] - n3["root_accuracy"]
    return rows


def _summary_row(method: str, level: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    size = len(records)
    precisions, recalls, false_rates, missed_rates = [], [], [], []
    for record in records:
        predicted = set(record["predicted_contaminated_nodes"])
        true = set(record["true_contaminated_nodes"])
        false = set(record["false_contaminated_nodes"])
        missed = set(record["missed_contaminated_nodes"])
        precisions.append(len(predicted & true) / max(1, len(predicted)))
        recalls.append(len(predicted & true) / max(1, len(true)))
        false_rates.append(len(false) / max(1, len(predicted)))
        missed_rates.append(len(missed) / max(1, len(true)))
    distances = [record["root_graph_distance"] for record in records if record["root_graph_distance"] is not None]
    return {
        "method": method,
        "noise_level": level,
        "instances": size,
        "root_accuracy": sum(record["prediction_type"] == "root" for record in records) / size,
        "mediator_selection_rate": sum(record["prediction_type"] == "mediator" for record in records) / size,
        "symptom_selection_rate": sum(record["prediction_type"] == "symptom" for record in records) / size,
        "benign_noise_selection_rate": sum(record["prediction_type"] == "benign_noise_node" for record in records) / size,
        "invalid_output_rate": sum(record["prediction_type"] == "invalid" for record in records) / size,
        "mean_graph_distance_to_root": mean(distances) if distances else None,
        "contaminated_subgraph_precision": mean(precisions),
        "contaminated_subgraph_recall": mean(recalls),
        "false_contamination_rate": mean(false_rates),
        "missed_contamination_rate": mean(missed_rates),
        "mean_latency_seconds": mean(record["latency_seconds"] for record in records),
        "api_call_count": sum(record["method"].startswith("deepseek") for record in records),
        "api_cost_usd": None,
        "root_accuracy_degradation_n0_to_n3": None,
    }


def _paired_correctness(records: list[dict[str, Any]], methods: list[str]) -> list[dict[str, Any]]:
    by_key = {(record["instance_id"], record["method"]): record for record in records}
    comparisons = []
    reference = "cascad_full"
    for method in methods:
        if method == reference:
            continue
        shared = sorted({
            record["instance_id"] for record in records
            if record["method"] == method and (record["instance_id"], reference) in by_key
        })
        comparisons.append({
            "method_a": reference,
            "method_b": method,
            "paired_instances": len(shared),
            "both_correct": sum(
                by_key[(instance, reference)]["prediction_type"] == "root"
                and by_key[(instance, method)]["prediction_type"] == "root" for instance in shared
            ),
            "a_correct_b_wrong": sum(
                by_key[(instance, reference)]["prediction_type"] == "root"
                and by_key[(instance, method)]["prediction_type"] != "root" for instance in shared
            ),
            "a_wrong_b_correct": sum(
                by_key[(instance, reference)]["prediction_type"] != "root"
                and by_key[(instance, method)]["prediction_type"] == "root" for instance in shared
            ),
            "both_wrong": sum(
                by_key[(instance, reference)]["prediction_type"] != "root"
                and by_key[(instance, method)]["prediction_type"] != "root" for instance in shared
            ),
        })
    return comparisons


def _mcnemar_tables(records: list[dict[str, Any]], methods: list[str]) -> list[dict[str, Any]]:
    """Exact 2x2 inputs only; no automatic significance claim."""
    return _paired_correctness(records, methods)


def _bootstrap_artifacts(records: list[dict[str, Any]], methods: list[str]) -> list[dict[str, Any]]:
    result = []
    for method in methods:
        for level in LEVEL_COUNTS:
            selected = [record for record in records if record["method"] == method and record["noise_level"] == level]
            for metric in ("root_accuracy", "false_contamination_rate", "subgraph_precision", "subgraph_recall"):
                values = [_record_metric(record, metric) for record in selected]
                low, high = _bootstrap_ci(values, seed=f"{method}:{level}:{metric}")
                result.append({
                    "method": method, "noise_level": level, "metric": metric,
                    "estimate": mean(values), "ci95_low": low, "ci95_high": high,
                    "bootstrap_samples": 2000,
                })
    return result


def _record_metric(record: dict[str, Any], metric: str) -> float:
    predicted = set(record["predicted_contaminated_nodes"])
    true = set(record["true_contaminated_nodes"])
    if metric == "root_accuracy":
        return float(record["prediction_type"] == "root")
    if metric == "false_contamination_rate":
        return len(predicted - true) / max(1, len(predicted))
    if metric == "subgraph_precision":
        return len(predicted & true) / max(1, len(predicted))
    return len(predicted & true) / max(1, len(true))


def _bootstrap_ci(values: list[float], seed: str) -> tuple[float, float]:
    random = Random(int.from_bytes(sha256(seed.encode()).digest()[:8], "big"))
    samples = sorted(
        mean(random.choice(values) for _ in values)
        for _ in range(2000)
    )
    return samples[49], samples[1949]


def _write_plots(directory: Path, summaries: list[dict[str, Any]]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    specs = {
        "root_accuracy_vs_noise.svg": ("root_accuracy", "Root accuracy"),
        "benign_noise_selection_vs_noise.svg": ("benign_noise_selection_rate", "Benign-noise selection rate"),
        "false_contamination_vs_noise.svg": ("false_contamination_rate", "False-contamination rate"),
        "subgraph_precision_vs_noise.svg": ("contaminated_subgraph_precision", "Subgraph precision"),
        "subgraph_recall_vs_noise.svg": ("contaminated_subgraph_recall", "Subgraph recall"),
        "latency_vs_noise.svg": ("mean_latency_seconds", "Mean latency (seconds)"),
    }
    for filename, (metric, title) in specs.items():
        _write_svg_plot(directory / filename, summaries, metric, title)


def _write_svg_plot(path: Path, rows: list[dict[str, Any]], metric: str, title: str) -> None:
    methods = sorted({row["method"] for row in rows})
    width, height, left, top, chart_w, chart_h = 900, 520, 80, 55, 760, 360
    values = [float(row[metric] or 0.0) for row in rows]
    maximum = max(values + [1.0 if "rate" in metric or "accuracy" in metric or "precision" in metric or "recall" in metric else 0.001])
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="18">{title}</text>',
        f'<path d="M {left} {top} V {top+chart_h} H {left+chart_w}" fill="none" stroke="currentColor"/>',
    ]
    for method_index, method in enumerate(methods):
        selected = {row["noise_level"]: row for row in rows if row["method"] == method}
        points = []
        for index, level in enumerate(LEVEL_COUNTS):
            value = float(selected[level][metric] or 0.0)
            x = left + index * chart_w / 3
            y = top + chart_h - value / maximum * chart_h
            points.append(f"{x:.1f},{y:.1f}")
            lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="none" stroke="currentColor"/>')
        dash = "" if method_index == 0 else f' stroke-dasharray="{2 + method_index},{2 + method_index}"'
        lines.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="currentColor"{dash}/>')
        lines.append(f'<text x="{left+chart_w+8}" y="{top+15*method_index}" font-family="sans-serif" font-size="10">{method}</text>')
    for index, level in enumerate(LEVEL_COUNTS):
        x = left + index * chart_w / 3
        lines.append(f'<text x="{x}" y="{top+chart_h+25}" text-anchor="middle" font-family="sans-serif">{level}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")


def _json_hash(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
