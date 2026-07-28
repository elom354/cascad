"""Branched-topology ablation for native dependency constraints."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from random import Random
from statistics import mean, stdev
from time import perf_counter
from typing import Any, Iterable

from cascad.attribution_baseline import build_attribution_prompt, serialize_trace_for_attribution
from cascad.divergence import DivergenceDistribution, corresponding_events, encoder_status, event_distance
from cascad.export import write_csv, write_json
from cascad.models import CausalEdge, NodeEvent, RunTrace


SCENARIO = "cloud_branched_dependency"
ROOT = "share"
MEDIATOR = "memory"
SYMPTOM = "notify"
MAIN_NODES = ("share", "memory", "notify", "responder")
INDEPENDENT_NODES = ("audit_log", "metrics_export")
ALL_NODES = ("planner", *MAIN_NODES, *INDEPENDENT_NODES)
NATIVE_EDGES = (
    ("planner", "share"),
    ("share", "memory"),
    ("memory", "notify"),
    ("notify", "responder"),
    ("planner", "audit_log"),
    ("audit_log", "metrics_export"),
)
LEVELS = ("B0", "B1", "B2", "B3")
CALIBRATION_TEMPLATE_IDS = ("branch-cal-format-01", "branch-cal-format-02")
EVALUATION_TEMPLATE_IDS = {
    "B0": (),
    "B1": ("branch-eval-format-01",),
    "B2": ("branch-eval-destination-01",),
    "B3": ("branch-eval-destination-02", "branch-eval-scope-01"),
}
SALIENT_SPECIFICATION = {
    "frozen_before_evaluation": True,
    "B2": {
        "node": "audit_log",
        "field": "destination",
        "clean_surface": "regional-records",
        "comparison_surface": "external-compliance-stream",
        "rationale": "Destination is independent of authorization and notification state.",
    },
    "B3": {
        "audit_log": {
            "field": "destination",
            "clean_surface": "regional-records",
            "comparison_surface": "long-term-governance-ledger",
        },
        "metrics_export": {
            "field": "aggregation_scope",
            "clean_surface": "daily-resource",
            "comparison_surface": "cross-region-quarterly",
        },
        "rationale": "Neither field is read by the main branch.",
    },
}


@dataclass
class BranchedPair:
    instance_id: str
    seed: int
    split: str
    level: str
    template_ids: tuple[str, ...]
    clean: RunTrace
    comparison: RunTrace
    independent_divergent_nodes: tuple[str, ...]


@dataclass(frozen=True)
class BranchedCalibration:
    split_id: str
    distributions: dict[str, DivergenceDistribution]
    variable_fields: dict[str, tuple[str, ...]]
    instance_ids: tuple[str, ...]
    template_ids: tuple[str, ...]


def generate_branched_pair(seed: int, level: str, split: str) -> BranchedPair:
    """Generate one fixed-topology pair with split-specific independent surfaces."""
    if level not in LEVELS:
        raise ValueError(f"unknown branched level: {level}")
    if split not in {"calibration", "evaluation"}:
        raise ValueError("split must be calibration or evaluation")
    clean = _build_trace(seed, corrupt=False, level=level, split=split)
    comparison = _build_trace(seed, corrupt=split == "evaluation", level=level, split=split)
    templates: tuple[str, ...]
    independent: tuple[str, ...]
    if split == "calibration":
        templates = (CALIBRATION_TEMPLATE_IDS[seed % len(CALIBRATION_TEMPLATE_IDS)],)
        _set_payload(
            comparison, "metrics_export", "format_hint",
            ("compact metrics summary", "concise measurement summary")[seed % 2],
        )
        independent = ("metrics_export",)
    else:
        templates = EVALUATION_TEMPLATE_IDS[level]
        independent = {
            "B0": (),
            "B1": ("metrics_export",),
            "B2": ("audit_log",),
            "B3": ("audit_log", "metrics_export"),
        }[level]
        if level == "B1":
            _set_payload(
                comparison, "metrics_export", "format_hint",
                ("brief operational summary", "short measurement overview")[seed % 2],
            )
        elif level == "B2":
            _set_payload(
                comparison, "audit_log", "destination",
                f"external-compliance-stream-{seed:05d}",
            )
        elif level == "B3":
            _set_payload(
                comparison, "audit_log", "destination",
                f"long-term-governance-ledger-{seed:05d}",
            )
            _set_payload(
                comparison, "metrics_export", "aggregation_scope",
                f"cross-region-quarterly-{seed:05d}",
            )
    instance_id = f"branched-{split}-{level.lower()}-{seed:05d}"
    clean.metadata["study_instance_id"] = instance_id
    comparison.metadata["study_instance_id"] = instance_id
    return BranchedPair(
        instance_id, seed, split, level, templates,
        clean, comparison, independent,
    )


def build_branched_calibration(
    pairs: Iterable[BranchedPair],
) -> BranchedCalibration:
    pairs = list(pairs)
    if not pairs or any(pair.split != "calibration" for pair in pairs):
        raise ValueError("branched calibration requires clean/clean pairs")
    variable = {"metrics_export": ("format_hint",)}
    samples: dict[str, list[float]] = {}
    for pair in pairs:
        for node, (observed, clean) in corresponding_events(
            pair.comparison, pair.clean
        ).items():
            fields = set(variable.get(node, ()))
            samples.setdefault(node, []).append(event_distance(
                _without_fields(observed, fields),
                _without_fields(clean, fields),
            ))
    return BranchedCalibration(
        split_id="branched-dependency-calibration-v1",
        distributions={
            node: DivergenceDistribution(
                mean(values),
                stdev(values) if len(values) > 1 else 0.0,
                tuple(values),
            )
            for node, values in samples.items()
        },
        variable_fields=variable,
        instance_ids=tuple(pair.instance_id for pair in pairs),
        template_ids=tuple(sorted({item for pair in pairs for item in pair.template_ids})),
    )


def diagnose_branched(
    pair: BranchedPair,
    calibration: BranchedCalibration,
    *,
    use_calibration: bool,
    use_dependencies: bool,
    epsilon: float = 0.05,
) -> tuple[str | None, set[str], dict[str, float]]:
    """Apply identical alignment/thresholding with an orthogonal graph filter."""
    distances: dict[str, float] = {}
    divergent: set[str] = set()
    for node, (observed, clean) in corresponding_events(
        pair.comparison, pair.clean
    ).items():
        fields = set(calibration.variable_fields.get(node, ())) if use_calibration else set()
        distance = event_distance(
            _without_fields(observed, fields),
            _without_fields(clean, fields),
        )
        distances[node] = distance
        baseline = (
            calibration.distributions.get(node, DivergenceDistribution(0.0, 0.0))
            if use_calibration else DivergenceDistribution(0.0, 0.0)
        )
        if distance > baseline.mean + baseline.stddev + epsilon:
            divergent.add(node)
    if not divergent:
        return None, set(), distances
    if not use_dependencies:
        prediction = _first_in_event_order(pair.comparison, divergent)
        return prediction, divergent, distances
    ancestors = _ancestors(pair.comparison, SYMPTOM) | {SYMPTOM}
    root_candidates = divergent & ancestors
    if not root_candidates:
        return None, set(), distances
    prediction = _first_topological(pair.comparison, root_candidates)
    contaminated = divergent & (_reachable(pair.comparison, prediction) | {prediction})
    return prediction, contaminated, distances


def temporal_adjacency_baseline(
    pair: BranchedPair, calibration: BranchedCalibration, epsilon: float = 0.05
) -> tuple[str | None, set[str]]:
    """Treat adjacent execution order as causality, the explicit temporal ablation."""
    _, divergent, _ = diagnose_branched(
        pair, calibration, use_calibration=True, use_dependencies=False,
        epsilon=epsilon,
    )
    if not divergent:
        return None, set()
    prediction = _first_in_event_order(pair.comparison, divergent)
    order = _observable_node_order(pair.comparison)
    start = order.index(prediction)
    return prediction, {node for node in order[start:] if node in divergent}


def naive_first_difference(pair: BranchedPair) -> str | None:
    left = _serialized_by_node(pair.clean)
    right = _serialized_by_node(pair.comparison)
    order = _observable_node_order(pair.comparison)
    return next((node for node in order if left.get(node) != right.get(node)), None)


def maximum_raw_divergence(pair: BranchedPair) -> str | None:
    distances = {
        node: event_distance(observed, clean)
        for node, (observed, clean) in corresponding_events(
            pair.comparison, pair.clean
        ).items()
    }
    return max(
        distances,
        key=lambda node: (distances[node], -_observable_node_order(pair.comparison).index(node)),
    )


def run_branched_dependency_study(
    out_dir: str | Path = "runs/branched-dependency-study",
    *,
    instances_per_level: int = 20,
    calibration_pairs: int = 24,
    epsilon: float = 0.05,
) -> dict[str, Any]:
    """Run the fixed local structural ablation and export all evidence."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    calibration_pairs_list = [
        generate_branched_pair(200_000 + index, "B1", "calibration")
        for index in range(calibration_pairs)
    ]
    evaluation_pairs = [
        generate_branched_pair(level_index * 10_000 + index, level, "evaluation")
        for level_index, level in enumerate(LEVELS)
        for index in range(instances_per_level)
    ]
    calibration = build_branched_calibration(calibration_pairs_list)
    split_audit = _split_audit(calibration_pairs_list, evaluation_pairs)
    topology_audit = graph_topology_audit(evaluation_pairs[0].comparison)
    threshold_audit = _threshold_audit(evaluation_pairs, calibration, epsilon)
    fairness = _fairness_audit(
        calibration_pairs_list, evaluation_pairs, split_audit,
        topology_audit, threshold_audit,
    )
    if not fairness["passed"]:
        raise ValueError(f"branched fairness audit failed: {fairness['failures']}")
    write_json(out / "calibration_manifest.json", {
        "instances": [_manifest_row(pair) for pair in calibration_pairs_list],
        "profile": _calibration_json(calibration),
    })
    write_json(out / "evaluation_manifest.json", {
        "instances": [_manifest_row(pair) for pair in evaluation_pairs],
        "salient_divergence_specification": SALIENT_SPECIFICATION,
    })
    write_json(out / "fairness_audit.json", fairness)
    write_json(out / "graph_topology_audit.json", topology_audit)

    raw_path = out / "raw_results.jsonl"
    raw_path.write_text("", encoding="utf-8")
    records = []
    for pair in evaluation_pairs:
        methods: list[tuple[str, str | None, set[str], float]] = []
        started = perf_counter()
        prediction, nodes = temporal_adjacency_baseline(pair, calibration, epsilon)
        methods.append(("temporal_adjacency", prediction, nodes, perf_counter() - started))
        started = perf_counter()
        prediction = naive_first_difference(pair)
        methods.append(("naive_first_raw_difference", prediction, _singleton(prediction), perf_counter() - started))
        started = perf_counter()
        prediction = maximum_raw_divergence(pair)
        methods.append(("maximum_raw_divergence", prediction, _singleton(prediction), perf_counter() - started))
        for name, use_calibration, use_dependencies in (
            ("cascad_no_calibration", False, True),
            ("cascad_no_dependencies", True, False),
            ("cascad_full", True, True),
        ):
            started = perf_counter()
            prediction, nodes, _ = diagnose_branched(
                pair, calibration,
                use_calibration=use_calibration,
                use_dependencies=use_dependencies,
                epsilon=epsilon,
            )
            methods.append((name, prediction, nodes, perf_counter() - started))
        for method, prediction, nodes, latency in methods:
            record = _record(pair, method, prediction, nodes, latency)
            records.append(record)
            _append_jsonl(raw_path, record)
    uniqueness = _uniqueness(evaluation_pairs)
    if not uniqueness["passed"]:
        raise ValueError(f"branched uniqueness failed: {uniqueness}")
    summaries = _summaries(records)
    paired = _paired(records)
    bootstrap = _bootstrap(records)
    write_json(out / "raw_results.json", records)
    write_json(out / "summary.json", summaries)
    write_csv(out / "summary.csv", summaries)
    write_json(out / "paired_correctness.json", paired)
    write_csv(out / "paired_correctness.csv", paired)
    write_json(out / "bootstrap_confidence_intervals.json", bootstrap)
    _plots(out / "plots", summaries)
    metadata = {
        "scenario": SCENARIO,
        "encoder": encoder_status(),
        "calibration_instances": len(calibration_pairs_list),
        "evaluation_instances": len(evaluation_pairs),
        "split_integrity": split_audit,
        "uniqueness": uniqueness,
        "salient_specification_frozen": True,
        "api_calls": 0,
    }
    write_json(out / "study_metadata.json", metadata)
    return metadata


def graph_topology_audit(trace: RunTrace) -> dict[str, Any]:
    actual = {(edge.source, edge.target) for edge in trace.causal_edges}
    required = set(NATIVE_EDGES)
    forbidden = {
        ("share", "audit_log"), ("share", "metrics_export"),
        ("memory", "audit_log"), ("memory", "metrics_export"),
        ("notify", "audit_log"), ("notify", "metrics_export"),
    }
    return {
        "required_edges": [list(edge) for edge in NATIVE_EDGES],
        "actual_edges": [list(edge) for edge in sorted(actual)],
        "missing_edges": [list(edge) for edge in sorted(required - actual)],
        "extra_edges": [list(edge) for edge in sorted(actual - required)],
        "forbidden_edges_present": [list(edge) for edge in sorted(actual & forbidden)],
        "share_reaches_audit_log": "audit_log" in _reachable(trace, "share"),
        "share_reaches_metrics_export": "metrics_export" in _reachable(trace, "share"),
        "passed": actual == required
        and not actual & forbidden
        and not (set(INDEPENDENT_NODES) & _reachable(trace, "share")),
    }


def _build_trace(
    seed: int, *, corrupt: bool, level: str, split: str,
    include_independent: bool = True,
) -> RunTrace:
    trace = RunTrace(metadata={"scenario": SCENARIO, "seed": seed, "split": split})
    clean_permission = ("viewer", "editor", "authorized", "approved")[seed % 4]
    changed_permission = {
        "viewer": "restricted", "editor": "read_only",
        "authorized": "limited", "approved": "blocked",
    }[clean_permission]
    permission = changed_permission if corrupt else clean_permission
    events = {
        "planner": ("plan_created", {
            "plan": ["share", "memory", "notify", "responder", "audit_log", "metrics_export"],
            "resource": f"branch-resource-{seed:05d}",
        }),
        "share": ("tool_call", {"ok": True, "permission": permission}),
        "memory": ("memory_write", {"authorization_state": permission}),
        "notify": ("tool_call", {
            "ok": not corrupt,
            "notified": not corrupt,
            **({"message": "completion recorded"} if not corrupt else {
                "message": "the recipient operation stopped after reading the stored state"
            }),
        }),
        "responder": ("final_answer", {
            "answer": (
                f"Workflow branch-resource-{seed:05d} completed."
                if not corrupt else "The workflow could not be completed."
            )
        }),
        "audit_log": ("tool_call", {
            "ok": True,
            "destination": f"regional-records-{seed:05d}",
            "entry": "workflow activity recorded",
        }),
        "metrics_export": ("tool_call", {
            "ok": True,
            "aggregation_scope": f"daily-resource-{seed:05d}",
            "format_hint": "standard measurement summary",
        }),
    }
    orders = {
        "B0": ("planner", "share", "memory", "audit_log", "notify", "metrics_export", "responder"),
        "B1": ("planner", "share", "memory", "audit_log", "notify", "metrics_export", "responder"),
        "B2": ("planner", "share", "audit_log", "memory", "notify", "metrics_export", "responder"),
        "B3": ("planner", "audit_log", "share", "memory", "metrics_export", "notify", "responder"),
    }
    for node in orders[level]:
        if not include_independent and node in INDEPENDENT_NODES:
            continue
        kind, payload = events[node]
        trace.add_event(NodeEvent(node, kind, trace.run_id, payload=payload))
    for source, target in NATIVE_EDGES:
        if include_independent or (source not in INDEPENDENT_NODES and target not in INDEPENDENT_NODES):
            trace.add_edge(CausalEdge(source, target, "data_dependency"))
    return trace


def main_outcome_without_independent_branch(seed: int, corrupt: bool) -> tuple[Any, Any]:
    """Return main result with/without the independent branch for fairness tests."""
    full = _build_trace(seed, corrupt=corrupt, level="B3", split="evaluation")
    reduced = _build_trace(
        seed, corrupt=corrupt, level="B3", split="evaluation",
        include_independent=False,
    )
    return _node_payload(full, "responder"), _node_payload(reduced, "responder")


def _threshold_audit(
    pairs: list[BranchedPair],
    calibration: BranchedCalibration,
    epsilon: float,
) -> dict[str, Any]:
    rows = []
    for pair in pairs:
        if pair.level == "B0":
            continue
        _, _, distances = diagnose_branched(
            pair, calibration, use_calibration=True, use_dependencies=False,
            epsilon=epsilon,
        )
        for node in pair.independent_divergent_nodes:
            baseline = calibration.distributions.get(node, DivergenceDistribution(0.0, 0.0))
            threshold = baseline.mean + baseline.stddev + epsilon
            fields = set(calibration.variable_fields.get(node, ()))
            observed, clean = corresponding_events(pair.comparison, pair.clean)[node]
            canonical_distance = event_distance(
                _without_fields(observed, fields), _without_fields(clean, fields)
            )
            rows.append({
                "instance_id": pair.instance_id,
                "level": pair.level,
                "node": node,
                "canonical_distance": canonical_distance,
                "threshold": threshold,
                "within_range": canonical_distance <= threshold,
                "expected_class": "calibrated_benign" if pair.level == "B1" else "salient_non_causal",
            })
    return {
        "rows": rows,
        "class_1_within_range": all(
            row["within_range"] for row in rows if row["level"] == "B1"
        ),
        "class_2_above_threshold": all(
            not row["within_range"] for row in rows if row["level"] in {"B2", "B3"}
        ),
    }


def _fairness_audit(
    calibration: list[BranchedPair],
    evaluation: list[BranchedPair],
    split: dict[str, Any],
    topology: dict[str, Any],
    threshold: dict[str, Any],
) -> dict[str, Any]:
    failures = []
    if not split["passed"]:
        failures.append("split overlap")
    if not topology["passed"]:
        failures.append("native topology")
    if not threshold["class_1_within_range"]:
        failures.append("class 1 exceeds expected range")
    if not threshold["class_2_above_threshold"]:
        failures.append("class 2 does not exceed threshold")
    outcomes = [
        {
            "seed": seed,
            "clean_equal": len(set(map(json.dumps, main_outcome_without_independent_branch(seed, False)))) == 1,
            "corrupt_equal": len(set(map(json.dumps, main_outcome_without_independent_branch(seed, True)))) == 1,
        }
        for seed in range(5)
    ]
    if not all(row["clean_equal"] and row["corrupt_equal"] for row in outcomes):
        failures.append("independent branch changes main result")
    serialized = json.dumps([
        serialize_trace_for_attribution(pair.comparison) for pair in evaluation
    ]).casefold()
    forbidden = (
        "ground_truth", "root_cause", "visible_symptom",
        "salient_non_causal", "true_contaminated",
    )
    leaked = [term for term in forbidden if term in serialized]
    if leaked:
        failures.append(f"evaluator metadata leaked: {leaked}")
    return {
        "passed": not failures,
        "failures": failures,
        "same_trace_pairs_for_all_methods": True,
        "native_edges_supplied_to_deepseek": False,
        "split_integrity": split,
        "topology": topology,
        "threshold_classes": threshold,
        "salient_specification": SALIENT_SPECIFICATION,
        "main_outcome_branch_removal": outcomes,
        "evaluator_terms_in_model_traces": leaked,
        "calibration_instances": len(calibration),
        "evaluation_instances": len(evaluation),
    }


def _record(
    pair: BranchedPair, method: str, prediction: str | None,
    contaminated: set[str], latency: float,
) -> dict[str, Any]:
    false = contaminated - set(MAIN_NODES)
    missed = set(MAIN_NODES) - contaminated
    independent = set(pair.independent_divergent_nodes)
    included_independent = contaminated & independent
    return {
        "instance_id": pair.instance_id,
        "seed": pair.seed,
        "scenario": SCENARIO,
        "noise_level": pair.level,
        "noise_template_ids": list(pair.template_ids),
        "method": method,
        "ground_truth_root": ROOT,
        "injection_node": ROOT,
        "estimated_source": prediction,
        "propagation_mediator": MEDIATOR,
        "visible_symptom": SYMPTOM,
        "prediction": prediction,
        "prediction_type": (
            "invalid" if prediction is None else
            "root" if prediction == ROOT else
            "mediator" if prediction == MEDIATOR else
            "symptom" if prediction == SYMPTOM else
            "independent_branch" if prediction in INDEPENDENT_NODES else "other"
        ),
        "root_graph_distance": _undirected_distance(prediction),
        "predicted_contaminated_nodes": sorted(contaminated),
        "true_contaminated_nodes": list(MAIN_NODES),
        "false_contaminated_nodes": sorted(false),
        "missed_contaminated_nodes": sorted(missed),
        "divergent_independent_nodes": sorted(independent),
        "included_independent_nodes": sorted(included_independent),
        "ibfir": len(included_independent) / max(1, len(independent)),
        "audit_log_included": "audit_log" in contaminated,
        "metrics_export_included": "metrics_export" in contaminated,
        "clean_trace_sha256": _hash(serialize_trace_for_attribution(pair.clean)),
        "corrupt_trace_sha256": _hash(serialize_trace_for_attribution(pair.comparison)),
        "latency_seconds": latency,
        "api_cost_usd": None,
    }


def _summaries(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for method in sorted({item["method"] for item in records}):
        for level in LEVELS:
            selected = [
                item for item in records
                if item["method"] == method and item["noise_level"] == level
            ]
            n = len(selected)
            precision = [
                len(set(item["predicted_contaminated_nodes"]) & set(MAIN_NODES))
                / max(1, len(item["predicted_contaminated_nodes"]))
                for item in selected
            ]
            recall = [
                len(set(item["predicted_contaminated_nodes"]) & set(MAIN_NODES))
                / len(MAIN_NODES)
                for item in selected
            ]
            distances = [
                item["root_graph_distance"] for item in selected
                if item["root_graph_distance"] is not None
            ]
            rows.append({
                "method": method, "noise_level": level, "instances": n,
                "root_accuracy": sum(item["prediction_type"] == "root" for item in selected) / n,
                "ibfir": mean(item["ibfir"] for item in selected),
                "contaminated_subgraph_precision": mean(precision),
                "contaminated_subgraph_recall": mean(recall),
                "false_contamination_rate": mean(
                    len(item["false_contaminated_nodes"])
                    / max(1, len(item["predicted_contaminated_nodes"]))
                    for item in selected
                ),
                "missed_contamination_rate": mean(
                    len(item["missed_contaminated_nodes"]) / len(MAIN_NODES)
                    for item in selected
                ),
                "audit_log_inclusion_rate": mean(item["audit_log_included"] for item in selected),
                "metrics_export_inclusion_rate": mean(item["metrics_export_included"] for item in selected),
                "mean_graph_distance_to_root": mean(distances) if distances else None,
                "mean_latency_seconds": mean(item["latency_seconds"] for item in selected),
                "api_cost_usd": None,
            })
    return rows


def _paired(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    index = {(item["instance_id"], item["method"]): item for item in records}
    comparisons = (
        ("cascad_full", "cascad_no_dependencies"),
        ("cascad_full", "cascad_no_calibration"),
        ("cascad_full", "temporal_adjacency"),
    )
    rows = []
    ids = sorted({item["instance_id"] for item in records})
    for a, b in comparisons:
        rows.append({
            "method_a": a, "method_b": b, "paired_instances": len(ids),
            "a_lower_ibfir": sum(index[(i, a)]["ibfir"] < index[(i, b)]["ibfir"] for i in ids),
            "equal_ibfir": sum(index[(i, a)]["ibfir"] == index[(i, b)]["ibfir"] for i in ids),
            "a_higher_ibfir": sum(index[(i, a)]["ibfir"] > index[(i, b)]["ibfir"] for i in ids),
            "both_root_correct": sum(
                index[(i, a)]["prediction_type"] == "root"
                and index[(i, b)]["prediction_type"] == "root" for i in ids
            ),
        })
    return rows


def _bootstrap(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for method in sorted({item["method"] for item in records}):
        for level in LEVELS:
            selected = [item for item in records if item["method"] == method and item["noise_level"] == level]
            metrics = {
                "root_accuracy": [float(item["prediction_type"] == "root") for item in selected],
                "ibfir": [item["ibfir"] for item in selected],
                "subgraph_precision": [
                    len(set(item["predicted_contaminated_nodes"]) & set(MAIN_NODES))
                    / max(1, len(item["predicted_contaminated_nodes"])) for item in selected
                ],
                "subgraph_recall": [
                    len(set(item["predicted_contaminated_nodes"]) & set(MAIN_NODES))
                    / len(MAIN_NODES) for item in selected
                ],
            }
            for metric, values in metrics.items():
                low, high = _bootstrap_ci(values, f"{method}:{level}:{metric}")
                rows.append({
                    "method": method, "noise_level": level, "metric": metric,
                    "estimate": mean(values), "ci95_low": low, "ci95_high": high,
                    "bootstrap_samples": 2000,
                })
    return rows


def _plots(directory: Path, rows: list[dict[str, Any]]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for filename, metric, title in (
        ("root_accuracy.svg", "root_accuracy", "Root accuracy"),
        ("ibfir.svg", "ibfir", "Independent Branch False Inclusion Rate"),
        ("subgraph_precision.svg", "contaminated_subgraph_precision", "Subgraph precision"),
        ("subgraph_recall.svg", "contaminated_subgraph_recall", "Subgraph recall"),
    ):
        _svg(directory / filename, rows, metric, title)


def _svg(path: Path, rows: list[dict[str, Any]], metric: str, title: str) -> None:
    methods = sorted({row["method"] for row in rows})
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="520" viewBox="0 0 900 520">',
        f'<text x="450" y="28" text-anchor="middle" font-family="sans-serif" font-size="18">{title}</text>',
        '<path d="M 80 55 V 415 H 840" fill="none" stroke="currentColor"/>',
    ]
    for method_index, method in enumerate(methods):
        selected = {row["noise_level"]: row for row in rows if row["method"] == method}
        points = []
        for index, level in enumerate(LEVELS):
            x = 80 + index * 760 / 3
            y = 415 - float(selected[level][metric]) * 360
            points.append(f"{x:.1f},{y:.1f}")
            lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="none" stroke="currentColor"/>')
        dash = "" if method_index == 0 else f' stroke-dasharray="{2+method_index},{2+method_index}"'
        lines.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="currentColor"{dash}/>')
        lines.append(f'<text x="848" y="{65+15*method_index}" font-family="sans-serif" font-size="10">{method}</text>')
    for index, level in enumerate(LEVELS):
        lines.append(f'<text x="{80+index*760/3:.1f}" y="440" text-anchor="middle" font-family="sans-serif">{level}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def _split_audit(calibration: list[BranchedPair], evaluation: list[BranchedPair]) -> dict[str, Any]:
    cal_ids = {pair.instance_id for pair in calibration}
    eval_ids = {pair.instance_id for pair in evaluation}
    cal_templates = {item for pair in calibration for item in pair.template_ids}
    eval_templates = {item for pair in evaluation for item in pair.template_ids}
    return {
        "instance_overlap": sorted(cal_ids & eval_ids),
        "template_overlap": sorted(cal_templates & eval_templates),
        "passed": not cal_ids & eval_ids and not cal_templates & eval_templates,
    }


def _uniqueness(pairs: list[BranchedPair]) -> dict[str, Any]:
    clean = {_hash(serialize_trace_for_attribution(pair.clean)) for pair in pairs}
    corrupt = {_hash(serialize_trace_for_attribution(pair.comparison)) for pair in pairs}
    prompts = {
        build_attribution_prompt(pair.comparison, mode="paired", clean_trace=pair.clean).prompt_sha256
        for pair in pairs
    }
    return {
        "instances": len(pairs),
        "unique_instance_ids": len({pair.instance_id for pair in pairs}),
        "unique_clean_trace_hashes": len(clean),
        "unique_corrupt_trace_hashes": len(corrupt),
        "unique_prompt_hashes": len(prompts),
        "passed": all(value == len(pairs) for value in (
            len({pair.instance_id for pair in pairs}), len(clean), len(corrupt), len(prompts)
        )),
    }


def _observable_node_order(trace: RunTrace) -> list[str]:
    return [
        event.node_id for event in trace.events
        if event.node_id in ALL_NODES
    ]


def _first_in_event_order(trace: RunTrace, nodes: set[str]) -> str:
    return next(node for node in _observable_node_order(trace) if node in nodes)


def _first_topological(trace: RunTrace, nodes: set[str]) -> str:
    ancestors_within = {
        node: len(_ancestors(trace, node) & nodes) for node in nodes
    }
    return min(nodes, key=lambda node: (ancestors_within[node], _observable_node_order(trace).index(node)))


def _reachable(trace: RunTrace, origin: str) -> set[str]:
    adjacency: dict[str, set[str]] = {}
    for edge in trace.causal_edges:
        adjacency.setdefault(edge.source, set()).add(edge.target)
    pending, seen = [origin], set()
    while pending:
        node = pending.pop()
        for target in adjacency.get(node, ()):
            if target not in seen:
                seen.add(target)
                pending.append(target)
    return seen


def _ancestors(trace: RunTrace, target: str) -> set[str]:
    reverse: dict[str, set[str]] = {}
    for edge in trace.causal_edges:
        reverse.setdefault(edge.target, set()).add(edge.source)
    pending, seen = [target], set()
    while pending:
        node = pending.pop()
        for source in reverse.get(node, ()):
            if source not in seen:
                seen.add(source)
                pending.append(source)
    return seen


def _without_fields(event: NodeEvent, fields: set[str]) -> NodeEvent:
    copy = deepcopy(event)
    copy.payload = {key: value for key, value in copy.payload.items() if key not in fields}
    return copy


def _set_payload(trace: RunTrace, node: str, field: str, value: Any) -> None:
    next(event for event in trace.events if event.node_id == node).payload[field] = value


def _node_payload(trace: RunTrace, node: str) -> dict[str, Any]:
    return next(event.payload for event in trace.events if event.node_id == node)


def _serialized_by_node(trace: RunTrace) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for event in serialize_trace_for_attribution(trace):
        result.setdefault(event["node_id"], []).append(event)
    return result


def _singleton(node: str | None) -> set[str]:
    return {node} if node else set()


def _undirected_distance(prediction: str | None) -> int | None:
    if prediction is None:
        return None
    adjacency: dict[str, set[str]] = {}
    for left, right in NATIVE_EDGES:
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)
    pending, seen = [(ROOT, 0)], {ROOT}
    while pending:
        node, distance = pending.pop(0)
        if node == prediction:
            return distance
        for target in adjacency.get(node, ()):
            if target not in seen:
                seen.add(target)
                pending.append((target, distance + 1))
    return None


def _manifest_row(pair: BranchedPair) -> dict[str, Any]:
    return {
        "instance_id": pair.instance_id,
        "seed": pair.seed,
        "split": pair.split,
        "noise_level": pair.level,
        "noise_template_ids": list(pair.template_ids),
        "independent_divergent_nodes": list(pair.independent_divergent_nodes),
    }


def _calibration_json(value: BranchedCalibration) -> dict[str, Any]:
    return {
        "split_id": value.split_id,
        "variable_fields": {key: list(fields) for key, fields in value.variable_fields.items()},
        "instance_ids": list(value.instance_ids),
        "template_ids": list(value.template_ids),
        "distributions": {
            node: {"mean": item.mean, "stddev": item.stddev, "samples": list(item.samples)}
            for node, item in value.distributions.items()
        },
    }


def _bootstrap_ci(values: list[float], seed: str) -> tuple[float, float]:
    random = Random(int.from_bytes(sha256(seed.encode()).digest()[:8], "big"))
    samples = sorted(mean(random.choice(values) for _ in values) for _ in range(2000))
    return samples[49], samples[1949]


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")


def _hash(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
