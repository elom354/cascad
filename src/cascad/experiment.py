"""Reproducible, repeated and statistically reported Cascad experiments."""

from __future__ import annotations

from dataclasses import dataclass, replace
from statistics import mean
from typing import Any, Callable, Iterable

from cascad.attribution_baseline import (
    attribute_failure_detailed,
    localization_accuracy,
    normalize_attribution_mode,
)
from cascad.causal import CausalGraph
from cascad.divergence import estimate_natural_divergence
from cascad.injection import FaultInjector
from cascad.intervention import CalibratedInterventionPolicy
from cascad.metrics import PropagationMetrics, compute_metrics, variance_report
from cascad.models import EventKind, FaultSpec
from cascad.simulator import ReActPropagationSimulator


@dataclass(frozen=True)
class ExperimentCase:
    name: str
    faults: list[FaultSpec]
    fault_factory: Callable[[int], list[FaultSpec]] | None = None
    task: str | None = None
    trials: int = 1
    seed_start: int = 0
    scenario: str = "weather"
    method: str = "counterfactual"
    natural_pairs: int = 8
    epsilon: float = 0.05
    attribution_llm: Callable[[str], str] | None = None
    intervention_policy: CalibratedInterventionPolicy | None = None
    attribution_record_callback: Callable[[dict[str, Any]], None] | None = None
    attribution_mode: str = "deepseek_single_guided"
    fixed_instance_seed: int | None = None
    require_unique_instances: bool = False
    propagation_mediator: str = "memory"


@dataclass(frozen=True)
class ExperimentSummary:
    name: str
    trials: int
    mean_depth: float
    mean_delay: float
    mean_breadth: float
    mean_memory_amplification: float
    affected_nodes_union: list[str]
    method: str = "counterfactual"
    variance: dict[str, dict[str, float]] | None = None
    localization_accuracy: float | None = None
    graph_localization_accuracy: float | None = None
    clean_intervention_trigger_rate: float = 0.0
    corrupt_intervention_block_rate: float = 0.0
    encoder_used: str = "not_applicable"
    encoder_reason: str = "no textual distance computed"
    attribution_records: list[dict[str, Any]] | None = None
    root_accuracy: float | None = None
    mediator_selection_rate: float = 0.0
    visible_symptom_selection_rate: float = 0.0
    other_node_rate: float = 0.0
    invalid_output_rate: float = 0.0
    mean_root_distance: float | None = None
    unique_prompt_count: int = 0
    unique_trace_count: int = 0
    api_call_count: int = 0
    temperature: float | None = None
    model: str | None = None
    experimental_role: str = "standard"


def run_experiment(cases: Iterable[ExperimentCase]) -> list[ExperimentSummary]:
    """Run matched clean/corrupt trajectories with configurable baseline modes."""
    summaries: list[ExperimentSummary] = []
    for case in cases:
        if case.method not in {"counterfactual", "temporal", "raw_logging", "attribution"}:
            raise ValueError("unknown method")
        def clean_factory(seed: int = 0) -> Any:
            return ReActPropagationSimulator(scenario=case.scenario).run(case.task, seed=seed)

        clean_factory.cache_key = f"{case.scenario}:{case.task}:natural"  # type: ignore[attr-defined]
        natural = estimate_natural_divergence(clean_factory, M=case.natural_pairs) if case.method in {"counterfactual", "attribution"} else {}
        metrics: list[PropagationMetrics] = []
        affected_nodes: set[str] = set()
        predictions: list[str | None] = []
        truth: list[str | None] = []
        graph_predictions: list[str | None] = []
        attribution_records: list[dict[str, Any]] = []
        clean_triggered = 0
        corrupt_blocked = 0
        for trial in range(case.seed_start, case.seed_start + case.trials):
            instance_seed = case.fixed_instance_seed if case.fixed_instance_seed is not None else trial
            clean = ReActPropagationSimulator(scenario=case.scenario, intervention_policy=case.intervention_policy).run(case.task, seed=instance_seed)
            specs = case.fault_factory(instance_seed) if case.fault_factory else [
                replace(spec, seed=instance_seed) if spec.seed is None else spec for spec in case.faults
            ]
            injector = FaultInjector(specs)
            corrupt = ReActPropagationSimulator(injector, scenario=case.scenario, intervention_policy=case.intervention_policy).run(case.task, seed=instance_seed)
            metric = _raw_logging_metrics(corrupt.trace) if case.method == "raw_logging" else compute_metrics(
                corrupt.trace,
                clean_trace=clean.trace if case.method in {"counterfactual", "attribution"} else None,
                natural_divergence=natural,
                epsilon=case.epsilon,
                construction_method="temporal" if case.method == "temporal" else "counterfactual",
            )
            metrics.append(metric)
            affected_nodes.update(metric.affected_nodes)
            truth_node = _fault_node(corrupt.trace)
            graph_prediction, graph = _graph_prediction(corrupt.trace, clean.trace, natural, case.epsilon)
            graph_predictions.append(graph_prediction)
            visible_failure = _visible_failure_node(corrupt.trace)
            clean_triggered += int(bool(clean.interventions))
            corrupt_blocked += int(bool(corrupt.interventions))
            if case.method == "attribution" and case.attribution_llm:
                attribution = attribute_failure_detailed(
                    corrupt.trace,
                    case.attribution_llm,
                    clean_trace=clean.trace,
                    mode=case.attribution_mode,
                )
                bundle = attribution.prompt_bundle
                assert bundle is not None
                predictions.append(attribution.predicted_node)
                truth.append(truth_node)
                prediction_type = _prediction_type(
                    attribution.predicted_node,
                    truth_node,
                    case.propagation_mediator,
                    visible_failure,
                )
                call_metadata = getattr(
                    case.attribution_llm,
                    "last_call_metadata",
                    None,
                ) or {}
                attribution_provider = getattr(
                    case.attribution_llm,
                    "provider",
                    "mock",
                )
                record = {
                    "run_id": corrupt.trace.run_id,
                    "seed": trial,
                    "scenario": case.scenario,
                    "instance_id": corrupt.trace.metadata.get("instance_id"),
                    "instance_seed": instance_seed,
                    "attribution_mode": normalize_attribution_mode(case.attribution_mode),
                    "model": getattr(case.attribution_llm, "model", "mock"),
                    "attribution_provider": attribution_provider,
                    "temperature": getattr(case.attribution_llm, "temperature", None),
                    "prompt_sha256": bundle.prompt_sha256,
                    "clean_trace_sha256": bundle.clean_trace_sha256,
                    "corrupt_trace_sha256": bundle.corrupt_trace_sha256,
                    "clean_observable_trace": bundle.clean_observable_trace,
                    "corrupt_observable_trace": bundle.corrupt_observable_trace,
                    "candidate_nodes": list(bundle.candidates),
                    "deepseek_raw_response": attribution.raw_response,
                    "deepseek_parsed_node": attribution.predicted_node,
                    "attribution_raw_response": attribution.raw_response,
                    "attribution_parsed_node": attribution.predicted_node,
                    "parse_valid": attribution.predicted_node is not None,
                    "ground_truth_root": truth_node,
                    "injection_node": truth_node,
                    "estimated_source": attribution.predicted_node,
                    "propagation_mediator": case.propagation_mediator,
                    "visible_symptom": visible_failure,
                    "prediction_type": prediction_type,
                    "cascad_prediction": graph_prediction,
                    "cascad_estimated_source": graph_prediction,
                    "cascad_prediction_type": "root" if graph_prediction == truth_node else "other",
                    "deepseek_confidence": attribution.confidence,
                    "attribution_confidence": attribution.confidence,
                    "graph_distance_to_root": _root_distance(corrupt.trace, truth_node, attribution.predicted_node),
                    "leaked_terms": list(bundle.leaked_terms),
                    "privileged_metadata_present": bundle.privileged_metadata_present,
                    "clean_trace_present": bundle.clean_trace_present,
                    "corrupt_trace_present": bundle.corrupt_trace_present,
                    "experimental_role": "repeated_call_stability" if case.fixed_instance_seed is not None else "varied_instances",
                    # Compatibility aliases for v2 audit consumers.
                    "deepseek_predicted_node": attribution.predicted_node,
                    "visible_failure_node": visible_failure,
                    "graph_predicted_node": graph_prediction,
                    "deepseek_prediction_type": _legacy_prediction_type(prediction_type),
                    "graph_prediction_type": "root_cause" if graph_prediction == truth_node else "other",
                    "prompt": attribution.prompt,
                    **call_metadata,
                }
                attribution_records.append(record)
                if case.attribution_record_callback:
                    case.attribution_record_callback(record)
            elif case.method != "raw_logging":
                truth.append(truth_node)

        reports = {
            "depth": variance_report([float(item.propagation_depth) for item in metrics]),
            "delay": variance_report([item.propagation_delay for item in metrics]),
            "breadth": variance_report([float(item.propagation_breadth) for item in metrics]),
            "memory_persistence": variance_report([item.memory_persistence for item in metrics]),
        }
        unique_prompts = len({record["prompt_sha256"] for record in attribution_records})
        unique_traces = len({record["corrupt_trace_sha256"] for record in attribution_records})
        if case.require_unique_instances and (
            unique_prompts != case.trials or unique_traces != case.trials
        ):
            raise ValueError(
                f"varied-instance requirement failed: trials={case.trials}, "
                f"unique_prompts={unique_prompts}, unique_traces={unique_traces}"
            )
        categories = [record["prediction_type"] for record in attribution_records]
        distances = [
            float(record["graph_distance_to_root"])
            for record in attribution_records
            if record["graph_distance_to_root"] is not None
        ]
        api_calls = len(attribution_records)
        root_accuracy = categories.count("root") / api_calls if api_calls else None
        summaries.append(ExperimentSummary(
            name=case.name, trials=case.trials,
            mean_depth=mean(item.propagation_depth for item in metrics),
            mean_delay=mean(item.propagation_delay for item in metrics),
            mean_breadth=mean(item.propagation_breadth for item in metrics),
            mean_memory_amplification=mean(item.memory_amplification_factor for item in metrics),
            affected_nodes_union=sorted(affected_nodes), method=case.method,
            variance=reports,
            localization_accuracy=localization_accuracy(predictions, truth) if predictions else None,
            graph_localization_accuracy=localization_accuracy(graph_predictions, truth) if truth else None,
            clean_intervention_trigger_rate=clean_triggered / max(1, case.trials),
            corrupt_intervention_block_rate=corrupt_blocked / max(1, case.trials),
            encoder_used=metrics[-1].encoder_used if metrics else "not_applicable",
            encoder_reason=metrics[-1].encoder_reason if metrics else "no metrics",
            attribution_records=attribution_records or None,
            root_accuracy=root_accuracy,
            mediator_selection_rate=categories.count("mediator") / max(1, api_calls),
            visible_symptom_selection_rate=categories.count("symptom") / max(1, api_calls),
            other_node_rate=categories.count("other") / max(1, api_calls),
            invalid_output_rate=categories.count("invalid") / max(1, api_calls),
            mean_root_distance=mean(distances) if distances else None,
            unique_prompt_count=unique_prompts,
            unique_trace_count=unique_traces,
            api_call_count=api_calls,
            temperature=getattr(case.attribution_llm, "temperature", None) if case.attribution_llm else None,
            model=getattr(case.attribution_llm, "model", None) if case.attribution_llm else None,
            experimental_role="repeated_call_stability" if case.fixed_instance_seed is not None else "varied_instances",
        ))
    return summaries


def calibrate_epsilon(
    pilot_runs: Iterable[tuple[float, bool]], candidates: Iterable[float] = (0.0, 0.01, 0.05, 0.1, 0.2), target_false_positive_rate: float = 0.05
) -> float:
    """Choose epsilon with best F1 under a target clean false-positive rate."""
    runs = list(pilot_runs)
    if not runs:
        raise ValueError("pilot_runs cannot be empty")
    best_epsilon, best_score = 0.05, -1.0
    for epsilon in candidates:
        predicted = [distance > epsilon for distance, _ in runs]
        labels = [label for _, label in runs]
        fp = sum(pred and not label for pred, label in zip(predicted, labels))
        negatives = max(1, sum(not label for label in labels))
        if fp / negatives > target_false_positive_rate:
            continue
        tp = sum(pred and label for pred, label in zip(predicted, labels))
        precision = tp / max(1, sum(predicted))
        recall = tp / max(1, sum(labels))
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        if f1 > best_score:
            best_epsilon, best_score = epsilon, f1
    return best_epsilon


def _fault_node(trace: Any) -> str | None:
    return next((event.node_id for event in trace.events if event.kind == EventKind.FAULT_INJECTED), None)


def _visible_failure_node(trace: Any) -> str | None:
    """First observable explicit failure, excluding privileged injection events."""
    for event in trace.events:
        if event.kind in {EventKind.FAULT_INJECTED, EventKind.NODE_START, EventKind.NODE_END}:
            continue
        if event.kind == EventKind.ERROR_OBSERVED or event.payload.get("ok") is False or event.payload.get("exception_type"):
            return event.node_id
    return None


def _graph_prediction(trace: Any, clean_trace: Any, natural: Any, epsilon: float) -> tuple[str | None, CausalGraph]:
    graph = CausalGraph.from_trace(
        trace, clean_trace=clean_trace, natural_divergence=natural, epsilon=epsilon,
        construction_method="counterfactual",
    )
    if not graph.contaminated_nodes:
        return None, graph
    targets = {edge.target for edge in graph.edges}
    roots = graph.contaminated_nodes - targets
    event_order = {event.node_id: index for index, event in enumerate(trace.events)}
    return min(roots or graph.contaminated_nodes, key=lambda node: event_order.get(node, 10**9)), graph


def _prediction_type(
    prediction: str | None,
    root: str | None,
    mediator: str | None,
    symptom: str | None,
) -> str:
    if prediction is None:
        return "invalid"
    if prediction == root:
        return "root"
    if prediction == mediator:
        return "mediator"
    if prediction == symptom:
        return "symptom"
    return "other"


def _legacy_prediction_type(category: str) -> str:
    return {"root": "root_cause", "symptom": "visible_symptom"}.get(category, category)


def _root_distance(trace: Any, root: str | None, prediction: str | None) -> int | None:
    if root is None or prediction is None:
        return None
    if prediction == root:
        return 0
    return CausalGraph.from_trace(trace).shortest_depths(root).get(prediction)


def _raw_logging_metrics(trace: Any) -> PropagationMetrics:
    """Placeholder only: raw logging deliberately does not infer propagation."""
    return PropagationMetrics(
        propagation_depth=0, propagation_delay=0.0, propagation_breadth=0,
        memory_amplification_factor=0.0, affected_nodes=[],
        first_fault_node=_fault_node(trace), first_visible_error_node=None,
    )
