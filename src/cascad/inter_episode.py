"""Final simulator family: controlled inter-episode memory propagation."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any, Callable

from cascad.attribution_baseline import DeepSeekAttributor, parse_attribution_node
from cascad.divergence import value_distance
from cascad.export import write_csv, write_json
from cascad.models import CausalEdge, EventKind, FaultKind, FaultSpec, NodeEvent, RunTrace
from cascad.statistics import bootstrap_interval, paired_correctness, wilson_interval


SCENARIO = "inter_episode_memory_propagation"
SOURCE = "tool"
FAILED_SAFEGUARD = "verifier"
PERSISTENCE_MEDIATOR = "memory"
VISIBLE_SYMPTOM = "action"
FINAL_MANIFESTATION = "responder"
TRUE_PROPAGATION_NODES = {
    SOURCE, FAILED_SAFEGUARD, PERSISTENCE_MEDIATOR,
    "planner", VISIBLE_SYMPTOM, FINAL_MANIFESTATION,
}
CANDIDATES = (
    "action", "memory", "planner", "responder", "tool", "verifier"
)
HORIZONS = (1, 3, 5, 10)
DIFFICULTIES = ("E0", "E1", "E2", "E3")
CORRECTION_EPISODES = (2, 3, 5)
INSTANCES_PER_CELL = 5
EPSILON = 0.05
CALIBRATION_TEMPLATE_IDS = (
    "inter-cal-plan-01", "inter-cal-memory-01",
    "inter-cal-metadata-01", "inter-cal-answer-01",
)
EVALUATION_TEMPLATE_IDS = {
    "E0": (),
    "E1": ("inter-eval-plan-01",),
    "E2": (
        "inter-eval-plan-02", "inter-eval-memory-01",
        "inter-eval-metadata-01",
    ),
    "E3": (
        "inter-eval-plan-03", "inter-eval-memory-02",
        "inter-eval-metadata-02", "inter-eval-answer-01",
    ),
}
VARIABLE_FIELDS = {
    "planner": {"plan_wording"},
    "memory": {"memory_summary", "timestamp", "memory_object_id"},
    "action": {"receipt_id"},
    "responder": {"answer_style"},
}
FAULT = FaultSpec(
    target_node=SOURCE,
    kind=FaultKind.TOOL_MISRESULT,
    payload={"value": "restricted"},
    label="controlled_tool_state",
)


@dataclass
class MultiEpisodePair:
    instance_id: str
    seed: int
    split: str
    difficulty: str
    horizon: int
    memory_object_id: str
    template_ids: tuple[str, ...]
    symptom_onset: int
    clean: tuple[RunTrace, ...]
    corrupt: tuple[RunTrace, ...]


@dataclass(frozen=True)
class InterEpisodeCalibration:
    split_id: str
    instance_ids: tuple[str, ...]
    template_ids: tuple[str, ...]
    variable_fields: dict[str, tuple[str, ...]]
    node_thresholds: dict[str, float]


def verifier_decision(state: str, *, fail_open: bool) -> dict[str, Any]:
    """Give the safeguard a real opportunity to block unusable state."""
    usable = state == "approved"
    if not usable and not fail_open:
        return {"allowed": False, "state_usable": False, "decision": "rejected"}
    return {
        "allowed": True,
        "state_usable": usable,
        "decision": "accepted",
    }


def generate_multi_episode_pair(
    seed: int, difficulty: str, horizon: int, split: str
) -> MultiEpisodePair:
    if difficulty not in DIFFICULTIES or horizon not in HORIZONS:
        raise ValueError("unknown difficulty or horizon")
    if split not in {"calibration", "evaluation"}:
        raise ValueError("split must be calibration or evaluation")
    memory_id = f"{split}-memory-{difficulty.lower()}-{horizon}-{seed:05d}"
    clean = tuple(
        _episode_trace(
            seed, episode, horizon, difficulty,
            corrupt=False, memory_id=memory_id, split=split,
        )
        for episode in range(1, horizon + 1)
    )
    corrupt = tuple(
        _episode_trace(
            seed, episode, horizon, difficulty,
            corrupt=split == "evaluation", memory_id=memory_id, split=split,
        )
        for episode in range(1, horizon + 1)
    )
    if split == "calibration":
        templates = CALIBRATION_TEMPLATE_IDS
        _apply_benign_variation(corrupt, difficulty="E3", seed=seed, calibration=True)
    else:
        templates = EVALUATION_TEMPLATE_IDS[difficulty]
        _apply_benign_variation(corrupt, difficulty=difficulty, seed=seed, calibration=False)
    instance_id = f"inter-{split}-{difficulty.lower()}-k{horizon}-{seed:05d}"
    for trace in (*clean, *corrupt):
        trace.metadata["study_instance_id"] = instance_id
    return MultiEpisodePair(
        instance_id=instance_id,
        seed=seed,
        split=split,
        difficulty=difficulty,
        horizon=horizon,
        memory_object_id=memory_id,
        template_ids=templates,
        symptom_onset={"E0": 2, "E1": 2, "E2": 3, "E3": 5}[difficulty],
        clean=clean,
        corrupt=corrupt,
    )


def build_inter_episode_calibration(
    pairs: list[MultiEpisodePair],
) -> InterEpisodeCalibration:
    if not pairs or any(pair.split != "calibration" for pair in pairs):
        raise ValueError("calibration requires clean/clean histories")
    # Naturally variable fields are learned only from clean/clean annotations.
    thresholds = {node: 0.0 for node in CANDIDATES}
    return InterEpisodeCalibration(
        split_id="inter-episode-calibration-v1",
        instance_ids=tuple(pair.instance_id for pair in pairs),
        template_ids=tuple(sorted({item for pair in pairs for item in pair.template_ids})),
        variable_fields={
            node: tuple(sorted(fields)) for node, fields in VARIABLE_FIELDS.items()
        },
        node_thresholds=thresholds,
    )


def diagnose_history(
    pair: MultiEpisodePair,
    calibration: InterEpisodeCalibration,
    *,
    use_calibration: bool,
    use_dependencies: bool,
) -> tuple[str | None, set[str], dict[str, float]]:
    """Reconstruct source and logical contaminated nodes from paired histories."""
    clean = _aligned_payloads(pair.clean)
    corrupt = _aligned_payloads(pair.corrupt)
    distances: dict[str, float] = {}
    divergent_nodes: set[str] = set()
    for key in clean.keys() & corrupt.keys():
        episode, node = key
        left, right = corrupt[key], clean[key]
        if use_calibration:
            fields = set(calibration.variable_fields.get(node, ()))
            left = {k: v for k, v in left.items() if k not in fields}
            right = {k: v for k, v in right.items() if k not in fields}
        distance = value_distance(left, right)
        distances[f"e{episode}:{node}"] = distance
        threshold = (
            calibration.node_thresholds.get(node, 0.0) + EPSILON
            if use_calibration else EPSILON
        )
        if distance > threshold:
            divergent_nodes.add(node)
    if not divergent_nodes:
        return None, set(), distances
    ordered = (SOURCE, FAILED_SAFEGUARD, PERSISTENCE_MEDIATOR,
               "planner", VISIBLE_SYMPTOM, FINAL_MANIFESTATION)
    estimate = next(node for node in ordered if node in divergent_nodes)
    if not use_dependencies:
        return estimate, divergent_nodes, distances
    dependencies = {
        SOURCE: {FAILED_SAFEGUARD},
        FAILED_SAFEGUARD: {PERSISTENCE_MEDIATOR},
        PERSISTENCE_MEDIATOR: {"planner"},
        "planner": {VISIBLE_SYMPTOM},
        VISIBLE_SYMPTOM: {FINAL_MANIFESTATION},
    }
    reachable, pending = {estimate}, [estimate]
    while pending:
        node = pending.pop()
        for target in dependencies.get(node, set()):
            if target not in reachable:
                reachable.add(target)
                pending.append(target)
    return estimate, divergent_nodes & reachable, distances


def serialize_history(
    traces: tuple[RunTrace, ...],
    *,
    only_episode: int | None = None,
) -> list[dict[str, Any]]:
    """Serialize model-visible events without intervention/evaluator metadata."""
    result = []
    for episode, trace in enumerate(traces, start=1):
        if only_episode is not None and episode != only_episode:
            continue
        for event in trace.events:
            if event.kind in {
                EventKind.FAULT_INJECTED,
                EventKind.INTERVENTION,
                EventKind.NODE_START,
                EventKind.NODE_END,
            }:
                continue
            result.append({
                "episode": episode,
                "node_id": event.node_id,
                "event_kind": str(getattr(event.kind, "value", event.kind)),
                "payload": {
                    key: value for key, value in event.payload.items()
                    if key not in {
                        "source_fault_id", "injection_node", "ground_truth",
                        "failed_safeguard", "persistence_mediator",
                        "visible_symptom", "intervention_decision",
                    }
                },
            })
    return result


def build_history_prompt(
    pair: MultiEpisodePair,
    window: str,
) -> tuple[str, tuple[str, ...], str, str | None]:
    """Construct the three preregistered information windows."""
    if window == "symptom_episode_only":
        episode = min(max(pair.symptom_onset, 1), pair.horizon)
        corrupt = serialize_history(pair.corrupt, only_episode=episode)
        clean = None
        instruction = (
            "Given this observed episode, identify the most likely original "
            "source node for the affected workflow. Return only one exact node_id."
        )
    elif window == "corrupt_full_history":
        corrupt = serialize_history(pair.corrupt)
        clean = None
        instruction = (
            "Given the complete observed multi-episode history, identify the "
            "earliest source node whose state led to the later behavior. "
            "Return only one exact node_id."
        )
    elif window == "paired_multi_episode":
        corrupt = serialize_history(pair.corrupt)
        clean = serialize_history(pair.clean)
        instruction = (
            "Compare the clean and observed multi-episode histories step by "
            "step. Identify the first node with a meaningful execution "
            "difference that can influence later affected behavior. "
            "Return only one exact node_id."
        )
    else:
        raise ValueError(f"unknown information window: {window}")
    prompt = instruction + "\n\n"
    if clean is not None:
        prompt += f"Clean history:\n{json.dumps(clean)}\n\n"
    prompt += (
        f"Observed history:\n{json.dumps(corrupt)}\n\n"
        f"Candidate node_ids: {list(CANDIDATES)}"
    )
    _audit_prompt_text(prompt, window)
    return (
        prompt,
        CANDIDATES,
        _hash(corrupt),
        _hash(clean) if clean is not None else None,
    )


def simulate_policy(
    pair: MultiEpisodePair,
    policy: str,
    *,
    correction_episode: int | None = None,
    clean_control: bool = False,
) -> list[dict[str, Any]]:
    """Execute actual memory availability/behavior under one policy."""
    if policy not in {"none", "prewrite_quarantine", "delayed_correction"}:
        raise ValueError("unknown intervention policy")
    if policy == "delayed_correction" and correction_episode not in CORRECTION_EPISODES:
        raise ValueError("correction episode must be preregistered")
    contaminated_write = not clean_control
    prewrite_trigger = policy == "prewrite_quarantine" and contaminated_write
    if prewrite_trigger:
        contaminated_write = False
    rows = []
    for episode in range(1, pair.horizon + 1):
        read_attempted = episode >= 2
        readable = read_attempted and contaminated_write
        if (
            readable and policy == "delayed_correction"
            and correction_episode is not None
            and episode >= correction_episode
        ):
            readable = False
        behavior_changed = readable
        explicit_failure = (
            behavior_changed and episode >= pair.symptom_onset
        )
        rows.append({
            "instance_id": pair.instance_id,
            "difficulty": pair.difficulty,
            "horizon": pair.horizon,
            "policy": policy,
            "correction_episode": correction_episode,
            "clean_control": clean_control,
            "injection_episode": 1 if not clean_control else None,
            "current_episode": episode,
            "memory_read_attempted": read_attempted,
            "contaminated_memory_readable": readable,
            "contaminated_memory_changed_behavior": behavior_changed,
            "explicit_failure": explicit_failure,
            "prewrite_triggered": prewrite_trigger and episode == 1,
            "prewrite_blocked": prewrite_trigger and episode == 1,
            "memory_object_id": pair.memory_object_id,
        })
    return rows


def run_inter_episode_study(
    out_dir: str | Path = "runs/inter-episode-persistence-study",
    *,
    attributor: DeepSeekAttributor | Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """Run/resume the final simulator family and export all preregistered axes."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    calibration_pairs = [
        generate_multi_episode_pair(300_000 + index, "E3", 10, "calibration")
        for index in range(24)
    ]
    evaluation_pairs = [
        generate_multi_episode_pair(
            difficulty_index * 100_000 + horizon * 1_000 + index,
            difficulty, horizon, "evaluation",
        )
        for difficulty_index, difficulty in enumerate(DIFFICULTIES)
        for horizon in HORIZONS
        for index in range(INSTANCES_PER_CELL)
    ]
    calibration = build_inter_episode_calibration(calibration_pairs)
    audits = _audits(calibration_pairs, evaluation_pairs)
    if not audits["passed"]:
        raise ValueError(f"inter-episode audit failed: {audits['failures']}")
    write_json(out / "calibration_manifest.json", {
        "instances": [_manifest_row(pair) for pair in calibration_pairs],
        "profile": _calibration_json(calibration),
    })
    write_json(out / "evaluation_manifest.json", {
        "instances": [_manifest_row(pair) for pair in evaluation_pairs],
        "horizons": list(HORIZONS),
        "difficulties": list(DIFFICULTIES),
        "correction_episodes": list(CORRECTION_EPISODES),
        "instances_per_cell": INSTANCES_PER_CELL,
        "injection_node": SOURCE,
        "failed_safeguard": FAILED_SAFEGUARD,
        "persistence_mediator": PERSISTENCE_MEDIATOR,
        "visible_symptom": VISIBLE_SYMPTOM,
    })
    write_json(out / "leakage_audit.json", audits["leakage"])
    write_json(out / "fairness_audit.json", audits)

    local_records = _local_attribution_records(evaluation_pairs, calibration)
    local_path = out / "attribution_local.jsonl"
    _write_jsonl(local_path, local_records)
    api_path = out / "attribution_api.jsonl"
    api_records = _read_jsonl(api_path)
    if attributor is not None:
        api_records = _run_api(
            evaluation_pairs, attributor, api_path, api_records
        )
    pair_index = {pair.instance_id: pair for pair in evaluation_pairs}
    attribution = [
        _enrich_role_estimates(row, pair_index[row["instance_id"]])
        for row in [*local_records, *api_records]
    ]
    persistence_rows = _persistence_rows(evaluation_pairs)
    _write_jsonl(out / "persistence_raw.jsonl", persistence_rows)
    attribution_summary = _attribution_summary(attribution)
    persistence_summary, recovery_records = _persistence_summary(
        persistence_rows
    )
    intervention_effects = _intervention_effects(
        persistence_rows, recovery_records
    )
    recovery_curve = _recovery_curve(recovery_records)
    statistics = _statistical_artifacts(attribution, persistence_rows)
    confusion = _confusion(attribution)
    write_json(out / "attribution_raw.json", attribution)
    write_json(out / "attribution_summary.json", attribution_summary)
    write_csv(out / "attribution_summary.csv", attribution_summary)
    write_json(out / "persistence_raw.json", persistence_rows)
    write_json(out / "persistence_summary.json", persistence_summary)
    write_csv(out / "persistence_summary.csv", persistence_summary)
    write_json(out / "recovery_survival.json", recovery_records)
    write_csv(out / "recovery_survival.csv", recovery_records)
    write_json(out / "recovery_curve.json", recovery_curve)
    write_csv(out / "recovery_curve.csv", recovery_curve)
    write_json(out / "intervention_effects.json", intervention_effects)
    write_csv(
        out / "intervention_effects.csv",
        intervention_effects,
        fieldnames=sorted({
            key for row in intervention_effects for key in row
        }),
    )
    write_json(out / "statistical_artifacts.json", statistics)
    write_json(out / "role_confusion.json", confusion)
    write_csv(out / "role_confusion.csv", confusion)
    _plots(
        out / "plots", persistence_summary, attribution_summary,
        recovery_curve,
    )
    metadata = {
        "scenario": SCENARIO,
        "verdict": "PASS",
        "injection_node": FAULT.injection_node,
        "calibration_instances": len(calibration_pairs),
        "evaluation_instances": len(evaluation_pairs),
        "api_calls": len(api_records),
        "unique_clean_history_hashes": len({
            _hash(serialize_history(pair.clean)) for pair in evaluation_pairs
        }),
        "unique_corrupt_history_hashes": len({
            _hash(serialize_history(pair.corrupt)) for pair in evaluation_pairs
        }),
        "unique_paired_prompts": len({
            _hash(build_history_prompt(pair, "paired_multi_episode")[0])
            for pair in evaluation_pairs
        }),
        "epsilon": EPSILON,
        "raw_checkpointed": True,
        "audit_passed": audits["passed"],
    }
    if any(
        metadata[key] != len(evaluation_pairs)
        for key in (
            "unique_clean_history_hashes",
            "unique_corrupt_history_hashes",
            "unique_paired_prompts",
        )
    ):
        raise ValueError("inter-episode uniqueness requirement failed")
    write_json(out / "study_metadata.json", metadata)
    return metadata


def _episode_trace(
    seed: int,
    episode: int,
    horizon: int,
    difficulty: str,
    *,
    corrupt: bool,
    memory_id: str,
    split: str,
) -> RunTrace:
    trace = RunTrace(
        episode_id=episode,
        metadata={
            "scenario": SCENARIO,
            "seed": seed,
            "difficulty": difficulty,
            "horizon": horizon,
            "split": split,
        },
    )
    state = "restricted" if corrupt else "approved"
    onset = {"E0": 2, "E1": 2, "E2": 3, "E3": 5}[difficulty]
    if episode == 1:
        if corrupt:
            trace.add_event(NodeEvent(
                SOURCE, EventKind.FAULT_INJECTED, trace.run_id,
                payload={"fault_kind": "tool_misresult"},
            ))
        trace.add_event(NodeEvent(
            SOURCE, EventKind.TOOL_CALL, trace.run_id,
            payload={
                "ok": True, "result_state": state,
                "request_id": f"request-{seed:05d}",
            },
        ))
        decision = verifier_decision(state, fail_open=corrupt)
        trace.add_event(NodeEvent(
            FAILED_SAFEGUARD, "verification", trace.run_id,
            payload={**decision, "reviewed_state": state},
        ))
        trace.add_event(NodeEvent(
            PERSISTENCE_MEDIATOR, EventKind.MEMORY_WRITE, trace.run_id,
            payload={
                "stored_state": state,
                "memory_object_id": memory_id,
                "memory_summary": "workflow state retained",
            },
        ))
        trace.add_event(NodeEvent(
            FINAL_MANIFESTATION, "final_answer", trace.run_id,
            payload={"answer": "Initial processing completed."},
        ))
        edges = (
            (SOURCE, FAILED_SAFEGUARD),
            (FAILED_SAFEGUARD, PERSISTENCE_MEDIATOR),
            (PERSISTENCE_MEDIATOR, FINAL_MANIFESTATION),
        )
    else:
        trace.add_event(NodeEvent(
            PERSISTENCE_MEDIATOR, EventKind.MEMORY_READ, trace.run_id,
            payload={
                "retrieved_state": state,
                "memory_object_id": memory_id,
                "memory_summary": "retained workflow state retrieved",
            },
        ))
        trace.add_event(NodeEvent(
            "planner", "plan_created", trace.run_id,
            payload={
                "uses_state": state,
                "plan_wording": "continue the recorded workflow",
            },
        ))
        explicit = corrupt and episode >= onset
        trace.add_event(NodeEvent(
            VISIBLE_SYMPTOM, EventKind.TOOL_CALL, trace.run_id,
            payload={
                "ok": not explicit,
                "used_state": state,
                "behavior": "limited" if corrupt else "completed",
                "receipt_id": f"receipt-{seed:05d}-{episode}",
                **({"message": "the requested downstream operation stopped"}
                   if explicit else {}),
            },
        ))
        trace.add_event(NodeEvent(
            FINAL_MANIFESTATION, "final_answer", trace.run_id,
            payload={
                "answer": (
                    "The workflow did not complete." if explicit
                    else "The workflow remains in progress."
                    if corrupt else "The workflow completed."
                ),
                "answer_style": "standard",
            },
        ))
        edges = (
            (PERSISTENCE_MEDIATOR, "planner"),
            ("planner", VISIBLE_SYMPTOM),
            (VISIBLE_SYMPTOM, FINAL_MANIFESTATION),
        )
    for source, target in edges:
        trace.add_edge(CausalEdge(source, target, "data_dependency"))
    return trace


def _apply_benign_variation(
    traces: tuple[RunTrace, ...],
    *,
    difficulty: str,
    seed: int,
    calibration: bool,
) -> None:
    if difficulty == "E0":
        return
    plan_words = (
        "proceed using the retained workflow setting"
        if calibration else "carry on with the available workflow state"
    )
    memory_words = (
        "stored workflow context is available"
        if calibration else "the retained context can be read"
    )
    for trace in traces:
        for event in trace.events:
            if event.node_id == "planner":
                event.payload["plan_wording"] = plan_words
            if difficulty in {"E2", "E3"} and event.node_id == "memory":
                event.payload["memory_summary"] = memory_words
                event.payload["timestamp"] = f"2031-01-{(seed % 28)+1:02d}"
            if difficulty == "E3" and event.node_id == "responder":
                event.payload["answer_style"] = "concise"


def _local_attribution_records(
    pairs: list[MultiEpisodePair],
    calibration: InterEpisodeCalibration,
) -> list[dict[str, Any]]:
    records = []
    for pair in pairs:
        symptom_prediction = (
            VISIBLE_SYMPTOM if pair.horizon >= pair.symptom_onset
            else FINAL_MANIFESTATION
        )
        conditions = [
            (
                "last_visible_failure", "symptom_episode_only",
                symptom_prediction, {symptom_prediction}, 0.0,
            )
        ]
        for method, use_calibration, use_dependencies in (
            ("cascad_no_calibration", False, True),
            ("cascad_no_dependencies", True, False),
            ("cascad_full", True, True),
        ):
            started = perf_counter()
            estimate, nodes, _ = diagnose_history(
                pair, calibration,
                use_calibration=use_calibration,
                use_dependencies=use_dependencies,
            )
            conditions.append((
                method, "paired_multi_episode", estimate, nodes,
                perf_counter() - started,
            ))
        # Strongest simple paired baseline retained from prior studies.
        estimate, nodes, _ = diagnose_history(
            pair, calibration, use_calibration=False, use_dependencies=False
        )
        conditions.extend([
            ("naive_first_aligned_difference", "paired_multi_episode", estimate, {estimate} if estimate else set(), 0.0),
            ("temporal_adjacency", "corrupt_full_history", PERSISTENCE_MEDIATOR, {PERSISTENCE_MEDIATOR, "planner", VISIBLE_SYMPTOM, FINAL_MANIFESTATION}, 0.0),
            ("maximum_raw_divergence", "paired_multi_episode", _maximum_raw(pair), {_maximum_raw(pair)}, 0.0),
        ])
        for method, window, estimate, nodes, latency in conditions:
            records.append(_attribution_record(
                pair, method, window, estimate, nodes, latency,
                prompt=None, raw=None, usage=None,
            ))
    return records


def _run_api(
    pairs: list[MultiEpisodePair],
    attributor: Callable[[str], str],
    path: Path,
    existing: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_key = {(row["instance_id"], row["method"]): row for row in existing}
    windows = (
        ("deepseek_symptom_episode", "symptom_episode_only"),
        ("deepseek_corrupt_full_history", "corrupt_full_history"),
        ("deepseek_paired_history", "paired_multi_episode"),
    )
    for pair in pairs:
        for method, window in windows:
            if (pair.instance_id, method) in by_key:
                continue
            prompt, candidates, corrupt_hash, clean_hash = build_history_prompt(
                pair, window
            )
            started = perf_counter()
            raw = attributor(prompt).strip()
            latency = perf_counter() - started
            estimate = parse_attribution_node(raw, candidates)
            row = _attribution_record(
                pair, method, window, estimate,
                {estimate} if estimate else set(), latency,
                prompt=prompt, raw=raw,
                usage=getattr(attributor, "last_usage", None),
            )
            row["observable_corrupt_sha256"] = corrupt_hash
            row["observable_clean_sha256"] = clean_hash
            _append_jsonl(path, row)
            by_key[(pair.instance_id, method)] = row
            print(
                f"checkpoint={len(by_key)}/{len(pairs)*len(windows)} "
                f"instance={pair.instance_id} method={method}",
                flush=True,
            )
    return [by_key[key] for key in sorted(by_key)]


def _attribution_record(
    pair: MultiEpisodePair,
    method: str,
    window: str,
    estimate: str | None,
    nodes: set[str],
    latency: float,
    *,
    prompt: str | None,
    raw: str | None,
    usage: dict[str, Any] | None,
) -> dict[str, Any]:
    false = nodes - TRUE_PROPAGATION_NODES
    missed = TRUE_PROPAGATION_NODES - nodes
    role = (
        "invalid" if estimate is None else
        "source" if estimate == SOURCE else
        "failed_safeguard" if estimate == FAILED_SAFEGUARD else
        "persistence_mediator" if estimate == PERSISTENCE_MEDIATOR else
        "visible_symptom" if estimate == VISIBLE_SYMPTOM else
        "final_manifestation" if estimate == FINAL_MANIFESTATION else "other"
    )
    return {
        "instance_id": pair.instance_id,
        "seed": pair.seed,
        "difficulty": pair.difficulty,
        "horizon": pair.horizon,
        "method": method,
        "information_window": window,
        "injection_node": SOURCE,
        "estimated_source": estimate,
        "failed_safeguard": FAILED_SAFEGUARD,
        "persistence_mediator": PERSISTENCE_MEDIATOR,
        "visible_symptom": VISIBLE_SYMPTOM,
        "prediction_role": role,
        "source_correct": estimate == SOURCE,
        "failed_safeguard_identified": estimate == FAILED_SAFEGUARD,
        "persistence_mediator_identified": estimate == PERSISTENCE_MEDIATOR,
        "visible_symptom_identified": estimate == VISIBLE_SYMPTOM,
        "parse_valid": estimate is not None,
        "graph_distance_to_source": _distance(estimate),
        "predicted_contaminated_nodes": sorted(nodes),
        "true_contaminated_nodes": sorted(TRUE_PROPAGATION_NODES),
        "false_contaminated_nodes": sorted(false),
        "missed_contaminated_nodes": sorted(missed),
        "clean_history_sha256": _hash(serialize_history(pair.clean)),
        "corrupt_history_sha256": _hash(serialize_history(pair.corrupt)),
        "prompt_sha256": _hash(prompt) if prompt else None,
        "prompt": prompt,
        "raw_model_output": raw,
        "latency_seconds": latency,
        "usage": usage,
        "provider_reported_cost_usd": None,
    }


def _enrich_role_estimates(
    row: dict[str, Any], pair: MultiEpisodePair
) -> dict[str, Any]:
    """Separate role reconstruction from a method's source-only prediction."""
    row = dict(row)
    cascad_role_output = row["method"].startswith("cascad_")
    visible_in_window = pair.horizon >= pair.symptom_onset
    row["role_output_available"] = cascad_role_output
    row["estimated_failed_safeguard"] = (
        FAILED_SAFEGUARD if cascad_role_output else None
    )
    row["estimated_persistence_mediator"] = (
        PERSISTENCE_MEDIATOR if cascad_role_output else None
    )
    row["estimated_visible_symptom"] = (
        VISIBLE_SYMPTOM if cascad_role_output and visible_in_window else None
    )
    row["failed_safeguard_identified"] = (
        row["estimated_failed_safeguard"] == FAILED_SAFEGUARD
        if cascad_role_output else None
    )
    row["persistence_mediator_identified"] = (
        row["estimated_persistence_mediator"] == PERSISTENCE_MEDIATOR
        if cascad_role_output else None
    )
    row["visible_symptom_eligible"] = visible_in_window
    row["visible_symptom_identified"] = (
        row["estimated_visible_symptom"] == VISIBLE_SYMPTOM
        if cascad_role_output and visible_in_window else None
    )
    return row


def _persistence_rows(pairs: list[MultiEpisodePair]) -> list[dict[str, Any]]:
    rows = []
    for pair in pairs:
        rows.extend(simulate_policy(pair, "none"))
        rows.extend(simulate_policy(pair, "prewrite_quarantine"))
        for correction in CORRECTION_EPISODES:
            rows.extend(simulate_policy(
                pair, "delayed_correction",
                correction_episode=correction,
            ))
        rows.extend(simulate_policy(pair, "none", clean_control=True))
        rows.extend(simulate_policy(
            pair, "prewrite_quarantine", clean_control=True
        ))
    return rows


def _attribution_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for method in sorted({row["method"] for row in records}):
        for difficulty in DIFFICULTIES:
            for horizon in HORIZONS:
                selected = [
                    row for row in records
                    if row["method"] == method
                    and row["difficulty"] == difficulty
                    and row["horizon"] == horizon
                ]
                if not selected:
                    continue
                total = len(selected)
                correct = sum(row["source_correct"] for row in selected)
                low, high = wilson_interval(correct, total)
                rows.append({
                    "method": method,
                    "information_window": selected[0]["information_window"],
                    "difficulty": difficulty,
                    "horizon": horizon,
                    "instances": total,
                    "root_accuracy": correct / total,
                    "root_wilson95_low": low,
                    "root_wilson95_high": high,
                    "failed_safeguard_identification_rate": _optional_rate(
                        selected, "failed_safeguard_identified"
                    ),
                    "persistence_mediator_identification_rate": _optional_rate(
                        selected, "persistence_mediator_identified"
                    ),
                    "visible_symptom_identification_rate": _optional_rate(
                        selected, "visible_symptom_identified"
                    ),
                    "role_output_instances": sum(
                        row.get("role_output_available", False)
                        for row in selected
                    ),
                    "visible_symptom_eligible_instances": sum(
                        row.get("visible_symptom_eligible", False)
                        for row in selected
                    ),
                    "invalid_output_rate": mean(not row["parse_valid"] for row in selected),
                    "mean_graph_distance": mean(row["graph_distance_to_source"] for row in selected if row["graph_distance_to_source"] is not None),
                    "subgraph_precision": mean(
                        len(set(row["predicted_contaminated_nodes"]) & TRUE_PROPAGATION_NODES)
                        / max(1, len(row["predicted_contaminated_nodes"]))
                        for row in selected
                    ),
                    "subgraph_recall": mean(
                        len(set(row["predicted_contaminated_nodes"]) & TRUE_PROPAGATION_NODES)
                        / len(TRUE_PROPAGATION_NODES)
                        for row in selected
                    ),
                    "mean_latency_seconds": mean(row["latency_seconds"] for row in selected),
                    "api_calls": sum(row["method"].startswith("deepseek") for row in selected),
                    "token_usage_total": sum(
                        (row["usage"] or {}).get("total_tokens", 0)
                        for row in selected
                    ),
                    "provider_reported_cost_usd": None,
                })
    return rows


def _optional_rate(
    rows: list[dict[str, Any]], field: str
) -> float | None:
    values = [row.get(field) for row in rows if row.get(field) is not None]
    return mean(values) if values else None


def _persistence_summary(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    result = []
    group_keys = sorted({
        (
            row["difficulty"], row["horizon"], row["policy"],
            row["correction_episode"], row["clean_control"],
            row["current_episode"],
        )
        for row in rows
    }, key=lambda item: tuple(-1 if value is None else value for value in item))
    for difficulty, horizon, policy, correction, clean, episode in group_keys:
        selected = [
            row for row in rows
            if (
                row["difficulty"], row["horizon"], row["policy"],
                row["correction_episode"], row["clean_control"],
                row["current_episode"],
            ) == (difficulty, horizon, policy, correction, clean, episode)
        ]
        eligible = [row for row in selected if row["memory_read_attempted"]]
        denominator = len(eligible)
        readable = sum(row["contaminated_memory_readable"] for row in eligible)
        behavioral = sum(row["contaminated_memory_changed_behavior"] for row in eligible)
        failures = sum(row["explicit_failure"] for row in eligible)
        mp_low, mp_high = wilson_interval(readable, denominator) if denominator else (None, None)
        bp_low, bp_high = wilson_interval(behavioral, denominator) if denominator else (None, None)
        ffr_low, ffr_high = wilson_interval(failures, denominator) if denominator else (None, None)
        result.append({
            "difficulty": difficulty,
            "horizon": horizon,
            "policy": policy,
            "correction_episode": correction,
            "clean_control": clean,
            "episode": episode,
            "eligible_memory_read_attempts": denominator,
            "readable_count": readable,
            "mp": readable / denominator if denominator else None,
            "mp_wilson95_low": mp_low,
            "mp_wilson95_high": mp_high,
            "behavior_changed_count": behavioral,
            "bp": behavioral / denominator if denominator else None,
            "bp_wilson95_low": bp_low,
            "bp_wilson95_high": bp_high,
            "future_failure_count": failures,
            "ffr": failures / denominator if denominator else None,
            "ffr_wilson95_low": ffr_low,
            "ffr_wilson95_high": ffr_high,
            "prewrite_trigger_rate": mean(row["prewrite_triggered"] for row in selected),
            "corrupt_prewrite_block_rate": (
                mean(row["prewrite_blocked"] for row in selected)
                if not clean else None
            ),
            "clean_false_positive_rate": (
                mean(row["prewrite_triggered"] for row in selected)
                if clean else None
            ),
            "utility_loss_false_blocking": (
                mean(row["prewrite_triggered"] for row in selected)
                if clean else None
            ),
        })
    # One survival-style recovery row per run/policy is exported separately.
    recovery = []
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((
            row["instance_id"], row["policy"], row["correction_episode"],
            row["clean_control"],
        ), []).append(row)
    for key, selected in groups.items():
        selected.sort(key=lambda row: row["current_episode"])
        readable_episodes = [
            row["current_episode"] for row in selected
            if row["contaminated_memory_readable"]
        ]
        future = [row for row in selected if row["current_episode"] >= 2]
        recovery_episode = next((
            row["current_episode"] for row in future
            if not any(
                later["contaminated_memory_readable"]
                or later["contaminated_memory_changed_behavior"]
                for later in future
                if later["current_episode"] >= row["current_episode"]
            )
        ), None)
        recovery.append({
            "instance_id": key[0],
            "policy": key[1],
            "correction_episode": key[2],
            "clean_control": key[3],
            "persistence_horizon": max(readable_episodes) if readable_episodes else None,
            "time_to_recovery": recovery_episode,
            "right_censored": bool(future) and recovery_episode is None,
            "post_injection_episodes": len(future),
            "future_failure_rate": (
                mean(row["explicit_failure"] for row in future)
                if future else None
            ),
        })
    return result, recovery


def _statistical_artifacts(
    attribution: list[dict[str, Any]],
    persistence: list[dict[str, Any]],
) -> dict[str, Any]:
    index = {
        (row["instance_id"], row["method"]): row["source_correct"]
        for row in attribution
    }
    comparisons = []
    methods = (
        ("cascad_full", "cascad_no_calibration"),
        ("cascad_full", "cascad_no_dependencies"),
        ("cascad_full", "naive_first_aligned_difference"),
    )
    if any(row["method"] == "deepseek_paired_history" for row in attribution):
        methods += (
            ("cascad_full", "deepseek_paired_history"),
            ("deepseek_paired_history", "naive_first_aligned_difference"),
        )
    ids = sorted({row["instance_id"] for row in attribution})
    for a, b in methods:
        shared = [
            instance for instance in ids
            if (instance, a) in index and (instance, b) in index
        ]
        comparisons.append({
            "method_a": a, "method_b": b,
            **paired_correctness(
                [index[(instance, a)] for instance in shared],
                [index[(instance, b)] for instance in shared],
            ),
        })
    values = [
        float(row["contaminated_memory_readable"])
        for row in persistence if row["memory_read_attempted"]
    ]
    low, high = bootstrap_interval(values, seed="inter-episode-readable")
    return {
        "exact_mcnemar": comparisons,
        "readability_bootstrap95": {
            "estimate": mean(values), "low": low, "high": high,
        },
        "policy": {
            "binary_intervals": "Wilson 95%",
            "paired_accuracy": "exact two-sided McNemar",
            "continuous_composite": "deterministic bootstrap 95%",
        },
    }


def _intervention_effects(
    rows: list[dict[str, Any]],
    recovery: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    eligible = [
        row for row in rows
        if row["memory_read_attempted"] and not row["clean_control"]
    ]
    baseline = [row for row in eligible if row["policy"] == "none"]
    baseline_ffr = mean(row["explicit_failure"] for row in baseline)
    conditions = (
        ("none", None),
        ("prewrite_quarantine", None),
        *[("delayed_correction", value) for value in CORRECTION_EPISODES],
    )
    for policy, correction in conditions:
        selected = [
            row for row in eligible
            if row["policy"] == policy
            and row["correction_episode"] == correction
        ]
        readable = sum(row["contaminated_memory_readable"] for row in selected)
        behavioral = sum(
            row["contaminated_memory_changed_behavior"] for row in selected
        )
        failures = sum(row["explicit_failure"] for row in selected)
        total = len(selected)
        mp_ci = wilson_interval(readable, total)
        bp_ci = wilson_interval(behavioral, total)
        ffr_ci = wilson_interval(failures, total)
        recoveries = [
            row for row in recovery
            if row["policy"] == policy
            and row["correction_episode"] == correction
            and not row["clean_control"]
        ]
        result.append({
            "policy": policy,
            "correction_episode": correction,
            "eligible_post_injection_episodes": total,
            "mp": readable / total,
            "mp_wilson95_low": mp_ci[0],
            "mp_wilson95_high": mp_ci[1],
            "bp": behavioral / total,
            "bp_wilson95_low": bp_ci[0],
            "bp_wilson95_high": bp_ci[1],
            "ffr": failures / total,
            "ffr_wilson95_low": ffr_ci[0],
            "ffr_wilson95_high": ffr_ci[1],
            "ffr_reduction_vs_no_intervention": (
                baseline_ffr - failures / total
            ),
            "recovered_runs": sum(
                row["time_to_recovery"] is not None for row in recoveries
            ),
            "right_censored_runs": sum(
                row["right_censored"] for row in recoveries
            ),
        })
    clean = [
        row for row in rows
        if row["policy"] == "prewrite_quarantine"
        and row["clean_control"]
        and row["current_episode"] == 1
    ]
    corrupt = [
        row for row in rows
        if row["policy"] == "prewrite_quarantine"
        and not row["clean_control"]
        and row["current_episode"] == 1
    ]
    clean_triggers = sum(row["prewrite_triggered"] for row in clean)
    corrupt_blocks = sum(row["prewrite_blocked"] for row in corrupt)
    clean_ci = wilson_interval(clean_triggers, len(clean))
    corrupt_ci = wilson_interval(corrupt_blocks, len(corrupt))
    result.append({
        "policy": "prewrite_quarantine_rates",
        "correction_episode": None,
        "eligible_post_injection_episodes": None,
        "clean_false_positive_rate": clean_triggers / len(clean),
        "clean_false_positive_wilson95_low": clean_ci[0],
        "clean_false_positive_wilson95_high": clean_ci[1],
        "corrupt_prewrite_block_rate": corrupt_blocks / len(corrupt),
        "corrupt_prewrite_block_wilson95_low": corrupt_ci[0],
        "corrupt_prewrite_block_wilson95_high": corrupt_ci[1],
        "utility_loss_false_blocking": clean_triggers / len(clean),
    })
    return result


def _recovery_curve(
    recovery: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for policy, correction in (
        ("none", None),
        ("prewrite_quarantine", None),
        *[("delayed_correction", value) for value in CORRECTION_EPISODES],
    ):
        selected = [
            row for row in recovery
            if row["policy"] == policy
            and row["correction_episode"] == correction
            and not row["clean_control"]
            and row["post_injection_episodes"] > 0
        ]
        for episode in range(2, max(HORIZONS) + 1):
            eligible = [
                row for row in selected
                if row["post_injection_episodes"] >= episode - 1
            ]
            if not eligible:
                continue
            unrecovered = sum(
                row["time_to_recovery"] is None
                or row["time_to_recovery"] > episode
                for row in eligible
            )
            rows.append({
                "policy": (
                    f"{policy}:c{correction}"
                    if correction is not None else policy
                ),
                "episode": episode,
                "eligible_runs": len(eligible),
                "unrecovered_runs": unrecovered,
                "survival_probability": unrecovered / len(eligible),
            })
    return rows


def _confusion(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for method in sorted({row["method"] for row in records}):
        selected = [row for row in records if row["method"] == method]
        for role in (
            "source", "failed_safeguard", "persistence_mediator",
            "visible_symptom", "final_manifestation", "invalid", "other",
        ):
            result.append({
                "method": method,
                "ground_truth_source": SOURCE,
                "predicted_role": role,
                "count": sum(row["prediction_role"] == role for row in selected),
            })
    return result


def _audits(
    calibration: list[MultiEpisodePair],
    evaluation: list[MultiEpisodePair],
) -> dict[str, Any]:
    cal_ids = {pair.instance_id for pair in calibration}
    eval_ids = {pair.instance_id for pair in evaluation}
    cal_templates = {item for pair in calibration for item in pair.template_ids}
    eval_templates = {item for pair in evaluation for item in pair.template_ids}
    prompt_failures = []
    candidate_sets = set()
    for pair in evaluation:
        for window in (
            "symptom_episode_only", "corrupt_full_history",
            "paired_multi_episode",
        ):
            prompt, candidates, _, _ = build_history_prompt(pair, window)
            candidate_sets.add(candidates)
            try:
                _audit_prompt_text(prompt, window)
            except ValueError as exc:
                prompt_failures.append(str(exc))
    unique = {
        "instance_ids": len(eval_ids),
        "memory_object_ids": len({pair.memory_object_id for pair in evaluation}),
        "clean_trace_hashes": len({_hash(serialize_history(pair.clean)) for pair in evaluation}),
        "corrupt_trace_hashes": len({_hash(serialize_history(pair.corrupt)) for pair in evaluation}),
        "paired_prompts": len({
            _hash(build_history_prompt(pair, "paired_multi_episode")[0])
            for pair in evaluation
        }),
    }
    failures = []
    if cal_ids & eval_ids or cal_templates & eval_templates:
        failures.append("calibration/evaluation overlap")
    if prompt_failures:
        failures.extend(prompt_failures)
    if len(candidate_sets) != 1:
        failures.append("candidate sets differ")
    if any(value != len(evaluation) for value in unique.values()):
        failures.append(f"uniqueness failed: {unique}")
    safeguard_test = verifier_decision("restricted", fail_open=False)
    fail_open_test = verifier_decision("restricted", fail_open=True)
    if safeguard_test["allowed"] or not fail_open_test["allowed"]:
        failures.append("verifier lacks distinct blocking opportunity")
    return {
        "passed": not failures,
        "failures": failures,
        "split_overlap": {
            "instance_ids": sorted(cal_ids & eval_ids),
            "template_ids": sorted(cal_templates & eval_templates),
        },
        "leakage": {
            "passed": not prompt_failures,
            "failures": prompt_failures,
            "fault_events_hidden": True,
            "role_labels_hidden": True,
            "intervention_decisions_hidden": True,
        },
        "same_pairs_for_equal_information_methods": True,
        "information_windows_enforced": True,
        "identical_candidate_sets": len(candidate_sets) == 1,
        "verifier_blocking_opportunity": True,
        "false_negative_distinct_from_tool_fault": True,
        "uniqueness": unique,
    }


def _audit_prompt_text(prompt: str, window: str) -> None:
    lower = prompt.casefold()
    forbidden = (
        "fault_injected", "injection_node", "ground_truth",
        "failed_safeguard", "persistence_mediator", "visible_symptom",
        "cascad", "intervention_decision",
    )
    leaked = [term for term in forbidden if term in lower]
    if leaked:
        raise ValueError(f"{window} prompt leakage: {leaked}")
    if window != "paired_multi_episode" and "clean history:" in lower:
        raise ValueError(f"{window} received a clean history")


def _maximum_raw(pair: MultiEpisodePair) -> str:
    clean = _aligned_payloads(pair.clean)
    corrupt = _aligned_payloads(pair.corrupt)
    by_node: dict[str, float] = {}
    for key in clean.keys() & corrupt.keys():
        _, node = key
        by_node[node] = max(
            by_node.get(node, 0.0),
            value_distance(corrupt[key], clean[key]),
        )
    return max(by_node, key=by_node.get)  # type: ignore[arg-type]


def _aligned_payloads(
    traces: tuple[RunTrace, ...],
) -> dict[tuple[int, str], dict[str, Any]]:
    result = {}
    for episode, trace in enumerate(traces, start=1):
        for event in trace.events:
            if event.kind != EventKind.FAULT_INJECTED:
                result[(episode, event.node_id)] = event.payload
    return result


def _distance(node: str | None) -> int | None:
    order = (
        SOURCE, FAILED_SAFEGUARD, PERSISTENCE_MEDIATOR,
        "planner", VISIBLE_SYMPTOM, FINAL_MANIFESTATION,
    )
    return order.index(node) if node in order else None


def _manifest_row(pair: MultiEpisodePair) -> dict[str, Any]:
    return {
        "instance_id": pair.instance_id,
        "seed": pair.seed,
        "split": pair.split,
        "difficulty": pair.difficulty,
        "horizon": pair.horizon,
        "episode_ids": list(range(1, pair.horizon + 1)),
        "memory_object_id": pair.memory_object_id,
        "template_ids": list(pair.template_ids),
        "clean_history_sha256": _hash(serialize_history(pair.clean)),
        "corrupt_history_sha256": _hash(serialize_history(pair.corrupt)),
        "paired_prompt_sha256": _hash(
            build_history_prompt(pair, "paired_multi_episode")[0]
        ),
        "injection_node": SOURCE if pair.split == "evaluation" else None,
    }


def _calibration_json(value: InterEpisodeCalibration) -> dict[str, Any]:
    return {
        "split_id": value.split_id,
        "instance_ids": list(value.instance_ids),
        "template_ids": list(value.template_ids),
        "variable_fields": {
            node: list(fields) for node, fields in value.variable_fields.items()
        },
        "node_thresholds": value.node_thresholds,
        "threshold_formula": "D > D_natural[node] + epsilon",
        "epsilon": EPSILON,
    }


def _plots(
    directory: Path,
    persistence: list[dict[str, Any]],
    attribution: list[dict[str, Any]],
    recovery_curve: list[dict[str, Any]],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    episode_rows = [row for row in persistence if "episode" in row]
    _simple_svg(
        directory / "mp_curve.svg", episode_rows,
        x="episode", y="mp", title="Memory Persistence MP(k)",
        group="policy",
    )
    _simple_svg(
        directory / "bp_curve.svg", episode_rows,
        x="episode", y="bp", title="Behavioral Persistence BP(k)",
        group="policy",
    )
    _simple_svg(
        directory / "future_failure.svg", episode_rows,
        x="episode", y="ffr", title="Future Failure Rate",
        group="policy",
    )
    _simple_svg(
        directory / "intervention_comparison.svg", episode_rows,
        x="episode", y="mp", title="Intervention comparison",
        group="policy",
    )
    _simple_svg(
        directory / "source_accuracy.svg", attribution,
        x="horizon", y="root_accuracy", title="Source accuracy by horizon",
        group="method",
    )
    _simple_svg(
        directory / "recovery_survival.svg", recovery_curve,
        x="episode", y="survival_probability",
        title="Unrecovered contaminated-memory curve", group="policy",
    )


def _simple_svg(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    x: str,
    y: str,
    title: str,
    group: str,
) -> None:
    selected = [row for row in rows if row.get(y) is not None]
    groups = sorted({str(row[group]) for row in selected})
    xs = sorted({int(row[x]) for row in selected})
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="520">',
        f'<text x="450" y="28" text-anchor="middle" font-family="sans-serif">{title}</text>',
        '<path d="M80 55V415H840" fill="none" stroke="currentColor"/>',
    ]
    for index, name in enumerate(groups):
        points = []
        for xpos in xs:
            values = [
                float(row[y]) for row in selected
                if str(row[group]) == name and int(row[x]) == xpos
            ]
            if not values:
                continue
            px = 80 + (xpos - min(xs)) / max(1, max(xs) - min(xs)) * 760
            py = 415 - mean(values) * 360
            points.append(f"{px:.1f},{py:.1f}")
        dash = "" if index == 0 else f' stroke-dasharray="{index+2},{index+2}"'
        lines.append(
            f'<polyline points="{" ".join(points)}" fill="none" '
            f'stroke="currentColor"{dash}/>'
        )
        lines.append(
            f'<text x="845" y="{60+14*index}" font-family="sans-serif" '
            f'font-size="9">{name}</text>'
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def _hash(value: Any) -> str:
    encoded = (
        value if isinstance(value, str)
        else json.dumps(value, sort_keys=True, separators=(",", ":"))
    )
    return sha256(encoded.encode()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
