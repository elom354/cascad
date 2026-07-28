"""Adapter for immutable raw traces emitted by the sibling LangGraph ReAct agent."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any

from cascad.divergence import DivergenceDistribution, value_distance
from cascad.models import CausalEdge, NodeEvent, RunTrace

_NONCAUSAL_KEYS = {
    "event_id",
    "id",
    "tool_call_id",
    "response_metadata",
    "usage_metadata",
}
_EVALUATOR_RUNTIME_KEYS = {
    "condition",
    "dataset",
    "fault_injection_enabled",
    "task_instance_id",
}
_ISO_TIME = re.compile(
    r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|\+00:00)\b"
)
_WALL_CLOCK_TOLERANCE = timedelta(days=1)
_IGNORED_KINDS = {
    "node_start",
    "node_end",
    "tool_observation_raw",
    "fault_injected",
}


@dataclass(frozen=True)
class AlignedDivergence:
    """One aligned observable event and its clean/perturbed distance."""

    event_key: str
    node_id: str
    event_kind: str
    occurrence: int
    distance: float
    alignment_status: str = "matched"


@dataclass(frozen=True)
class RealAgentPairResult:
    """Independent source estimates for one immutable real-agent pair."""

    task_instance_id: str
    agent_configuration: str
    injection_node: str | None
    cascad_estimated_source: str | None
    simple_baseline_estimated_source: str | None
    divergences: tuple[AlignedDivergence, ...]
    clean_raw_sha256: str
    perturbed_raw_sha256: str


def load_react_agent_trace(path: str | Path) -> RunTrace:
    """Load raw JSONL, retain its hash, and produce a separately canonicalized trace."""
    source = Path(path)
    raw = source.read_bytes()
    records = [
        json.loads(line)
        for line in raw.decode("utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError(f"empty real-agent trace: {source}")

    runtime = _runtime_metadata(records)
    wall_clock_anchors = _wall_clock_anchors(records)
    trace = RunTrace(
        run_id=str(records[0]["run_id"]),
        metadata={
            "source": "react-agent-real",
            "raw_trace_path": str(source),
            "raw_trace_sha256": sha256(raw).hexdigest(),
            "canonicalization": {
                "noncausal_keys_removed": sorted(_NONCAUSAL_KEYS),
                "evaluator_runtime_keys_removed": sorted(_EVALUATOR_RUNTIME_KEYS),
                "wall_clock_iso_replaced": True,
                "wall_clock_policy": (
                    "replace ISO values within 24 hours of the raw pre-injection "
                    "tool clock; preserve temporally distant observations"
                ),
                "raw_tool_observations_excluded": True,
            },
            "runtime": runtime,
        },
    )
    for record in records:
        kind = _kind(record)
        if kind == "tool_observation_raw":
            continue
        trace.add_event(
            NodeEvent(
                node_id=str(record["node_id"]),
                kind=kind,
                run_id=str(record["run_id"]),
                event_id=str(record.get("event_id", "")),
                parent_event_id=record.get("parent_event_id"),
                timestamp=float(record.get("timestamp", 0.0)),
                payload=_canonicalize(
                    record.get("payload", {}),
                    wall_clock_anchors=wall_clock_anchors,
                ),
                tags=set(record.get("tags", [])),
            )
        )
    _attach_native_edges(trace)
    return trace


def aligned_divergences(
    clean: RunTrace,
    perturbed: RunTrace,
) -> tuple[AlignedDivergence, ...]:
    """Align repeated events by node, kind and occurrence without using fault truth."""
    clean_events = _indexed_observable_events(clean)
    perturbed_events = _indexed_observable_events(perturbed)
    if tuple(clean_events) != tuple(perturbed_events):
        missing_clean = sorted(set(perturbed_events) - set(clean_events))
        missing_perturbed = sorted(set(clean_events) - set(perturbed_events))
        raise ValueError(
            "unalignable observable trajectories; "
            f"clean_missing={missing_clean}, perturbed_missing={missing_perturbed}"
        )
    output = []
    for key, clean_event in clean_events.items():
        event = perturbed_events[key]
        node_id, kind, occurrence = key
        output.append(
            AlignedDivergence(
                event_key=f"{node_id}|{kind}|{occurrence}",
                node_id=node_id,
                event_kind=kind,
                occurrence=occurrence,
                distance=value_distance(clean_event.payload, event.payload),
            )
        )
    return tuple(output)


def aligned_divergences_with_structure(
    clean: RunTrace,
    perturbed: RunTrace,
) -> tuple[AlignedDivergence, ...]:
    """Align the union and retain path additions/removals as structural changes."""
    clean_events = _indexed_observable_events(clean)
    perturbed_events = _indexed_observable_events(perturbed)
    keys = list(clean_events)
    keys.extend(key for key in perturbed_events if key not in clean_events)
    output = []
    for key in keys:
        clean_event = clean_events.get(key)
        perturbed_event = perturbed_events.get(key)
        node_id, kind, occurrence = key
        if clean_event is None:
            distance = 1.0
            status = "missing_clean"
        elif perturbed_event is None:
            distance = 1.0
            status = "missing_perturbed"
        else:
            distance = value_distance(clean_event.payload, perturbed_event.payload)
            status = "matched"
        output.append(
            AlignedDivergence(
                event_key=f"{node_id}|{kind}|{occurrence}",
                node_id=node_id,
                event_kind=kind,
                occurrence=occurrence,
                distance=distance,
                alignment_status=status,
            )
        )
    return tuple(output)


def canonical_trace_sha256(trace: RunTrace) -> str:
    """Hash only the canonical observable event sequence."""
    payload = [
        {
            "node_id": event.node_id,
            "event_kind": str(getattr(event.kind, "value", event.kind)),
            "payload": event.payload,
        }
        for event in trace.events
        if str(getattr(event.kind, "value", event.kind)) not in _IGNORED_KINDS
    ]
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def analyze_real_agent_pair(
    clean_path: str | Path,
    perturbed_path: str | Path,
    *,
    natural_divergence: dict[str, DivergenceDistribution] | None = None,
    epsilon: float = 0.0,
) -> RealAgentPairResult:
    """Estimate a source from paired real executions without reading truth first."""
    clean = load_react_agent_trace(clean_path)
    perturbed = load_react_agent_trace(perturbed_path)
    _validate_pair_metadata(clean, perturbed)
    divergences = aligned_divergences(clean, perturbed)
    natural = natural_divergence or {}

    calibrated = next(
        (
            item.node_id
            for item in divergences
            if item.distance
            > natural.get(item.event_key, DivergenceDistribution(0.0, 0.0)).mean
            + natural.get(item.event_key, DivergenceDistribution(0.0, 0.0)).stddev
            + epsilon
        ),
        None,
    )
    simple = next((item.node_id for item in divergences if item.distance > 0.0), None)
    runtime = clean.metadata["runtime"]
    return RealAgentPairResult(
        task_instance_id=str(runtime.get("task_instance_id") or ""),
        agent_configuration=str(runtime.get("agent_configuration") or ""),
        injection_node=_injection_node(perturbed),
        cascad_estimated_source=calibrated,
        simple_baseline_estimated_source=simple,
        divergences=divergences,
        clean_raw_sha256=str(clean.metadata["raw_trace_sha256"]),
        perturbed_raw_sha256=str(perturbed.metadata["raw_trace_sha256"]),
    )


def _indexed_observable_events(
    trace: RunTrace,
) -> dict[tuple[str, str, int], NodeEvent]:
    counts: Counter[tuple[str, str]] = Counter()
    output: dict[tuple[str, str, int], NodeEvent] = {}
    for event in trace.events:
        kind = str(getattr(event.kind, "value", event.kind))
        if kind in _IGNORED_KINDS:
            continue
        base = (event.node_id, kind)
        occurrence = counts[base]
        counts[base] += 1
        output[(event.node_id, kind, occurrence)] = event
    return output


def _runtime_metadata(records: list[dict[str, Any]]) -> dict[str, Any]:
    for record in records:
        if _kind(record) == "model_request":
            runtime = record.get("payload", {}).get("runtime", {})
            if isinstance(runtime, dict):
                return runtime
    return {}


def _injection_node(trace: RunTrace) -> str | None:
    for event in trace.events:
        if str(getattr(event.kind, "value", event.kind)) == "fault_injected":
            value = event.payload.get("injection_node")
            return str(value) if value else event.node_id
    return None


def _validate_pair_metadata(clean: RunTrace, perturbed: RunTrace) -> None:
    left = clean.metadata["runtime"]
    right = perturbed.metadata["runtime"]
    for key in (
        "task_instance_id",
        "agent_configuration",
        "requested_model",
        "temperature",
        "memory_enabled",
        "tool_profile",
    ):
        if left.get(key) != right.get(key):
            raise ValueError(f"paired execution mismatch for {key}: {left.get(key)!r} != {right.get(key)!r}")
    if left.get("condition") != "clean" or right.get("condition") != "perturbed":
        raise ValueError("pair must contain clean then perturbed validation conditions")


def _attach_native_edges(trace: RunTrace) -> None:
    nodes = []
    for event in trace.events:
        if event.node_id not in nodes:
            nodes.append(event.node_id)
    for source, target in zip(nodes, nodes[1:]):
        relation = (
            "tool_dependency"
            if source == "call_model" or target == "call_model"
            else "memory_dependency"
            if "memory" in source or "memory" in target
            else "control_dependency"
        )
        trace.add_edge(
            CausalEdge(
                source=source,
                target=target,
                relation=relation,
                evidence="native LangGraph execution order; not inferred causality",
            )
        )


def _canonicalize(
    value: Any,
    *,
    wall_clock_anchors: tuple[datetime, ...] = (),
) -> Any:
    if isinstance(value, dict):
        output = {}
        for key, item in value.items():
            if key in _NONCAUSAL_KEYS:
                continue
            if key == "runtime" and isinstance(item, dict):
                output[key] = {
                    child_key: _canonicalize(
                        child,
                        wall_clock_anchors=wall_clock_anchors,
                    )
                    for child_key, child in item.items()
                    if child_key not in _EVALUATOR_RUNTIME_KEYS
                }
            else:
                output[key] = _canonicalize(
                    item,
                    wall_clock_anchors=wall_clock_anchors,
                )
        return output
    if isinstance(value, list):
        return [
            _canonicalize(item, wall_clock_anchors=wall_clock_anchors)
            for item in value
        ]
    if isinstance(value, str):
        return _ISO_TIME.sub(
            lambda match: (
                "<WALL_CLOCK_TIME>"
                if _is_wall_clock_value(match.group(), wall_clock_anchors)
                else match.group()
            ),
            value,
        )
    return value


def _wall_clock_anchors(records: list[dict[str, Any]]) -> tuple[datetime, ...]:
    """Extract nuisance clock values before injection without exposing them."""
    anchors = []
    for record in records:
        if _kind(record) != "tool_observation_raw":
            continue
        encoded = json.dumps(record.get("payload", {}), ensure_ascii=False)
        anchors.extend(_parse_iso(match.group()) for match in _ISO_TIME.finditer(encoded))
    return tuple(anchor for anchor in anchors if anchor is not None)


def _is_wall_clock_value(
    value: str,
    anchors: tuple[datetime, ...],
) -> bool:
    parsed = _parse_iso(value)
    return bool(
        parsed is not None
        and any(abs(parsed - anchor) <= _WALL_CLOCK_TOLERANCE for anchor in anchors)
    )


def _parse_iso(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _kind(record: dict[str, Any]) -> str:
    kind = record.get("kind", "")
    return str(getattr(kind, "value", kind))
