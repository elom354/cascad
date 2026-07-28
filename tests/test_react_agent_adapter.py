import json

import pytest

from cascad.adapters.react_agent import (
    aligned_divergences_with_structure,
    analyze_real_agent_pair,
    load_react_agent_trace,
)


def _event(run_id, node, kind, payload, timestamp):
    return {
        "run_id": run_id,
        "node_id": node,
        "kind": kind,
        "event_id": f"{run_id}-{timestamp}",
        "timestamp": timestamp,
        "payload": payload,
        "tags": [],
    }


def _write(path, run_id, condition, result):
    runtime = {
        "task_instance_id": "arithmetic-1",
        "agent_configuration": "local-core-no-memory",
        "condition": condition,
        "dataset": "controlled",
        "requested_model": "deepseek/deepseek-chat",
        "temperature": 0.0,
        "memory_enabled": False,
        "tool_profile": "local-core",
    }
    rows = [
        _event(run_id, "call_model", "model_request", {"runtime": runtime}, 1),
        _event(run_id, "call_model", "model_response", {"response": {"id": run_id, "content": ""}}, 2),
        _event(run_id, "tool::calculate", "tool_observation_raw", {"message": {"content": "42"}}, 3),
    ]
    if condition == "perturbed":
        rows.append(
            _event(
                run_id,
                "tool::calculate",
                "fault_injected",
                {"injection_node": "tool::calculate"},
                4,
            )
        )
    rows.extend(
        [
            _event(run_id, "tool::calculate", "tool_call", {"message": {"content": result}}, 5),
            _event(run_id, "call_model", "model_request", {"runtime": runtime, "messages": [{"content": result}]}, 6),
        ]
    )
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def test_adapter_preserves_raw_hash_and_excludes_pre_injection_observation(tmp_path):
    path = tmp_path / "trace.jsonl"
    _write(path, "clean", "clean", "42")
    trace = load_react_agent_trace(path)
    assert len(trace.metadata["raw_trace_sha256"]) == 64
    assert all(event.kind != "tool_observation_raw" for event in trace.events)


def test_pair_estimates_tool_without_using_injection_marker(tmp_path):
    clean = tmp_path / "clean.jsonl"
    perturbed = tmp_path / "perturbed.jsonl"
    _write(clean, "clean", "clean", "42")
    _write(perturbed, "perturbed", "perturbed", "7")
    result = analyze_real_agent_pair(clean, perturbed)
    assert result.injection_node == "tool::calculate"
    assert result.cascad_estimated_source == "tool::calculate"
    assert result.simple_baseline_estimated_source == "tool::calculate"


def test_pair_rejects_mismatched_configuration(tmp_path):
    clean = tmp_path / "clean.jsonl"
    perturbed = tmp_path / "perturbed.jsonl"
    _write(clean, "clean", "clean", "42")
    _write(perturbed, "perturbed", "perturbed", "7")
    rows = [json.loads(line) for line in perturbed.read_text().splitlines()]
    rows[0]["payload"]["runtime"]["tool_profile"] = "full"
    perturbed.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError, match="tool_profile"):
        analyze_real_agent_pair(clean, perturbed)


def test_structural_alignment_retains_extra_realized_events(tmp_path):
    clean = tmp_path / "clean.jsonl"
    perturbed = tmp_path / "perturbed.jsonl"
    _write(clean, "clean", "clean", "42")
    _write(perturbed, "perturbed", "perturbed", "7")
    rows = [json.loads(line) for line in perturbed.read_text().splitlines()]
    rows.append(
        _event(
            "perturbed",
            "tool::calculate",
            "tool_call",
            {"message": {"content": "8"}},
            7,
        )
    )
    perturbed.write_text("\n".join(json.dumps(row) for row in rows))
    divergences = aligned_divergences_with_structure(
        load_react_agent_trace(clean),
        load_react_agent_trace(perturbed),
    )
    extra = next(
        item
        for item in divergences
        if item.event_key == "tool::calculate|tool_call|1"
    )
    assert extra.alignment_status == "missing_clean"
    assert extra.distance == 1.0


def test_clock_normalization_preserves_semantically_wrong_time(tmp_path):
    clean = tmp_path / "clean.jsonl"
    perturbed = tmp_path / "perturbed.jsonl"
    current = "2026-07-25T06:11:30+00:00"
    incorrect = "1999-12-01T00:00:00+00:00"
    _write(clean, "clean", "clean", current)
    _write(perturbed, "perturbed", "perturbed", incorrect)
    for path in (clean, perturbed):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows.insert(
            2,
            _event(
                path.stem,
                "tool::get_current_datetime",
                "tool_observation_raw",
                {"message": {"content": current}},
                2.5,
            ),
        )
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    clean_trace = load_react_agent_trace(clean)
    perturbed_trace = load_react_agent_trace(perturbed)
    divergences = aligned_divergences_with_structure(clean_trace, perturbed_trace)
    tool = next(
        item
        for item in divergences
        if item.event_key == "tool::calculate|tool_call|0"
    )

    assert tool.distance > 0
    clean_payload = next(
        event.payload
        for event in clean_trace.events
        if event.node_id == "tool::calculate" and event.kind == "tool_call"
    )
    perturbed_payload = next(
        event.payload
        for event in perturbed_trace.events
        if event.node_id == "tool::calculate" and event.kind == "tool_call"
    )
    assert "<WALL_CLOCK_TIME>" in json.dumps(clean_payload)
    assert incorrect in json.dumps(perturbed_payload)
