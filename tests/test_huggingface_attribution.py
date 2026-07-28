import csv
import importlib.util
from pathlib import Path

import pytest

from cascad.attribution_baseline import attribute_failure_detailed
from cascad.huggingface_attribution import (
    HuggingFaceAttributor,
    model_spec,
)
from cascad.models import NodeEvent, RunTrace


class FakeBackend:
    def __init__(self, response: str = "tool") -> None:
        self.response = response
        self.loaded = False

    def load(self) -> None:
        self.loaded = True

    def generate(self, prompt: str):
        return self.response, {
            "input_tokens": 20,
            "output_tokens": 1,
            "total_tokens": 21,
            "hardware": "fake-gpu",
        }


def _runner_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_huggingface_attribution.py"
    )
    spec = importlib.util.spec_from_file_location("hf_runner", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_local_attributor_uses_shared_parser_and_records_provenance() -> None:
    backend = FakeBackend()
    client = HuggingFaceAttributor(
        model_spec("qwen3-4b"),
        resolved_revision="a" * 40,
        backend=backend,
    )
    trace = RunTrace(run_id="hf-test")
    trace.add_event(NodeEvent("tool", "tool_call", trace.run_id))

    result = attribute_failure_detailed(
        trace,
        client,
        mode="single-guided",
    )

    assert result.predicted_node == "tool"
    assert client.last_call_metadata is not None
    assert client.last_call_metadata["resolved_revision"] == "a" * 40
    assert client.last_call_metadata["quantization"] == "4bit"
    assert client.last_call_metadata["attention_backend"] == "sdpa"
    assert client.last_call_metadata["cache_implementation"] == "dynamic"
    assert client.last_call_metadata["hardware"] == "fake-gpu"
    assert client.last_call_metadata["parse_valid"] is True
    assert client.last_call_metadata["parsed_response"] == "tool"


def test_model_registry_is_explicit_and_rejects_unknown_alias() -> None:
    assert model_spec("qwen3-4b").disable_thinking is True
    assert model_spec("qwen3-4b").do_sample is True
    assert model_spec("qwen3-4b").temperature == 0.7
    assert model_spec("mistral-7b").do_sample is False
    assert model_spec("mistral-7b").model_id.startswith("mistralai/")
    with pytest.raises(ValueError, match="unknown Hugging Face model"):
        model_spec("unregistered")


def test_huggingface_summary_counts_invalid_output_as_wrong() -> None:
    module = _runner_module()
    records = [
        {
            "status": "completed",
            "model_id": "model",
            "resolved_revision": "revision",
            "instance_id": "one",
            "correct": True,
            "parse_valid": True,
            "graph_correct": True,
            "latency_ms": 10.0,
            "total_tokens": 12,
        },
        {
            "status": "completed",
            "model_id": "model",
            "resolved_revision": "revision",
            "instance_id": "two",
            "correct": False,
            "parse_valid": False,
            "graph_correct": True,
            "latency_ms": 20.0,
            "total_tokens": 13,
        },
    ]

    summary = module.summarize(records)
    model = summary["models"][0]

    assert model["n"] == 2
    assert model["root_accuracy"] == 0.5
    assert model["invalid_parse_count"] == 1
    assert model["graph_vs_model_paired"]["a_correct_b_wrong"] == 1


def test_csv_export_accepts_completed_and_error_rows(tmp_path: Path) -> None:
    module = _runner_module()
    destination = tmp_path / "mixed.csv"

    module._write_csv(
        destination,
        [
            {"status": "completed", "instance_id": "one"},
            {
                "status": "error",
                "instance_id": "two",
                "error": "CUDA out of memory",
            },
        ],
    )

    with destination.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["error"] == ""
    assert rows[1]["error"] == "CUDA out of memory"


def test_summary_rejects_an_incomplete_frozen_split() -> None:
    module = _runner_module()
    config_id = module._execution_config_id(
        quantization="4bit",
        attention_backend="sdpa",
        cache_implementation="offloaded",
        max_new_tokens=32,
    )
    records = [
        {
            "status": "completed",
            "model_id": "model",
            "resolved_revision": "revision",
            "execution_config_id": config_id,
            "instance_id": "one",
            "correct": True,
            "parse_valid": True,
            "graph_correct": True,
            "latency_ms": 10.0,
            "total_tokens": 12,
        }
    ]

    summary = module.summarize(
        records,
        execution_config_id=config_id,
        expected_instance_ids={"one", "two"},
    )

    assert summary["study_complete"] is False
    assert summary["models"][0]["missing_instance_ids"] == ["two"]


def test_runtime_conformance_gate_rejects_corrupted_generation() -> None:
    module = _runner_module()
    client = HuggingFaceAttributor(
        model_spec("qwen3-4b"),
        resolved_revision="a" * 40,
        backend=FakeBackend("tool multilingual garbage"),
    )

    result = module._runtime_conformance_check(
        client,
        alias="qwen3-4b",
        execution_config_id="config",
    )

    assert result["passed"] is False
    assert result["raw_response"] == "tool multilingual garbage"


def test_capacity_gate_orders_largest_pair_first(tmp_path: Path) -> None:
    module = _runner_module()
    small_clean = tmp_path / "small-clean"
    small_bad = tmp_path / "small-bad"
    large_clean = tmp_path / "large-clean"
    large_bad = tmp_path / "large-bad"
    small_clean.write_text("a")
    small_bad.write_text("b")
    large_clean.write_text("a" * 100)
    large_bad.write_text("b" * 100)

    ordered = module._capacity_first_order(
        ["small", "large"],
        {
            "small": (small_clean, small_bad),
            "large": (large_clean, large_bad),
        },
    )

    assert ordered == ["large", "small"]
