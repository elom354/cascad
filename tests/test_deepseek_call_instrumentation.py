import json
from io import BytesIO
from urllib.error import HTTPError

import pytest

from cascad.attribution_baseline import (
    DeepSeekAttributor,
    OpenAIAttributor,
    attribute_failure_detailed,
    configured_attributors,
)
from cascad.models import NodeEvent, RunTrace


class FakeResponse:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self):
        return json.dumps(
            {
                "model": "deepseek-chat-provider-id",
                "choices": [{"message": {"content": "tool"}}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                },
            }
        ).encode()

    def getcode(self):
        return 200


def test_deepseek_call_metadata_distinguishes_wrapper_and_provider_retries(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "cascad.attribution_baseline.urlopen",
        lambda request, timeout: FakeResponse(),
    )
    trace = RunTrace(run_id="instrumented")
    trace.add_event(
        NodeEvent(
            node_id="tool",
            kind="tool_call",
            run_id=trace.run_id,
            payload={"ok": False},
        )
    )
    client = DeepSeekAttributor(api_key="test", configured_max_retries=0)
    result = attribute_failure_detailed(trace, client, mode="single-guided")
    metadata = client.last_call_metadata
    assert result.predicted_node == "tool"
    assert metadata is not None
    assert metadata["http_status"] == 200
    assert metadata["provider_returned_model"] == "deepseek-chat-provider-id"
    assert metadata["input_tokens"] == 10
    assert metadata["output_tokens"] == 2
    assert metadata["total_tokens"] == 12
    assert metadata["raw_response"] == "tool"
    assert metadata["parsed_response"] == "tool"
    assert metadata["parse_valid"] is True
    assert metadata["wrapper_attempt_count"] == 1
    assert metadata["configured_max_retries"] == 0
    assert metadata["provider_internal_retry_count"] is None
    assert metadata["error"] is None
    assert metadata["latency_ms"] >= 0


def test_openai_uses_same_instrumentation_and_explicit_reasoning_none(
    monkeypatch,
) -> None:
    captured = {}

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data.decode())
        return FakeResponse()

    monkeypatch.setattr("cascad.attribution_baseline.urlopen", fake_urlopen)
    trace = RunTrace(run_id="openai-instrumented")
    trace.add_event(
        NodeEvent(
            node_id="tool",
            kind="tool_call",
            run_id=trace.run_id,
            payload={"ok": False},
        )
    )
    client = OpenAIAttributor(api_key="test")
    result = attribute_failure_detailed(trace, client, mode="single-guided")
    assert result.predicted_node == "tool"
    assert client.last_call_metadata is not None
    assert client.last_call_metadata["provider_returned_model"] == (
        "deepseek-chat-provider-id"
    )
    assert captured["body"]["model"] == "gpt-5.6-sol"
    assert captured["body"]["reasoning_effort"] == "none"
    assert "temperature" not in captured["body"]


def test_optional_openai_key_enables_both_attributors(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "DEEPSEEK_API_KEY=deepseek-test",
                "OPENAI_API_KEY=openai-test",
                "OPENAI_MODEL=gpt-5.6-sol",
            ]
        ),
        encoding="utf-8",
    )
    clients = configured_attributors(str(env_file))
    assert [client.provider for client in clients] == ["deepseek", "openai"]


def test_missing_optional_openai_key_runs_deepseek_only(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "DEEPSEEK_API_KEY=deepseek-test\nOPENAI_API_KEY=\n",
        encoding="utf-8",
    )
    clients = configured_attributors(str(env_file))
    assert [client.provider for client in clients] == ["deepseek"]


def test_provider_http_error_keeps_bounded_diagnostic(monkeypatch) -> None:
    error = HTTPError(
        "https://example.test",
        400,
        "Bad Request",
        {},
        BytesIO(
            json.dumps(
                {"error": {"message": "unsupported model alias"}}
            ).encode()
        ),
    )
    monkeypatch.setattr(
        "cascad.attribution_baseline.urlopen",
        lambda request, timeout: (_ for _ in ()).throw(error),
    )
    client = DeepSeekAttributor(api_key="test")

    with pytest.raises(RuntimeError, match="unsupported model alias"):
        client("test prompt")

    assert client.last_call_metadata is not None
    assert client.last_call_metadata["http_status"] == 400
    assert "unsupported model alias" in client.last_call_metadata["error"]
