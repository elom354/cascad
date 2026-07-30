"""Auditable single-trace and paired-trace LLM attribution baselines."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
from time import perf_counter
from typing import Any, Callable, Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from cascad.models import EventKind, RunTrace


AttributionMode = Literal[
    "deepseek_single_neutral",
    "deepseek_single_guided",
    "deepseek_paired",
]
TraceSerialization = Literal["full-v1", "compact-v1", "compact-v2"]

MODE_ALIASES: dict[str, AttributionMode] = {
    "single-neutral": "deepseek_single_neutral",
    "single-guided": "deepseek_single_guided",
    "paired": "deepseek_paired",
    "deepseek_single_neutral": "deepseek_single_neutral",
    "deepseek_single_guided": "deepseek_single_guided",
    "deepseek_paired": "deepseek_paired",
}


@dataclass(frozen=True)
class DeepSeekAttributor:
    """Minimal OpenAI-compatible DeepSeek client for attribution experiments."""

    api_key: str
    model: str = "deepseek-v4-flash"
    temperature: float = 0.0
    endpoint: str = "https://api.deepseek.com/chat/completions"
    timeout_seconds: float = 30.0
    configured_max_retries: int = 0
    provider: str = "deepseek"
    last_usage: dict[str, Any] | None = field(
        default=None, init=False, compare=False, repr=False
    )
    last_call_metadata: dict[str, Any] | None = field(
        default=None, init=False, compare=False, repr=False
    )

    @classmethod
    def from_environment(cls, env_file: str | None = None) -> "DeepSeekAttributor":
        values = _environment_values(env_file)
        api_key = os.getenv("DEEPSEEK_API_KEY") or values.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY is required for the DeepSeek attribution baseline"
            )
        return cls(
            api_key=api_key,
            model=os.getenv("DEEPSEEK_MODEL")
            or values.get("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        )

    def __call__(self, prompt: str) -> str:
        body = json.dumps(self._request_body(prompt)).encode("utf-8")
        request = Request(
            self.endpoint,
            data=body,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        started_at = datetime.now(tz=UTC).isoformat()
        started_clock = perf_counter()
        base_metadata = {
            "request_started_at": started_at,
            "request_finished_at": None,
            "latency_ms": None,
            "requested_model": self.model,
            "provider_returned_model": None,
            "http_status": None,
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "raw_response": None,
            "parsed_response": None,
            "parse_valid": None,
            "wrapper_attempt_count": 1,
            "configured_max_retries": self.configured_max_retries,
            "provider_internal_retry_count": None,
            "error": None,
        }
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
                http_status = getattr(response, "status", None) or response.getcode()
        except HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            provider_error = _provider_error_message(error_body)
            _finish_call_metadata(
                self,
                base_metadata,
                started_clock,
                http_status=exc.code,
                error=f"HTTPError: {exc.code}: {provider_error}",
            )
            raise RuntimeError(
                f"{self.provider} attribution request failed with HTTP "
                f"{exc.code}: {provider_error}"
            ) from exc
        except URLError as exc:
            _finish_call_metadata(
                self,
                base_metadata,
                started_clock,
                error=f"URLError: {exc.reason}",
            )
            raise RuntimeError(
                f"{self.provider} attribution request failed: {exc.reason}"
            ) from exc
        except Exception as exc:
            _finish_call_metadata(
                self,
                base_metadata,
                started_clock,
                error=f"{exc.__class__.__name__}: {exc}",
            )
            raise
        usage = payload.get("usage") or {}
        raw = str(payload["choices"][0]["message"]["content"]).strip()
        object.__setattr__(self, "last_usage", usage)
        _finish_call_metadata(
            self,
            base_metadata,
            started_clock,
            http_status=http_status,
            provider_returned_model=payload.get("model"),
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
            raw_response=raw,
        )
        return raw

    def _request_body(self, prompt: str) -> dict[str, Any]:
        """Build one provider request without changing the model-visible prompt."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Follow the attribution instruction exactly and return "
                        "only one candidate node_id."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        }


@dataclass(frozen=True)
class OpenAIAttributor(DeepSeekAttributor):
    """OpenAI attribution baseline using the same observable prompt contract."""

    model: str = "gpt-5.6-sol"
    endpoint: str = "https://api.openai.com/v1/chat/completions"
    provider: str = "openai"
    reasoning_effort: str = "none"

    @classmethod
    def from_environment(cls, env_file: str | None = None) -> "OpenAIAttributor":
        values = _environment_values(env_file)
        api_key = os.getenv("OPENAI_API_KEY") or values.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is required for the OpenAI attribution baseline"
            )
        return cls(
            api_key=api_key,
            model=os.getenv("OPENAI_MODEL")
            or values.get("OPENAI_MODEL", "gpt-5.6-sol"),
            reasoning_effort=os.getenv("OPENAI_REASONING_EFFORT")
            or values.get("OPENAI_REASONING_EFFORT", "none"),
        )

    def _request_body(self, prompt: str) -> dict[str, Any]:
        """Use explicit reasoning none for the Chat Completions comparison."""
        return {
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Follow the attribution instruction exactly and return "
                        "only one candidate node_id."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        }


def configured_attributors(
    env_file: str | None = ".env",
) -> tuple[DeepSeekAttributor, ...]:
    """Load DeepSeek and add OpenAI only when its optional key is configured."""
    values = _environment_values(env_file)
    deepseek_key = os.getenv("DEEPSEEK_API_KEY") or values.get("DEEPSEEK_API_KEY")
    if not deepseek_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY is required in Cascad's .env for comparison"
        )
    clients: list[DeepSeekAttributor] = [
        DeepSeekAttributor.from_environment(env_file)
    ]
    if os.getenv("OPENAI_API_KEY") or values.get("OPENAI_API_KEY"):
        clients.append(OpenAIAttributor.from_environment(env_file))
    return tuple(clients)


@dataclass(frozen=True)
class AttributionPrompt:
    """Exact model input plus reproducibility and leakage metadata."""

    mode: AttributionMode
    serialization_version: TraceSerialization
    prompt: str
    prompt_sha256: str
    clean_observable_trace: list[dict[str, Any]] | None
    corrupt_observable_trace: list[dict[str, Any]]
    shared_observable_context: dict[str, Any] | None
    clean_trace_sha256: str | None
    corrupt_trace_sha256: str
    candidates: tuple[str, ...]
    leaked_terms: tuple[str, ...]
    privileged_metadata_present: bool
    clean_trace_present: bool
    corrupt_trace_present: bool


@dataclass(frozen=True)
class AttributionResult:
    """Auditable LLM attribution with raw response and exact prompt."""

    predicted_node: str | None
    raw_response: str
    prompt: str
    candidates: tuple[str, ...]
    confidence: float | None = None
    prompt_bundle: AttributionPrompt | None = None


def normalize_attribution_mode(mode: str) -> AttributionMode:
    try:
        return MODE_ALIASES[mode]
    except KeyError as exc:
        raise ValueError(f"unknown attribution mode: {mode}") from exc


def attribute_failure(
    trace: RunTrace,
    llm: Callable[[str], str],
    *,
    clean_trace: RunTrace | None = None,
    mode: str = "deepseek_single_guided",
) -> str | None:
    """Backward-compatible parsed node attribution."""
    return attribute_failure_detailed(trace, llm, clean_trace=clean_trace, mode=mode).predicted_node


def attribute_failure_detailed(
    trace: RunTrace,
    llm: Callable[[str], str],
    *,
    clean_trace: RunTrace | None = None,
    mode: str = "deepseek_single_guided",
    serialization_version: TraceSerialization = "full-v1",
) -> AttributionResult:
    """Build, audit, execute and parse one attribution call."""
    bundle = build_attribution_prompt(
        trace,
        mode=mode,
        clean_trace=clean_trace,
        serialization_version=serialization_version,
    )
    if bundle.leaked_terms or bundle.privileged_metadata_present:
        raise ValueError(
            f"attribution prompt failed leakage audit: terms={list(bundle.leaked_terms)} "
            f"privileged={bundle.privileged_metadata_present}"
        )
    raw = llm(bundle.prompt).strip()
    predicted = _parse_node(raw, bundle.candidates)
    if getattr(llm, "last_call_metadata", None) is not None:
        finalize_parse_metadata(llm, predicted)
    return AttributionResult(
        predicted,
        raw,
        bundle.prompt,
        bundle.candidates,
        None,
        bundle,
    )


def finalize_parse_metadata(client: Any, predicted: str | None) -> None:
    """Attach parser output without changing request/retry provenance."""
    if client.last_call_metadata is None:
        return
    object.__setattr__(
        client,
        "last_call_metadata",
        {
            **client.last_call_metadata,
            "parsed_response": predicted,
            "parse_valid": predicted is not None,
        },
    )


def _finish_call_metadata(
    client: DeepSeekAttributor,
    metadata: dict[str, Any],
    started_clock: float,
    **updates: Any,
) -> None:
    finished = {
        **metadata,
        **updates,
        "request_finished_at": datetime.now(tz=UTC).isoformat(),
        "latency_ms": (perf_counter() - started_clock) * 1000,
    }
    object.__setattr__(client, "last_call_metadata", finished)


def build_attribution_prompt(
    corrupt_trace: RunTrace,
    *,
    mode: str = "deepseek_single_guided",
    clean_trace: RunTrace | None = None,
    serialization_version: TraceSerialization = "full-v1",
) -> AttributionPrompt:
    """Construct one of the three stable information-ablation prompts."""
    if serialization_version not in {"full-v1", "compact-v1", "compact-v2"}:
        raise ValueError(
            f"unknown trace serialization: {serialization_version}"
        )
    normalized_mode = normalize_attribution_mode(mode)
    observed = serialize_trace_for_attribution(
        corrupt_trace,
        serialization_version=serialization_version,
    )
    reference = (
        serialize_trace_for_attribution(
            clean_trace,
            serialization_version=serialization_version,
        )
        if clean_trace
        else None
    )
    shared_context = None
    if normalized_mode == "deepseek_paired" and serialization_version == "compact-v2":
        assert reference is not None
        shared_context, reference, observed = compact_paired_shared_context(
            reference,
            observed,
        )
    observed_candidates = tuple(sorted({event["node_id"] for event in observed}))
    if normalized_mode == "deepseek_paired":
        if reference is None:
            raise ValueError("deepseek_paired requires a clean_trace")
        reference_candidates = tuple(sorted({event["node_id"] for event in reference}))
        paired_candidates = tuple(
            sorted(set(reference_candidates) | set(observed_candidates))
        )
        serialization_note = (
            "\nCompact trace rule: in model_request events, omitted static "
            "fields inherit their previous value and messages_delta is "
            "appended to the previous messages list.\n"
            if serialization_version in {"compact-v1", "compact-v2"}
            else ""
        )
        if serialization_version == "compact-v2":
            serialization_note += (
                "Fields listed in inherits_shared are restored from the "
                "shared model-request context below.\n"
            )

        def dumps(value: Any) -> str:
            if serialization_version == "compact-v2":
                return json.dumps(
                    value,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
            return json.dumps(value)

        shared_section = (
            f"Shared model-request context:\n{dumps(shared_context)}\n\n"
            if shared_context
            else ""
        )
        prompt = (
            "Below are two execution traces of the same task: a clean reference execution and an observed execution in which a fault may have occurred.\n\n"
            "Compare the traces step by step and identify the first node where the observed execution diverges meaningfully from the clean reference.\n\n"
            "Return only one exact node_id.\n\n"
            f"Trace serialization: {serialization_version}.{serialization_note}\n"
            f"{shared_section}"
            f"Clean reference trace:\n{dumps(reference)}\n\n"
            f"Observed trace:\n{dumps(observed)}\n\n"
            f"Candidate node_ids: {list(paired_candidates)}"
        )
        observed_candidates = paired_candidates
    elif normalized_mode == "deepseek_single_neutral":
        prompt = (
            "Given this agent execution trace, identify the node most responsible for the final failure. Return only one exact node_id.\n\n"
            f"Observed trace:\n{json.dumps(observed)}\n\n"
            f"Candidate node_ids: {list(observed_candidates)}"
        )
    else:
        prompt = (
            "Given this complete observable agent trace, identify the root-cause node, not merely the node where a downstream symptom became visible. Return only one exact node_id.\n"
            + json.dumps(observed)
            + f"\nCandidate node_ids: {list(observed_candidates)}"
        )
    audit = audit_prompt(
        prompt,
        mode=normalized_mode,
        clean_observable_trace=reference if normalized_mode == "deepseek_paired" else None,
        corrupt_observable_trace=observed,
        shared_observable_context=shared_context,
    )
    return AttributionPrompt(
        mode=normalized_mode,
        serialization_version=serialization_version,
        prompt=prompt,
        prompt_sha256=sha256(prompt.encode("utf-8")).hexdigest(),
        clean_observable_trace=reference if normalized_mode == "deepseek_paired" else None,
        corrupt_observable_trace=observed,
        shared_observable_context=shared_context,
        clean_trace_sha256=(
            _trace_sha256(
                {"shared": shared_context, "trace": reference}
                if shared_context is not None
                else reference
            )
            if normalized_mode == "deepseek_paired"
            else None
        ),
        corrupt_trace_sha256=_trace_sha256(
            {"shared": shared_context, "trace": observed}
            if shared_context is not None
            else observed
        ),
        candidates=observed_candidates,
        leaked_terms=audit["leaked_terms"],
        privileged_metadata_present=audit["privileged_metadata_present"],
        clean_trace_present=audit["clean_trace_present"],
        corrupt_trace_present=audit["corrupt_trace_present"],
    )


def serialize_trace_for_attribution(
    trace: RunTrace | None,
    *,
    serialization_version: TraceSerialization = "full-v1",
) -> list[dict[str, Any]]:
    """Serialize observable events without evaluator-only metadata."""
    if trace is None:
        return []
    hidden = {
        EventKind.NODE_START,
        EventKind.NODE_END,
        EventKind.FAULT_INJECTED,
        EventKind.INTERVENTION,
    }
    forbidden_payload_keys = {"source_fault_id", "ground_truth", "root_cause", "visible_failure", "injection_node"}
    full = [
        {
            "node_id": event.node_id,
            "event_kind": getattr(event.kind, "value", event.kind),
            "payload": {
                key: value for key, value in event.payload.items()
                if key not in forbidden_payload_keys
            },
        }
        for event in trace.events
        if event.kind not in hidden
    ]
    if serialization_version == "full-v1":
        return full
    if serialization_version in {"compact-v1", "compact-v2"}:
        return compact_observable_trace(full)
    raise ValueError(f"unknown trace serialization: {serialization_version}")


def compact_observable_trace(
    full_trace: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Remove cumulative model-request repetition without losing information."""
    compact: list[dict[str, Any]] = []
    previous_static: dict[str, Any] = {}
    previous_messages: list[Any] = []
    static_keys = ("runtime", "system_message", "available_tools")
    for event in full_trace:
        if event["event_kind"] != "model_request":
            compact.append(event)
            continue
        payload = event["payload"]
        encoded: dict[str, Any] = {
            key: value
            for key, value in payload.items()
            if key not in {*static_keys, "messages"}
        }
        inherited = []
        for key in static_keys:
            if key not in payload:
                continue
            if key in previous_static and payload[key] == previous_static[key]:
                inherited.append(key)
            else:
                encoded[key] = payload[key]
                previous_static[key] = payload[key]
        messages = list(payload.get("messages") or [])
        if (
            previous_messages
            and messages[: len(previous_messages)] == previous_messages
        ):
            encoded["messages_delta"] = messages[len(previous_messages) :]
            encoded["messages_total"] = len(messages)
        else:
            encoded["messages"] = messages
        previous_messages = messages
        if inherited:
            encoded["inherits"] = inherited
        compact.append({**event, "payload": encoded})
    return compact


def expand_compact_observable_trace(
    compact_trace: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Reconstruct the full-v1 observable trace for losslessness tests."""
    expanded: list[dict[str, Any]] = []
    previous_static: dict[str, Any] = {}
    previous_messages: list[Any] = []
    static_keys = ("runtime", "system_message", "available_tools")
    for event in compact_trace:
        if event["event_kind"] != "model_request":
            expanded.append(event)
            continue
        encoded = event["payload"]
        payload = {
            key: value
            for key, value in encoded.items()
            if key not in {"inherits", "messages_delta", "messages_total"}
        }
        for key in encoded.get("inherits", []):
            payload[key] = previous_static[key]
        for key in static_keys:
            if key in payload:
                previous_static[key] = payload[key]
        if "messages_delta" in encoded:
            messages = [*previous_messages, *encoded["messages_delta"]]
            if len(messages) != encoded["messages_total"]:
                raise ValueError("compact messages_total does not match delta")
            payload["messages"] = messages
        previous_messages = list(payload.get("messages") or [])
        expanded.append({**event, "payload": payload})
    return expanded


def compact_paired_shared_context(
    clean_trace: list[dict[str, Any]],
    observed_trace: list[dict[str, Any]],
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Store identical first-request context once across a paired prompt."""
    clean_first = _first_model_request(clean_trace)
    observed_first = _first_model_request(observed_trace)
    shared = {
        key: clean_first["payload"][key]
        for key in ("runtime", "system_message", "available_tools")
        if key in clean_first["payload"]
        and clean_first["payload"].get(key)
        == observed_first["payload"].get(key)
    }
    return (
        shared,
        _replace_first_request_with_shared(clean_trace, shared),
        _replace_first_request_with_shared(observed_trace, shared),
    )


def expand_paired_shared_context(
    shared: dict[str, Any],
    trace: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Restore compact-v1 form before expanding a compact-v2 trace."""
    restored = []
    restored_first = False
    for event in trace:
        if event["event_kind"] != "model_request" or restored_first:
            restored.append(event)
            continue
        payload = dict(event["payload"])
        inherited = payload.pop("inherits_shared", [])
        for key in inherited:
            payload[key] = shared[key]
        restored.append({**event, "payload": payload})
        restored_first = True
    return expand_compact_observable_trace(restored)


def _first_model_request(
    trace: list[dict[str, Any]],
) -> dict[str, Any]:
    try:
        return next(
            event
            for event in trace
            if event["event_kind"] == "model_request"
        )
    except StopIteration as exc:
        raise ValueError("compact-v2 requires a model_request event") from exc


def _replace_first_request_with_shared(
    trace: list[dict[str, Any]],
    shared: dict[str, Any],
) -> list[dict[str, Any]]:
    replaced = []
    replaced_first = False
    for event in trace:
        if event["event_kind"] != "model_request" or replaced_first:
            replaced.append(event)
            continue
        payload = {
            key: value
            for key, value in event["payload"].items()
            if key not in shared
        }
        if shared:
            payload["inherits_shared"] = list(shared)
        replaced.append({**event, "payload": payload})
        replaced_first = True
    return replaced


def audit_prompt(
    prompt: str,
    *,
    mode: AttributionMode,
    clean_observable_trace: list[dict[str, Any]] | None,
    corrupt_observable_trace: list[dict[str, Any]],
    shared_observable_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return leakage flags before any external API call."""
    serialized = json.dumps(
        {
            "shared": shared_observable_context,
            "clean": clean_observable_trace,
            "observed": corrupt_observable_trace,
        }
    ).casefold()
    privileged_terms = (
        "fault_injected",
        "fault_label",
        "source_fault_id",
        "ground_truth",
        "root_cause",
        "visible_failure",
        "injection_node",
        "share caused",
        "caused by share",
    )
    leaked = {term for term in privileged_terms if term in serialized}
    prompt_lower = prompt.casefold()
    if mode != "deepseek_paired":
        for term in ("clean reference trace:", "clean run", "corrupted run", "first divergence", "injected fault"):
            if term in prompt_lower:
                leaked.add(term)
    if mode == "deepseek_single_neutral":
        for term in ("root cause", "root-cause", "downstream symptom"):
            if term in prompt_lower:
                leaked.add(term)
    return {
        "leaked_terms": tuple(sorted(leaked)),
        "privileged_metadata_present": any(term in serialized for term in privileged_terms),
        "clean_trace_present": clean_observable_trace is not None,
        "corrupt_trace_present": bool(corrupt_observable_trace),
    }


def localization_accuracy(predictions: list[str | None], ground_truth: list[str | None]) -> float:
    if len(predictions) != len(ground_truth):
        raise ValueError("predictions and ground_truth must have the same length")
    return sum(prediction == truth for prediction, truth in zip(predictions, ground_truth)) / max(1, len(ground_truth))


def parse_attribution_node(raw: str, candidates: tuple[str, ...]) -> str | None:
    """Public shared parser for auditable study-specific attribution prompts."""
    return _parse_node(raw, candidates)


def _provider_error_message(body: str) -> str:
    """Return a bounded provider error without exposing request credentials."""
    try:
        payload = json.loads(body)
        error = payload.get("error", payload)
        if isinstance(error, dict):
            message = error.get("message") or error.get("type") or error
        else:
            message = error
    except (json.JSONDecodeError, AttributeError):
        message = body
    normalized = " ".join(str(message).split())
    return normalized[:500] or "empty provider response"


def _trace_sha256(trace: Any | None) -> str | None:
    if trace is None:
        return None
    encoded = json.dumps(trace, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return sha256(encoded).hexdigest()


def _read_env_file(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    values: dict[str, str] = {}
    with open(path, encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _environment_values(env_file: str | None) -> dict[str, str]:
    """Read the explicit file, or Cascad's local .env when it exists."""
    selected = env_file
    if selected is None and os.path.isfile(".env"):
        selected = ".env"
    return _read_env_file(selected)


def _parse_node(raw: str, candidates: tuple[str, ...]) -> str | None:
    """Use identical parsing for every attribution mode."""
    normalized = raw.strip().strip("`*# ").casefold()
    exact = {candidate.casefold(): candidate for candidate in candidates}
    if normalized in exact:
        return exact[normalized]
    tokens = re.findall(r"[a-zA-Z0-9_-]+", normalized)
    matches = {exact[token] for token in tokens if token in exact}
    return next(iter(matches)) if len(matches) == 1 else None
