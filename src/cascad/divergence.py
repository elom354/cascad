"""Counterfactual divergence primitives used by Cascad's causal reconstruction.

The module deliberately has no mandatory model dependency.  When
``sentence-transformers`` is installed it is used for textual values; otherwise
a deterministic hashed bag-of-words embedding preserves offline reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from hashlib import blake2b
from math import sqrt
from statistics import mean, stdev
from typing import Any, Callable, Iterable

from cascad.models import NodeEvent, RunTrace


_NATURAL_CACHE: dict[tuple[str, str | None, int], dict[str, "DivergenceDistribution"]] = {}
_EMBEDDING_MODEL: Any | None = None
_ENCODER_PROBED = False
_ENCODER_STATUS: dict[str, str] = {"encoder_used": "not_applicable", "reason": "no textual distance computed"}


@dataclass(frozen=True)
class DivergenceDistribution:
    """Natural variation observed at one node type."""

    mean: float
    stddev: float
    samples: tuple[float, ...] = ()


def event_distance(left: NodeEvent, right: NodeEvent) -> float:
    """Return a normalized distance between corresponding event payloads."""
    return value_distance(_event_value(left), _event_value(right))


def value_distance(left: Any, right: Any) -> float:
    """Distance for scalar, text and structured agent values, in ``[0, 1]``."""
    if left == right:
        return 0.0
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        scale = max(abs(float(left)), abs(float(right)), 1.0)
        return min(1.0, abs(float(left) - float(right)) / scale)
    if isinstance(left, str) and isinstance(right, str):
        return _text_distance(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return _mapping_distance(left, right)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return _sequence_distance(left, right)
    return 1.0


def encoder_status() -> dict[str, str]:
    """Return the encoder provenance of the most recent textual distance."""
    return dict(_ENCODER_STATUS)


def probe_embedding_encoder() -> tuple[Any | None, dict[str, str]]:
    """Attempt one local MiniLM load and retain an explicit success/failure reason."""
    global _EMBEDDING_MODEL, _ENCODER_PROBED, _ENCODER_STATUS
    if _ENCODER_PROBED:
        return _EMBEDDING_MODEL, encoder_status()
    _ENCODER_PROBED = True
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        _EMBEDDING_MODEL = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            local_files_only=True,
        )
        _ENCODER_STATUS = {"encoder_used": "embedding", "reason": "all-MiniLM-L6-v2 loaded"}
    except Exception as exc:  # dependency, offline cache, and model load failures are all audit-relevant
        _EMBEDDING_MODEL = None
        _ENCODER_STATUS = {"encoder_used": "fallback", "reason": f"{exc.__class__.__name__}: {exc}"}
    return _EMBEDDING_MODEL, encoder_status()


def fallback_text_distance(left: str, right: str) -> float:
    """Deterministic lexical fallback, publicly exposed for audit/testing."""
    dimensions = 128
    a = [0.0] * dimensions
    b = [0.0] * dimensions
    for token in left.lower().split():
        a[_stable_bucket(token, dimensions)] += 1.0
    for token in right.lower().split():
        b[_stable_bucket(token, dimensions)] += 1.0
    denom = sqrt(sum(item * item for item in a)) * sqrt(sum(item * item for item in b))
    return 1.0 if denom == 0 else max(0.0, 1.0 - sum(x * y for x, y in zip(a, b)) / denom)


def text_distance(left: str, right: str) -> float:
    """Public text-distance API using MiniLM when available, fallback otherwise."""
    return _text_distance(left, right)


def estimate_natural_divergence(
    scenario: Callable[..., Any], node_type: str | None = None, M: int = 8
) -> dict[str, DivergenceDistribution]:
    """Estimate clean/clean divergence from ``M`` independently seeded pairs.

    ``scenario`` may return a :class:`RunTrace` or an object exposing ``trace``
    (such as ``SimulationResult``).  It is called with ``seed=`` when supported.
    ``node_type`` optionally restricts the returned profile to a single node id.
    """
    if M < 1:
        raise ValueError("M must be at least 1")
    scenario_key = getattr(scenario, "cache_key", None) or getattr(scenario, "__qualname__", repr(scenario))
    cache_key = (str(scenario_key), node_type, M)
    if cache_key in _NATURAL_CACHE:
        return _NATURAL_CACHE[cache_key]
    values: dict[str, list[float]] = {}
    for index in range(M):
        first = _run_scenario(scenario, 2 * index)
        second = _run_scenario(scenario, 2 * index + 1)
        for key, (left, right) in corresponding_events(first, second).items():
            if node_type is None or key == node_type:
                values.setdefault(key, []).append(event_distance(left, right))
    result = {
        key: DivergenceDistribution(
            mean=mean(samples),
            stddev=stdev(samples) if len(samples) > 1 else 0.0,
            samples=tuple(samples),
        )
        for key, samples in values.items()
    }
    _NATURAL_CACHE[cache_key] = result
    return result


def corresponding_events(
    left: RunTrace, right: RunTrace
) -> dict[str, tuple[NodeEvent, NodeEvent]]:
    """Match the semantically meaningful terminal event for each node.

    Start/end span bookkeeping is intentionally ignored: it has no user-state
    content and would create a spurious zero-distance signal.
    """
    left_by_node = _meaningful_events(left.events)
    right_by_node = _meaningful_events(right.events)
    return {
        node: (left_by_node[node], right_by_node[node])
        for node in left_by_node.keys() & right_by_node.keys()
    }


def _meaningful_events(events: Iterable[NodeEvent]) -> dict[str, NodeEvent]:
    ignored = {"node_start", "node_end", "fault_injected"}
    result: dict[str, NodeEvent] = {}
    for event in events:
        if str(event.kind) not in ignored and getattr(event.kind, "value", event.kind) not in ignored:
            result[event.node_id] = event
    return result


def _run_scenario(scenario: Callable[..., Any], seed: int) -> RunTrace:
    try:
        result = scenario(seed=seed)
    except TypeError:
        result = scenario()
    return result.trace if hasattr(result, "trace") else result


def _event_value(event: NodeEvent) -> Any:
    payload = event.payload
    # The payload itself is the canonical trace contract.  These keys merely
    # remove tracing-only fields that differ across otherwise equal trajectories.
    return {key: value for key, value in payload.items() if key not in {"started_by"}}


def _mapping_distance(left: dict[str, Any], right: dict[str, Any]) -> float:
    keys = set(left) | set(right)
    if not keys:
        return 0.0
    distances = [0.0 if key in left and key in right else 1.0 for key in keys]
    for key in set(left) & set(right):
        distances.append(value_distance(left[key], right[key]))
    return min(1.0, sum(distances) / len(distances))


def _sequence_distance(left: list[Any] | tuple[Any, ...], right: list[Any] | tuple[Any, ...]) -> float:
    if not left and not right:
        return 0.0
    longest = max(len(left), len(right), 1)
    shared = sum(value_distance(a, b) for a, b in zip(left, right))
    missing = abs(len(left) - len(right))
    return min(1.0, (shared + missing) / longest)


@lru_cache(maxsize=16_384)
def _text_distance(left: str, right: str) -> float:
    # Optional local semantic encoder. It is intentionally lazy so the base
    # package remains installable in offline experimental environments.
    model, _ = probe_embedding_encoder()
    if model is not None:
        try:
            vectors = model.encode([left, right], normalize_embeddings=True)
            similarity = sum(float(a) * float(b) for a, b in zip(vectors[0], vectors[1]))
            distance = max(0.0, min(1.0, 1.0 - similarity))
            # Sentence embeddings frequently under-represent logical negation
            # ("eligible" / "not eligible"). A polarity mismatch is therefore
            # a semantic contradiction, never a near-zero divergence.
            if _has_polarity_mismatch(left, right):
                return max(distance, 0.75)
            return distance
        except Exception as exc:
            # Runtime encoder failures retain provenance instead of silently
            # looking like semantic distances.
            _ENCODER_STATUS.update({"encoder_used": "fallback", "reason": f"runtime {exc.__class__.__name__}: {exc}"})
    return fallback_text_distance(left, right)


def _stable_bucket(token: str, dimensions: int) -> int:
    return int.from_bytes(blake2b(token.encode("utf-8"), digest_size=4).digest(), "big") % dimensions


def _has_polarity_mismatch(left: str, right: str) -> bool:
    negations = {"not", "no", "never", "none", "without", "cannot", "can't", "failed"}
    left_tokens = set(left.lower().replace("n't", " not").split())
    right_tokens = set(right.lower().replace("n't", " not").split())
    return bool(left_tokens & negations) != bool(right_tokens & negations)
