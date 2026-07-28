"""Serialization helpers."""

from __future__ import annotations

import json
import csv
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from cascad.causal import CausalGraph
from cascad.models import RunTrace


def to_jsonable(value: Any) -> Any:
    """Convert dataclasses and sets to JSON-compatible values."""
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(value)
    return value


def write_json(path: str | Path, value: Any) -> None:
    """Write JSON file."""
    Path(path).write_text(
        json.dumps(to_jsonable(value), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    """Write a stable UTF-8 CSV for experimental result tables."""
    columns = fieldnames or (list(rows[0]) if rows else [])
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_trace_bundle(
    directory: str | Path, trace: RunTrace, metrics: Any, graph: CausalGraph | None = None
) -> None:
    """Write trace, metrics and causal graph outputs."""
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "trace.json", trace)
    write_json(out / "metrics.json", metrics)
    graph = graph or CausalGraph.from_trace(trace)
    (out / "causal_graph.dot").write_text(
        graph.to_dot(trace.affected_nodes()),
        encoding="utf-8",
    )
