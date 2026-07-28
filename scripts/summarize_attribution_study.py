"""Aggregate difficult-attribution outputs into JSON/CSV study artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from cascad.export import write_csv, write_json


RUN_DIRS = {
    "support_neutral": "support-neutral-attribution",
    "document_neutral": "document-neutral-attribution",
    "cloud_neutral": "cloud-neutral-attribution",
    "cloud_distant_symptom": "cloud-distant-attribution",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--out", default="runs/attribution-study")
    args = parser.parse_args()
    runs_dir, out = Path(args.runs_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []
    confusion: list[dict] = []
    for scenario, directory in RUN_DIRS.items():
        records = json.loads((runs_dir / directory / "attribution_raw.json").read_text(encoding="utf-8"))["runs"]
        deepseek_correct = sum(record["deepseek_prediction_type"] == "root_cause" for record in records)
        cascad_correct = sum(record["graph_prediction_type"] == "root_cause" for record in records)
        summaries.append({
            "scenario": scenario,
            "runs": len(records),
            "root_cause": records[0]["injection_node"],
            "visible_symptom": records[0]["visible_failure_node"],
            "deepseek_accuracy": deepseek_correct / len(records),
            "cascad_accuracy": cascad_correct / len(records),
            "deepseek_distribution": dict(Counter(record["deepseek_predicted_node"] for record in records)),
            "cascad_distribution": dict(Counter(record["graph_predicted_node"] for record in records)),
        })
        confusion.extend({
            "scenario": scenario,
            "seed": record["seed"],
            "ground_truth_root_cause": record["injection_node"],
            "visible_symptom": record["visible_failure_node"],
            "deepseek_prediction": record["deepseek_predicted_node"],
            "cascad_prediction": record["graph_predicted_node"],
            "deepseek_prediction_type": record["deepseek_prediction_type"],
            "cascad_prediction_type": record["graph_prediction_type"],
        } for record in records)
    write_json(out / "summary.json", summaries)
    write_csv(out / "summary.csv", summaries, [
        "scenario", "runs", "root_cause", "visible_symptom",
        "deepseek_accuracy", "cascad_accuracy", "deepseek_distribution", "cascad_distribution",
    ])
    write_json(out / "confusion.json", confusion)
    write_csv(out / "confusion.csv", confusion)
    print(f"wrote={out} scenarios={len(summaries)} runs={len(confusion)}")


if __name__ == "__main__":
    main()
