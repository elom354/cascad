"""Aggregate information-ablation modes and repeated-call stability evidence."""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path

from cascad.export import write_csv, write_json


VARIED = {
    "DeepSeek single-neutral": ("paired-study-single-neutral", "corrupt observable trace", "neutral observational"),
    "DeepSeek single-guided": ("paired-study-single-guided", "corrupt observable trace", "guided root-cause"),
    "DeepSeek paired": ("paired-study-paired", "clean + corrupt observable traces", "paired comparison"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--out", default="runs/paired-attribution-study")
    parser.add_argument("--stability-dir", default="runs/cloud-distant-attribution")
    args = parser.parse_args()
    runs_dir, out = Path(args.runs_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    all_calls: list[dict] = []
    records_by_method: dict[str, list[dict]] = {}

    for method, (directory, information, prompt_type) in VARIED.items():
        records = json.loads((runs_dir / directory / "attribution_raw.json").read_text(encoding="utf-8"))["runs"]
        records_by_method[method] = records
        all_calls.extend(records)
        rows.append(_summary_row(method, information, prompt_type, records))

    paired_records = records_by_method["DeepSeek paired"]
    rows.append({
        "method": "Cascad counterfactual graph",
        "information_available": "paired traces + natural-divergence calibration + native dependencies",
        "prompt_type": "structured algorithm (no LLM prompt)",
        "unique_instances": len({item["instance_id"] for item in paired_records}),
        "unique_prompt_count": 0,
        "api_call_count": 0,
        "root_accuracy": sum(item["cascad_prediction_type"] == "root" for item in paired_records) / len(paired_records),
        "mediator_rate": 0.0,
        "symptom_rate": 0.0,
        "other_rate": 0.0,
        "invalid_rate": 0.0,
        "mean_root_distance": 0.0,
        "model": None,
        "temperature": None,
        "experimental_role": "varied_instances",
    })

    stability_records = json.loads((Path(args.stability_dir) / "attribution_raw.json").read_text(encoding="utf-8"))["runs"]
    enriched_stability = []
    for record in stability_records:
        prompt = record["prompt"]
        prompt_hash = sha256(prompt.encode("utf-8")).hexdigest()
        observable_text = prompt.split("\n", 1)[-1].rsplit("\nCandidate node_ids:", 1)[0]
        trace_hash = sha256(observable_text.encode("utf-8")).hexdigest()
        prediction = record.get("deepseek_predicted_node")
        category = "root" if prediction == "share" else "mediator" if prediction == "memory" else "symptom" if prediction == "notify" else "invalid" if prediction is None else "other"
        root_distance = {"root": 0, "mediator": 1, "symptom": 2}.get(category)
        enriched_stability.append({
            **record,
            "attribution_mode": "deepseek_single_guided",
            "prompt_sha256": prompt_hash,
            "corrupt_trace_sha256": trace_hash,
            "prediction_type": category,
            "graph_distance_to_root": root_distance,
            "experimental_role": "repeated_call_stability",
            "model": "deepseek-chat",
            "temperature": 0,
        })
    rows.append(_summary_row(
        "Repeated-call stability check",
        "one identical corrupt observable trace",
        "guided root-cause, repeated identical input",
        enriched_stability,
        role="repeated_call_stability",
    ))

    paired_correctness = []
    cascad_by_seed = {item["instance_seed"]: item["cascad_prediction_type"] == "root" for item in paired_records}
    for method, records in records_by_method.items():
        deep_by_seed = {item["instance_seed"]: item["prediction_type"] == "root" for item in records}
        seeds = sorted(set(cascad_by_seed) & set(deep_by_seed))
        paired_correctness.append({
            "comparison": f"{method} vs Cascad",
            "paired_instances": len(seeds),
            "both_correct": sum(deep_by_seed[s] and cascad_by_seed[s] for s in seeds),
            "deepseek_only_correct": sum(deep_by_seed[s] and not cascad_by_seed[s] for s in seeds),
            "cascad_only_correct": sum(not deep_by_seed[s] and cascad_by_seed[s] for s in seeds),
            "both_wrong": sum(not deep_by_seed[s] and not cascad_by_seed[s] for s in seeds),
        })

    write_json(out / "summary.json", rows)
    write_csv(out / "summary.csv", rows)
    write_json(out / "paired_correctness.json", paired_correctness)
    write_csv(out / "paired_correctness.csv", paired_correctness)
    write_json(out / "raw_calls.json", all_calls)
    write_json(out / "repeated_stability.json", enriched_stability)
    print(f"wrote={out} varied_calls={len(all_calls)} stability_calls={len(enriched_stability)}")


def _summary_row(
    method: str,
    information: str,
    prompt_type: str,
    records: list[dict],
    role: str = "varied_instances",
) -> dict:
    categories = [item["prediction_type"] for item in records]
    distances = [item.get("graph_distance_to_root") for item in records if item.get("graph_distance_to_root") is not None]
    return {
        "method": method,
        "information_available": information,
        "prompt_type": prompt_type,
        "unique_instances": len({item.get("instance_id", item.get("corrupt_trace_sha256")) for item in records}),
        "unique_prompt_count": len({item["prompt_sha256"] for item in records}),
        "api_call_count": len(records),
        "root_accuracy": categories.count("root") / len(records),
        "mediator_rate": categories.count("mediator") / len(records),
        "symptom_rate": categories.count("symptom") / len(records),
        "other_rate": categories.count("other") / len(records),
        "invalid_rate": categories.count("invalid") / len(records),
        "mean_root_distance": sum(distances) / len(distances) if distances else None,
        "model": records[0].get("model"),
        "temperature": records[0].get("temperature"),
        "experimental_role": role,
    }


if __name__ == "__main__":
    main()
