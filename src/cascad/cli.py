"""Cascad command line interface."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from cascad.causal import CausalGraph
from cascad.attribution_baseline import DeepSeekAttributor
from cascad.divergence import estimate_natural_divergence
from cascad.export import write_csv, write_trace_bundle
from cascad.injection import FaultInjector
from cascad.metrics import compute_metrics
from cascad.experiment import ExperimentCase, run_experiment
from cascad.export import write_json
from cascad.simulator import ReActPropagationSimulator, default_fault
from cascad.scenarios import ATTRIBUTION_SCENARIOS, SCENARIOS, attribution_fault


def main() -> None:
    """Run the CLI."""
    parser = argparse.ArgumentParser(prog="cascad")
    subparsers = parser.add_subparsers(dest="command", required=True)

    simulate = subparsers.add_parser("simulate", help="Run a reference simulation")
    simulate.add_argument("--fault-kind", default="tool_misresult")
    simulate.add_argument("--target-node", default="tool")
    simulate.add_argument("--task", default=None)
    simulate.add_argument("--scenario", choices=sorted(SCENARIOS), default="weather")
    simulate.add_argument("--method", choices=["counterfactual", "temporal"], default="counterfactual")
    simulate.add_argument("--out", default="runs/latest")

    view = subparsers.add_parser("view", help="Open the Cascad visual viewer")
    view.add_argument("--runs-dir", default="runs")
    view.add_argument("--host", default="127.0.0.1")
    view.add_argument("--port", type=int, default=8765)

    experiment = subparsers.add_parser("experiment", help="Run repeated comparable experimental conditions")
    experiment.add_argument("--scenario", choices=sorted(SCENARIOS), default="weather")
    experiment.add_argument("--method", choices=["counterfactual", "temporal", "raw_logging", "attribution"], default="counterfactual")
    experiment.add_argument("--n-repeats", type=int, default=5)
    experiment.add_argument("--seed-start", type=int, default=0)
    experiment.add_argument("--append", action="store_true")
    experiment.add_argument("--attribution", choices=["none", "deepseek"], default="none")
    experiment.add_argument(
        "--attribution-mode",
        choices=["single-neutral", "single-guided", "paired"],
        default="single-guided",
    )
    experiment.add_argument("--fixed-instance-seed", type=int, default=None)
    experiment.add_argument("--require-unique-instances", action="store_true")
    experiment.add_argument("--env-file", default=None)
    experiment.add_argument("--out", default=None)

    natural_noise = subparsers.add_parser(
        "natural-noise-study",
        help="Run the held-out controlled natural-divergence robustness pilot",
    )
    natural_noise.add_argument("--instances-per-level", type=int, default=20)
    natural_noise.add_argument("--calibration-pairs", type=int, default=24)
    natural_noise.add_argument("--epsilon", type=float, default=0.05)
    natural_noise.add_argument("--deepseek", action="store_true")
    natural_noise.add_argument("--env-file", default=None)
    natural_noise.add_argument("--out", default="runs/natural-noise-study")

    natural_noise_api = subparsers.add_parser(
        "natural-noise-deepseek",
        help="Complete the frozen natural-noise pilot with paired DeepSeek baselines",
    )
    natural_noise_api.add_argument(
        "--frozen-dir", default="runs/natural-noise-study"
    )
    natural_noise_api.add_argument(
        "--out", default="runs/natural-noise-study-deepseek"
    )
    natural_noise_api.add_argument("--env-file", required=True)

    branched = subparsers.add_parser(
        "branched-dependency-study",
        help="Run the native dependency-constraint structural ablation",
    )
    branched.add_argument("--instances-per-level", type=int, default=20)
    branched.add_argument("--calibration-pairs", type=int, default=24)
    branched.add_argument("--epsilon", type=float, default=0.05)
    branched.add_argument("--out", default="runs/branched-dependency-study")

    inter_episode = subparsers.add_parser(
        "inter-episode-study",
        help="Run the final simulator inter-episode persistence protocol",
    )
    inter_episode.add_argument(
        "--out", default="runs/inter-episode-persistence-study"
    )
    inter_episode.add_argument("--deepseek", action="store_true")
    inter_episode.add_argument("--env-file", default=None)

    compare_llms = subparsers.add_parser(
        "compare-attribution",
        help=(
            "Run matched DeepSeek/OpenAI attribution baselines; OpenAI is "
            "included automatically when OPENAI_API_KEY is configured"
        ),
    )
    compare_llms.add_argument("--env-file", default=".env")
    compare_llms.add_argument("--n-repeats", type=int, default=20)
    compare_llms.add_argument("--seed-start", type=int, default=0)
    compare_llms.add_argument(
        "--attribution-mode",
        choices=["single-neutral", "single-guided", "paired"],
        default="paired",
    )
    compare_llms.add_argument(
        "--scenarios",
        nargs="+",
        default=None,
    )
    compare_llms.add_argument(
        "--out",
        default="runs/llm-attribution-comparison",
    )

    args = parser.parse_args()
    if args.command == "simulate":
        fault = attribution_fault(args.scenario) if args.scenario in ATTRIBUTION_SCENARIOS else default_fault(kind=args.fault_kind, target_node=args.target_node)
        simulator = ReActPropagationSimulator(FaultInjector([fault]), scenario=args.scenario)
        result = simulator.run(args.task)
        clean = ReActPropagationSimulator(scenario=args.scenario).run(args.task)
        natural = estimate_natural_divergence(
            lambda seed=0: ReActPropagationSimulator(scenario=args.scenario).run(args.task, seed=seed), M=4
        ) if args.method == "counterfactual" else {}
        metrics = compute_metrics(
            result.trace, clean_trace=clean.trace if args.method == "counterfactual" else None,
            natural_divergence=natural, construction_method=args.method,
        )
        graph = CausalGraph.from_trace(
            result.trace, clean_trace=clean.trace if args.method == "counterfactual" else None,
            natural_divergence=natural, construction_method=args.method,
        )
        write_trace_bundle(Path(args.out), result.trace, metrics, graph=graph)
        print(f"run_id={result.trace.run_id}")
        print(f"final_answer={result.final_answer}")
        print(f"metrics={metrics}")
        print(f"wrote={args.out}")
    elif args.command == "view":
        from cascad.viewer import run_viewer

        run_viewer(args.runs_dir, host=args.host, port=args.port)
    elif args.command == "experiment":
        targets = {"weather": "tool", "support": "refund_api", "document": "generate_report", "cloud": "share"}
        attributor = DeepSeekAttributor.from_environment(args.env_file) if args.attribution == "deepseek" else None
        method = "attribution" if args.attribution == "deepseek" else args.method
        fault = attribution_fault(args.scenario) if args.scenario in ATTRIBUTION_SCENARIOS else default_fault(target_node=targets[args.scenario])
        fault_factory = (
            (lambda seed: [attribution_fault(args.scenario, seed)])
            if args.scenario in ATTRIBUTION_SCENARIOS else None
        )
        out = Path(args.out or f"runs/{args.scenario}-{method}-experiment")
        out.mkdir(parents=True, exist_ok=True)
        raw_path = out / "attribution_raw.json"
        checkpoint_records: list[dict] = []
        if args.append and raw_path.exists():
            checkpoint_records = json.loads(raw_path.read_text(encoding="utf-8")).get("runs", [])

        def checkpoint(record: dict) -> None:
            checkpoint_records[:] = [item for item in checkpoint_records if item.get("seed") != record.get("seed")]
            checkpoint_records.append(record)
            checkpoint_records.sort(key=lambda item: item["seed"])
            write_json(raw_path, {"scenario": args.scenario, "runs": checkpoint_records})
            print(f"attribution_records={len(checkpoint_records)} latest_seed={record['seed']}", flush=True)

        summary = run_experiment([ExperimentCase(
            name=f"{args.scenario}_{method}", scenario=args.scenario, method=method,
            faults=[fault], fault_factory=fault_factory,
            trials=args.n_repeats, seed_start=args.seed_start,
            attribution_llm=attributor,
            attribution_record_callback=checkpoint if attributor else None,
            attribution_mode=args.attribution_mode,
            fixed_instance_seed=args.fixed_instance_seed,
            require_unique_instances=args.require_unique_instances,
        )])[0]
        if checkpoint_records:
            categories = [item["prediction_type"] for item in checkpoint_records]
            distances = [item["graph_distance_to_root"] for item in checkpoint_records if item["graph_distance_to_root"] is not None]
            summary = replace(
                summary,
                trials=len(checkpoint_records),
                localization_accuracy=categories.count("root") / len(checkpoint_records),
                graph_localization_accuracy=sum(item["cascad_prediction_type"] == "root" for item in checkpoint_records) / len(checkpoint_records),
                root_accuracy=categories.count("root") / len(checkpoint_records),
                mediator_selection_rate=categories.count("mediator") / len(checkpoint_records),
                visible_symptom_selection_rate=categories.count("symptom") / len(checkpoint_records),
                other_node_rate=categories.count("other") / len(checkpoint_records),
                invalid_output_rate=categories.count("invalid") / len(checkpoint_records),
                mean_root_distance=sum(distances) / len(distances) if distances else None,
                unique_prompt_count=len({item["prompt_sha256"] for item in checkpoint_records}),
                unique_trace_count=len({item["corrupt_trace_sha256"] for item in checkpoint_records}),
                api_call_count=len(checkpoint_records),
                attribution_records=checkpoint_records,
            )
        if summary.attribution_records:
            write_json(out / "attribution_raw.json", {"scenario": args.scenario, "runs": summary.attribution_records})
            confusion = [
                {
                    "scenario": record["scenario"],
                    "seed": record["seed"],
                    "ground_truth_root_cause": record["ground_truth_root"],
                    "visible_symptom": record["visible_symptom"],
                    "deepseek_prediction": record["deepseek_parsed_node"],
                    "cascad_prediction": record["cascad_prediction"],
                    "deepseek_prediction_type": record["prediction_type"],
                    "cascad_prediction_type": record["cascad_prediction_type"],
                }
                for record in summary.attribution_records
            ]
            write_json(out / "confusion.json", confusion)
            write_csv(out / "confusion.csv", confusion)
        exported_summary = replace(summary, attribution_records=None)
        write_json(out / "metrics.json", exported_summary)
        write_json(out / "summary.json", exported_summary)
        write_csv(out / "summary.csv", [{
            "scenario": args.scenario,
            "trials": summary.trials,
            "deepseek_accuracy": summary.localization_accuracy,
            "cascad_accuracy": summary.graph_localization_accuracy,
            "mediator_rate": summary.mediator_selection_rate,
            "symptom_rate": summary.visible_symptom_selection_rate,
            "invalid_rate": summary.invalid_output_rate,
            "mean_root_distance": summary.mean_root_distance,
            "unique_prompt_count": summary.unique_prompt_count,
            "unique_trace_count": summary.unique_trace_count,
            "api_call_count": summary.api_call_count,
            "temperature": summary.temperature,
            "model": summary.model,
            "encoder_used": summary.encoder_used,
        }])
        print(f"wrote={out / 'metrics.json'}")
        print(f"summary={summary}")
    elif args.command == "natural-noise-study":
        from cascad.natural_noise import run_natural_noise_study

        attributor = (
            DeepSeekAttributor.from_environment(args.env_file)
            if args.deepseek else None
        )
        result = run_natural_noise_study(
            args.out,
            instances_per_level=args.instances_per_level,
            calibration_pairs=args.calibration_pairs,
            epsilon=args.epsilon,
            attributor=attributor,
        )
        print(f"wrote={args.out}")
        print(f"study={result}")
    elif args.command == "natural-noise-deepseek":
        from cascad.natural_noise_api import complete_frozen_deepseek_baselines

        attributor = DeepSeekAttributor.from_environment(args.env_file)
        result = complete_frozen_deepseek_baselines(
            frozen_dir=args.frozen_dir,
            out_dir=args.out,
            attributor=attributor,
        )
        print(f"wrote={args.out}")
        print(f"study={result}")
    elif args.command == "branched-dependency-study":
        from cascad.branched_dependency import run_branched_dependency_study

        result = run_branched_dependency_study(
            args.out,
            instances_per_level=args.instances_per_level,
            calibration_pairs=args.calibration_pairs,
            epsilon=args.epsilon,
        )
        print(f"wrote={args.out}")
        print(f"study={result}")
    elif args.command == "inter-episode-study":
        from cascad.inter_episode import run_inter_episode_study

        attributor = (
            DeepSeekAttributor.from_environment(args.env_file)
            if args.deepseek else None
        )
        result = run_inter_episode_study(args.out, attributor=attributor)
        print(f"wrote={args.out}")
        print(f"study={result}")
    elif args.command == "compare-attribution":
        from cascad.llm_comparison import (
            DEFAULT_COMPARISON_SCENARIOS,
            run_llm_attribution_comparison,
        )

        result = run_llm_attribution_comparison(
            args.out,
            env_file=args.env_file,
            scenarios=args.scenarios or DEFAULT_COMPARISON_SCENARIOS,
            n_repeats=args.n_repeats,
            seed_start=args.seed_start,
            attribution_mode=args.attribution_mode,
        )
        print(f"wrote={args.out}")
        print(f"comparison={result}")


if __name__ == "__main__":
    main()
