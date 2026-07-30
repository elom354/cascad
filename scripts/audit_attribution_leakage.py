"""Pre-API leakage and uniqueness audit for attribution-ablation prompts."""

from __future__ import annotations

import argparse
from pathlib import Path

from cascad.attribution_baseline import build_attribution_prompt
from cascad.export import write_json
from cascad.injection import FaultInjector
from cascad.scenarios import ATTRIBUTION_SCENARIOS, attribution_fault
from cascad.simulator import ReActPropagationSimulator


FORBIDDEN = {
    "support_neutral": {"refund_api", "refund service", "refund endpoint"},
    "document_neutral": {"generate_report", "report generator", "generation tool"},
    "cloud_neutral": {"share", "sharing tool", "share service"},
    "cloud_distant_symptom": {"share caused", "caused by share", "fault at share"},
    "cloud_distant_symptom_natural_noise": {
        "share caused",
        "caused by share",
        "fault at share",
    },
}
MODES = ("single-neutral", "single-guided", "paired")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", type=int, default=20)
    parser.add_argument("--out", default="runs/paired-attribution-study")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    scenario_checks: list[dict] = []
    prompt_checks: list[dict] = []

    for scenario in ATTRIBUTION_SCENARIOS:
        fault = attribution_fault(scenario)
        message = str(fault.payload["value"].get("error", "")).casefold()
        leaked = sorted(term for term in FORBIDDEN[scenario] if term in message)
        status = "PASS" if not leaked else "FAIL"
        check = {
            "scenario": scenario,
            "status": status,
            "target": fault.target_node,
            "fault_message": message,
            "leaked_terms": leaked,
        }
        scenario_checks.append(check)
        print(f"{scenario}: {status} message_leaks={leaked}")
        if leaked:
            failures.append(f"{scenario}:fault-message")

    scenario = "cloud_distant_symptom"
    for seed in range(args.instances):
        clean = ReActPropagationSimulator(scenario=scenario).run(seed=seed).trace
        corrupt = ReActPropagationSimulator(
            FaultInjector([attribution_fault(scenario, seed)]), scenario=scenario
        ).run(seed=seed).trace
        for mode in MODES:
            bundle = build_attribution_prompt(corrupt, mode=mode, clean_trace=clean)
            check = {
                "scenario": scenario,
                "instance_seed": seed,
                "attribution_mode": bundle.mode,
                "prompt_sha256": bundle.prompt_sha256,
                "clean_trace_sha256": bundle.clean_trace_sha256,
                "corrupt_trace_sha256": bundle.corrupt_trace_sha256,
                "leaked_terms": list(bundle.leaked_terms),
                "privileged_metadata_present": bundle.privileged_metadata_present,
                "clean_trace_present": bundle.clean_trace_present,
                "corrupt_trace_present": bundle.corrupt_trace_present,
            }
            prompt_checks.append(check)
            if bundle.leaked_terms or bundle.privileged_metadata_present:
                failures.append(f"{scenario}:{seed}:{mode}")

    hash_audit = []
    for mode in MODES:
        selected = [item for item in prompt_checks if item["attribution_mode"].endswith(mode.replace("-", "_"))]
        prompt_count = len({item["prompt_sha256"] for item in selected})
        trace_count = len({item["corrupt_trace_sha256"] for item in selected})
        passed = prompt_count == args.instances and trace_count == args.instances
        hash_audit.append({
            "attribution_mode": selected[0]["attribution_mode"],
            "number_of_instances": args.instances,
            "unique_prompt_count": prompt_count,
            "unique_trace_count": trace_count,
            "status": "PASS" if passed else "FAIL",
        })
        print(f"{selected[0]['attribution_mode']}: prompts={prompt_count} traces={trace_count} status={'PASS' if passed else 'FAIL'}")
        if not passed:
            failures.append(f"uniqueness:{mode}")

    write_json(out / "leakage_audit.json", {
        "scenario_fault_messages": scenario_checks,
        "prompt_checks": prompt_checks,
        "status": "PASS" if not failures else "FAIL",
    })
    write_json(out / "prompt_trace_hash_audit.json", hash_audit)
    if failures:
        raise SystemExit(f"leakage/uniqueness audit failed: {', '.join(failures)}")


if __name__ == "__main__":
    main()
