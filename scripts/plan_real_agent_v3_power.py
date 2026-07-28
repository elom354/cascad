#!/usr/bin/env python3
"""Create an auditable, assumption-explicit V3 sample-size plan."""

from __future__ import annotations

import argparse
import json

from cascad.study_design import (
    balanced_target,
    paired_mcnemar_normal_approx_n,
    worst_case_proportion_n,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p-cascad-only-correct", type=float, default=0.08)
    parser.add_argument("--p-baseline-only-correct", type=float, default=0.03)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--power", type=float, default=0.90)
    parser.add_argument("--precision-half-width", type=float, default=0.04)
    parser.add_argument("--task-domains", type=int, default=3)
    parser.add_argument("--agent-configurations", type=int, default=2)
    parser.add_argument("--root-roles", type=int, default=3)
    parser.add_argument("--observability-conditions", type=int, default=3)
    args = parser.parse_args()

    strata = (
        args.task_domains
        * args.agent_configurations
        * args.root_roles
        * args.observability_conditions
    )
    test_n = paired_mcnemar_normal_approx_n(
        p_method_a_only_correct=args.p_cascad_only_correct,
        p_method_b_only_correct=args.p_baseline_only_correct,
        alpha=args.alpha,
        power=args.power,
    )
    precision_n = worst_case_proportion_n(
        half_width=args.precision_half_width,
        confidence=1 - args.alpha,
    )
    target, per_stratum = balanced_target(
        max(test_n, precision_n),
        strata=strata,
    )
    print(
        json.dumps(
            {
                "role": "prospective_sample_size_plan",
                "method": (
                    "normal-approximation planning for two-sided paired "
                    "McNemar; exact McNemar reserved for final inference"
                ),
                "assumptions": {
                    "p_cascad_only_correct": args.p_cascad_only_correct,
                    "p_baseline_only_correct": args.p_baseline_only_correct,
                    "alpha_two_sided": args.alpha,
                    "target_power": args.power,
                    "worst_case_accuracy_ci_half_width": (
                        args.precision_half_width
                    ),
                },
                "design_axes": {
                    "task_domains": args.task_domains,
                    "agent_configurations": args.agent_configurations,
                    "root_roles": args.root_roles,
                    "observability_conditions": (
                        args.observability_conditions
                    ),
                    "strata": strata,
                },
                "minimum_for_paired_test": test_n,
                "minimum_for_overall_precision": precision_n,
                "balanced_confirmatory_pairs": target,
                "pairs_per_stratum": per_stratum,
                "required_agent_executions": 2 * target,
                "warning": (
                    "Repeated calls on one trace are not independent units. "
                    "Every pair must use a distinct preregistered task instance."
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
