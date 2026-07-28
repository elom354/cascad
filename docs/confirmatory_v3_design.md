# Prospective real-agent validation V3

## Why V3 is required

The 200-pair V2 collection is useful evidence, but its main weakness is not the
number 200 by itself. It contains one agent-model family, three task families,
two injected root nodes, and only 40 multi-step cases. Task, fault, causal
position, and configuration are not fully crossed. In addition, the final
clock-normalization rule was corrected after controlled outcomes were
inspected. V2 must therefore remain a post-hoc-corrected validation rather than
being renamed as untouched confirmatory evidence.

V3 is a fresh held-out collection. Existing V2 traces may be used for sample
planning and engineering tests, but not for threshold selection or outcome
tuning.

## Prospective sample size

The planning command is:

```bash
PYTHONPATH=src uv run python scripts/plan_real_agent_v3_power.py
```

Default assumptions are declared rather than inferred after execution:

- two-sided alpha: 0.05;
- target power: 0.90;
- discordant probabilities: 0.08 Cascad-only correct and 0.03
  baseline-only correct;
- maximum overall 95% interval half-width under worst-case variance: 0.04.

The paired McNemar approximation requires 459 pairs; the precision target
requires 601. Rounding upward across 54 balanced strata gives **648 distinct
pairs**, 12 per stratum and 1,296 agent executions. Final paired inference uses
the exact McNemar test, not the planning approximation.

These assumptions are a prospective design choice. A sensitivity analysis
with alternative discordance rates must be frozen alongside the protocol.

## Factorial coverage

The 54 primary strata cross:

| Axis | Levels |
|---|---|
| task domain | support, document, cloud |
| agent configuration | stateless/local, persistent-memory/full-tools |
| root role | upstream tool, state/memory boundary, downstream decision/action |
| observability | explicit local failure, silent corruption, distant visible symptom |

Every pair must use a distinct task instance and surface realization. Repeated
model calls on one identical trace measure stability and are not independent
sample units.

The implementation must ensure that each root role is reachable in every
declared cell. Impossible combinations must be removed before freezing the
design, with the revised number of strata passed back to the power script.

## Difficulty controls

The confirmatory split must include conditions where observation alone cannot
trivially reveal the intervention:

- neutral fault payloads that do not name the target or an obvious synonym;
- silent upstream corruption with the first explicit failure downstream;
- memory-mediated propagation and successful recovery cases;
- distractor failures that are causally unrelated to the intervention;
- structurally divergent trajectories retained in the denominator.

Difficulty is not created by hiding information from one baseline. Cascad,
DeepSeek, OpenAI when available, Qwen, and Mistral receive the information
specified for their declared ablation. Any paired-LLM comparison must verify
identical prompt and trace hashes.

## Freeze and analysis policy

Before the first confirmatory execution, freeze:

1. task generators and template IDs;
2. all instance seeds and cell assignments;
3. injected root and fault specifications;
4. calibration and canonicalization rules;
5. candidate-node policy and observable serialization version;
6. exclusion policy, which retains absorbed, recovered, repeated-exposure,
   and structurally divergent runs;
7. primary endpoint, family-wise testing order, subgroup status, and missing
   data policy;
8. exact package versions, model revisions, prompts, and hashes.

Calibration, engineering smoke tests, and confirmatory instances must use
disjoint IDs. Operational preflight may test formatting and hardware capacity
without reading labels. Any outcome-dependent amendment creates V4; it does
not silently modify V3.

## External validity

The 648-pair primary design strengthens causal-position and task diversity, but
it remains a one-agent-model study unless a second agent model executes a
separately frozen replication split. Local Hugging Face attribution models are
baseline observers; they do not count as additional agent-model families.
