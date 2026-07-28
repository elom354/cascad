# Paired attribution information-ablation study

## Scope and implementation controls

This study isolates three explanations for attribution behavior on
`cloud_distant_symptom`: information access, prompt wording, and Cascad's
structured comparison. It uses a deterministic simulator and does not support
publication-level claims about arbitrary LLM agents.

The implementation was inspected before modification. The relevant controls
are:

- `src/cascad/attribution_baseline.py`
  - `DeepSeekAttributor`: HTTP client, model and temperature;
  - `serialize_trace_for_attribution`: common observable-event filter;
  - `build_attribution_prompt`: the three explicit prompt modes;
  - `audit_prompt`: pre-API privileged-information checks;
  - `_parse_node`: one shared parser for all modes.
- `src/cascad/scenarios/attribution.py`
  - `generate_cloud_distant_instance`: deterministic surface variation;
  - `attribution_fault`: seed-matched silent state deviation at `share`.
- `src/cascad/simulator.py`
  - `ReActPropagationSimulator.run`: paired instance construction, memory
    propagation and the visible `notify` failure.
- `src/cascad/experiment.py`
  - `run_experiment`: same-seed clean/corrupt replay, natural-divergence
    calibration, Cascad graph prediction, raw record schema and metrics.
- `src/cascad/cli.py`
  - `--attribution-mode`, uniqueness enforcement, checkpoint/resume and
    CSV/JSON exports.
- `src/cascad/export.py`
  - deterministic JSON and CSV serialization.

The causal reconstruction in `causal.py` was not changed for this study.

## Conditions

All varied conditions use 20 paired instances with the same structure:

```text
planner -> upload -> share -> memory -> notify -> responder
```

`share` silently changes the first state (`ok=true`), memory persists it,
`notify` emits the first explicit failure, and the responder reports failure.
Each seed varies resource ID, clean/observed state pair, task wording, memory
field, notification wording, harmless fields and final response wording.

The modes are:

1. `deepseek_single_neutral`: corrupt observable trace only, neutral prompt.
2. `deepseek_single_guided`: corrupt observable trace only, existing
   anti-symptom/root-cause instruction.
3. `deepseek_paired`: complete clean and observed traces, with no generated
   textual diff.
4. Cascad counterfactual graph: paired traces, clean/clean natural-divergence
   calibration, semantic/structured distance and native dependencies.

The earlier 20 guided calls are retained separately as a stability check. They
used one deterministic input 20 times: `unique_trace_count=1`,
`unique_prompt_count=1`, `api_call_count=20`. They are repeated measurements,
not 20 independent samples.

## Leakage and uniqueness audit

Before API execution:

```bash
PYTHONPATH=src python3 scripts/audit_attribution_leakage.py \
  --instances 20 --out runs/paired-attribution-study
```

Actual output:

```text
support_neutral: PASS message_leaks=[]
document_neutral: PASS message_leaks=[]
cloud_neutral: PASS message_leaks=[]
cloud_distant_symptom: PASS message_leaks=[]
deepseek_single_neutral: prompts=20 traces=20 status=PASS
deepseek_single_guided: prompts=20 traces=20 status=PASS
deepseek_paired: prompts=20 traces=20 status=PASS
```

Single-trace prompts contain one observed trace and no clean reference or
privileged injection metadata. Paired prompts contain both traces under the
necessary reference/observed headings. Neither view contains `FAULT_INJECTED`,
source-fault IDs, evaluator labels, interventions, graph outputs or dependency
edges. Every raw call exports the exact SHA-256 prompt and observable-trace
hashes plus the audit flags.

## Commands

One-instance smoke test:

```bash
PYTHONPATH=src .venv/bin/python -m cascad.cli experiment \
  --scenario cloud_distant_symptom --n-repeats 1 \
  --attribution deepseek --attribution-mode paired \
  --env-file ../react-agent/.env --out runs/paired-attribution-smoke
```

Repeated-call stability check (one identical instance):

```bash
PYTHONPATH=src .venv/bin/python -m cascad.cli experiment \
  --scenario cloud_distant_symptom --n-repeats 20 --fixed-instance-seed 0 \
  --attribution deepseek --attribution-mode single-guided \
  --env-file ../react-agent/.env --out runs/paired-study-stability
```

Varied-instance study (repeat for `single-guided` and `paired`):

```bash
PYTHONPATH=src .venv/bin/python -m cascad.cli experiment \
  --scenario cloud_distant_symptom --n-repeats 20 \
  --attribution deepseek --attribution-mode single-neutral \
  --require-unique-instances --env-file ../react-agent/.env \
  --out runs/paired-study-single-neutral
```

Checkpointed runs can resume with `--seed-start N --append`. Aggregation:

```bash
PYTHONPATH=src python3 scripts/summarize_paired_attribution_study.py
```

Complete tests:

```bash
python3 -m pytest tests -q
PYTHONPATH=src .venv/bin/python -m pytest tests -q
```

Exact final output in both environments:

```text
..........................                                               [100%]
26 passed
```

## Results

| Method | Information available | Prompt type | Unique instances | Root accuracy | Mediator rate | Symptom rate | Invalid rate | Mean root distance |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DeepSeek single-neutral | corrupt trace | neutral | 20 | 0.00 | 0.00 | 1.00 | 0.00 | 2.00 |
| DeepSeek single-guided | corrupt trace | anti-symptom guided | 20 | 0.10 | 0.90 | 0.00 | 0.00 | 0.90 |
| DeepSeek paired | clean + corrupt traces | stepwise comparison | 20 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Cascad counterfactual graph | paired + calibration + dependencies | structured algorithm | 20 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Repeated-call stability check | one corrupt trace repeated | guided | 1 | 0.00 | 1.00 | 0.00 | 0.00 | 1.00 |

DeepSeek returned valid candidates in all 60 varied calls. No calibrated
confidence was available, so confidence remains `null` rather than being
invented.

Prompt wording matters: without coaching, DeepSeek selects the visible
`notify` symptom in 20/20 cases. The guided prompt moves 18/20 decisions to the
`memory` mediator and 2/20 to the `share` root. This is explicit prompt
sensitivity, not random invalid output.

Providing the paired clean reference closes the gap: DeepSeek identifies
`share` in 20/20 instances, matching Cascad. Therefore the single-trace result
does **not** establish superior Cascad reasoning accuracy. It primarily shows
an information-availability limitation.

Cascad's contribution in these experiments is the systematic creation,
alignment, calibration and dependency-constrained exploitation of paired
evidence. Once the same paired observations are exposed in raw form, DeepSeek
can also locate the first divergence on this scenario family.

The paired correctness artifact is suitable for an exact McNemar calculation
but no significance is claimed automatically. For paired DeepSeek versus
Cascad, all 20 cases are jointly correct and there are no discordant pairs.

## Artifacts

- `runs/paired-study-single-neutral/attribution_raw.json`
- `runs/paired-study-single-guided/attribution_raw.json`
- `runs/paired-study-paired/attribution_raw.json`
- `runs/paired-attribution-study/raw_calls.json`
- `runs/paired-attribution-study/summary.json` and `summary.csv`
- `runs/paired-attribution-study/paired_correctness.json` and `.csv`
- `runs/paired-attribution-study/leakage_audit.json`
- `runs/paired-attribution-study/prompt_trace_hash_audit.json`

## Bounded conclusion

On these 20 deterministic simulator instances, paired information—not an
observed intrinsic reasoning advantage—explains the separation between Cascad
and single-trace DeepSeek. The structured framework still provides a
reproducible way to generate and constrain that evidence. This conclusion is
limited to the evaluated simulator and should not be generalized to real,
non-deterministic agents without further controlled experiments.
