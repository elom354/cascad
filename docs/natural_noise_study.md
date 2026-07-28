# Controlled Natural-Divergence Robustness Study

## Frozen baseline and implementation map

The study extends the earlier `cloud_distant_symptom` protocol without
replacing or modifying its saved commands and outputs. Its exact inputs,
configuration, encoder identity, and output hashes are retained in the
versioned study artifacts.

The implementation points frozen before this extension were:

| Concern | Existing implementation |
|---|---|
| clean/clean calibration | `estimate_natural_divergence` in `src/cascad/divergence.py` |
| event alignment | `corresponding_events` in `src/cascad/divergence.py` |
| text and structured distances | `text_distance`, `event_distance`, and `value_distance` in `src/cascad/divergence.py` |
| dependency-constrained reconstruction | `CausalGraph._from_counterfactual` in `src/cascad/causal.py` |
| DeepSeek paired prompt | `build_attribution_prompt` in `src/cascad/attribution_baseline.py` |
| scenario generation | `generate_cloud_distant_instance` and `Scenario.instantiate` |
| raw exports and summaries | `run_experiment`, the `experiment` CLI branch, and `cascad.export` |

The new study is implemented separately in `src/cascad/natural_noise.py` and
exposed through `cascad natural-noise-study`. This freezes the earlier paired
attribution protocol while allowing explicit held-out calibration, controlled
noise annotations, ablations, and study-specific exports.

## Diagnostic definitions

- **Root:** the earliest node whose paired divergence exceeds its calibrated
  natural-divergence threshold and which lies on a dependency-constrained
  contaminated path.
- **Mediator:** a downstream node that carries, transforms, amplifies or
  persists the contaminated state.
- **Visible symptom:** the first node that emits an explicit externally
  observable failure.

These labels are not interchangeable and evaluator-only labels are never
included in model prompts.

## Protocol and commands

Implementation and pilot results are appended by the study runner. The local
pilot (no paid API calls) is:

```bash
PYTHONPATH=src python3 -m cascad natural-noise-study \
  --instances-per-level 20 --calibration-pairs 24 \
  --out runs/natural-noise-study
```

DeepSeek is opt-in and checkpoints every response:

```bash
PYTHONPATH=src python3 -m cascad natural-noise-study \
  --instances-per-level 20 --calibration-pairs 24 \
  --deepseek --env-file ../react-agent/.env \
  --out runs/natural-noise-study
```

The semantic environment required for paper-facing measurements is:

```bash
uv sync --extra semantic
uv run --extra semantic python -m cascad natural-noise-study \
  --instances-per-level 20 --calibration-pairs 24 \
  --out runs/natural-noise-study
```

## Scenario and controlled noise

The new family keeps the linear native dependency skeleton:

```text
planner -> upload -> share -> memory -> notify -> responder
```

The executable authorization first changes silently at `share`, `memory`
persists that state, and `notify` emits the first explicit failure. Noise is
drawn from split-specific template banks:

- N0: no benign difference;
- N1: one semantic planner paraphrase;
- N2: three differences across planner, upload, and memory;
- N3: five semantic and structured differences across planner, upload,
  memory, share, and notify.

Semantic additions are display-only prose that no execution branch consumes.
Structured identifiers, receipts, correlation values, timestamps, and optional
metadata are declared non-causal and are likewise never read by the simulator.
All rationales and the four evaluator categories are stored only in the
manifests. They are absent from serialized model traces. The topology is a
strict dependency chain, so there is no independent native branch on which an
event can be safely reordered. This pilot therefore does not manufacture a
benign event reordering that the graph cannot justify.

Calibration uses 24 clean/clean pairs. Evaluation uses 80 held-out
clean/corrupt pairs (20 at each level). Instance IDs and template IDs are
disjoint. Exact template surfaces and identifiers are not reused across the
split. Canonicalization and calibration use the same event representation; a
pre-pilot semantic test caught and corrected an earlier representation
mismatch before the 80-instance evaluation was run.

## Methods

Every method receives the same held-out trace pairs:

1. naive first exact serialized difference;
2. canonicalized first difference, ignoring only declared non-causal fields;
3. maximum uncalibrated event distance;
4. Cascad without calibration, retaining dependency constraints;
5. Cascad without dependency constraints, retaining calibration;
6. full Cascad with held-out calibration and native reachability.

The opt-in DeepSeek conditions use the same existing paired instruction. The
calibrated condition adds the exact clean-only per-node field/event/distance
summary exported in `fairness_audit.json`; it receives no evaluator label or
evaluation example.

## Local semantic pilot results

The completed pilot used the local `all-MiniLM-L6-v2` embedding encoder.
Fairness, leakage, split-overlap, and all three uniqueness checks passed.

| Method | Root accuracy N0 | N1 | N2 | N3 | Benign selection N3 | False contamination N3 |
|---|---:|---:|---:|---:|---:|---:|
| naive first raw difference | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 |
| canonicalized first difference | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 |
| maximum raw divergence | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Cascad without calibration | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.333 |
| Cascad without dependencies | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| full Cascad | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 |

Full Cascad recovered the full four-node contaminated subgraph at every level
(precision and recall 1.00). Removing calibration made the benign planner
surface the selected node at N1–N3. Removing dependency constraints made no
measurable difference on this strictly linear topology. This result therefore
supports a benefit for `D_natural` on this simulator family, but does not
support a separate reachability benefit under this topology.

DeepSeek was deliberately not called by the local command: API execution is
opt-in and the runner reports `api_call_count=0` and unknown cost rather than
inventing an estimate. Run the documented `--deepseek` command to add the raw
and calibrated LLM rows; each response is appended immediately to
`raw_results.jsonl`.

## Outputs

The runner produces split manifests, full JSON/JSONL records, summary CSV/JSON,
an ablation table, latency/cost table, fairness audit, bootstrap 95% intervals,
paired-correctness and McNemar 2x2 inputs, and six SVG plots under
`runs/natural-noise-study/`.

## Bounded conclusion

On 80 held-out instances sharing one simulated cloud-workflow skeleton,
clean/clean field calibration prevented controlled benign differences from
being selected as root causes. Native dependency filtering added no measurable
value because every studied node lies on one linear reachable path. No claim
of cross-domain generalization or superiority over DeepSeek follows from this
local, no-API pilot.

## API baselines on the frozen evaluation set

This section completes the equal-information paired comparison; it does not
replace the local pilot above. The committed 24 calibration and 80 evaluation
pairs were kept unchanged. Because the original pilot retained observable trace
hashes rather than clear trace payloads, each trace was materialized
deterministically in memory and accepted only after its clean and corrupt hashes
matched the frozen per-instance hashes. Candidate sets, levels, template IDs,
the clean-only calibration context, and all uniqueness counts were also
verified before API execution.

Both DeepSeek conditions used `deepseek-chat`, temperature `0`, the same clean
and corrupt observable traces, and the same candidate node IDs. The raw
condition received only the existing first-meaningful-divergence prompt. The
calibrated condition additionally received the exact context exported in
`runs/natural-noise-study-deepseek/calibration_context.json`. No evaluator
labels, injection metadata, graph edges, Cascad predictions, or worked examples
were provided.

| Method | N0 root | N1 root | N2 root | N3 root | N3 mediator | N3 symptom |
|---|---:|---:|---:|---:|---:|---:|
| DeepSeek paired raw | 1.00 | 1.00 | 1.00 | 0.90 | 0.00 | 0.10 |
| DeepSeek paired calibrated | 1.00 | 1.00 | 1.00 | 0.90 | 0.05 | 0.05 |
| full Cascad | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 |

Neither DeepSeek condition selected a benign-noise node or produced an invalid
output. Both made two N3 errors, although the raw and calibrated conditions
failed on different paired instances: their paired table contains 76
both-correct cases, two raw-only correct cases, and two calibrated-only correct
cases. Against full Cascad, each condition has 78 both-correct and two cases
where only Cascad is correct. These are McNemar table inputs; no automatic
significance claim is made.

Mean latency ranged from approximately 1.66 to 2.06 seconds per call. All 160
API responses contain token-usage records. The API did not return a monetary
charge, so cost remains `null` rather than being inferred from an unstated
pricing schedule.

### Bounded API interpretation

The controlled noise did not cause progressive degradation for either paired
DeepSeek condition through N2. At N3, both were slightly below full Cascad
(`0.90` versus `1.00`), while supplying calibration context did not change
aggregate root accuracy. On this simulator family, these observations are
consistent with a small robustness difference in structured threshold
application, but the 2/20 N3 gap is not by itself evidence of general
superiority or statistical significance. The earlier single-trace conditions
are not used as a fair same-information comparison.
