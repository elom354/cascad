# Inter-Episode Memory Propagation Study

## Verdict: PASS

This is the final deterministic simulator family for Cascad v2. The protocol
was fixed before evaluation and no additional simulator benchmark family is
permitted after this study.

## Questions and fixed roles

The study evaluates how attribution changes with the information window,
whether diagnostic roles remain distinguishable, how long contaminated memory
remains active, and how pre-write or delayed correction changes future
behavior.

The independently preregistered roles are:

```text
ground-truth source s* = tool in Episode 1
failed safeguard       = verifier in Episode 1
persistence mediator   = memory
visible symptom        = action
final manifestation    = responder
```

`FaultSpec.injection_node` fixes `s*` before reconstruction. The verifier has a
real blocking opportunity: with `fail_open=False`, a restricted state is
rejected and cannot be written; the experimental false-negative condition
accepts the unusable state. It is a failed safeguard, not a mediator or a
second `FaultSpec` source.

## Design

- Horizons: `K ∈ {1, 3, 5, 10}`.
- Difficulties: E0–E3.
- Five unique held-out instances per difficulty×horizon cell: 80 total.
- Calibration: 24 disjoint clean/clean K=10 histories.
- Delayed correction episodes: `c ∈ {2, 3, 5}`.
- No development split was used.
- All instance IDs, memory-object IDs, clean/corrupt history hashes, and paired
  prompts are unique.

E0 has no benign variation. E1 adds held-out planning wording. E2 combines
multiple benign fields with a symptom delayed to episode 3. E3 combines all
held-out variations with the first explicit symptom at episode 5.

The information windows are kept separate:

- A1: symptom episode only;
- A2: complete corrupt history;
- A3: paired clean/corrupt history.

A1 and A2 are information ablations, not equal-information comparisons with
Cascad.

## Persistence definitions

For every episode `k ≥ 2`, the denominator for MP(k) and BP(k) is the number of
runs in which a memory read is attempted. Episode 1 has no read opportunity and
therefore a null denominator, not a successful recovery.

```text
MP(k) = readable contaminated memory / eligible memory-read attempts
BP(k) = behavior changed by contaminated memory / eligible attempts
FFR   = explicit post-injection failures / post-injection episodes
```

Persistence Horizon is the final readable episode. Time to Recovery is the
first episode after which neither readable nor behaviorally active
contamination remains. Runs still active at K are right-censored.

## Attribution results

| Method/window | Source accuracy | Invalid rate | Total tokens |
|---|---:|---:|---:|
| DeepSeek symptom episode only | 0.250 | 0.000 | 24,460 |
| DeepSeek corrupt full history | 0.925 | 0.075 | 95,313 |
| DeepSeek paired history | 0.988 | 0.013 | 176,562 |
| naive first aligned difference | 1.000 | 0.000 | 0 |
| full Cascad | 1.000 | 0.000 | 0 |
| Cascad without calibration | 1.000 | 0.000 | 0 |
| Cascad without dependencies | 1.000 | 0.000 | 0 |

The symptom-only model selected `memory` in 60/80 histories and `tool` in the
20 K=1 histories, where Episode 1 was also the only available episode. Full
corrupt history materially improved source localization. Paired history
produced 79/80 valid/correct sources. One paired and six corrupt-history
responses contained explanatory prose instead of the required single node ID;
the preregistered strict parser retains them as invalid rather than reparsing
them after observation.

Full Cascad independently estimated `tool`, `verifier`, and `memory` correctly
in all 80 histories. It identified `action` in every history whose horizon
actually contained the delayed explicit symptom; shorter horizons are excluded
from that role denominator.

The exact paired Cascad/DeepSeek-history comparison has one discordant case,
accuracy difference `0.0125`, and exact two-sided McNemar `p=1.0`. Cascad,
its two ablations, and the naive paired baseline all score `1.0`; calibration
and dependency constraints therefore add no measurable source-localization
value on this simple chain. The study does not establish attribution
superiority over DeepSeek or the simple aligned baseline.

## Persistence and intervention results

Across the 300 eligible post-injection episode records per policy:

| Policy | MP | BP | FFR | FFR reduction vs C0 |
|---|---:|---:|---:|---:|
| C0 no intervention | 1.000 | 1.000 | 0.817 | 0.000 |
| C1 pre-write quarantine | 0.000 | 0.000 | 0.000 | 0.817 |
| C2 correction at 2 | 0.000 | 0.000 | 0.000 | 0.817 |
| C2 correction at 3 | 0.200 | 0.200 | 0.100 | 0.717 |
| C2 correction at 5 | 0.533 | 0.533 | 0.350 | 0.467 |

Without intervention, all 60 runs with at least one post-injection episode are
right-censored: contamination remains readable through their horizon.
Pre-write quarantine blocks 80/80 corrupt writes, triggers on 0/80 clean
controls, and creates zero measured utility loss from false blocking.

Delayed corrections recover at the preregistered episode when it occurs within
the horizon. Runs whose horizon ends before their correction point are retained
as unrecovered/right-censored where a read opportunity exists; they are not
assigned recovery at K.

Wilson intervals and explicit denominators are exported for MP(k), BP(k), FFR,
clean false positives, and block rates. Recovery records and an unrecovered
survival-style curve are exported separately.

## Fairness and leakage

Both audits pass before and after API execution:

- fault events and evaluator roles are absent from prompts;
- intervention decisions are hidden from the attribution study;
- A1/A2 never receive clean traces;
- equal-information candidates are identical;
- calibration and evaluation IDs/templates are disjoint;
- all 240 API responses are checkpointed individually;
- monetary cost remains `null` because the provider did not return a charge.

## Commands

Local structural/persistence pass:

```bash
uv run --extra semantic python -m cascad inter-episode-study \
  --out runs/inter-episode-persistence-study
```

Checkpointed DeepSeek completion:

```bash
uv run --extra semantic python -m cascad inter-episode-study \
  --out runs/inter-episode-persistence-study \
  --deepseek \
  --env-file ../react-agent/.env
```

## Bounded conclusion

Phase 2 passes. On this controlled multi-episode simulator, complete history
substantially improves LLM source attribution; paired information nearly closes
the remaining gap. Cascad separates the injected source, failed safeguard,
persistence mediator, and observed symptom, while pre-write quarantine prevents
future contaminated reads without measured clean false positives. Earlier
correction shortens persistence and reduces future failures. These results are
limited to the fixed simulator histories, roles, policies, and trace
distribution.
