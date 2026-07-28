# Real-agent controlled study v2

## Collection status

**CALIBRATION, SPECIFICITY, CONTROLLED COLLECTION, AND DEEPSEEK ATTRIBUTION:
COMPLETE. OPENAI BASELINE: UNAVAILABLE FOR THIS COLLECTION.**

The full 60-pair v2 calibration and the distinct 60-pair held-out specificity
split were completed before the controlled split was opened. The controlled
collection then completed all 200 preregistered clean/perturbed pairs. Its
explicit confirmation gate, matching frozen threshold file, and prerequisite
checks remain enforced for reproduction.

V1 calibration and pilot artifacts remain immutable. V2 is a new protocol
because multi-step memory tasks introduce `remember`, `memory_write`,
`search_memory`, and later decision events that the v1 calibration did not
cover.

SHA-256 of the frozen preexecution protocol:

```text
72b6b95e3571ddaaeef92757028e374016d941356db6e5a78a5bb49305e0fb85
```

## Frozen datasets

| Split | Pairs | Purpose |
|---|---:|---|
| calibration clean/clean | 60 | freeze thresholds, including multi-step keys |
| specificity clean/clean | 60 | estimate false-positive behavior out of calibration |
| controlled clean/perturbed | 200 | root localization and propagation |

Specificity has 30 pairs per configuration, with 15 local arithmetic and 15
local time pairs. Its IDs, prompts, expressions, and templates are disjoint
from calibration and controlled instances. Root accuracy is not defined for
this no-fault split.

The controlled split has 100 pairs per configuration. Under
`full-tools-persistent-memory`, 40/100 are structurally eligible multi-step
tasks. Overall, 40/200 controlled pairs are multi-step. Eligibility is fixed
from the protocol:

```text
calculate observation
-> later model decision
-> remember / persistent write
-> search_memory / later read
-> final decision
```

Realization is assessed only after execution and never used as an exclusion.
Absorbed, detected, blocked, and recovered faults remain in all denominators.

## Coverage gate

`coverage_matrix.json/csv` records, before execution:

- task family and configuration;
- expected node/event keys;
- memory read and write opportunities;
- expected tool-call count;
- downstream decision opportunity;
- matching v2 calibration stratum.

There are nine structural rows. Every specificity and controlled stratum maps
to a v2 calibration stratum. The threshold-freeze script must additionally
observe every expected key. Missing multi-step keys fail calibration; v1
thresholds are never silently reused.

All 60 calibration pairs are complete (120 executions), with no invalid
execution, hash failure, or fault event. All ten multi-step calibration pairs
realized `calculate`, `remember`, and `search_memory` in both repetitions,
including explicit `memory_write` and tool-level `memory_read` events.

The resulting freeze contains 45 event-key thresholds and passes the full
coverage gate. It was produced with the real `all-MiniLM-L6-v2` embedding
encoder and calibration-only data; `heldout_data_accessed` is `false`.

## Input and result hashes

Preexecution manifests contain only:

```text
instance_spec_sha256
task_template_sha256
fault_spec_sha256
configuration_sha256
prompt_template_sha256
candidate_policy_sha256
canonicalization_policy_sha256
```

They contain no raw or canonical trace hash. The runner writes raw trace hashes
to a separate postcollection integrity manifest. Cascad analysis later adds
canonical clean/corrupt hashes without modifying preregistration.

## Memory isolation

Every execution receives its own SQLite namespace. The namespace is cleared
before execution, only its SHA-256 is exported, and production memory cannot be
cleared through this API. Multi-step writes and reads remain available within
the same execution while clean and perturbed conditions cannot contaminate one
another.

## Required analyses

Specificity reports global, configuration, family, and event-key
false-positive rates; pairs with false subgraphs; mean falsely contaminated
nodes; and Wilson 95% intervals.

The held-out specificity collection is complete (60 pairs, 120 executions,
zero injected faults, zero invalid records). It observed one false-positive
pair:

- global: 1/60 = 1.67%, Wilson 95% CI [0.29%, 8.86%];
- full tools with persistent memory: 0/30, CI [0%, 11.35%];
- local core without memory: 1/30 = 3.33%, CI [0.59%, 16.67%];
- arithmetic: 0/30; local time: 1/30;
- false-subgraph pair rate: 1/60;
- mean falsely contaminated nodes per pair: 1/60 = 0.0167.

The single exceedance is
`specificity--local-core-no-memory--local_time_tool--010`, at
`call_model|model_response|1`: observed distance 0.0044622 versus frozen
threshold 0.0041745. Root accuracy remains explicitly not applicable and was
not computed.

Controlled propagation uses non-exclusive outcome flags plus one primary class:

```text
propagated_to_memory
propagated_without_memory
absorbed_by_agent
detected_and_blocked
no_persistence_opportunity_realized
final_failure
successful_recovery
```

It also reports realized propagation, absorption, persistence, observed depth,
subgraph precision/recall, and performance by realized path length.

## Controlled results

The append-only execution manifest contains 403 attempts. Exactly 400
successful execution keys form the 200 preregistered pairs. Three earlier
`GraphRecursionError` attempts are retained as failed attempts; their later
successful retries are used. Eight perturbed pairs called the preregistered
target more than once and therefore contain repeated fault exposure. These are
realized agent behaviors, not exclusions. No pair was excluded.

The counterfactual analysis reports:

- root localization: 200/200 = 100%, Wilson 95% CI [98.12%, 100%];
- realized propagation: 185/200 = 92.5%;
- absorption: 15/200 = 7.5%;
- persistence: 19/200 = 9.5%;
- final-response failure: 108/200 = 54.0%;
- successful recovery: 77/200 = 38.5%;
- 189 aligned and 11 structurally divergent trajectory pairs.

The time-observation analysis uses a context-aware nuisance-clock policy.
Timestamps within 24 hours of the raw pre-injection clock observation are
canonicalized as wall-clock variation. Temporally distant observations remain
observable. This prevents the injected 1999 observation from being erased as
if it were an ordinary run timestamp. The raw pre-injection observation stays
excluded from the causal event sequence and all original traces remain
immutable.

On the exact same 200 pairs, DeepSeek `deepseek-v4-flash` completed 200/200
attribution calls without request error and localized 190/200 roots = 95.0%,
Wilson 95% CI [91.04%, 97.26%]. It selected `call_model` in four incorrect
cases and produced no parsed candidate in six. Cascad was correct and DeepSeek
incorrect in ten paired cases; the exact two-sided McNemar p-value is 0.001953.
This result is reported as observed, not as evidence that an LLM baseline must
fail.

The optional OpenAI baseline could not be collected: its one preflight request
returned HTTP 429, `account is not active`. No mass OpenAI run was attempted.
The same-run comparison remains pending activation of billing for that API
account.

## Attribution instrumentation

Every new DeepSeek attribution call records request start/end, latency,
requested and provider-returned models, HTTP status, token counts, raw and
parsed output, parser validity, wrapper attempt count, configured maximum
retries, provider-internal retry count, and error. The provider-internal retry
count remains `null` when unavailable. A successful response is not interpreted
as proof of zero internal retries.

## Ground-truth policy

Root-cause labels are fixed by the controlled fault-injection protocol before
execution. Accuracy therefore measures recovery of an intervention-defined
root, not agreement with a subjective human judgment. No human annotation
study was executed and no claim about human agreement is made.

The earlier plan included a secondary 80-pair blinded human subset. It was
removed before annotation because it is not required to establish the known
injection location and would answer a different question: whether observers
infer the same cause from the visible trace. The retained records still permit
such a separate study, but it is outside the reported analysis.

## Protocol amendment

The context-aware nuisance-clock rule described above was introduced after the
controlled outcomes had been inspected. The raw traces, injected roots,
candidate nodes, and execution inclusion policy were not altered. Nevertheless,
the corrected metrics are post-hoc evidence for this normalization rule and
must not be described as an untouched confirmatory result. Confirmatory support
for the corrected rule requires a new held-out protocol version.

The local Qwen and Mistral attribution baselines were also added after the
controlled collection. They are secondary analyses of frozen traces and do not
change the primary data.

## Claim boundary

Claims remain limited to one primary agent model, two agent configurations,
preregistered local tools, controlled fault families, and the trace
distributions actually observed. V2 is not a multi-agent-model validation. See
`docs/limitations.md` for the full reporting boundary.

## Evidence and reproduction

Versioned evidence is stored in:

```text
runs/real-agent-confirmatory-v2-calibration/
runs/real-agent-confirmatory-v2-specificity/
runs/real-agent-confirmatory-v2-controlled/
```

From the `Cascad` repository, the completed analyses reproduce with:

```bash
PYTHONPATH=src .venv/bin/python scripts/analyze_real_agent_calibration_v2.py
PYTHONPATH=src .venv/bin/python scripts/analyze_real_agent_specificity_v2.py
PYTHONPATH=src .venv/bin/python scripts/analyze_real_agent_controlled_v2.py \
  --attribution deepseek
```

The first command is overwrite-protected for the threshold freeze. The second
uses that exact freeze and never reports root accuracy for clean/clean pairs.
The third reuses its append-only LLM checkpoint and makes no duplicate
successful DeepSeek calls.
