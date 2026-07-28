# Difficult attribution conditions — experimental report

Date: 22 July 2026. These are controlled simulator results, not final paper
results. DeepSeek was called with `deepseek-chat`, temperature 0, on the same
20 seeds used by Cascad's paired clean/corrupt replay.

## Scenario design

The original `support`, `document`, and `cloud` scenarios remain unchanged.
Three additional neutral-observation variants preserve their topology, clean
trajectory and injection location while replacing only the visible faulty
state:

| Variant | Injection node | Neutral observation |
| --- | --- | --- |
| `support_neutral` | `refund_api` | inconsistent decision state |
| `document_neutral` | `generate_report` | inconsistent artifact state |
| `cloud_neutral` | `share` | inconsistent authorization state |

The `cloud_distant_symptom` condition uses the native chain:

```text
planner -> upload -> share -> memory -> notify -> responder
```

The clean run stores `viewer`. In the corrupted run, `share` silently returns
`restricted` with `ok=true`; memory persists that state; `notify` is the first
node to emit `ok=false`; the responder then reports failure. The observable
trace never says that `share` caused the failure.

DeepSeek receives the complete **observable** trace. Span bookkeeping,
`FAULT_INJECTED`, interventions, source-fault IDs and causal edges are excluded
because they are privileged evaluator information. Each prompt still includes
all observable planner, tool, memory and response events, including `share`.

## Leakage audit

The requested source search was executed:

```bash
for node in refund_api generate_report share upload notify tool memory; do
    grep -ri "$node" src/cascad/scenarios/*.py
done
```

Every match was inspected manually. Matches in topology tuples, dictionary
keys, class attributes and target identifiers are internal identifiers and are
expected. None of the neutral fault messages contains its target node or a
direct synonym. The automated observable-trace audit produced:

```text
support_neutral: PASS ... leaked_terms=[] privileged_metadata=False
document_neutral: PASS ... leaked_terms=[] privileged_metadata=False
cloud_neutral: PASS ... leaked_terms=[] privileged_metadata=False
cloud_distant_symptom: PASS ... leaked_terms=[] privileged_metadata=False
```

Reproduce it with:

```bash
PYTHONPATH=src python3 scripts/audit_attribution_leakage.py
```

## Experiment commands

Each attribution command performs both DeepSeek attribution and paired
counterfactual graph diagnosis on the exact same corrupt runs:

```bash
PYTHONPATH=src .venv/bin/python -m cascad.cli experiment \
  --scenario support_neutral --n-repeats 20 --attribution deepseek \
  --env-file ../react-agent/.env --out runs/support-neutral-attribution

PYTHONPATH=src .venv/bin/python -m cascad.cli experiment \
  --scenario document_neutral --n-repeats 20 --attribution deepseek \
  --env-file ../react-agent/.env --out runs/document-neutral-attribution

PYTHONPATH=src .venv/bin/python -m cascad.cli experiment \
  --scenario cloud_neutral --n-repeats 20 --attribution deepseek \
  --env-file ../react-agent/.env --out runs/cloud-neutral-attribution

PYTHONPATH=src .venv/bin/python -m cascad.cli experiment \
  --scenario cloud_distant_symptom --n-repeats 20 --attribution deepseek \
  --env-file ../react-agent/.env --out runs/cloud-distant-attribution

PYTHONPATH=src python3 scripts/summarize_attribution_study.py
```

Long API runs support `--seed-start` and `--append`; raw evidence is
checkpointed after every response.

## Results

| Scenario | Root cause | Visible symptom | DeepSeek distribution | Cascad distribution | DeepSeek accuracy | Cascad accuracy |
| --- | --- | --- | --- | --- | ---: | ---: |
| support neutral | refund_api | refund_api | refund_api: 20 | refund_api: 20 | 1.00 | 1.00 |
| document neutral | generate_report | generate_report | generate_report: 20 | generate_report: 20 | 1.00 | 1.00 |
| cloud neutral | share | share | share: 20 | share: 20 | 1.00 | 1.00 |
| cloud distant symptom | share | notify | memory: 20 | share: 20 | 0.00 | 1.00 |

DeepSeek confidence is exported as `null`: this endpoint/configuration did not
provide a calibrated confidence value, and Cascad does not invent one.

Every raw record contains scenario, seed, injection node, visible failure,
raw DeepSeek output, parsed prediction, optional confidence, graph prediction,
root/symptom classification, candidates and exact prompt. The combined
confusion table contains 80 rows in `runs/attribution-study/confusion.csv` and
`.json`.

## Interpretation

DeepSeek and Cascad both solve all three neutral-payload variants. These
conditions are therefore insufficiently challenging to separate the methods:
the faulty node itself emits the first explicit failure, so observation alone
is enough even without a revealing message.

The distant-symptom condition creates a genuine distinction. Cascad identifies
`share`, the first clean/corrupt divergence, in all 20 runs. DeepSeek selects
`memory` in all 20 runs. It does **not** simply choose the visible `notify`
failure; it identifies the intermediate persistence component that carries the
incorrect state. This is a meaningful diagnostic answer, but it is not the
known injected root cause.

The result supports a narrower claim: paired counterfactual replay distinguishes
the first divergent cause from both downstream symptoms and propagation
mediators, whereas single-trace LLM attribution may select a causally involved
intermediate component. It does not establish general superiority over
DeepSeek, and it must be replicated on non-deterministic agent traces before
publication.
