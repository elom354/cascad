# Scope and limitations

Cascad measures causal propagation under controlled interventions. The current
evidence supports conclusions about the tested traces and fault families, not
unrestricted autonomous agents.

## Supported claims

- A known intervention can be localized by comparing matched clean and
  perturbed trajectories.
- Propagation depth, delay, breadth, memory persistence, recovery, and
  containment can be measured on instrumented traces.
- Counterfactual graph attribution can be compared with observation-only
  language-model attribution on identical candidate sets and traces.

## Current limits

- The real-agent collection uses one primary agent model, two configurations,
  local tools, and controlled fault families.
- Root causes are intervention-defined. The project does not report agreement
  with human causal judgments.
- Task family, target tool, and injected fault family are not fully crossed in
  the current real-agent dataset.
- Some graph edges are reconstructed from event order when a runtime does not
  expose an explicit dependency edge. Such edges support trace-level
  propagation analysis but should not be interpreted as a complete structural
  causal model.
- The final V2 clock-normalization rule was corrected after controlled-outcome
  inspection. Results affected by it are post-hoc corrected evidence.
- DeepSeek is the completed API baseline. OpenAI depends on account access;
  Hugging Face results are a secondary analysis of frozen traces.
- Free Kaggle and Colab accelerators are capacity-limited and not guaranteed.

## Required extensions

Stronger external validity requires a fresh held-out protocol with:

- multiple agent-model families;
- more environments and stateful tools;
- factorial crossing of task, fault, and root-node families;
- native runtime dependency edges where available;
- additional unobstructed baselines;
- independent replication of the corrected canonicalization policy.

These limits are part of the reporting boundary and are not exclusion criteria
for the released runs.
