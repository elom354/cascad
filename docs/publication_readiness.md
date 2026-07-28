# Publication-readiness audit

## Verdict

As of 28 July 2026, Cascad is a reproducible research prototype with useful
V2 evidence. It is **not yet ready for broad confirmatory claims** such as
general superiority over LLM diagnosis or validation across autonomous-agent
architectures. No checklist can guarantee acceptance; the venue and reviewers
make that decision.

The current defensible paper can present the framework, simulator studies,
real-agent feasibility evidence, and explicit limitations. A stronger
confirmatory paper requires the fresh V3 study in
`docs/confirmatory_v3_design.md`.

## Evidence audit

| Area | Current status | Required action |
|---|---|---|
| intervention ground truth | strong | keep roots frozen before execution |
| matched counterfactual design | strong | retain all paired and failed-attempt evidence |
| natural divergence calibration | implemented | independently confirm the post-hoc clock rule in V3 |
| held-out specificity | 60 pairs, one false-positive pair | expand across V3 event/root roles |
| V2 controlled sample | 200 distinct pairs | do not call it underpowered without qualification; its diversity is the larger problem |
| root diversity | two injected nodes | cross three causal roles and three domains in V3 |
| multi-step coverage | 40/200 | increase balanced silent/distant/memory cases |
| agent-model diversity | one family | add a separately frozen second-family replication when feasible |
| DeepSeek baseline | 200/200 complete, 95% root accuracy | report honestly; it shows V2 is relatively easy |
| OpenAI baseline | unavailable in V2 | report missing account access, never impute results |
| Qwen T4 baseline | invalid runtime | exclude; rerun with the corrected capacity gates |
| statistical inference | Wilson intervals and exact paired McNemar | freeze multiplicity and subgroup policy prospectively |
| raw reproducibility | traces, hashes, lockfile, CI | create an archival release and DOI for submission |

The executable audit is:

```bash
PYTHONPATH=src uv run python scripts/audit_publication_readiness.py \
  --hf-results cascad-huggingface-qwen3-4b-sdpa-offloaded
```

## Main anticipated reviewer criticisms

1. **“The benchmark is too narrow or easy.”** DeepSeek reaches 95% on the
   real V2 paired traces, and the simulator paired condition reaches 100%.
   Resolution: keep these findings, narrow the claim, and run V3 with silent
   upstream corruption, distant symptoms, distractors, more root roles, and
   balanced task domains.

2. **“Two hundred runs are pseudo-replication.”** They are 200 distinct paired
   instances, not 200 calls on one trace, but they share a small number of task
   structures. Resolution: report the experimental unit precisely and use 648
   distinct, balanced V3 pairs from the prospective power/precision plan.

3. **“The method benefits from more information than the LLM.”** The existing
   ablation already shows that paired DeepSeek can match Cascad on the simple
   distant-symptom simulator. Resolution: separate information-access claims
   from algorithmic claims and verify identical prompt/trace hashes for matched
   baselines.

4. **“Ground truth is circular.”** The injected root is intervention-defined,
   not assigned by Cascad or a human annotator. Resolution: state that accuracy
   measures recovery of a known intervention; do not claim agreement with
   natural-world human causal judgments.

5. **“Post-hoc rules inflate performance.”** The V2 clock rule was amended
   after outcome inspection. Resolution: label V2 accordingly and test the
   frozen rule on untouched V3 data.

6. **“The perfect Cascad score indicates leakage.”** Existing prompts remove
   evaluator-only injection metadata and leakage audits exist. Resolution:
   retain automated prompt audits, publish exact prompt hashes, and add negative
   controls/distractor faults in V3.

7. **“Local-model comparison is invalid.”** The supplied Qwen run is incomplete
   and incoherent. Resolution: exclude it rather than counting it as 0%, apply
   label-free runtime conformance and capacity gates, and require full coverage.

8. **“Results may not reproduce.”** Model APIs, GPU kernels, and mutable model
   revisions can change. Resolution: retain provider-returned model IDs, raw
   outputs, immutable Hugging Face commits, package versions, hardware, prompt
   hashes, seeds, lockfile, and integrity manifests.

## Publication ethics and repository release

IEEE encourages detailed methods plus shared code and data in accessible
repositories:
<https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/research-reproducibility/>.
Before submission, create a versioned release, archive it with a DOI (for
example Zenodo), add `CITATION.cff` with the real authors/title, and state which
artifacts cannot be redistributed.

IEEE currently requires disclosure of AI-generated content, including code,
text, figures, and images, in the acknowledgments, identifying the system,
affected sections, and level of use:
<https://open.ieee.org/author-guidelines-for-artificial-intelligence-ai-generated-text/>.
Removing development notes or rewriting Git history does not remove that
obligation. The authors remain responsible for verifying every claim, result,
reference, and line of released code.

No human annotation is used for the intervention-defined root labels. If a
future study recruits human annotators, IEEE policy requires an IRB/ethics
statement (or an explanation) and consent reporting:
<https://journals.ieeeauthorcenter.ieee.org/become-an-ieee-journal-author/publishing-ethics/guidelines-and-policies/submission-and-peer-review-policies/>.

The local `.env` and result export directories are ignored by Git. Before a
public release, run a dedicated secret scanner over the full Git history,
dependency/license scanning, the complete test suite, and a clean-room
reproduction from the tagged archive.

## Stop conditions

Do not label the next results “confirmatory” until:

- the V3 protocol, seeds, sample size, thresholds, and analysis policy are
  frozen before execution;
- every declared cell is feasible and has unique instances;
- the corrected clock policy is untouched;
- the paired-information and leakage audits pass;
- local-model runtime preflight passes before mass execution;
- missing/failed calls are retained under the frozen policy;
- the exact release environment can reproduce the exported tables.
