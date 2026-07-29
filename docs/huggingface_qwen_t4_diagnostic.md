# Qwen3-4B T4 execution diagnostic

## Decision

The directory `cascad-huggingface-qwen3-4b-sdpa-offloaded` is preserved
locally and ignored by Git. Its run is **excluded from scientific comparison**.
It is a failed runtime diagnostic, not evidence that Qwen has zero attribution
accuracy.

## Integrity and outcome

The SHA-256 and byte count of every artifact match the supplied integrity
manifest. The frozen model revision is
`1cfa9a7208912126459214e8b04321603b3df60c`.

| Outcome | Count |
|---|---:|
| attempted instances | 200 |
| completed generation calls | 160 |
| valid parsed outputs | 0 |
| multi-step out-of-memory errors | 40 |

All 160 generated responses consumed the full 32-token allowance and contained
incoherent multilingual token sequences. All 40 multi-step traces failed with
CUDA `OutOfMemoryError` on a T4. Because both generation validity and coverage
failed, neither an accuracy estimate nor a paired comparison is reported.

## Root-cause boundary

The evidence identifies an invalid inference configuration, not a defect in the
strict attribution parser. It also revealed that the run used greedy decoding,
whereas the Qwen3 model card recommends sampling in non-thinking mode and warns
against greedy decoding:
<https://huggingface.co/Qwen/Qwen3-4B#best-practices>.

- relaxing the parser cannot turn incoherent text into a scientific
  prediction;
- KV-cache offloading does not remove the memory required by long prompt
  prefill;
- changing the prompt only for Qwen would break matched-information fairness.

The runner now uses dynamic caching and model-card Qwen sampling by default,
derives and records a stable seed for each prompt, records generation-library
versions in the execution configuration, performs a label-free exact-sentinel
conformance check, and evaluates the largest pending trace first as a capacity
gate. An invalid backend therefore stops before a mass run.

## Next valid execution

Run the corrected notebook in a fresh runtime and a new output directory. Do
not resume the offloaded directory because it belongs to a different execution
configuration. Accept the new result only if:

1. the runtime conformance gate passes;
2. the largest trace completes without truncation;
3. all 200 frozen instances complete;
4. raw invalid outputs remain counted, not silently reparsed;
5. `summary.json` reports `study_complete: true`.

The corrected free-T4 notebook uses the new, reversible `compact-v1`
serialization. It removes cumulative duplication but no observable data.
Results from this prompt version cannot be paired with the old `full-v1`
DeepSeek responses: DeepSeek and every other comparison baseline must be rerun
with the identical serialization and prompt hashes.
