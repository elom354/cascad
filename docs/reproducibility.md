# Reproducibility

## Environment

Cascad targets Python 3.11 and 3.12. Create the core development environment
with:

```bash
uv sync --extra dev
```

Install the semantic encoder separately when reproducing calibrated
counterfactual analyses:

```bash
uv sync --extra semantic
PYTHONPATH=src uv run python scripts/check_encoder.py
```

The check must report `encoder_used: embedding`. A fallback encoder result is
not interchangeable with the reported embedding-based experiments.

## Simulator studies

The versioned reports provide the exact commands for the paired attribution,
natural-noise, branched-dependency, and inter-episode studies:

- [paired attribution](paired_attribution_study.md);
- [natural noise](natural_noise_study.md);
- [branched dependency](branched_dependency_study.md);
- [inter-episode persistence](inter_episode_persistence_study.md).

Raw records and statistical exports are stored in the corresponding
directories under `runs/`.

## Real-agent controlled study

The frozen real-agent datasets are:

```text
runs/real-agent-confirmatory-v2-calibration/
runs/real-agent-confirmatory-v2-specificity/
runs/real-agent-confirmatory-v2-controlled/
```

Reproduce their analyses without making new provider calls:

```bash
PYTHONPATH=src uv run python scripts/analyze_real_agent_calibration_v2.py
PYTHONPATH=src uv run python scripts/analyze_real_agent_specificity_v2.py
PYTHONPATH=src uv run python scripts/analyze_real_agent_controlled_v2.py
```

The controlled records retain all preregistered pairs, including absorbed,
blocked, recovered, structurally divergent, and repeated-exposure executions.
Integrity manifests contain SHA-256 hashes for the frozen artifacts.

## Attribution baselines

DeepSeek and optional OpenAI calls require a local, ignored `.env`:

```bash
cp .env.example .env
```

Local Hugging Face baselines require a CUDA runtime:

```bash
python -m pip install -e ".[huggingface]"
PYTHONPATH=src python scripts/run_huggingface_attribution.py \
  --models qwen3-4b \
  --quantization 4bit
```

Run one model at a time on a free notebook GPU. The runner is append-only and
resumes completed model/revision/instance triples. A paper-facing result must
contain all 200 controlled instances; `--limit` is only a smoke test.

## Ground truth

Controlled root-cause accuracy uses the node selected by the fault-injection
protocol before execution. This intervention-defined label does not require a
human annotator. No claim about agreement with human causal judgments is made.

## Protocol amendment

The context-aware clock canonicalization used in the final V2 analysis was
introduced after inspection of the controlled outcomes. Raw traces and
injection labels were not changed, but estimates that depend on this rule must
be treated as post-hoc corrected evidence rather than an untouched
confirmatory result. A new held-out protocol version is required before making
a fully confirmatory claim about that normalization policy.
