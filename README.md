# Cascad

Cascad is a research framework for detecting, tracing, and measuring error
propagation in tool-using AI agents.

It is built around a standard observability/evaluation architecture:

- instrumentation and tracing;
- controlled fault injection;
- rule-based error detection;
- counterfactual causal graph reconstruction calibrated against natural LLM divergence;
- propagation metrics;
- intervention policy evaluation;
- reproducible simulation runner and exports.

## Why

Knowing that an agent failed is not enough. Cascad is designed to answer:

- When did the error appear?
- Where did it enter the agent graph?
- Which nodes became contaminated?
- How deep, delayed, and broad was the propagation?
- Did memory amplify or attenuate the error?
- Where should an intervention happen?

## Installation

```bash
uv sync --extra dev
```

Or:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Python 3.11 or 3.12 is recommended.

## Quick start

```bash
uv run cascad simulate --scenario weather --fault-kind tool_misresult --target-node tool --out runs/demo
```

Outputs:

- `runs/demo/trace.json`
- `runs/demo/metrics.json`
- `runs/demo/causal_graph.dot`

Open the interactive viewer without installing Graphviz:

```bash
uv run cascad view --runs-dir runs
```

Then visit `http://127.0.0.1:8765`.

## LLM attribution comparison

Cascad has its own local `.env`. It is ignored by Git. Initialize it from the
tracked template and add the DeepSeek key:

```bash
cp .env.example .env
```

```env
DEEPSEEK_API_KEY=your-deepseek-api-key
DEEPSEEK_MODEL=deepseek-v4-flash
```

OpenAI is optional:

```env
OPENAI_API_KEY=your-openai-api-key
OPENAI_MODEL=gpt-5.6-sol
OPENAI_REASONING_EFFORT=none
```

Run the matched comparison:

```bash
uv run cascad compare-attribution \
  --env-file .env \
  --n-repeats 20 \
  --out runs/llm-attribution-comparison
```

DeepSeek always runs. If `OPENAI_API_KEY` is non-empty, OpenAI runs
automatically on the same scenarios, seeds, candidate nodes, prompts, clean
traces, and corrupted traces. The command checkpoints every provider call and
exports raw outputs, token/latency metadata, Wilson intervals, an exact paired
McNemar table, and a pairing audit.

This is a comparison of **attribution baselines**, not a second agent-model
validation. Running the real agent itself with OpenAI requires a separately
versioned protocol and fresh calibration; those runs must not be mixed into the
frozen DeepSeek V2 dataset.

## Local Hugging Face baselines

Cascad includes two public, cross-family attribution baselines:

- `Qwen/Qwen3-4B`;
- `mistralai/Mistral-7B-Instruct-v0.3`.

They run locally with Transformers and 4-bit quantization on a CUDA GPU. The
runner applies the same paired trace prompt and strict parser as the DeepSeek
baseline, resolves every model revision to an immutable Hub commit, forbids
input truncation, and checkpoints every call.

```bash
python -m pip install -e ".[huggingface]"
PYTHONPATH=src python scripts/run_huggingface_attribution.py \
  --models qwen3-4b \
  --quantization 4bit
```

The repository contains a ready-to-run
[Kaggle/Colab notebook](notebooks/cascad_huggingface_cloud.ipynb), an
[Azure launcher](cloud/azure/run_huggingface.sh), and a
[GPU container](Dockerfile.gpu). See
[docs/cloud_execution.md](docs/cloud_execution.md) for the exact procedure.

## Research status

The 200-pair V2 real-agent study is preserved as post-hoc-corrected evidence:
its clock normalization was amended after controlled results were inspected.
It must not be presented as an untouched confirmatory study. The
[publication-readiness audit](docs/publication_readiness.md) records the
supported claim boundary, and the
[prospective V3 design](docs/confirmatory_v3_design.md) addresses sample size,
factorial coverage, difficult attribution, and external validity.

Reproduce the current evidence audit and the prospective sample-size
calculation with:

```bash
PYTHONPATH=src uv run python scripts/audit_publication_readiness.py
PYTHONPATH=src uv run python scripts/plan_real_agent_v3_power.py
```

The V2 controlled split was opened only after calibration and specificity
passed. To reproduce its gated collection from `react-agent`, use:

```bash
cd ../react-agent
PYTHONPATH=src uv run python scripts/run_real_agent_confirmatory_v2.py \
  --stage controlled \
  --env-file ../Cascad/.env \
  --threshold-freeze experiments/real_agent_confirmatory_v2/threshold_freeze.json \
  --confirm-controlled \
  --execute
```

This performs the 400 real-agent executions needed for 200 pairs. It is not
launched merely by adding a key; the explicit
`--confirm-controlled --execute` gate prevents accidental provider cost.

Analyze the completed split and run the checkpointed DeepSeek attribution from
`Cascad` with:

```bash
PYTHONPATH=src .venv/bin/python scripts/analyze_real_agent_controlled_v2.py \
  --attribution deepseek
```

Use `--attribution openai` after the optional OpenAI account is active, or
`--attribution both` when both providers are available. Existing successful
calls are skipped.

## Visual viewer

Cascad includes a dependency-free web viewer. It does not require Graphviz.

```bash
uv run cascad view --runs-dir runs
```

For traces produced by the sibling `react-agent` project:

```bash
uv run cascad view --runs-dir ../react-agent/agent_data/cascad_runs
```

Then open:

```text
http://127.0.0.1:8765
```

The viewer lets you inspect runs, propagation metrics, event timelines, causal
graphs, affected nodes, and per-node observations.

## Research mapping

Cascad directly supports the metrics from the article draft:

- Propagation Depth
- Propagation Delay
- contamination breadth curve $CB(t)$ and final breadth
- memory persistence rate across episodes
- causal graph of affected nodes
- intervention decisions before propagation expands

See [docs/methodology.md](docs/methodology.md).

The versioned study artifacts under `runs/` preserve raw outputs, pairing
audits, uncertainty intervals, and integrity hashes. Reproduction commands and
claim boundaries are documented in
[docs/reproducibility.md](docs/reproducibility.md) and
[docs/limitations.md](docs/limitations.md).

## Python API

```python
from cascad.injection import FaultInjector
from cascad.simulator import ReActPropagationSimulator, default_fault

fault = default_fault(kind="tool_misresult", target_node="tool")
result = ReActPropagationSimulator(FaultInjector([fault])).run()

print(result.metrics.propagation_depth)
print(result.metrics.propagation_breadth)
```

## Architecture

```text
cascad/
├── models.py        # trace, event, fault, observation and edge models
├── tracer.py        # span/event instrumentation
├── injection.py     # controlled fault injection
├── detectors.py     # error detectors
├── divergence.py     # semantic/structured divergence and clean-run baseline
├── causal.py         # counterfactual or temporal causal graph reconstruction
├── metrics.py        # PD, delay, CB(t), memory persistence and variance
├── intervention.py   # calibrated and schema-based containment policies
├── simulator.py      # scenarios and multi-episode memory simulation
├── attribution_baseline.py # LLM single-shot localization baseline
├── staged_injection.py # S1→S2→S3 propagation baseline
├── experiment.py    # repeated-trial experiment aggregation
├── export.py        # JSON and DOT exports
├── viewer.py        # dependency-free visual web viewer
└── cli.py           # command line interface
```

The core framework has no mandatory third-party dependency. Semantic encoders
and local language models are isolated in optional dependency groups so Cascad
can still be embedded into LangGraph, benchmark environments, or custom agent
runners.

## Quality checks

```bash
uv sync --extra dev
uv run ruff check .
uv run pytest -q
```

## License

MIT. See [LICENSE](LICENSE).
