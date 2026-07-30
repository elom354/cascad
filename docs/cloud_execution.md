# GPU execution on Kaggle, Colab, and Azure

## Purpose

The Hugging Face study is a secondary cross-family attribution baseline. It
reuses the frozen 200 real-agent trace pairs, the same paired observable prompt,
the same candidate nodes, and the same strict parser used for DeepSeek. It does
not retrain any model and does not modify the frozen traces.

Three public Apache-2.0 models are registered:

| Alias | Hub model | Native context | Recommended use |
|---|---|---:|---|
| `qwen3-1.7b` | `Qwen/Qwen3-1.7B` | 32,768 | free T4 notebook |
| `qwen3-4b` | `Qwen/Qwen3-4B` | 32,768 | GPU with at least 24 GiB |
| `mistral-7b` | `mistralai/Mistral-7B-Instruct-v0.3` | 32,768 | 24+ GiB independent-family replication |

The runner resolves `main` to an immutable Hub commit before downloading the
weights. That resolved revision, GPU, quantization, token counts, latency, raw
response, parsed response, and prompt hash are exported for every instance.
Input truncation is forbidden.

Free T4 execution uses the versioned `compact-v2` observable serialization.
Repeated model-request state is represented through inheritance and cumulative
message histories through deltas. Context identical between the clean and
observed traces is stored once. The transformation is exactly reversible and is
covered by round-trip tests; it does not truncate an event or payload.
Because its prompt hash differs from the original `full-v1` DeepSeek calls, the
runner refuses to calculate a cross-provider comparison until DeepSeek has been
rerun with the same serialization.

Qwen runs in non-thinking mode with its model-card sampling recommendation
(`temperature=0.7`, `top_p=0.8`, `top_k=20`, `min_p=0`). A per-prompt seed is
derived from the immutable revision and prompt hash and exported with the
record. Mistral retains deterministic decoding because its registered model
spec does not prescribe Qwen's sampling policy.

## Kaggle

1. Create a notebook and enable a GPU accelerator.
2. Upload `notebooks/cascad_huggingface_cloud.ipynb`.
3. Set `REPO_URL` in the first code cell after the repository is published.
4. Keep `MODEL_ALIAS = "qwen3-1.7b"` for a free T4.
5. Run every cell.
6. Download the generated ZIP before closing the session.
7. Start a fresh GPU session with `MODEL_ALIAS = "mistral-7b"` if disk space is
   insufficient to cache both models.

The models are public. `HF_TOKEN` is optional, but a read token can be stored in
Kaggle Secrets to reduce anonymous-download rate limits. Never paste a token
directly into a committed notebook.

Kaggle documents a weekly GPU quota that varies with demand:
<https://www.kaggle.com/docs/efficient-gpu-usage>.

## Google Colab

The same notebook detects `/content` automatically:

1. open the notebook in Colab;
2. select a GPU runtime;
3. set the public repository URL;
4. keep the default `qwen3-1.7b` model and run one model per session;
5. download the result ZIP or copy it to Drive.

Colab free resources are not guaranteed or unlimited:
<https://research.google.com/colaboratory/faq.html>.

## Azure GPU VM

Azure for Students can be used when free notebook capacity is unavailable.
Create a Linux GPU VM from an image that already includes NVIDIA drivers and
CUDA, clone the repository, and run:

```bash
chmod +x cloud/azure/run_huggingface.sh
cloud/azure/run_huggingface.sh qwen3-4b
cloud/azure/run_huggingface.sh mistral-7b
```

Stop or deallocate the VM immediately after copying the results. A stopped VM
may continue to incur storage charges; a running GPU VM consumes student credit
quickly.

Azure for Students currently advertises a finite credit without requiring a
credit card for eligible students:
<https://azure.microsoft.com/free/students/>.

## Direct command

On any CUDA machine:

```bash
python -m pip install -e ".[huggingface]"
PYTHONPATH=src python scripts/run_huggingface_attribution.py \
  --models qwen3-4b \
  --quantization 4bit \
  --attention-backend sdpa \
  --cache-implementation dynamic \
  --trace-serialization compact-v2
```

Download weights in advance when a compute session has limited network time:

```bash
PYTHONPATH=src python scripts/download_huggingface_models.py \
  --models qwen3-4b
```

The append-only `raw_results.jsonl` checkpoint allows the same command to
resume after interruption. Successfully completed model/revision/instance
triples are never called twice.

Before attribution begins, the runner performs a label-free exact-sentinel
generation check. It then evaluates the largest pending trace pair first. A
broken generation backend or an out-of-memory failure therefore stops the
campaign before spending a full 200-call run. `offloaded` remains available
for diagnostics, but it is not the accepted default: the July 2026 T4 run
produced incoherent generations under that cache mode.

## Result acceptance

A paper-facing run must satisfy all of the following:

- all 200 frozen instances completed;
- `summary.json` reports `study_complete: true`;
- no input truncation;
- one immutable resolved revision per model;
- no missing raw response or prompt hash;
- all errors and invalid parser outputs retained;
- `integrity_manifest.json` regenerated after completion;
- results copied out of the temporary notebook before shutdown.

Runs using `--limit` are smoke tests and must not be reported as final results.
