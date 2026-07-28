# GPU execution on Kaggle, Colab, and Azure

## Purpose

The Hugging Face study is a secondary cross-family attribution baseline. It
reuses the frozen 200 real-agent trace pairs, the same paired observable prompt,
the same candidate nodes, and the same strict parser used for DeepSeek. It does
not retrain any model and does not modify the frozen traces.

Two public Apache-2.0 models are registered:

| Alias | Hub model | Native context | Recommended use |
|---|---|---:|---|
| `qwen3-4b` | `Qwen/Qwen3-4B` | 32,768 | first free-GPU run |
| `mistral-7b` | `mistralai/Mistral-7B-Instruct-v0.3` | 32,768 | independent model-family replication |

The runner resolves `main` to an immutable Hub commit before downloading the
weights. That resolved revision, GPU, quantization, token counts, latency, raw
response, parsed response, and prompt hash are exported for every instance.
Input truncation is forbidden.

## Kaggle

1. Create a notebook and enable a GPU accelerator.
2. Upload `notebooks/cascad_huggingface_cloud.ipynb`.
3. Set `REPO_URL` in the first code cell after the repository is published.
4. Keep `MODEL_ALIAS = "qwen3-4b"` for the first run.
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
4. run one model per session;
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
  --quantization 4bit
```

Download weights in advance when a compute session has limited network time:

```bash
PYTHONPATH=src python scripts/download_huggingface_models.py \
  --models qwen3-4b
```

The append-only `raw_results.jsonl` checkpoint allows the same command to
resume after interruption. Successfully completed model/revision/instance
triples are never called twice.

## Result acceptance

A paper-facing run must satisfy all of the following:

- all 200 frozen instances completed;
- no input truncation;
- one immutable resolved revision per model;
- no missing raw response or prompt hash;
- all errors and invalid parser outputs retained;
- `integrity_manifest.json` regenerated after completion;
- results copied out of the temporary notebook before shutdown.

Runs using `--limit` are smoke tests and must not be reported as final results.
