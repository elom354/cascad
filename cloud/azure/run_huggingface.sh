#!/usr/bin/env bash
set -euo pipefail

MODEL_ALIAS="${1:-qwen3-4b}"
OUTPUT_DIR="${2:-runs/real-agent-confirmatory-v2-huggingface-${MODEL_ALIAS}}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "No NVIDIA runtime detected. Use an Azure GPU image with CUDA enabled." >&2
  exit 1
fi

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi

.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[huggingface]"
nvidia-smi

PYTHONPATH=src .venv/bin/python scripts/run_huggingface_attribution.py \
  --models "${MODEL_ALIAS}" \
  --quantization 4bit \
  --out "${OUTPUT_DIR}"
