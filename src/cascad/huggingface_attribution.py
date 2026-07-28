"""Local Hugging Face attribution baselines for GPU-backed experiments."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from time import perf_counter
from typing import Any, Literal


QuantizationMode = Literal["4bit", "8bit", "none"]


@dataclass(frozen=True)
class HuggingFaceModelSpec:
    """Frozen public configuration for one local attribution model."""

    alias: str
    model_id: str
    requested_revision: str
    context_tokens: int
    disable_thinking: bool = False


MODEL_SPECS: dict[str, HuggingFaceModelSpec] = {
    "qwen3-4b": HuggingFaceModelSpec(
        alias="qwen3-4b",
        model_id="Qwen/Qwen3-4B",
        requested_revision="main",
        context_tokens=32_768,
        disable_thinking=True,
    ),
    "mistral-7b": HuggingFaceModelSpec(
        alias="mistral-7b",
        model_id="mistralai/Mistral-7B-Instruct-v0.3",
        requested_revision="main",
        context_tokens=32_768,
    ),
}

DEFAULT_MODEL_ALIASES = ("qwen3-4b", "mistral-7b")


def model_spec(alias: str) -> HuggingFaceModelSpec:
    """Return a supported model configuration by stable experiment alias."""
    try:
        return MODEL_SPECS[alias]
    except KeyError as exc:
        choices = ", ".join(sorted(MODEL_SPECS))
        raise ValueError(
            f"unknown Hugging Face model {alias!r}; choose {choices}"
        ) from exc


def resolve_model_revision(
    spec: HuggingFaceModelSpec,
    *,
    token: str | None = None,
) -> str:
    """Resolve a mutable Hub reference to the immutable commit downloaded."""
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError(
            "Hugging Face dependencies are missing; install .[huggingface]"
        ) from exc
    info = HfApi(token=token).model_info(
        spec.model_id,
        revision=spec.requested_revision,
    )
    if not info.sha:
        raise RuntimeError(f"the Hub did not return a commit for {spec.model_id}")
    return str(info.sha)


class HuggingFaceAttributor:
    """Callable local model using the same attribution prompt as API baselines."""

    provider = "huggingface-local"
    temperature = 0.0
    configured_max_retries = 0

    def __init__(
        self,
        spec: HuggingFaceModelSpec,
        *,
        resolved_revision: str | None = None,
        quantization: QuantizationMode = "4bit",
        max_new_tokens: int = 32,
        token: str | None = None,
        backend: Any | None = None,
    ) -> None:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if quantization not in {"4bit", "8bit", "none"}:
            raise ValueError(f"unsupported quantization mode: {quantization}")
        self.spec = spec
        self.model = spec.model_id
        self.requested_revision = spec.requested_revision
        self.resolved_revision = resolved_revision
        self.quantization = quantization
        self.max_new_tokens = max_new_tokens
        self.token = token or os.getenv("HF_TOKEN")
        self._backend = backend
        self.last_call_metadata: dict[str, Any] | None = None

    @classmethod
    def from_alias(
        cls,
        alias: str,
        **kwargs: Any,
    ) -> "HuggingFaceAttributor":
        """Create a client from a preregistered model alias."""
        return cls(model_spec(alias), **kwargs)

    def load(self) -> None:
        """Resolve, download, quantize, and load the model once."""
        if self._backend is not None:
            return
        revision = self.resolved_revision or resolve_model_revision(
            self.spec,
            token=self.token,
        )
        self.resolved_revision = revision
        self._backend = _TransformersBackend(
            self.spec,
            resolved_revision=revision,
            quantization=self.quantization,
            max_new_tokens=self.max_new_tokens,
            token=self.token,
        )
        self._backend.load()

    def __call__(self, prompt: str) -> str:
        """Generate exactly one auditable attribution response."""
        self.load()
        started_at = datetime.now(tz=UTC).isoformat()
        started_clock = perf_counter()
        base = {
            "request_started_at": started_at,
            "request_finished_at": None,
            "latency_ms": None,
            "provider": self.provider,
            "requested_model": self.model,
            "requested_revision": self.requested_revision,
            "resolved_revision": self.resolved_revision,
            "quantization": self.quantization,
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "raw_response": None,
            "parsed_response": None,
            "parse_valid": None,
            "wrapper_attempt_count": 1,
            "configured_max_retries": self.configured_max_retries,
            "provider_internal_retry_count": None,
            "prompt_sha256": sha256(prompt.encode("utf-8")).hexdigest(),
            "hardware": None,
            "error": None,
        }
        try:
            raw, generation = self._backend.generate(prompt)
        except Exception as exc:
            self.last_call_metadata = {
                **base,
                "request_finished_at": datetime.now(tz=UTC).isoformat(),
                "latency_ms": (perf_counter() - started_clock) * 1000,
                "error": f"{exc.__class__.__name__}: {exc}",
            }
            raise
        self.last_call_metadata = {
            **base,
            **generation,
            "request_finished_at": datetime.now(tz=UTC).isoformat(),
            "latency_ms": (perf_counter() - started_clock) * 1000,
            "raw_response": raw,
        }
        return raw


class _TransformersBackend:
    """Lazy optional Transformers backend; imported only on GPU execution."""

    def __init__(
        self,
        spec: HuggingFaceModelSpec,
        *,
        resolved_revision: str,
        quantization: QuantizationMode,
        max_new_tokens: int,
        token: str | None,
    ) -> None:
        self.spec = spec
        self.resolved_revision = resolved_revision
        self.quantization = quantization
        self.max_new_tokens = max_new_tokens
        self.token = token
        self.tokenizer: Any | None = None
        self.model: Any | None = None
        self.torch: Any | None = None

    def load(self) -> None:
        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
            )
        except ImportError as exc:
            raise RuntimeError(
                "GPU inference dependencies are missing; install .[huggingface]"
            ) from exc
        if not torch.cuda.is_available() and self.quantization != "none":
            raise RuntimeError(
                "4/8-bit execution requires a CUDA GPU; select a GPU runtime "
                "or use --quantization none only for a small CPU smoke test"
            )
        quantization_config = None
        if self.quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif self.quantization == "8bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        model_kwargs: dict[str, Any] = {
            "revision": self.resolved_revision,
            "token": self.token,
            "low_cpu_mem_usage": True,
        }
        if torch.cuda.is_available():
            model_kwargs.update({"device_map": "auto", "dtype": torch.float16})
        else:
            model_kwargs["dtype"] = torch.float32
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.spec.model_id,
            revision=self.resolved_revision,
            token=self.token,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.spec.model_id,
            **model_kwargs,
        )
        self.model.eval()
        self.torch = torch

    def generate(self, prompt: str) -> tuple[str, dict[str, Any]]:
        if self.model is None or self.tokenizer is None or self.torch is None:
            raise RuntimeError("Hugging Face backend is not loaded")
        messages = [
            {
                "role": "system",
                "content": (
                    "Follow the attribution instruction exactly and return "
                    "only one candidate node_id."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        template_kwargs: dict[str, Any] = {}
        if self.spec.disable_thinking:
            template_kwargs["enable_thinking"] = False
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            **template_kwargs,
        )
        input_tokens = int(inputs["input_ids"].shape[-1])
        maximum = self.spec.context_tokens - self.max_new_tokens
        if input_tokens > maximum:
            raise ValueError(
                f"prompt has {input_tokens} tokens but {self.spec.alias} allows "
                f"at most {maximum}; truncation is forbidden by the protocol"
            )
        device = _input_device(self.model)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=(
                    self.tokenizer.pad_token_id
                    if self.tokenizer.pad_token_id is not None
                    else self.tokenizer.eos_token_id
                ),
            )
        new_tokens = generated[0, input_tokens:]
        raw = self.tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
        ).strip()
        output_tokens = int(new_tokens.shape[-1])
        hardware = (
            self.torch.cuda.get_device_name(0)
            if self.torch.cuda.is_available()
            else "CPU"
        )
        return raw, {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "hardware": hardware,
        }


def _input_device(model: Any) -> Any:
    try:
        return model.get_input_embeddings().weight.device
    except (AttributeError, TypeError):
        return model.device
