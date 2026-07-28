"""Audit the actual textual divergence encoder used by this environment."""

from __future__ import annotations

from cascad.divergence import probe_embedding_encoder


def main() -> None:
    model, status = probe_embedding_encoder()
    print(f"encoder_used: {status['encoder_used']}")
    print(f"reason: {status['reason']}")
    if model is None:
        print("WARNING: semantic embedding is unavailable; deterministic lexical fallback is ACTIVE.")
    else:
        print("PASS: all-MiniLM-L6-v2 is loaded and will be used for textual divergence.")


if __name__ == "__main__":
    main()
