#!/usr/bin/env python3
"""Resolve and download the frozen local attribution models."""

from __future__ import annotations

import argparse
import json
import os

from huggingface_hub import snapshot_download

from cascad.huggingface_attribution import (
    DEFAULT_MODEL_ALIASES,
    model_spec,
    resolve_model_revision,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODEL_ALIASES),
    )
    args = parser.parse_args()
    token = os.getenv("HF_TOKEN")
    records = []
    for alias in args.models:
        spec = model_spec(alias)
        revision = resolve_model_revision(spec, token=token)
        path = snapshot_download(
            repo_id=spec.model_id,
            revision=revision,
            token=token,
        )
        records.append(
            {
                "alias": alias,
                "model_id": spec.model_id,
                "requested_revision": spec.requested_revision,
                "resolved_revision": revision,
                "cache_path": path,
            }
        )
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
