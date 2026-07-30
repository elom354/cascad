import importlib.util
import json
from hashlib import sha256
from pathlib import Path


def _module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "reparse_huggingface_results.py"
    )
    spec = importlib.util.spec_from_file_location("hf_reparse", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_source_integrity_verification(tmp_path: Path) -> None:
    module = _module()
    raw = tmp_path / "raw_results.jsonl"
    raw.write_text('{"status":"completed"}\n')
    manifest = {
        "artifacts": {
            raw.name: {
                "sha256": sha256(raw.read_bytes()).hexdigest(),
                "bytes": raw.stat().st_size,
            }
        }
    }
    (tmp_path / "integrity_manifest.json").write_text(
        json.dumps(manifest)
    )

    assert module.verify_integrity(tmp_path) == "PASS"
