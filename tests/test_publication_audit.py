import importlib.util
import json
from pathlib import Path


def _module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "audit_publication_readiness.py"
    )
    spec = importlib.util.spec_from_file_location("publication_audit", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_huggingface_audit_rejects_incomplete_invalid_run(
    tmp_path: Path,
) -> None:
    module = _module()
    raw = (
        '{"status":"completed","parse_valid":false}\n'
        '{"status":"error","error":{"type":"OutOfMemoryError"}}\n'
    )
    (tmp_path / "raw_results.jsonl").write_text(raw)
    (tmp_path / "summary.json").write_text(
        json.dumps({"study_complete": False})
    )
    artifacts = {}
    for name in ("raw_results.jsonl", "summary.json"):
        path = tmp_path / name
        artifacts[name] = {
            "sha256": module.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }
    (tmp_path / "integrity_manifest.json").write_text(
        json.dumps({"artifacts": artifacts})
    )

    report = module.audit_huggingface_run(tmp_path)

    assert report["integrity"] == "PASS"
    assert report["acceptance_decision"] == "EXCLUDE_INVALID_RUNTIME"
    assert report["invalid_parse_count"] == 1
