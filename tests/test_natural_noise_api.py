import json

from cascad.natural_noise_api import (
    complete_frozen_deepseek_baselines,
    load_and_verify_frozen_study,
)


def test_frozen_study_hashes_and_context_verify() -> None:
    frozen = load_and_verify_frozen_study()
    assert len(frozen.calibration_pairs) == 24
    assert len(frozen.evaluation_pairs) == 80
    assert frozen.verification["trace_hashes_match"]
    assert frozen.verification["calibration_context_matches"]


def test_phase_a_mock_exports_all_equal_information_rows(tmp_path) -> None:
    result = complete_frozen_deepseek_baselines(
        out_dir=tmp_path,
        attributor=lambda _: "share",
    )
    assert result["api_calls"] == 160
    assert result["unique_prompt_count"] == 160
    records = [
        json.loads(line)
        for line in (tmp_path / "raw_attribution.jsonl").read_text().splitlines()
    ]
    assert len(records) == 160
    assert {item["prediction"] for item in records} == {"share"}
    assert all("fault_injected" not in item["prompt"].lower() for item in records)
