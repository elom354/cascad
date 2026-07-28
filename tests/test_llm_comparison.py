import json

from cascad.llm_comparison import run_llm_attribution_comparison


class FakeAttributor:
    temperature = 0.0
    last_call_metadata = None

    def __init__(self, provider: str) -> None:
        self.provider = provider
        self.model = f"{provider}-test-model"

    def __call__(self, prompt: str) -> str:
        for candidate in ("refund_api", "generate_report", "share"):
            if f'"node_id": "{candidate}"' in prompt:
                return candidate
        raise AssertionError("expected controlled root candidate in prompt")


def test_matched_comparison_runs_both_providers_on_identical_inputs(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "cascad.llm_comparison.configured_attributors",
        lambda env_file: (
            FakeAttributor("deepseek"),
            FakeAttributor("openai"),
        ),
    )
    result = run_llm_attribution_comparison(
        tmp_path,
        scenarios=("support_neutral", "cloud_distant_symptom"),
        n_repeats=1,
    )
    summary = json.loads((tmp_path / "summary.json").read_text())
    pairing = json.loads((tmp_path / "pairing_audit.json").read_text())
    assert result["completed_calls"] == 4
    assert summary["configured_providers"] == ["deepseek", "openai"]
    assert summary["paired_provider_statistics"]["pair_count"] == 2
    assert pairing["verdict"] == "PASS"
    assert pairing["matched_provider_pairs"] == 2
