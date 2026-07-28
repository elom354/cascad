from cascad.real_specificity import specificity_summary


def test_specificity_reports_pair_and_event_denominators() -> None:
    rows = [
        {
            "configuration_id": "a",
            "task_family": "math",
            "clean_pair_false_positive": False,
            "falsely_contaminated_nodes": [],
            "event_threshold_results": [
                {"event_key": "tool", "exceeds_threshold": False}
            ],
        },
        {
            "configuration_id": "a",
            "task_family": "time",
            "clean_pair_false_positive": True,
            "falsely_contaminated_nodes": ["model"],
            "event_threshold_results": [
                {"event_key": "tool", "exceeds_threshold": True}
            ],
        },
    ]
    summary = specificity_summary(rows)
    assert summary["root_accuracy"] == "not_applicable_clean_clean"
    assert summary["global"]["false_positive_pairs"] == 1
    assert summary["global"]["denominator_pairs"] == 2
    assert summary["global"]["rate"] == 0.5
    assert summary["mean_falsely_contaminated_nodes"] == 0.5
    assert summary["by_event_key"][0]["denominator_events"] == 2
