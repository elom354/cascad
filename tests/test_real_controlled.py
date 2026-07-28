from cascad.real_controlled import controlled_summary


def _record(**updates):
    row = {
        "configuration_id": "full",
        "task_family": "multi",
        "fault_family": "numeric",
        "propagation_opportunity": "multi_step",
        "realized_path_length": 3,
        "primary_class": "propagated_to_memory",
        "trajectory_aligned": True,
        "uncalibrated_realized_extension_count": 0,
        "graph_correct": True,
        "realized_propagation": True,
        "absorbed_by_agent": False,
        "propagated_to_memory": True,
        "final_failure": True,
        "successful_recovery": False,
        "observed_depth": 2,
        "subgraph_precision": 1.0,
        "subgraph_recall": 0.5,
        "fault_event_count": 1,
    }
    row.update(updates)
    return row


def test_controlled_summary_retains_recovery_and_repeated_exposure() -> None:
    summary = controlled_summary(
        [
            _record(),
            _record(
                graph_correct=False,
                primary_class="successful_recovery",
                propagated_to_memory=False,
                final_failure=False,
                successful_recovery=True,
                fault_event_count=3,
                trajectory_aligned=False,
                uncalibrated_realized_extension_count=2,
            ),
        ]
    )
    assert summary["global"]["n"] == 2
    assert summary["global"]["graph_root_accuracy"] == 0.5
    assert summary["global"]["repeated_exposure_pairs"] == 1
    assert summary["global"]["successful_recovery_rate"] == 0.5
    assert summary["alignment"]["structurally_divergent_pairs"] == 1
