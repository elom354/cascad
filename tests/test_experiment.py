from cascad.experiment import ExperimentCase, run_experiment
from cascad.simulator import default_fault


def test_experiment_summary() -> None:
    summaries = run_experiment(
        [
            ExperimentCase(
                name="tool_failure",
                faults=[default_fault(kind="tool_misresult", target_node="tool")],
                trials=2,
            )
        ]
    )

    assert summaries[0].name == "tool_failure"
    assert summaries[0].trials == 2
    assert summaries[0].mean_breadth >= 1

