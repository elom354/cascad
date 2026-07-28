from cascad.divergence import event_distance
from cascad.models import NodeEvent


def test_divergence_is_zero_for_equal_values_and_positive_after_corruption() -> None:
    clean = NodeEvent("planner", "reasoning", "run", payload={"text": "send the report"})
    same = NodeEvent("planner", "reasoning", "run", payload={"text": "send the report"})
    corrupt = NodeEvent("planner", "reasoning", "run", payload={"text": "discard the report"})

    assert event_distance(clean, same) == 0.0
    assert event_distance(clean, corrupt) > 0.0
