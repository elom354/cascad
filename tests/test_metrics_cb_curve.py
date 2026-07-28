from cascad.injection import FaultInjector
from cascad.metrics import contamination_breadth_curve, compute_metrics
from cascad.simulator import ReActPropagationSimulator, default_fault


def test_contamination_breadth_curve_is_monotonic() -> None:
    result = ReActPropagationSimulator(FaultInjector([default_fault()])).run()
    curve = contamination_breadth_curve(result.trace)
    assert curve
    assert all(next_value >= value for (_, value), (_, next_value) in zip(curve, curve[1:]))
    assert compute_metrics(result.trace).contamination_breadth_curve == tuple(curve)
