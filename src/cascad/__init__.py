"""Cascad: causal error propagation tracing for AI agents."""

from cascad.detectors import RuleBasedDetector
from cascad.divergence import DivergenceDistribution, estimate_natural_divergence, event_distance
from cascad.injection import FaultInjector
from cascad.metrics import (
    PropagationMetrics,
    compute_metrics,
    contamination_breadth_curve,
    contamination_breadth_final,
    memory_persistence_rate,
    variance_report,
)
from cascad.models import (
    CausalEdge,
    ErrorObservation,
    FaultSpec,
    InterventionDecision,
    NodeEvent,
    RunTrace,
)
from cascad.tracer import CascadTracer

__all__ = [
    "CascadTracer",
    "CausalEdge",
    "DivergenceDistribution",
    "ErrorObservation",
    "FaultInjector",
    "FaultSpec",
    "InterventionDecision",
    "NodeEvent",
    "PropagationMetrics",
    "RuleBasedDetector",
    "RunTrace",
    "contamination_breadth_curve",
    "contamination_breadth_final",
    "compute_metrics",
    "estimate_natural_divergence",
    "event_distance",
    "memory_persistence_rate",
    "variance_report",
]
