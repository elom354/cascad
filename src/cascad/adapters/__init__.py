"""Adapters for traces emitted by external, non-simulated agents."""

from cascad.adapters.react_agent import (
    RealAgentPairResult,
    analyze_real_agent_pair,
    canonical_trace_sha256,
    load_react_agent_trace,
)

__all__ = [
    "RealAgentPairResult",
    "analyze_real_agent_pair",
    "canonical_trace_sha256",
    "load_react_agent_trace",
]
