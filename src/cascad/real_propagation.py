"""Predeclared outcome classification for structurally eligible real-agent faults."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PropagationOutcome:
    """Non-exclusive propagation flags plus one deterministic primary class."""

    primary_class: str
    propagated_to_memory: bool
    propagated_without_memory: bool
    absorbed_by_agent: bool
    detected_and_blocked: bool
    no_persistence_opportunity_realized: bool
    final_failure: bool
    successful_recovery: bool


def classify_propagation(
    *,
    structurally_multi_step: bool,
    persistence_opportunity_realized: bool,
    memory_contaminated: bool,
    downstream_contaminated: bool,
    explicit_detection: bool,
    final_failure: bool,
) -> PropagationOutcome:
    """Classify an injected instance without excluding absorption or recovery."""
    no_opportunity = structurally_multi_step and not persistence_opportunity_realized
    detected = explicit_detection and not memory_contaminated
    to_memory = memory_contaminated
    without_memory = downstream_contaminated and not memory_contaminated
    absorbed = not downstream_contaminated and not final_failure
    recovered = downstream_contaminated and not final_failure
    if no_opportunity:
        primary = "no_persistence_opportunity_realized"
    elif detected:
        primary = "detected_and_blocked"
    elif to_memory:
        primary = "propagated_to_memory"
    elif recovered:
        primary = "successful_recovery"
    elif without_memory:
        primary = "propagated_without_memory"
    else:
        primary = "absorbed_by_agent"
    return PropagationOutcome(
        primary_class=primary,
        propagated_to_memory=to_memory,
        propagated_without_memory=without_memory,
        absorbed_by_agent=absorbed,
        detected_and_blocked=detected,
        no_persistence_opportunity_realized=no_opportunity,
        final_failure=final_failure,
        successful_recovery=recovered,
    )
