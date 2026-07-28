"""Built-in experimental scenarios and controlled fault conditions."""

from cascad.scenarios.attribution import ATTRIBUTION_SCENARIOS, attribution_fault
from cascad.scenarios.base import Scenario, ScenarioInstance
from cascad.scenarios.standard import STANDARD_SCENARIOS

SCENARIOS = {**STANDARD_SCENARIOS, **ATTRIBUTION_SCENARIOS}

__all__ = ["ATTRIBUTION_SCENARIOS", "SCENARIOS", "STANDARD_SCENARIOS", "Scenario", "ScenarioInstance", "attribution_fault"]
