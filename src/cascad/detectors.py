"""Error detectors for Cascad traces."""

from __future__ import annotations

from cascad.models import ErrorObservation, EventKind, NodeEvent, RunTrace


class RuleBasedDetector:
    """Detect explicit and propagated errors using event payload rules."""

    name = "rule_based"

    def detect(self, trace: RunTrace) -> list[ErrorObservation]:
        """Return error observations for the trace."""
        observations: list[ErrorObservation] = []
        fault_event_id: str | None = None
        contaminated = False

        for event in trace.events:
            explicit = self._is_explicit_error(event)
            if event.kind == EventKind.FAULT_INJECTED:
                contaminated = True
                fault_event_id = event.event_id
                explicit = True

            if explicit:
                observations.append(
                    ErrorObservation(
                        node_id=event.node_id,
                        event_id=event.event_id,
                        detector=self.name,
                        confidence=1.0,
                        reason=self._reason(event),
                        timestamp=event.timestamp,
                        source_fault_id=fault_event_id,
                    )
                )
                contaminated = True
            elif contaminated and self._carries_contamination(event):
                observations.append(
                    ErrorObservation(
                        node_id=event.node_id,
                        event_id=event.event_id,
                        detector=self.name,
                        confidence=0.65,
                        reason="event depends on a contaminated upstream state",
                        timestamp=event.timestamp,
                        source_fault_id=fault_event_id,
                    )
                )
        return observations

    def attach(self, trace: RunTrace) -> RunTrace:
        """Detect and attach observations to the trace."""
        seen = {(obs.node_id, obs.event_id, obs.detector) for obs in trace.observations}
        for observation in self.detect(trace):
            key = (observation.node_id, observation.event_id, observation.detector)
            if key not in seen:
                trace.add_observation(observation)
        return trace

    def _is_explicit_error(self, event: NodeEvent) -> bool:
        payload = event.payload
        if event.kind == EventKind.ERROR_OBSERVED:
            return True
        if event.kind == EventKind.FAULT_INJECTED:
            return True
        if payload.get("ok") is False:
            return True
        if payload.get("exception_type"):
            return True
        text = str(payload).lower()
        return "__corrupted__" in text or "error" in text or "failed" in text

    def _carries_contamination(self, event: NodeEvent) -> bool:
        text = str(event.payload).lower()
        return any(
            token in text
            for token in ("__corrupted__", "wrong", "missing", "poisoned", "failed")
        )

    def _reason(self, event: NodeEvent) -> str:
        if event.kind == EventKind.FAULT_INJECTED:
            return "controlled fault injection"
        if event.payload.get("exception_type"):
            return f"exception: {event.payload['exception_type']}"
        if event.payload.get("ok") is False:
            return str(event.payload.get("error", "tool returned ok=false"))
        return "payload matched error signature"

