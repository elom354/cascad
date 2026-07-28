"""Reference agent simulations and scenarios used by Cascad experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol
from uuid import uuid4

from cascad.detectors import RuleBasedDetector
from cascad.injection import FaultInjector
from cascad.intervention import CalibratedInterventionPolicy, GenericInterceptorPolicy
from cascad.metrics import PropagationMetrics, compute_metrics
from cascad.models import EventKind, FaultKind, FaultSpec, RunTrace
from cascad.scenarios import SCENARIOS, Scenario
from cascad.tracer import CascadTracer


class MemoryBackend(Protocol):
    """Minimal persistence contract; MongoDB adapters can implement this later."""

    def read(self, session_id: str) -> list[Any]: ...

    def write(self, session_id: str, value: Any) -> None: ...

    def clear(self, session_id: str) -> None: ...


class InMemoryBackend:
    """Deterministic process-local backend for controlled experiments."""

    def __init__(self) -> None:
        self._data: dict[str, list[Any]] = {}

    def read(self, session_id: str) -> list[Any]:
        return list(self._data.get(session_id, []))

    def write(self, session_id: str, value: Any) -> None:
        self._data.setdefault(session_id, []).append(value)

    def clear(self, session_id: str) -> None:
        """Remove persisted values as a controlled manual remediation."""
        self._data.pop(session_id, None)


@dataclass(frozen=True)
class SimulationResult:
    trace: RunTrace
    metrics: PropagationMetrics
    final_answer: str
    interventions: tuple[str, ...] = ()


class ReActPropagationSimulator:
    """Small ReAct-style simulator supporting controlled multi-episode studies."""

    nodes = SCENARIOS["weather"].nodes

    def __init__(
        self,
        injector: FaultInjector | None = None,
        detector: RuleBasedDetector | None = None,
        memory_backend: MemoryBackend | None = None,
        scenario: str | Scenario = "weather",
        intervention_policy: CalibratedInterventionPolicy | GenericInterceptorPolicy | None = None,
    ) -> None:
        self.injector = injector or FaultInjector()
        self.detector = detector or RuleBasedDetector()
        self.memory_backend = memory_backend or InMemoryBackend()
        self.scenario = SCENARIOS[scenario] if isinstance(scenario, str) else scenario
        self.intervention_policy = intervention_policy

    def run(
        self,
        user_task: str | None = None,
        *,
        seed: int | None = None,
        session_id: str | None = None,
        episode_id: int = 1,
    ) -> SimulationResult:
        """Run one episode; session memory is shared when ``session_id`` repeats."""
        actual_seed = 0 if seed is None else seed
        instance = self.scenario.instantiate(actual_seed)
        session_id = session_id or str(uuid4())
        task = user_task or instance.task
        tracer = CascadTracer(
            task=task,
            scenario=self.scenario.name,
            session_id=session_id,
            episode_id=episode_id,
            instance_id=instance.instance_id,
            instance_seed=instance.instance_seed,
        )
        tracer.trace.session_id, tracer.trace.episode_id = session_id, episode_id
        interventions: list[str] = []

        with tracer.span("input", user_task=task):
            state: dict[str, Any] = {"task": task, "intent": self.scenario.name.split("_", 1)[0]}
            state = self.injector.apply("input", state, tracer)

        with tracer.span("planner"):
            plan = list(self.scenario.nodes[2:])
            plan = self.injector.apply("planner", plan, tracer)
            tracer.emit("planner", "plan_created", {"plan": plan})
            tracer.link("input", "planner", "data_dependency", evidence="task -> plan")

        memory = self.memory_backend.read(session_id)
        memory_seen = False
        workflow_state: dict[str, Any] = {}
        previous = "planner"
        tool_ok = True
        # Weather has an explicit verifier stage below; other scenarios use
        # their declared tools directly.
        execution_nodes = [node for node in self.scenario.nodes[2:-1] if node != "verifier"]
        for node in execution_nodes:
            if node == "memory":
                with tracer.span(node):
                    if self.scenario.distant_symptom:
                        memories = [{instance.memory_field: workflow_state.get("authorization_state", instance.clean_state)}]
                    else:
                        memories = memory or [instance.memory_default]
                    memories = self.injector.apply(node, memories, tracer)
                    tracer.emit(node, EventKind.MEMORY_READ, {"memories": memories})
                    tracer.link(previous, node, "memory_dependency", evidence="state uses memory")
                    decision = self._intervene(node, {"memories": memories}, tracer, interventions)
                    if decision == "quarantine_memory_write":
                        memories = []
                    elif decision == "discard_retrieved_memory":
                        memories = []
                    if memories:
                        self.memory_backend.write(session_id, memories)
                        tracer.emit(node, EventKind.MEMORY_WRITE, {"memories": memories})
                    memory_seen = bool(memories)
                previous = node
                continue

            with tracer.span(node):
                value = dict(instance.tool_values.get(node, {"ok": True, "result": node}))
                if self.scenario.distant_symptom and node == "notify" and workflow_state.get("authorization_state") != instance.clean_state:
                    value = {
                        "ok": False,
                        "notified": False,
                        "error": instance.notification_error,
                        **{key: item for key, item in value.items() if key not in {"ok", "notified"}},
                    }
                # Staged injection changes the arguments just before tool use.
                args = self.injector.apply(f"{node}_args", {"task": task, "memory": memory_seen}, tracer)
                if isinstance(self.intervention_policy, GenericInterceptorPolicy):
                    decision = self.intervention_policy.validate(node, args, {"task": str, "memory": bool})
                    if decision.action != "continue":
                        interventions.append(decision.action)
                        tracer.emit(node, EventKind.INTERVENTION, {"action": decision.action, "reason": decision.reason})
                        value = {"ok": False, "error": "tool arguments blocked"}
                value = self.injector.apply(node, value, tracer)
                if self.scenario.distant_symptom and node == "share":
                    workflow_state["authorization_state"] = value.get("permission") if isinstance(value, dict) else value
                tracer.emit(node, EventKind.TOOL_CALL, {**value, "arguments": args})
                tracer.link(previous, node, "tool_dependency", evidence=f"{previous} guides {node}")
                if isinstance(value, dict) and value.get("ok") is False:
                    tool_ok = False
                self._intervene(node, {**value, "arguments": args}, tracer, interventions)
            previous = node

        if "verifier" in self.scenario.nodes:
            with tracer.span("verifier"):
                verdict = {"ok": tool_ok, "error": None if tool_ok else "tool failed"}
                verdict = self.injector.apply("verifier", verdict, tracer)
                tracer.emit("verifier", "verification", verdict)
                tracer.link(previous, "verifier", "data_dependency", evidence="tool output checked")
                tool_ok = isinstance(verdict, dict) and bool(verdict.get("ok"))
            previous = "verifier"

        with tracer.span("responder"):
            final_answer = instance.success_answer if tool_ok else instance.failure_answer
            final_answer = self.injector.apply("responder", final_answer, tracer)
            decision = self._intervene("responder", {"answer": final_answer}, tracer, interventions)
            if decision == "block_final_action":
                final_answer = "Action blocked by Cascad safety intervention."
            tracer.emit("responder", "final_answer", {"answer": final_answer})
            tracer.link(previous, "responder", "control_dependency", evidence="state -> answer")

        self.detector.attach(tracer.trace)
        metrics = compute_metrics(tracer.trace)
        return SimulationResult(tracer.trace, metrics, str(final_answer), tuple(interventions))

    def run_episodes(self, K: int, user_task: str | None = None, session_id: str | None = None) -> list[SimulationResult]:
        """Run ``K`` sequential episodes against one memory backend/session."""
        session = session_id or str(uuid4())
        return [self.run(user_task, session_id=session, episode_id=index + 1) for index in range(K)]

    def clear_memory(self, session_id: str) -> None:
        """Apply an explicit recovery action between multi-episode runs."""
        self.memory_backend.clear(session_id)

    def _intervene(self, node: str, value: Any, tracer: CascadTracer, actions: list[str]) -> str:
        if not isinstance(self.intervention_policy, CalibratedInterventionPolicy):
            return "continue"
        decision = self.intervention_policy.decide_value(node, value)
        if decision.action != "continue":
            actions.append(decision.action)
            tracer.emit(node, EventKind.INTERVENTION, {"action": decision.action, "reason": decision.reason})
        return decision.action


def default_fault(kind: str = "tool_misresult", target_node: str = "tool") -> FaultSpec:
    """Create a convenient controlled fault for the reference scenarios."""
    if kind in {"memory_poison", "memory_poisoning"}:
        return FaultSpec(target_node=target_node, kind=FaultKind.MEMORY_POISON, payload={"memory": "poisoned upload policy"}, label="memory_poison")
    if kind in {"staged_tool_argument", "staged"}:
        return FaultSpec(target_node=f"{target_node}_args", kind=FaultKind.STAGED_TOOL_ARGUMENT, payload={"value": {"task": 7, "memory": "invalid"}}, label="staged_tool_argument")
    failures = {
        "refund_api": {"ok": False, "eligible": False, "error": "refund eligibility was corrupted"},
        "generate_report": {"ok": False, "error": "report generation failed because document metadata was corrupted"},
        "share": {"ok": False, "error": "cloud sharing permission was corrupted"},
        "tool": {"ok": False, "error": "cloud upload silently failed"},
    }
    return FaultSpec(
        target_node=target_node,
        kind=FaultKind.TOOL_MISRESULT,
        payload={"value": failures.get(target_node, {"ok": False, "error": f"{target_node} returned a corrupted result"})},
        label="tool_misresult",
    )
