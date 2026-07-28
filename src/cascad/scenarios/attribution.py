"""Hard attribution variants designed to separate root cause from symptom."""

from dataclasses import replace

from cascad.models import FaultKind, FaultSpec
from cascad.scenarios.base import Scenario, ScenarioInstance
from cascad.scenarios.standard import STANDARD_SCENARIOS


def _variant(
    base_name: str,
    variant_name: str,
    *,
    distant_symptom: bool = False,
) -> Scenario:
    base = STANDARD_SCENARIOS[base_name]
    return Scenario(
        name=variant_name,
        nodes=base.nodes,
        task=base.task,
        tool_values=base.tool_values,
        memory_default=base.memory_default,
        attribution_variant="distant_symptom" if distant_symptom else "neutral_fault",
        distant_symptom=distant_symptom,
        instance_generator=generate_cloud_distant_instance if distant_symptom else None,
    )


def generate_cloud_distant_instance(seed: int) -> ScenarioInstance:
    """Generate genuinely distinct surfaces while preserving the causal graph."""
    state_pairs = (
        ("viewer", "restricted"),
        ("editor", "read_only"),
        ("approved", "blocked"),
        ("authorized", "limited"),
    )
    task_templates = (
        "process resource {resource} and send the completion notice",
        "complete workflow {resource} and inform the recipient",
        "handle asset {resource} through the delivery workflow",
        "execute the access workflow for item {resource}",
        "finish processing object {resource} and report completion",
    )
    memory_fields = ("authorization_state", "access_state", "entitlement_state", "policy_state")
    notification_errors = (
        "the downstream action could not be completed with the persisted state",
        "the completion step rejected the state recovered from storage",
        "the final operation could not proceed using the recorded state",
        "the recipient operation stopped after reading the stored state",
        "the completion request was declined using the retained state",
    )
    failure_answers = (
        "The workflow could not be completed.",
        "The requested operation did not finish.",
        "Completion could not be confirmed.",
        "The delivery workflow ended without success.",
        "The final operation was not completed.",
    )
    clean_state, _ = state_pairs[seed % len(state_pairs)]
    resource = f"resource-{seed:04d}"
    task = task_templates[seed % len(task_templates)].format(resource=resource)
    correlation = f"corr-{(seed * 7919 + 17) % 100000:05d}"
    tool_values = {
        "upload": {"ok": True, "file_id": resource, "correlation": correlation},
        "share": {"ok": True, "permission": clean_state, "correlation": correlation},
        "notify": {"ok": True, "notified": True, "receipt": f"receipt-{seed:04d}"},
    }
    return ScenarioInstance(
        instance_id=f"cloud-distant-{seed:04d}",
        instance_seed=seed,
        task=task,
        tool_values=tool_values,
        memory_default={memory_fields[seed % len(memory_fields)]: clean_state},
        clean_state=clean_state,
        memory_field=memory_fields[seed % len(memory_fields)],
        notification_error=notification_errors[seed % len(notification_errors)],
        success_answer=f"Workflow {resource} completed.",
        failure_answer=failure_answers[seed % len(failure_answers)],
    )


def generate_cloud_natural_noise_instance(seed: int) -> ScenarioInstance:
    """Preserve the distant-symptom mechanism under a separate study identity."""
    instance = generate_cloud_distant_instance(seed)
    return replace(
        instance,
        instance_id=f"cloud-natural-noise-{seed:04d}",
    )


ATTRIBUTION_SCENARIOS: dict[str, Scenario] = {
    "support_neutral": _variant("support", "support_neutral"),
    "document_neutral": _variant("document", "document_neutral"),
    "cloud_neutral": _variant("cloud", "cloud_neutral"),
    "cloud_distant_symptom": _variant("cloud", "cloud_distant_symptom", distant_symptom=True),
    "cloud_distant_symptom_natural_noise": replace(
        _variant("cloud", "cloud_distant_symptom_natural_noise", distant_symptom=True),
        instance_generator=generate_cloud_natural_noise_instance,
    ),
}


def attribution_fault(scenario_name: str, seed: int | None = None) -> FaultSpec:
    """Return a scenario-consistent fault whose observation does not name its source."""
    conditions = {
        "support_neutral": (
            "refund_api",
            {"ok": False, "eligible": False, "error": "the workflow produced an inconsistent decision state during execution"},
        ),
        "document_neutral": (
            "generate_report",
            {"ok": False, "file": None, "error": "the workflow produced an inconsistent artifact state during execution"},
        ),
        "cloud_neutral": (
            "share",
            {"ok": False, "permission": "unknown", "error": "the workflow produced an inconsistent authorization state during execution"},
        ),
    }
    if scenario_name in {"cloud_distant_symptom", "cloud_distant_symptom_natural_noise"}:
        actual_seed = 0 if seed is None else seed
        generator = (
            generate_cloud_natural_noise_instance
            if scenario_name == "cloud_distant_symptom_natural_noise"
            else generate_cloud_distant_instance
        )
        instance = generator(actual_seed)
        clean_permission = instance.tool_values["share"]["permission"]
        state_pairs = {
            "viewer": "restricted",
            "editor": "read_only",
            "approved": "blocked",
            "authorized": "limited",
        }
        return FaultSpec(
            target_node="share",
            kind=FaultKind.TOOL_MISRESULT,
            payload={"value": {
                **instance.tool_values["share"],
                "permission": state_pairs[clean_permission],
            }},
            seed=actual_seed,
            label="controlled_state_deviation",
        )
    if scenario_name not in conditions:
        raise ValueError(f"no attribution fault defined for scenario {scenario_name!r}")
    target, value = conditions[scenario_name]
    return FaultSpec(
        target_node=target,
        kind=FaultKind.TOOL_MISRESULT,
        payload={"value": value},
        seed=seed,
        label="controlled_state_deviation",
    )
