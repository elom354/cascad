"""Original reference scenarios retained for backward compatibility."""

from cascad.scenarios.base import Scenario


STANDARD_SCENARIOS: dict[str, Scenario] = {
    "weather": Scenario(
        "weather", ("input", "planner", "memory", "tool", "verifier", "responder"),
        "send a weather PDF", {"tool": {"ok": True, "file": "weather.pdf", "cloud_url": "b2://file"}},
    ),
    "support": Scenario(
        "support", ("input", "planner", "refund_api", "memory", "responder"),
        "process a customer refund", {"refund_api": {"ok": True, "eligible": True, "refund_id": "rf-001"}},
    ),
    "document": Scenario(
        "document", ("input", "planner", "retrieve_doc", "summarize", "generate_report", "responder"),
        "generate a documented report", {
            "retrieve_doc": {"ok": True, "document_id": "doc-001", "author": "verified"},
            "summarize": {"ok": True, "summary": "source summary"},
            "generate_report": {"ok": True, "file": "report.pdf"},
        },
    ),
    "cloud": Scenario(
        "cloud", ("input", "planner", "upload", "share", "memory", "notify", "responder"),
        "upload, share and notify", {
            "upload": {"ok": True, "file_id": "cloud-001"},
            "share": {"ok": True, "permission": "viewer"},
            "notify": {"ok": True, "notified": True},
        },
        memory_default="permission=viewer",
    ),
}
