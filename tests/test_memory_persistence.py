from cascad.injection import FaultInjector
from cascad.metrics import memory_persistence_rate
from cascad.simulator import InMemoryBackend, ReActPropagationSimulator, default_fault


def test_memory_poison_persists_across_episodes() -> None:
    backend = InMemoryBackend()
    simulator = ReActPropagationSimulator(
        FaultInjector([default_fault("memory_poison", "memory")]), memory_backend=backend
    )
    first = simulator.run(session_id="shared", episode_id=1)
    clean_followup = ReActPropagationSimulator(memory_backend=backend).run(session_id="shared", episode_id=2)
    assert memory_persistence_rate([first.trace, clean_followup.trace], k_turns=1) == 1.0
