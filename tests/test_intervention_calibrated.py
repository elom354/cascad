from cascad.injection import FaultInjector
from cascad.intervention import CalibratedInterventionPolicy, calibrate_intervention
from cascad.simulator import ReActPropagationSimulator, default_fault


def test_calibrated_intervention_blocks_corrupted_memory_but_not_clean_run() -> None:
    profile = calibrate_intervention(lambda seed=0: ReActPropagationSimulator().run(seed=seed), M=2)
    policy = CalibratedInterventionPolicy(profile)
    clean = ReActPropagationSimulator(intervention_policy=policy).run()
    corrupt = ReActPropagationSimulator(
        FaultInjector([default_fault("memory_poison", "memory")]), intervention_policy=policy
    ).run()
    assert not clean.interventions
    assert corrupt.interventions
