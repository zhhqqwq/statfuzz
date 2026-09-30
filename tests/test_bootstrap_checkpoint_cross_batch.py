import copy

import numpy as np
import pytest

import statfuzz.bootstrap_coverage as coverage_module
from statfuzz import bootstrap_mean_coverage
from statfuzz.bootstrap_checkpoint import (
    BootstrapCoverageCheckpoint,
    BootstrapCoverageExperimentSpec,
)
from statfuzz.bootstrap_coverage import (
    _resume_bootstrap_mean_coverage,
    _simulate_bootstrap_coverage_range,
    _simulate_bootstrap_coverage_scalar,
)
from statfuzz.dgp import LogNormal, MixtureNormal, Normal, StudentT
from statfuzz.methods import BootstrapMeanPercentile
from statfuzz.targets import resolve_mean_target

ROOT_SEED = 20260930
SIMULATIONS = 83
METHOD = BootstrapMeanPercentile(resamples=7, interval_level=0.8)
BATCH_TRANSITIONS = [
    pytest.param(1, 7, 10, id="1-to-7"),
    pytest.param(7, 2, 14, id="7-to-2"),
    pytest.param(2, 64, 12, id="2-to-64"),
    pytest.param(64, 1, 64, id="64-to-1"),
]


class _RecordingDGP:
    def __init__(self, base):
        self.base = base
        self.samples = []

    @property
    def name(self):
        return self.base.name

    @property
    def population_mean(self):
        return self.base.population_mean

    @property
    def identity(self):
        return self.base.identity

    def sample(self, rng, n):
        sample = self.base.sample(rng, n)
        self.samples.append(np.asarray(sample, dtype=float).copy())
        return sample


def _dgp_factories():
    return [
        pytest.param(
            lambda: Normal(mean=0.0, sd=1.0),
            id="normal",
        ),
        pytest.param(
            lambda: LogNormal(sigma=0.7, mean=0.0),
            id="lognormal",
        ),
        pytest.param(
            lambda: StudentT(df=5.0, mean=0.0, scale=1.0),
            id="student-t",
        ),
        pytest.param(
            lambda: MixtureNormal(
                weight=0.8,
                mean1=-1.0,
                sd1=1.0,
                mean2=4.0,
                sd2=2.0,
                mean=0.0,
            ),
            id="mixture-normal",
        ),
    ]


def _experiment(dgp):
    return BootstrapCoverageExperimentSpec.from_coverage_config(
        dgp=dgp,
        n=6,
        simulations=SIMULATIONS,
        method=METHOD,
        tolerance=0.02,
        seed=ROOT_SEED,
        evidence_confidence_level=0.9,
        evidence_interval_method="wilson",
    )


def _record_events(monkeypatch, target):
    original_event = coverage_module.bootstrap_mean_coverage_event

    def record_event(sample, **kwargs):
        event = original_event(sample, **kwargs)
        target.append(event)
        return event

    monkeypatch.setattr(
        coverage_module,
        "bootstrap_mean_coverage_event",
        record_event,
    )
    return original_event


def _run_scalar_oracle(monkeypatch, base):
    dgp = _RecordingDGP(base)
    rng = np.random.Generator(np.random.PCG64(ROOT_SEED))
    target_check = resolve_mean_target(dgp, None)
    events = []
    original_event = _record_events(monkeypatch, events)

    try:
        covered = _simulate_bootstrap_coverage_scalar(
            dgp=dgp,
            n=6,
            simulations=SIMULATIONS,
            target_check=target_check,
            method=METHOD,
            root_seed=ROOT_SEED,
            rng=rng,
        )
    finally:
        monkeypatch.setattr(
            coverage_module,
            "bootstrap_mean_coverage_event",
            original_event,
        )

    final_state = copy.deepcopy(rng.bit_generator.state)
    next_stream = rng.integers(
        0,
        2**31,
        size=20,
        dtype=np.int64,
    )
    return dgp.samples, events, covered, final_state, next_stream


def _run_cross_batch_resume(
    monkeypatch,
    base,
    *,
    prefix_batch_size,
    resume_batch_size,
    split,
):
    dgp = _RecordingDGP(base)
    experiment = _experiment(dgp)
    rng = np.random.Generator(np.random.PCG64(ROOT_SEED))
    target_check = resolve_mean_target(dgp, None)
    events = []
    original_event = _record_events(monkeypatch, events)

    try:
        prefix_covered = _simulate_bootstrap_coverage_range(
            dgp=dgp,
            n=6,
            simulations=SIMULATIONS,
            target_check=target_check,
            method=METHOD,
            root_seed=ROOT_SEED,
            rng=rng,
            batch_size=prefix_batch_size,
            start=0,
            stop=split,
            initial_covered=0,
        )

        checkpoint = BootstrapCoverageCheckpoint.create(
            experiment=experiment,
            rng=rng,
            completed=split,
            covered=prefix_covered,
        )

        result, final_rng = _resume_bootstrap_mean_coverage(
            dgp=dgp,
            checkpoint=checkpoint,
            batch_size=resume_batch_size,
        )
    finally:
        monkeypatch.setattr(
            coverage_module,
            "bootstrap_mean_coverage_event",
            original_event,
        )

    final_state = copy.deepcopy(final_rng.bit_generator.state)
    next_stream = final_rng.integers(
        0,
        2**31,
        size=20,
        dtype=np.int64,
    )
    return dgp.samples, events, result, final_state, next_stream, checkpoint


@pytest.mark.parametrize("factory", _dgp_factories())
@pytest.mark.parametrize(
    ("prefix_batch_size", "resume_batch_size", "split"),
    BATCH_TRANSITIONS,
)
def test_cross_batch_resume_is_exactly_equal_to_scalar_oracle(
    monkeypatch,
    factory,
    prefix_batch_size,
    resume_batch_size,
    split,
):
    reference = _run_scalar_oracle(monkeypatch, factory())
    resumed = _run_cross_batch_resume(
        monkeypatch,
        factory(),
        prefix_batch_size=prefix_batch_size,
        resume_batch_size=resume_batch_size,
        split=split,
    )

    (
        reference_samples,
        reference_events,
        reference_covered,
        reference_state,
        reference_next,
    ) = reference
    (
        resumed_samples,
        resumed_events,
        resumed_result,
        resumed_state,
        resumed_next,
        checkpoint,
    ) = resumed

    assert split % prefix_batch_size == 0
    assert checkpoint.state.completed == split
    assert checkpoint.state.covered == sum(
        int(event.covered) for event in resumed_events[:split]
    )

    assert len(resumed_samples) == len(reference_samples) == SIMULATIONS
    for actual, expected in zip(
        resumed_samples,
        reference_samples,
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)

    assert resumed_events == reference_events
    assert resumed_result.coverage_count == reference_covered
    assert resumed_state == reference_state
    np.testing.assert_array_equal(resumed_next, reference_next)

    uninterrupted_result = bootstrap_mean_coverage(
        dgp=factory(),
        n=6,
        simulations=SIMULATIONS,
        method=METHOD,
        tolerance=0.02,
        seed=ROOT_SEED,
        evidence_confidence_level=0.9,
        evidence_interval_method="wilson",
        batch_size=1,
    )
    assert resumed_result == uninterrupted_result


@pytest.mark.parametrize(
    ("prefix_batch_size", "resume_batch_size", "split"),
    BATCH_TRANSITIONS,
)
def test_cross_batch_resume_does_not_change_checkpoint_identity(
    prefix_batch_size,
    resume_batch_size,
    split,
):
    dgp = Normal()
    experiment = _experiment(dgp)
    rng = np.random.Generator(np.random.PCG64(ROOT_SEED))
    target_check = resolve_mean_target(dgp, None)

    covered = _simulate_bootstrap_coverage_range(
        dgp=dgp,
        n=6,
        simulations=SIMULATIONS,
        target_check=target_check,
        method=METHOD,
        root_seed=ROOT_SEED,
        rng=rng,
        batch_size=prefix_batch_size,
        start=0,
        stop=split,
        initial_covered=0,
    )
    checkpoint = BootstrapCoverageCheckpoint.create(
        experiment=experiment,
        rng=rng,
        completed=split,
        covered=covered,
    )

    # Batch scheduling is intentionally absent from experiment/fingerprint
    # identity, so the same checkpoint must validate for any resume batch size.
    result, _ = _resume_bootstrap_mean_coverage(
        dgp=Normal(),
        checkpoint=checkpoint,
        batch_size=resume_batch_size,
    )

    assert result.simulations == SIMULATIONS
    assert result.seed == ROOT_SEED
