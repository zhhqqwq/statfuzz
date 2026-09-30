import numpy as np
import pytest

import statfuzz.bootstrap_coverage as coverage_module
from statfuzz import bootstrap_mean_coverage
from statfuzz.bootstrap_coverage import (
    BootstrapCoverageResult,
    _simulate_bootstrap_coverage_batched,
    _simulate_bootstrap_coverage_scalar,
)
from statfuzz.dgp import LogNormal, MixtureNormal, Normal, StudentT
from statfuzz.methods import BootstrapMeanPercentile
from statfuzz.targets import resolve_mean_target

ROOT_SEED = 20260930
SIMULATIONS = 11
METHOD = BootstrapMeanPercentile(resamples=9, interval_level=0.8)
BATCH_SIZES = [1, 2, 7, 64, SIMULATIONS + 5]


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


def _run_internal_trace(monkeypatch, base, *, batch_size=None):
    dgp = _RecordingDGP(base)
    target_check = resolve_mean_target(dgp, None)
    rng = np.random.Generator(np.random.PCG64(ROOT_SEED))
    events = []
    original_event = coverage_module.bootstrap_mean_coverage_event

    def record_event(sample, **kwargs):
        event = original_event(sample, **kwargs)
        events.append(event)
        return event

    monkeypatch.setattr(
        coverage_module,
        "bootstrap_mean_coverage_event",
        record_event,
    )
    try:
        if batch_size is None:
            coverage_count = _simulate_bootstrap_coverage_scalar(
                dgp=dgp,
                n=6,
                simulations=SIMULATIONS,
                target_check=target_check,
                method=METHOD,
                root_seed=ROOT_SEED,
                rng=rng,
            )
        else:
            coverage_count = _simulate_bootstrap_coverage_batched(
                dgp=dgp,
                n=6,
                simulations=SIMULATIONS,
                target_check=target_check,
                method=METHOD,
                root_seed=ROOT_SEED,
                rng=rng,
                batch_size=batch_size,
            )
    finally:
        monkeypatch.setattr(
            coverage_module,
            "bootstrap_mean_coverage_event",
            original_event,
        )

    next_stream = rng.integers(
        0,
        2**31,
        size=20,
        dtype=np.int64,
    )
    return dgp.samples, events, coverage_count, rng.bit_generator.state, next_stream


@pytest.mark.parametrize("factory", _dgp_factories())
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_outer_batching_is_strictly_equivalent_to_scalar_oracle(
    monkeypatch,
    factory,
    batch_size,
):
    reference = _run_internal_trace(
        monkeypatch,
        factory(),
        batch_size=None,
    )
    actual = _run_internal_trace(
        monkeypatch,
        factory(),
        batch_size=batch_size,
    )

    reference_samples, reference_events, reference_count, reference_state, reference_next = (
        reference
    )
    actual_samples, actual_events, actual_count, actual_state, actual_next = actual

    assert len(actual_samples) == len(reference_samples) == SIMULATIONS
    for actual_sample, reference_sample in zip(
        actual_samples,
        reference_samples,
        strict=True,
    ):
        np.testing.assert_array_equal(actual_sample, reference_sample)

    assert actual_events == reference_events
    assert actual_count == reference_count
    assert actual_state == reference_state
    np.testing.assert_array_equal(actual_next, reference_next)


@pytest.mark.parametrize("factory", _dgp_factories())
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_public_batched_result_is_exactly_equal_to_batch_size_one(
    factory,
    batch_size,
):
    reference = bootstrap_mean_coverage(
        dgp=factory(),
        n=6,
        simulations=SIMULATIONS,
        method=METHOD,
        tolerance=0.02,
        seed=ROOT_SEED,
        evidence_confidence_level=0.9,
        batch_size=1,
    )
    actual = bootstrap_mean_coverage(
        dgp=factory(),
        n=6,
        simulations=SIMULATIONS,
        method=METHOD,
        tolerance=0.02,
        seed=ROOT_SEED,
        evidence_confidence_level=0.9,
        batch_size=batch_size,
    )

    assert isinstance(actual, BootstrapCoverageResult)
    assert actual == reference


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5, "2", None])
def test_invalid_batch_size_fails_before_first_outer_draw(batch_size):
    dgp = _RecordingDGP(Normal())

    with pytest.raises((TypeError, ValueError), match="batch_size"):
        bootstrap_mean_coverage(
            dgp=dgp,
            n=6,
            simulations=SIMULATIONS,
            method=METHOD,
            seed=ROOT_SEED,
            batch_size=batch_size,
        )

    assert dgp.samples == []
