import copy
from dataclasses import FrozenInstanceError
from itertools import pairwise

import numpy as np
import pytest

import statfuzz.bootstrap_coverage as coverage_module
from statfuzz import (
    BootstrapCoverageProgress,
    bootstrap_mean_coverage,
)
from statfuzz.bootstrap_coverage import _simulate_bootstrap_coverage_batched
from statfuzz.dgp import LogNormal, MixtureNormal, Normal, StudentT
from statfuzz.methods import BootstrapMeanPercentile
from statfuzz.targets import resolve_mean_target

ROOT_SEED = 20260930
SIMULATIONS = 19
METHOD = BootstrapMeanPercentile(resamples=9, interval_level=0.8)
BATCH_SIZES = [1, 2, 7, 64, 1_000]


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


def _run_internal_trace(monkeypatch, base, *, batch_size, with_progress):
    dgp = _RecordingDGP(base)
    target_check = resolve_mean_target(dgp, None)
    rng = np.random.Generator(np.random.PCG64(ROOT_SEED))
    events = []
    progress = []
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
        coverage_count = _simulate_bootstrap_coverage_batched(
            dgp=dgp,
            n=6,
            simulations=SIMULATIONS,
            target_check=target_check,
            method=METHOD,
            root_seed=ROOT_SEED,
            rng=rng,
            batch_size=batch_size,
            progress_callback=progress.append if with_progress else None,
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
    return (
        dgp.samples,
        events,
        coverage_count,
        final_state,
        next_stream,
        progress,
    )


def test_bootstrap_coverage_progress_is_immutable():
    event = BootstrapCoverageProgress(
        completed=4,
        total=10,
        covered=3,
        empirical=0.75,
    )

    with pytest.raises(FrozenInstanceError):
        event.completed = 5


def test_progress_is_emitted_after_each_committed_outer_batch():
    events = []

    result = bootstrap_mean_coverage(
        dgp=Normal(),
        n=6,
        simulations=10,
        method=METHOD,
        seed=ROOT_SEED,
        batch_size=4,
        progress_callback=events.append,
    )

    assert [event.completed for event in events] == [4, 8, 10]
    assert all(event.total == 10 for event in events)
    assert all(
        left.completed < right.completed
        for left, right in pairwise(events)
    )
    assert all(0 <= event.covered <= event.completed for event in events)
    assert all(
        event.empirical == event.covered / event.completed
        for event in events
    )
    assert events[-1].covered == result.coverage_count
    assert events[-1].empirical == result.empirical


@pytest.mark.parametrize("factory", _dgp_factories())
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_progress_callback_preserves_exact_public_result(
    factory,
    batch_size,
):
    kwargs = {
        "dgp": factory(),
        "n": 6,
        "simulations": SIMULATIONS,
        "method": METHOD,
        "tolerance": 0.02,
        "seed": ROOT_SEED,
        "batch_size": batch_size,
        "evidence_confidence_level": 0.9,
    }
    reference = bootstrap_mean_coverage(**kwargs)

    observed = bootstrap_mean_coverage(
        **{**kwargs, "dgp": factory()},
        progress_callback=lambda event: False,
    )

    assert observed == reference


@pytest.mark.parametrize("factory", _dgp_factories())
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_progress_callback_preserves_samples_events_and_outer_rng(
    monkeypatch,
    factory,
    batch_size,
):
    reference = _run_internal_trace(
        monkeypatch,
        factory(),
        batch_size=batch_size,
        with_progress=False,
    )
    observed = _run_internal_trace(
        monkeypatch,
        factory(),
        batch_size=batch_size,
        with_progress=True,
    )

    (
        reference_samples,
        reference_events,
        reference_count,
        reference_state,
        reference_next,
        reference_progress,
    ) = reference
    (
        observed_samples,
        observed_events,
        observed_count,
        observed_state,
        observed_next,
        observed_progress,
    ) = observed

    assert reference_progress == []
    assert len(observed_samples) == len(reference_samples) == SIMULATIONS
    for actual, expected in zip(
        observed_samples,
        reference_samples,
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)

    assert observed_events == reference_events
    assert observed_count == reference_count
    assert observed_state == reference_state
    np.testing.assert_array_equal(observed_next, reference_next)

    assert observed_progress
    assert observed_progress[-1].completed == SIMULATIONS
    assert observed_progress[-1].covered == observed_count


def test_progress_callback_exception_propagates_after_committed_batch(monkeypatch):
    dgp = _RecordingDGP(Normal())
    observed_events = []
    progress = []
    original_event = coverage_module.bootstrap_mean_coverage_event

    def record_event(sample, **kwargs):
        event = original_event(sample, **kwargs)
        observed_events.append(event)
        return event

    def fail(event):
        progress.append(event)
        raise RuntimeError("progress consumer failed")

    monkeypatch.setattr(
        coverage_module,
        "bootstrap_mean_coverage_event",
        record_event,
    )

    with pytest.raises(RuntimeError, match="progress consumer failed"):
        bootstrap_mean_coverage(
            dgp=dgp,
            n=6,
            simulations=10,
            method=METHOD,
            seed=ROOT_SEED,
            batch_size=4,
            progress_callback=fail,
        )

    assert len(dgp.samples) == 4
    assert len(observed_events) == 4
    assert len(progress) == 1
    assert progress[0].completed == 4
    assert progress[0].covered == sum(
        int(event.covered) for event in observed_events
    )


def test_failed_batch_does_not_emit_progress(monkeypatch):
    dgp = _RecordingDGP(Normal())
    progress = []
    original_event = coverage_module.bootstrap_mean_coverage_event

    def fail_on_index_two(sample, **kwargs):
        if kwargs["logical_outer_index"] == 2:
            raise ValueError("synthetic coverage failure")
        return original_event(sample, **kwargs)

    monkeypatch.setattr(
        coverage_module,
        "bootstrap_mean_coverage_event",
        fail_on_index_two,
    )

    with pytest.raises(
        RuntimeError,
        match="simulation 2: bootstrap coverage evaluation failed",
    ):
        bootstrap_mean_coverage(
            dgp=dgp,
            n=6,
            simulations=10,
            method=METHOD,
            seed=ROOT_SEED,
            batch_size=4,
            progress_callback=progress.append,
        )

    assert len(dgp.samples) == 4
    assert progress == []


def test_non_callable_progress_callback_fails_before_first_outer_draw():
    dgp = _RecordingDGP(Normal())

    with pytest.raises(
        TypeError,
        match="progress_callback must be callable or None",
    ):
        bootstrap_mean_coverage(
            dgp=dgp,
            n=6,
            simulations=5,
            method=METHOD,
            seed=ROOT_SEED,
            progress_callback=object(),
        )

    assert dgp.samples == []


@pytest.mark.parametrize(
    ("completed", "total", "covered", "empirical", "error"),
    [
        (0, 10, 0, 0.0, ValueError),
        (11, 10, 0, 0.0, ValueError),
        (4, 10, 5, 1.25, ValueError),
        (4, 10, 3, 0.5, ValueError),
        (4, 10, True, 0.25, TypeError),
        (4, 10, 1, float("nan"), ValueError),
    ],
)
def test_bootstrap_coverage_progress_rejects_inconsistent_state(
    completed,
    total,
    covered,
    empirical,
    error,
):
    with pytest.raises(error):
        BootstrapCoverageProgress(
            completed=completed,
            total=total,
            covered=covered,
            empirical=empirical,
        )
