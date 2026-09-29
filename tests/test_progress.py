import copy
from dataclasses import FrozenInstanceError
from itertools import pairwise

import numpy as np
import pytest

from statfuzz import SimulationProgress, simulation, stress_test
from statfuzz.dgp import DGPIdentity, Normal


BATCH_SIZES = (1, 2, 7, 64, 1_000)


def test_simulation_progress_is_immutable():
    event = SimulationProgress(
        completed=4,
        total=10,
        rejections=1,
        empirical=0.25,
    )

    with pytest.raises(FrozenInstanceError):
        event.completed = 5


def test_progress_is_emitted_after_each_committed_logical_batch():
    events = []

    result = stress_test(
        method="welch_ttest",
        metric="type1_error",
        dgp=Normal(),
        n1=11,
        n2=13,
        simulations=10,
        seed=20260930,
        batch_size=4,
        progress_callback=events.append,
    )

    assert [event.completed for event in events] == [4, 8, 10]
    assert all(event.total == 10 for event in events)
    assert all(
        left.completed < right.completed
        for left, right in pairwise(events)
    )
    assert all(0 <= event.rejections <= event.completed for event in events)
    assert all(
        event.empirical == event.rejections / event.completed
        for event in events
    )
    assert events[-1].rejections == result.rejection_count
    assert events[-1].empirical == result.empirical


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_progress_callback_preserves_exact_stress_result(batch_size):
    kwargs = {
        "method": "welch_ttest",
        "metric": "type1_error",
        "dgp": Normal(),
        "n1": 17,
        "n2": 23,
        "simulations": 73,
        "seed": 8472,
        "batch_size": batch_size,
    }
    reference = stress_test(**kwargs)
    events = []

    observed = stress_test(
        **kwargs,
        progress_callback=lambda event: events.append(event) or False,
    )

    assert observed == reference
    assert events[-1].completed == 73
    assert events[-1].rejections == observed.rejection_count


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_progress_callback_preserves_final_numpy_rng_state(batch_size):
    dgp = Normal()
    reference_rng = np.random.default_rng(12345)
    observed_rng = np.random.default_rng(12345)
    events = []

    reference_rejections = simulation._simulate_rejections(
        dgp=dgp,
        other=dgp,
        n1=13,
        n2=19,
        simulations=41,
        alpha=0.05,
        rng=reference_rng,
        batch_size=batch_size,
    )
    observed_rejections = simulation._simulate_rejections(
        dgp=dgp,
        other=dgp,
        n1=13,
        n2=19,
        simulations=41,
        alpha=0.05,
        rng=observed_rng,
        batch_size=batch_size,
        progress_callback=events.append,
    )

    assert observed_rejections == reference_rejections
    assert copy.deepcopy(observed_rng.bit_generator.state) == (
        reference_rng.bit_generator.state
    )
    assert events[-1].completed == 41


class _RecordingNormal:
    name = "recording-normal"
    population_mean = 0.0
    identity = DGPIdentity.from_mapping(
        "tests.ProgressRecordingNormal",
        {"mean": 0.0, "sd": 1.0},
    )

    def __init__(self):
        self.draws = []

    def sample(self, rng, n):
        sample = rng.normal(size=n)
        self.draws.append(sample.copy())
        return sample


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_progress_callback_preserves_logical_sample_sequence(batch_size):
    reference_dgp = _RecordingNormal()
    observed_dgp = _RecordingNormal()

    reference = stress_test(
        method="welch_ttest",
        metric="type1_error",
        dgp=reference_dgp,
        n1=5,
        n2=7,
        simulations=19,
        seed=9876,
        batch_size=batch_size,
    )
    observed = stress_test(
        method="welch_ttest",
        metric="type1_error",
        dgp=observed_dgp,
        n1=5,
        n2=7,
        simulations=19,
        seed=9876,
        batch_size=batch_size,
        progress_callback=lambda event: None,
    )

    assert observed == reference
    assert len(observed_dgp.draws) == len(reference_dgp.draws) == 38
    for actual, expected in zip(observed_dgp.draws, reference_dgp.draws):
        np.testing.assert_array_equal(actual, expected)


def test_progress_callback_exception_propagates_after_committed_batch():
    events = []

    def fail(event):
        events.append(event)
        raise RuntimeError("progress consumer failed")

    with pytest.raises(RuntimeError, match="progress consumer failed"):
        stress_test(
            method="welch_ttest",
            metric="type1_error",
            dgp=Normal(),
            n1=8,
            n2=8,
            simulations=10,
            seed=7,
            batch_size=4,
            progress_callback=fail,
        )

    assert [event.completed for event in events] == [4]


def test_stress_test_rejects_non_callable_progress_callback():
    with pytest.raises(TypeError, match="progress_callback must be callable or None"):
        stress_test(
            method="welch_ttest",
            metric="type1_error",
            dgp=Normal(),
            simulations=2,
            progress_callback=object(),
        )
