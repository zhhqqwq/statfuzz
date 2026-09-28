import copy

import numpy as np
import pytest

from statfuzz import simulation, stress_test
from statfuzz.dgp import LogNormal, MixtureNormal, Normal, StudentT
from statfuzz.methods import welch_ttest_pvalue, welch_ttest_pvalues_batch

BATCH_SIZES = (1, 2, 7, 64, 10_000)


@pytest.mark.parametrize(
    "dgp",
    [
        Normal(),
        LogNormal(sigma=1.0),
        StudentT(df=5.0),
        MixtureNormal(),
    ],
)
def test_batch_size_preserves_exact_stress_result(dgp):
    kwargs = {
        "method": "welch_ttest",
        "metric": "type1_error",
        "dgp": dgp,
        "n1": 23,
        "n2": 23,
        "simulations": 137,
        "seed": 20260929,
    }
    reference = stress_test(**kwargs, batch_size=1)

    for batch_size in BATCH_SIZES[1:]:
        assert stress_test(**kwargs, batch_size=batch_size) == reference


def test_batched_welch_pvalues_match_scalar_oracle_exactly():
    rng = np.random.default_rng(42)
    xs = [rng.normal(size=17) for _ in range(31)]
    ys = [rng.normal(scale=2.0, size=13) for _ in range(31)]

    scalar = np.asarray(
        [welch_ttest_pvalue(x, y) for x, y in zip(xs, ys)],
        dtype=float,
    )
    batched = welch_ttest_pvalues_batch(xs, ys)

    np.testing.assert_array_equal(batched, scalar)


def test_batched_welch_preserves_zero_variance_reference_cases():
    xs = [
        np.ones(5),
        np.ones(5),
        np.array([0.0, 1.0, 2.0, 3.0]),
    ]
    ys = [
        np.ones(5),
        np.full(5, 2.0),
        np.array([0.0, 2.0, 4.0, 6.0]),
    ]

    scalar = np.asarray(
        [welch_ttest_pvalue(x, y) for x, y in zip(xs, ys)],
        dtype=float,
    )

    np.testing.assert_array_equal(
        welch_ttest_pvalues_batch(xs, ys),
        scalar,
    )


class _RecordingDGP:
    population_mean = 0.0

    def __init__(self, label, log):
        self.label = label
        self.log = log

    @property
    def name(self):
        return f"Recording({self.label})"

    def sample(self, rng, n):
        sample = rng.normal(size=n)
        self.log.append((self.label, sample.copy()))
        return sample


def _recorded_run(batch_size):
    log = []
    first = _RecordingDGP("x", log)
    second = _RecordingDGP("y", log)
    rng = np.random.default_rng(12345)

    rejections = simulation._simulate_rejections(
        dgp=first,
        other=second,
        n1=5,
        n2=7,
        simulations=19,
        alpha=0.05,
        rng=rng,
        batch_size=batch_size,
    )
    return rejections, log, copy.deepcopy(rng.bit_generator.state)


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_batch_size_preserves_sample_call_order_and_values(batch_size):
    reference_rejections, reference_log, reference_state = _recorded_run(1)
    rejections, log, state = _recorded_run(batch_size)

    assert rejections == reference_rejections
    assert [label for label, _ in log] == [
        item
        for _ in range(19)
        for item in ("x", "y")
    ]
    assert len(log) == len(reference_log)
    for (label, sample), (reference_label, reference_sample) in zip(
        log,
        reference_log,
    ):
        assert label == reference_label
        np.testing.assert_array_equal(sample, reference_sample)
    assert state == reference_state


@pytest.mark.parametrize("batch_size", [0, -1, 1.5, True, None])
def test_stress_test_rejects_invalid_batch_size(batch_size):
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        stress_test(
            method="welch_ttest",
            metric="type1_error",
            dgp=Normal(),
            simulations=2,
            batch_size=batch_size,
        )


class _OverflowOnThirdReplicate:
    name = "overflow-on-third"
    population_mean = 0.0

    def __init__(self):
        self.calls = 0

    def sample(self, rng, n):
        del rng
        self.calls += 1
        if self.calls == 5:
            sample = np.empty(n, dtype=float)
            sample[::2] = 1e308
            sample[1::2] = -1e308
            return sample
        return np.zeros(n, dtype=float)


def test_batched_welch_failure_reports_exact_logical_simulation():
    with pytest.raises(RuntimeError, match=r"simulation 2: Welch evaluation failed"):
        stress_test(
            method="welch_ttest",
            metric="type1_error",
            dgp=_OverflowOnThirdReplicate(),
            n1=6,
            n2=6,
            simulations=7,
            batch_size=7,
        )
