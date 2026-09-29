# ruff: noqa: I001
import copy

import numpy as np
import pytest

import statfuzz.simulation as simulation_module
from statfuzz import stress_test
from statfuzz.dgp import DGPIdentity, LogNormal, MixtureNormal, Normal, StudentT
from statfuzz.methods import welch_ttest_pvalue, welch_ttest_pvalues_batch


BATCH_SIZES = (1, 2, 7, 64, 1_000)


@pytest.mark.parametrize(
    "dgp",
    [
        Normal(),
        LogNormal(sigma=1.0),
        StudentT(df=5.0),
        MixtureNormal(),
    ],
    ids=["normal", "lognormal", "student_t", "mixture_normal"],
)
@pytest.mark.parametrize("n", [10, 37])
def test_stress_result_is_exactly_batch_size_invariant(dgp, n):
    kwargs = {
        "method": "welch_ttest",
        "metric": "type1_error",
        "dgp": dgp,
        "n1": n,
        "n2": n,
        "simulations": 73,
        "seed": 20260929,
        "alpha": 0.05,
        "tolerance": 0.01,
    }

    scalar = stress_test(**kwargs, batch_size=1)

    for batch_size in BATCH_SIZES[1:]:
        batched = stress_test(**kwargs, batch_size=batch_size)
        assert batched == scalar


class _RecordingNormal:
    name = "recording-normal"
    population_mean = 0.0
    identity = DGPIdentity.from_mapping(
        "tests.RecordingNormal",
        {"mean": 0.0, "sd": 1.0},
    )

    def __init__(self):
        self.draws: list[tuple[float, ...]] = []

    def sample(self, rng, n):
        sample = rng.normal(size=n)
        self.draws.append(tuple(float(value) for value in sample))
        return sample


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_batch_size_preserves_logical_sample_draw_sequence(batch_size):
    dgp = _RecordingNormal()

    result = stress_test(
        method="welch_ttest",
        metric="type1_error",
        dgp=dgp,
        n1=5,
        n2=5,
        simulations=11,
        seed=12345,
        batch_size=batch_size,
    )

    reference = _RecordingNormal()
    scalar = stress_test(
        method="welch_ttest",
        metric="type1_error",
        dgp=reference,
        n1=5,
        n2=5,
        simulations=11,
        seed=12345,
        batch_size=1,
    )

    assert result == scalar
    assert dgp.draws == reference.draws
    assert len(dgp.draws) == 22


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_batch_size_preserves_final_numpy_rng_state(batch_size):
    dgp = Normal()
    rng = np.random.default_rng(8472)

    rejections = simulation_module._simulate_rejections(
        dgp=dgp,
        other=dgp,
        n1=13,
        n2=17,
        simulations=41,
        alpha=0.05,
        rng=rng,
        batch_size=batch_size,
    )
    state = copy.deepcopy(rng.bit_generator.state)

    scalar_rng = np.random.default_rng(8472)
    scalar_rejections = simulation_module._simulate_rejections(
        dgp=dgp,
        other=dgp,
        n1=13,
        n2=17,
        simulations=41,
        alpha=0.05,
        rng=scalar_rng,
        batch_size=1,
    )

    assert rejections == scalar_rejections
    assert state == scalar_rng.bit_generator.state


def test_batched_welch_preserves_scalar_rejection_decisions():
    rng = np.random.default_rng(2026)
    xs = [rng.normal(size=11) for _ in range(31)]
    ys = [rng.normal(scale=1.7, size=19) for _ in range(31)]

    scalar = np.asarray(
        [welch_ttest_pvalue(x, y) for x, y in zip(xs, ys)],
        dtype=float,
    )
    batched = welch_ttest_pvalues_batch(xs, ys)

    assert batched.shape == scalar.shape
    assert np.array_equal(batched < 0.05, scalar < 0.05)
    assert np.allclose(batched, scalar, rtol=1e-14, atol=0.0)


def test_batched_welch_preserves_zero_variance_override():
    xs = [
        np.ones(8),
        np.zeros(8),
        np.arange(8, dtype=float),
    ]
    ys = [
        np.ones(8),
        np.ones(8),
        np.arange(8, dtype=float),
    ]

    scalar = np.asarray(
        [welch_ttest_pvalue(x, y) for x, y in zip(xs, ys)],
        dtype=float,
    )
    batched = welch_ttest_pvalues_batch(xs, ys)

    assert np.array_equal(batched, scalar)


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5])
def test_stress_test_rejects_invalid_batch_size(batch_size):
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        stress_test(
            method="welch_ttest",
            metric="type1_error",
            dgp=Normal(),
            simulations=2,
            batch_size=batch_size,
        )
