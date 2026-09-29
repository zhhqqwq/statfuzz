import numpy as np
import pytest

from statfuzz.methods import (
    BootstrapMeanPercentile,
    bootstrap_mean_percentile_interval,
)


def test_fixed_rng_scalar_reference_interval_is_stable():
    sample = np.array([1.0, 2.0, 4.0, 8.0])
    method = BootstrapMeanPercentile(
        resamples=7,
        interval_level=0.8,
    )
    rng = np.random.Generator(np.random.PCG64(12345))

    interval = bootstrap_mean_percentile_interval(sample, rng, method)

    assert interval == (2.5, 5.1)


def test_scalar_oracle_matches_explicit_replicate_draw_order_and_rng_consumption():
    sample = np.array([1.0, 2.0, 4.0, 8.0])
    method = BootstrapMeanPercentile(
        resamples=7,
        interval_level=0.8,
    )

    reference_rng = np.random.Generator(np.random.PCG64(12345))
    expected_indices = [
        [2, 0, 3, 1],
        [0, 3, 2, 2],
        [3, 1, 3, 1],
        [2, 2, 0, 0],
        [0, 2, 2, 3],
        [2, 0, 3, 3],
        [2, 2, 0, 0],
    ]
    expected_means = [3.75, 4.25, 5.0, 2.5, 4.25, 5.25, 2.5]

    actual_indices = []
    actual_means = []
    for _ in range(method.resamples):
        indices = reference_rng.integers(0, sample.size, size=sample.size)
        actual_indices.append(indices.tolist())
        actual_means.append(float(np.mean(sample[indices])))

    assert actual_indices == expected_indices
    assert actual_means == expected_means

    tail = (1.0 - method.interval_level) / 2.0
    expected_interval = tuple(
        float(value)
        for value in np.quantile(
            np.asarray(expected_means),
            [tail, 1.0 - tail],
            method="linear",
        )
    )

    oracle_rng = np.random.Generator(np.random.PCG64(12345))
    actual_interval = bootstrap_mean_percentile_interval(
        sample,
        oracle_rng,
        method,
    )

    assert actual_interval == expected_interval
    assert oracle_rng.bit_generator.state == reference_rng.bit_generator.state

    np.testing.assert_array_equal(
        oracle_rng.integers(0, 2**31, size=20, dtype=np.int64),
        reference_rng.integers(0, 2**31, size=20, dtype=np.int64),
    )


def test_scalar_oracle_does_not_mutate_input_sample():
    sample = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
    before = sample.copy()

    bootstrap_mean_percentile_interval(
        sample,
        np.random.Generator(np.random.PCG64(7)),
        BootstrapMeanPercentile(resamples=11, interval_level=0.9),
    )

    np.testing.assert_array_equal(sample, before)


def test_constant_sample_returns_exact_degenerate_interval():
    interval = bootstrap_mean_percentile_interval(
        np.array([4.5, 4.5, 4.5]),
        np.random.Generator(np.random.PCG64(123)),
        BootstrapMeanPercentile(resamples=17, interval_level=0.95),
    )

    assert interval == (4.5, 4.5)


def test_single_observation_sample_is_supported():
    interval = bootstrap_mean_percentile_interval(
        np.array([2.25]),
        np.random.Generator(np.random.PCG64(123)),
        BootstrapMeanPercentile(resamples=5, interval_level=0.8),
    )

    assert interval == (2.25, 2.25)


@pytest.mark.parametrize(
    "sample",
    [
        np.array([]),
        np.array([[1.0, 2.0]]),
        np.array([1.0, np.nan]),
        np.array([1.0, np.inf]),
        np.array([1.0, -np.inf]),
    ],
)
def test_scalar_oracle_rejects_invalid_samples(sample):
    with pytest.raises(ValueError):
        bootstrap_mean_percentile_interval(
            sample,
            np.random.Generator(np.random.PCG64(1)),
            BootstrapMeanPercentile(resamples=5),
        )


def test_scalar_oracle_rejects_non_finite_bootstrap_mean():
    sample = np.array([1e308, 1e308])

    with pytest.raises(ValueError, match="replicate 0 mean is non-finite"):
        bootstrap_mean_percentile_interval(
            sample,
            np.random.Generator(np.random.PCG64(1)),
            BootstrapMeanPercentile(resamples=5),
        )


def test_scalar_oracle_requires_numpy_generator():
    with pytest.raises(TypeError, match="numpy.random.Generator"):
        bootstrap_mean_percentile_interval(
            np.array([1.0, 2.0]),
            object(),
            BootstrapMeanPercentile(resamples=5),
        )


def test_scalar_oracle_requires_bootstrap_method_config():
    with pytest.raises(TypeError, match="BootstrapMeanPercentile"):
        bootstrap_mean_percentile_interval(
            np.array([1.0, 2.0]),
            np.random.Generator(np.random.PCG64(1)),
            object(),
        )
