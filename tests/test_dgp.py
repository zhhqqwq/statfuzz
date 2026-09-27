import numpy as np
import pytest

from statfuzz.dgp import LogNormal, MixtureNormal, Normal, StudentT


def test_normal_sample_shape():
    x = Normal().sample(np.random.default_rng(0), 10)
    assert x.shape == (10,)


def test_lognormal_is_centered_in_expectation():
    rng = np.random.default_rng(123)
    x = LogNormal(sigma=0.8, mean=2.0).sample(rng, 200_000)
    assert np.mean(x) == pytest.approx(2.0, abs=0.03)


def test_student_t_requires_existing_mean():
    with pytest.raises(ValueError):
        StudentT(df=1.0)


def test_mixture_sample_shape():
    x = MixtureNormal().sample(np.random.default_rng(0), 25)
    assert x.shape == (25,)
