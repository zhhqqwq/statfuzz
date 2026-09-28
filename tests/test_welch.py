import numpy as np
import pytest
from scipy.stats import ttest_ind

from statfuzz.methods import welch_ttest_pvalue


def test_welch_matches_scipy():
    rng = np.random.default_rng(7)
    x = rng.normal(size=30)
    y = rng.normal(scale=2.0, size=21)

    ours = welch_ttest_pvalue(x, y)
    scipy_p = ttest_ind(x, y, equal_var=False).pvalue
    assert ours == pytest.approx(scipy_p, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize(
    ("x", "y", "message"),
    [
        (
            np.array([0.0, np.nan, 1.0]),
            np.array([0.0, 1.0, 2.0]),
            "finite observations",
        ),
        (
            np.array([0.0, 1.0, 2.0]),
            np.array([0.0, np.inf, 2.0]),
            "finite observations",
        ),
        (
            np.zeros((2, 2)),
            np.array([0.0, 1.0, 2.0]),
            "one-dimensional",
        ),
    ],
)
def test_welch_rejects_invalid_samples(x, y, message):
    with pytest.raises(ValueError, match=message):
        welch_ttest_pvalue(x, y)


def test_welch_rejects_non_finite_intermediate_moments():
    huge = np.array([1e308, 1e308, 1e308])

    with pytest.raises(ValueError, match="mean|variance"):
        welch_ttest_pvalue(huge, huge)
