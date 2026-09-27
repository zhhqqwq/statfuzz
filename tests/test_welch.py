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
