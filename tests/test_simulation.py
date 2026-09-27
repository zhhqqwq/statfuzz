import pytest

from statfuzz import stress_test
from statfuzz.dgp import Normal


def test_stress_test_is_reproducible():
    kwargs = {
        "method": "welch_ttest",
        "metric": "type1_error",
        "dgp": Normal(),
        "n1": 20,
        "simulations": 300,
        "seed": 99,
    }
    a = stress_test(**kwargs)
    b = stress_test(**kwargs)
    assert a.empirical == b.empirical
    assert a.mcse == b.mcse


def test_normal_null_is_reasonable():
    result = stress_test(
        method="welch_ttest",
        metric="type1_error",
        dgp=Normal(),
        n1=30,
        simulations=4000,
        seed=2026,
        tolerance=0.02,
    )
    assert result.empirical == pytest.approx(0.05, abs=0.02)


def test_unsupported_method_fails_loudly():
    with pytest.raises(ValueError, match="supports only"):
        stress_test(method="student_ttest", metric="type1_error", dgp=Normal())
