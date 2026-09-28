import numpy as np
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


class _NonFiniteDGP:
    name = "non-finite"
    population_mean = 0.0

    def sample(self, rng, n):
        del rng
        return np.full(n, np.nan)


class _WrongShapeDGP:
    name = "wrong-shape"
    population_mean = 0.0

    def sample(self, rng, n):
        del rng
        return np.zeros((n, 1))


def test_stress_test_rejects_non_finite_generated_samples():
    with pytest.raises(RuntimeError, match="group 1 sample contains non-finite"):
        stress_test(
            method="welch_ttest",
            metric="type1_error",
            dgp=_NonFiniteDGP(),
            simulations=1,
        )


def test_stress_test_rejects_wrong_sample_shape():
    with pytest.raises(RuntimeError, match="sample must have shape"):
        stress_test(
            method="welch_ttest",
            metric="type1_error",
            dgp=_WrongShapeDGP(),
            simulations=1,
        )


def test_stress_test_rejects_invalid_method_pvalue(monkeypatch):
    monkeypatch.setattr(
        "statfuzz.simulation.welch_ttest_pvalue",
        lambda x, y: float("nan"),
    )

    with pytest.raises(RuntimeError, match="invalid p-value"):
        stress_test(
            method="welch_ttest",
            metric="type1_error",
            dgp=Normal(),
            simulations=1,
        )


@pytest.mark.parametrize(
    ("alpha", "tolerance", "message"),
    [
        (float("nan"), 0.01, "alpha must be finite"),
        (float("inf"), 0.01, "alpha must be finite"),
        (0.05, float("nan"), "tolerance must be finite"),
        (0.05, float("inf"), "tolerance must be finite"),
    ],
)
def test_stress_test_rejects_non_finite_configuration(alpha, tolerance, message):
    with pytest.raises(ValueError, match=message):
        stress_test(
            method="welch_ttest",
            metric="type1_error",
            dgp=Normal(),
            alpha=alpha,
            tolerance=tolerance,
            simulations=1,
        )


def test_stress_test_preserves_structured_dgp_identity():
    first = Normal(sd=1.0000001)
    second = Normal(sd=1.0000002)

    result = stress_test(
        method="welch_ttest",
        metric="type1_error",
        dgp=first,
        dgp2=second,
        simulations=2,
        seed=123,
    )

    assert result.dgp1 == result.dgp2 == "Normal(mean=0, sd=1)"
    assert result.dgp1_identity == first.identity
    assert result.dgp2_identity == second.identity
    assert result.dgp1_identity != result.dgp2_identity
    assert result.null_check is not None
    assert result.null_check.source == "population_means"
    assert result.null_check.common_mean == 0.0
    assert result.rejection_count is not None
    assert result.rejection_count == round(result.empirical * result.simulations)
