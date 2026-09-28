import pytest

from statfuzz import MeanEqualityNull, StressTestResult, stress_test
from statfuzz.dgp import Normal


class _UnknownMeanDGP:
    name = "unknown-mean"

    def __init__(self):
        self.sample_calls = 0

    def sample(self, rng, n):
        self.sample_calls += 1
        return rng.normal(size=n)


class _KnownMeanDGP:
    name = "known-mean"
    population_mean = 0.0

    def sample(self, rng, n):
        return rng.normal(size=n)


def test_unequal_builtin_means_fail_before_sampling(monkeypatch):
    def forbidden_sample(self, rng, n):
        raise AssertionError("sampling should not start under a false null")

    monkeypatch.setattr(Normal, "sample", forbidden_sample)

    with pytest.raises(ValueError, match="requires equal population means"):
        stress_test(
            method="welch_ttest",
            metric="type1_error",
            dgp=Normal(mean=0.0),
            dgp2=Normal(mean=5.0),
            simulations=5,
        )


def test_custom_dgp_without_population_mean_requires_declaration_before_sampling():
    dgp = _UnknownMeanDGP()

    with pytest.raises(ValueError, match="MeanEqualityNull"):
        stress_test(
            method="welch_ttest",
            metric="type1_error",
            dgp=dgp,
            simulations=5,
        )

    assert dgp.sample_calls == 0


def test_explicit_custom_null_is_recorded():
    dgp = _UnknownMeanDGP()

    result = stress_test(
        method="welch_ttest",
        metric="type1_error",
        dgp=dgp,
        simulations=5,
        seed=17,
        null=MeanEqualityNull(
            mean=0.0,
            note="custom generator is analytically centered",
        ),
    )

    assert result.null_check is not None
    assert result.null_check.source == "declaration"
    assert result.null_check.common_mean == 0.0
    assert result.null_check.group1_population_mean is None
    assert result.null_check.group2_population_mean is None
    assert result.null_check.note == "custom generator is analytically centered"
    assert dgp.sample_calls == 10


def test_partial_population_mean_is_cross_checked_against_declaration():
    result = stress_test(
        method="welch_ttest",
        metric="type1_error",
        dgp=Normal(mean=0.0),
        dgp2=_UnknownMeanDGP(),
        simulations=5,
        null=MeanEqualityNull(mean=0.0, note="group 2 is centered"),
    )

    assert result.null_check is not None
    assert result.null_check.source == "declaration"
    assert result.null_check.group1_population_mean == 0.0
    assert result.null_check.group2_population_mean is None

    with pytest.raises(ValueError, match="group 1 population_mean"):
        stress_test(
            method="welch_ttest",
            metric="type1_error",
            dgp=Normal(mean=0.0),
            dgp2=_UnknownMeanDGP(),
            simulations=1,
            null=MeanEqualityNull(mean=1.0),
        )


def test_verified_builtin_null_records_population_means():
    result = stress_test(
        method="welch_ttest",
        metric="type1_error",
        dgp=Normal(mean=2.0, sd=1.0),
        dgp2=Normal(mean=2.0, sd=3.0),
        simulations=5,
        seed=3,
    )

    assert result.null_check is not None
    assert result.null_check.source == "population_means"
    assert result.null_check.common_mean == 2.0
    assert result.null_check.group1_population_mean == 2.0
    assert result.null_check.group2_population_mean == 2.0


def test_known_custom_population_mean_needs_no_declaration():
    result = stress_test(
        method="welch_ttest",
        metric="type1_error",
        dgp=_KnownMeanDGP(),
        simulations=3,
    )

    assert result.null_check is not None
    assert result.null_check.source == "population_means"


def test_zero_rejections_report_nonzero_wilson_uncertainty(monkeypatch):
    monkeypatch.setattr(
        "statfuzz.simulation.welch_ttest_pvalue",
        lambda x, y: 1.0,
    )

    result = stress_test(
        method="welch_ttest",
        metric="type1_error",
        dgp=Normal(),
        simulations=20,
        seed=11,
        batch_size=1,
    )

    assert result.rejection_count == 0
    assert result.empirical == 0.0
    assert result.mcse == 0.0
    assert result.confidence_level == 0.95
    assert result.interval_method == "wilson"
    assert result.interval_low == 0.0
    assert result.interval_high == pytest.approx(0.16112515805281935)
    assert "95% wilson CI" in str(result)


def test_all_rejections_report_wilson_lower_bound(monkeypatch):
    monkeypatch.setattr(
        "statfuzz.simulation.welch_ttest_pvalue",
        lambda x, y: 0.0,
    )

    result = stress_test(
        method="welch_ttest",
        metric="type1_error",
        dgp=Normal(),
        simulations=20,
        batch_size=1,
    )

    assert result.rejection_count == 20
    assert result.empirical == 1.0
    assert result.mcse == 0.0
    assert result.interval_low == pytest.approx(0.8388748419471806)
    assert result.interval_high == 1.0


def test_confidence_level_controls_reported_interval(monkeypatch):
    monkeypatch.setattr(
        "statfuzz.simulation.welch_ttest_pvalue",
        lambda x, y: 1.0,
    )

    ci95 = stress_test(
        method="welch_ttest",
        metric="type1_error",
        dgp=Normal(),
        simulations=20,
        confidence_level=0.95,
        batch_size=1,
    )
    ci90 = stress_test(
        method="welch_ttest",
        metric="type1_error",
        dgp=Normal(),
        simulations=20,
        confidence_level=0.90,
        batch_size=1,
    )

    assert ci90.interval_high < ci95.interval_high


@pytest.mark.parametrize(
    ("confidence_level", "interval_method", "message"),
    [
        (float("nan"), "wilson", "confidence_level"),
        (1.0, "wilson", "confidence_level"),
        (0.95, "exact", "interval_method"),
    ],
)
def test_interval_configuration_is_explicitly_validated(
    confidence_level,
    interval_method,
    message,
):
    with pytest.raises(ValueError, match=message):
        stress_test(
            method="welch_ttest",
            metric="type1_error",
            dgp=Normal(),
            simulations=1,
            confidence_level=confidence_level,
            interval_method=interval_method,
        )


def test_stress_result_rejects_inconsistent_rejection_count():
    with pytest.raises(ValueError, match="empirical is inconsistent"):
        StressTestResult(
            method="welch_ttest",
            metric="type1_error",
            dgp1="Normal(mean=0, sd=1)",
            dgp2="Normal(mean=0, sd=1)",
            n1=20,
            n2=20,
            simulations=20,
            seed=1,
            nominal=0.05,
            empirical=0.1,
            mcse=0.0,
            tolerance=0.01,
            rejection_count=0,
            confidence_level=0.95,
            interval_method="wilson",
            interval_low=0.0,
            interval_high=0.16112515805281935,
        )
