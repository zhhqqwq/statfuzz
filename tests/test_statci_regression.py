import json
import math

import pytest

from statfuzz import (
    RegressionPolicy,
    StatCIRegressionError,
    StatCIResult,
    StatCISuiteResult,
    assert_regression_suite,
    compare_results,
    compare_suites,
    render_regression_summary,
    write_regression_summary,
)


def _result(
    *,
    property="type1_error",
    observed=0.05,
    target=0.05,
    tolerance=0.01,
    mcse=0.002,
    seed=42,
    simulations=10_000,
    method="welch_ttest",
    dgp1="Normal(mean=0, sd=1)",
    dgp2="Normal(mean=0, sd=1)",
    n1=20,
    n2=20,
):
    deviation = observed - target
    absolute_deviation = abs(deviation)
    return StatCIResult(
        property=property,
        target=target,
        tolerance=tolerance,
        observed=observed,
        deviation=deviation,
        absolute_deviation=absolute_deviation,
        passed=absolute_deviation <= tolerance,
        method=method,
        metric=property,
        dgp1=dgp1,
        dgp2=dgp2,
        n1=n1,
        n2=n2,
        simulations=simulations,
        seed=seed,
        mcse=mcse,
    )


def test_matching_key_ignores_seed_and_simulation_budget():
    baseline = _result(observed=0.052, seed=1, simulations=5_000)
    current = _result(observed=0.053, seed=999, simulations=50_000)

    comparison = compare_results(baseline, current)

    assert comparison.key.property == "type1_error"
    assert comparison.baseline.seed == 1
    assert comparison.current.seed == 999


def test_conservative_guard_prevents_noise_scale_change_from_being_regression():
    baseline = _result(observed=0.055, mcse=0.002, seed=1)
    current = _result(observed=0.059, mcse=0.002, seed=2)

    comparison = compare_results(baseline, current)

    assert comparison.worsening == pytest.approx(0.004)
    assert comparison.comparison_mcse == pytest.approx(0.004)
    assert comparison.uncertainty_allowance == pytest.approx(0.008)
    assert comparison.regression_threshold == pytest.approx(0.008)
    assert not comparison.regressed
    assert comparison.direction == "WITHIN_GUARD"


def test_large_worsening_is_regression():
    baseline = _result(observed=0.052, mcse=0.002, seed=1)
    current = _result(observed=0.08, mcse=0.002, seed=2)

    comparison = compare_results(baseline, current)

    assert comparison.worsening == pytest.approx(0.028)
    assert comparison.regressed
    assert comparison.status == "REGRESSION"
    assert comparison.direction == "REGRESSION"


def test_change_toward_target_is_improvement_not_regression():
    baseline = _result(observed=0.08, seed=1)
    current = _result(observed=0.06, seed=2)

    comparison = compare_results(baseline, current)

    assert comparison.observed_change == pytest.approx(-0.02)
    assert comparison.worsening == pytest.approx(-0.02)
    assert comparison.direction == "IMPROVED"
    assert not comparison.regressed


def test_pass_to_fail_is_recorded_but_does_not_bypass_uncertainty_guard():
    baseline = _result(observed=0.059, mcse=0.005, seed=1)
    current = _result(observed=0.061, mcse=0.005, seed=2)

    comparison = compare_results(baseline, current)

    assert baseline.passed
    assert not current.passed
    assert comparison.pass_to_fail
    assert comparison.worsening == pytest.approx(0.002)
    assert comparison.regression_threshold == pytest.approx(0.02)
    assert not comparison.regressed
    assert comparison.direction == "WITHIN_GUARD"


def test_fail_to_pass_is_recorded():
    baseline = _result(observed=0.08, seed=1)
    current = _result(observed=0.055, seed=2)

    comparison = compare_results(baseline, current)

    assert comparison.fail_to_pass
    assert comparison.direction == "IMPROVED"


def test_independent_mode_uses_root_sum_square_mcse():
    baseline = _result(observed=0.052, mcse=0.003, seed=1)
    current = _result(observed=0.07, mcse=0.004, seed=2)
    policy = RegressionPolicy(
        uncertainty_mode="independent",
        uncertainty_multiplier=1.0,
    )

    comparison = compare_results(baseline, current, policy=policy)

    assert comparison.comparison_mcse == pytest.approx(0.005)
    assert comparison.uncertainty_allowance == pytest.approx(0.005)


@pytest.mark.parametrize(
    ("baseline_seed", "current_seed"),
    [
        (1, 1),
        (None, 2),
        (1, None),
    ],
)
def test_independent_mode_requires_known_distinct_seeds(
    baseline_seed,
    current_seed,
):
    policy = RegressionPolicy(uncertainty_mode="independent")

    with pytest.raises(ValueError, match="independent uncertainty mode"):
        compare_results(
            _result(seed=baseline_seed),
            _result(seed=current_seed),
            policy=policy,
        )


def test_minimum_worsening_adds_engineering_guard():
    baseline = _result(observed=0.052, mcse=0.0, seed=1)
    current = _result(observed=0.06, mcse=0.0, seed=2)
    policy = RegressionPolicy(
        minimum_worsening=0.01,
        uncertainty_multiplier=0.0,
    )

    comparison = compare_results(baseline, current, policy=policy)

    assert comparison.worsening == pytest.approx(0.008)
    assert not comparison.regressed


def test_compare_results_requires_exact_matching_contract_and_context():
    baseline = _result(n1=20, n2=20)
    current = _result(n1=30, n2=20)

    with pytest.raises(ValueError, match="same comparison key"):
        compare_results(baseline, current)


def test_suite_duplicate_matching_key_is_rejected():
    baseline = StatCISuiteResult.from_results(
        [
            _result(seed=1),
            _result(seed=2, simulations=20_000),
        ],
        name="baseline",
    )
    current = StatCISuiteResult.from_results(
        [_result(seed=3)],
        name="current",
    )

    with pytest.raises(ValueError, match="duplicate comparison key"):
        compare_suites(baseline, current)


def test_strict_suite_matching_rejects_missing_and_new_checks():
    baseline = StatCISuiteResult.from_results(
        [
            _result(property="type1_error", seed=1),
            _result(property="coverage", target=0.95, observed=0.95, seed=2),
        ],
        name="baseline",
    )
    current = StatCISuiteResult.from_results(
        [
            _result(property="type1_error", seed=3),
            _result(property="bias", target=0.0, observed=0.0, seed=4),
        ],
        name="current",
    )

    with pytest.raises(ValueError, match="missing current checks") as exc_info:
        compare_suites(baseline, current)

    message = str(exc_info.value)
    assert "unexpected current checks" in message
    assert "coverage" in message
    assert "bias" in message


def test_non_strict_matching_compares_intersection_and_records_unmatched():
    baseline = StatCISuiteResult.from_results(
        [
            _result(property="type1_error", seed=1),
            _result(property="coverage", target=0.95, observed=0.95, seed=2),
        ],
        name="baseline",
    )
    current = StatCISuiteResult.from_results(
        [
            _result(property="type1_error", observed=0.051, seed=3),
            _result(property="bias", target=0.0, observed=0.0, seed=4),
        ],
        name="current",
    )
    policy = RegressionPolicy(strict_matching=False)

    result = compare_suites(baseline, current, policy=policy)

    assert result.total == 1
    assert len(result.missing_current) == 1
    assert result.missing_current[0].property == "coverage"
    assert len(result.new_current) == 1
    assert result.new_current[0].property == "bias"


def test_regression_suite_aggregates_and_serializes_deterministically(tmp_path):
    baseline = StatCISuiteResult.from_results(
        [
            _result(property="type1_error", observed=0.052, seed=1),
            _result(
                property="coverage",
                target=0.95,
                observed=0.95,
                seed=2,
            ),
        ],
        name="baseline",
    )
    current = StatCISuiteResult.from_results(
        [
            _result(property="type1_error", observed=0.08, seed=3),
            _result(
                property="coverage",
                target=0.95,
                observed=0.951,
                seed=4,
            ),
        ],
        name="current",
    )

    result = compare_suites(baseline, current, name="nightly regression")

    assert not result.passed
    assert result.status == "FAIL"
    assert result.regression_count == 1
    assert result.total == 2

    first = result.to_json()
    second = result.to_json()
    assert first == second

    parsed = json.loads(first)
    assert parsed["name"] == "nightly regression"
    assert parsed["status"] == "FAIL"
    assert parsed["regression_count"] == 1
    assert parsed["policy"]["uncertainty_mode"] == "conservative"

    path = result.write_json(tmp_path / "regression.json")
    assert path.read_text(encoding="utf-8") == first + "\n"


def test_assert_regression_suite_carries_full_result():
    baseline = StatCISuiteResult.from_results(
        [_result(observed=0.052, seed=1)],
        name="baseline",
    )
    current = StatCISuiteResult.from_results(
        [_result(observed=0.08, seed=2)],
        name="current",
    )
    result = compare_suites(baseline, current)

    with pytest.raises(StatCIRegressionError) as exc_info:
        assert_regression_suite(result)

    assert isinstance(exc_info.value, AssertionError)
    assert exc_info.value.result is result
    assert "1/1 matched checks regressed" in str(exc_info.value)


def test_passing_regression_suite_returns_itself():
    baseline = StatCISuiteResult.from_results(
        [_result(observed=0.052, seed=1)],
        name="baseline",
    )
    current = StatCISuiteResult.from_results(
        [_result(observed=0.053, seed=2)],
        name="current",
    )
    result = compare_suites(baseline, current)

    assert assert_regression_suite(result) is result


def test_regression_summary_contains_policy_and_regressions(tmp_path):
    baseline = StatCISuiteResult.from_results(
        [_result(observed=0.052, seed=1)],
        name="baseline",
    )
    current = StatCISuiteResult.from_results(
        [_result(observed=0.08, seed=2)],
        name="current",
    )
    result = compare_suites(baseline, current, name="PR regression")

    summary = render_regression_summary(result)

    assert "## StatCI Regression — PR regression" in summary
    assert "**Overall:** FAIL" in summary
    assert "conservative comparison MCSE" in summary
    assert "### Regressions" in summary
    assert "type1_error" in summary

    path = tmp_path / "summary.md"
    path.write_text("# Existing\n", encoding="utf-8")
    returned = write_regression_summary(result, path)

    assert returned == path
    assert path.read_text(encoding="utf-8").startswith("# Existing\n")


def test_regression_summary_uses_github_environment(tmp_path, monkeypatch):
    baseline = StatCISuiteResult.from_results(
        [_result(observed=0.052, seed=1)],
        name="baseline",
    )
    current = StatCISuiteResult.from_results(
        [_result(observed=0.053, seed=2)],
        name="current",
    )
    result = compare_suites(baseline, current)
    path = tmp_path / "github-summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(path))

    assert write_regression_summary(result) == path
    assert "StatCI Regression" in path.read_text(encoding="utf-8")


def test_policy_validates_configuration():
    with pytest.raises(ValueError, match="minimum_worsening"):
        RegressionPolicy(minimum_worsening=-0.1)
    with pytest.raises(ValueError, match="uncertainty_multiplier"):
        RegressionPolicy(uncertainty_multiplier=-1.0)
    with pytest.raises(ValueError, match="uncertainty_mode"):
        RegressionPolicy(uncertainty_mode="invalid")


def test_independent_formula_is_smaller_than_conservative_for_positive_mcse():
    baseline = _result(observed=0.052, mcse=0.003, seed=1)
    current = _result(observed=0.06, mcse=0.004, seed=2)

    conservative = compare_results(
        baseline,
        current,
        policy=RegressionPolicy(uncertainty_mode="conservative"),
    )
    independent = compare_results(
        baseline,
        current,
        policy=RegressionPolicy(uncertainty_mode="independent"),
    )

    assert conservative.comparison_mcse == pytest.approx(0.007)
    assert independent.comparison_mcse == pytest.approx(math.sqrt(0.000025))
    assert independent.comparison_mcse < conservative.comparison_mcse
