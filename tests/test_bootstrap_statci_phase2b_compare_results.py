import math

import pytest

from statfuzz import StatCIResult
from statfuzz.methods import BootstrapMeanPercentile
from statfuzz.statci import (
    BootstrapCoverageComparisonKey,
    RegressionPolicy,
    compare_results,
)

METHOD = BootstrapMeanPercentile(
    resamples=19,
    interval_level=0.8,
)


def _coverage_result(
    *,
    observed,
    mcse,
    seed,
    target=0.8,
    tolerance=0.05,
    n=8,
    simulations=100,
    sd=1.0,
):
    coverage_count = round(observed * simulations)
    assert coverage_count / simulations == observed
    deviation = observed - target
    absolute_deviation = abs(deviation)
    passed = absolute_deviation <= tolerance

    return StatCIResult.from_dict(
        {
            "schema_version": "1.2",
            "property": "coverage",
            "target": target,
            "tolerance": tolerance,
            "observed": observed,
            "deviation": deviation,
            "absolute_deviation": absolute_deviation,
            "passed": passed,
            "status": "PASS" if passed else "FAIL",
            "evidence": {
                "kind": "bootstrap_coverage",
                "method": "bootstrap_mean_percentile",
                "metric": "coverage",
                "dgp": f"Normal(mean=0, sd={sd:g})",
                "dgp_identity": {
                    "family": "statfuzz.dgp.Normal",
                    "parameters": {
                        "mean": 0.0,
                        "sd": sd,
                    },
                },
                "n": n,
                "simulations": simulations,
                "seed": seed,
                "mcse": mcse,
                "target_check": {
                    "kind": "mean",
                    "source": "population_mean",
                    "mean": 0.0,
                    "population_mean": 0.0,
                    "note": None,
                },
                "coverage_count": coverage_count,
                "bootstrap_method": METHOD.as_dict(),
                "evidence_interval": {
                    "level": 0.95,
                    "method": "wilson",
                    "low": max(0.0, observed - 0.1),
                    "high": min(1.0, observed + 0.1),
                },
            },
        }
    )


def test_bootstrap_compare_results_stable_reference_case():
    baseline = _coverage_result(
        observed=0.76,
        mcse=0.01,
        seed=101,
    )
    current = _coverage_result(
        observed=0.84,
        mcse=0.01,
        seed=102,
    )

    comparison = compare_results(baseline, current)

    assert isinstance(comparison.key, BootstrapCoverageComparisonKey)
    assert comparison.key == BootstrapCoverageComparisonKey.from_result(
        baseline
    )
    assert comparison.observed_change == pytest.approx(0.08)
    assert comparison.worsening == pytest.approx(0.0)
    assert comparison.direction == "STABLE"
    assert not comparison.regressed
    assert comparison.status == "PASS"


def test_bootstrap_compare_results_improved_reference_case():
    baseline = _coverage_result(
        observed=0.65,
        mcse=0.01,
        seed=201,
        tolerance=0.20,
    )
    current = _coverage_result(
        observed=0.75,
        mcse=0.01,
        seed=202,
        tolerance=0.20,
    )

    comparison = compare_results(baseline, current)

    assert comparison.worsening == pytest.approx(-0.10)
    assert comparison.direction == "IMPROVED"
    assert not comparison.regressed
    assert comparison.fail_to_pass is False


def test_bootstrap_compare_results_worsened_reference_case():
    baseline = _coverage_result(
        observed=0.78,
        mcse=0.01,
        seed=301,
        tolerance=0.25,
    )
    current = _coverage_result(
        observed=0.60,
        mcse=0.01,
        seed=302,
        tolerance=0.25,
    )

    comparison = compare_results(baseline, current)

    assert comparison.uncertainty_scale == pytest.approx(0.02)
    assert comparison.uncertainty_allowance == pytest.approx(0.04)
    assert comparison.regression_threshold == pytest.approx(0.04)
    assert comparison.worsening == pytest.approx(0.18)
    assert comparison.regressed
    assert comparison.direction == "REGRESSION"
    assert comparison.status == "REGRESSION"


def test_bootstrap_pass_to_fail_is_recorded_without_bypassing_guard():
    baseline = _coverage_result(
        observed=0.78,
        mcse=0.02,
        seed=401,
        tolerance=0.05,
    )
    current = _coverage_result(
        observed=0.74,
        mcse=0.02,
        seed=402,
        tolerance=0.05,
    )

    comparison = compare_results(baseline, current)

    assert baseline.passed
    assert not current.passed
    assert comparison.pass_to_fail
    assert not comparison.fail_to_pass
    assert comparison.worsening == pytest.approx(0.04)
    assert comparison.regression_threshold == pytest.approx(0.08)
    assert not comparison.regressed
    assert comparison.direction == "WITHIN_GUARD"


def test_bootstrap_compare_results_rejects_phase2a_identity_mismatch():
    baseline = _coverage_result(
        observed=0.78,
        mcse=0.01,
        seed=501,
        n=8,
    )
    current = _coverage_result(
        observed=0.78,
        mcse=0.01,
        seed=502,
        n=9,
    )

    baseline_key = BootstrapCoverageComparisonKey.from_result(baseline)
    current_key = BootstrapCoverageComparisonKey.from_result(current)
    assert baseline_key != current_key

    with pytest.raises(ValueError, match="same comparison key"):
        compare_results(baseline, current)


def test_bootstrap_independent_seed_uncertainty_uses_root_sum_square_mcse():
    baseline = _coverage_result(
        observed=0.78,
        mcse=0.03,
        seed=601,
        tolerance=0.20,
    )
    current = _coverage_result(
        observed=0.70,
        mcse=0.04,
        seed=602,
        tolerance=0.20,
    )
    policy = RegressionPolicy(
        uncertainty_mode="independent",
        uncertainty_multiplier=1.0,
    )

    comparison = compare_results(
        baseline,
        current,
        policy=policy,
    )

    assert comparison.uncertainty_scale == pytest.approx(0.05)
    assert comparison.uncertainty_scale == pytest.approx(
        math.sqrt(0.03**2 + 0.04**2)
    )
    assert comparison.uncertainty_allowance == pytest.approx(0.05)
    assert comparison.regression_threshold == pytest.approx(0.05)
    assert comparison.worsening == pytest.approx(0.08)
    assert comparison.regressed


@pytest.mark.parametrize(
    ("baseline_seed", "current_seed"),
    [
        (701, 701),
        (None, 702),
        (703, None),
    ],
)
def test_bootstrap_independent_mode_requires_known_distinct_seeds(
    baseline_seed,
    current_seed,
):
    baseline = _coverage_result(
        observed=0.78,
        mcse=0.03,
        seed=baseline_seed,
        tolerance=0.20,
    )
    current = _coverage_result(
        observed=0.70,
        mcse=0.04,
        seed=current_seed,
        tolerance=0.20,
    )

    with pytest.raises(
        ValueError,
        match="independent uncertainty mode",
    ):
        compare_results(
            baseline,
            current,
            policy=RegressionPolicy(
                uncertainty_mode="independent",
            ),
        )


def test_bootstrap_compare_result_serializes_coverage_key_and_evidence():
    baseline = _coverage_result(
        observed=0.78,
        mcse=0.01,
        seed=801,
    )
    current = _coverage_result(
        observed=0.75,
        mcse=0.01,
        seed=802,
    )

    comparison = compare_results(baseline, current)
    payload = comparison.as_dict()

    assert payload["key"]["kind"] == "bootstrap_coverage"
    assert payload["key"] == BootstrapCoverageComparisonKey.from_result(
        baseline
    ).as_dict()
    assert payload["baseline"]["evidence"]["kind"] == "bootstrap_coverage"
    assert payload["current"]["evidence"]["kind"] == "bootstrap_coverage"
