import pytest

from statfuzz import StatCIResult, StatCISuiteResult
from statfuzz.methods import BootstrapMeanPercentile
from statfuzz.statci import (
    BootstrapCoverageComparisonKey,
    RegressionPolicy,
    StatCIComparisonKey,
    compare_suites,
)

METHOD = BootstrapMeanPercentile(
    resamples=19,
    interval_level=0.8,
)


def _coverage_result(
    *,
    observed,
    seed,
    n=8,
    sd=1.0,
    target=0.8,
    tolerance=0.10,
    mcse=0.01,
    simulations=100,
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


def _type1_result(
    *,
    observed,
    seed,
    n1=20,
    n2=20,
    target=0.05,
    tolerance=0.01,
    mcse=0.002,
    simulations=10_000,
):
    deviation = observed - target
    absolute_deviation = abs(deviation)
    return StatCIResult(
        property="type1_error",
        target=target,
        tolerance=tolerance,
        observed=observed,
        deviation=deviation,
        absolute_deviation=absolute_deviation,
        passed=absolute_deviation <= tolerance,
        method="welch_ttest",
        metric="type1_error",
        dgp1="Normal(mean=0, sd=1)",
        dgp2="Normal(mean=0, sd=1)",
        n1=n1,
        n2=n2,
        simulations=simulations,
        seed=seed,
        mcse=mcse,
    )


def test_compare_suites_supports_pure_bootstrap_coverage_suite():
    baseline = StatCISuiteResult.from_results(
        [
            _coverage_result(
                observed=0.78,
                seed=101,
                n=8,
            ),
            _coverage_result(
                observed=0.76,
                seed=102,
                n=12,
            ),
        ],
        name="coverage baseline",
    )
    current = StatCISuiteResult.from_results(
        [
            _coverage_result(
                observed=0.70,
                seed=201,
                n=12,
            ),
            _coverage_result(
                observed=0.79,
                seed=202,
                n=8,
            ),
        ],
        name="coverage current",
    )

    result = compare_suites(
        baseline,
        current,
        name="coverage regression",
    )

    assert result.total == 2
    assert all(
        isinstance(comparison.key, BootstrapCoverageComparisonKey)
        for comparison in result.comparisons
    )
    assert [comparison.key.n for comparison in result.comparisons] == [
        8,
        12,
    ]
    assert result.missing_current == ()
    assert result.new_current == ()


def test_compare_suites_preserves_pure_type1_ordering():
    baseline = StatCISuiteResult.from_results(
        [
            _type1_result(
                observed=0.052,
                seed=101,
                n1=30,
                n2=20,
            ),
            _type1_result(
                observed=0.053,
                seed=102,
                n1=20,
                n2=20,
            ),
        ],
        name="type1 baseline",
    )
    current = StatCISuiteResult.from_results(
        [
            _type1_result(
                observed=0.054,
                seed=201,
                n1=20,
                n2=20,
            ),
            _type1_result(
                observed=0.055,
                seed=202,
                n1=30,
                n2=20,
            ),
        ],
        name="type1 current",
    )

    result = compare_suites(baseline, current)

    assert all(
        isinstance(comparison.key, StatCIComparisonKey)
        for comparison in result.comparisons
    )
    assert [
        (comparison.key.n1, comparison.key.n2)
        for comparison in result.comparisons
    ] == [
        (20, 20),
        (30, 20),
    ]


def test_compare_suites_supports_mixed_type1_and_bootstrap_suite():
    type1_baseline = _type1_result(
        observed=0.052,
        seed=301,
    )
    type1_current = _type1_result(
        observed=0.08,
        seed=302,
    )
    coverage_baseline = _coverage_result(
        observed=0.78,
        seed=303,
    )
    coverage_current = _coverage_result(
        observed=0.75,
        seed=304,
    )

    baseline = StatCISuiteResult.from_results(
        [coverage_baseline, type1_baseline],
        name="mixed baseline",
    )
    current = StatCISuiteResult.from_results(
        [type1_current, coverage_current],
        name="mixed current",
    )

    result = compare_suites(baseline, current)

    assert result.total == 2
    assert isinstance(result.comparisons[0].key, StatCIComparisonKey)
    assert isinstance(
        result.comparisons[1].key,
        BootstrapCoverageComparisonKey,
    )
    assert [comparison.key.property for comparison in result.comparisons] == [
        "type1_error",
        "coverage",
    ]
    assert result.regression_count == 1


def test_compare_suites_rejects_duplicate_bootstrap_comparison_key():
    baseline = StatCISuiteResult.from_results(
        [
            _coverage_result(
                observed=0.78,
                seed=401,
                simulations=100,
            ),
            _coverage_result(
                observed=0.79,
                seed=402,
                simulations=200,
            ),
        ],
        name="duplicate coverage baseline",
    )
    current = StatCISuiteResult.from_results(
        [
            _coverage_result(
                observed=0.78,
                seed=403,
            )
        ],
        name="coverage current",
    )

    with pytest.raises(
        ValueError,
        match="duplicate comparison key",
    ) as exc_info:
        compare_suites(baseline, current)

    assert "coverage" in str(exc_info.value)


def test_compare_suites_strict_matching_reports_bootstrap_missing_and_new():
    baseline = StatCISuiteResult.from_results(
        [
            _coverage_result(
                observed=0.78,
                seed=501,
                n=8,
            )
        ],
        name="coverage baseline",
    )
    current = StatCISuiteResult.from_results(
        [
            _coverage_result(
                observed=0.78,
                seed=502,
                n=12,
            )
        ],
        name="coverage current",
    )

    with pytest.raises(
        ValueError,
        match="missing current checks",
    ) as exc_info:
        compare_suites(baseline, current)

    message = str(exc_info.value)
    assert "unexpected current checks" in message
    assert "coverage" in message
    assert "n=8" in message
    assert "n=12" in message


def test_compare_suites_non_strict_records_mixed_missing_and_new_keys():
    baseline = StatCISuiteResult.from_results(
        [
            _type1_result(
                observed=0.052,
                seed=601,
            ),
            _coverage_result(
                observed=0.78,
                seed=602,
                n=8,
            ),
        ],
        name="mixed baseline",
    )
    current = StatCISuiteResult.from_results(
        [
            _type1_result(
                observed=0.053,
                seed=603,
            ),
            _coverage_result(
                observed=0.78,
                seed=604,
                n=12,
            ),
        ],
        name="mixed current",
    )

    result = compare_suites(
        baseline,
        current,
        policy=RegressionPolicy(strict_matching=False),
    )

    assert result.total == 1
    assert len(result.missing_current) == 1
    assert isinstance(
        result.missing_current[0],
        BootstrapCoverageComparisonKey,
    )
    assert result.missing_current[0].n == 8
    assert len(result.new_current) == 1
    assert isinstance(
        result.new_current[0],
        BootstrapCoverageComparisonKey,
    )
    assert result.new_current[0].n == 12


def test_compare_suites_comparison_order_is_deterministic_across_input_order():
    type1_baseline = _type1_result(
        observed=0.052,
        seed=701,
    )
    type1_current = _type1_result(
        observed=0.053,
        seed=702,
    )
    coverage8_baseline = _coverage_result(
        observed=0.78,
        seed=703,
        n=8,
    )
    coverage8_current = _coverage_result(
        observed=0.79,
        seed=704,
        n=8,
    )
    coverage12_baseline = _coverage_result(
        observed=0.76,
        seed=705,
        n=12,
    )
    coverage12_current = _coverage_result(
        observed=0.77,
        seed=706,
        n=12,
    )

    first = compare_suites(
        StatCISuiteResult.from_results(
            [
                coverage12_baseline,
                type1_baseline,
                coverage8_baseline,
            ],
            name="baseline",
        ),
        StatCISuiteResult.from_results(
            [
                coverage8_current,
                coverage12_current,
                type1_current,
            ],
            name="current",
        ),
    )
    second = compare_suites(
        StatCISuiteResult.from_results(
            [
                coverage8_baseline,
                coverage12_baseline,
                type1_baseline,
            ],
            name="baseline",
        ),
        StatCISuiteResult.from_results(
            [
                type1_current,
                coverage12_current,
                coverage8_current,
            ],
            name="current",
        ),
    )

    assert first == second
    assert first.to_json() == second.to_json()
    assert [
        (
            type(comparison.key).__name__,
            getattr(comparison.key, "n1", None),
            getattr(comparison.key, "n", None),
        )
        for comparison in first.comparisons
    ] == [
        ("StatCIComparisonKey", 20, None),
        ("BootstrapCoverageComparisonKey", None, 8),
        ("BootstrapCoverageComparisonKey", None, 12),
    ]
