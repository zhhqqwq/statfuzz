from statfuzz import StatCIResult, StatCISuiteResult
from statfuzz.methods import BootstrapMeanPercentile
from statfuzz.statci import (
    compare_suites,
    render_regression_summary,
)

METHOD = BootstrapMeanPercentile(
    resamples=19,
    interval_level=0.8,
)


def _type1_result(*, observed, seed):
    target = 0.05
    tolerance = 0.01
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
        n1=20,
        n2=20,
        simulations=10_000,
        seed=seed,
        mcse=0.002,
    )


def _coverage_result(*, observed, seed):
    target = 0.8
    tolerance = 0.10
    simulations = 100
    coverage_count = round(observed * simulations)
    deviation = observed - target
    absolute_deviation = abs(deviation)
    return StatCIResult.from_dict(
        {
            "schema_version": "1.2",
            "property": "coverage",
            "target": target,
            "tolerance": tolerance,
            "observed": observed,
            "deviation": deviation,
            "absolute_deviation": absolute_deviation,
            "passed": absolute_deviation <= tolerance,
            "status": (
                "PASS"
                if absolute_deviation <= tolerance
                else "FAIL"
            ),
            "evidence": {
                "kind": "bootstrap_coverage",
                "method": "bootstrap_mean_percentile",
                "metric": "coverage",
                "dgp": "Normal(mean=0, sd=1)",
                "dgp_identity": {
                    "family": "statfuzz.dgp.Normal",
                    "parameters": {
                        "mean": 0.0,
                        "sd": 1.0,
                    },
                },
                "n": 8,
                "simulations": simulations,
                "seed": seed,
                "mcse": 0.01,
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


def test_type1_regression_summary_is_byte_for_byte_unchanged():
    result = compare_suites(
        StatCISuiteResult.from_results(
            [_type1_result(observed=0.052, seed=1)],
            name="baseline",
        ),
        StatCISuiteResult.from_results(
            [_type1_result(observed=0.08, seed=2)],
            name="current",
        ),
        name="Type-I regression",
    )

    assert render_regression_summary(result) == (
        "## StatCI Regression — Type-I regression\n"
        "\n"
        "**Overall:** FAIL  \n"
        "**Matched checks:** 1  \n"
        "**Regressions:** 1  \n"
        "**PASS→FAIL transitions:** 1  \n"
        "**Improvements:** 0\n"
        "\n"
        "Policy: minimum worsening=0, uncertainty=2 × "
        "conservative uncertainty scale.\n"
        "\n"
        "| Property | Direction | Baseline | Current | Worsening | "
        "Uncertainty scale | Guard threshold | PASS→FAIL |\n"
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |\n"
        "| type1_error | REGRESSION | 0.002 | 0.03 | 0.028 | "
        "0.004 | 0.008 | yes |\n"
        "\n"
        "### Regressions\n"
        "\n"
        "- **type1_error \\| welch_ttest \\| "
        "Normal(mean=0, sd=1) vs Normal(mean=0, sd=1) \\| "
        "n=20/20 \\| target=0.05 ± 0.01**: "
        "worsening=0.028 > guard=0.008\n"
    )


def test_bootstrap_coverage_regression_summary_renders_full_comparison_key():
    result = compare_suites(
        StatCISuiteResult.from_results(
            [_coverage_result(observed=0.78, seed=11)],
            name="baseline",
        ),
        StatCISuiteResult.from_results(
            [_coverage_result(observed=0.60, seed=12)],
            name="current",
        ),
        name="Coverage regression",
    )

    assert render_regression_summary(result) == (
        "## StatCI Regression — Coverage regression\n"
        "\n"
        "**Overall:** FAIL  \n"
        "**Matched checks:** 1  \n"
        "**Regressions:** 1  \n"
        "**PASS→FAIL transitions:** 1  \n"
        "**Improvements:** 0\n"
        "\n"
        "Policy: minimum worsening=0, uncertainty=2 × "
        "conservative uncertainty scale.\n"
        "\n"
        "| Check | Direction | Baseline | Current | Worsening | "
        "Uncertainty scale | Guard threshold | PASS→FAIL |\n"
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |\n"
        "| coverage \\| bootstrap_mean_percentile \\| "
        "Normal(mean=0, sd=1) \\| n=8 \\| target=0.8 ± 0.1 | "
        "REGRESSION | 0.02 | 0.2 | 0.18 | 0.02 | 0.04 | yes |\n"
        "\n"
        "### Regressions\n"
        "\n"
        "- **coverage \\| bootstrap_mean_percentile \\| "
        "Normal(mean=0, sd=1) \\| n=8 \\| target=0.8 ± 0.1**: "
        "worsening=0.18 > guard=0.04\n"
    )


def test_mixed_regression_summary_renders_both_key_families():
    result = compare_suites(
        StatCISuiteResult.from_results(
            [
                _coverage_result(observed=0.78, seed=21),
                _type1_result(observed=0.052, seed=22),
            ],
            name="baseline",
        ),
        StatCISuiteResult.from_results(
            [
                _type1_result(observed=0.053, seed=23),
                _coverage_result(observed=0.60, seed=24),
            ],
            name="current",
        ),
        name="Mixed regression",
    )

    assert render_regression_summary(result) == (
        "## StatCI Regression — Mixed regression\n"
        "\n"
        "**Overall:** FAIL  \n"
        "**Matched checks:** 2  \n"
        "**Regressions:** 1  \n"
        "**PASS→FAIL transitions:** 1  \n"
        "**Improvements:** 0\n"
        "\n"
        "Policy: minimum worsening=0, uncertainty=2 × "
        "conservative uncertainty scale.\n"
        "\n"
        "| Check | Direction | Baseline | Current | Worsening | "
        "Uncertainty scale | Guard threshold | PASS→FAIL |\n"
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |\n"
        "| type1_error \\| welch_ttest \\| "
        "Normal(mean=0, sd=1) vs Normal(mean=0, sd=1) \\| "
        "n=20/20 \\| target=0.05 ± 0.01 | WITHIN_GUARD | "
        "0.002 | 0.003 | 0.001 | 0.004 | 0.008 | no |\n"
        "| coverage \\| bootstrap_mean_percentile \\| "
        "Normal(mean=0, sd=1) \\| n=8 \\| target=0.8 ± 0.1 | "
        "REGRESSION | 0.02 | 0.2 | 0.18 | 0.02 | 0.04 | yes |\n"
        "\n"
        "### Regressions\n"
        "\n"
        "- **coverage \\| bootstrap_mean_percentile \\| "
        "Normal(mean=0, sd=1) \\| n=8 \\| target=0.8 ± 0.1**: "
        "worsening=0.18 > guard=0.04\n"
    )


def test_mixed_regression_summary_is_deterministic_across_suite_input_order():
    type1_baseline = _type1_result(observed=0.052, seed=31)
    type1_current = _type1_result(observed=0.053, seed=32)
    coverage_baseline = _coverage_result(observed=0.78, seed=33)
    coverage_current = _coverage_result(observed=0.60, seed=34)

    first = compare_suites(
        StatCISuiteResult.from_results(
            [coverage_baseline, type1_baseline],
            name="baseline",
        ),
        StatCISuiteResult.from_results(
            [type1_current, coverage_current],
            name="current",
        ),
        name="Deterministic mixed",
    )
    second = compare_suites(
        StatCISuiteResult.from_results(
            [type1_baseline, coverage_baseline],
            name="baseline",
        ),
        StatCISuiteResult.from_results(
            [coverage_current, type1_current],
            name="current",
        ),
        name="Deterministic mixed",
    )

    assert first == second
    assert render_regression_summary(first) == render_regression_summary(second)
