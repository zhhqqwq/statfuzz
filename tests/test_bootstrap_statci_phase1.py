import json

import pytest

from statfuzz import (
    StatCIResult,
    StatCISuiteResult,
    StatisticalAssertion,
    StatisticalAssertionError,
    StressTestResult,
    assert_property,
    bootstrap_mean_coverage,
    check_property,
)
from statfuzz.dgp import LogNormal, Normal
from statfuzz.methods import BootstrapMeanPercentile
from statfuzz.statci import (
    STATCI_SCHEMA_VERSION,
    render_github_summary,
)

METHOD = BootstrapMeanPercentile(
    resamples=19,
    interval_level=0.8,
)
NORMAL_SEED = 16896585173923650117
LOGNORMAL_SEED = 9415301123600923416


def _coverage_result(*, dgp, seed):
    return bootstrap_mean_coverage(
        dgp=dgp,
        n=8,
        simulations=40,
        method=METHOD,
        tolerance=0.10,
        seed=seed,
        batch_size=7,
        evidence_confidence_level=0.9,
    )


def _type1_result(*, empirical=0.057):
    return StressTestResult(
        method="welch_ttest",
        metric="type1_error",
        dgp1="normal",
        dgp2="normal",
        n1=20,
        n2=20,
        simulations=10_000,
        seed=42,
        nominal=0.05,
        empirical=empirical,
        mcse=0.002,
        tolerance=0.01,
    )


def test_check_property_accepts_real_bootstrap_coverage_pass_and_fail():
    passing_source = _coverage_result(
        dgp=Normal(mean=0.0, sd=1.0),
        seed=NORMAL_SEED,
    )
    failing_source = _coverage_result(
        dgp=LogNormal(sigma=1.0, mean=0.0),
        seed=LOGNORMAL_SEED,
    )

    assert passing_source.coverage_count == 31
    assert passing_source.empirical == 0.775
    assert failing_source.coverage_count == 16
    assert failing_source.empirical == 0.4

    passing = check_property(
        passing_source,
        property="coverage",
        target=0.8,
        tolerance=0.05,
    )
    failing = check_property(
        failing_source,
        property="coverage",
        target=0.8,
        tolerance=0.10,
    )

    assert isinstance(passing, StatCIResult)
    assert isinstance(failing, StatCIResult)
    assert passing.passed
    assert passing.status == "PASS"
    assert passing.observed == 0.775
    assert passing.deviation == pytest.approx(-0.025)
    assert failing.status == "FAIL"
    assert not failing.passed
    assert failing.observed == 0.4
    assert failing.deviation == pytest.approx(-0.4)


def test_statistical_assertion_and_assert_property_support_bootstrap_coverage():
    passing_source = _coverage_result(
        dgp=Normal(mean=0.0, sd=1.0),
        seed=NORMAL_SEED,
    )
    failing_source = _coverage_result(
        dgp=LogNormal(sigma=1.0, mean=0.0),
        seed=LOGNORMAL_SEED,
    )

    assertion = StatisticalAssertion(
        property="coverage",
        target=0.8,
        tolerance=0.05,
    )
    assert assertion.evaluate(passing_source).passed

    with pytest.raises(StatisticalAssertionError) as exc_info:
        assert_property(
            failing_source,
            property="coverage",
            target=0.8,
            tolerance=0.10,
        )

    assert exc_info.value.result.status == "FAIL"
    assert exc_info.value.result.property == "coverage"
    assert exc_info.value.result.observed == 0.4


def test_bootstrap_statci_evidence_snapshot_is_coverage_specific():
    source = _coverage_result(
        dgp=Normal(mean=0.0, sd=1.0),
        seed=NORMAL_SEED,
    )
    result = check_property(
        source,
        property="coverage",
        target=0.8,
        tolerance=0.05,
    )

    payload = result.as_dict()
    evidence = payload["evidence"]

    assert payload["schema_version"] == STATCI_SCHEMA_VERSION == "1.2"
    assert payload["property"] == "coverage"
    assert payload["status"] == "PASS"
    assert evidence["kind"] == "bootstrap_coverage"
    assert evidence["method"] == "bootstrap_mean_percentile"
    assert evidence["metric"] == "coverage"
    assert evidence["dgp"] == source.dgp
    assert evidence["dgp_identity"] == source.dgp_identity.as_dict()
    assert evidence["n"] == 8
    assert evidence["simulations"] == 40
    assert evidence["seed"] == NORMAL_SEED
    assert evidence["coverage_count"] == 31
    assert evidence["target_check"] == source.target_check.as_dict()
    assert evidence["bootstrap_method"] == METHOD.as_dict()
    assert evidence["evidence_interval"] == {
        "level": source.evidence_confidence_level,
        "method": source.evidence_interval_method,
        "low": source.evidence_interval_low,
        "high": source.evidence_interval_high,
    }
    assert evidence["evidence_interval"]["low"] <= source.empirical
    assert source.empirical <= evidence["evidence_interval"]["high"]
    assert "rejection_count" not in evidence
    assert "null_check" not in evidence


def test_bootstrap_statci_snapshot_json_is_deterministic():
    source = _coverage_result(
        dgp=Normal(mean=0.0, sd=1.0),
        seed=NORMAL_SEED,
    )
    first = check_property(
        source,
        property="coverage",
        target=0.8,
        tolerance=0.05,
    )
    second = check_property(
        source,
        property="coverage",
        target=0.8,
        tolerance=0.05,
    )

    assert first == second
    assert first.as_dict() == second.as_dict()
    assert first.to_json() == second.to_json()
    assert json.loads(first.to_json()) == first.as_dict()


def test_type1_statci_json_shape_remains_exactly_legacy():
    result = check_property(
        _type1_result(),
        property="type1_error",
        target=0.05,
        tolerance=0.01,
    )
    payload = result.as_dict()

    assert type(result) is StatCIResult
    assert set(payload) == {
        "schema_version",
        "property",
        "target",
        "tolerance",
        "observed",
        "deviation",
        "absolute_deviation",
        "passed",
        "status",
        "evidence",
    }
    assert payload["schema_version"] == "1.2"
    assert payload["property"] == "type1_error"
    assert payload["status"] == "PASS"
    assert payload["evidence"] == {
        "method": "welch_ttest",
        "metric": "type1_error",
        "dgp1": "normal",
        "dgp2": "normal",
        "dgp1_identity": None,
        "dgp2_identity": None,
        "n1": 20,
        "n2": 20,
        "simulations": 10_000,
        "seed": 42,
        "mcse": 0.002,
        "null_check": None,
        "rejection_count": None,
        "confidence_interval": None,
    }


def test_type1_github_summary_is_byte_for_byte_unchanged():
    result = check_property(
        _type1_result(),
        property="type1_error",
        target=0.05,
        tolerance=0.01,
    )
    suite = StatCISuiteResult.from_results(
        [result],
        name="Type-I",
    )

    assert render_github_summary(suite) == (
        "## StatCI — Type-I\n"
        "\n"
        "**Overall:** PASS  \n"
        "**Checks:** 1 passed / 0 failed / 1 total\n"
        "\n"
        "| Property | Status | Observed | Target | Tolerance | Deviation | MCSE | Simulations | Seed |\n"
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n"
        "| type1_error | PASS | 0.057 | 0.05 | 0.01 | 0.007 | 0.002 | 10000 | 42 |\n"
    )


def test_github_summary_needs_no_bootstrap_specific_renderer():
    passing = check_property(
        _coverage_result(
            dgp=Normal(mean=0.0, sd=1.0),
            seed=NORMAL_SEED,
        ),
        property="coverage",
        target=0.8,
        tolerance=0.05,
    )
    failing = check_property(
        _coverage_result(
            dgp=LogNormal(sigma=1.0, mean=0.0),
            seed=LOGNORMAL_SEED,
        ),
        property="coverage",
        target=0.8,
        tolerance=0.10,
    )
    suite = StatCISuiteResult.from_results(
        [passing, failing],
        name="Bootstrap coverage",
    )

    summary = render_github_summary(suite)

    assert "**Overall:** FAIL" in summary
    assert "1 passed / 1 failed / 2 total" in summary
    assert "| coverage | PASS | 0.775 | 0.8 | 0.05 | -0.025 |" in summary
    assert "| coverage | FAIL | 0.4 | 0.8 | 0.1 | -0.4 |" in summary
    assert "### Failed checks" in summary
