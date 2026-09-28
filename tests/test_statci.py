import json

import pytest

from statfuzz import (
    StatCIResult,
    StatisticalAssertion,
    StatisticalAssertionError,
    StressTestResult,
    assert_property,
    check_property,
    stress_test,
)
from statfuzz.dgp import DGPIdentity, Normal
from statfuzz.statci import STATCI_SCHEMA_VERSION


def _result(
    *,
    metric="type1_error",
    empirical=0.05,
    mcse=0.002,
    simulations=10_000,
    seed=42,
    dgp_identity=None,
):
    return StressTestResult(
        method="welch_ttest",
        metric=metric,
        dgp1="normal",
        dgp2="normal",
        n1=20,
        n2=20,
        simulations=simulations,
        seed=seed,
        nominal=0.05,
        empirical=empirical,
        mcse=mcse,
        tolerance=0.01,
        dgp1_identity=dgp_identity,
        dgp2_identity=dgp_identity,
    )


def test_check_property_returns_machine_readable_pass():
    result = check_property(
        _result(empirical=0.057),
        property="type1_error",
        target=0.05,
        tolerance=0.01,
    )

    assert isinstance(result, StatCIResult)
    assert result.passed
    assert result.status == "PASS"
    assert result.property == "type1_error"
    assert result.target == 0.05
    assert result.observed == 0.057
    assert result.deviation == pytest.approx(0.007)
    assert result.absolute_deviation == pytest.approx(0.007)
    assert result.mcse == 0.002
    assert result.simulations == 10_000
    assert result.seed == 42


def test_boundary_is_inclusive():
    result = check_property(
        _result(empirical=0.06),
        property="type1_error",
        target=0.05,
        tolerance=0.01,
    )

    assert result.passed


def test_check_property_returns_fail_without_raising():
    result = check_property(
        _result(empirical=0.071),
        property="type1_error",
        target=0.05,
        tolerance=0.01,
    )

    assert not result.passed
    assert result.status == "FAIL"
    assert result.absolute_deviation == pytest.approx(0.021)


def test_assert_property_returns_result_on_pass():
    result = assert_property(
        _result(empirical=0.052),
        property="type1_error",
        target=0.05,
        tolerance=0.01,
    )

    assert result.passed


def test_assert_property_raises_assertion_error_with_result_payload():
    with pytest.raises(StatisticalAssertionError) as exc_info:
        assert_property(
            _result(empirical=0.08, mcse=0.003, simulations=5_000, seed=99),
            property="type1_error",
            target=0.05,
            tolerance=0.01,
        )

    error = exc_info.value
    assert isinstance(error, AssertionError)
    assert not error.result.passed
    assert error.result.status == "FAIL"
    assert error.result.seed == 99
    assert "type1_error" in str(error)
    assert "observed=0.08" in str(error)
    assert "target=0.05" in str(error)
    assert "tolerance=0.01" in str(error)
    assert "mcse=0.003" in str(error)
    assert "simulations=5000" in str(error)


def test_requested_property_must_match_result_metric():
    with pytest.raises(ValueError, match="does not match"):
        check_property(
            _result(metric="type1_error"),
            property="coverage",
            target=0.95,
            tolerance=0.01,
        )


@pytest.mark.parametrize(
    ("target", "tolerance", "message"),
    [
        (float("nan"), 0.01, "target must be finite"),
        (float("inf"), 0.01, "target must be finite"),
        (0.05, float("nan"), "tolerance must be finite"),
        (0.05, float("inf"), "tolerance must be finite"),
        (0.05, -0.01, "tolerance must be non-negative"),
    ],
)
def test_assertion_configuration_requires_finite_valid_values(
    target,
    tolerance,
    message,
):
    with pytest.raises(ValueError, match=message):
        StatisticalAssertion(
            property="type1_error",
            target=target,
            tolerance=tolerance,
        )


def test_assertion_requires_non_empty_property():
    with pytest.raises(ValueError, match="non-empty"):
        StatisticalAssertion(property="", target=0.05, tolerance=0.01)


def test_result_empirical_and_mcse_are_validated():
    with pytest.raises(ValueError, match="result.empirical must be finite"):
        check_property(
            _result(empirical=float("nan")),
            property="type1_error",
            target=0.05,
            tolerance=0.01,
        )

    with pytest.raises(ValueError, match="result.mcse must be non-negative"):
        check_property(
            _result(mcse=-0.001),
            property="type1_error",
            target=0.05,
            tolerance=0.01,
        )


def test_mcse_is_evidence_not_part_of_threshold():
    tiny_mcse = check_property(
        _result(empirical=0.061, mcse=0.00001),
        property="type1_error",
        target=0.05,
        tolerance=0.01,
    )
    large_mcse = check_property(
        _result(empirical=0.061, mcse=0.1),
        property="type1_error",
        target=0.05,
        tolerance=0.01,
    )

    assert not tiny_mcse.passed
    assert not large_mcse.passed
    assert tiny_mcse.absolute_deviation == large_mcse.absolute_deviation


def test_statci_json_is_deterministic_and_contains_evidence(tmp_path):
    result = check_property(
        _result(empirical=0.08, mcse=0.003, simulations=5_000, seed=99),
        property="type1_error",
        target=0.05,
        tolerance=0.01,
    )

    first = result.to_json()
    second = result.to_json()

    assert first == second

    parsed = json.loads(first)
    assert parsed["schema_version"] == STATCI_SCHEMA_VERSION
    assert parsed["status"] == "FAIL"
    assert parsed["property"] == "type1_error"
    assert parsed["evidence"]["method"] == "welch_ttest"
    assert parsed["evidence"]["mcse"] == 0.003
    assert parsed["evidence"]["simulations"] == 5_000
    assert parsed["evidence"]["seed"] == 99

    path = result.write_json(tmp_path / "statci.json")
    assert path.read_text(encoding="utf-8") == first + "\n"


def test_statistical_assertion_can_be_reused():
    assertion = StatisticalAssertion(
        property="type1_error",
        target=0.05,
        tolerance=0.01,
    )

    passing = assertion.evaluate(_result(empirical=0.055))
    failing = assertion.evaluate(_result(empirical=0.08))

    assert passing.passed
    assert not failing.passed


def test_statci_json_preserves_structured_dgp_identity():
    identity = DGPIdentity.from_mapping(
        "statfuzz.dgp.Normal",
        {"mean": 0.0, "sd": 1.0000001},
    )
    result = check_property(
        _result(dgp_identity=identity),
        property="type1_error",
        target=0.05,
        tolerance=0.01,
    )

    parsed = json.loads(result.to_json())

    assert result.dgp1_identity == identity
    assert result.dgp2_identity == identity
    assert parsed["evidence"]["dgp1_identity"] == identity.as_dict()
    assert parsed["evidence"]["dgp2_identity"] == identity.as_dict()


def test_statci_preserves_null_and_binomial_interval_evidence(monkeypatch):
    monkeypatch.setattr(
        "statfuzz.simulation.welch_ttest_pvalue",
        lambda x, y: 1.0,
    )
    stress = stress_test(
        method="welch_ttest",
        metric="type1_error",
        dgp=Normal(),
        simulations=20,
        seed=7,
    )
    result = check_property(
        stress,
        property="type1_error",
        target=0.05,
        tolerance=0.1,
    )

    parsed = json.loads(result.to_json())
    evidence = parsed["evidence"]

    assert evidence["null_check"]["source"] == "population_means"
    assert evidence["null_check"]["common_mean"] == 0.0
    assert evidence["rejection_count"] == 0
    assert evidence["confidence_interval"]["level"] == 0.95
    assert evidence["confidence_interval"]["method"] == "wilson"
    assert evidence["confidence_interval"]["low"] == 0.0
    assert evidence["confidence_interval"]["high"] == pytest.approx(
        0.16112515805281935
    )

    loaded = StatCIResult.from_dict(parsed)
    assert loaded.null_check == result.null_check
    assert loaded.rejection_count == 0
    assert loaded.interval_method == "wilson"
    assert loaded.interval_high == pytest.approx(0.16112515805281935)


def test_statci_tolerance_rule_is_unchanged_by_interval_evidence(monkeypatch):
    monkeypatch.setattr(
        "statfuzz.simulation.welch_ttest_pvalue",
        lambda x, y: 1.0,
    )
    stress = stress_test(
        method="welch_ttest",
        metric="type1_error",
        dgp=Normal(),
        simulations=20,
    )

    result = check_property(
        stress,
        property="type1_error",
        target=0.05,
        tolerance=0.04,
    )

    assert stress.interval_high > 0.05
    assert not result.passed
    assert result.absolute_deviation == pytest.approx(0.05)
