from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from statfuzz.bootstrap_coverage import (
    BootstrapCoverageEvent,
    BootstrapCoverageResult,
    bootstrap_mean_coverage_event,
)
from statfuzz.dgp import DGPIdentity
from statfuzz.methods import BootstrapMeanPercentile
from statfuzz.metrics import BinomialRateEvidence, binomial_rate_evidence
from statfuzz.result import StatisticalPropertyResult
from statfuzz.targets import MeanTargetCheck

ROOT_SEED = 20260930
SAMPLE = np.array([1.0, 2.0, 4.0, 8.0])
METHOD = BootstrapMeanPercentile(resamples=7, interval_level=0.8)


def _target(mean):
    return MeanTargetCheck(
        source="declaration",
        mean=mean,
        population_mean=None,
    )


def test_single_coverage_event_fixed_reference():
    event = bootstrap_mean_coverage_event(
        SAMPLE,
        target_check=_target(4.0),
        method=METHOD,
        root_seed=ROOT_SEED,
        logical_outer_index=0,
    )

    assert event == BootstrapCoverageEvent(
        logical_outer_index=0,
        target_mean=4.0,
        interval_low=3.3499999999999996,
        interval_high=5.5,
        covered=True,
    )


def test_single_coverage_event_is_inclusive_at_lower_boundary():
    event = bootstrap_mean_coverage_event(
        SAMPLE,
        target_check=_target(1.5),
        method=METHOD,
        root_seed=ROOT_SEED,
        logical_outer_index=3,
    )

    assert event.interval_low == 1.5
    assert event.target_mean == event.interval_low
    assert event.covered is True


def test_single_coverage_event_is_inclusive_at_upper_boundary():
    event = bootstrap_mean_coverage_event(
        SAMPLE,
        target_check=_target(5.5),
        method=METHOD,
        root_seed=ROOT_SEED,
        logical_outer_index=0,
    )

    assert event.interval_high == 5.5
    assert event.target_mean == event.interval_high
    assert event.covered is True


def test_single_coverage_event_outside_interval_is_not_covered():
    event = bootstrap_mean_coverage_event(
        SAMPLE,
        target_check=_target(6.0),
        method=METHOD,
        root_seed=ROOT_SEED,
        logical_outer_index=0,
    )

    assert event.interval_high == 5.5
    assert event.covered is False


def test_single_coverage_event_is_deterministic_per_logical_index():
    first = bootstrap_mean_coverage_event(
        SAMPLE,
        target_check=_target(4.0),
        method=METHOD,
        root_seed=ROOT_SEED,
        logical_outer_index=3,
    )

    bootstrap_mean_coverage_event(
        SAMPLE,
        target_check=_target(4.0),
        method=METHOD,
        root_seed=ROOT_SEED,
        logical_outer_index=7,
    )

    second = bootstrap_mean_coverage_event(
        SAMPLE,
        target_check=_target(4.0),
        method=METHOD,
        root_seed=ROOT_SEED,
        logical_outer_index=3,
    )

    assert second == first


def test_coverage_event_contract_rejects_inconsistent_covered_flag():
    with pytest.raises(ValueError, match="inclusive"):
        BootstrapCoverageEvent(
            logical_outer_index=0,
            target_mean=4.0,
            interval_low=3.0,
            interval_high=5.0,
            covered=False,
        )


def test_coverage_event_contract_rejects_invalid_bounds():
    with pytest.raises(ValueError, match="low <= high"):
        BootstrapCoverageEvent(
            logical_outer_index=0,
            target_mean=4.0,
            interval_low=5.0,
            interval_high=3.0,
            covered=False,
        )


def test_single_coverage_event_requires_target_check():
    with pytest.raises(TypeError, match="MeanTargetCheck"):
        bootstrap_mean_coverage_event(
            SAMPLE,
            target_check=object(),
            method=METHOD,
            root_seed=ROOT_SEED,
            logical_outer_index=0,
        )


def test_single_coverage_event_propagates_scalar_oracle_sample_validation():
    with pytest.raises(ValueError, match="finite"):
        bootstrap_mean_coverage_event(
            np.array([1.0, np.nan]),
            target_check=_target(0.0),
            method=METHOD,
            root_seed=ROOT_SEED,
            logical_outer_index=0,
        )


def _identity():
    return DGPIdentity.from_mapping(
        "tests.BootstrapCoverageDGP",
        {"mean": 0.0},
    )


def _result(coverage_count=95, tolerance=0.01):
    return BootstrapCoverageResult.from_coverage_count(
        dgp="coverage-dgp",
        dgp_identity=_identity(),
        n=20,
        simulations=100,
        seed=ROOT_SEED,
        method_config=BootstrapMeanPercentile(
            resamples=199,
            interval_level=0.95,
        ),
        target_check=_target(0.0),
        coverage_count=coverage_count,
        tolerance=tolerance,
        evidence_confidence_level=0.9,
        evidence_interval_method="wilson",
    )


def test_bootstrap_coverage_result_satisfies_statistical_property_contract():
    result = _result()

    assert isinstance(result, StatisticalPropertyResult)
    assert result.method == "bootstrap_mean_percentile"
    assert result.metric == "coverage"
    assert result.nominal == 0.95
    assert result.empirical == 0.95
    assert result.mcse > 0.0
    assert result.coverage_count == 95
    assert result.bootstrap_resamples == 199
    assert result.bootstrap_interval_level == 0.95
    assert result.bootstrap_quantile_method == "linear"
    assert result.evidence_confidence_level == 0.9
    assert result.evidence_interval_method == "wilson"
    assert result.evidence_interval_low <= result.empirical
    assert result.empirical <= result.evidence_interval_high
    assert result.deviation == 0.0
    assert result.passed is True
    assert result.status == "PASS"


def test_bootstrap_coverage_result_uses_tolerance_for_status():
    result = _result(coverage_count=90, tolerance=0.01)

    assert result.deviation == pytest.approx(-0.05)
    assert result.passed is False
    assert result.status == "OUTSIDE_TOLERANCE"


def test_bootstrap_coverage_result_is_immutable():
    result = _result()

    with pytest.raises(FrozenInstanceError):
        result.n = 21


def test_bootstrap_coverage_result_rejects_evidence_trial_mismatch():
    with pytest.raises(ValueError, match="trials must equal simulations"):
        BootstrapCoverageResult(
            dgp="coverage-dgp",
            dgp_identity=_identity(),
            n=20,
            simulations=100,
            seed=ROOT_SEED,
            method_config=BootstrapMeanPercentile(),
            target_check=_target(0.0),
            evidence=binomial_rate_evidence(95, 101),
            tolerance=0.01,
        )


def test_bootstrap_coverage_result_rejects_internally_inconsistent_evidence():
    with pytest.raises(ValueError, match="inconsistent"):
        BootstrapCoverageResult(
            dgp="coverage-dgp",
            dgp_identity=_identity(),
            n=20,
            simulations=100,
            seed=ROOT_SEED,
            method_config=BootstrapMeanPercentile(),
            target_check=_target(0.0),
            evidence=BinomialRateEvidence(
                event_count=95,
                trials=100,
                empirical=0.5,
                mcse=0.01,
                confidence_level=0.95,
                interval_method="wilson",
                interval_low=0.4,
                interval_high=0.6,
            ),
            tolerance=0.01,
        )


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("n", 0, ValueError),
        ("n", True, TypeError),
        ("simulations", 0, ValueError),
        ("seed", -1, ValueError),
        ("seed", True, TypeError),
        ("tolerance", -0.1, ValueError),
        ("tolerance", float("nan"), ValueError),
    ],
)
def test_bootstrap_coverage_result_rejects_invalid_core_fields(
    field,
    value,
    error,
):
    kwargs = {
        "dgp": "coverage-dgp",
        "dgp_identity": _identity(),
        "n": 20,
        "simulations": 100,
        "seed": ROOT_SEED,
        "method_config": BootstrapMeanPercentile(),
        "target_check": _target(0.0),
        "evidence": binomial_rate_evidence(95, 100),
        "tolerance": 0.01,
    }
    kwargs[field] = value

    with pytest.raises(error):
        BootstrapCoverageResult(**kwargs)
