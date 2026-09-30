import copy
import json

import pytest

from statfuzz import (
    StatCIResult,
    StatCISuiteResult,
    bootstrap_mean_coverage,
    check_property,
)
from statfuzz.dgp import Normal
from statfuzz.methods import BootstrapMeanPercentile
from statfuzz.statci import (
    BootstrapCoverageComparisonKey,
    compare_results,
    compare_suites,
)
from statfuzz.targets import MeanTarget

ROOT_SEED = 20260930
METHOD = BootstrapMeanPercentile(
    resamples=19,
    interval_level=0.8,
)


def _coverage_check(
    *,
    dgp=None,
    method=METHOD,
    n=8,
    simulations=40,
    seed=ROOT_SEED,
    target=0.8,
    tolerance=0.10,
    mean_target=None,
    evidence_confidence_level=0.9,
):
    source = bootstrap_mean_coverage(
        dgp=Normal() if dgp is None else dgp,
        n=n,
        simulations=simulations,
        method=method,
        tolerance=0.10,
        seed=seed,
        mean_target=mean_target,
        batch_size=7,
        evidence_confidence_level=evidence_confidence_level,
    )
    return check_property(
        source,
        property="coverage",
        target=target,
        tolerance=tolerance,
    )


def test_bootstrap_comparison_key_locks_canonical_identity():
    result = _coverage_check(
        mean_target=MeanTarget(
            mean=0.0,
            note="reviewed mean target",
        )
    )

    key = BootstrapCoverageComparisonKey.from_result(result)

    assert key.as_dict() == {
        "kind": "bootstrap_coverage",
        "property": "coverage",
        "target": 0.8,
        "tolerance": 0.10,
        "method": "bootstrap_mean_percentile",
        "metric": "coverage",
        "dgp": result.dgp,
        "dgp_identity": {
            "family": "statfuzz.dgp.Normal",
            "parameters": {
                "mean": 0.0,
                "sd": 1.0,
            },
        },
        "target_check": {
            "kind": "mean",
            "source": "population_mean",
            "mean": 0.0,
            "population_mean": 0.0,
            "note": "reviewed mean target",
        },
        "bootstrap_method": METHOD.as_dict(),
        "n": 8,
    }

    restored = BootstrapCoverageComparisonKey.from_dict(key.as_dict())
    assert restored == key
    assert restored.as_dict() == key.as_dict()
    assert restored.canonical_json() == key.canonical_json()


@pytest.mark.parametrize(
    "changed",
    [
        lambda: _coverage_check(dgp=Normal(sd=2.0)),
        lambda: _coverage_check(
            mean_target=MeanTarget(
                mean=0.0,
                note="different reviewed target",
            )
        ),
        lambda: _coverage_check(
            method=BootstrapMeanPercentile(
                resamples=20,
                interval_level=0.8,
            )
        ),
        lambda: _coverage_check(
            method=BootstrapMeanPercentile(
                resamples=19,
                interval_level=0.9,
            ),
            target=0.8,
        ),
        lambda: _coverage_check(n=9),
        lambda: _coverage_check(target=0.79),
        lambda: _coverage_check(tolerance=0.11),
    ],
)
def test_bootstrap_comparison_key_changes_with_assertion_or_experiment_identity(
    changed,
):
    baseline = BootstrapCoverageComparisonKey.from_result(_coverage_check())
    observed = BootstrapCoverageComparisonKey.from_result(changed())

    assert observed != baseline


@pytest.mark.parametrize(
    "changed",
    [
        lambda: _coverage_check(seed=ROOT_SEED + 1),
        lambda: _coverage_check(simulations=41),
        lambda: _coverage_check(evidence_confidence_level=0.95),
    ],
)
def test_bootstrap_comparison_key_excludes_monte_carlo_realization_controls(
    changed,
):
    baseline = BootstrapCoverageComparisonKey.from_result(_coverage_check())
    observed = BootstrapCoverageComparisonKey.from_result(changed())

    assert observed == baseline
    assert observed.canonical_json() == baseline.canonical_json()


def test_bootstrap_statci_result_strictly_round_trips_for_baseline_loading():
    result = _coverage_check(
        mean_target=MeanTarget(
            mean=0.0,
            note="baseline truth",
        )
    )

    payload = result.as_dict()
    restored = StatCIResult.from_dict(copy.deepcopy(payload))

    assert restored == result
    assert restored.as_dict() == payload
    assert restored.to_json() == result.to_json()
    assert restored.evidence_kind == "bootstrap_coverage"


def test_bootstrap_statci_suite_round_trips_as_baseline():
    first = _coverage_check()
    second = _coverage_check(
        dgp=Normal(sd=1.5),
        seed=ROOT_SEED + 1,
    )
    suite = StatCISuiteResult.from_results(
        [first, second],
        name="bootstrap baseline",
    )

    restored = StatCISuiteResult.from_json(suite.to_json())

    assert restored.as_dict() == suite.as_dict()
    assert restored.to_json() == suite.to_json()
    assert all(
        result.evidence_kind == "bootstrap_coverage"
        for result in restored.results
    )


def test_bootstrap_baseline_loader_rejects_unknown_evidence_fields():
    payload = _coverage_check().as_dict()
    payload["evidence"]["batch_size"] = 64

    with pytest.raises(ValueError, match="unexpected keys"):
        StatCIResult.from_dict(payload)


def test_bootstrap_baseline_loader_rejects_inconsistent_coverage_count():
    payload = _coverage_check().as_dict()
    payload["evidence"]["coverage_count"] += 1

    with pytest.raises(
        ValueError,
        match="coverage_count / simulations",
    ):
        StatCIResult.from_dict(payload)


def test_bootstrap_baseline_loader_rejects_method_identity_mismatch():
    payload = _coverage_check().as_dict()
    payload["evidence"]["method"] = "different-bootstrap-method"

    with pytest.raises(
        ValueError,
        match="must match evidence.bootstrap_method.method",
    ):
        StatCIResult.from_dict(payload)


def test_bootstrap_baseline_loader_rejects_corrupt_target_check_shape():
    payload = _coverage_check().as_dict()
    payload["evidence"]["target_check"]["extra"] = "not canonical"

    with pytest.raises(ValueError, match="unexpected keys"):
        StatCIResult.from_dict(payload)


def test_bootstrap_comparison_key_strictly_rejects_unknown_fields():
    key = BootstrapCoverageComparisonKey.from_result(
        _coverage_check()
    ).as_dict()
    key["seed"] = ROOT_SEED

    with pytest.raises(ValueError, match="unexpected keys"):
        BootstrapCoverageComparisonKey.from_dict(key)


def test_phase2b_compare_results_reuses_phase2a_identity():
    baseline = _coverage_check(seed=ROOT_SEED)
    current = _coverage_check(seed=ROOT_SEED + 1)

    comparison = compare_results(baseline, current)

    assert isinstance(comparison.key, BootstrapCoverageComparisonKey)
    assert comparison.key == BootstrapCoverageComparisonKey.from_result(
        baseline
    )


def test_phase2a_does_not_enable_bootstrap_compare_suites():
    baseline = StatCISuiteResult.from_results(
        [_coverage_check(seed=ROOT_SEED)],
        name="baseline",
    )
    current = StatCISuiteResult.from_results(
        [_coverage_check(seed=ROOT_SEED + 1)],
        name="current",
    )

    with pytest.raises(TypeError, match="suite matching"):
        compare_suites(baseline, current)


def test_baseline_json_payload_remains_deterministic_after_round_trip():
    result = _coverage_check()
    first = result.to_json()
    restored = StatCIResult.from_dict(json.loads(first))

    assert restored.to_json() == first
