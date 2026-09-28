import json

import pytest

from statfuzz import (
    STATCI_STATUS_SCHEMA_VERSION,
    StatCIResult,
    StatCIStatusArtifact,
    StatCISuiteResult,
)


def _result(*, property, passed):
    observed = 0.05 if passed else 0.08
    return StatCIResult(
        property=property,
        target=0.05,
        tolerance=0.01,
        observed=observed,
        deviation=observed - 0.05,
        absolute_deviation=abs(observed - 0.05),
        passed=passed,
        method="welch_ttest",
        metric=property,
        dgp1="normal",
        dgp2="normal",
        n1=20,
        n2=20,
        simulations=5_000,
        seed=42,
        mcse=0.003,
    )


def test_status_artifact_from_passing_suite():
    suite = StatCISuiteResult.from_results(
        [
            _result(property="type1_error", passed=True),
            _result(property="coverage", passed=True),
        ],
        name="core checks",
    )

    artifact = StatCIStatusArtifact.from_suite(suite)

    assert artifact.suite_name == "core checks"
    assert artifact.status == "PASS"
    assert artifact.passed
    assert artifact.total == 2
    assert artifact.passed_count == 2
    assert artifact.failed_count == 0
    assert artifact.badge_message == "PASS · 2/2"
    assert artifact.badge_color == "brightgreen"


def test_status_artifact_from_failing_suite():
    suite = StatCISuiteResult.from_results(
        [
            _result(property="type1_error", passed=True),
            _result(property="coverage", passed=False),
            _result(property="bias", passed=True),
        ],
        name="release gate",
    )

    artifact = StatCIStatusArtifact.from_suite(suite)

    assert artifact.status == "FAIL"
    assert not artifact.passed
    assert artifact.total == 3
    assert artifact.passed_count == 2
    assert artifact.failed_count == 1
    assert artifact.badge_message == "FAIL · 2/3"
    assert artifact.badge_color == "red"


def test_badge_payload_matches_shields_endpoint_shape():
    suite = StatCISuiteResult.from_results(
        [_result(property="type1_error", passed=True)],
    )
    artifact = StatCIStatusArtifact.from_suite(suite)

    assert artifact.badge_as_dict() == {
        "schemaVersion": 1,
        "label": "StatCI",
        "message": "PASS · 1/1",
        "color": "brightgreen",
    }


def test_custom_badge_label_is_preserved():
    suite = StatCISuiteResult.from_results(
        [_result(property="type1_error", passed=False)],
    )
    artifact = StatCIStatusArtifact.from_suite(
        suite,
        badge_label="statistical CI",
    )

    assert artifact.badge_as_dict()["label"] == "statistical CI"


def test_status_json_is_deterministic_and_lightweight(tmp_path):
    suite = StatCISuiteResult.from_results(
        [
            _result(property="type1_error", passed=True),
            _result(property="coverage", passed=False),
        ],
        name="nightly",
    )
    artifact = StatCIStatusArtifact.from_suite(suite)

    first = artifact.to_json()
    second = artifact.to_json()
    assert first == second

    parsed = json.loads(first)
    assert parsed["schema_version"] == STATCI_STATUS_SCHEMA_VERSION
    assert parsed["suite_name"] == "nightly"
    assert parsed["status"] == "FAIL"
    assert parsed["checks"] == {
        "total": 2,
        "passed": 1,
        "failed": 1,
    }
    assert parsed["badge"]["schemaVersion"] == 1
    assert "results" not in parsed

    path = artifact.write_json(tmp_path / "statci-status.json")
    assert path.read_text(encoding="utf-8") == first + "\n"


def test_badge_json_is_deterministic_and_writable(tmp_path):
    suite = StatCISuiteResult.from_results(
        [_result(property="type1_error", passed=True)],
    )
    artifact = StatCIStatusArtifact.from_suite(suite)

    first = artifact.badge_to_json()
    second = artifact.badge_to_json()
    assert first == second

    parsed = json.loads(first)
    assert parsed == artifact.badge_as_dict()

    path = artifact.write_badge_json(tmp_path / "statci-badge.json")
    assert path.read_text(encoding="utf-8") == first + "\n"


def test_from_suite_requires_suite():
    with pytest.raises(TypeError, match="StatCISuiteResult"):
        StatCIStatusArtifact.from_suite(object())


def test_badge_label_must_be_non_empty():
    suite = StatCISuiteResult.from_results(
        [_result(property="type1_error", passed=True)]
    )

    with pytest.raises(ValueError, match="badge_label"):
        StatCIStatusArtifact.from_suite(suite, badge_label="")


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "suite_name": "",
            "status": "PASS",
            "passed": True,
            "total": 1,
            "passed_count": 1,
            "failed_count": 0,
        },
        {
            "suite_name": "x",
            "status": "UNKNOWN",
            "passed": False,
            "total": 1,
            "passed_count": 0,
            "failed_count": 1,
        },
        {
            "suite_name": "x",
            "status": "PASS",
            "passed": True,
            "total": 0,
            "passed_count": 0,
            "failed_count": 0,
        },
        {
            "suite_name": "x",
            "status": "PASS",
            "passed": True,
            "total": 2,
            "passed_count": 1,
            "failed_count": 0,
        },
        {
            "suite_name": "x",
            "status": "FAIL",
            "passed": True,
            "total": 1,
            "passed_count": 0,
            "failed_count": 1,
        },
    ],
)
def test_invalid_status_artifact_invariants_are_rejected(kwargs):
    with pytest.raises(ValueError):
        StatCIStatusArtifact(**kwargs)
