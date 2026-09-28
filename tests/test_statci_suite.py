import json

import pytest

from statfuzz import (
    StatCIResult,
    StatCISuiteError,
    StatCISuiteResult,
    assert_suite,
    render_github_summary,
    write_github_summary,
)


def _result(*, property, passed, seed=42):
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
        seed=seed,
        mcse=0.003,
    )


def test_suite_aggregates_pass_fail_counts():
    suite = StatCISuiteResult.from_results(
        [
            _result(property="type1_error", passed=True),
            _result(property="coverage", passed=False),
            _result(property="bias", passed=True),
        ],
        name="core stats",
    )

    assert suite.name == "core stats"
    assert suite.total == 3
    assert suite.passed_count == 2
    assert suite.failed_count == 1
    assert not suite.passed
    assert suite.status == "FAIL"


def test_passing_suite_reports_pass():
    suite = StatCISuiteResult.from_results(
        [
            _result(property="type1_error", passed=True),
            _result(property="coverage", passed=True),
        ]
    )

    assert suite.passed
    assert suite.status == "PASS"
    assert assert_suite(suite) is suite


def test_empty_suite_is_rejected():
    with pytest.raises(ValueError, match="at least one"):
        StatCISuiteResult(name="empty", results=())


def test_suite_requires_statci_results():
    with pytest.raises(TypeError, match="StatCIResult"):
        StatCISuiteResult(name="bad", results=(_result(property="x", passed=True), object()))


def test_suite_json_is_deterministic_and_embeds_children(tmp_path):
    suite = StatCISuiteResult.from_results(
        [
            _result(property="type1_error", passed=True, seed=1),
            _result(property="coverage", passed=False, seed=2),
        ],
        name="nightly",
    )

    first = suite.to_json()
    second = suite.to_json()
    assert first == second

    parsed = json.loads(first)
    assert parsed["name"] == "nightly"
    assert parsed["status"] == "FAIL"
    assert parsed["total"] == 2
    assert parsed["passed_count"] == 1
    assert parsed["failed_count"] == 1
    assert [result["property"] for result in parsed["results"]] == [
        "type1_error",
        "coverage",
    ]

    path = suite.write_json(tmp_path / "suite.json")
    assert path.read_text(encoding="utf-8") == first + "\n"


def test_assert_suite_raises_with_full_suite_payload():
    suite = StatCISuiteResult.from_results(
        [
            _result(property="type1_error", passed=True),
            _result(property="coverage", passed=False),
        ],
        name="release gate",
    )

    with pytest.raises(StatCISuiteError) as exc_info:
        assert_suite(suite)

    error = exc_info.value
    assert isinstance(error, AssertionError)
    assert error.suite is suite
    assert "1/2 checks failed" in str(error)
    assert "release gate" in str(error)


def test_github_summary_contains_overall_table_and_failed_section():
    suite = StatCISuiteResult.from_results(
        [
            _result(property="type1_error", passed=True, seed=1),
            _result(property="coverage", passed=False, seed=2),
        ],
        name="PR statistical checks",
    )

    summary = render_github_summary(suite)

    assert "## StatCI — PR statistical checks" in summary
    assert "**Overall:** FAIL" in summary
    assert "1 passed / 1 failed / 2 total" in summary
    assert "| Property | Status | Observed | Target | Tolerance |" in summary
    assert "| type1_error | PASS |" in summary
    assert "| coverage | FAIL |" in summary
    assert "### Failed checks" in summary
    assert "simulations" not in summary.lower() or "Simulations" in summary


def test_github_summary_escapes_markdown_table_breakers():
    suite = StatCISuiteResult.from_results(
        [
            _result(property="coverage|evil\nrow", passed=False),
        ],
        name="suite|name",
    )

    summary = render_github_summary(suite)

    assert "suite\\|name" in summary
    assert "coverage\\|evil<br>row" in summary


def test_write_github_summary_appends_to_explicit_path(tmp_path):
    suite = StatCISuiteResult.from_results(
        [_result(property="type1_error", passed=True)]
    )
    path = tmp_path / "summary.md"
    path.write_text("# Existing\n", encoding="utf-8")

    returned = write_github_summary(suite, path)
    content = path.read_text(encoding="utf-8")

    assert returned == path
    assert content.startswith("# Existing\n")
    assert content.count("## StatCI") == 1


def test_write_github_summary_uses_environment_variable(tmp_path, monkeypatch):
    suite = StatCISuiteResult.from_results(
        [_result(property="type1_error", passed=True)]
    )
    path = tmp_path / "github-summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(path))

    returned = write_github_summary(suite)

    assert returned == path
    assert "**Overall:** PASS" in path.read_text(encoding="utf-8")


def test_write_github_summary_requires_path_or_environment(monkeypatch):
    suite = StatCISuiteResult.from_results(
        [_result(property="type1_error", passed=True)]
    )
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)

    with pytest.raises(RuntimeError, match="GITHUB_STEP_SUMMARY"):
        write_github_summary(suite)


def test_summary_renderer_requires_suite():
    with pytest.raises(TypeError, match="StatCISuiteResult"):
        render_github_summary(object())


def test_assert_suite_requires_suite():
    with pytest.raises(TypeError, match="StatCISuiteResult"):
        assert_suite(object())
