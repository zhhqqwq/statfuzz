from importlib.metadata import version

import statfuzz
from statfuzz import dgp, report, search, statci


def _assert_public_api(module, expected):
    assert set(module.__all__) == expected
    for name in expected:
        assert hasattr(module, name), f"{module.__name__}.{name} is missing"


def test_package_version_matches_distribution_metadata():
    assert statfuzz.__version__ == "0.1.0"
    assert version("statfuzz") == statfuzz.__version__


def test_top_level_public_api_is_intentionally_small():
    expected = {
        "RegressionPolicy",
        "StatCIRegressionError",
        "StatCIResult",
        "StatCIStatusArtifact",
        "StatCISuiteError",
        "StatCISuiteResult",
        "StatisticalAssertion",
        "StatisticalAssertionError",
        "StressTestResult",
        "__version__",
        "assert_property",
        "assert_regression_suite",
        "assert_suite",
        "check_property",
        "compare_suites",
        "stress_test",
        "write_github_summary",
        "write_regression_summary",
    }
    _assert_public_api(statfuzz, expected)

    assert "STATCI_SCHEMA_VERSION" not in statfuzz.__all__
    assert "StatCIComparisonKey" not in statfuzz.__all__
    assert "StatCIRegressionResult" not in statfuzz.__all__
    assert "compare_results" not in statfuzz.__all__
    assert "render_github_summary" not in statfuzz.__all__


def test_dgp_namespace_public_api():
    expected = {
        "DGPIdentity",
        "DataGenerator",
        "LogNormal",
        "MixtureNormal",
        "Normal",
        "StudentT",
    }
    _assert_public_api(dgp, expected)


def test_search_namespace_public_api():
    expected = {
        "AbsoluteDeviationObjective",
        "CandidateValidationResult",
        "CounterexampleDiscoveryResult",
        "CounterexampleShrinkResult",
        "DiscoveryBudget",
        "FailureCriterion",
        "FamilyPoint",
        "FamilyShrinkPlan",
        "FamilyShrinkResult",
        "FamilyShrinkStep",
        "GridSearchResult",
        "NegativeDeviationObjective",
        "ObjectiveThresholdCriterion",
        "OutsideToleranceCriterion",
        "ParameterPoint",
        "ParameterSpace",
        "PositiveDeviationObjective",
        "RandomSearchResult",
        "SearchMultiplicitySummary",
        "SearchObjective",
        "SearchRecord",
        "SearchResult",
        "SelectionEffectDiagnostic",
        "ShrinkDimension",
        "ShrinkPlan",
        "ShrinkStep",
        "find_counterexample",
        "grid_search",
        "random_search",
        "shrink_counterexample",
        "shrink_dgp_family",
        "summarize_search_multiplicity",
        "summarize_selection_effect",
        "validate_candidate",
    }
    _assert_public_api(search, expected)


def test_report_namespace_exposes_workflow_not_snapshot_internals():
    expected = {
        "REPORT_SCHEMA_VERSION",
        "FailureMap2D",
        "StatFuzzReport",
        "build_report",
        "failure_map_2d",
        "render_html",
        "write_html",
    }
    _assert_public_api(report, expected)

    for internal_name in (
        "FailureMapCell",
        "SearchRecordSnapshot",
        "SearchSnapshot",
        "ShrinkSnapshot",
        "StressTestSnapshot",
        "ValidationSnapshot",
    ):
        assert internal_name not in report.__all__


def test_statci_namespace_exposes_full_advanced_api():
    expected = {
        "STATCI_REGRESSION_SCHEMA_VERSION",
        "STATCI_SCHEMA_VERSION",
        "STATCI_STATUS_SCHEMA_VERSION",
        "RegressionPolicy",
        "StatCIComparisonKey",
        "StatCIRegressionError",
        "StatCIRegressionResult",
        "StatCIRegressionSuiteResult",
        "StatCIResult",
        "StatCIStatusArtifact",
        "StatCISuiteError",
        "StatCISuiteResult",
        "StatisticalAssertion",
        "StatisticalAssertionError",
        "assert_property",
        "assert_regression_suite",
        "assert_suite",
        "check_property",
        "compare_results",
        "compare_suites",
        "render_github_summary",
        "render_regression_summary",
        "write_github_summary",
        "write_regression_summary",
    }
    _assert_public_api(statci, expected)
