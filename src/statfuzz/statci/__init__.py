"""Statistical assertions for continuous integration."""

from .artifact import STATCI_STATUS_SCHEMA_VERSION, StatCIStatusArtifact
from .assertions import StatisticalAssertionError, assert_property, check_property
from .github import render_github_summary, write_github_summary
from .model import STATCI_SCHEMA_VERSION, StatCIResult, StatisticalAssertion
from .regression import (
    STATCI_REGRESSION_SCHEMA_VERSION,
    BootstrapCoverageComparisonKey,
    RegressionPolicy,
    StatCIComparisonKey,
    StatCIRegressionError,
    StatCIRegressionResult,
    StatCIRegressionSuiteResult,
    assert_regression_suite,
    compare_results,
    compare_suites,
    render_regression_summary,
    write_regression_summary,
)
from .suite import StatCISuiteError, StatCISuiteResult, assert_suite

__all__ = [
    "STATCI_REGRESSION_SCHEMA_VERSION",
    "STATCI_SCHEMA_VERSION",
    "STATCI_STATUS_SCHEMA_VERSION",
    "BootstrapCoverageComparisonKey",
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
]
