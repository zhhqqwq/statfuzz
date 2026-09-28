"""StatFuzz: stress testing for statistical methods."""

from ._version import __version__
from .nulls import MeanEqualityNull, MeanNullCheck
from .result import StressTestResult
from .simulation import stress_test
from .statci import (
    RegressionPolicy,
    StatCIRegressionError,
    StatCIResult,
    StatCIStatusArtifact,
    StatCISuiteError,
    StatCISuiteResult,
    StatisticalAssertion,
    StatisticalAssertionError,
    assert_property,
    assert_regression_suite,
    assert_suite,
    check_property,
    compare_suites,
    write_github_summary,
    write_regression_summary,
)

__all__ = [
    "MeanEqualityNull",
    "MeanNullCheck",
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
]
