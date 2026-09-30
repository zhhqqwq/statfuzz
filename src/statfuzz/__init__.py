"""StatFuzz: stress testing for statistical methods."""

from ._version import __version__
from .bootstrap_coverage import BootstrapCoverageResult
from .nulls import MeanEqualityNull, MeanNullCheck
from .result import StressTestResult
from .simulation import SimulationProgress, stress_test
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
from .targets import MeanTarget, MeanTargetCheck

__all__ = [
    "BootstrapCoverageResult",
    "MeanEqualityNull",
    "MeanNullCheck",
    "MeanTarget",
    "MeanTargetCheck",
    "RegressionPolicy",
    "SimulationProgress",
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
