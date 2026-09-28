"""StatFuzz: stress testing for statistical methods."""

from .result import StressTestResult
from .simulation import stress_test
from .statci import (
    STATCI_SCHEMA_VERSION,
    STATCI_STATUS_SCHEMA_VERSION,
    StatCIResult,
    StatCIStatusArtifact,
    StatCISuiteError,
    StatCISuiteResult,
    StatisticalAssertion,
    StatisticalAssertionError,
    assert_property,
    assert_suite,
    check_property,
    render_github_summary,
    write_github_summary,
)

__all__ = [
    "STATCI_SCHEMA_VERSION",
    "STATCI_STATUS_SCHEMA_VERSION",
    "StatCIResult",
    "StatCIStatusArtifact",
    "StatCISuiteError",
    "StatCISuiteResult",
    "StatisticalAssertion",
    "StatisticalAssertionError",
    "StressTestResult",
    "assert_property",
    "assert_suite",
    "check_property",
    "render_github_summary",
    "stress_test",
    "write_github_summary",
]
__version__ = "0.1.0"
