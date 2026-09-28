"""Statistical assertions for continuous integration."""

from .artifact import STATCI_STATUS_SCHEMA_VERSION, StatCIStatusArtifact
from .assertions import StatisticalAssertionError, assert_property, check_property
from .github import render_github_summary, write_github_summary
from .model import STATCI_SCHEMA_VERSION, StatCIResult, StatisticalAssertion
from .suite import StatCISuiteError, StatCISuiteResult, assert_suite

__all__ = [
    "STATCI_SCHEMA_VERSION",
    "STATCI_STATUS_SCHEMA_VERSION",
    "StatCIResult",
    "StatCISuiteError",
    "StatCISuiteResult",
    "StatCIStatusArtifact",
    "StatisticalAssertion",
    "StatisticalAssertionError",
    "assert_property",
    "assert_suite",
    "check_property",
    "render_github_summary",
    "write_github_summary",
]
