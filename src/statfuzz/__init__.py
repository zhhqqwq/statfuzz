"""StatFuzz: stress testing for statistical methods."""

from .result import StressTestResult
from .simulation import stress_test
from .statci import (
    STATCI_SCHEMA_VERSION,
    StatCIResult,
    StatisticalAssertion,
    StatisticalAssertionError,
    assert_property,
    check_property,
)

__all__ = [
    "STATCI_SCHEMA_VERSION",
    "StatCIResult",
    "StatisticalAssertion",
    "StatisticalAssertionError",
    "StressTestResult",
    "assert_property",
    "check_property",
    "stress_test",
]
__version__ = "0.1.0"
