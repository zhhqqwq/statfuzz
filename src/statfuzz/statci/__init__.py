"""Statistical assertions for continuous integration."""

from .assertions import StatisticalAssertionError, assert_property, check_property
from .model import STATCI_SCHEMA_VERSION, StatCIResult, StatisticalAssertion

__all__ = [
    "STATCI_SCHEMA_VERSION",
    "StatCIResult",
    "StatisticalAssertion",
    "StatisticalAssertionError",
    "assert_property",
    "check_property",
]
