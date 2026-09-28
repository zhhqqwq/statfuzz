from __future__ import annotations

from ..result import StressTestResult
from .model import StatCIResult, StatisticalAssertion


class StatisticalAssertionError(AssertionError):
    """Pytest-friendly assertion error carrying the failed StatCIResult."""

    def __init__(self, result: StatCIResult) -> None:
        self.result = result
        super().__init__(self._message(result))

    @staticmethod
    def _message(result: StatCIResult) -> str:
        return (
            f"StatCI assertion failed for {result.property!r}: "
            f"observed={result.observed:.6g}, "
            f"target={result.target:.6g}, "
            f"deviation={result.deviation:+.6g}, "
            f"absolute_deviation={result.absolute_deviation:.6g}, "
            f"tolerance={result.tolerance:.6g}, "
            f"mcse={result.mcse:.6g}, "
            f"simulations={result.simulations}, "
            f"seed={result.seed}"
        )


def check_property(
    result: StressTestResult,
    *,
    property: str,
    target: float,
    tolerance: float,
) -> StatCIResult:
    """Evaluate a statistical property assertion without raising on FAIL."""

    assertion = StatisticalAssertion(
        property=property,
        target=target,
        tolerance=tolerance,
    )
    return assertion.evaluate(result)


def assert_property(
    result: StressTestResult,
    *,
    property: str,
    target: float,
    tolerance: float,
) -> StatCIResult:
    """Evaluate a statistical property and raise AssertionError on failure.

    The returned/raised StatCIResult uses the engineering rule
    abs(observed - target) <= tolerance. MCSE is retained as evidence but is not
    folded into the threshold.
    """

    statci_result = check_property(
        result,
        property=property,
        target=target,
        tolerance=tolerance,
    )
    if not statci_result.passed:
        raise StatisticalAssertionError(statci_result)
    return statci_result
