"""StatFuzz: stress testing for statistical methods."""

from .result import StressTestResult
from .simulation import stress_test

__all__ = ["StressTestResult", "stress_test"]
__version__ = "0.1.0"
