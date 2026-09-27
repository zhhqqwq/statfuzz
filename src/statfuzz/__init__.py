"""StatFuzz: stress testing for statistical methods."""

from .simulation import stress_test
from .result import StressTestResult

__all__ = ["stress_test", "StressTestResult"]
__version__ = "0.1.0"
