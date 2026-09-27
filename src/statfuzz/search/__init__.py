"""Search primitives for StatFuzz."""

from .budget import DiscoveryBudget
from .discovery import CounterexampleDiscoveryResult, find_counterexample
from .grid import GridSearchResult, SearchRecord, grid_search
from .objective import (
    AbsoluteDeviationObjective,
    NegativeDeviationObjective,
    PositiveDeviationObjective,
    SearchObjective,
)
from .space import ParameterPoint, ParameterSpace
from .validation import CandidateValidationResult, validate_candidate

__all__ = [
    "AbsoluteDeviationObjective",
    "CandidateValidationResult",
    "CounterexampleDiscoveryResult",
    "DiscoveryBudget",
    "GridSearchResult",
    "NegativeDeviationObjective",
    "ParameterPoint",
    "ParameterSpace",
    "PositiveDeviationObjective",
    "SearchObjective",
    "SearchRecord",
    "find_counterexample",
    "grid_search",
    "validate_candidate",
]
