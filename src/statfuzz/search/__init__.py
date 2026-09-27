"""Search primitives for StatFuzz."""

from .budget import DiscoveryBudget
from .discovery import CounterexampleDiscoveryResult, find_counterexample
from .grid import GridSearchResult, grid_search
from .objective import (
    AbsoluteDeviationObjective,
    NegativeDeviationObjective,
    PositiveDeviationObjective,
    SearchObjective,
)
from .random import RandomSearchResult, random_search
from .result import SearchRecord, SearchResult
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
    "RandomSearchResult",
    "SearchObjective",
    "SearchRecord",
    "SearchResult",
    "find_counterexample",
    "grid_search",
    "random_search",
    "validate_candidate",
]
