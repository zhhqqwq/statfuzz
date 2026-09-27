"""Search primitives for StatFuzz."""

from .discovery import CounterexampleDiscoveryResult, find_counterexample
from .grid import GridSearchResult, SearchRecord, grid_search
from .space import ParameterPoint, ParameterSpace
from .validation import CandidateValidationResult, validate_candidate

__all__ = [
    "CandidateValidationResult",
    "CounterexampleDiscoveryResult",
    "GridSearchResult",
    "ParameterPoint",
    "ParameterSpace",
    "SearchRecord",
    "find_counterexample",
    "grid_search",
    "validate_candidate",
]
