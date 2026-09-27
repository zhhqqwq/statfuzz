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
from .shrink import (
    CounterexampleShrinkResult,
    FailureCriterion,
    ObjectiveThresholdCriterion,
    OutsideToleranceCriterion,
    ShrinkDimension,
    ShrinkPlan,
    ShrinkStep,
    shrink_counterexample,
)
from .space import ParameterPoint, ParameterSpace
from .validation import CandidateValidationResult, validate_candidate

__all__ = [
    "AbsoluteDeviationObjective",
    "CandidateValidationResult",
    "CounterexampleDiscoveryResult",
    "CounterexampleShrinkResult",
    "DiscoveryBudget",
    "FailureCriterion",
    "GridSearchResult",
    "NegativeDeviationObjective",
    "ObjectiveThresholdCriterion",
    "OutsideToleranceCriterion",
    "ParameterPoint",
    "ParameterSpace",
    "PositiveDeviationObjective",
    "RandomSearchResult",
    "SearchObjective",
    "SearchRecord",
    "SearchResult",
    "ShrinkDimension",
    "ShrinkPlan",
    "ShrinkStep",
    "find_counterexample",
    "grid_search",
    "random_search",
    "shrink_counterexample",
    "validate_candidate",
]
