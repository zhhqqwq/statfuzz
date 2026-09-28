"""Search primitives for StatFuzz."""

from .budget import DiscoveryBudget
from .discovery import CounterexampleDiscoveryResult, find_counterexample
from .family import (
    FamilyPoint,
    FamilyShrinkPlan,
    FamilyShrinkResult,
    FamilyShrinkStep,
    shrink_dgp_family,
)
from .grid import GridSearchResult, grid_search
from .multiplicity import (
    SearchMultiplicitySummary,
    SelectionEffectDiagnostic,
    summarize_search_multiplicity,
    summarize_selection_effect,
)
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
    "FamilyPoint",
    "FamilyShrinkPlan",
    "FamilyShrinkResult",
    "FamilyShrinkStep",
    "GridSearchResult",
    "NegativeDeviationObjective",
    "ObjectiveThresholdCriterion",
    "OutsideToleranceCriterion",
    "ParameterPoint",
    "ParameterSpace",
    "PositiveDeviationObjective",
    "RandomSearchResult",
    "SearchMultiplicitySummary",
    "SearchObjective",
    "SearchRecord",
    "SearchResult",
    "SelectionEffectDiagnostic",
    "ShrinkDimension",
    "ShrinkPlan",
    "ShrinkStep",
    "find_counterexample",
    "grid_search",
    "random_search",
    "shrink_counterexample",
    "shrink_dgp_family",
    "summarize_search_multiplicity",
    "summarize_selection_effect",
    "validate_candidate",
]
