from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ..result import StressTestResult
from .grid import GridSearchResult, Objective, grid_search
from .space import ParameterPoint, ParameterSpace
from .validation import CandidateValidationResult, validate_candidate

DiscoveryEvaluator = Callable[[ParameterPoint, int], StressTestResult]


@dataclass(frozen=True)
class CounterexampleDiscoveryResult:
    """Complete search and independent-validation record for one discovery run."""

    search: GridSearchResult
    validation: CandidateValidationResult

    @property
    def point(self) -> ParameterPoint:
        return self.validation.point

    @property
    def search_result(self) -> StressTestResult:
        return self.validation.search_result

    @property
    def validation_result(self) -> StressTestResult:
        return self.validation.validation_result

    def as_row(self) -> dict[str, object]:
        row = self.validation.as_row()
        row.update(
            {
                "objective": self.search.objective,
                "search_points": len(self.search.records),
            }
        )
        return row


def find_counterexample(
    *,
    space: ParameterSpace,
    search_evaluate: DiscoveryEvaluator,
    validation_evaluate: DiscoveryEvaluator,
    search_root_seed: int = 0,
    validation_root_seed: int = 1,
    objective: Objective = "absolute_deviation",
) -> CounterexampleDiscoveryResult:
    """Search a finite parameter space and independently validate the top candidate.

    The search stage evaluates every point using search_root_seed and the
    requested exploratory objective. The top-ranked point is then re-simulated
    with validation_root_seed through the independent validation API.

    The returned object keeps the complete search table and both stage-specific
    Monte Carlo estimates. It does not merge the estimates or claim a
    mathematical proof of a universal counterexample.
    """

    if search_root_seed < 0:
        raise ValueError("search_root_seed must be non-negative")
    if validation_root_seed < 0:
        raise ValueError("validation_root_seed must be non-negative")
    if search_root_seed == validation_root_seed:
        raise ValueError("search_root_seed and validation_root_seed must differ")

    search = grid_search(
        space=space,
        evaluate=search_evaluate,
        seed=search_root_seed,
        objective=objective,
    )
    validation = validate_candidate(
        search=search,
        evaluate=validation_evaluate,
        validation_root_seed=validation_root_seed,
        rank=0,
    )

    return CounterexampleDiscoveryResult(search=search, validation=validation)
