from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ..result import StatisticalPropertyResult
from .budget import DiscoveryBudget
from .grid import grid_search
from .multiplicity import summarize_selection_effect
from .objective import ObjectiveLike
from .random import random_search
from .result import SearchResult
from .space import ParameterPoint, ParameterSpace
from .validation import CandidateValidationResult, validate_candidate

DiscoveryEvaluator = Callable[[ParameterPoint, int, int], StatisticalPropertyResult]


@dataclass(frozen=True)
class CounterexampleDiscoveryResult:
    """Complete search and independent-validation record for one discovery run."""

    search: SearchResult
    validation: CandidateValidationResult
    budget: DiscoveryBudget

    @property
    def point(self) -> ParameterPoint:
        return self.validation.point

    @property
    def search_result(self) -> StatisticalPropertyResult:
        return self.validation.search_result

    @property
    def validation_result(self) -> StatisticalPropertyResult:
        return self.validation.validation_result

    def as_row(self) -> dict[str, object]:
        row = self.validation.as_row()
        selected = self.search.ranked()[self.validation.candidate_rank]
        diagnostic = summarize_selection_effect(self.search, self.validation)
        row.update(
            {
                "objective": self.search.objective_name,
                "objective_score": selected.objective_score(self.search.objective),
                "search_points": len(self.search.records),
                "search_budget": self.budget.search_simulations,
                "validation_budget": self.budget.validation_simulations,
                "search_candidate_rank": diagnostic.candidate_rank,
                "search_candidate_rank_one_based": diagnostic.candidate_rank + 1,
                "search_empirical_percentile": (
                    diagnostic.multiplicity.empirical_percentile
                ),
                "search_empirical_upper_tail_fraction": (
                    diagnostic.multiplicity.empirical_upper_tail_fraction
                ),
                "search_objective_median": (
                    diagnostic.multiplicity.objective_median
                ),
                "search_objective_p95": diagnostic.multiplicity.objective_p95,
                "validation_objective_score": diagnostic.validation_score,
                "search_minus_validation_gap": (
                    diagnostic.search_minus_validation_gap
                ),
            }
        )
        return row


def _run_budgeted(
    *,
    evaluate: DiscoveryEvaluator,
    point: ParameterPoint,
    seed: int,
    simulations: int,
    stage: str,
) -> StatisticalPropertyResult:
    result = evaluate(point, seed, simulations)
    if not isinstance(result, StatisticalPropertyResult):
        raise TypeError(
            f"{stage} evaluator must return a StatisticalPropertyResult"
        )
    if result.simulations != simulations:
        raise ValueError(
            f"{stage} evaluator returned simulations={result.simulations}, "
            f"expected the requested budget {simulations}"
        )
    return result


def find_counterexample(
    *,
    space: ParameterSpace,
    search_evaluate: DiscoveryEvaluator,
    validation_evaluate: DiscoveryEvaluator,
    budget: DiscoveryBudget,
    search_root_seed: int = 0,
    validation_root_seed: int = 1,
    objective: ObjectiveLike = "absolute_deviation",
    search_draws: int | None = None,
) -> CounterexampleDiscoveryResult:
    """Search a finite parameter space and independently validate the top candidate.

    The discovery budget is executable: its search simulation count is passed
    into every search evaluation and its validation count is passed into the
    independent validation evaluation. If search_draws is None, discovery uses
    exhaustive grid search. Otherwise it samples search_draws points without
    replacement using random_search.

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

    def search_stage(point: ParameterPoint, seed: int | None) -> StatisticalPropertyResult:
        if seed is None:
            raise RuntimeError("search seed unexpectedly resolved to None")
        return _run_budgeted(
            evaluate=search_evaluate,
            point=point,
            seed=seed,
            simulations=budget.search_simulations,
            stage="search",
        )

    def validation_stage(point: ParameterPoint, seed: int) -> StatisticalPropertyResult:
        return _run_budgeted(
            evaluate=validation_evaluate,
            point=point,
            seed=seed,
            simulations=budget.validation_simulations,
            stage="validation",
        )

    if search_draws is None:
        search = grid_search(
            space=space,
            evaluate=search_stage,
            seed=search_root_seed,
            objective=objective,
        )
    else:
        search = random_search(
            space=space,
            evaluate=search_stage,
            draws=search_draws,
            seed=search_root_seed,
            objective=objective,
        )
    validation = validate_candidate(
        search=search,
        evaluate=validation_stage,
        validation_root_seed=validation_root_seed,
        rank=0,
    )

    return CounterexampleDiscoveryResult(
        search=search,
        validation=validation,
        budget=budget,
    )
