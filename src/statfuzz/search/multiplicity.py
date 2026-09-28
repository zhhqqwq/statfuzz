from __future__ import annotations

import math
from dataclasses import dataclass

from .result import SearchResult
from .validation import CandidateValidationResult


def _quantile(sorted_values: tuple[float, ...], probability: float) -> float:
    if not sorted_values:
        raise ValueError("quantile requires at least one value")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between 0 and 1")
    if len(sorted_values) == 1:
        return sorted_values[0]

    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]

    weight = position - lower
    return (
        sorted_values[lower] * (1.0 - weight)
        + sorted_values[upper] * weight
    )


@dataclass(frozen=True)
class SearchMultiplicitySummary:
    """Descriptive multiplicity metadata for one search-selected candidate."""

    objective: str
    evaluated_points: int
    candidate_rank: int
    candidate_score: float
    tie_count: int
    empirical_percentile: float
    empirical_upper_tail_fraction: float
    objective_min: float
    objective_median: float
    objective_p90: float
    objective_p95: float
    objective_max: float
    candidate_minus_median: float
    candidate_minus_p95: float
    outside_tolerance_count: int
    outside_tolerance_fraction: float

    @property
    def candidate_rank_one_based(self) -> int:
        return self.candidate_rank + 1

    @property
    def selection_opportunities(self) -> int:
        return self.evaluated_points

    def as_dict(self) -> dict[str, object]:
        return {
            "objective": self.objective,
            "evaluated_points": self.evaluated_points,
            "selection_opportunities": self.selection_opportunities,
            "candidate_rank": self.candidate_rank,
            "candidate_rank_one_based": self.candidate_rank_one_based,
            "candidate_score": self.candidate_score,
            "tie_count": self.tie_count,
            "empirical_percentile": self.empirical_percentile,
            "empirical_upper_tail_fraction": self.empirical_upper_tail_fraction,
            "objective_min": self.objective_min,
            "objective_median": self.objective_median,
            "objective_p90": self.objective_p90,
            "objective_p95": self.objective_p95,
            "objective_max": self.objective_max,
            "candidate_minus_median": self.candidate_minus_median,
            "candidate_minus_p95": self.candidate_minus_p95,
            "outside_tolerance_count": self.outside_tolerance_count,
            "outside_tolerance_fraction": self.outside_tolerance_fraction,
        }


@dataclass(frozen=True)
class SelectionEffectDiagnostic:
    """Search-to-validation objective change for one selected candidate.

    This is a descriptive diagnostic, not an unbiased estimate of selection bias.
    """

    objective: str
    evaluated_points: int
    candidate_rank: int
    search_score: float
    validation_score: float
    search_minus_validation_gap: float
    validation_minus_search_change: float
    search_mcse: float
    validation_mcse: float
    search_simulations: int
    validation_simulations: int
    search_status: str
    validation_status: str
    multiplicity: SearchMultiplicitySummary

    def as_dict(self) -> dict[str, object]:
        return {
            "objective": self.objective,
            "evaluated_points": self.evaluated_points,
            "candidate_rank": self.candidate_rank,
            "candidate_rank_one_based": self.candidate_rank + 1,
            "search_score": self.search_score,
            "validation_score": self.validation_score,
            "search_minus_validation_gap": self.search_minus_validation_gap,
            "validation_minus_search_change": self.validation_minus_search_change,
            "search_mcse": self.search_mcse,
            "validation_mcse": self.validation_mcse,
            "search_simulations": self.search_simulations,
            "validation_simulations": self.validation_simulations,
            "search_status": self.search_status,
            "validation_status": self.validation_status,
            "multiplicity": self.multiplicity.as_dict(),
        }


def summarize_search_multiplicity(
    search: SearchResult,
    *,
    candidate_rank: int = 0,
) -> SearchMultiplicitySummary:
    """Summarize observed extremeness after selecting from multiple search points.

    The empirical percentile and upper-tail fraction are descriptive ranks among
    the evaluated objective scores. They are not p-values and do not provide
    family-wise error control.
    """

    if not isinstance(search, SearchResult):
        raise TypeError("search must be a SearchResult")
    if candidate_rank < 0:
        raise ValueError("candidate_rank must be non-negative")
    if not search.records:
        raise ValueError("cannot summarize multiplicity for an empty search")

    ranked = search.ranked()
    if candidate_rank >= len(ranked):
        raise IndexError("candidate_rank is outside the available search results")

    scores = tuple(
        record.objective_score(search.objective)
        for record in search.records
    )
    sorted_scores = tuple(sorted(scores))
    candidate = ranked[candidate_rank]
    candidate_score = candidate.objective_score(search.objective)
    n = len(scores)

    tie_count = sum(score == candidate_score for score in scores)
    less_or_equal = sum(score <= candidate_score for score in scores)
    greater_or_equal = sum(score >= candidate_score for score in scores)
    outside_tolerance_count = sum(
        not record.result.passed
        for record in search.records
    )

    median = _quantile(sorted_scores, 0.5)
    p90 = _quantile(sorted_scores, 0.9)
    p95 = _quantile(sorted_scores, 0.95)

    return SearchMultiplicitySummary(
        objective=search.objective_name,
        evaluated_points=n,
        candidate_rank=candidate_rank,
        candidate_score=candidate_score,
        tie_count=tie_count,
        empirical_percentile=less_or_equal / n,
        empirical_upper_tail_fraction=greater_or_equal / n,
        objective_min=sorted_scores[0],
        objective_median=median,
        objective_p90=p90,
        objective_p95=p95,
        objective_max=sorted_scores[-1],
        candidate_minus_median=candidate_score - median,
        candidate_minus_p95=candidate_score - p95,
        outside_tolerance_count=outside_tolerance_count,
        outside_tolerance_fraction=outside_tolerance_count / n,
    )


def summarize_selection_effect(
    search: SearchResult,
    validation: CandidateValidationResult,
) -> SelectionEffectDiagnostic:
    """Compare a selected search score with its independent validation score.

    A positive search_minus_validation_gap means the independently re-estimated
    objective was less extreme than the search-stage estimate. The gap mixes
    selection, Monte Carlo noise, validation budget, and genuine local behavior;
    it must not be interpreted as an unbiased winner's-curse estimate.
    """

    if not isinstance(search, SearchResult):
        raise TypeError("search must be a SearchResult")
    if not isinstance(validation, CandidateValidationResult):
        raise TypeError("validation must be a CandidateValidationResult")

    multiplicity = summarize_search_multiplicity(
        search,
        candidate_rank=validation.candidate_rank,
    )
    selected = search.ranked()[validation.candidate_rank]
    if selected.point != validation.point:
        raise ValueError("validation point does not match the selected search candidate")
    if selected.result != validation.search_result:
        raise ValueError("validation search_result does not match the search candidate")

    validation_score = float(search.objective.score(validation.validation_result))
    if not math.isfinite(validation_score):
        raise ValueError(
            f"objective {search.objective_name!r} returned a non-finite "
            "validation score"
        )

    search_score = multiplicity.candidate_score
    return SelectionEffectDiagnostic(
        objective=search.objective_name,
        evaluated_points=len(search.records),
        candidate_rank=validation.candidate_rank,
        search_score=search_score,
        validation_score=validation_score,
        search_minus_validation_gap=search_score - validation_score,
        validation_minus_search_change=validation_score - search_score,
        search_mcse=validation.search_result.mcse,
        validation_mcse=validation.validation_result.mcse,
        search_simulations=validation.search_result.simulations,
        validation_simulations=validation.validation_result.simulations,
        search_status=validation.search_result.status,
        validation_status=validation.validation_result.status,
        multiplicity=multiplicity,
    )
