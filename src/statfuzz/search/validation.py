from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ..result import StressTestResult
from .grid import _point_seed
from .result import SearchResult
from .space import ParameterPoint

ValidationEvaluator = Callable[[ParameterPoint, int], StressTestResult]


@dataclass(frozen=True)
class CandidateValidationResult:
    """Search-stage and independent validation-stage results for one candidate."""

    candidate_rank: int
    point: ParameterPoint
    search_result: StressTestResult
    validation_result: StressTestResult
    search_root_seed: int | None
    validation_root_seed: int
    search_seed: int | None
    validation_seed: int

    @property
    def search_absolute_deviation(self) -> float:
        return abs(self.search_result.deviation)

    @property
    def validation_absolute_deviation(self) -> float:
        return abs(self.validation_result.deviation)

    def as_row(self) -> dict[str, object]:
        row: dict[str, object] = {
            f"param:{name}": value for name, value in self.point.items
        }
        row.update(
            {
                "candidate_rank": self.candidate_rank,
                "search_root_seed": self.search_root_seed,
                "validation_root_seed": self.validation_root_seed,
                "search_seed": self.search_seed,
                "validation_seed": self.validation_seed,
                "search_simulations": self.search_result.simulations,
                "validation_simulations": self.validation_result.simulations,
                "search_empirical": self.search_result.empirical,
                "validation_empirical": self.validation_result.empirical,
                "nominal": self.validation_result.nominal,
                "search_deviation": self.search_result.deviation,
                "validation_deviation": self.validation_result.deviation,
                "search_mcse": self.search_result.mcse,
                "validation_mcse": self.validation_result.mcse,
                "search_status": self.search_result.status,
                "validation_status": self.validation_result.status,
            }
        )
        return row


def validate_candidate(
    *,
    search: SearchResult,
    evaluate: ValidationEvaluator,
    validation_root_seed: int,
    rank: int = 0,
) -> CandidateValidationResult:
    """Re-simulate one search-selected point with an independent random stream.

    The candidate is selected from search.ranked(). The validation evaluator
    receives the exact same ParameterPoint but a new deterministic child seed
    derived from validation_root_seed.

    The validation evaluator should normally use a larger simulation budget.
    Search and validation estimates are retained separately.
    """

    if validation_root_seed < 0:
        raise ValueError("validation_root_seed must be non-negative")
    if search.root_seed is not None and validation_root_seed == search.root_seed:
        raise ValueError(
            "validation_root_seed must differ from the search root seed "
            "to enforce independent validation"
        )
    if rank < 0:
        raise ValueError("rank must be non-negative")

    ranked = search.ranked()
    if not ranked:
        raise ValueError("cannot validate a candidate from an empty search result")
    if rank >= len(ranked):
        raise IndexError("rank is outside the available search results")

    selected = ranked[rank]
    validation_seed = _point_seed(validation_root_seed, selected.point)
    if validation_seed is None:
        raise RuntimeError("validation seed derivation unexpectedly returned None")
    if validation_seed == selected.seed:
        raise RuntimeError("validation seed collided with the search seed")

    validation_result = evaluate(selected.point, validation_seed)
    if not isinstance(validation_result, StressTestResult):
        raise TypeError("evaluate(point, seed) must return a StressTestResult")
    if validation_result.seed != validation_seed:
        raise ValueError(
            "the validation evaluator must pass the provided seed through to "
            "stress_test so validation is reproducible"
        )

    return CandidateValidationResult(
        candidate_rank=rank,
        point=selected.point,
        search_result=selected.result,
        validation_result=validation_result,
        search_root_seed=search.root_seed,
        validation_root_seed=validation_root_seed,
        search_seed=selected.seed,
        validation_seed=validation_seed,
    )
