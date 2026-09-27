from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass

from ..result import StressTestResult
from .objective import ObjectiveLike, resolve_objective
from .result import SearchRecord, SearchResult
from .space import ParameterPoint, ParameterSpace

Evaluator = Callable[[ParameterPoint, int | None], StressTestResult]


def _point_seed(root_seed: int | None, point: ParameterPoint) -> int | None:
    if root_seed is None:
        return None
    if root_seed < 0:
        raise ValueError("seed must be non-negative or None")

    payload = f"{root_seed}:{point.to_json()}".encode()
    digest = hashlib.blake2b(payload, digest_size=8, person=b"statfuzz").digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


@dataclass(frozen=True)
class GridSearchResult(SearchResult):
    """Exhaustive search result over every point in a ParameterSpace."""


def grid_search(
    *,
    space: ParameterSpace,
    evaluate: Evaluator,
    seed: int | None = 0,
    objective: ObjectiveLike = "absolute_deviation",
) -> GridSearchResult:
    """Evaluate every point in a finite parameter space.

    The caller supplies evaluate(point, seed) and must return a StressTestResult.
    The search layer does not know how to build DGPs or run statistical methods.

    Each point receives a deterministic child seed derived from the root seed
    and the canonical serialized point. Results are exploratory: important
    candidates should later be confirmed with independent validation draws.
    """

    if seed is not None and seed < 0:
        raise ValueError("seed must be non-negative or None")
    resolved_objective = resolve_objective(objective)

    records: list[SearchRecord] = []
    for point in space:
        child_seed = _point_seed(seed, point)
        result = evaluate(point, child_seed)
        if not isinstance(result, StressTestResult):
            raise TypeError("evaluate(point, seed) must return a StressTestResult")
        if result.seed != child_seed:
            raise ValueError(
                "the evaluator must pass the provided seed through to stress_test "
                "so the search remains reproducible"
            )
        record = SearchRecord(point=point, result=result, seed=child_seed)
        record.objective_score(resolved_objective)
        records.append(record)

    return GridSearchResult(
        records=tuple(records),
        objective=resolved_objective,
        root_seed=seed,
        parameter_names=space.names,
    )
