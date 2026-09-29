from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass

from ..result import StatisticalPropertyResult
from .grid import _point_seed
from .objective import ObjectiveLike, resolve_objective
from .result import SearchRecord, SearchResult
from .space import ParameterPoint, ParameterSpace

Evaluator = Callable[[ParameterPoint, int | None], StatisticalPropertyResult]
_UINT64_SIZE = 1 << 64


def _uniform_offset(*, seed: int, step: int, bound: int) -> int:
    """Return an unbiased deterministic integer in range(bound)."""

    if bound <= 0:
        raise ValueError("bound must be positive")

    limit = _UINT64_SIZE - (_UINT64_SIZE % bound)
    attempt = 0
    while True:
        payload = f"{seed}:{step}:{attempt}".encode()
        digest = hashlib.blake2b(
            payload,
            digest_size=8,
            person=b"statfuzz-rand",
        ).digest()
        value = int.from_bytes(digest, byteorder="big", signed=False)
        if value < limit:
            return value % bound
        attempt += 1


def _sample_indices(*, population_size: int, draws: int, seed: int) -> tuple[int, ...]:
    """Sample ordered unique indices using a sparse partial Fisher-Yates shuffle."""

    if seed < 0:
        raise ValueError("seed must be non-negative")
    if population_size > _UINT64_SIZE:
        raise ValueError(
            "random_search currently supports ParameterSpace sizes up to 2**64"
        )
    if draws <= 0:
        raise ValueError("draws must be positive")
    if draws > population_size:
        raise ValueError(
            f"draws={draws} exceeds ParameterSpace size {population_size}"
        )

    swaps: dict[int, int] = {}
    sampled: list[int] = []

    for step in range(draws):
        offset = _uniform_offset(
            seed=seed,
            step=step,
            bound=population_size - step,
        )
        selected_position = step + offset

        left_value = swaps.get(step, step)
        selected_value = swaps.get(selected_position, selected_position)

        swaps[step] = selected_value
        swaps[selected_position] = left_value
        sampled.append(selected_value)

    return tuple(sampled)


@dataclass(frozen=True)
class RandomSearchResult(SearchResult):
    """Search result for a reproducible subset sampled without replacement."""

    sampled_indices: tuple[int, ...]
    space_size: int

    @property
    def draws(self) -> int:
        return len(self.sampled_indices)

    @property
    def coverage_fraction(self) -> float:
        return self.draws / self.space_size


def random_search(
    *,
    space: ParameterSpace,
    evaluate: Evaluator,
    draws: int,
    seed: int = 0,
    objective: ObjectiveLike = "absolute_deviation",
) -> RandomSearchResult:
    """Evaluate a reproducible subset of a finite ParameterSpace.

    Points are sampled without replacement using a deterministic sparse
    partial Fisher-Yates shuffle. The full Cartesian product is never
    materialized.

    The same root seed also derives deterministic per-point evaluation seeds,
    so a sampled point receives the same evaluation seed regardless of the
    order in which sampled points are evaluated.
    """

    if seed < 0:
        raise ValueError("seed must be non-negative")

    resolved_objective = resolve_objective(objective)
    sampled_indices = _sample_indices(
        population_size=len(space),
        draws=draws,
        seed=seed,
    )

    records: list[SearchRecord] = []
    for index in sampled_indices:
        point = space.point_at(index)
        child_seed = _point_seed(seed, point)
        if child_seed is None:
            raise RuntimeError("random-search child seed unexpectedly resolved to None")

        result = evaluate(point, child_seed)
        if not isinstance(result, StatisticalPropertyResult):
            raise TypeError("evaluate(point, seed) must return a StatisticalPropertyResult")
        if result.seed != child_seed:
            raise ValueError(
                "the evaluator must pass the provided seed through to stress_test "
                "so the search remains reproducible"
            )

        record = SearchRecord(point=point, result=result, seed=child_seed)
        record.objective_score(resolved_objective)
        records.append(record)

    return RandomSearchResult(
        records=tuple(records),
        objective=resolved_objective,
        root_seed=seed,
        parameter_names=space.names,
        sampled_indices=sampled_indices,
        space_size=len(space),
    )
