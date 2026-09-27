from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from dataclasses import dataclass

from ..result import StressTestResult
from .objective import ObjectiveLike, SearchObjective, resolve_objective
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
class SearchRecord:
    point: ParameterPoint
    result: StressTestResult
    seed: int | None

    @property
    def absolute_deviation(self) -> float:
        return abs(self.result.deviation)

    def objective_score(self, objective: SearchObjective) -> float:
        score = float(objective.score(self.result))
        if not math.isfinite(score):
            raise ValueError(
                f"objective {objective.name!r} returned a non-finite score"
            )
        return score

    def as_row(self, objective: SearchObjective | None = None) -> dict[str, object]:
        row: dict[str, object] = {
            f"param:{name}": value for name, value in self.point.items
        }
        row.update(
            {
                "nominal": self.result.nominal,
                "empirical": self.result.empirical,
                "deviation": self.result.deviation,
                "absolute_deviation": self.absolute_deviation,
                "mcse": self.result.mcse,
                "simulations": self.result.simulations,
                "seed": self.seed,
                "status": self.result.status,
            }
        )
        if objective is not None:
            row["objective"] = objective.name
            row["objective_score"] = self.objective_score(objective)
        return row


@dataclass(frozen=True)
class GridSearchResult:
    records: tuple[SearchRecord, ...]
    objective: SearchObjective
    root_seed: int | None
    parameter_names: tuple[str, ...]

    @property
    def objective_name(self) -> str:
        return self.objective.name

    def ranked(self) -> tuple[SearchRecord, ...]:
        """Return records from largest to smallest objective score."""

        return tuple(
            sorted(
                self.records,
                key=lambda record: record.objective_score(self.objective),
                reverse=True,
            )
        )

    def to_rows(self, *, ranked: bool = False) -> list[dict[str, object]]:
        records = self.ranked() if ranked else self.records
        return [record.as_row(self.objective) for record in records]

    def to_markdown(self, *, ranked: bool = True, digits: int = 4) -> str:
        rows = self.to_rows(ranked=ranked)
        if not rows:
            return ""

        headers = list(rows[0])
        rendered: list[list[str]] = []
        for row in rows:
            cells: list[str] = []
            for header in headers:
                value = row[header]
                if isinstance(value, float):
                    cells.append(f"{value:.{digits}f}")
                elif value is None:
                    cells.append("None")
                else:
                    cells.append(str(value))
            rendered.append(cells)

        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ]
        lines.extend("| " + " | ".join(cells) + " |" for cells in rendered)
        return "\n".join(lines)


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
