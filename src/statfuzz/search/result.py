from __future__ import annotations

import math
from dataclasses import dataclass

from ..result import StatisticalPropertyResult
from .objective import SearchObjective
from .space import ParameterPoint


@dataclass(frozen=True)
class SearchRecord:
    point: ParameterPoint
    result: StatisticalPropertyResult
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
class SearchResult:
    """Common result interface shared by grid and random search."""

    records: tuple[SearchRecord, ...]
    objective: SearchObjective
    root_seed: int | None
    parameter_names: tuple[str, ...]

    @property
    def objective_name(self) -> str:
        return self.objective.name

    def ranked(self) -> tuple[SearchRecord, ...]:
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
