from __future__ import annotations

import json
from dataclasses import dataclass

from ..search.result import SearchResult
from ..search.space import JSONScalar
from .model import StressTestSnapshot


def _identity(value: JSONScalar) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sort_key(value: JSONScalar) -> tuple[int, object]:
    if value is None:
        return (4, "")
    if isinstance(value, bool):
        return (0, int(value))
    if isinstance(value, (int, float)):
        return (1, float(value))
    if isinstance(value, str):
        return (2, value)
    return (3, _identity(value))


def _sorted_unique(values: list[JSONScalar]) -> tuple[JSONScalar, ...]:
    by_identity: dict[str, JSONScalar] = {}
    for value in values:
        by_identity.setdefault(_identity(value), value)
    return tuple(sorted(by_identity.values(), key=_sort_key))


@dataclass(frozen=True)
class FailureMapCell:
    x: JSONScalar
    y: JSONScalar
    parameters: dict[str, JSONScalar]
    objective_score: float
    result: StressTestSnapshot

    def as_dict(self) -> dict[str, object]:
        return {
            "x": self.x,
            "y": self.y,
            "parameters": dict(self.parameters),
            "objective_score": self.objective_score,
            "result": self.result.as_dict(),
        }


@dataclass(frozen=True)
class FailureMap2D:
    x_parameter: str
    y_parameter: str
    x_values: tuple[JSONScalar, ...]
    y_values: tuple[JSONScalar, ...]
    fixed_parameters: dict[str, JSONScalar]
    cells: tuple[tuple[FailureMapCell | None, ...], ...]
    objective: str
    uncertainty: str = "mcse"

    @property
    def covered_cells(self) -> int:
        return sum(cell is not None for row in self.cells for cell in row)

    @property
    def total_cells(self) -> int:
        return len(self.x_values) * len(self.y_values)

    @property
    def missing_cells(self) -> int:
        return self.total_cells - self.covered_cells

    @property
    def coverage_fraction(self) -> float:
        if self.total_cells == 0:
            return 0.0
        return self.covered_cells / self.total_cells

    def as_dict(self) -> dict[str, object]:
        return {
            "x_parameter": self.x_parameter,
            "y_parameter": self.y_parameter,
            "x_values": list(self.x_values),
            "y_values": list(self.y_values),
            "fixed_parameters": dict(self.fixed_parameters),
            "objective": self.objective,
            "uncertainty": self.uncertainty,
            "covered_cells": self.covered_cells,
            "missing_cells": self.missing_cells,
            "coverage_fraction": self.coverage_fraction,
            "cells": [
                [None if cell is None else cell.as_dict() for cell in row]
                for row in self.cells
            ],
        }


def failure_map_2d(
    *,
    search: SearchResult,
    x_parameter: str,
    y_parameter: str,
    fixed: dict[str, JSONScalar] | None = None,
) -> FailureMap2D:
    """Build a 2D failure-map snapshot from existing search records.

    No simulation is rerun. For spaces with more than two parameters, every
    non-axis parameter must be fixed explicitly to prevent accidental projection
    of multiple records onto the same map coordinate.
    """

    if x_parameter == y_parameter:
        raise ValueError("x_parameter and y_parameter must differ")
    if x_parameter not in search.parameter_names:
        raise ValueError(f"unknown x_parameter: {x_parameter!r}")
    if y_parameter not in search.parameter_names:
        raise ValueError(f"unknown y_parameter: {y_parameter!r}")

    fixed_parameters = {} if fixed is None else dict(fixed)
    remaining = tuple(
        name
        for name in search.parameter_names
        if name not in {x_parameter, y_parameter}
    )

    missing_fixed = [name for name in remaining if name not in fixed_parameters]
    extra_fixed = [
        name
        for name in fixed_parameters
        if name not in remaining
    ]
    if missing_fixed:
        raise ValueError(
            "all non-axis parameters must be fixed explicitly; missing: "
            + ", ".join(missing_fixed)
        )
    if extra_fixed:
        raise ValueError(
            "fixed contains parameters that are not non-axis dimensions: "
            + ", ".join(extra_fixed)
        )

    selected = []
    for record in search.records:
        parameters = record.point.as_dict()
        if all(
            _identity(parameters[name]) == _identity(value)
            for name, value in fixed_parameters.items()
        ):
            selected.append(record)

    if not selected:
        raise ValueError("no search records match the requested failure-map slice")

    x_values = _sorted_unique(
        [record.point[x_parameter] for record in selected]
    )
    y_values = _sorted_unique(
        [record.point[y_parameter] for record in selected]
    )
    x_index = {_identity(value): index for index, value in enumerate(x_values)}
    y_index = {_identity(value): index for index, value in enumerate(y_values)}

    matrix: list[list[FailureMapCell | None]] = [
        [None for _ in x_values]
        for _ in y_values
    ]

    for record in selected:
        x = record.point[x_parameter]
        y = record.point[y_parameter]
        xi = x_index[_identity(x)]
        yi = y_index[_identity(y)]

        if matrix[yi][xi] is not None:
            raise ValueError(
                "multiple search records map to the same 2D coordinate; "
                "fix all remaining parameters more specifically"
            )

        matrix[yi][xi] = FailureMapCell(
            x=x,
            y=y,
            parameters=record.point.as_dict(),
            objective_score=record.objective_score(search.objective),
            result=StressTestSnapshot.from_result(record.result),
        )

    return FailureMap2D(
        x_parameter=x_parameter,
        y_parameter=y_parameter,
        x_values=x_values,
        y_values=y_values,
        fixed_parameters=fixed_parameters,
        cells=tuple(tuple(row) for row in matrix),
        objective=search.objective_name,
    )
