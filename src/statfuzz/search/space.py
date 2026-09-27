from __future__ import annotations

import itertools
import json
import math
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import TypeAlias

JSONScalar: TypeAlias = str | int | float | bool | None


def _validate_scalar(value: object) -> JSONScalar:
    if not isinstance(value, (str, int, float, bool)) and value is not None:
        raise TypeError(
            "parameter values must be JSON scalar values: str, int, float, bool, or None"
        )
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("floating-point parameter values must be finite")
    return value


@dataclass(frozen=True)
class ParameterPoint:
    """One immutable, canonically ordered point in a parameter space."""

    items: tuple[tuple[str, JSONScalar], ...]

    def as_dict(self) -> dict[str, JSONScalar]:
        return dict(self.items)

    def to_json(self) -> str:
        return json.dumps(
            self.as_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )

    def __getitem__(self, name: str) -> JSONScalar:
        for key, value in self.items:
            if key == name:
                return value
        raise KeyError(name)


class ParameterSpace:
    """Finite, deterministic Cartesian product of named parameter candidates."""

    def __init__(self, parameters: Mapping[str, Iterable[JSONScalar]]) -> None:
        if not parameters:
            raise ValueError("ParameterSpace requires at least one parameter")

        axes: list[tuple[str, tuple[JSONScalar, ...]]] = []
        for name in sorted(parameters):
            if not isinstance(name, str) or not name:
                raise ValueError("parameter names must be non-empty strings")

            values = tuple(_validate_scalar(value) for value in parameters[name])
            if not values:
                raise ValueError(f"parameter {name!r} must have at least one candidate value")

            encoded = [
                json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)
                for value in values
            ]
            if len(set(encoded)) != len(encoded):
                raise ValueError(f"parameter {name!r} contains duplicate candidate values")

            axes.append((name, values))

        self._axes = tuple(axes)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self._axes)

    @property
    def parameters(self) -> dict[str, tuple[JSONScalar, ...]]:
        return {name: values for name, values in self._axes}

    def __len__(self) -> int:
        return math.prod(len(values) for _, values in self._axes)

    def __iter__(self) -> Iterator[ParameterPoint]:
        names = self.names
        axes = [values for _, values in self._axes]
        for combination in itertools.product(*axes):
            yield ParameterPoint(tuple(zip(names, combination)))

    def points(self) -> tuple[ParameterPoint, ...]:
        """Materialize all points. Prefer iteration for very large grids."""
        return tuple(self)
