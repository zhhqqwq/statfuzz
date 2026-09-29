from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal, Protocol, runtime_checkable

from ..result import StatisticalPropertyResult

ObjectiveAlias = Literal[
    "absolute_deviation",
    "positive_deviation",
    "negative_deviation",
]


@runtime_checkable
class SearchObjective(Protocol):
    """Ranking rule for exploratory search results."""

    name: str

    def score(self, result: StatisticalPropertyResult) -> float:
        """Return a score where larger values rank as more extreme."""
        ...


@dataclass(frozen=True)
class AbsoluteDeviationObjective:
    """Rank by absolute distance from the nominal target."""

    name: ClassVar[str] = "absolute_deviation"

    def score(self, result: StatisticalPropertyResult) -> float:
        return abs(result.deviation)


@dataclass(frozen=True)
class PositiveDeviationObjective:
    """Rank large positive empirical-minus-nominal deviations first."""

    name: ClassVar[str] = "positive_deviation"

    def score(self, result: StatisticalPropertyResult) -> float:
        return result.deviation


@dataclass(frozen=True)
class NegativeDeviationObjective:
    """Rank large negative empirical-minus-nominal deviations first."""

    name: ClassVar[str] = "negative_deviation"

    def score(self, result: StatisticalPropertyResult) -> float:
        return -result.deviation


ObjectiveLike = SearchObjective | ObjectiveAlias

_BUILT_INS: dict[str, SearchObjective] = {
    AbsoluteDeviationObjective.name: AbsoluteDeviationObjective(),
    PositiveDeviationObjective.name: PositiveDeviationObjective(),
    NegativeDeviationObjective.name: NegativeDeviationObjective(),
}


def resolve_objective(objective: ObjectiveLike) -> SearchObjective:
    """Resolve a built-in string alias or validate a custom objective."""

    if isinstance(objective, str):
        try:
            return _BUILT_INS[objective]
        except KeyError as exc:
            supported = ", ".join(sorted(_BUILT_INS))
            raise ValueError(
                f"unsupported objective {objective!r}; supported aliases: {supported}"
            ) from exc

    if not isinstance(objective, SearchObjective):
        raise TypeError("objective must be a SearchObjective or a supported string alias")
    if not objective.name:
        raise ValueError("objective.name must be a non-empty string")
    return objective
