from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from .nulls import population_mean_of

MeanTargetSource = Literal["population_mean", "declaration"]


def _finite_mean_target(name: str, value: object) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


@dataclass(frozen=True)
class MeanTarget:
    """Explicit caller declaration of the population mean used as truth."""

    mean: float
    note: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "mean",
            _finite_mean_target("mean target", self.mean),
        )
        if self.note is not None and (
            not isinstance(self.note, str) or not self.note.strip()
        ):
            raise ValueError(
                "mean target note must be a non-empty string or None"
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": "mean",
            "mean": self.mean,
            "note": self.note,
        }


@dataclass(frozen=True)
class MeanTargetCheck:
    """Recorded evidence for the population-mean truth used by an experiment."""

    source: MeanTargetSource
    mean: float
    population_mean: float | None = None
    note: str | None = None

    def __post_init__(self) -> None:
        if self.source not in {"population_mean", "declaration"}:
            raise ValueError(
                "mean target source must be 'population_mean' or 'declaration'"
            )
        object.__setattr__(
            self,
            "mean",
            _finite_mean_target("mean target check mean", self.mean),
        )
        if self.population_mean is not None:
            object.__setattr__(
                self,
                "population_mean",
                _finite_mean_target(
                    "mean target check population_mean",
                    self.population_mean,
                ),
            )
        if self.source == "population_mean" and self.population_mean is None:
            raise ValueError(
                "population_mean source requires a verified population_mean"
            )
        if (
            self.population_mean is not None
            and self.population_mean != self.mean
        ):
            raise ValueError(
                "mean target check population_mean must equal the target mean"
            )
        if self.note is not None and (
            not isinstance(self.note, str) or not self.note.strip()
        ):
            raise ValueError(
                "mean target check note must be a non-empty string or None"
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": "mean",
            "source": self.source,
            "mean": self.mean,
            "population_mean": self.population_mean,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> MeanTargetCheck:
        if not isinstance(data, dict):
            raise TypeError("mean target check must be an object")
        if data.get("kind") != "mean":
            raise ValueError("unsupported mean target check kind")
        source = data.get("source")
        if source not in {"population_mean", "declaration"}:
            raise ValueError("unsupported mean target check source")
        return cls(
            source=source,
            mean=data.get("mean"),
            population_mean=data.get("population_mean"),
            note=data.get("note"),
        )


def resolve_mean_target(
    dgp: object,
    declaration: MeanTarget | None,
) -> MeanTargetCheck:
    """Resolve a known population mean before a mean-coverage experiment."""

    population_mean = population_mean_of(dgp)

    if population_mean is not None:
        if declaration is not None and declaration.mean != population_mean:
            raise ValueError(
                "mean target does not match the DGP population_mean"
            )
        return MeanTargetCheck(
            source="population_mean",
            mean=population_mean,
            population_mean=population_mean,
            note=None if declaration is None else declaration.note,
        )

    if declaration is None:
        raise ValueError(
            "mean coverage requires a known population mean; "
            "the DGP does not expose population_mean. "
            "Provide MeanTarget(mean=..., note=...) explicitly."
        )

    return MeanTargetCheck(
        source="declaration",
        mean=declaration.mean,
        population_mean=None,
        note=declaration.note,
    )


__all__ = ["MeanTarget", "MeanTargetCheck"]
