from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

NullCheckSource = Literal["population_means", "declaration"]


def _finite_mean(name: str, value: object) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


@dataclass(frozen=True)
class MeanEqualityNull:
    """Explicit caller declaration for an equal-population-means null hypothesis."""

    mean: float
    note: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "mean", _finite_mean("null mean", self.mean))
        if self.note is not None and (
            not isinstance(self.note, str) or not self.note.strip()
        ):
            raise ValueError("null declaration note must be a non-empty string or None")

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": "equal_means",
            "mean": self.mean,
            "note": self.note,
        }


@dataclass(frozen=True)
class MeanNullCheck:
    """Recorded evidence that an equal-means Type-I null was established."""

    source: NullCheckSource
    common_mean: float
    group1_population_mean: float | None = None
    group2_population_mean: float | None = None
    note: str | None = None

    def __post_init__(self) -> None:
        if self.source not in {"population_means", "declaration"}:
            raise ValueError(
                "null check source must be 'population_means' or 'declaration'"
            )
        object.__setattr__(
            self,
            "common_mean",
            _finite_mean("null common_mean", self.common_mean),
        )
        for field_name in ("group1_population_mean", "group2_population_mean"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self,
                    field_name,
                    _finite_mean(field_name, value),
                )
        if self.note is not None and (
            not isinstance(self.note, str) or not self.note.strip()
        ):
            raise ValueError("null check note must be a non-empty string or None")

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": "equal_means",
            "source": self.source,
            "common_mean": self.common_mean,
            "group1_population_mean": self.group1_population_mean,
            "group2_population_mean": self.group2_population_mean,
            "note": self.note,
        }


def population_mean_of(dgp: object) -> float | None:
    """Return a finite declared population mean if the DGP exposes one."""

    if not hasattr(dgp, "population_mean"):
        return None
    value = getattr(dgp, "population_mean")
    if value is None:
        return None
    return _finite_mean("DGP population_mean", value)


def verify_mean_equality_null(
    dgp1: object,
    dgp2: object,
    declaration: MeanEqualityNull | None,
) -> MeanNullCheck:
    """Verify or explicitly record an equal-means null before simulation."""

    mean1 = population_mean_of(dgp1)
    mean2 = population_mean_of(dgp2)

    if mean1 is not None and mean2 is not None:
        if mean1 != mean2:
            raise ValueError(
                "type1_error requires equal population means; "
                f"group 1 mean={mean1!r}, group 2 mean={mean2!r}"
            )
        if declaration is not None and declaration.mean != mean1:
            raise ValueError(
                "null declaration mean does not match verified DGP population means"
            )
        return MeanNullCheck(
            source="population_means",
            common_mean=mean1,
            group1_population_mean=mean1,
            group2_population_mean=mean2,
            note=None if declaration is None else declaration.note,
        )

    if declaration is None:
        missing = []
        if mean1 is None:
            missing.append("group 1")
        if mean2 is None:
            missing.append("group 2")
        missing_text = " and ".join(missing)
        raise ValueError(
            "type1_error requires a verifiable equal-means null; "
            f"{missing_text} DGP does not expose population_mean. "
            "Provide MeanEqualityNull(mean=..., note=...) explicitly."
        )

    if mean1 is not None and mean1 != declaration.mean:
        raise ValueError(
            "group 1 population_mean does not match the declared null mean"
        )
    if mean2 is not None and mean2 != declaration.mean:
        raise ValueError(
            "group 2 population_mean does not match the declared null mean"
        )

    return MeanNullCheck(
        source="declaration",
        common_mean=declaration.mean,
        group1_population_mean=mean1,
        group2_population_mean=mean2,
        note=declaration.note,
    )
