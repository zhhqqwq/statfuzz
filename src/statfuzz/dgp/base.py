from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Protocol, TypeAlias

import numpy as np

IdentityScalar: TypeAlias = str | int | float | bool | None


def _require_finite(name: str, value: float) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _validate_identity_scalar(name: str, value: object) -> IdentityScalar:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"DGP identity parameter {name!r} must be finite")
        return numeric
    raise TypeError(f"DGP identity parameter {name!r} must be a JSON scalar")


@dataclass(frozen=True, order=True)
class DGPIdentity:
    """Canonical machine identity for a data-generating process.

    Human-readable names may round parameters for display. This identity instead
    preserves the DGP family and full scalar parameter values for matching,
    serialization, and regression baselines.
    """

    family: str
    parameters: tuple[tuple[str, IdentityScalar], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.family, str) or not self.family:
            raise ValueError("DGP identity family must be a non-empty string")

        names = [name for name, _ in self.parameters]
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("DGP identity parameter names must be non-empty strings")
        if len(set(names)) != len(names):
            raise ValueError("DGP identity parameter names must be unique")
        if tuple(sorted(names)) != tuple(names):
            raise ValueError("DGP identity parameters must be sorted by name")

        for name, value in self.parameters:
            _validate_identity_scalar(name, value)

    @classmethod
    def from_mapping(
        cls,
        family: str,
        parameters: dict[str, IdentityScalar],
    ) -> DGPIdentity:
        items = tuple(
            (name, _validate_identity_scalar(name, value))
            for name, value in sorted(parameters.items())
        )
        return cls(family=family, parameters=items)

    def as_dict(self) -> dict[str, object]:
        return {
            "family": self.family,
            "parameters": {name: value for name, value in self.parameters},
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.as_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> DGPIdentity:
        if not isinstance(data, dict):
            raise TypeError("DGP identity must be an object")
        family = data.get("family")
        parameters = data.get("parameters")
        if not isinstance(family, str) or not family:
            raise ValueError("DGP identity family must be a non-empty string")
        if not isinstance(parameters, dict):
            raise TypeError("DGP identity parameters must be an object")

        normalized: dict[str, IdentityScalar] = {}
        for name, value in parameters.items():
            if not isinstance(name, str) or not name:
                raise ValueError("DGP identity parameter names must be non-empty strings")
            normalized[name] = _validate_identity_scalar(name, value)

        return cls.from_mapping(family, normalized)


class DataGenerator(Protocol):
    """Protocol for a data-generating process used by StatFuzz."""

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Draw a sample of size n."""
        ...

    @property
    def name(self) -> str:
        """Human-readable generator name."""
        ...

    @property
    def identity(self) -> DGPIdentity:
        """Stable structured identity used by machine-readable results."""
        ...


def get_dgp_identity(dgp: object) -> DGPIdentity:
    """Return a structured identity, with a conservative custom-DGP fallback.

    Built-in StatFuzz DGPs implement identity directly. For older/custom DGPs
    without that property, the fallback includes the fully qualified class name,
    display name, and repr. Custom DGP authors should implement identity for
    stable cross-run matching.
    """

    identity = getattr(dgp, "identity", None)
    if isinstance(identity, DGPIdentity):
        return identity

    display_name = getattr(dgp, "name", None)
    if not isinstance(display_name, str) or not display_name:
        raise TypeError("DGP must expose a non-empty name string")

    cls = type(dgp)
    family = f"{cls.__module__}.{cls.__qualname__}"
    return DGPIdentity.from_mapping(
        family,
        {
            "display_name": display_name,
            "repr": repr(dgp),
        },
    )
