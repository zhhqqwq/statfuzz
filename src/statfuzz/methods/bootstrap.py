from __future__ import annotations

import json
import math
from dataclasses import dataclass
from numbers import Integral, Real

BOOTSTRAP_MEAN_PERCENTILE_METHOD = "bootstrap_mean_percentile"
BOOTSTRAP_MEAN_PERCENTILE_SEMANTICS_VERSION = "1"
BOOTSTRAP_MEAN_PERCENTILE_QUANTILE_METHOD = "linear"


@dataclass(frozen=True)
class BootstrapMeanPercentile:
    """Immutable configuration for percentile bootstrap mean intervals."""

    resamples: int = 999
    interval_level: float = 0.95
    quantile_method: str = BOOTSTRAP_MEAN_PERCENTILE_QUANTILE_METHOD

    def __post_init__(self) -> None:
        if isinstance(self.resamples, bool) or not isinstance(
            self.resamples,
            Integral,
        ):
            raise TypeError("resamples must be an integer")
        resamples = int(self.resamples)
        if resamples <= 0:
            raise ValueError("resamples must be positive")
        object.__setattr__(self, "resamples", resamples)

        if isinstance(self.interval_level, bool) or not isinstance(
            self.interval_level,
            Real,
        ):
            raise TypeError("interval_level must be a real number")
        interval_level = float(self.interval_level)
        if not math.isfinite(interval_level):
            raise ValueError("interval_level must be finite")
        if not 0.0 < interval_level < 1.0:
            raise ValueError("interval_level must be strictly between 0 and 1")
        object.__setattr__(self, "interval_level", interval_level)

        if self.quantile_method != BOOTSTRAP_MEAN_PERCENTILE_QUANTILE_METHOD:
            raise ValueError(
                "quantile_method must currently be 'linear'"
            )

    @property
    def method(self) -> str:
        return BOOTSTRAP_MEAN_PERCENTILE_METHOD

    @property
    def semantics_version(self) -> str:
        return BOOTSTRAP_MEAN_PERCENTILE_SEMANTICS_VERSION

    def as_dict(self) -> dict[str, object]:
        """Return the stable machine identity for this method configuration."""

        return {
            "method": self.method,
            "semantics_version": self.semantics_version,
            "resamples": self.resamples,
            "interval_level": self.interval_level,
            "quantile_method": self.quantile_method,
        }

    def canonical_json(self) -> str:
        """Return canonical JSON suitable for fingerprints and exact matching."""

        return json.dumps(
            self.as_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> BootstrapMeanPercentile:
        if not isinstance(data, dict):
            raise TypeError("bootstrap method identity must be an object")

        expected = {
            "method",
            "semantics_version",
            "resamples",
            "interval_level",
            "quantile_method",
        }
        actual = set(data)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            details = []
            if missing:
                details.append(f"missing={missing!r}")
            if extra:
                details.append(f"extra={extra!r}")
            raise ValueError(
                "bootstrap method identity has unexpected keys "
                f"({', '.join(details)})"
            )

        if data["method"] != BOOTSTRAP_MEAN_PERCENTILE_METHOD:
            raise ValueError("unsupported bootstrap method identity")
        if (
            data["semantics_version"]
            != BOOTSTRAP_MEAN_PERCENTILE_SEMANTICS_VERSION
        ):
            raise ValueError("unsupported bootstrap method semantics version")

        return cls(
            resamples=data["resamples"],
            interval_level=data["interval_level"],
            quantile_method=data["quantile_method"],
        )


__all__ = ["BootstrapMeanPercentile"]
