from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

BOOTSTRAP_MEAN_PERCENTILE_METHOD = "bootstrap_mean_percentile"
BOOTSTRAP_MEAN_PERCENTILE_SEMANTICS_VERSION = "1"
BOOTSTRAP_MEAN_PERCENTILE_QUANTILE_METHOD = "linear"
BOOTSTRAP_MEAN_PERCENTILE_RNG_DOMAIN = (
    "statfuzz.bootstrap_mean_percentile.child_rng"
)
BOOTSTRAP_MEAN_PERCENTILE_RNG_SEMANTICS_VERSION = "1"
_BOOTSTRAP_MEAN_PERCENTILE_RNG_PERSON = b"statfuzz-bsprng1"


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


def _non_negative_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{name} must be non-negative")
    return normalized


def bootstrap_mean_percentile_child_seed(
    root_seed: int,
    logical_outer_index: int,
) -> int:
    """Derive the stable bootstrap child seed for one logical outer replicate."""

    normalized_root_seed = _non_negative_integer("root_seed", root_seed)
    normalized_index = _non_negative_integer(
        "logical_outer_index",
        logical_outer_index,
    )

    payload = {
        "domain": BOOTSTRAP_MEAN_PERCENTILE_RNG_DOMAIN,
        "logical_outer_index": normalized_index,
        "rng_semantics_version": (
            BOOTSTRAP_MEAN_PERCENTILE_RNG_SEMANTICS_VERSION
        ),
        "root_seed": normalized_root_seed,
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.blake2b(
        canonical,
        digest_size=16,
        person=_BOOTSTRAP_MEAN_PERCENTILE_RNG_PERSON,
    ).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def bootstrap_mean_percentile_child_rng(
    root_seed: int,
    logical_outer_index: int,
) -> np.random.Generator:
    """Create the explicit PCG64 child Generator for one outer replicate."""

    child_seed = bootstrap_mean_percentile_child_seed(
        root_seed,
        logical_outer_index,
    )
    return np.random.Generator(np.random.PCG64(child_seed))


def _as_finite_bootstrap_sample(sample: np.ndarray) -> np.ndarray:
    values = np.asarray(sample, dtype=float)
    if values.ndim != 1:
        raise ValueError("sample must be one-dimensional")
    if values.size == 0:
        raise ValueError("sample must contain at least one observation")
    if not np.all(np.isfinite(values)):
        raise ValueError("sample must contain only finite observations")
    return values


def bootstrap_mean_percentile_interval(
    sample: np.ndarray,
    rng: np.random.Generator,
    method: BootstrapMeanPercentile,
) -> tuple[float, float]:
    """Scalar reference oracle for a percentile bootstrap mean interval.

    For each bootstrap replicate, exactly n replacement indices are drawn with
    one rng.integers(0, n, size=n) call, then the resampled mean is computed.
    Quantiles are evaluated only after all replicate means exist.
    """

    values = _as_finite_bootstrap_sample(sample)
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    if not isinstance(method, BootstrapMeanPercentile):
        raise TypeError("method must be a BootstrapMeanPercentile")

    n = values.size
    bootstrap_means = np.empty(method.resamples, dtype=float)
    for replicate in range(method.resamples):
        indices = rng.integers(0, n, size=n)
        with np.errstate(over="ignore", invalid="ignore"):
            mean = float(np.mean(values[indices]))
        if not math.isfinite(mean):
            raise ValueError(
                f"bootstrap replicate {replicate} mean is non-finite"
            )
        bootstrap_means[replicate] = mean

    tail_probability = (1.0 - method.interval_level) / 2.0
    interval = np.asarray(
        np.quantile(
            bootstrap_means,
            [tail_probability, 1.0 - tail_probability],
            method=method.quantile_method,
        ),
        dtype=float,
    )
    if interval.shape != (2,):
        raise ValueError("bootstrap quantile output has an unexpected shape")
    if not np.all(np.isfinite(interval)):
        raise ValueError("bootstrap percentile interval is non-finite")

    low = float(interval[0])
    high = float(interval[1])
    if low > high:
        raise ValueError("bootstrap percentile interval has invalid bounds")
    return low, high


__all__ = [
    "BootstrapMeanPercentile",
    "bootstrap_mean_percentile_child_rng",
    "bootstrap_mean_percentile_child_seed",
    "bootstrap_mean_percentile_interval",
]
