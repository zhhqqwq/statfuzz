from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import DGPIdentity, _require_finite


@dataclass(frozen=True)
class LogNormal:
    """Shifted log-normal with an explicitly controlled arithmetic mean.

    The raw log-normal draw has log-location 0 and log-scale sigma.
    We subtract its theoretical mean and add mean so that two groups can
    differ in shape while remaining under a mean-equality null hypothesis.
    """

    sigma: float = 1.0
    mean: float = 0.0

    def __post_init__(self) -> None:
        _require_finite("sigma", self.sigma)
        _require_finite("mean", self.mean)
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")

    @property
    def name(self) -> str:
        return f"ShiftedLogNormal(sigma={self.sigma:g}, mean={self.mean:g})"

    @property
    def identity(self) -> DGPIdentity:
        return DGPIdentity.from_mapping(
            "statfuzz.dgp.LogNormal",
            {"mean": self.mean, "sigma": self.sigma},
        )

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        if n < 2:
            raise ValueError("n must be at least 2")
        raw_mean = np.exp(0.5 * self.sigma**2)
        if not np.isfinite(raw_mean):
            raise ValueError(
                "sigma produces a non-finite theoretical log-normal mean"
            )
        raw = rng.lognormal(mean=0.0, sigma=self.sigma, size=n)
        return raw - raw_mean + self.mean
