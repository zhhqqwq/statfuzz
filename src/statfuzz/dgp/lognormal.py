from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LogNormal:
    """Shifted log-normal with an explicitly controlled arithmetic mean.

    The raw log-normal draw has log-location 0 and log-scale ``sigma``.
    We subtract its theoretical mean and add ``mean`` so that two groups can
    differ in shape while remaining under a mean-equality null hypothesis.
    """

    sigma: float = 1.0
    mean: float = 0.0

    def __post_init__(self) -> None:
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")

    @property
    def name(self) -> str:
        return f"ShiftedLogNormal(sigma={self.sigma:g}, mean={self.mean:g})"

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        if n < 2:
            raise ValueError("n must be at least 2")
        raw = rng.lognormal(mean=0.0, sigma=self.sigma, size=n)
        raw_mean = np.exp(0.5 * self.sigma**2)
        return raw - raw_mean + self.mean
