from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MixtureNormal:
    """Two-component normal mixture with controlled arithmetic mean."""

    weight: float = 0.9
    mean1: float = 0.0
    sd1: float = 1.0
    mean2: float = 0.0
    sd2: float = 5.0
    mean: float = 0.0

    def __post_init__(self) -> None:
        if not 0 < self.weight < 1:
            raise ValueError("weight must be strictly between 0 and 1")
        if self.sd1 <= 0 or self.sd2 <= 0:
            raise ValueError("component standard deviations must be positive")

    @property
    def name(self) -> str:
        return (
            "MixtureNormal("
            f"weight={self.weight:g}, mean1={self.mean1:g}, sd1={self.sd1:g}, "
            f"mean2={self.mean2:g}, sd2={self.sd2:g}, mean={self.mean:g})"
        )

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        if n < 2:
            raise ValueError("n must be at least 2")
        choose_first = rng.random(n) < self.weight
        x = np.empty(n, dtype=float)
        n1 = int(choose_first.sum())
        x[choose_first] = rng.normal(self.mean1, self.sd1, size=n1)
        x[~choose_first] = rng.normal(self.mean2, self.sd2, size=n - n1)

        mixture_mean = self.weight * self.mean1 + (1 - self.weight) * self.mean2
        return x - mixture_mean + self.mean
