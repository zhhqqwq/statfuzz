from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Normal:
    mean: float = 0.0
    sd: float = 1.0

    def __post_init__(self) -> None:
        if self.sd <= 0:
            raise ValueError("sd must be positive")

    @property
    def name(self) -> str:
        return f"Normal(mean={self.mean:g}, sd={self.sd:g})"

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        if n < 2:
            raise ValueError("n must be at least 2")
        return rng.normal(loc=self.mean, scale=self.sd, size=n)
