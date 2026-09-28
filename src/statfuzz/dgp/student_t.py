from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import DGPIdentity, _require_finite


@dataclass(frozen=True)
class StudentT:
    df: float = 5.0
    mean: float = 0.0
    scale: float = 1.0

    def __post_init__(self) -> None:
        _require_finite("df", self.df)
        _require_finite("mean", self.mean)
        _require_finite("scale", self.scale)
        if self.df <= 1:
            raise ValueError("df must be greater than 1 so the mean exists")
        if self.scale <= 0:
            raise ValueError("scale must be positive")

    @property
    def name(self) -> str:
        return f"StudentT(df={self.df:g}, mean={self.mean:g}, scale={self.scale:g})"

    @property
    def identity(self) -> DGPIdentity:
        return DGPIdentity.from_mapping(
            "statfuzz.dgp.StudentT",
            {"df": self.df, "mean": self.mean, "scale": self.scale},
        )

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        if n < 2:
            raise ValueError("n must be at least 2")
        return self.mean + self.scale * rng.standard_t(df=self.df, size=n)
