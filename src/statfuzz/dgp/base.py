from __future__ import annotations

from typing import Protocol

import numpy as np


class DataGenerator(Protocol):
    """Protocol for a data-generating process used by StatFuzz."""

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Draw a sample of size n."""
        ...

    @property
    def name(self) -> str:
        """Human-readable generator name."""
        ...
