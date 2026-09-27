from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DiscoveryBudget:
    """Monte Carlo budgets for search and independent validation."""

    search_simulations: int
    validation_simulations: int

    def __post_init__(self) -> None:
        if self.search_simulations <= 0:
            raise ValueError("search_simulations must be positive")
        if self.validation_simulations <= 0:
            raise ValueError("validation_simulations must be positive")

    def as_dict(self) -> dict[str, int]:
        return {
            "search_simulations": self.search_simulations,
            "validation_simulations": self.validation_simulations,
        }
