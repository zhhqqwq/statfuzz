from __future__ import annotations

from dataclasses import dataclass

from .dgp.base import DGPIdentity


@dataclass(frozen=True)
class StressTestResult:
    method: str
    metric: str
    dgp1: str
    dgp2: str
    n1: int
    n2: int
    simulations: int
    seed: int | None
    nominal: float
    empirical: float
    mcse: float
    tolerance: float
    dgp1_identity: DGPIdentity | None = None
    dgp2_identity: DGPIdentity | None = None

    @property
    def deviation(self) -> float:
        return self.empirical - self.nominal

    @property
    def passed(self) -> bool:
        return abs(self.deviation) <= self.tolerance

    @property
    def status(self) -> str:
        return "PASS" if self.passed else "OUTSIDE_TOLERANCE"

    def __str__(self) -> str:
        return (
            "StatFuzz stress test\n"
            f"Method:              {self.method}\n"
            f"Metric:              {self.metric}\n"
            f"Group 1 DGP:         {self.dgp1}\n"
            f"Group 2 DGP:         {self.dgp2}\n"
            f"n1 / n2:             {self.n1} / {self.n2}\n"
            f"Simulations:         {self.simulations}\n"
            f"Nominal alpha:       {self.nominal:.4f}\n"
            f"Empirical alpha:     {self.empirical:.4f}\n"
            f"Monte Carlo SE:      {self.mcse:.4f}\n"
            f"Deviation:           {self.deviation:+.4f}\n"
            f"Tolerance:           ±{self.tolerance:.4f}\n"
            f"Status:              {self.status}"
        )
