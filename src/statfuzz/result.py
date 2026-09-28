from __future__ import annotations

import math
from dataclasses import dataclass

from .dgp.base import DGPIdentity
from .nulls import MeanNullCheck


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
    null_check: MeanNullCheck | None = None
    rejection_count: int | None = None
    confidence_level: float | None = None
    interval_method: str | None = None
    interval_low: float | None = None
    interval_high: float | None = None

    def __post_init__(self) -> None:
        if self.null_check is not None and not isinstance(self.null_check, MeanNullCheck):
            raise TypeError("null_check must be a MeanNullCheck or None")

        interval_values = (
            self.rejection_count,
            self.confidence_level,
            self.interval_method,
            self.interval_low,
            self.interval_high,
        )
        if any(value is not None for value in interval_values):
            if any(value is None for value in interval_values):
                raise ValueError(
                    "rejection_count and interval evidence must be provided together"
                )
            if (
                not isinstance(self.rejection_count, int)
                or isinstance(self.rejection_count, bool)
            ):
                raise TypeError("rejection_count must be an integer")
            if not 0 <= self.rejection_count <= self.simulations:
                raise ValueError(
                    "rejection_count must be between 0 and simulations"
                )
            expected_empirical = self.rejection_count / self.simulations
            if not math.isclose(
                self.empirical,
                expected_empirical,
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                raise ValueError(
                    "empirical is inconsistent with rejection_count / simulations"
                )
            if (
                not math.isfinite(self.confidence_level)
                or not 0 < self.confidence_level < 1
            ):
                raise ValueError(
                    "confidence_level must be finite and strictly between 0 and 1"
                )
            if not isinstance(self.interval_method, str) or not self.interval_method:
                raise ValueError("interval_method must be a non-empty string")
            if not math.isfinite(self.interval_low) or not math.isfinite(
                self.interval_high
            ):
                raise ValueError("interval bounds must be finite")
            if not 0 <= self.interval_low <= self.empirical <= self.interval_high <= 1:
                raise ValueError(
                    "interval bounds must satisfy "
                    "0 <= low <= empirical <= high <= 1"
                )

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
        lines = [
            "StatFuzz stress test",
            f"Method:              {self.method}",
            f"Metric:              {self.metric}",
            f"Group 1 DGP:         {self.dgp1}",
            f"Group 2 DGP:         {self.dgp2}",
            f"n1 / n2:             {self.n1} / {self.n2}",
            f"Simulations:         {self.simulations}",
            f"Nominal alpha:       {self.nominal:.4f}",
            f"Empirical alpha:     {self.empirical:.4f}",
            f"Monte Carlo SE:      {self.mcse:.4f}",
        ]
        if self.rejection_count is not None:
            confidence_pct = 100.0 * self.confidence_level
            lines.extend(
                [
                    f"Rejections:          {self.rejection_count}",
                    (
                        f"{confidence_pct:g}% {self.interval_method} CI:  "
                        f"[{self.interval_low:.4f}, {self.interval_high:.4f}]"
                    ),
                ]
            )
        if self.null_check is not None:
            lines.append(
                f"Null verification:   {self.null_check.source} "
                f"(mean={self.null_check.common_mean:g})"
            )
        lines.extend(
            [
                f"Deviation:           {self.deviation:+.4f}",
                f"Tolerance:           ±{self.tolerance:.4f}",
                f"Status:              {self.status}",
            ]
        )
        return "\n".join(lines)
