from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.stats import binomtest


@dataclass(frozen=True)
class Type1ErrorEvidence:
    rejection_count: int
    simulations: int
    empirical: float
    mcse: float
    confidence_level: float
    interval_method: str
    interval_low: float
    interval_high: float


def type1_error_evidence(
    rejections: int,
    simulations: int,
    nominal: float,
    *,
    confidence_level: float = 0.95,
    interval_method: str = "wilson",
) -> Type1ErrorEvidence:
    if not isinstance(rejections, int) or isinstance(rejections, bool):
        raise TypeError("rejections must be an integer")
    if simulations <= 0:
        raise ValueError("simulations must be positive")
    if not 0 <= rejections <= simulations:
        raise ValueError("rejections must be between 0 and simulations")
    if not math.isfinite(nominal) or not 0 < nominal < 1:
        raise ValueError("nominal must be finite and strictly between 0 and 1")
    if not math.isfinite(confidence_level) or not 0 < confidence_level < 1:
        raise ValueError(
            "confidence_level must be finite and strictly between 0 and 1"
        )
    if interval_method != "wilson":
        raise ValueError("interval_method must currently be 'wilson'")

    empirical = rejections / simulations
    mcse = math.sqrt(empirical * (1.0 - empirical) / simulations)
    ci = binomtest(rejections, simulations).proportion_ci(
        confidence_level=confidence_level,
        method=interval_method,
    )

    return Type1ErrorEvidence(
        rejection_count=rejections,
        simulations=simulations,
        empirical=empirical,
        mcse=mcse,
        confidence_level=confidence_level,
        interval_method=interval_method,
        interval_low=float(ci.low),
        interval_high=float(ci.high),
    )


def type1_error_summary(
    rejections: int,
    simulations: int,
    nominal: float,
) -> tuple[float, float]:
    """Backward-compatible empirical rate + MCSE summary."""

    evidence = type1_error_evidence(rejections, simulations, nominal)
    return evidence.empirical, evidence.mcse
