from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.stats import binomtest


@dataclass(frozen=True)
class BinomialRateEvidence:
    """Monte Carlo evidence for a repeated Bernoulli event rate."""

    event_count: int
    trials: int
    empirical: float
    mcse: float
    confidence_level: float
    interval_method: str
    interval_low: float
    interval_high: float


def binomial_rate_evidence(
    event_count: int,
    trials: int,
    *,
    confidence_level: float = 0.95,
    interval_method: str = "wilson",
) -> BinomialRateEvidence:
    if not isinstance(event_count, int) or isinstance(event_count, bool):
        raise TypeError("event_count must be an integer")
    if not isinstance(trials, int) or isinstance(trials, bool):
        raise TypeError("trials must be an integer")
    if trials <= 0:
        raise ValueError("trials must be positive")
    if not 0 <= event_count <= trials:
        raise ValueError("event_count must be between 0 and trials")
    if not math.isfinite(confidence_level) or not 0 < confidence_level < 1:
        raise ValueError(
            "confidence_level must be finite and strictly between 0 and 1"
        )
    if interval_method != "wilson":
        raise ValueError("interval_method must currently be 'wilson'")

    empirical = event_count / trials
    mcse = math.sqrt(empirical * (1.0 - empirical) / trials)
    ci = binomtest(event_count, trials).proportion_ci(
        confidence_level=confidence_level,
        method=interval_method,
    )

    return BinomialRateEvidence(
        event_count=event_count,
        trials=trials,
        empirical=empirical,
        mcse=mcse,
        confidence_level=confidence_level,
        interval_method=interval_method,
        interval_low=float(ci.low),
        interval_high=float(ci.high),
    )
