from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

from .dgp.base import DGPIdentity
from .methods.bootstrap import (
    BootstrapMeanPercentile,
    bootstrap_mean_percentile_child_rng,
    bootstrap_mean_percentile_interval,
)
from .metrics.binomial_rate import BinomialRateEvidence, binomial_rate_evidence
from .targets import MeanTargetCheck

BOOTSTRAP_COVERAGE_METRIC = "coverage"


def _positive_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{name} must be positive")
    return normalized


def _non_negative_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{name} must be non-negative")
    return normalized


def _non_negative_finite_real(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    if normalized < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return normalized


@dataclass(frozen=True)
class BootstrapCoverageEvent:
    """One logical outer-replicate coverage decision."""

    logical_outer_index: int
    target_mean: float
    interval_low: float
    interval_high: float
    covered: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "logical_outer_index",
            _non_negative_integer(
                "logical_outer_index",
                self.logical_outer_index,
            ),
        )

        for name in ("target_mean", "interval_low", "interval_high"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real number")
            normalized = float(value)
            if not math.isfinite(normalized):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, normalized)

        if self.interval_low > self.interval_high:
            raise ValueError("coverage interval bounds must satisfy low <= high")
        if not isinstance(self.covered, bool):
            raise TypeError("covered must be a bool")

        expected = self.interval_low <= self.target_mean <= self.interval_high
        if self.covered is not expected:
            raise ValueError(
                "covered is inconsistent with inclusive interval membership"
            )


def bootstrap_mean_coverage_event(
    sample: np.ndarray,
    *,
    target_check: MeanTargetCheck,
    method: BootstrapMeanPercentile,
    root_seed: int,
    logical_outer_index: int,
) -> BootstrapCoverageEvent:
    """Evaluate one inclusive mean-coverage event for a logical outer replicate."""

    if not isinstance(target_check, MeanTargetCheck):
        raise TypeError("target_check must be a MeanTargetCheck")
    if not isinstance(method, BootstrapMeanPercentile):
        raise TypeError("method must be a BootstrapMeanPercentile")

    rng = bootstrap_mean_percentile_child_rng(
        root_seed,
        logical_outer_index,
    )
    low, high = bootstrap_mean_percentile_interval(
        sample,
        rng,
        method,
    )
    target_mean = target_check.mean
    return BootstrapCoverageEvent(
        logical_outer_index=logical_outer_index,
        target_mean=target_mean,
        interval_low=low,
        interval_high=high,
        covered=low <= target_mean <= high,
    )


@dataclass(frozen=True)
class BootstrapCoverageResult:
    """Aggregate coverage result contract for percentile bootstrap mean intervals."""

    dgp: str
    dgp_identity: DGPIdentity
    n: int
    simulations: int
    seed: int
    method_config: BootstrapMeanPercentile
    target_check: MeanTargetCheck
    evidence: BinomialRateEvidence
    tolerance: float

    def __post_init__(self) -> None:
        if not isinstance(self.dgp, str) or not self.dgp:
            raise ValueError("dgp must be a non-empty string")
        if not isinstance(self.dgp_identity, DGPIdentity):
            raise TypeError("dgp_identity must be a DGPIdentity")

        object.__setattr__(self, "n", _positive_integer("n", self.n))
        object.__setattr__(
            self,
            "simulations",
            _positive_integer("simulations", self.simulations),
        )
        object.__setattr__(
            self,
            "seed",
            _non_negative_integer("seed", self.seed),
        )
        object.__setattr__(
            self,
            "tolerance",
            _non_negative_finite_real("tolerance", self.tolerance),
        )

        if not isinstance(self.method_config, BootstrapMeanPercentile):
            raise TypeError(
                "method_config must be a BootstrapMeanPercentile"
            )
        if not isinstance(self.target_check, MeanTargetCheck):
            raise TypeError("target_check must be a MeanTargetCheck")
        if not isinstance(self.evidence, BinomialRateEvidence):
            raise TypeError("evidence must be a BinomialRateEvidence")
        if self.evidence.trials != self.simulations:
            raise ValueError(
                "evidence.trials must equal simulations"
            )
        expected_evidence = binomial_rate_evidence(
            self.evidence.event_count,
            self.evidence.trials,
            confidence_level=self.evidence.confidence_level,
            interval_method=self.evidence.interval_method,
        )
        if self.evidence != expected_evidence:
            raise ValueError(
                "evidence is inconsistent with its event count and trials"
            )

    @classmethod
    def from_coverage_count(
        cls,
        *,
        dgp: str,
        dgp_identity: DGPIdentity,
        n: int,
        simulations: int,
        seed: int,
        method_config: BootstrapMeanPercentile,
        target_check: MeanTargetCheck,
        coverage_count: int,
        tolerance: float,
        evidence_confidence_level: float = 0.95,
        evidence_interval_method: str = "wilson",
    ) -> BootstrapCoverageResult:
        evidence = binomial_rate_evidence(
            coverage_count,
            simulations,
            confidence_level=evidence_confidence_level,
            interval_method=evidence_interval_method,
        )
        return cls(
            dgp=dgp,
            dgp_identity=dgp_identity,
            n=n,
            simulations=simulations,
            seed=seed,
            method_config=method_config,
            target_check=target_check,
            evidence=evidence,
            tolerance=tolerance,
        )

    @property
    def method(self) -> str:
        return self.method_config.method

    @property
    def metric(self) -> str:
        return BOOTSTRAP_COVERAGE_METRIC

    @property
    def nominal(self) -> float:
        return self.method_config.interval_level

    @property
    def empirical(self) -> float:
        return self.evidence.empirical

    @property
    def mcse(self) -> float:
        return self.evidence.mcse

    @property
    def coverage_count(self) -> int:
        return self.evidence.event_count

    @property
    def bootstrap_resamples(self) -> int:
        return self.method_config.resamples

    @property
    def bootstrap_interval_level(self) -> float:
        return self.method_config.interval_level

    @property
    def bootstrap_quantile_method(self) -> str:
        return self.method_config.quantile_method

    @property
    def evidence_confidence_level(self) -> float:
        return self.evidence.confidence_level

    @property
    def evidence_interval_method(self) -> str:
        return self.evidence.interval_method

    @property
    def evidence_interval_low(self) -> float:
        return self.evidence.interval_low

    @property
    def evidence_interval_high(self) -> float:
        return self.evidence.interval_high

    @property
    def deviation(self) -> float:
        return self.empirical - self.nominal

    @property
    def passed(self) -> bool:
        return abs(self.deviation) <= self.tolerance

    @property
    def status(self) -> str:
        return "PASS" if self.passed else "OUTSIDE_TOLERANCE"


__all__ = [
    "BootstrapCoverageEvent",
    "BootstrapCoverageResult",
    "bootstrap_mean_coverage_event",
]
