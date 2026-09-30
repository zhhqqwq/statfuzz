from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

from .dgp.base import DataGenerator, DGPIdentity, get_dgp_identity
from .methods.bootstrap import (
    BootstrapMeanPercentile,
    bootstrap_mean_percentile_child_rng,
    bootstrap_mean_percentile_interval,
)
from .metrics.binomial_rate import BinomialRateEvidence, binomial_rate_evidence
from .targets import MeanTarget, MeanTargetCheck, resolve_mean_target

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


def _strict_probability(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    if not 0.0 < normalized < 1.0:
        raise ValueError(f"{name} must be strictly between 0 and 1")
    return normalized


def _checked_outer_sample(
    dgp: DataGenerator,
    rng: np.random.Generator,
    n: int,
    *,
    logical_outer_index: int,
) -> np.ndarray:
    try:
        raw = dgp.sample(rng, n)
        sample = np.asarray(raw, dtype=float)
    except Exception as exc:
        raise RuntimeError(
            f"simulation {logical_outer_index}: failed to generate outer sample"
        ) from exc

    if sample.shape != (n,):
        raise RuntimeError(
            f"simulation {logical_outer_index}: outer sample must have shape {(n,)}, "
            f"got {sample.shape}"
        )
    if not np.all(np.isfinite(sample)):
        raise RuntimeError(
            f"simulation {logical_outer_index}: outer sample contains non-finite values"
        )
    return sample


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


def _simulate_bootstrap_coverage_scalar(
    *,
    dgp: DataGenerator,
    n: int,
    simulations: int,
    target_check: MeanTargetCheck,
    method: BootstrapMeanPercentile,
    root_seed: int,
    rng: np.random.Generator,
) -> int:
    """Scalar outer reference executor for mean-interval coverage."""

    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")

    coverage_count = 0
    for logical_outer_index in range(simulations):
        sample = _checked_outer_sample(
            dgp,
            rng,
            n,
            logical_outer_index=logical_outer_index,
        )
        try:
            event = bootstrap_mean_coverage_event(
                sample,
                target_check=target_check,
                method=method,
                root_seed=root_seed,
                logical_outer_index=logical_outer_index,
            )
        except Exception as exc:
            raise RuntimeError(
                f"simulation {logical_outer_index}: "
                "bootstrap coverage evaluation failed"
            ) from exc
        coverage_count += int(event.covered)

    return coverage_count


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


def bootstrap_mean_coverage(
    *,
    dgp: DataGenerator,
    n: int = 20,
    simulations: int = 1_000,
    method: BootstrapMeanPercentile | None = None,
    tolerance: float = 0.01,
    seed: int = 0,
    mean_target: MeanTarget | None = None,
    evidence_confidence_level: float = 0.95,
    evidence_interval_method: str = "wilson",
) -> BootstrapCoverageResult:
    """Estimate percentile-bootstrap mean interval coverage by scalar Monte Carlo.

    This Phase C2 reference executor runs logical outer replicates strictly in
    index order. It has no batching, progress, checkpoint, or resume semantics.
    """

    normalized_n = _positive_integer("n", n)
    normalized_simulations = _positive_integer("simulations", simulations)
    normalized_seed = _non_negative_integer("seed", seed)
    normalized_tolerance = _non_negative_finite_real("tolerance", tolerance)
    normalized_evidence_confidence = _strict_probability(
        "evidence_confidence_level",
        evidence_confidence_level,
    )
    if evidence_interval_method != "wilson":
        raise ValueError("evidence_interval_method must currently be 'wilson'")
    if mean_target is not None and not isinstance(mean_target, MeanTarget):
        raise TypeError("mean_target must be a MeanTarget or None")

    if method is None:
        method_config = BootstrapMeanPercentile()
    elif isinstance(method, BootstrapMeanPercentile):
        method_config = method
    else:
        raise TypeError("method must be a BootstrapMeanPercentile or None")

    target_check = resolve_mean_target(dgp, mean_target)
    dgp_identity = get_dgp_identity(dgp)
    dgp_name = getattr(dgp, "name", None)
    if not isinstance(dgp_name, str) or not dgp_name:
        raise TypeError("DGP must expose a non-empty name string")

    outer_rng = np.random.Generator(np.random.PCG64(normalized_seed))
    coverage_count = _simulate_bootstrap_coverage_scalar(
        dgp=dgp,
        n=normalized_n,
        simulations=normalized_simulations,
        target_check=target_check,
        method=method_config,
        root_seed=normalized_seed,
        rng=outer_rng,
    )

    return BootstrapCoverageResult.from_coverage_count(
        dgp=dgp_name,
        dgp_identity=dgp_identity,
        n=normalized_n,
        simulations=normalized_simulations,
        seed=normalized_seed,
        method_config=method_config,
        target_check=target_check,
        coverage_count=coverage_count,
        tolerance=normalized_tolerance,
        evidence_confidence_level=normalized_evidence_confidence,
        evidence_interval_method=evidence_interval_method,
    )


__all__ = [
    "BootstrapCoverageEvent",
    "BootstrapCoverageResult",
    "bootstrap_mean_coverage",
    "bootstrap_mean_coverage_event",
]
