from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

from .bootstrap_checkpoint import (
    BootstrapCoverageCheckpoint,
    BootstrapCoverageExperimentSpec,
)
from .checkpoint import ExecutionContract
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
class BootstrapCoverageProgress:
    """Cumulative snapshot emitted after a committed outer coverage batch."""

    completed: int
    total: int
    covered: int
    empirical: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "completed",
            _positive_integer("completed", self.completed),
        )
        object.__setattr__(
            self,
            "total",
            _positive_integer("total", self.total),
        )
        object.__setattr__(
            self,
            "covered",
            _non_negative_integer("covered", self.covered),
        )
        if self.completed > self.total:
            raise ValueError("completed must not exceed total")
        if self.covered > self.completed:
            raise ValueError("covered must not exceed completed")
        if isinstance(self.empirical, bool) or not isinstance(self.empirical, Real):
            raise TypeError("empirical must be a real number")
        empirical = float(self.empirical)
        if not math.isfinite(empirical):
            raise ValueError("empirical must be finite")
        expected = self.covered / self.completed
        if empirical != expected:
            raise ValueError(
                "empirical must equal covered / completed"
            )
        object.__setattr__(self, "empirical", empirical)


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


def _simulate_bootstrap_coverage_range(
    *,
    dgp: DataGenerator,
    n: int,
    simulations: int,
    target_check: MeanTargetCheck,
    method: BootstrapMeanPercentile,
    root_seed: int,
    rng: np.random.Generator,
    batch_size: int,
    start: int,
    stop: int,
    initial_covered: int,
    progress_callback: Callable[[BootstrapCoverageProgress], object] | None = None,
) -> int:
    """Run a committed logical range without changing RNG assignment."""

    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    normalized_batch_size = _positive_integer("batch_size", batch_size)
    normalized_start = _non_negative_integer("start", start)
    normalized_stop = _non_negative_integer("stop", stop)
    normalized_initial_covered = _non_negative_integer(
        "initial_covered",
        initial_covered,
    )
    if normalized_start > normalized_stop:
        raise ValueError("start must not exceed stop")
    if normalized_stop > simulations:
        raise ValueError("stop must not exceed simulations")
    if normalized_initial_covered > normalized_start:
        raise ValueError("initial_covered must not exceed start")
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("progress_callback must be callable or None")

    coverage_count = normalized_initial_covered
    for batch_start in range(
        normalized_start,
        normalized_stop,
        normalized_batch_size,
    ):
        batch_stop = min(
            batch_start + normalized_batch_size,
            normalized_stop,
        )
        batch_samples: list[tuple[int, np.ndarray]] = []

        for logical_outer_index in range(batch_start, batch_stop):
            sample = _checked_outer_sample(
                dgp,
                rng,
                n,
                logical_outer_index=logical_outer_index,
            )
            batch_samples.append((logical_outer_index, sample))

        for logical_outer_index, sample in batch_samples:
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

        if progress_callback is not None:
            progress_callback(
                BootstrapCoverageProgress(
                    completed=batch_stop,
                    total=simulations,
                    covered=coverage_count,
                    empirical=coverage_count / batch_stop,
                )
            )

    return coverage_count


def _simulate_bootstrap_coverage_batched(
    *,
    dgp: DataGenerator,
    n: int,
    simulations: int,
    target_check: MeanTargetCheck,
    method: BootstrapMeanPercentile,
    root_seed: int,
    rng: np.random.Generator,
    batch_size: int,
    progress_callback: Callable[[BootstrapCoverageProgress], object] | None = None,
) -> int:
    """Run the complete outer experiment through the range kernel."""

    return _simulate_bootstrap_coverage_range(
        dgp=dgp,
        n=n,
        simulations=simulations,
        target_check=target_check,
        method=method,
        root_seed=root_seed,
        rng=rng,
        batch_size=batch_size,
        start=0,
        stop=simulations,
        initial_covered=0,
        progress_callback=progress_callback,
    )


def _resume_bootstrap_mean_coverage(
    *,
    dgp: DataGenerator,
    checkpoint: BootstrapCoverageCheckpoint,
    batch_size: int,
) -> tuple[BootstrapCoverageResult, np.random.Generator]:
    """Resume one validated coverage checkpoint without exposing a public API."""

    if not isinstance(checkpoint, BootstrapCoverageCheckpoint):
        raise TypeError("checkpoint must be a BootstrapCoverageCheckpoint")
    normalized_batch_size = _positive_integer("batch_size", batch_size)

    experiment = checkpoint.experiment.bind_dgp(dgp)
    origin_rng = np.random.Generator(
        np.random.PCG64(experiment.root_seed)
    )
    execution = ExecutionContract.from_generator(origin_rng)
    checkpoint.validate_for(experiment, execution)

    outer_rng = checkpoint.state.rng.restore_generator()
    coverage_count = _simulate_bootstrap_coverage_range(
        dgp=dgp,
        n=experiment.n,
        simulations=experiment.simulations,
        target_check=experiment.target_check,
        method=experiment.method_config,
        root_seed=experiment.root_seed,
        rng=outer_rng,
        batch_size=normalized_batch_size,
        start=checkpoint.state.completed,
        stop=experiment.simulations,
        initial_covered=checkpoint.state.covered,
    )

    dgp_name = getattr(dgp, "name", None)
    if not isinstance(dgp_name, str) or not dgp_name:
        raise TypeError("DGP must expose a non-empty name string")

    result = BootstrapCoverageResult.from_coverage_count(
        dgp=dgp_name,
        dgp_identity=experiment.dgp_identity,
        n=experiment.n,
        simulations=experiment.simulations,
        seed=experiment.root_seed,
        method_config=experiment.method_config,
        target_check=experiment.target_check,
        coverage_count=coverage_count,
        tolerance=experiment.tolerance,
        evidence_confidence_level=experiment.evidence_confidence_level,
        evidence_interval_method=experiment.evidence_interval_method,
    )
    return result, outer_rng


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
    batch_size: int = 64,
    progress_callback: Callable[[BootstrapCoverageProgress], object] | None = None,
) -> BootstrapCoverageResult:
    """Estimate percentile-bootstrap mean interval coverage by Monte Carlo.

    Outer batching changes only execution scheduling. DGP samples and bootstrap
    child streams remain assigned by logical outer index. No bootstrap
    vectorization, progress, checkpoint, or resume semantics are present.
    """

    normalized_n = _positive_integer("n", n)
    normalized_simulations = _positive_integer("simulations", simulations)
    normalized_seed = _non_negative_integer("seed", seed)
    normalized_tolerance = _non_negative_finite_real("tolerance", tolerance)
    normalized_evidence_confidence = _strict_probability(
        "evidence_confidence_level",
        evidence_confidence_level,
    )
    normalized_batch_size = _positive_integer("batch_size", batch_size)
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("progress_callback must be callable or None")
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
    coverage_count = _simulate_bootstrap_coverage_batched(
        dgp=dgp,
        n=normalized_n,
        simulations=normalized_simulations,
        target_check=target_check,
        method=method_config,
        root_seed=normalized_seed,
        rng=outer_rng,
        batch_size=normalized_batch_size,
        progress_callback=progress_callback,
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
    "BootstrapCoverageProgress",
    "BootstrapCoverageResult",
    "bootstrap_mean_coverage",
    "bootstrap_mean_coverage_event",
]
