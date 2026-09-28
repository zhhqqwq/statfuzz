from __future__ import annotations

import math

import numpy as np

from .dgp import DataGenerator
from .dgp.base import get_dgp_identity
from .methods import welch_ttest_pvalue, welch_ttest_pvalues_batch
from .metrics import type1_error_evidence
from .nulls import MeanEqualityNull, verify_mean_equality_null
from .result import StressTestResult

DEFAULT_BATCH_SIZE = 64


def _checked_sample(
    dgp: DataGenerator,
    rng: np.random.Generator,
    n: int,
    *,
    group: str,
    simulation_index: int,
) -> np.ndarray:
    try:
        raw = dgp.sample(rng, n)
        sample = np.asarray(raw, dtype=float)
    except Exception as exc:
        raise RuntimeError(
            f"simulation {simulation_index}: failed to generate {group} sample"
        ) from exc

    if sample.shape != (n,):
        raise RuntimeError(
            f"simulation {simulation_index}: {group} sample must have shape {(n,)}, "
            f"got {sample.shape}"
        )
    if not np.all(np.isfinite(sample)):
        raise RuntimeError(
            f"simulation {simulation_index}: {group} sample contains non-finite values"
        )
    return sample


def _checked_batch_pvalues(
    xs: list[np.ndarray],
    ys: list[np.ndarray],
    *,
    simulation_start: int,
) -> np.ndarray:
    if len(xs) == 1:
        pvalues = np.asarray([welch_ttest_pvalue(xs[0], ys[0])], dtype=float)
    else:
        try:
            pvalues = welch_ttest_pvalues_batch(xs, ys)
        except Exception as exc:
            simulation_end = simulation_start + len(xs) - 1
            raise RuntimeError(
                "failed to evaluate Welch p-values for simulations "
                f"{simulation_start}..{simulation_end}"
            ) from exc

    if pvalues.shape != (len(xs),):
        raise RuntimeError("method returned an unexpected number of p-values")

    for offset, pvalue in enumerate(pvalues):
        value = float(pvalue)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            simulation_index = simulation_start + offset
            raise RuntimeError(
                f"simulation {simulation_index}: method returned invalid "
                f"p-value {value!r}"
            )
    return pvalues


def _simulate_rejections(
    *,
    dgp: DataGenerator,
    other: DataGenerator,
    n1: int,
    n2: int,
    simulations: int,
    alpha: float,
    rng: np.random.Generator,
    batch_size: int,
) -> int:
    """Run logical replicates without letting batch boundaries alter RNG use."""

    rejections = 0
    for batch_start in range(0, simulations, batch_size):
        batch_stop = min(batch_start + batch_size, simulations)
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []

        for simulation_index in range(batch_start, batch_stop):
            xs.append(
                _checked_sample(
                    dgp,
                    rng,
                    n1,
                    group="group 1",
                    simulation_index=simulation_index,
                )
            )
            ys.append(
                _checked_sample(
                    other,
                    rng,
                    n2,
                    group="group 2",
                    simulation_index=simulation_index,
                )
            )

        pvalues = _checked_batch_pvalues(
            xs,
            ys,
            simulation_start=batch_start,
        )
        rejections += int(np.count_nonzero(pvalues < alpha))

    return rejections


def stress_test(
    *,
    method: str,
    metric: str,
    dgp: DataGenerator,
    dgp2: DataGenerator | None = None,
    n1: int = 20,
    n2: int | None = None,
    simulations: int = 10_000,
    alpha: float = 0.05,
    tolerance: float = 0.01,
    seed: int | None = 0,
    null: MeanEqualityNull | None = None,
    confidence_level: float = 0.95,
    interval_method: str = "wilson",
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> StressTestResult:
    """Estimate a statistical method's finite-sample property by Monte Carlo.

    Logical replicates always consume randomness in the same order. batch_size
    changes only when already-generated samples are statistically evaluated.
    batch_size=1 uses the original scalar Welch reference path.
    """

    if method != "welch_ttest":
        raise ValueError("v0.1 supports only method='welch_ttest'")
    if metric != "type1_error":
        raise ValueError("v0.1 supports only metric='type1_error'")
    if n1 < 2:
        raise ValueError("n1 must be at least 2")
    if n2 is None:
        n2 = n1
    if n2 < 2:
        raise ValueError("n2 must be at least 2")
    if simulations <= 0:
        raise ValueError("simulations must be positive")
    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size <= 0
    ):
        raise ValueError("batch_size must be a positive integer")
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be finite and strictly between 0 and 1")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative")
    if null is not None and not isinstance(null, MeanEqualityNull):
        raise TypeError("null must be a MeanEqualityNull or None")
    if not math.isfinite(confidence_level) or not 0 < confidence_level < 1:
        raise ValueError(
            "confidence_level must be finite and strictly between 0 and 1"
        )
    if interval_method != "wilson":
        raise ValueError("interval_method must currently be 'wilson'")

    other = dgp if dgp2 is None else dgp2
    null_check = verify_mean_equality_null(dgp, other, null)
    dgp1_identity = get_dgp_identity(dgp)
    dgp2_identity = get_dgp_identity(other)
    rng = np.random.default_rng(seed)

    rejections = _simulate_rejections(
        dgp=dgp,
        other=other,
        n1=n1,
        n2=n2,
        simulations=simulations,
        alpha=alpha,
        rng=rng,
        batch_size=batch_size,
    )

    evidence = type1_error_evidence(
        rejections,
        simulations,
        alpha,
        confidence_level=confidence_level,
        interval_method=interval_method,
    )

    return StressTestResult(
        method=method,
        metric=metric,
        dgp1=dgp.name,
        dgp2=other.name,
        n1=n1,
        n2=n2,
        simulations=simulations,
        seed=seed,
        nominal=alpha,
        empirical=evidence.empirical,
        mcse=evidence.mcse,
        tolerance=tolerance,
        dgp1_identity=dgp1_identity,
        dgp2_identity=dgp2_identity,
        null_check=null_check,
        rejection_count=evidence.rejection_count,
        confidence_level=evidence.confidence_level,
        interval_method=evidence.interval_method,
        interval_low=evidence.interval_low,
        interval_high=evidence.interval_high,
    )
