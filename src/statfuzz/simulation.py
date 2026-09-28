from __future__ import annotations

import math

import numpy as np

from .dgp import DataGenerator
from .dgp.base import get_dgp_identity
from .methods import welch_ttest_pvalue
from .metrics import type1_error_summary
from .result import StressTestResult


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
) -> StressTestResult:
    """Estimate a statistical method's finite-sample property by Monte Carlo.

    StatFuzz v0.1 intentionally supports one validated path:
    method='welch_ttest' + metric='type1_error'.

    dgp and dgp2 should represent a null hypothesis with equal means.
    Built-in generators expose explicit mean controls to make that easy.
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
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be finite and strictly between 0 and 1")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative")

    other = dgp if dgp2 is None else dgp2
    dgp1_identity = get_dgp_identity(dgp)
    dgp2_identity = get_dgp_identity(other)
    rng = np.random.default_rng(seed)

    rejections = 0
    for simulation_index in range(simulations):
        x = _checked_sample(
            dgp,
            rng,
            n1,
            group="group 1",
            simulation_index=simulation_index,
        )
        y = _checked_sample(
            other,
            rng,
            n2,
            group="group 2",
            simulation_index=simulation_index,
        )
        pvalue = welch_ttest_pvalue(x, y)
        if not math.isfinite(pvalue) or not 0.0 <= pvalue <= 1.0:
            raise RuntimeError(
                f"simulation {simulation_index}: method returned invalid "
                f"p-value {pvalue!r}"
            )
        rejections += pvalue < alpha

    empirical, mcse = type1_error_summary(rejections, simulations, alpha)

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
        empirical=empirical,
        mcse=mcse,
        tolerance=tolerance,
        dgp1_identity=dgp1_identity,
        dgp2_identity=dgp2_identity,
    )
