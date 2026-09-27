from __future__ import annotations

import numpy as np

from .dgp import DataGenerator
from .methods import welch_ttest_pvalue
from .metrics import type1_error_summary
from .result import StressTestResult


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
    ``method='welch_ttest'`` + ``metric='type1_error'``.

    ``dgp`` and ``dgp2`` should represent a null hypothesis with equal means.
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
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between 0 and 1")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")

    other = dgp if dgp2 is None else dgp2
    rng = np.random.default_rng(seed)

    rejections = 0
    for _ in range(simulations):
        x = dgp.sample(rng, n1)
        y = other.sample(rng, n2)
        pvalue = welch_ttest_pvalue(x, y)
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
    )
