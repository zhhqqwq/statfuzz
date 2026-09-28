from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy.stats import t


def _as_finite_sample(name: str, values: np.ndarray) -> np.ndarray:
    sample = np.asarray(values, dtype=float)
    if sample.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if sample.size < 2:
        raise ValueError(f"{name} must contain at least 2 observations")
    if not np.all(np.isfinite(sample)):
        raise ValueError(f"{name} must contain only finite observations")
    return sample


def _welch_components(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[float | None, float | None, float | None]:
    """Return statistic/df or a deterministic zero-variance p-value override."""

    x = _as_finite_sample("x", x)
    y = _as_finite_sample("y", y)
    n1 = x.size
    n2 = y.size

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        mean1 = float(np.mean(x))
        mean2 = float(np.mean(y))
        if not math.isfinite(mean1) or not math.isfinite(mean2):
            raise ValueError("sample mean is non-finite")

        v1 = float(np.var(x, ddof=1))
        v2 = float(np.var(y, ddof=1))
        if not math.isfinite(v1) or not math.isfinite(v2):
            raise ValueError("sample variance is non-finite")

        denom2 = v1 / n1 + v2 / n2
        if not math.isfinite(denom2):
            raise ValueError("Welch denominator is non-finite")

        if denom2 == 0:
            return None, None, 1.0 if mean1 == mean2 else 0.0
        if denom2 < 0:
            raise ValueError("Welch denominator must be non-negative")

        statistic = (mean1 - mean2) / math.sqrt(denom2)
        df_num = denom2**2
        df_den = (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)

    if not math.isfinite(statistic):
        raise ValueError("Welch statistic is non-finite")
    if not math.isfinite(df_den) or df_den <= 0:
        raise ValueError("Welch degrees-of-freedom denominator is invalid")

    df = df_num / df_den
    if not math.isfinite(df) or df <= 0:
        raise ValueError("Welch degrees of freedom are invalid")

    return statistic, df, None


def welch_ttest_pvalue(x: np.ndarray, y: np.ndarray) -> float:
    """Two-sided scalar Welch t-test p-value used as the reference oracle."""

    statistic, df, override = _welch_components(x, y)
    if override is not None:
        return override

    assert statistic is not None
    assert df is not None
    pvalue = float(2.0 * t.sf(abs(statistic), df=df))
    if not math.isfinite(pvalue) or not 0.0 <= pvalue <= 1.0:
        raise ValueError(f"Welch p-value is invalid: {pvalue!r}")
    return pvalue


def welch_ttest_pvalues_batch(
    xs: Sequence[np.ndarray],
    ys: Sequence[np.ndarray],
) -> np.ndarray:
    """Welch p-values with scalar moments and one vectorized SciPy tail call.

    Sample validation plus mean/variance/statistic/df calculations intentionally
    reuse the scalar reference operations. Only scipy.stats.t.sf is vectorized.
    """

    if len(xs) != len(ys):
        raise ValueError("xs and ys must contain the same number of samples")
    if not xs:
        return np.empty(0, dtype=float)

    pvalues = np.empty(len(xs), dtype=float)
    active_indices: list[int] = []
    statistics: list[float] = []
    dfs: list[float] = []

    for index, (x, y) in enumerate(zip(xs, ys)):
        statistic, df, override = _welch_components(x, y)
        if override is not None:
            pvalues[index] = override
            continue

        assert statistic is not None
        assert df is not None
        active_indices.append(index)
        statistics.append(statistic)
        dfs.append(df)

    if active_indices:
        statistic_array = np.asarray(statistics, dtype=float)
        df_array = np.asarray(dfs, dtype=float)
        tail_values = np.asarray(
            2.0 * t.sf(np.abs(statistic_array), df=df_array),
            dtype=float,
        )
        if tail_values.shape != (len(active_indices),):
            raise ValueError("batched Welch p-value output has an unexpected shape")
        if not np.all(np.isfinite(tail_values)):
            raise ValueError("batched Welch p-value output contains non-finite values")
        if np.any((tail_values < 0.0) | (tail_values > 1.0)):
            raise ValueError("batched Welch p-value output must lie in [0, 1]")

        for index, pvalue in zip(active_indices, tail_values):
            pvalues[index] = pvalue

    return pvalues
