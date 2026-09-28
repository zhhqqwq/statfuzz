from __future__ import annotations

import math

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


def welch_ttest_pvalue(x: np.ndarray, y: np.ndarray) -> float:
    """Two-sided Welch t-test p-value.

    Implemented directly rather than wrapping scipy.stats.ttest_ind so the
    statistical calculation remains explicit and easy to audit.
    """

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
            return 1.0 if mean1 == mean2 else 0.0
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

    pvalue = float(2.0 * t.sf(abs(statistic), df=df))
    if not math.isfinite(pvalue) or not 0.0 <= pvalue <= 1.0:
        raise ValueError(f"Welch p-value is invalid: {pvalue!r}")
    return pvalue
