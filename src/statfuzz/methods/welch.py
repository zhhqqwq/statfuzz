from __future__ import annotations

import numpy as np
from scipy.stats import t


def welch_ttest_pvalue(x: np.ndarray, y: np.ndarray) -> float:
    """Two-sided Welch t-test p-value.

    Implemented directly rather than wrapping scipy.stats.ttest_ind so the
    statistical calculation remains explicit and easy to audit.
    """

    n1 = x.size
    n2 = y.size
    if n1 < 2 or n2 < 2:
        raise ValueError("both samples must contain at least 2 observations")

    v1 = np.var(x, ddof=1)
    v2 = np.var(y, ddof=1)
    denom2 = v1 / n1 + v2 / n2

    if denom2 == 0:
        return 1.0 if np.mean(x) == np.mean(y) else 0.0

    statistic = (np.mean(x) - np.mean(y)) / np.sqrt(denom2)
    df_num = denom2**2
    df_den = (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
    df = df_num / df_den
    return float(2.0 * t.sf(abs(statistic), df=df))
