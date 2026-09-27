from __future__ import annotations

import math


def type1_error_summary(rejections: int, simulations: int, nominal: float) -> tuple[float, float]:
    if simulations <= 0:
        raise ValueError("simulations must be positive")
    if not 0 < nominal < 1:
        raise ValueError("nominal must be strictly between 0 and 1")
    empirical = rejections / simulations
    mcse = math.sqrt(empirical * (1.0 - empirical) / simulations)
    return empirical, mcse
