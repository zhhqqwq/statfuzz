from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from ..result import StressTestResult

STATCI_SCHEMA_VERSION = "1.0"


def _finite(name: str, value: float) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


@dataclass(frozen=True)
class StatCIResult:
    """Machine-readable result of one statistical property assertion."""

    property: str
    target: float
    tolerance: float
    observed: float
    deviation: float
    absolute_deviation: float
    passed: bool
    method: str
    metric: str
    dgp1: str
    dgp2: str
    n1: int
    n2: int
    simulations: int
    seed: int | None
    mcse: float
    schema_version: str = STATCI_SCHEMA_VERSION

    @property
    def status(self) -> str:
        return "PASS" if self.passed else "FAIL"

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "property": self.property,
            "target": self.target,
            "tolerance": self.tolerance,
            "observed": self.observed,
            "deviation": self.deviation,
            "absolute_deviation": self.absolute_deviation,
            "passed": self.passed,
            "status": self.status,
            "evidence": {
                "method": self.method,
                "metric": self.metric,
                "dgp1": self.dgp1,
                "dgp2": self.dgp2,
                "n1": self.n1,
                "n2": self.n2,
                "simulations": self.simulations,
                "seed": self.seed,
                "mcse": self.mcse,
            },
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(
            self.as_dict(),
            ensure_ascii=False,
            sort_keys=True,
            indent=indent,
            allow_nan=False,
        )

    def write_json(
        self,
        path: str | Path,
        *,
        indent: int = 2,
    ) -> Path:
        target = Path(path)
        target.write_text(self.to_json(indent=indent) + "\n", encoding="utf-8")
        return target


@dataclass(frozen=True)
class StatisticalAssertion:
    """Engineering tolerance assertion for one statistical property."""

    property: str
    target: float
    tolerance: float

    def __post_init__(self) -> None:
        if not isinstance(self.property, str) or not self.property:
            raise ValueError("property must be a non-empty string")

        target = _finite("target", self.target)
        tolerance = _finite("tolerance", self.tolerance)
        if tolerance < 0:
            raise ValueError("tolerance must be non-negative")

        object.__setattr__(self, "target", target)
        object.__setattr__(self, "tolerance", tolerance)

    def evaluate(self, result: StressTestResult) -> StatCIResult:
        """Evaluate the assertion without raising on statistical failure."""

        if not isinstance(result, StressTestResult):
            raise TypeError("result must be a StressTestResult")
        if result.metric != self.property:
            raise ValueError(
                f"assertion property {self.property!r} does not match "
                f"StressTestResult.metric {result.metric!r}"
            )

        observed = _finite("result.empirical", result.empirical)
        mcse = _finite("result.mcse", result.mcse)
        if mcse < 0:
            raise ValueError("result.mcse must be non-negative")

        deviation = observed - self.target
        absolute_deviation = abs(deviation)
        passed = absolute_deviation <= self.tolerance

        return StatCIResult(
            property=self.property,
            target=self.target,
            tolerance=self.tolerance,
            observed=observed,
            deviation=deviation,
            absolute_deviation=absolute_deviation,
            passed=passed,
            method=result.method,
            metric=result.metric,
            dgp1=result.dgp1,
            dgp2=result.dgp2,
            n1=result.n1,
            n2=result.n2,
            simulations=result.simulations,
            seed=result.seed,
            mcse=mcse,
        )
