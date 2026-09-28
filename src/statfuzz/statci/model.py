from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from ..result import StressTestResult

STATCI_SCHEMA_VERSION = "1.0"


def _finite(name: str, value: object) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
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

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> StatCIResult:
        """Reconstruct and validate a StatCIResult from its JSON-shaped payload."""

        if not isinstance(data, dict):
            raise TypeError("StatCIResult payload must be a dictionary")
        if data.get("schema_version") != STATCI_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported StatCI schema_version: {data.get('schema_version')!r}"
            )

        evidence = data.get("evidence")
        if not isinstance(evidence, dict):
            raise TypeError("StatCIResult payload requires an evidence object")

        property_name = data.get("property")
        metric = evidence.get("metric")
        method = evidence.get("method")
        dgp1 = evidence.get("dgp1")
        dgp2 = evidence.get("dgp2")
        if not isinstance(property_name, str) or not property_name:
            raise ValueError("property must be a non-empty string")
        if not isinstance(metric, str) or not metric:
            raise ValueError("evidence.metric must be a non-empty string")
        if property_name != metric:
            raise ValueError("property must match evidence.metric")
        for name, value in (("method", method), ("dgp1", dgp1), ("dgp2", dgp2)):
            if not isinstance(value, str) or not value:
                raise ValueError(f"evidence.{name} must be a non-empty string")

        passed = data.get("passed")
        if not isinstance(passed, bool):
            raise TypeError("passed must be a boolean")

        target = _finite("target", data.get("target"))
        tolerance = _finite("tolerance", data.get("tolerance"))
        observed = _finite("observed", data.get("observed"))
        deviation = _finite("deviation", data.get("deviation"))
        absolute_deviation = _finite(
            "absolute_deviation",
            data.get("absolute_deviation"),
        )
        mcse = _finite("evidence.mcse", evidence.get("mcse"))
        if tolerance < 0:
            raise ValueError("tolerance must be non-negative")
        if mcse < 0:
            raise ValueError("evidence.mcse must be non-negative")

        n1 = evidence.get("n1")
        n2 = evidence.get("n2")
        simulations = evidence.get("simulations")
        seed = evidence.get("seed")
        for name, value in (("n1", n1), ("n2", n2), ("simulations", simulations)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"evidence.{name} must be a positive integer")
        if seed is not None and (
            not isinstance(seed, int) or isinstance(seed, bool) or seed < 0
        ):
            raise ValueError("evidence.seed must be a non-negative integer or null")

        expected_deviation = observed - target
        if not math.isclose(deviation, expected_deviation, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("deviation is inconsistent with observed - target")
        if not math.isclose(
            absolute_deviation,
            abs(deviation),
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("absolute_deviation is inconsistent with deviation")

        expected_passed = absolute_deviation <= tolerance
        if passed != expected_passed:
            raise ValueError("passed is inconsistent with tolerance")
        if data.get("status") != ("PASS" if passed else "FAIL"):
            raise ValueError("status is inconsistent with passed")

        return cls(
            property=property_name,
            target=target,
            tolerance=tolerance,
            observed=observed,
            deviation=deviation,
            absolute_deviation=absolute_deviation,
            passed=passed,
            method=method,
            metric=metric,
            dgp1=dgp1,
            dgp2=dgp2,
            n1=n1,
            n2=n2,
            simulations=simulations,
            seed=seed,
            mcse=mcse,
        )


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
