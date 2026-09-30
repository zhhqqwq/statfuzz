from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from ..bootstrap_coverage import BootstrapCoverageResult
from ..dgp.base import DGPIdentity
from ..nulls import MeanNullCheck
from ..result import StatisticalPropertyResult, StressTestResult

STATCI_SCHEMA_VERSION = "1.2"
_SUPPORTED_STATCI_SCHEMA_VERSIONS = {"1.0", "1.1", STATCI_SCHEMA_VERSION}


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
    dgp1_identity: DGPIdentity | None = None
    dgp2_identity: DGPIdentity | None = None
    null_check: MeanNullCheck | None = None
    rejection_count: int | None = None
    confidence_level: float | None = None
    interval_method: str | None = None
    interval_low: float | None = None
    interval_high: float | None = None
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
                "dgp1_identity": (
                    None
                    if self.dgp1_identity is None
                    else self.dgp1_identity.as_dict()
                ),
                "dgp2_identity": (
                    None
                    if self.dgp2_identity is None
                    else self.dgp2_identity.as_dict()
                ),
                "n1": self.n1,
                "n2": self.n2,
                "simulations": self.simulations,
                "seed": self.seed,
                "mcse": self.mcse,
                "null_check": (
                    None if self.null_check is None else self.null_check.as_dict()
                ),
                "rejection_count": self.rejection_count,
                "confidence_interval": (
                    None
                    if self.confidence_level is None
                    else {
                        "level": self.confidence_level,
                        "method": self.interval_method,
                        "low": self.interval_low,
                        "high": self.interval_high,
                    }
                ),
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
        schema_version = data.get("schema_version")
        if schema_version not in _SUPPORTED_STATCI_SCHEMA_VERSIONS:
            raise ValueError(
                f"unsupported StatCI schema_version: {schema_version!r}"
            )

        evidence = data.get("evidence")
        if not isinstance(evidence, dict):
            raise TypeError("StatCIResult payload requires an evidence object")

        property_name = data.get("property")
        metric = evidence.get("metric")
        method = evidence.get("method")
        dgp1 = evidence.get("dgp1")
        dgp2 = evidence.get("dgp2")
        raw_dgp1_identity = evidence.get("dgp1_identity")
        raw_dgp2_identity = evidence.get("dgp2_identity")
        raw_null_check = evidence.get("null_check")
        if not isinstance(property_name, str) or not property_name:
            raise ValueError("property must be a non-empty string")
        if not isinstance(metric, str) or not metric:
            raise ValueError("evidence.metric must be a non-empty string")
        if property_name != metric:
            raise ValueError("property must match evidence.metric")
        for name, value in (("method", method), ("dgp1", dgp1), ("dgp2", dgp2)):
            if not isinstance(value, str) or not value:
                raise ValueError(f"evidence.{name} must be a non-empty string")

        dgp1_identity = (
            None
            if raw_dgp1_identity is None
            else DGPIdentity.from_dict(raw_dgp1_identity)
        )
        dgp2_identity = (
            None
            if raw_dgp2_identity is None
            else DGPIdentity.from_dict(raw_dgp2_identity)
        )
        null_check = (
            None
            if raw_null_check is None
            else MeanNullCheck.from_dict(raw_null_check)
        )

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

        rejection_count = evidence.get("rejection_count")
        raw_interval = evidence.get("confidence_interval")
        confidence_level = None
        interval_method = None
        interval_low = None
        interval_high = None

        if rejection_count is not None or raw_interval is not None:
            if (
                not isinstance(rejection_count, int)
                or isinstance(rejection_count, bool)
            ):
                raise ValueError("evidence.rejection_count must be an integer")
            if not 0 <= rejection_count <= simulations:
                raise ValueError(
                    "evidence.rejection_count must be between 0 and simulations"
                )
            if not math.isclose(
                observed,
                rejection_count / simulations,
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                raise ValueError(
                    "observed is inconsistent with rejection_count / simulations"
                )
            if not isinstance(raw_interval, dict):
                raise TypeError("evidence.confidence_interval must be an object")
            confidence_level = _finite(
                "evidence.confidence_interval.level",
                raw_interval.get("level"),
            )
            if not 0 < confidence_level < 1:
                raise ValueError(
                    "evidence.confidence_interval.level must be between 0 and 1"
                )
            interval_method = raw_interval.get("method")
            if not isinstance(interval_method, str) or not interval_method:
                raise ValueError(
                    "evidence.confidence_interval.method must be a non-empty string"
                )
            interval_low = _finite(
                "evidence.confidence_interval.low",
                raw_interval.get("low"),
            )
            interval_high = _finite(
                "evidence.confidence_interval.high",
                raw_interval.get("high"),
            )
            if not 0 <= interval_low <= observed <= interval_high <= 1:
                raise ValueError(
                    "confidence interval must contain observed within [0, 1]"
                )

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
            dgp1_identity=dgp1_identity,
            dgp2_identity=dgp2_identity,
            null_check=null_check,
            rejection_count=rejection_count,
            confidence_level=confidence_level,
            interval_method=interval_method,
            interval_low=interval_low,
            interval_high=interval_high,
        )


@dataclass(frozen=True)
class _BootstrapCoverageStatCIResult(StatCIResult):
    """Current-run StatCI evidence for one Bootstrap coverage assertion."""

    dgp: str = ""
    dgp_identity: DGPIdentity | None = None
    target_check: dict[str, object] | None = None
    coverage_count: int | None = None
    bootstrap_method: dict[str, object] | None = None
    evidence_confidence_level: float | None = None
    evidence_interval_method: str | None = None
    evidence_interval_low: float | None = None
    evidence_interval_high: float | None = None
    n: int | None = None

    def as_dict(self) -> dict[str, object]:
        if self.dgp_identity is None:
            raise ValueError(
                "Bootstrap coverage StatCI evidence requires dgp_identity"
            )
        if self.target_check is None:
            raise ValueError(
                "Bootstrap coverage StatCI evidence requires target_check"
            )
        if self.coverage_count is None:
            raise ValueError(
                "Bootstrap coverage StatCI evidence requires coverage_count"
            )
        if self.bootstrap_method is None:
            raise ValueError(
                "Bootstrap coverage StatCI evidence requires bootstrap_method"
            )
        if self.evidence_confidence_level is None:
            raise ValueError(
                "Bootstrap coverage StatCI evidence requires evidence interval"
            )
        if self.evidence_interval_method is None:
            raise ValueError(
                "Bootstrap coverage StatCI evidence requires evidence interval"
            )
        if self.evidence_interval_low is None or self.evidence_interval_high is None:
            raise ValueError(
                "Bootstrap coverage StatCI evidence requires evidence interval"
            )
        if self.n is None:
            raise ValueError("Bootstrap coverage StatCI evidence requires n")

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
                "kind": "bootstrap_coverage",
                "method": self.method,
                "metric": self.metric,
                "dgp": self.dgp,
                "dgp_identity": self.dgp_identity.as_dict(),
                "n": self.n,
                "simulations": self.simulations,
                "seed": self.seed,
                "mcse": self.mcse,
                "target_check": dict(self.target_check),
                "coverage_count": self.coverage_count,
                "bootstrap_method": dict(self.bootstrap_method),
                "evidence_interval": {
                    "level": self.evidence_confidence_level,
                    "method": self.evidence_interval_method,
                    "low": self.evidence_interval_low,
                    "high": self.evidence_interval_high,
                },
            },
        }


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

    def evaluate(
        self,
        result: StatisticalPropertyResult,
    ) -> StatCIResult:
        """Evaluate the assertion without raising on statistical failure."""

        if not isinstance(result, StatisticalPropertyResult):
            raise TypeError(
                "result must satisfy StatisticalPropertyResult"
            )
        if result.metric != self.property:
            raise ValueError(
                f"assertion property {self.property!r} does not match "
                f"result.metric {result.metric!r}"
            )

        observed = _finite("result.empirical", result.empirical)
        mcse = _finite("result.mcse", result.mcse)
        if mcse < 0:
            raise ValueError("result.mcse must be non-negative")

        deviation = observed - self.target
        absolute_deviation = abs(deviation)
        passed = absolute_deviation <= self.tolerance

        common = {
            "property": self.property,
            "target": self.target,
            "tolerance": self.tolerance,
            "observed": observed,
            "deviation": deviation,
            "absolute_deviation": absolute_deviation,
            "passed": passed,
            "method": result.method,
            "metric": result.metric,
            "simulations": result.simulations,
            "seed": result.seed,
            "mcse": mcse,
        }

        if isinstance(result, StressTestResult):
            return StatCIResult(
                **common,
                dgp1=result.dgp1,
                dgp2=result.dgp2,
                n1=result.n1,
                n2=result.n2,
                dgp1_identity=result.dgp1_identity,
                dgp2_identity=result.dgp2_identity,
                null_check=result.null_check,
                rejection_count=result.rejection_count,
                confidence_level=result.confidence_level,
                interval_method=result.interval_method,
                interval_low=result.interval_low,
                interval_high=result.interval_high,
            )

        if isinstance(result, BootstrapCoverageResult):
            return _BootstrapCoverageStatCIResult(
                **common,
                dgp1=result.dgp,
                dgp2=result.dgp,
                n1=result.n,
                n2=result.n,
                dgp1_identity=result.dgp_identity,
                dgp2_identity=result.dgp_identity,
                dgp=result.dgp,
                dgp_identity=result.dgp_identity,
                target_check=result.target_check.as_dict(),
                coverage_count=result.coverage_count,
                bootstrap_method=result.method_config.as_dict(),
                evidence_confidence_level=result.evidence_confidence_level,
                evidence_interval_method=result.evidence_interval_method,
                evidence_interval_low=result.evidence_interval_low,
                evidence_interval_high=result.evidence_interval_high,
                n=result.n,
            )

        raise TypeError(
            "StatCI currently supports StressTestResult and "
            "BootstrapCoverageResult"
        )
