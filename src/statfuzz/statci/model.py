from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from ..bootstrap_coverage import BootstrapCoverageResult
from ..dgp.base import DGPIdentity
from ..methods.bootstrap import BootstrapMeanPercentile
from ..nulls import MeanNullCheck
from ..result import StatisticalPropertyResult, StressTestResult
from ..targets import MeanTargetCheck

STATCI_SCHEMA_VERSION = "1.2"
_SUPPORTED_STATCI_SCHEMA_VERSIONS = {"1.0", "1.1", STATCI_SCHEMA_VERSION}


def _require_exact_keys(
    data: dict[str, object],
    expected: set[str],
    name: str,
) -> None:
    actual = set(data)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        details = []
        if missing:
            details.append(f"missing={missing!r}")
        if extra:
            details.append(f"extra={extra!r}")
        raise ValueError(
            f"{name} has unexpected keys ({', '.join(details)})"
        )


def _non_empty_string(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _positive_int(name: str, value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _non_negative_int(name: str, value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


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

    @property
    def evidence_kind(self) -> str:
        return "stress_test"

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
        if evidence.get("kind") == "bootstrap_coverage":
            return _bootstrap_coverage_result_from_dict(data, evidence)

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
    target_check: MeanTargetCheck | None = None
    coverage_count: int | None = None
    bootstrap_method: BootstrapMeanPercentile | None = None
    evidence_confidence_level: float | None = None
    evidence_interval_method: str | None = None
    evidence_interval_low: float | None = None
    evidence_interval_high: float | None = None
    n: int | None = None

    @property
    def evidence_kind(self) -> str:
        return "bootstrap_coverage"

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
                "target_check": self.target_check.as_dict(),
                "coverage_count": self.coverage_count,
                "bootstrap_method": self.bootstrap_method.as_dict(),
                "evidence_interval": {
                    "level": self.evidence_confidence_level,
                    "method": self.evidence_interval_method,
                    "low": self.evidence_interval_low,
                    "high": self.evidence_interval_high,
                },
            },
        }


def _bootstrap_coverage_result_from_dict(
    data: dict[str, object],
    evidence: dict[str, object],
) -> StatCIResult:
    """Strictly reconstruct one persisted Bootstrap coverage StatCI result."""

    _require_exact_keys(
        data,
        {
            "schema_version",
            "property",
            "target",
            "tolerance",
            "observed",
            "deviation",
            "absolute_deviation",
            "passed",
            "status",
            "evidence",
        },
        "Bootstrap coverage StatCI result",
    )
    _require_exact_keys(
        evidence,
        {
            "kind",
            "method",
            "metric",
            "dgp",
            "dgp_identity",
            "n",
            "simulations",
            "seed",
            "mcse",
            "target_check",
            "coverage_count",
            "bootstrap_method",
            "evidence_interval",
        },
        "Bootstrap coverage StatCI evidence",
    )
    if evidence["kind"] != "bootstrap_coverage":
        raise ValueError("unsupported Bootstrap coverage evidence kind")

    property_name = _non_empty_string("property", data["property"])
    method = _non_empty_string("evidence.method", evidence["method"])
    metric = _non_empty_string("evidence.metric", evidence["metric"])
    if property_name != metric:
        raise ValueError("property must match evidence.metric")
    if metric != "coverage":
        raise ValueError("Bootstrap coverage evidence metric must be 'coverage'")

    dgp = _non_empty_string("evidence.dgp", evidence["dgp"])
    raw_dgp_identity = evidence["dgp_identity"]
    raw_target_check = evidence["target_check"]
    raw_bootstrap_method = evidence["bootstrap_method"]
    raw_interval = evidence["evidence_interval"]
    if not isinstance(raw_dgp_identity, dict):
        raise TypeError("evidence.dgp_identity must be an object")
    if not isinstance(raw_target_check, dict):
        raise TypeError("evidence.target_check must be an object")
    if not isinstance(raw_bootstrap_method, dict):
        raise TypeError("evidence.bootstrap_method must be an object")
    if not isinstance(raw_interval, dict):
        raise TypeError("evidence.evidence_interval must be an object")

    _require_exact_keys(
        raw_target_check,
        {"kind", "source", "mean", "population_mean", "note"},
        "evidence.target_check",
    )
    _require_exact_keys(
        raw_interval,
        {"level", "method", "low", "high"},
        "evidence.evidence_interval",
    )

    dgp_identity = DGPIdentity.from_dict(raw_dgp_identity)
    target_check = MeanTargetCheck.from_dict(raw_target_check)
    bootstrap_method = BootstrapMeanPercentile.from_dict(
        raw_bootstrap_method
    )
    if method != bootstrap_method.method:
        raise ValueError(
            "evidence.method must match evidence.bootstrap_method.method"
        )

    n = _positive_int("evidence.n", evidence["n"])
    simulations = _positive_int(
        "evidence.simulations",
        evidence["simulations"],
    )
    seed = evidence["seed"]
    if seed is not None:
        seed = _non_negative_int("evidence.seed", seed)
    coverage_count = _non_negative_int(
        "evidence.coverage_count",
        evidence["coverage_count"],
    )
    if coverage_count > simulations:
        raise ValueError(
            "evidence.coverage_count must not exceed simulations"
        )

    target = _finite("target", data["target"])
    tolerance = _finite("tolerance", data["tolerance"])
    observed = _finite("observed", data["observed"])
    deviation = _finite("deviation", data["deviation"])
    absolute_deviation = _finite(
        "absolute_deviation",
        data["absolute_deviation"],
    )
    mcse = _finite("evidence.mcse", evidence["mcse"])
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    if mcse < 0:
        raise ValueError("evidence.mcse must be non-negative")

    expected_observed = coverage_count / simulations
    if not math.isclose(
        observed,
        expected_observed,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError(
            "observed is inconsistent with coverage_count / simulations"
        )

    confidence_level = _finite(
        "evidence.evidence_interval.level",
        raw_interval["level"],
    )
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(
            "evidence.evidence_interval.level must be between 0 and 1"
        )
    interval_method = _non_empty_string(
        "evidence.evidence_interval.method",
        raw_interval["method"],
    )
    interval_low = _finite(
        "evidence.evidence_interval.low",
        raw_interval["low"],
    )
    interval_high = _finite(
        "evidence.evidence_interval.high",
        raw_interval["high"],
    )
    if not 0.0 <= interval_low <= observed <= interval_high <= 1.0:
        raise ValueError(
            "Bootstrap coverage evidence interval must contain observed "
            "within [0, 1]"
        )

    expected_deviation = observed - target
    if not math.isclose(
        deviation,
        expected_deviation,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("deviation is inconsistent with observed - target")
    if not math.isclose(
        absolute_deviation,
        abs(deviation),
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError(
            "absolute_deviation is inconsistent with deviation"
        )

    passed = data["passed"]
    if not isinstance(passed, bool):
        raise TypeError("passed must be a boolean")
    expected_passed = absolute_deviation <= tolerance
    if passed != expected_passed:
        raise ValueError("passed is inconsistent with tolerance")
    if data["status"] != ("PASS" if passed else "FAIL"):
        raise ValueError("status is inconsistent with passed")

    return _BootstrapCoverageStatCIResult(
        property=property_name,
        target=target,
        tolerance=tolerance,
        observed=observed,
        deviation=deviation,
        absolute_deviation=absolute_deviation,
        passed=passed,
        method=method,
        metric=metric,
        dgp1=dgp,
        dgp2=dgp,
        n1=n,
        n2=n,
        simulations=simulations,
        seed=seed,
        mcse=mcse,
        dgp1_identity=dgp_identity,
        dgp2_identity=dgp_identity,
        dgp=dgp,
        dgp_identity=dgp_identity,
        target_check=target_check,
        coverage_count=coverage_count,
        bootstrap_method=bootstrap_method,
        evidence_confidence_level=confidence_level,
        evidence_interval_method=interval_method,
        evidence_interval_low=interval_low,
        evidence_interval_high=interval_high,
        n=n,
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
                target_check=result.target_check,
                coverage_count=result.coverage_count,
                bootstrap_method=result.method_config,
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
