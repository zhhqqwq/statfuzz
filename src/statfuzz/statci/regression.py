from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from ..dgp.base import DGPIdentity
from ..methods.bootstrap import BootstrapMeanPercentile
from ..targets import MeanTargetCheck
from .model import StatCIResult, _BootstrapCoverageStatCIResult
from .suite import StatCISuiteResult

STATCI_REGRESSION_SCHEMA_VERSION = "1.2"
UncertaintyMode = Literal["conservative", "independent"]


def _canonical_json(data: object) -> str:
    return json.dumps(
        data,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _dgp_identity_key(display_name: str, identity: object) -> str:
    canonical = getattr(identity, "canonical_json", None)
    if callable(canonical):
        return str(canonical())
    return f"legacy-display:{display_name}"


def _null_identity_key(result: StatCIResult) -> str:
    if result.null_check is None:
        return "legacy-null:unrecorded"
    payload = {
        "kind": "equal_means",
        "common_mean": result.null_check.common_mean,
    }
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )



@dataclass(frozen=True, order=True)
class StatCIComparisonKey:
    """Stable experiment/assertion identity used to match baseline and current checks."""

    property: str
    target: float
    tolerance: float
    method: str
    metric: str
    dgp1_identity: str
    dgp2_identity: str
    null_identity: str
    n1: int
    n2: int
    dgp1_display: str = field(compare=False)
    dgp2_display: str = field(compare=False)

    @classmethod
    def from_result(cls, result: StatCIResult) -> StatCIComparisonKey:
        if not isinstance(result, StatCIResult):
            raise TypeError("result must be a StatCIResult")
        if result.evidence_kind == "bootstrap_coverage":
            raise TypeError(
                "Bootstrap coverage suite matching is not enabled yet; "
                "use BootstrapCoverageComparisonKey for coverage identity"
            )
        return cls(
            property=result.property,
            target=result.target,
            tolerance=result.tolerance,
            method=result.method,
            metric=result.metric,
            dgp1_identity=_dgp_identity_key(
                result.dgp1,
                result.dgp1_identity,
            ),
            dgp2_identity=_dgp_identity_key(
                result.dgp2,
                result.dgp2_identity,
            ),
            null_identity=_null_identity_key(result),
            n1=result.n1,
            n2=result.n2,
            dgp1_display=result.dgp1,
            dgp2_display=result.dgp2,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "property": self.property,
            "target": self.target,
            "tolerance": self.tolerance,
            "method": self.method,
            "metric": self.metric,
            "dgp1": self.dgp1_display,
            "dgp2": self.dgp2_display,
            "dgp1_identity": self.dgp1_identity,
            "dgp2_identity": self.dgp2_identity,
            "null_identity": self.null_identity,
            "n1": self.n1,
            "n2": self.n2,
        }

    def describe(self) -> str:
        return (
            f"{self.property} | {self.method} | "
            f"{self.dgp1_display} vs {self.dgp2_display} | "
            f"n={self.n1}/{self.n2} | "
            f"target={self.target:g} ± {self.tolerance:g}"
        )


@dataclass(frozen=True, order=True)
class BootstrapCoverageComparisonKey:
    """Stable identity for one Bootstrap coverage StatCI check."""

    property: str
    target: float
    tolerance: float
    method: str
    metric: str
    dgp_identity: str
    target_identity: str
    bootstrap_method_identity: str
    n: int
    dgp_display: str = field(compare=False)

    @classmethod
    def from_result(
        cls,
        result: StatCIResult,
    ) -> BootstrapCoverageComparisonKey:
        if not isinstance(result, _BootstrapCoverageStatCIResult):
            raise TypeError(
                "result must be a Bootstrap coverage StatCI result"
            )
        if result.dgp_identity is None:
            raise ValueError("Bootstrap coverage result requires dgp_identity")
        if result.target_check is None:
            raise ValueError("Bootstrap coverage result requires target_check")
        if result.bootstrap_method is None:
            raise ValueError(
                "Bootstrap coverage result requires bootstrap_method"
            )
        if result.n is None:
            raise ValueError("Bootstrap coverage result requires n")

        return cls(
            property=result.property,
            target=result.target,
            tolerance=result.tolerance,
            method=result.method,
            metric=result.metric,
            dgp_identity=result.dgp_identity.canonical_json(),
            target_identity=_canonical_json(result.target_check.as_dict()),
            bootstrap_method_identity=result.bootstrap_method.canonical_json(),
            n=result.n,
            dgp_display=result.dgp,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": "bootstrap_coverage",
            "property": self.property,
            "target": self.target,
            "tolerance": self.tolerance,
            "method": self.method,
            "metric": self.metric,
            "dgp": self.dgp_display,
            "dgp_identity": json.loads(self.dgp_identity),
            "target_check": json.loads(self.target_identity),
            "bootstrap_method": json.loads(self.bootstrap_method_identity),
            "n": self.n,
        }

    def canonical_json(self) -> str:
        return _canonical_json(self.as_dict())

    @classmethod
    def from_dict(
        cls,
        data: dict[str, object],
    ) -> BootstrapCoverageComparisonKey:
        if not isinstance(data, dict):
            raise TypeError(
                "Bootstrap coverage comparison key must be an object"
            )
        expected = {
            "kind",
            "property",
            "target",
            "tolerance",
            "method",
            "metric",
            "dgp",
            "dgp_identity",
            "target_check",
            "bootstrap_method",
            "n",
        }
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
                "Bootstrap coverage comparison key has unexpected keys "
                f"({', '.join(details)})"
            )
        if data["kind"] != "bootstrap_coverage":
            raise ValueError(
                "unsupported Bootstrap coverage comparison key kind"
            )

        property_name = data["property"]
        method = data["method"]
        metric = data["metric"]
        dgp_display = data["dgp"]
        for name, value in (
            ("property", property_name),
            ("method", method),
            ("metric", metric),
            ("dgp", dgp_display),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if property_name != metric:
            raise ValueError("property must match metric")
        if metric != "coverage":
            raise ValueError("Bootstrap coverage key metric must be 'coverage'")

        target = data["target"]
        tolerance = data["tolerance"]
        if isinstance(target, bool) or not isinstance(target, (int, float)):
            raise TypeError("target must be a real number")
        if isinstance(tolerance, bool) or not isinstance(
            tolerance,
            (int, float),
        ):
            raise TypeError("tolerance must be a real number")
        target = float(target)
        tolerance = float(tolerance)
        if not math.isfinite(target):
            raise ValueError("target must be finite")
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError("tolerance must be finite and non-negative")

        n = data["n"]
        if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
            raise ValueError("n must be a positive integer")

        raw_dgp = data["dgp_identity"]
        raw_target = data["target_check"]
        raw_method = data["bootstrap_method"]
        if not isinstance(raw_dgp, dict):
            raise TypeError("dgp_identity must be an object")
        if not isinstance(raw_target, dict):
            raise TypeError("target_check must be an object")
        if not isinstance(raw_method, dict):
            raise TypeError("bootstrap_method must be an object")

        dgp_identity = DGPIdentity.from_dict(raw_dgp)
        target_check = MeanTargetCheck.from_dict(raw_target)
        bootstrap_method = BootstrapMeanPercentile.from_dict(raw_method)
        if method != bootstrap_method.method:
            raise ValueError(
                "method must match bootstrap_method.method"
            )

        return cls(
            property=property_name,
            target=target,
            tolerance=tolerance,
            method=method,
            metric=metric,
            dgp_identity=dgp_identity.canonical_json(),
            target_identity=_canonical_json(target_check.as_dict()),
            bootstrap_method_identity=bootstrap_method.canonical_json(),
            n=n,
            dgp_display=dgp_display,
        )

    def describe(self) -> str:
        return (
            f"{self.property} | {self.method} | {self.dgp_display} | "
            f"n={self.n} | target={self.target:g} ± {self.tolerance:g}"
        )


@dataclass(frozen=True)
class RegressionPolicy:
    """Engineering policy for deciding whether a matched check regressed."""

    minimum_worsening: float = 0.0
    uncertainty_multiplier: float = 2.0
    uncertainty_mode: UncertaintyMode = "conservative"
    strict_matching: bool = True

    def __post_init__(self) -> None:
        for name, value in (
            ("minimum_worsening", self.minimum_worsening),
            ("uncertainty_multiplier", self.uncertainty_multiplier),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.uncertainty_mode not in {"conservative", "independent"}:
            raise ValueError(
                "uncertainty_mode must be 'conservative' or 'independent'"
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "minimum_worsening": self.minimum_worsening,
            "uncertainty_multiplier": self.uncertainty_multiplier,
            "uncertainty_mode": self.uncertainty_mode,
            "strict_matching": self.strict_matching,
        }


def _uncertainty_scale(
    baseline: StatCIResult,
    current: StatCIResult,
    policy: RegressionPolicy,
) -> float:
    if policy.uncertainty_mode == "conservative":
        return baseline.mcse + current.mcse

    if baseline.seed is None or current.seed is None:
        raise ValueError(
            "independent uncertainty mode requires known baseline and current seeds"
        )
    if baseline.seed == current.seed:
        raise ValueError(
            "independent uncertainty mode requires distinct baseline and current seeds"
        )
    return math.sqrt(baseline.mcse**2 + current.mcse**2)


StatCIRegressionKey = StatCIComparisonKey | BootstrapCoverageComparisonKey


def _comparison_key_for_result(
    result: StatCIResult,
) -> StatCIRegressionKey:
    if result.evidence_kind == "bootstrap_coverage":
        return BootstrapCoverageComparisonKey.from_result(result)
    return StatCIComparisonKey.from_result(result)


@dataclass(frozen=True)
class StatCIRegressionResult:
    """Uncertainty-aware comparison for one matched StatCI check."""

    key: StatCIRegressionKey
    baseline: StatCIResult
    current: StatCIResult
    uncertainty_scale: float
    uncertainty_allowance: float
    regression_threshold: float
    worsening: float
    regressed: bool
    direction: str
    pass_to_fail: bool
    fail_to_pass: bool

    @property
    def status(self) -> str:
        return "REGRESSION" if self.regressed else "PASS"

    @property
    def observed_change(self) -> float:
        return self.current.observed - self.baseline.observed

    def as_dict(self) -> dict[str, object]:
        return {
            "key": self.key.as_dict(),
            "status": self.status,
            "regressed": self.regressed,
            "direction": self.direction,
            "pass_to_fail": self.pass_to_fail,
            "fail_to_pass": self.fail_to_pass,
            "observed_change": self.observed_change,
            "worsening": self.worsening,
            "uncertainty_scale": self.uncertainty_scale,
            "uncertainty_allowance": self.uncertainty_allowance,
            "regression_threshold": self.regression_threshold,
            "baseline": self.baseline.as_dict(),
            "current": self.current.as_dict(),
        }


def compare_results(
    baseline: StatCIResult,
    current: StatCIResult,
    *,
    policy: RegressionPolicy | None = None,
) -> StatCIRegressionResult:
    """Compare one exactly matched baseline/current StatCI result pair."""

    if not isinstance(baseline, StatCIResult) or not isinstance(current, StatCIResult):
        raise TypeError("baseline and current must be StatCIResult instances")

    resolved_policy = RegressionPolicy() if policy is None else policy
    if not isinstance(resolved_policy, RegressionPolicy):
        raise TypeError("policy must be a RegressionPolicy")

    baseline_key = _comparison_key_for_result(baseline)
    current_key = _comparison_key_for_result(current)
    if baseline_key != current_key:
        raise ValueError(
            "baseline and current checks do not have the same comparison key"
        )

    uncertainty_scale = _uncertainty_scale(baseline, current, resolved_policy)
    uncertainty_allowance = (
        resolved_policy.uncertainty_multiplier * uncertainty_scale
    )
    regression_threshold = (
        resolved_policy.minimum_worsening + uncertainty_allowance
    )
    worsening = current.absolute_deviation - baseline.absolute_deviation
    regressed = worsening > regression_threshold

    if regressed:
        direction = "REGRESSION"
    elif worsening < 0:
        direction = "IMPROVED"
    elif worsening == 0:
        direction = "STABLE"
    else:
        direction = "WITHIN_GUARD"

    return StatCIRegressionResult(
        key=baseline_key,
        baseline=baseline,
        current=current,
        uncertainty_scale=uncertainty_scale,
        uncertainty_allowance=uncertainty_allowance,
        regression_threshold=regression_threshold,
        worsening=worsening,
        regressed=regressed,
        direction=direction,
        pass_to_fail=baseline.passed and not current.passed,
        fail_to_pass=not baseline.passed and current.passed,
    )


def _regression_key_sort_key(
    key: StatCIRegressionKey,
) -> tuple[object, ...]:
    if isinstance(key, StatCIComparisonKey):
        return (
            0,
            key.property,
            key.target,
            key.tolerance,
            key.method,
            key.metric,
            key.dgp1_identity,
            key.dgp2_identity,
            key.null_identity,
            key.n1,
            key.n2,
        )
    if isinstance(key, BootstrapCoverageComparisonKey):
        return (
            1,
            key.property,
            key.target,
            key.tolerance,
            key.method,
            key.metric,
            key.dgp_identity,
            key.target_identity,
            key.bootstrap_method_identity,
            key.n,
        )
    raise TypeError("unsupported StatCI regression comparison key")


def _index_suite(
    suite: StatCISuiteResult,
    *,
    role: str,
) -> dict[StatCIRegressionKey, StatCIResult]:
    index: dict[StatCIRegressionKey, StatCIResult] = {}
    for result in suite.results:
        key = _comparison_key_for_result(result)
        if key in index:
            raise ValueError(
                f"{role} suite contains duplicate comparison key: {key.describe()}"
            )
        index[key] = result
    return index


@dataclass(frozen=True)
class StatCIRegressionSuiteResult:
    """Aggregate baseline-vs-current regression comparison."""

    name: str
    baseline_name: str
    current_name: str
    policy: RegressionPolicy
    comparisons: tuple[StatCIRegressionResult, ...]
    missing_current: tuple[StatCIRegressionKey, ...] = ()
    new_current: tuple[StatCIRegressionKey, ...] = ()
    schema_version: str = STATCI_REGRESSION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string")
        if not self.comparisons:
            raise ValueError("regression suite requires at least one comparison")

    @property
    def passed(self) -> bool:
        return not any(comparison.regressed for comparison in self.comparisons)

    @property
    def status(self) -> str:
        return "PASS" if self.passed else "FAIL"

    @property
    def total(self) -> int:
        return len(self.comparisons)

    @property
    def regression_count(self) -> int:
        return sum(comparison.regressed for comparison in self.comparisons)

    @property
    def pass_to_fail_count(self) -> int:
        return sum(comparison.pass_to_fail for comparison in self.comparisons)

    @property
    def improvement_count(self) -> int:
        return sum(
            comparison.direction == "IMPROVED"
            for comparison in self.comparisons
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "baseline_name": self.baseline_name,
            "current_name": self.current_name,
            "status": self.status,
            "passed": self.passed,
            "total": self.total,
            "regression_count": self.regression_count,
            "pass_to_fail_count": self.pass_to_fail_count,
            "improvement_count": self.improvement_count,
            "policy": self.policy.as_dict(),
            "missing_current": [key.as_dict() for key in self.missing_current],
            "new_current": [key.as_dict() for key in self.new_current],
            "comparisons": [
                comparison.as_dict()
                for comparison in self.comparisons
            ],
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


def compare_suites(
    baseline: StatCISuiteResult,
    current: StatCISuiteResult,
    *,
    policy: RegressionPolicy | None = None,
    name: str = "StatCI regression",
) -> StatCIRegressionSuiteResult:
    """Match and compare two StatCI suites using an explicit regression policy."""

    if not isinstance(baseline, StatCISuiteResult):
        raise TypeError("baseline must be a StatCISuiteResult")
    if not isinstance(current, StatCISuiteResult):
        raise TypeError("current must be a StatCISuiteResult")

    resolved_policy = RegressionPolicy() if policy is None else policy
    if not isinstance(resolved_policy, RegressionPolicy):
        raise TypeError("policy must be a RegressionPolicy")

    baseline_index = _index_suite(baseline, role="baseline")
    current_index = _index_suite(current, role="current")

    baseline_keys = set(baseline_index)
    current_keys = set(current_index)
    missing_current = tuple(
        sorted(
            baseline_keys - current_keys,
            key=_regression_key_sort_key,
        )
    )
    new_current = tuple(
        sorted(
            current_keys - baseline_keys,
            key=_regression_key_sort_key,
        )
    )

    if resolved_policy.strict_matching and (missing_current or new_current):
        pieces = []
        if missing_current:
            pieces.append(
                "missing current checks: "
                + "; ".join(key.describe() for key in missing_current)
            )
        if new_current:
            pieces.append(
                "unexpected current checks: "
                + "; ".join(key.describe() for key in new_current)
            )
        raise ValueError("suite comparison key mismatch; " + " | ".join(pieces))

    matched_keys = tuple(
        sorted(
            baseline_keys & current_keys,
            key=_regression_key_sort_key,
        )
    )
    if not matched_keys:
        raise ValueError("baseline and current suites have no matching checks")

    comparisons = tuple(
        compare_results(
            baseline_index[key],
            current_index[key],
            policy=resolved_policy,
        )
        for key in matched_keys
    )

    return StatCIRegressionSuiteResult(
        name=name,
        baseline_name=baseline.name,
        current_name=current.name,
        policy=resolved_policy,
        comparisons=comparisons,
        missing_current=missing_current,
        new_current=new_current,
    )


class StatCIRegressionError(AssertionError):
    """Assertion failure carrying a failed regression-comparison suite."""

    def __init__(self, result: StatCIRegressionSuiteResult) -> None:
        self.result = result
        super().__init__(
            f"StatCI regression suite {result.name!r} failed: "
            f"{result.regression_count}/{result.total} matched checks regressed"
        )


def assert_regression_suite(
    result: StatCIRegressionSuiteResult,
) -> StatCIRegressionSuiteResult:
    if not isinstance(result, StatCIRegressionSuiteResult):
        raise TypeError("result must be a StatCIRegressionSuiteResult")
    if not result.passed:
        raise StatCIRegressionError(result)
    return result


def _md(value: object) -> str:
    text = str(value)
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")


def _number(value: float) -> str:
    return f"{value:.6g}"


def render_regression_summary(result: StatCIRegressionSuiteResult) -> str:
    """Render an uncertainty-aware baseline comparison for GitHub Actions."""

    if not isinstance(result, StatCIRegressionSuiteResult):
        raise TypeError("result must be a StatCIRegressionSuiteResult")

    lines = [
        f"## StatCI Regression — {_md(result.name)}",
        "",
        f"**Overall:** {result.status}  ",
        f"**Matched checks:** {result.total}  ",
        f"**Regressions:** {result.regression_count}  ",
        f"**PASS→FAIL transitions:** {result.pass_to_fail_count}  ",
        f"**Improvements:** {result.improvement_count}",
        "",
        (
            f"Policy: minimum worsening={_number(result.policy.minimum_worsening)}, "
            f"uncertainty={_number(result.policy.uncertainty_multiplier)} × "
            f"{result.policy.uncertainty_mode} uncertainty scale."
        ),
        "",
        "| Property | Direction | Baseline | Current | Worsening | Uncertainty scale | Guard threshold | PASS→FAIL |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for comparison in result.comparisons:
        lines.append(
            "| "
            + " | ".join(
                (
                    _md(comparison.key.property),
                    comparison.direction,
                    _number(comparison.baseline.absolute_deviation),
                    _number(comparison.current.absolute_deviation),
                    _number(comparison.worsening),
                    _number(comparison.uncertainty_scale),
                    _number(comparison.regression_threshold),
                    "yes" if comparison.pass_to_fail else "no",
                )
            )
            + " |"
        )

    regressions = [
        comparison
        for comparison in result.comparisons
        if comparison.regressed
    ]
    if regressions:
        lines.extend(["", "### Regressions", ""])
        for comparison in regressions:
            lines.append(
                f"- **{_md(comparison.key.describe())}**: "
                f"worsening={_number(comparison.worsening)} > "
                f"guard={_number(comparison.regression_threshold)}"
            )

    if result.missing_current or result.new_current:
        lines.extend(["", "### Unmatched checks", ""])
        for key in result.missing_current:
            lines.append(f"- Missing current: {_md(key.describe())}")
        for key in result.new_current:
            lines.append(f"- New current: {_md(key.describe())}")

    return "\n".join(lines) + "\n"


def write_regression_summary(
    result: StatCIRegressionSuiteResult,
    path: str | Path | None = None,
) -> Path:
    if path is None:
        configured = os.environ.get("GITHUB_STEP_SUMMARY")
        if not configured:
            raise RuntimeError(
                "GITHUB_STEP_SUMMARY is not set; provide path explicitly "
                "when running outside GitHub Actions"
            )
        target = Path(configured)
    else:
        target = Path(path)

    with target.open("a", encoding="utf-8") as handle:
        handle.write(render_regression_summary(result))
    return target
