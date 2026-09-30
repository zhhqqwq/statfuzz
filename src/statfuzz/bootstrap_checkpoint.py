from __future__ import annotations

import json
import math
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

from .checkpoint import (
    CheckpointSchemaError,
    ExecutionContract,
    ExperimentFingerprint,
    RNGSnapshot,
)
from .dgp import DGPIdentity
from .methods.bootstrap import BootstrapMeanPercentile
from .targets import MeanTarget, MeanTargetCheck, resolve_mean_target

BOOTSTRAP_COVERAGE_EXPERIMENT_SCHEMA = "statfuzz.bootstrap_coverage.experiment/1"
BOOTSTRAP_COVERAGE_METRIC = "coverage"
BOOTSTRAP_COVERAGE_METRIC_SEMANTICS_VERSION = "1"
BOOTSTRAP_COVERAGE_OUTER_BIT_GENERATOR = (
    f"{np.random.PCG64.__module__}.{np.random.PCG64.__qualname__}"
)


def _canonical_json(data: object) -> str:
    try:
        return json.dumps(
            data,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise CheckpointSchemaError(
            "bootstrap coverage checkpoint data is not canonical JSON"
        ) from exc


def _require_object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise CheckpointSchemaError(f"{name} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise CheckpointSchemaError(f"{name} keys must be strings")
    return value


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
        raise CheckpointSchemaError(
            f"{name} has unexpected keys ({', '.join(details)})"
        )


def _require_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise CheckpointSchemaError(f"{name} must be a non-empty string")
    return value


def _require_int(
    value: object,
    name: str,
    *,
    minimum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise CheckpointSchemaError(f"{name} must be an integer")
    normalized = int(value)
    if minimum is not None and normalized < minimum:
        raise CheckpointSchemaError(f"{name} must be at least {minimum}")
    return normalized


def _require_finite_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise CheckpointSchemaError(f"{name} must be a finite real number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise CheckpointSchemaError(f"{name} must be finite")
    return normalized


def _persistent_dgp_identity(dgp: object) -> DGPIdentity:
    identity = getattr(dgp, "identity", None)
    if not isinstance(identity, DGPIdentity):
        raise CheckpointSchemaError(
            "DGP must provide an explicit stable DGPIdentity "
            "for persistent bootstrap coverage checkpointing"
        )
    return identity


@dataclass(frozen=True)
class BootstrapCoverageExperimentSpec:
    """Canonical semantic identity for a bootstrap mean coverage experiment."""

    dgp_identity: DGPIdentity
    target_check: MeanTargetCheck
    method_config: BootstrapMeanPercentile
    n: int
    simulations: int
    tolerance: float
    evidence_confidence_level: float
    evidence_interval_method: str
    root_seed: int
    schema: str = BOOTSTRAP_COVERAGE_EXPERIMENT_SCHEMA
    metric: str = BOOTSTRAP_COVERAGE_METRIC
    metric_semantics_version: str = BOOTSTRAP_COVERAGE_METRIC_SEMANTICS_VERSION

    def __post_init__(self) -> None:
        if self.schema != BOOTSTRAP_COVERAGE_EXPERIMENT_SCHEMA:
            raise CheckpointSchemaError(
                f"unsupported bootstrap coverage experiment schema {self.schema!r}"
            )
        if self.metric != BOOTSTRAP_COVERAGE_METRIC:
            raise CheckpointSchemaError(
                "bootstrap coverage metric must be 'coverage'"
            )
        if (
            self.metric_semantics_version
            != BOOTSTRAP_COVERAGE_METRIC_SEMANTICS_VERSION
        ):
            raise CheckpointSchemaError(
                "unsupported bootstrap coverage metric semantics version"
            )
        if not isinstance(self.dgp_identity, DGPIdentity):
            raise CheckpointSchemaError("dgp_identity must be a DGPIdentity")
        if not isinstance(self.target_check, MeanTargetCheck):
            raise CheckpointSchemaError("target_check must be a MeanTargetCheck")
        if not isinstance(self.method_config, BootstrapMeanPercentile):
            raise CheckpointSchemaError(
                "method_config must be a BootstrapMeanPercentile"
            )

        object.__setattr__(
            self,
            "n",
            _require_int(self.n, "n", minimum=1),
        )
        object.__setattr__(
            self,
            "simulations",
            _require_int(self.simulations, "simulations", minimum=1),
        )

        tolerance = _require_finite_real(self.tolerance, "tolerance")
        if tolerance < 0:
            raise CheckpointSchemaError("tolerance must be non-negative")
        object.__setattr__(self, "tolerance", tolerance)

        confidence = _require_finite_real(
            self.evidence_confidence_level,
            "evidence_confidence_level",
        )
        if not 0.0 < confidence < 1.0:
            raise CheckpointSchemaError(
                "evidence_confidence_level must be strictly between 0 and 1"
            )
        object.__setattr__(
            self,
            "evidence_confidence_level",
            confidence,
        )

        if self.evidence_interval_method != "wilson":
            raise CheckpointSchemaError(
                "evidence_interval_method must currently be 'wilson'"
            )

        object.__setattr__(
            self,
            "root_seed",
            _require_int(self.root_seed, "root_seed", minimum=0),
        )

    @classmethod
    def from_coverage_config(
        cls,
        *,
        dgp: object,
        n: int = 20,
        simulations: int = 1_000,
        method: BootstrapMeanPercentile | None = None,
        tolerance: float = 0.01,
        seed: int = 0,
        mean_target: MeanTarget | None = None,
        evidence_confidence_level: float = 0.95,
        evidence_interval_method: str = "wilson",
    ) -> BootstrapCoverageExperimentSpec:
        if mean_target is not None and not isinstance(mean_target, MeanTarget):
            raise CheckpointSchemaError(
                "mean_target must be a MeanTarget or None"
            )
        if method is None:
            method_config = BootstrapMeanPercentile()
        elif isinstance(method, BootstrapMeanPercentile):
            method_config = method
        else:
            raise CheckpointSchemaError(
                "method must be a BootstrapMeanPercentile or None"
            )

        try:
            target_check = resolve_mean_target(dgp, mean_target)
        except (TypeError, ValueError) as exc:
            raise CheckpointSchemaError(
                "failed to resolve bootstrap coverage mean target"
            ) from exc

        return cls(
            dgp_identity=_persistent_dgp_identity(dgp),
            target_check=target_check,
            method_config=method_config,
            n=n,
            simulations=simulations,
            tolerance=tolerance,
            evidence_confidence_level=evidence_confidence_level,
            evidence_interval_method=evidence_interval_method,
            root_seed=seed,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "method": self.method_config.as_dict(),
            "metric": {
                "name": self.metric,
                "semantics_version": self.metric_semantics_version,
            },
            "dgp": self.dgp_identity.as_dict(),
            "target": self.target_check.as_dict(),
            "sample_size": {
                "n": self.n,
            },
            "budget": {
                "simulations": self.simulations,
            },
            "result_semantics": {
                "tolerance": self.tolerance,
                "evidence_confidence_level": self.evidence_confidence_level,
                "evidence_interval_method": self.evidence_interval_method,
            },
            "rng_origin": {
                "root_seed": self.root_seed,
            },
        }

    def canonical_json(self) -> str:
        return _canonical_json(self.as_dict())

    def fingerprint(
        self,
        execution: ExecutionContract,
    ) -> ExperimentFingerprint:
        if not isinstance(execution, ExecutionContract):
            raise TypeError("execution must be an ExecutionContract")
        if execution.bit_generator != BOOTSTRAP_COVERAGE_OUTER_BIT_GENERATOR:
            raise CheckpointSchemaError(
                "bootstrap coverage outer execution must use PCG64"
            )
        return ExperimentFingerprint.compute(self, execution)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, object],
    ) -> BootstrapCoverageExperimentSpec:
        data = _require_object(data, "bootstrap coverage experiment")
        _require_exact_keys(
            data,
            {
                "schema",
                "method",
                "metric",
                "dgp",
                "target",
                "sample_size",
                "budget",
                "result_semantics",
                "rng_origin",
            },
            "bootstrap coverage experiment",
        )

        method = _require_object(
            data["method"],
            "bootstrap coverage experiment.method",
        )
        metric = _require_object(
            data["metric"],
            "bootstrap coverage experiment.metric",
        )
        dgp = _require_object(
            data["dgp"],
            "bootstrap coverage experiment.dgp",
        )
        target = _require_object(
            data["target"],
            "bootstrap coverage experiment.target",
        )
        sample_size = _require_object(
            data["sample_size"],
            "bootstrap coverage experiment.sample_size",
        )
        budget = _require_object(
            data["budget"],
            "bootstrap coverage experiment.budget",
        )
        result_semantics = _require_object(
            data["result_semantics"],
            "bootstrap coverage experiment.result_semantics",
        )
        rng_origin = _require_object(
            data["rng_origin"],
            "bootstrap coverage experiment.rng_origin",
        )

        _require_exact_keys(
            metric,
            {"name", "semantics_version"},
            "bootstrap coverage experiment.metric",
        )
        _require_exact_keys(
            dgp,
            {"family", "parameters"},
            "bootstrap coverage experiment.dgp",
        )
        _require_exact_keys(
            target,
            {"kind", "source", "mean", "population_mean", "note"},
            "bootstrap coverage experiment.target",
        )
        _require_exact_keys(
            sample_size,
            {"n"},
            "bootstrap coverage experiment.sample_size",
        )
        _require_exact_keys(
            budget,
            {"simulations"},
            "bootstrap coverage experiment.budget",
        )
        _require_exact_keys(
            result_semantics,
            {
                "tolerance",
                "evidence_confidence_level",
                "evidence_interval_method",
            },
            "bootstrap coverage experiment.result_semantics",
        )
        _require_exact_keys(
            rng_origin,
            {"root_seed"},
            "bootstrap coverage experiment.rng_origin",
        )

        try:
            method_config = BootstrapMeanPercentile.from_dict(method)
            dgp_identity = DGPIdentity.from_dict(dgp)
            target_check = MeanTargetCheck.from_dict(target)
        except (TypeError, ValueError) as exc:
            raise CheckpointSchemaError(
                "bootstrap coverage experiment contains invalid semantic identity"
            ) from exc

        return cls(
            schema=_require_string(
                data["schema"],
                "bootstrap coverage experiment.schema",
            ),
            method_config=method_config,
            metric=_require_string(
                metric["name"],
                "bootstrap coverage experiment.metric.name",
            ),
            metric_semantics_version=_require_string(
                metric["semantics_version"],
                "bootstrap coverage experiment.metric.semantics_version",
            ),
            dgp_identity=dgp_identity,
            target_check=target_check,
            n=sample_size["n"],
            simulations=budget["simulations"],
            tolerance=result_semantics["tolerance"],
            evidence_confidence_level=(
                result_semantics["evidence_confidence_level"]
            ),
            evidence_interval_method=(
                result_semantics["evidence_interval_method"]
            ),
            root_seed=rng_origin["root_seed"],
        )


@dataclass(frozen=True)
class BootstrapCoverageCheckpointState:
    """Committed bootstrap coverage state at a complete outer-batch boundary."""

    completed: int
    covered: int
    rng: RNGSnapshot

    def __post_init__(self) -> None:
        completed = _require_int(
            self.completed,
            "state.completed",
            minimum=0,
        )
        covered = _require_int(
            self.covered,
            "state.covered",
            minimum=0,
        )
        if covered > completed:
            raise CheckpointSchemaError(
                "state.covered cannot exceed state.completed"
            )
        if not isinstance(self.rng, RNGSnapshot):
            raise CheckpointSchemaError("state.rng must be an RNGSnapshot")
        if self.rng.bit_generator != BOOTSTRAP_COVERAGE_OUTER_BIT_GENERATOR:
            raise CheckpointSchemaError(
                "bootstrap coverage outer RNG snapshot must use PCG64"
            )

        object.__setattr__(self, "completed", completed)
        object.__setattr__(self, "covered", covered)

    @classmethod
    def capture(
        cls,
        *,
        completed: int,
        covered: int,
        rng: np.random.Generator,
    ) -> BootstrapCoverageCheckpointState:
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator")
        snapshot = RNGSnapshot.from_generator(rng)
        return cls(
            completed=completed,
            covered=covered,
            rng=snapshot,
        )

    def validate_for(
        self,
        experiment: BootstrapCoverageExperimentSpec,
    ) -> None:
        if not isinstance(experiment, BootstrapCoverageExperimentSpec):
            raise TypeError(
                "experiment must be a BootstrapCoverageExperimentSpec"
            )
        if self.completed > experiment.simulations:
            raise CheckpointSchemaError(
                "state.completed cannot exceed experiment simulations"
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "completed": self.completed,
            "covered": self.covered,
            "rng": self.rng.as_dict(),
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, object],
    ) -> BootstrapCoverageCheckpointState:
        data = _require_object(data, "bootstrap coverage state")
        _require_exact_keys(
            data,
            {"completed", "covered", "rng"},
            "bootstrap coverage state",
        )
        return cls(
            completed=data["completed"],
            covered=data["covered"],
            rng=RNGSnapshot.from_dict(
                _require_object(
                    data["rng"],
                    "bootstrap coverage state.rng",
                )
            ),
        )


__all__ = [
    "BOOTSTRAP_COVERAGE_EXPERIMENT_SCHEMA",
    "BOOTSTRAP_COVERAGE_METRIC_SEMANTICS_VERSION",
    "BOOTSTRAP_COVERAGE_OUTER_BIT_GENERATOR",
    "BootstrapCoverageCheckpointState",
    "BootstrapCoverageExperimentSpec",
]
