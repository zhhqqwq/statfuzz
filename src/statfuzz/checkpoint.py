from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy

from ._version import __version__
from .dgp import DGPIdentity
from .nulls import MeanEqualityNull, MeanNullCheck, verify_mean_equality_null

EXPERIMENT_SCHEMA = "statfuzz.experiment/1"
CHECKPOINT_SCHEMA = "statfuzz.checkpoint/1"
EXECUTION_SEMANTICS_VERSION = "1"
METHOD_SEMANTICS_VERSION = "1"
METRIC_SEMANTICS_VERSION = "1"
FINGERPRINT_ALGORITHM = "sha256"

_STATE_TAG = "__statfuzz_type__"


class CheckpointError(ValueError):
    """Base error for invalid or incompatible checkpoint data."""


class CheckpointSchemaError(CheckpointError):
    """Raised when checkpoint data does not satisfy the supported schema."""


class CheckpointFingerprintError(CheckpointError):
    """Raised when checkpoint identity does not match canonical inputs."""


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
        raise CheckpointSchemaError("checkpoint data is not canonical JSON") from exc


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
        detail_text = ", ".join(details)
        raise CheckpointSchemaError(f"{name} has unexpected keys ({detail_text})")


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
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise CheckpointSchemaError(f"{name} must be an integer")
    normalized = int(value)
    if minimum is not None and normalized < minimum:
        raise CheckpointSchemaError(f"{name} must be at least {minimum}")
    return normalized


def _require_finite_float(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise CheckpointSchemaError(f"{name} must be a finite number")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise CheckpointSchemaError(f"{name} must be a finite number") from exc
    if not math.isfinite(normalized):
        raise CheckpointSchemaError(f"{name} must be finite")
    return normalized


def _qualified_type_name(value: object) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _persistent_dgp_identity(dgp: object, group: str) -> DGPIdentity:
    identity = getattr(dgp, "identity", None)
    if not isinstance(identity, DGPIdentity):
        raise CheckpointSchemaError(
            f"{group} DGP must provide an explicit stable DGPIdentity "
            "for persistent checkpointing"
        )
    return identity


@dataclass(frozen=True)
class ExperimentSpec:
    """Canonical semantic identity for the current stress-test experiment."""

    method: str
    metric: str
    dgp1_identity: DGPIdentity
    dgp2_identity: DGPIdentity
    null_check: MeanNullCheck
    n1: int
    n2: int
    simulations: int
    alpha: float
    tolerance: float
    confidence_level: float
    interval_method: str
    root_seed: int
    schema: str = EXPERIMENT_SCHEMA
    method_semantics_version: str = METHOD_SEMANTICS_VERSION
    metric_semantics_version: str = METRIC_SEMANTICS_VERSION

    def __post_init__(self) -> None:
        if self.schema != EXPERIMENT_SCHEMA:
            raise CheckpointSchemaError(
                f"unsupported experiment schema {self.schema!r}"
            )
        if self.method != "welch_ttest":
            raise CheckpointSchemaError("only method='welch_ttest' is supported")
        if self.metric != "type1_error":
            raise CheckpointSchemaError("only metric='type1_error' is supported")
        if self.method_semantics_version != METHOD_SEMANTICS_VERSION:
            raise CheckpointSchemaError("unsupported method semantics version")
        if self.metric_semantics_version != METRIC_SEMANTICS_VERSION:
            raise CheckpointSchemaError("unsupported metric semantics version")
        if not isinstance(self.dgp1_identity, DGPIdentity):
            raise CheckpointSchemaError("dgp1_identity must be a DGPIdentity")
        if not isinstance(self.dgp2_identity, DGPIdentity):
            raise CheckpointSchemaError("dgp2_identity must be a DGPIdentity")
        if not isinstance(self.null_check, MeanNullCheck):
            raise CheckpointSchemaError("null_check must be a MeanNullCheck")

        object.__setattr__(self, "n1", _require_int(self.n1, "n1", minimum=2))
        object.__setattr__(self, "n2", _require_int(self.n2, "n2", minimum=2))
        object.__setattr__(
            self,
            "simulations",
            _require_int(self.simulations, "simulations", minimum=1),
        )

        alpha = _require_finite_float(self.alpha, "alpha")
        if not 0 < alpha < 1:
            raise CheckpointSchemaError("alpha must be strictly between 0 and 1")
        object.__setattr__(self, "alpha", alpha)

        tolerance = _require_finite_float(self.tolerance, "tolerance")
        if tolerance < 0:
            raise CheckpointSchemaError("tolerance must be non-negative")
        object.__setattr__(self, "tolerance", tolerance)

        confidence_level = _require_finite_float(
            self.confidence_level,
            "confidence_level",
        )
        if not 0 < confidence_level < 1:
            raise CheckpointSchemaError(
                "confidence_level must be strictly between 0 and 1"
            )
        object.__setattr__(self, "confidence_level", confidence_level)

        if self.interval_method != "wilson":
            raise CheckpointSchemaError("interval_method must currently be 'wilson'")

        root_seed = _require_int(self.root_seed, "root_seed", minimum=0)
        object.__setattr__(self, "root_seed", root_seed)

    @classmethod
    def from_stress_test_config(
        cls,
        *,
        method: str,
        metric: str,
        dgp: object,
        dgp2: object | None = None,
        n1: int = 20,
        n2: int | None = None,
        simulations: int = 10_000,
        alpha: float = 0.05,
        tolerance: float = 0.01,
        seed: int | None = 0,
        null: MeanEqualityNull | None = None,
        confidence_level: float = 0.95,
        interval_method: str = "wilson",
    ) -> ExperimentSpec:
        if seed is None:
            raise CheckpointSchemaError(
                "persistent checkpointing does not yet support seed=None"
            )

        resolved_n2 = n1 if n2 is None else n2
        other = dgp if dgp2 is None else dgp2
        null_check = verify_mean_equality_null(dgp, other, null)

        return cls(
            method=method,
            metric=metric,
            dgp1_identity=_persistent_dgp_identity(dgp, "group 1"),
            dgp2_identity=_persistent_dgp_identity(other, "group 2"),
            null_check=null_check,
            n1=n1,
            n2=resolved_n2,
            simulations=simulations,
            alpha=alpha,
            tolerance=tolerance,
            confidence_level=confidence_level,
            interval_method=interval_method,
            root_seed=seed,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "method": {
                "name": self.method,
                "semantics_version": self.method_semantics_version,
            },
            "metric": {
                "name": self.metric,
                "semantics_version": self.metric_semantics_version,
            },
            "dgp1": self.dgp1_identity.as_dict(),
            "dgp2": self.dgp2_identity.as_dict(),
            "null": self.null_check.as_dict(),
            "sample_sizes": {
                "n1": self.n1,
                "n2": self.n2,
            },
            "budget": {
                "simulations": self.simulations,
            },
            "decision": {
                "alpha": self.alpha,
            },
            "result_semantics": {
                "tolerance": self.tolerance,
                "confidence_level": self.confidence_level,
                "interval_method": self.interval_method,
            },
            "rng_origin": {
                "root_seed": self.root_seed,
            },
        }

    def canonical_json(self) -> str:
        return _canonical_json(self.as_dict())

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> ExperimentSpec:
        data = _require_object(data, "experiment")
        _require_exact_keys(
            data,
            {
                "schema",
                "method",
                "metric",
                "dgp1",
                "dgp2",
                "null",
                "sample_sizes",
                "budget",
                "decision",
                "result_semantics",
                "rng_origin",
            },
            "experiment",
        )

        method = _require_object(data["method"], "experiment.method")
        metric = _require_object(data["metric"], "experiment.metric")
        sample_sizes = _require_object(
            data["sample_sizes"],
            "experiment.sample_sizes",
        )
        budget = _require_object(data["budget"], "experiment.budget")
        decision = _require_object(data["decision"], "experiment.decision")
        result_semantics = _require_object(
            data["result_semantics"],
            "experiment.result_semantics",
        )
        rng_origin = _require_object(data["rng_origin"], "experiment.rng_origin")

        _require_exact_keys(
            method,
            {"name", "semantics_version"},
            "experiment.method",
        )
        _require_exact_keys(
            metric,
            {"name", "semantics_version"},
            "experiment.metric",
        )
        _require_exact_keys(sample_sizes, {"n1", "n2"}, "experiment.sample_sizes")
        _require_exact_keys(budget, {"simulations"}, "experiment.budget")
        _require_exact_keys(decision, {"alpha"}, "experiment.decision")
        _require_exact_keys(
            result_semantics,
            {"tolerance", "confidence_level", "interval_method"},
            "experiment.result_semantics",
        )
        _require_exact_keys(rng_origin, {"root_seed"}, "experiment.rng_origin")

        return cls(
            schema=_require_string(data["schema"], "experiment.schema"),
            method=_require_string(method["name"], "experiment.method.name"),
            metric=_require_string(metric["name"], "experiment.metric.name"),
            method_semantics_version=_require_string(
                method["semantics_version"],
                "experiment.method.semantics_version",
            ),
            metric_semantics_version=_require_string(
                metric["semantics_version"],
                "experiment.metric.semantics_version",
            ),
            dgp1_identity=DGPIdentity.from_dict(
                _require_object(data["dgp1"], "experiment.dgp1")
            ),
            dgp2_identity=DGPIdentity.from_dict(
                _require_object(data["dgp2"], "experiment.dgp2")
            ),
            null_check=MeanNullCheck.from_dict(
                _require_object(data["null"], "experiment.null")
            ),
            n1=sample_sizes["n1"],
            n2=sample_sizes["n2"],
            simulations=budget["simulations"],
            alpha=decision["alpha"],
            tolerance=result_semantics["tolerance"],
            confidence_level=result_semantics["confidence_level"],
            interval_method=_require_string(
                result_semantics["interval_method"],
                "experiment.result_semantics.interval_method",
            ),
            root_seed=rng_origin["root_seed"],
        )


@dataclass(frozen=True)
class ExecutionContract:
    """Strict runtime compatibility contract for exact checkpoint reuse."""

    statfuzz_execution_semantics: str
    statfuzz_version: str
    numpy_version: str
    scipy_version: str
    bit_generator: str

    def __post_init__(self) -> None:
        if self.statfuzz_execution_semantics != EXECUTION_SEMANTICS_VERSION:
            raise CheckpointSchemaError("unsupported StatFuzz execution semantics")
        for field_name in (
            "statfuzz_version",
            "numpy_version",
            "scipy_version",
            "bit_generator",
        ):
            _require_string(getattr(self, field_name), field_name)

    @classmethod
    def from_generator(cls, rng: np.random.Generator) -> ExecutionContract:
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator")
        return cls(
            statfuzz_execution_semantics=EXECUTION_SEMANTICS_VERSION,
            statfuzz_version=__version__,
            numpy_version=np.__version__,
            scipy_version=scipy.__version__,
            bit_generator=_qualified_type_name(rng.bit_generator),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "statfuzz_execution_semantics": self.statfuzz_execution_semantics,
            "statfuzz_version": self.statfuzz_version,
            "numpy_version": self.numpy_version,
            "scipy_version": self.scipy_version,
            "bit_generator": self.bit_generator,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> ExecutionContract:
        data = _require_object(data, "execution")
        _require_exact_keys(
            data,
            {
                "statfuzz_execution_semantics",
                "statfuzz_version",
                "numpy_version",
                "scipy_version",
                "bit_generator",
            },
            "execution",
        )
        return cls(
            statfuzz_execution_semantics=_require_string(
                data["statfuzz_execution_semantics"],
                "execution.statfuzz_execution_semantics",
            ),
            statfuzz_version=_require_string(
                data["statfuzz_version"],
                "execution.statfuzz_version",
            ),
            numpy_version=_require_string(
                data["numpy_version"],
                "execution.numpy_version",
            ),
            scipy_version=_require_string(
                data["scipy_version"],
                "execution.scipy_version",
            ),
            bit_generator=_require_string(
                data["bit_generator"],
                "execution.bit_generator",
            ),
        )


@dataclass(frozen=True)
class ExperimentFingerprint:
    value: str
    algorithm: str = FINGERPRINT_ALGORITHM

    def __post_init__(self) -> None:
        if self.algorithm != FINGERPRINT_ALGORITHM:
            raise CheckpointSchemaError("unsupported fingerprint algorithm")
        if (
            not isinstance(self.value, str)
            or len(self.value) != 64
            or any(char not in "0123456789abcdef" for char in self.value)
        ):
            raise CheckpointSchemaError(
                "fingerprint value must be a lowercase SHA-256 hex digest"
            )

    @classmethod
    def compute(
        cls,
        experiment: ExperimentSpec,
        execution: ExecutionContract,
    ) -> ExperimentFingerprint:
        payload = {
            "experiment": experiment.as_dict(),
            "execution": execution.as_dict(),
        }
        digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
        return cls(value=digest)

    def as_dict(self) -> dict[str, object]:
        return {
            "algorithm": self.algorithm,
            "value": self.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> ExperimentFingerprint:
        data = _require_object(data, "fingerprint")
        _require_exact_keys(data, {"algorithm", "value"}, "fingerprint")
        return cls(
            algorithm=_require_string(
                data["algorithm"],
                "fingerprint.algorithm",
            ),
            value=_require_string(data["value"], "fingerprint.value"),
        )


def encode_rng_state(value: object) -> object:
    """Losslessly convert supported NumPy RNG-state values to JSON-safe data."""

    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        normalized = float(value)
        if not math.isfinite(normalized):
            raise CheckpointSchemaError("RNG state contains a non-finite float")
        return normalized
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject or value.dtype.fields is not None:
            raise CheckpointSchemaError(
                "RNG state arrays with object or structured dtype are unsupported"
            )
        contiguous = np.ascontiguousarray(value)
        return {
            _STATE_TAG: "ndarray",
            "dtype": contiguous.dtype.str,
            "shape": list(contiguous.shape),
            "data": base64.b64encode(contiguous.tobytes(order="C")).decode("ascii"),
        }
    if isinstance(value, dict):
        items = []
        for key, item in value.items():
            if not isinstance(key, str):
                raise CheckpointSchemaError("RNG state dictionary keys must be strings")
            items.append([key, encode_rng_state(item)])
        return {
            _STATE_TAG: "dict",
            "items": items,
        }
    if isinstance(value, list):
        return {
            _STATE_TAG: "list",
            "items": [encode_rng_state(item) for item in value],
        }
    if isinstance(value, tuple):
        return {
            _STATE_TAG: "tuple",
            "items": [encode_rng_state(item) for item in value],
        }
    raise CheckpointSchemaError(
        f"unsupported RNG state value type: {type(value).__name__}"
    )


def decode_rng_state(value: object) -> object:
    """Reverse :func:`encode_rng_state` with strict tagged-data validation."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CheckpointSchemaError("encoded RNG state contains non-finite float")
        return value
    if not isinstance(value, dict):
        raise CheckpointSchemaError("encoded RNG state contains an invalid value")

    data = _require_object(value, "encoded RNG state")
    tag = data.get(_STATE_TAG)
    if tag == "ndarray":
        _require_exact_keys(
            data,
            {_STATE_TAG, "dtype", "shape", "data"},
            "encoded RNG ndarray",
        )
        dtype_text = _require_string(data["dtype"], "encoded RNG ndarray dtype")
        shape_value = data["shape"]
        if not isinstance(shape_value, list):
            raise CheckpointSchemaError("encoded RNG ndarray shape must be a list")
        shape = tuple(
            _require_int(size, "encoded RNG ndarray dimension", minimum=0)
            for size in shape_value
        )
        payload = _require_string(data["data"], "encoded RNG ndarray data")
        try:
            raw = base64.b64decode(payload.encode("ascii"), validate=True)
            dtype = np.dtype(dtype_text)
        except (ValueError, TypeError, UnicodeEncodeError) as exc:
            raise CheckpointSchemaError("encoded RNG ndarray metadata is invalid") from exc
        if dtype.hasobject or dtype.fields is not None:
            raise CheckpointSchemaError(
                "encoded RNG arrays with object or structured dtype are unsupported"
            )
        expected_size = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if len(raw) != expected_size:
            raise CheckpointSchemaError(
                "encoded RNG ndarray byte length does not match dtype and shape"
            )
        return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()

    if tag in {"dict", "list", "tuple"}:
        _require_exact_keys(data, {_STATE_TAG, "items"}, "encoded RNG container")
        items = data["items"]
        if not isinstance(items, list):
            raise CheckpointSchemaError("encoded RNG container items must be a list")

        if tag == "dict":
            result: dict[str, object] = {}
            for pair in items:
                if not isinstance(pair, list) or len(pair) != 2:
                    raise CheckpointSchemaError(
                        "encoded RNG dictionary entries must be key/value pairs"
                    )
                key = _require_string(pair[0], "encoded RNG dictionary key")
                if key in result:
                    raise CheckpointSchemaError(
                        "encoded RNG dictionary keys must be unique"
                    )
                result[key] = decode_rng_state(pair[1])
            return result

        decoded = [decode_rng_state(item) for item in items]
        if tag == "tuple":
            return tuple(decoded)
        return decoded

    raise CheckpointSchemaError("encoded RNG state has an unsupported type tag")


@dataclass(frozen=True)
class RNGSnapshot:
    bit_generator: str
    state: object

    def __post_init__(self) -> None:
        _require_string(self.bit_generator, "rng.bit_generator")
        raw_state = self.raw_state()
        if not isinstance(raw_state, dict):
            raise CheckpointSchemaError("decoded BitGenerator state must be an object")
        marker = raw_state.get("bit_generator")
        if marker is not None:
            if not isinstance(marker, str) or not marker:
                raise CheckpointSchemaError(
                    "BitGenerator state marker must be a non-empty string"
                )
            expected_marker = self.bit_generator.rsplit(".", 1)[-1]
            if marker != expected_marker:
                raise CheckpointSchemaError(
                    "BitGenerator state marker does not match declared class"
                )

    @classmethod
    def from_generator(cls, rng: np.random.Generator) -> RNGSnapshot:
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator")
        return cls(
            bit_generator=_qualified_type_name(rng.bit_generator),
            state=encode_rng_state(rng.bit_generator.state),
        )

    def raw_state(self) -> object:
        return decode_rng_state(self.state)

    def as_dict(self) -> dict[str, object]:
        return {
            "bit_generator": self.bit_generator,
            "state": self.state,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> RNGSnapshot:
        data = _require_object(data, "state.rng")
        _require_exact_keys(
            data,
            {"bit_generator", "state"},
            "state.rng",
        )
        return cls(
            bit_generator=_require_string(
                data["bit_generator"],
                "state.rng.bit_generator",
            ),
            state=data["state"],
        )


@dataclass(frozen=True)
class CheckpointState:
    completed: int
    rejections: int
    rng: RNGSnapshot

    def __post_init__(self) -> None:
        completed = _require_int(self.completed, "state.completed", minimum=0)
        rejections = _require_int(self.rejections, "state.rejections", minimum=0)
        if rejections > completed:
            raise CheckpointSchemaError(
                "state.rejections cannot exceed state.completed"
            )
        object.__setattr__(self, "completed", completed)
        object.__setattr__(self, "rejections", rejections)
        if not isinstance(self.rng, RNGSnapshot):
            raise CheckpointSchemaError("state.rng must be an RNGSnapshot")

    def as_dict(self) -> dict[str, object]:
        return {
            "completed": self.completed,
            "rejections": self.rejections,
            "rng": self.rng.as_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> CheckpointState:
        data = _require_object(data, "state")
        _require_exact_keys(data, {"completed", "rejections", "rng"}, "state")
        return cls(
            completed=data["completed"],
            rejections=data["rejections"],
            rng=RNGSnapshot.from_dict(
                _require_object(data["rng"], "state.rng")
            ),
        )


@dataclass(frozen=True)
class Checkpoint:
    experiment: ExperimentSpec
    execution: ExecutionContract
    state: CheckpointState
    fingerprint: ExperimentFingerprint
    schema: str = CHECKPOINT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != CHECKPOINT_SCHEMA:
            raise CheckpointSchemaError(
                f"unsupported checkpoint schema {self.schema!r}"
            )
        if not isinstance(self.experiment, ExperimentSpec):
            raise CheckpointSchemaError("experiment must be an ExperimentSpec")
        if not isinstance(self.execution, ExecutionContract):
            raise CheckpointSchemaError("execution must be an ExecutionContract")
        if not isinstance(self.state, CheckpointState):
            raise CheckpointSchemaError("state must be a CheckpointState")
        if not isinstance(self.fingerprint, ExperimentFingerprint):
            raise CheckpointSchemaError(
                "fingerprint must be an ExperimentFingerprint"
            )

        expected = ExperimentFingerprint.compute(self.experiment, self.execution)
        if self.fingerprint != expected:
            raise CheckpointFingerprintError(
                "checkpoint fingerprint does not match experiment and execution"
            )
        if self.state.completed > self.experiment.simulations:
            raise CheckpointSchemaError(
                "state.completed cannot exceed experiment simulations"
            )
        if self.state.rng.bit_generator != self.execution.bit_generator:
            raise CheckpointSchemaError(
                "checkpoint RNG type does not match execution contract"
            )

    @classmethod
    def create(
        cls,
        *,
        experiment: ExperimentSpec,
        rng: np.random.Generator,
        completed: int,
        rejections: int,
    ) -> Checkpoint:
        execution = ExecutionContract.from_generator(rng)
        state = CheckpointState(
            completed=completed,
            rejections=rejections,
            rng=RNGSnapshot.from_generator(rng),
        )
        fingerprint = ExperimentFingerprint.compute(experiment, execution)
        return cls(
            experiment=experiment,
            execution=execution,
            state=state,
            fingerprint=fingerprint,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "fingerprint": self.fingerprint.as_dict(),
            "experiment": self.experiment.as_dict(),
            "execution": self.execution.as_dict(),
            "state": self.state.as_dict(),
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(
            self.as_dict(),
            ensure_ascii=False,
            sort_keys=True,
            indent=indent,
            allow_nan=False,
        )

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> Checkpoint:
        data = _require_object(data, "checkpoint")
        _require_exact_keys(
            data,
            {"schema", "fingerprint", "experiment", "execution", "state"},
            "checkpoint",
        )
        return cls(
            schema=_require_string(data["schema"], "checkpoint.schema"),
            fingerprint=ExperimentFingerprint.from_dict(
                _require_object(data["fingerprint"], "checkpoint.fingerprint")
            ),
            experiment=ExperimentSpec.from_dict(
                _require_object(data["experiment"], "checkpoint.experiment")
            ),
            execution=ExecutionContract.from_dict(
                _require_object(data["execution"], "checkpoint.execution")
            ),
            state=CheckpointState.from_dict(
                _require_object(data["state"], "checkpoint.state")
            ),
        )

    @classmethod
    def from_json(cls, text: str) -> Checkpoint:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise CheckpointSchemaError("checkpoint is not valid JSON") from exc
        return cls.from_dict(_require_object(data, "checkpoint"))

    @classmethod
    def read(cls, path: str | Path) -> Checkpoint:
        source = Path(path)
        try:
            text = source.read_text(encoding="utf-8")
        except OSError as exc:
            raise CheckpointError(f"failed to read checkpoint {source}") from exc
        return cls.from_json(text)

    def validate_for(
        self,
        experiment: ExperimentSpec,
        execution: ExecutionContract,
    ) -> None:
        expected = ExperimentFingerprint.compute(experiment, execution)
        if self.fingerprint != expected:
            raise CheckpointFingerprintError(
                "checkpoint does not match the requested experiment and execution"
            )

    def write_atomic(self, path: str | Path, *, indent: int = 2) -> Path:
        target = Path(path)
        payload = self.to_json(indent=indent) + "\n"

        temp_path: Path | None = None
        try:
            fd, temp_name = tempfile.mkstemp(
                prefix=f".{target.name}.",
                suffix=".tmp",
                dir=target.parent,
            )
            temp_path = Path(temp_name)
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, target)
        except OSError as exc:
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    pass
            raise CheckpointError(
                f"failed to atomically write checkpoint {target}"
            ) from exc
        return target


__all__ = [
    "CHECKPOINT_SCHEMA",
    "EXPERIMENT_SCHEMA",
    "Checkpoint",
    "CheckpointError",
    "CheckpointFingerprintError",
    "CheckpointSchemaError",
    "CheckpointState",
    "ExecutionContract",
    "ExperimentFingerprint",
    "ExperimentSpec",
    "RNGSnapshot",
    "decode_rng_state",
    "encode_rng_state",
]
