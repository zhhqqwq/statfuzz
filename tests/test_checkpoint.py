# ruff: noqa: I001
import copy
import json
import os

import numpy as np
import pytest

from statfuzz import MeanEqualityNull
from statfuzz.checkpoint import (
    CHECKPOINT_SCHEMA,
    Checkpoint,
    CheckpointError,
    CheckpointFingerprintError,
    CheckpointSchemaError,
    CheckpointState,
    ExecutionContract,
    ExperimentFingerprint,
    ExperimentSpec,
    RNGSnapshot,
    decode_rng_state,
    encode_rng_state,
)
from statfuzz.dgp import DGPIdentity, Normal


def _experiment(**overrides):
    kwargs = {
        "method": "welch_ttest",
        "metric": "type1_error",
        "dgp": Normal(),
        "n1": 11,
        "n2": 13,
        "simulations": 100,
        "alpha": 0.05,
        "tolerance": 0.01,
        "seed": 20260930,
        "confidence_level": 0.95,
        "interval_method": "wilson",
    }
    kwargs.update(overrides)
    return ExperimentSpec.from_stress_test_config(**kwargs)


def _checkpoint():
    experiment = _experiment()
    rng = np.random.default_rng(experiment.root_seed)
    rng.normal(size=19)
    return Checkpoint.create(
        experiment=experiment,
        rng=rng,
        completed=8,
        rejections=2,
    )


def _assert_nested_equal(actual, expected):
    if isinstance(expected, np.ndarray):
        assert isinstance(actual, np.ndarray)
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        np.testing.assert_array_equal(actual, expected)
        return
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert set(actual) == set(expected)
        for key in expected:
            _assert_nested_equal(actual[key], expected[key])
        return
    if isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_nested_equal(actual_item, expected_item)
        return
    assert actual == expected


def test_experiment_spec_is_canonical_and_round_trips():
    experiment = _experiment()
    restored = ExperimentSpec.from_dict(experiment.as_dict())

    assert restored == experiment
    assert restored.canonical_json() == experiment.canonical_json()
    assert restored.as_dict()["dgp1"] == Normal().identity.as_dict()
    assert "batch_size" not in restored.canonical_json()
    assert "progress_callback" not in restored.canonical_json()


def test_experiment_fingerprint_is_deterministic():
    experiment = _experiment()
    rng = np.random.default_rng(experiment.root_seed)
    execution = ExecutionContract.from_generator(rng)

    first = ExperimentFingerprint.compute(experiment, execution)
    second = ExperimentFingerprint.compute(
        ExperimentSpec.from_dict(experiment.as_dict()),
        ExecutionContract.from_dict(execution.as_dict()),
    )

    assert first == second
    assert len(first.value) == 64


def test_null_note_participates_in_fingerprint():
    first_experiment = _experiment(
        null=MeanEqualityNull(mean=0.0, note="first declaration")
    )
    second_experiment = _experiment(
        null=MeanEqualityNull(mean=0.0, note="second declaration")
    )
    rng = np.random.default_rng(first_experiment.root_seed)
    execution = ExecutionContract.from_generator(rng)

    assert ExperimentFingerprint.compute(
        first_experiment,
        execution,
    ) != ExperimentFingerprint.compute(second_experiment, execution)


def test_persistent_checkpoint_rejects_seed_none():
    with pytest.raises(CheckpointSchemaError, match="seed=None"):
        _experiment(seed=None)


class _NoIdentityDGP:
    name = "no-identity"
    population_mean = 0.0

    def sample(self, rng, n):
        return rng.normal(size=n)


class _StableIdentityDGP(_NoIdentityDGP):
    identity = DGPIdentity.from_mapping(
        "tests.StableIdentityDGP",
        {"mean": 0.0, "scale": 1.0},
    )


def test_persistent_checkpoint_requires_explicit_custom_dgp_identity():
    with pytest.raises(CheckpointSchemaError, match="stable DGPIdentity"):
        _experiment(dgp=_NoIdentityDGP())

    experiment = _experiment(dgp=_StableIdentityDGP())
    assert experiment.dgp1_identity == _StableIdentityDGP.identity


def test_rng_state_encoding_round_trips_numpy_scalars_arrays_and_containers():
    state = {
        "large": np.uint64(2**63 + 123),
        "array": np.arange(16, dtype=np.uint32).reshape(4, 4),
        "tuple": (np.int64(7), [np.float64(0.25), True]),
    }

    encoded = encode_rng_state(state)
    json.dumps(encoded, allow_nan=False)
    decoded = decode_rng_state(encoded)

    _assert_nested_equal(decoded, {
        "large": int(2**63 + 123),
        "array": state["array"],
        "tuple": (7, [0.25, True]),
    })


@pytest.mark.parametrize(
    "bit_generator_factory",
    [
        lambda: np.random.PCG64(1234),
        lambda: np.random.MT19937(1234),
    ],
    ids=["pcg64", "mt19937"],
)
def test_rng_snapshot_restores_exact_next_random_stream(bit_generator_factory):
    bit_generator = bit_generator_factory()
    rng = np.random.Generator(bit_generator)
    rng.normal(size=37)

    snapshot = RNGSnapshot.from_generator(rng)
    restored_bit_generator = type(bit_generator)()
    restored_bit_generator.state = snapshot.raw_state()
    restored = np.random.Generator(restored_bit_generator)

    np.testing.assert_array_equal(
        restored.integers(0, 2**31, size=100, dtype=np.int64),
        rng.integers(0, 2**31, size=100, dtype=np.int64),
    )


def test_checkpoint_json_round_trip_preserves_identity_and_state():
    checkpoint = _checkpoint()

    restored = Checkpoint.from_json(checkpoint.to_json())

    assert restored.experiment == checkpoint.experiment
    assert restored.execution == checkpoint.execution
    assert restored.fingerprint == checkpoint.fingerprint
    assert restored.state.completed == checkpoint.state.completed
    assert restored.state.rejections == checkpoint.state.rejections
    _assert_nested_equal(
        restored.state.rng.raw_state(),
        checkpoint.state.rng.raw_state(),
    )


def test_checkpoint_rejects_tampered_experiment_with_old_fingerprint():
    checkpoint = _checkpoint()
    data = checkpoint.as_dict()
    data["experiment"]["decision"]["alpha"] = 0.01

    with pytest.raises(CheckpointFingerprintError, match="fingerprint"):
        Checkpoint.from_dict(data)


def test_checkpoint_validate_for_rejects_different_result_semantics():
    checkpoint = _checkpoint()
    different = _experiment(tolerance=0.02)

    with pytest.raises(CheckpointFingerprintError, match="does not match"):
        checkpoint.validate_for(different, checkpoint.execution)


def test_checkpoint_rejects_state_beyond_total_budget():
    checkpoint = _checkpoint()

    with pytest.raises(CheckpointSchemaError, match="cannot exceed"):
        Checkpoint(
            experiment=checkpoint.experiment,
            execution=checkpoint.execution,
            state=CheckpointState(
                completed=101,
                rejections=2,
                rng=checkpoint.state.rng,
            ),
            fingerprint=checkpoint.fingerprint,
        )


def test_checkpoint_rejects_rng_type_mismatch():
    checkpoint = _checkpoint()
    mismatched_rng = RNGSnapshot(
        bit_generator="numpy.random._mt19937.MT19937",
        state=encode_rng_state(
            {
                "bit_generator": "MT19937",
                "state": {
                    "key": np.zeros(624, dtype=np.uint32),
                    "pos": 0,
                },
            }
        ),
    )

    with pytest.raises(CheckpointSchemaError, match="execution contract"):
        Checkpoint(
            experiment=checkpoint.experiment,
            execution=checkpoint.execution,
            state=CheckpointState(
                completed=checkpoint.state.completed,
                rejections=checkpoint.state.rejections,
                rng=mismatched_rng,
            ),
            fingerprint=checkpoint.fingerprint,
        )


def test_checkpoint_read_rejects_corrupt_json(tmp_path):
    path = tmp_path / "checkpoint.json"
    path.write_text('{"schema":', encoding="utf-8")

    with pytest.raises(CheckpointSchemaError, match="valid JSON"):
        Checkpoint.read(path)


def test_checkpoint_rejects_unsupported_schema():
    data = _checkpoint().as_dict()
    data["schema"] = "statfuzz.checkpoint/999"

    with pytest.raises(CheckpointSchemaError, match="unsupported checkpoint schema"):
        Checkpoint.from_dict(data)


def test_checkpoint_rejects_unknown_schema_fields():
    data = _checkpoint().as_dict()
    data["unexpected"] = True

    with pytest.raises(CheckpointSchemaError, match="unexpected keys"):
        Checkpoint.from_dict(data)


def test_atomic_write_round_trips_and_leaves_no_temp_file(tmp_path):
    checkpoint = _checkpoint()
    path = tmp_path / "checkpoint.json"

    returned = checkpoint.write_atomic(path)

    assert returned == path
    assert Checkpoint.read(path).as_dict() == checkpoint.as_dict()
    assert list(tmp_path.glob(".checkpoint.json.*.tmp")) == []


def test_atomic_write_failure_preserves_old_file_and_cleans_temp(
    tmp_path,
    monkeypatch,
):
    checkpoint = _checkpoint()
    path = tmp_path / "checkpoint.json"
    path.write_text("old checkpoint\n", encoding="utf-8")

    def fail_replace(source, target):
        assert os.path.exists(source)
        assert target == path
        raise OSError("replace failed")

    monkeypatch.setattr("statfuzz.checkpoint.os.replace", fail_replace)

    with pytest.raises(CheckpointError, match="atomically write"):
        checkpoint.write_atomic(path)

    assert path.read_text(encoding="utf-8") == "old checkpoint\n"
    assert list(tmp_path.glob(".checkpoint.json.*.tmp")) == []


def test_checkpoint_schema_constant_matches_serialized_document():
    checkpoint = _checkpoint()
    assert checkpoint.as_dict()["schema"] == CHECKPOINT_SCHEMA
