import json

import numpy as np
import pytest

from statfuzz.bootstrap_checkpoint import (
    BOOTSTRAP_COVERAGE_EXPERIMENT_SCHEMA,
    BOOTSTRAP_COVERAGE_OUTER_BIT_GENERATOR,
    BootstrapCoverageCheckpointState,
    BootstrapCoverageExperimentSpec,
)
from statfuzz.checkpoint import (
    CheckpointSchemaError,
    ExecutionContract,
)
from statfuzz.dgp import DGPIdentity, Normal
from statfuzz.methods import BootstrapMeanPercentile
from statfuzz.targets import MeanTarget

ROOT_SEED = 20260930


def _experiment(**overrides):
    kwargs = {
        "dgp": Normal(mean=0.0, sd=1.0),
        "n": 20,
        "simulations": 100,
        "method": BootstrapMeanPercentile(
            resamples=199,
            interval_level=0.9,
        ),
        "tolerance": 0.02,
        "seed": ROOT_SEED,
        "mean_target": None,
        "evidence_confidence_level": 0.95,
        "evidence_interval_method": "wilson",
    }
    kwargs.update(overrides)
    return BootstrapCoverageExperimentSpec.from_coverage_config(**kwargs)


def _execution(seed=ROOT_SEED):
    rng = np.random.Generator(np.random.PCG64(seed))
    return ExecutionContract.from_generator(rng)


def test_bootstrap_coverage_experiment_canonical_round_trip():
    experiment = _experiment()

    restored = BootstrapCoverageExperimentSpec.from_dict(
        experiment.as_dict()
    )

    assert restored == experiment
    assert restored.as_dict() == experiment.as_dict()
    assert restored.canonical_json() == experiment.canonical_json()
    assert experiment.schema == BOOTSTRAP_COVERAGE_EXPERIMENT_SCHEMA


def test_bootstrap_coverage_experiment_identity_contains_required_semantics():
    experiment = _experiment(
        mean_target=MeanTarget(
            mean=0.0,
            note="explicit review note",
        )
    )

    assert experiment.as_dict() == {
        "schema": "statfuzz.bootstrap_coverage.experiment/1",
        "method": {
            "method": "bootstrap_mean_percentile",
            "semantics_version": "1",
            "resamples": 199,
            "interval_level": 0.9,
            "quantile_method": "linear",
        },
        "metric": {
            "name": "coverage",
            "semantics_version": "1",
        },
        "dgp": Normal(mean=0.0, sd=1.0).identity.as_dict(),
        "target": {
            "kind": "mean",
            "source": "population_mean",
            "mean": 0.0,
            "population_mean": 0.0,
            "note": "explicit review note",
        },
        "sample_size": {"n": 20},
        "budget": {"simulations": 100},
        "result_semantics": {
            "tolerance": 0.02,
            "evidence_confidence_level": 0.95,
            "evidence_interval_method": "wilson",
        },
        "rng_origin": {"root_seed": ROOT_SEED},
    }


def test_batch_size_and_progress_callback_are_excluded_from_identity():
    experiment = _experiment()
    canonical = experiment.canonical_json()

    assert "batch_size" not in canonical
    assert "progress_callback" not in canonical

    payload = json.loads(canonical)
    assert set(payload) == {
        "schema",
        "method",
        "metric",
        "dgp",
        "target",
        "sample_size",
        "budget",
        "result_semantics",
        "rng_origin",
    }


def test_bootstrap_coverage_fingerprint_is_deterministic():
    experiment = _experiment()
    execution = _execution()

    first = experiment.fingerprint(execution)
    second = experiment.fingerprint(execution)

    assert first == second
    assert len(first.value) == 64


@pytest.mark.parametrize(
    "changed",
    [
        lambda: _experiment(
            method=BootstrapMeanPercentile(
                resamples=200,
                interval_level=0.9,
            )
        ),
        lambda: _experiment(
            method=BootstrapMeanPercentile(
                resamples=199,
                interval_level=0.8,
            )
        ),
        lambda: _experiment(
            dgp=Normal(mean=0.0, sd=2.0),
        ),
        lambda: _experiment(n=21),
        lambda: _experiment(simulations=101),
        lambda: _experiment(tolerance=0.03),
        lambda: _experiment(evidence_confidence_level=0.9),
        lambda: _experiment(seed=ROOT_SEED + 1),
        lambda: _experiment(
            mean_target=MeanTarget(
                mean=0.0,
                note="reviewed assumption",
            )
        ),
    ],
)
def test_semantic_changes_change_bootstrap_coverage_fingerprint(changed):
    baseline = _experiment().fingerprint(_execution())
    observed = changed().fingerprint(_execution())

    assert observed != baseline


def test_execution_contract_is_part_of_fingerprint():
    experiment = _experiment()
    baseline = experiment.fingerprint(_execution())

    changed_execution = ExecutionContract(
        statfuzz_execution_semantics="1",
        statfuzz_version="different-version",
        numpy_version=np.__version__,
        scipy_version=_execution().scipy_version,
        bit_generator=BOOTSTRAP_COVERAGE_OUTER_BIT_GENERATOR,
    )

    assert experiment.fingerprint(changed_execution) != baseline


def test_fingerprint_rejects_non_pcg64_outer_execution():
    experiment = _experiment()
    execution = ExecutionContract.from_generator(
        np.random.Generator(np.random.MT19937(ROOT_SEED))
    )

    with pytest.raises(CheckpointSchemaError, match="PCG64"):
        experiment.fingerprint(execution)


class _CustomWithoutIdentity:
    name = "custom-without-identity"
    population_mean = 0.0

    def sample(self, rng, n):
        return rng.normal(size=n)


class _CustomWithIdentity:
    name = "custom-with-identity"
    population_mean = 0.0
    identity = DGPIdentity.from_mapping(
        "tests.CustomBootstrapCoverageDGP",
        {"mean": 0.0, "scale": 1.0},
    )

    def sample(self, rng, n):
        return rng.normal(size=n)


def test_persistent_bootstrap_experiment_requires_explicit_dgp_identity():
    with pytest.raises(CheckpointSchemaError, match="stable DGPIdentity"):
        BootstrapCoverageExperimentSpec.from_coverage_config(
            dgp=_CustomWithoutIdentity(),
        )

    experiment = BootstrapCoverageExperimentSpec.from_coverage_config(
        dgp=_CustomWithIdentity(),
    )
    assert experiment.dgp_identity == _CustomWithIdentity.identity


def test_invalid_evidence_method_is_rejected():
    with pytest.raises(CheckpointSchemaError, match="wilson"):
        _experiment(evidence_interval_method="exact")


def test_checkpoint_state_capture_round_trips_exact_outer_rng():
    rng = np.random.Generator(np.random.PCG64(ROOT_SEED))
    rng.normal(size=37)

    state = BootstrapCoverageCheckpointState.capture(
        completed=7,
        covered=5,
        rng=rng,
    )
    restored = BootstrapCoverageCheckpointState.from_dict(state.as_dict())

    assert restored == state
    assert restored.completed == 7
    assert restored.covered == 5
    assert restored.rng.bit_generator == BOOTSTRAP_COVERAGE_OUTER_BIT_GENERATOR

    restored_rng = restored.rng.restore_generator()
    np.testing.assert_array_equal(
        restored_rng.integers(0, 2**31, size=50, dtype=np.int64),
        rng.integers(0, 2**31, size=50, dtype=np.int64),
    )


def test_zero_committed_state_is_valid():
    rng = np.random.Generator(np.random.PCG64(ROOT_SEED))

    state = BootstrapCoverageCheckpointState.capture(
        completed=0,
        covered=0,
        rng=rng,
    )

    assert state.completed == 0
    assert state.covered == 0


def test_checkpoint_state_rejects_covered_above_completed():
    rng = np.random.Generator(np.random.PCG64(ROOT_SEED))

    with pytest.raises(CheckpointSchemaError, match="cannot exceed"):
        BootstrapCoverageCheckpointState.capture(
            completed=4,
            covered=5,
            rng=rng,
        )


def test_checkpoint_state_rejects_non_pcg64_outer_rng():
    rng = np.random.Generator(np.random.MT19937(ROOT_SEED))

    with pytest.raises(CheckpointSchemaError, match="PCG64"):
        BootstrapCoverageCheckpointState.capture(
            completed=4,
            covered=2,
            rng=rng,
        )


def test_checkpoint_state_validates_against_experiment_budget():
    state = BootstrapCoverageCheckpointState.capture(
        completed=101,
        covered=90,
        rng=np.random.Generator(np.random.PCG64(ROOT_SEED)),
    )

    with pytest.raises(
        CheckpointSchemaError,
        match="cannot exceed experiment simulations",
    ):
        state.validate_for(_experiment(simulations=100))


def test_checkpoint_state_round_trip_rejects_unknown_fields():
    state = BootstrapCoverageCheckpointState.capture(
        completed=4,
        covered=2,
        rng=np.random.Generator(np.random.PCG64(ROOT_SEED)),
    )
    payload = state.as_dict()
    payload["batch_size"] = 64

    with pytest.raises(CheckpointSchemaError, match="unexpected keys"):
        BootstrapCoverageCheckpointState.from_dict(payload)


def test_experiment_round_trip_rejects_unknown_execution_controls():
    payload = _experiment().as_dict()
    payload["batch_size"] = 64

    with pytest.raises(CheckpointSchemaError, match="unexpected keys"):
        BootstrapCoverageExperimentSpec.from_dict(payload)
