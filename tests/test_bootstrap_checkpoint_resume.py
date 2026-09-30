import json
import os

import numpy as np
import pytest

import statfuzz.bootstrap_coverage as coverage_module
from statfuzz import bootstrap_mean_coverage
from statfuzz.bootstrap_checkpoint import (
    BOOTSTRAP_COVERAGE_CHECKPOINT_SCHEMA,
    BootstrapCoverageCheckpoint,
    BootstrapCoverageExperimentSpec,
)
from statfuzz.bootstrap_coverage import (
    _resume_bootstrap_mean_coverage,
    _simulate_bootstrap_coverage_range,
)
from statfuzz.checkpoint import (
    CheckpointError,
    CheckpointFingerprintError,
    CheckpointSchemaError,
    ExecutionContract,
)
from statfuzz.dgp import LogNormal, MixtureNormal, Normal, StudentT
from statfuzz.methods import BootstrapMeanPercentile
from statfuzz.targets import resolve_mean_target

ROOT_SEED = 20260930
SIMULATIONS = 18
BATCH_SIZE = 4
SPLIT = 8
METHOD = BootstrapMeanPercentile(resamples=7, interval_level=0.8)


class _RecordingDGP:
    def __init__(self, base):
        self.base = base
        self.samples = []

    @property
    def name(self):
        return self.base.name

    @property
    def population_mean(self):
        return self.base.population_mean

    @property
    def identity(self):
        return self.base.identity

    def sample(self, rng, n):
        sample = self.base.sample(rng, n)
        self.samples.append(np.asarray(sample, dtype=float).copy())
        return sample


def _dgp_factories():
    return [
        pytest.param(
            lambda: Normal(mean=0.0, sd=1.0),
            id="normal",
        ),
        pytest.param(
            lambda: LogNormal(sigma=0.7, mean=0.0),
            id="lognormal",
        ),
        pytest.param(
            lambda: StudentT(df=5.0, mean=0.0, scale=1.0),
            id="student-t",
        ),
        pytest.param(
            lambda: MixtureNormal(
                weight=0.8,
                mean1=-1.0,
                sd1=1.0,
                mean2=4.0,
                sd2=2.0,
                mean=0.0,
            ),
            id="mixture-normal",
        ),
    ]


def _experiment(dgp, *, simulations=SIMULATIONS):
    return BootstrapCoverageExperimentSpec.from_coverage_config(
        dgp=dgp,
        n=6,
        simulations=simulations,
        method=METHOD,
        tolerance=0.02,
        seed=ROOT_SEED,
        evidence_confidence_level=0.9,
        evidence_interval_method="wilson",
    )


def _checkpoint():
    dgp = Normal()
    experiment = _experiment(dgp)
    rng = np.random.Generator(np.random.PCG64(ROOT_SEED))
    target = resolve_mean_target(dgp, None)
    covered = _simulate_bootstrap_coverage_range(
        dgp=dgp,
        n=6,
        simulations=SIMULATIONS,
        target_check=target,
        method=METHOD,
        root_seed=ROOT_SEED,
        rng=rng,
        batch_size=BATCH_SIZE,
        start=0,
        stop=SPLIT,
        initial_covered=0,
    )
    return BootstrapCoverageCheckpoint.create(
        experiment=experiment,
        rng=rng,
        completed=SPLIT,
        covered=covered,
    )


def test_bootstrap_coverage_checkpoint_strict_json_round_trip():
    checkpoint = _checkpoint()

    restored = BootstrapCoverageCheckpoint.from_json(checkpoint.to_json())

    assert restored == checkpoint
    assert restored.as_dict() == checkpoint.as_dict()
    assert restored.schema == BOOTSTRAP_COVERAGE_CHECKPOINT_SCHEMA


def test_bootstrap_coverage_checkpoint_rejects_stale_fingerprint():
    payload = _checkpoint().as_dict()
    payload["experiment"]["result_semantics"]["tolerance"] = 0.03

    with pytest.raises(CheckpointFingerprintError, match="fingerprint"):
        BootstrapCoverageCheckpoint.from_dict(payload)


def test_bootstrap_coverage_checkpoint_validate_for_rejects_other_experiment():
    checkpoint = _checkpoint()
    changed = BootstrapCoverageExperimentSpec.from_coverage_config(
        dgp=Normal(),
        n=7,
        simulations=SIMULATIONS,
        method=METHOD,
        tolerance=0.02,
        seed=ROOT_SEED,
        evidence_confidence_level=0.9,
    )
    execution = ExecutionContract.from_generator(
        np.random.Generator(np.random.PCG64(ROOT_SEED))
    )

    with pytest.raises(CheckpointFingerprintError, match="requested"):
        checkpoint.validate_for(changed, execution)


@pytest.mark.parametrize(
    "text",
    [
        "{",
        "[]",
        '{"schema":"broken"}',
    ],
)
def test_bootstrap_coverage_checkpoint_rejects_corrupt_json(text):
    with pytest.raises(CheckpointSchemaError):
        BootstrapCoverageCheckpoint.from_json(text)


def test_bootstrap_coverage_checkpoint_rejects_unknown_top_level_field():
    payload = _checkpoint().as_dict()
    payload["batch_size"] = BATCH_SIZE

    with pytest.raises(CheckpointSchemaError, match="unexpected keys"):
        BootstrapCoverageCheckpoint.from_dict(payload)


def test_bootstrap_coverage_checkpoint_rejects_unsupported_schema():
    payload = _checkpoint().as_dict()
    payload["schema"] = "statfuzz.bootstrap_coverage.checkpoint/999"

    with pytest.raises(CheckpointSchemaError, match="unsupported"):
        BootstrapCoverageCheckpoint.from_dict(payload)


def test_bootstrap_coverage_checkpoint_atomic_write_round_trips(tmp_path):
    checkpoint = _checkpoint()
    path = tmp_path / "coverage.json"

    returned = checkpoint.write_atomic(path)

    assert returned == path
    assert BootstrapCoverageCheckpoint.read(path) == checkpoint
    assert list(tmp_path.glob(".coverage.json.*.tmp")) == []


def test_bootstrap_coverage_checkpoint_atomic_failure_preserves_old_file(
    tmp_path,
    monkeypatch,
):
    checkpoint = _checkpoint()
    path = tmp_path / "coverage.json"
    path.write_text("old coverage checkpoint\n", encoding="utf-8")

    def fail_replace(source, target):
        assert os.path.exists(source)
        assert target == path
        raise OSError("replace failed")

    monkeypatch.setattr(
        "statfuzz.bootstrap_checkpoint.os.replace",
        fail_replace,
    )

    with pytest.raises(CheckpointError, match="atomically write"):
        checkpoint.write_atomic(path)

    assert path.read_text(encoding="utf-8") == "old coverage checkpoint\n"
    assert list(tmp_path.glob(".coverage.json.*.tmp")) == []


def test_checkpoint_create_rejects_completed_beyond_budget():
    experiment = _experiment(Normal())
    rng = np.random.Generator(np.random.PCG64(ROOT_SEED))

    with pytest.raises(
        CheckpointSchemaError,
        match="cannot exceed experiment simulations",
    ):
        BootstrapCoverageCheckpoint.create(
            experiment=experiment,
            rng=rng,
            completed=SIMULATIONS + 1,
            covered=0,
        )


def _run_uninterrupted_trace(monkeypatch, base):
    dgp = _RecordingDGP(base)
    rng = np.random.Generator(np.random.PCG64(ROOT_SEED))
    target = resolve_mean_target(dgp, None)
    events = []
    original_event = coverage_module.bootstrap_mean_coverage_event

    def record_event(sample, **kwargs):
        event = original_event(sample, **kwargs)
        events.append(event)
        return event

    monkeypatch.setattr(
        coverage_module,
        "bootstrap_mean_coverage_event",
        record_event,
    )
    try:
        covered = _simulate_bootstrap_coverage_range(
            dgp=dgp,
            n=6,
            simulations=SIMULATIONS,
            target_check=target,
            method=METHOD,
            root_seed=ROOT_SEED,
            rng=rng,
            batch_size=BATCH_SIZE,
            start=0,
            stop=SIMULATIONS,
            initial_covered=0,
        )
    finally:
        monkeypatch.setattr(
            coverage_module,
            "bootstrap_mean_coverage_event",
            original_event,
        )

    state = json.dumps(rng.bit_generator.state, sort_keys=True)
    next_stream = rng.integers(0, 2**31, size=20, dtype=np.int64)
    return dgp.samples, events, covered, state, next_stream


def _run_resumed_trace(monkeypatch, base, tmp_path):
    dgp = _RecordingDGP(base)
    experiment = _experiment(dgp)
    rng = np.random.Generator(np.random.PCG64(ROOT_SEED))
    target = resolve_mean_target(dgp, None)
    events = []
    original_event = coverage_module.bootstrap_mean_coverage_event

    def record_event(sample, **kwargs):
        event = original_event(sample, **kwargs)
        events.append(event)
        return event

    monkeypatch.setattr(
        coverage_module,
        "bootstrap_mean_coverage_event",
        record_event,
    )
    try:
        covered = _simulate_bootstrap_coverage_range(
            dgp=dgp,
            n=6,
            simulations=SIMULATIONS,
            target_check=target,
            method=METHOD,
            root_seed=ROOT_SEED,
            rng=rng,
            batch_size=BATCH_SIZE,
            start=0,
            stop=SPLIT,
            initial_covered=0,
        )

        checkpoint = BootstrapCoverageCheckpoint.create(
            experiment=experiment,
            rng=rng,
            completed=SPLIT,
            covered=covered,
        )
        path = tmp_path / "resume.json"
        checkpoint.write_atomic(path)
        restored = BootstrapCoverageCheckpoint.read(path)

        result, final_rng = _resume_bootstrap_mean_coverage(
            dgp=dgp,
            checkpoint=restored,
            batch_size=BATCH_SIZE,
        )
    finally:
        monkeypatch.setattr(
            coverage_module,
            "bootstrap_mean_coverage_event",
            original_event,
        )

    state = json.dumps(final_rng.bit_generator.state, sort_keys=True)
    next_stream = final_rng.integers(0, 2**31, size=20, dtype=np.int64)
    return dgp.samples, events, result, state, next_stream


@pytest.mark.parametrize("factory", _dgp_factories())
def test_fixed_batch_resume_is_exactly_equivalent_to_uninterrupted(
    monkeypatch,
    tmp_path,
    factory,
):
    reference = _run_uninterrupted_trace(monkeypatch, factory())
    resumed = _run_resumed_trace(monkeypatch, factory(), tmp_path)

    (
        reference_samples,
        reference_events,
        reference_covered,
        reference_state,
        reference_next,
    ) = reference
    (
        resumed_samples,
        resumed_events,
        resumed_result,
        resumed_state,
        resumed_next,
    ) = resumed

    assert len(resumed_samples) == len(reference_samples) == SIMULATIONS
    for actual, expected in zip(
        resumed_samples,
        reference_samples,
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)

    assert resumed_events == reference_events
    assert resumed_result.coverage_count == reference_covered
    assert resumed_state == reference_state
    np.testing.assert_array_equal(resumed_next, reference_next)

    uninterrupted_result = bootstrap_mean_coverage(
        dgp=factory(),
        n=6,
        simulations=SIMULATIONS,
        method=METHOD,
        tolerance=0.02,
        seed=ROOT_SEED,
        evidence_confidence_level=0.9,
        batch_size=BATCH_SIZE,
    )
    assert resumed_result == uninterrupted_result


@pytest.mark.parametrize("batch_size", [1, 2, 7])
def test_fixed_batch_resume_equivalence_for_multiple_boundaries(
    monkeypatch,
    tmp_path,
    batch_size,
):
    simulations = 17
    split = batch_size * 2

    base = Normal()
    reference_dgp = _RecordingDGP(base)
    reference_rng = np.random.Generator(np.random.PCG64(ROOT_SEED))
    target = resolve_mean_target(reference_dgp, None)
    reference_events = []
    original_event = coverage_module.bootstrap_mean_coverage_event

    def record_reference(sample, **kwargs):
        event = original_event(sample, **kwargs)
        reference_events.append(event)
        return event

    monkeypatch.setattr(
        coverage_module,
        "bootstrap_mean_coverage_event",
        record_reference,
    )
    reference_covered = _simulate_bootstrap_coverage_range(
        dgp=reference_dgp,
        n=6,
        simulations=simulations,
        target_check=target,
        method=METHOD,
        root_seed=ROOT_SEED,
        rng=reference_rng,
        batch_size=batch_size,
        start=0,
        stop=simulations,
        initial_covered=0,
    )
    reference_state = json.dumps(
        reference_rng.bit_generator.state,
        sort_keys=True,
    )

    resumed_dgp = _RecordingDGP(Normal())
    resumed_rng = np.random.Generator(np.random.PCG64(ROOT_SEED))
    resumed_target = resolve_mean_target(resumed_dgp, None)
    resumed_events = []

    def record_resumed(sample, **kwargs):
        event = original_event(sample, **kwargs)
        resumed_events.append(event)
        return event

    monkeypatch.setattr(
        coverage_module,
        "bootstrap_mean_coverage_event",
        record_resumed,
    )
    prefix_covered = _simulate_bootstrap_coverage_range(
        dgp=resumed_dgp,
        n=6,
        simulations=simulations,
        target_check=resumed_target,
        method=METHOD,
        root_seed=ROOT_SEED,
        rng=resumed_rng,
        batch_size=batch_size,
        start=0,
        stop=split,
        initial_covered=0,
    )
    experiment = BootstrapCoverageExperimentSpec.from_coverage_config(
        dgp=resumed_dgp,
        n=6,
        simulations=simulations,
        method=METHOD,
        tolerance=0.02,
        seed=ROOT_SEED,
        evidence_confidence_level=0.9,
    )
    checkpoint = BootstrapCoverageCheckpoint.create(
        experiment=experiment,
        rng=resumed_rng,
        completed=split,
        covered=prefix_covered,
    )
    result, final_rng = _resume_bootstrap_mean_coverage(
        dgp=resumed_dgp,
        checkpoint=checkpoint,
        batch_size=batch_size,
    )

    assert result.coverage_count == reference_covered
    assert resumed_events == reference_events
    assert len(resumed_dgp.samples) == len(reference_dgp.samples)
    for actual, expected in zip(
        resumed_dgp.samples,
        reference_dgp.samples,
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)
    assert json.dumps(final_rng.bit_generator.state, sort_keys=True) == reference_state


class _CountingNormal:
    def __init__(self, sd):
        self.base = Normal(mean=0.0, sd=sd)
        self.calls = 0

    @property
    def name(self):
        return self.base.name

    @property
    def population_mean(self):
        return self.base.population_mean

    @property
    def identity(self):
        return self.base.identity

    def sample(self, rng, n):
        self.calls += 1
        return self.base.sample(rng, n)


def test_resume_fingerprint_mismatch_fails_before_new_outer_draw():
    checkpoint = _checkpoint()
    dgp = _CountingNormal(sd=2.0)

    with pytest.raises(CheckpointFingerprintError):
        _resume_bootstrap_mean_coverage(
            dgp=dgp,
            checkpoint=checkpoint,
            batch_size=BATCH_SIZE,
        )

    assert dgp.calls == 0


def test_completed_checkpoint_resumes_without_new_draw():
    dgp = _RecordingDGP(Normal())
    experiment = _experiment(dgp, simulations=4)
    rng = np.random.Generator(np.random.PCG64(ROOT_SEED))
    target = resolve_mean_target(dgp, None)
    covered = _simulate_bootstrap_coverage_range(
        dgp=dgp,
        n=6,
        simulations=4,
        target_check=target,
        method=METHOD,
        root_seed=ROOT_SEED,
        rng=rng,
        batch_size=2,
        start=0,
        stop=4,
        initial_covered=0,
    )
    checkpoint = BootstrapCoverageCheckpoint.create(
        experiment=experiment,
        rng=rng,
        completed=4,
        covered=covered,
    )

    fresh = _RecordingDGP(Normal())
    result, restored_rng = _resume_bootstrap_mean_coverage(
        dgp=fresh,
        checkpoint=checkpoint,
        batch_size=2,
    )

    assert fresh.samples == []
    assert result.coverage_count == covered
    assert restored_rng.bit_generator.state == rng.bit_generator.state
