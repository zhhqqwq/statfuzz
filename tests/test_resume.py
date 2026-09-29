# ruff: noqa: I001
import numpy as np
import pytest

from statfuzz import simulation, stress_test
from statfuzz.checkpoint import (
    Checkpoint,
    CheckpointFingerprintError,
    ExperimentSpec,
    RNGSnapshot,
)
from statfuzz.dgp import (
    DGPIdentity,
    LogNormal,
    MixtureNormal,
    Normal,
    StudentT,
)


SEED = 20260930
TOTAL_SIMULATIONS = 137


def _make_checkpoint(dgp, *, batch_size, completed):
    experiment = ExperimentSpec.from_stress_test_config(
        method="welch_ttest",
        metric="type1_error",
        dgp=dgp,
        n1=11,
        n2=17,
        simulations=TOTAL_SIMULATIONS,
        alpha=0.05,
        tolerance=0.01,
        seed=SEED,
        confidence_level=0.95,
        interval_method="wilson",
    )
    rng = np.random.default_rng(SEED)
    rejections = simulation._simulate_rejections(
        dgp=dgp,
        other=dgp,
        n1=experiment.n1,
        n2=experiment.n2,
        simulations=completed,
        alpha=experiment.alpha,
        rng=rng,
        batch_size=batch_size,
    )
    checkpoint = Checkpoint.create(
        experiment=experiment,
        rng=rng,
        completed=completed,
        rejections=rejections,
    )
    return Checkpoint.from_json(checkpoint.to_json())


DGP_FACTORIES = (
    pytest.param(lambda: Normal(), id="normal"),
    pytest.param(lambda: LogNormal(sigma=1.0), id="lognormal"),
    pytest.param(lambda: StudentT(df=5.0), id="student_t"),
    pytest.param(lambda: MixtureNormal(), id="mixture_normal"),
)


@pytest.mark.parametrize("dgp_factory", DGP_FACTORIES)
@pytest.mark.parametrize("batch_size", [1, 2, 7, 64])
def test_fixed_batch_resume_matches_uninterrupted_result_and_rng(
    dgp_factory,
    batch_size,
):
    resumed_dgp = dgp_factory()
    checkpoint = _make_checkpoint(
        resumed_dgp,
        batch_size=batch_size,
        completed=batch_size,
    )

    resumed_result, resumed_rng = simulation._resume_stress_test(
        checkpoint=checkpoint,
        dgp=resumed_dgp,
        batch_size=batch_size,
    )

    reference_dgp = dgp_factory()
    reference_result = stress_test(
        method="welch_ttest",
        metric="type1_error",
        dgp=reference_dgp,
        n1=11,
        n2=17,
        simulations=TOTAL_SIMULATIONS,
        alpha=0.05,
        tolerance=0.01,
        seed=SEED,
        confidence_level=0.95,
        interval_method="wilson",
        batch_size=batch_size,
    )

    reference_rng = np.random.default_rng(SEED)
    reference_rejections = simulation._simulate_rejections(
        dgp=dgp_factory(),
        other=dgp_factory(),
        n1=11,
        n2=17,
        simulations=TOTAL_SIMULATIONS,
        alpha=0.05,
        rng=reference_rng,
        batch_size=batch_size,
    )

    assert resumed_result == reference_result
    assert resumed_result.rejection_count == reference_rejections
    assert RNGSnapshot.from_generator(resumed_rng).state == (
        RNGSnapshot.from_generator(reference_rng).state
    )


class _RecordingNormal:
    name = "resume-recording-normal"
    population_mean = 0.0
    identity = DGPIdentity.from_mapping(
        "tests.ResumeRecordingNormal",
        {"mean": 0.0, "sd": 1.0},
    )

    def __init__(self):
        self.draws = []

    def sample(self, rng, n):
        sample = rng.normal(size=n)
        self.draws.append(sample.copy())
        return sample


def test_fixed_batch_resume_preserves_complete_logical_sample_sequence():
    batch_size = 7
    completed = 21
    resumed_dgp = _RecordingNormal()
    checkpoint = _make_checkpoint(
        resumed_dgp,
        batch_size=batch_size,
        completed=completed,
    )

    resumed_result, _ = simulation._resume_stress_test(
        checkpoint=checkpoint,
        dgp=resumed_dgp,
        batch_size=batch_size,
    )

    reference_dgp = _RecordingNormal()
    reference_result = stress_test(
        method="welch_ttest",
        metric="type1_error",
        dgp=reference_dgp,
        n1=11,
        n2=17,
        simulations=TOTAL_SIMULATIONS,
        alpha=0.05,
        tolerance=0.01,
        seed=SEED,
        confidence_level=0.95,
        interval_method="wilson",
        batch_size=batch_size,
    )

    assert resumed_result == reference_result
    assert len(resumed_dgp.draws) == len(reference_dgp.draws) == 2 * TOTAL_SIMULATIONS
    for actual, expected in zip(resumed_dgp.draws, reference_dgp.draws):
        np.testing.assert_array_equal(actual, expected)


def test_resumed_progress_uses_global_completed_and_rejection_counts():
    batch_size = 7
    completed = 21
    dgp = Normal()
    checkpoint = _make_checkpoint(
        dgp,
        batch_size=batch_size,
        completed=completed,
    )
    events = []

    result, _ = simulation._resume_stress_test(
        checkpoint=checkpoint,
        dgp=dgp,
        batch_size=batch_size,
        progress_callback=events.append,
    )

    assert events[0].completed == completed + batch_size
    assert events[-1].completed == TOTAL_SIMULATIONS
    assert events[-1].rejections == result.rejection_count
    assert events[-1].empirical == result.empirical


class _DifferentRecordingNormal(_RecordingNormal):
    identity = DGPIdentity.from_mapping(
        "tests.DifferentResumeRecordingNormal",
        {"mean": 0.0, "sd": 1.0},
    )


def test_resume_rejects_different_experiment_before_new_sample_draw():
    checkpoint_dgp = _RecordingNormal()
    checkpoint = _make_checkpoint(
        checkpoint_dgp,
        batch_size=7,
        completed=21,
    )
    incompatible_dgp = _DifferentRecordingNormal()

    with pytest.raises(CheckpointFingerprintError, match="does not match"):
        simulation._resume_stress_test(
            checkpoint=checkpoint,
            dgp=incompatible_dgp,
            batch_size=7,
        )

    assert incompatible_dgp.draws == []


@pytest.mark.parametrize(
    ("start_index", "initial_rejections", "message"),
    [
        (-1, 0, "start_index"),
        (4, 5, "initial_rejections"),
    ],
)
def test_resumable_executor_rejects_invalid_committed_state(
    start_index,
    initial_rejections,
    message,
):
    rng = np.random.default_rng(SEED)
    dgp = Normal()

    with pytest.raises(ValueError, match=message):
        simulation._simulate_rejections(
            dgp=dgp,
            other=dgp,
            n1=5,
            n2=5,
            simulations=10,
            alpha=0.05,
            rng=rng,
            batch_size=2,
            start_index=start_index,
            initial_rejections=initial_rejections,
        )


CROSS_BATCH_CASES = (
    pytest.param(1, 2, 17, id="1-to-2"),
    pytest.param(1, 7, 17, id="1-to-7"),
    pytest.param(1, 64, 17, id="1-to-64"),
    pytest.param(1, 1_000, 17, id="1-to-over-remaining"),
    pytest.param(2, 1, 18, id="2-to-1"),
    pytest.param(2, 7, 18, id="2-to-7"),
    pytest.param(2, 64, 18, id="2-to-64"),
    pytest.param(2, 1_000, 18, id="2-to-over-remaining"),
    pytest.param(7, 1, 21, id="7-to-1"),
    pytest.param(7, 2, 21, id="7-to-2"),
    pytest.param(7, 64, 21, id="7-to-64"),
    pytest.param(7, 1_000, 21, id="7-to-over-remaining"),
    pytest.param(64, 1, 64, id="64-to-1"),
    pytest.param(64, 2, 64, id="64-to-2"),
    pytest.param(64, 7, 64, id="64-to-7"),
    pytest.param(64, 1_000, 64, id="64-to-over-remaining"),
)


class _RecordingDGP:
    def __init__(self, delegate):
        self.delegate = delegate
        self.draws = []

    @property
    def name(self):
        return self.delegate.name

    @property
    def identity(self):
        return self.delegate.identity

    @property
    def population_mean(self):
        return self.delegate.population_mean

    def sample(self, rng, n):
        sample = self.delegate.sample(rng, n)
        self.draws.append(np.asarray(sample).copy())
        return sample


@pytest.mark.parametrize("dgp_factory", DGP_FACTORIES)
@pytest.mark.parametrize(
    ("pre_batch_size", "post_batch_size", "completed"),
    CROSS_BATCH_CASES,
)
def test_cross_batch_resume_is_exactly_equivalent_to_uninterrupted_run(
    dgp_factory,
    pre_batch_size,
    post_batch_size,
    completed,
):
    resumed_dgp = _RecordingDGP(dgp_factory())
    checkpoint = _make_checkpoint(
        resumed_dgp,
        batch_size=pre_batch_size,
        completed=completed,
    )

    assert completed % pre_batch_size == 0
    assert len(resumed_dgp.draws) == 2 * completed

    resumed_result, resumed_rng = simulation._resume_stress_test(
        checkpoint=checkpoint,
        dgp=resumed_dgp,
        batch_size=post_batch_size,
    )

    reference_dgp = _RecordingDGP(dgp_factory())
    reference_result = stress_test(
        method="welch_ttest",
        metric="type1_error",
        dgp=reference_dgp,
        n1=11,
        n2=17,
        simulations=TOTAL_SIMULATIONS,
        alpha=0.05,
        tolerance=0.01,
        seed=SEED,
        confidence_level=0.95,
        interval_method="wilson",
        batch_size=1,
    )

    reference_rng = np.random.default_rng(SEED)
    rng_reference_dgp = dgp_factory()
    reference_rejections = simulation._simulate_rejections(
        dgp=rng_reference_dgp,
        other=rng_reference_dgp,
        n1=11,
        n2=17,
        simulations=TOTAL_SIMULATIONS,
        alpha=0.05,
        rng=reference_rng,
        batch_size=1,
    )

    assert pre_batch_size != post_batch_size
    assert resumed_result == reference_result
    assert resumed_result.rejection_count == reference_rejections
    assert RNGSnapshot.from_generator(resumed_rng).state == (
        RNGSnapshot.from_generator(reference_rng).state
    )

    assert len(resumed_dgp.draws) == len(reference_dgp.draws)
    assert len(reference_dgp.draws) == 2 * TOTAL_SIMULATIONS
    for actual, expected in zip(resumed_dgp.draws, reference_dgp.draws):
        np.testing.assert_array_equal(actual, expected)
