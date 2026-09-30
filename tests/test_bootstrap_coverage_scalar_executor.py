import numpy as np
import pytest

import statfuzz.bootstrap_coverage as coverage_module
from statfuzz import bootstrap_mean_coverage
from statfuzz.bootstrap_coverage import (
    BootstrapCoverageEvent,
    _simulate_bootstrap_coverage_scalar,
)
from statfuzz.dgp import LogNormal, MixtureNormal, Normal, StudentT
from statfuzz.methods import BootstrapMeanPercentile
from statfuzz.targets import MeanTargetCheck

ROOT_SEED = 20260930
METHOD = BootstrapMeanPercentile(resamples=7, interval_level=0.8)

EXPECTED_SAMPLES = [
    [
        0.6366033914118914,
        -1.2255299187110698,
        0.23287523257300355,
        -0.9013322370602812,
    ],
    [
        -0.5863798887358305,
        0.2779833004633182,
        0.8054137799398718,
        1.38654675170384,
    ],
    [
        -0.8022088102670425,
        0.8094827651759504,
        -0.016934338961166008,
        -1.127470310436621,
    ],
    [
        0.344077623021908,
        -0.5460253502294526,
        0.3783164441826578,
        -0.42099603620843984,
    ],
    [
        0.7429542377117021,
        -0.7981693821273931,
        -0.04603958133657777,
        -0.45898528280269096,
    ],
]

EXPECTED_EVENTS = [
    BootstrapCoverageEvent(
        logical_outer_index=0,
        target_mean=0.0,
        interval_low=-0.731249558229736,
        interval_high=-0.08775973605042892,
        covered=False,
    ),
    BootstrapCoverageEvent(
        logical_outer_index=1,
        target_mean=0.0,
        interval_low=0.4345125902561697,
        interval_high=0.9797514105138373,
        covered=False,
    ),
    BootstrapCoverageEvent(
        logical_outer_index=2,
        target_mean=0.0,
        interval_low=-0.6445583769047952,
        interval_high=0.4362980934203532,
        covered=True,
    ),
    BootstrapCoverageEvent(
        logical_outer_index=3,
        target_mean=0.0,
        interval_low=-0.10097386360377233,
        interval_high=0.159656972446471,
        covered=True,
    ),
    BootstrapCoverageEvent(
        logical_outer_index=4,
        target_mean=0.0,
        interval_low=-0.24698282933101467,
        interval_high=-0.033137174946465175,
        covered=False,
    ),
]

EXPECTED_NEXT_OUTER_STREAM = [
    1556937507,
    1667086065,
    161528815,
    907865985,
    1079487114,
    1158766560,
    1387601418,
    1032041977,
    350394214,
    1799368966,
]


class _RecordingNormal:
    def __init__(self):
        self._base = Normal(mean=0.0, sd=1.0)
        self.samples = []

    @property
    def name(self):
        return self._base.name

    @property
    def population_mean(self):
        return self._base.population_mean

    @property
    def identity(self):
        return self._base.identity

    def sample(self, rng, n):
        sample = self._base.sample(rng, n)
        self.samples.append(sample.copy())
        return sample


def _target_check():
    return MeanTargetCheck(
        source="population_mean",
        mean=0.0,
        population_mean=0.0,
    )


def test_scalar_outer_executor_fixed_reference_trace(monkeypatch):
    dgp = _RecordingNormal()
    outer_rng = np.random.Generator(np.random.PCG64(ROOT_SEED))
    recorded_events = []
    original_event = coverage_module.bootstrap_mean_coverage_event

    def record_event(sample, **kwargs):
        event = original_event(sample, **kwargs)
        recorded_events.append(event)
        return event

    monkeypatch.setattr(
        coverage_module,
        "bootstrap_mean_coverage_event",
        record_event,
    )

    coverage_count = _simulate_bootstrap_coverage_scalar(
        dgp=dgp,
        n=4,
        simulations=5,
        target_check=_target_check(),
        method=METHOD,
        root_seed=ROOT_SEED,
        rng=outer_rng,
    )

    assert coverage_count == 2
    assert recorded_events == EXPECTED_EVENTS
    assert len(dgp.samples) == len(EXPECTED_SAMPLES)
    for actual, expected in zip(dgp.samples, EXPECTED_SAMPLES, strict=True):
        np.testing.assert_array_equal(actual, np.asarray(expected))

    reference_rng = np.random.Generator(np.random.PCG64(ROOT_SEED))
    for expected in EXPECTED_SAMPLES:
        np.testing.assert_array_equal(
            reference_rng.normal(0.0, 1.0, size=4),
            np.asarray(expected),
        )

    assert outer_rng.bit_generator.state == reference_rng.bit_generator.state

    actual_next = outer_rng.integers(
        0,
        2**31,
        size=10,
        dtype=np.int64,
    )
    reference_next = reference_rng.integers(
        0,
        2**31,
        size=10,
        dtype=np.int64,
    )
    np.testing.assert_array_equal(actual_next, reference_next)
    assert actual_next.tolist() == EXPECTED_NEXT_OUTER_STREAM


def test_bootstrap_resample_count_does_not_change_outer_sample_stream():
    left_dgp = _RecordingNormal()
    right_dgp = _RecordingNormal()
    left_rng = np.random.Generator(np.random.PCG64(ROOT_SEED))
    right_rng = np.random.Generator(np.random.PCG64(ROOT_SEED))

    _simulate_bootstrap_coverage_scalar(
        dgp=left_dgp,
        n=4,
        simulations=5,
        target_check=_target_check(),
        method=BootstrapMeanPercentile(
            resamples=7,
            interval_level=0.8,
        ),
        root_seed=ROOT_SEED,
        rng=left_rng,
    )
    _simulate_bootstrap_coverage_scalar(
        dgp=right_dgp,
        n=4,
        simulations=5,
        target_check=_target_check(),
        method=BootstrapMeanPercentile(
            resamples=19,
            interval_level=0.8,
        ),
        root_seed=ROOT_SEED,
        rng=right_rng,
    )

    assert len(left_dgp.samples) == len(right_dgp.samples) == 5
    for left, right in zip(left_dgp.samples, right_dgp.samples, strict=True):
        np.testing.assert_array_equal(left, right)
    assert left_rng.bit_generator.state == right_rng.bit_generator.state


def test_public_scalar_coverage_executor_builds_expected_result():
    result = bootstrap_mean_coverage(
        dgp=Normal(mean=0.0, sd=1.0),
        n=4,
        simulations=5,
        method=METHOD,
        tolerance=0.01,
        seed=ROOT_SEED,
        evidence_confidence_level=0.9,
    )

    assert result.coverage_count == 2
    assert result.simulations == 5
    assert result.empirical == 0.4
    assert result.nominal == 0.8
    assert result.deviation == pytest.approx(-0.4)
    assert result.status == "OUTSIDE_TOLERANCE"
    assert result.seed == ROOT_SEED
    assert result.n == 4
    assert result.method_config == METHOD
    assert result.target_check == _target_check()
    assert result.dgp_identity == Normal().identity
    assert result.evidence_confidence_level == 0.9
    assert result.evidence_interval_method == "wilson"


@pytest.mark.parametrize(
    "dgp",
    [
        Normal(mean=0.0, sd=1.0),
        LogNormal(sigma=0.7, mean=0.0),
        StudentT(df=5.0, mean=0.0, scale=1.0),
        MixtureNormal(
            weight=0.8,
            mean1=-1.0,
            sd1=1.0,
            mean2=4.0,
            sd2=2.0,
            mean=0.0,
        ),
    ],
)
def test_scalar_coverage_executor_supports_all_builtin_dgps(dgp):
    result = bootstrap_mean_coverage(
        dgp=dgp,
        n=6,
        simulations=4,
        method=BootstrapMeanPercentile(
            resamples=9,
            interval_level=0.8,
        ),
        seed=123,
    )

    assert result.dgp == dgp.name
    assert result.dgp_identity == dgp.identity
    assert result.target_check.source == "population_mean"
    assert result.target_check.mean == dgp.population_mean
    assert result.coverage_count in range(5)


class _CountingUnknownMeanDGP:
    name = "counting-unknown-mean"

    def __init__(self):
        self.calls = 0

    def sample(self, rng, n):
        self.calls += 1
        return rng.normal(size=n)


def test_unknown_mean_fails_before_any_outer_draw():
    dgp = _CountingUnknownMeanDGP()

    with pytest.raises(ValueError, match="Provide MeanTarget"):
        bootstrap_mean_coverage(
            dgp=dgp,
            n=4,
            simulations=3,
            method=BootstrapMeanPercentile(resamples=5),
            seed=1,
        )

    assert dgp.calls == 0


class _BadShapeDGP:
    name = "bad-shape"
    population_mean = 0.0

    @property
    def identity(self):
        return Normal().identity

    def __init__(self):
        self.calls = 0

    def sample(self, rng, n):
        index = self.calls
        self.calls += 1
        if index == 2:
            return np.zeros(n + 1)
        return np.zeros(n)


def test_outer_sample_failure_reports_logical_index():
    with pytest.raises(RuntimeError, match="simulation 2: outer sample must have shape"):
        bootstrap_mean_coverage(
            dgp=_BadShapeDGP(),
            n=4,
            simulations=5,
            method=BootstrapMeanPercentile(resamples=5),
            seed=1,
        )


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("n", 0, ValueError),
        ("simulations", 0, ValueError),
        ("seed", -1, ValueError),
        ("seed", None, TypeError),
        ("tolerance", float("nan"), ValueError),
        ("evidence_confidence_level", 1.0, ValueError),
        ("evidence_interval_method", "exact", ValueError),
    ],
)
def test_public_executor_rejects_invalid_config_before_sampling(
    field,
    value,
    error,
):
    dgp = _RecordingNormal()
    kwargs = {
        "dgp": dgp,
        "n": 4,
        "simulations": 3,
        "method": BootstrapMeanPercentile(resamples=5),
        "tolerance": 0.01,
        "seed": 1,
        "evidence_confidence_level": 0.95,
        "evidence_interval_method": "wilson",
    }
    kwargs[field] = value

    with pytest.raises(error):
        bootstrap_mean_coverage(**kwargs)

    assert dgp.samples == []
