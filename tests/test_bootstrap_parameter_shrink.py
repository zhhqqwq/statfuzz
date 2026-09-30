import pytest

from statfuzz import BootstrapCoverageResult, bootstrap_mean_coverage
from statfuzz.dgp import LogNormal
from statfuzz.methods import BootstrapMeanPercentile
from statfuzz.search import (
    ParameterPoint,
    ShrinkDimension,
    ShrinkPlan,
    shrink_counterexample,
)

ROOT_SEED = 23
SIMULATIONS = 40
METHOD = BootstrapMeanPercentile(
    resamples=19,
    interval_level=0.8,
)
TOLERANCE = 0.10

EXPECTED_POINT_SEEDS = {
    (20, 1.4): 4814316633320268937,
    (4, 1.4): 2123792986122574088,
    (4, 0.1): 501745526428469941,
    (4, 0.4): 11317074504781442213,
}


def _point(*, n, sigma):
    return ParameterPoint(
        (
            ("n", n),
            ("sigma", sigma),
        )
    )


def _plan():
    return ShrinkPlan(
        (
            ShrinkDimension("n", (4, 8, 12, 20)),
            ShrinkDimension("sigma", (0.1, 0.4, 0.8, 1.4)),
        )
    )


def _evaluate(point, seed, simulations):
    return bootstrap_mean_coverage(
        dgp=LogNormal(
            sigma=point["sigma"],
            mean=0.0,
        ),
        n=point["n"],
        simulations=simulations,
        method=METHOD,
        tolerance=TOLERANCE,
        seed=seed,
        batch_size=7,
        evidence_confidence_level=0.9,
    )


def test_bootstrap_parameter_shrink_locks_reference_trace():
    result = shrink_counterexample(
        point=_point(n=20, sigma=1.4),
        plan=_plan(),
        evaluate=_evaluate,
        simulations=SIMULATIONS,
        root_seed=ROOT_SEED,
    )

    assert isinstance(result.start_result, BootstrapCoverageResult)
    assert isinstance(result.final_result, BootstrapCoverageResult)
    assert result.start_result.coverage_count == 23
    assert result.start_result.empirical == 0.575
    assert result.start_result.seed == EXPECTED_POINT_SEEDS[(20, 1.4)]
    assert not result.start_result.passed

    assert [
        (
            step.attempt,
            step.pass_index,
            step.parameter,
            step.from_value,
            step.to_value,
            step.point["n"],
            step.point["sigma"],
            step.seed,
            step.result.coverage_count,
            step.result.empirical,
            step.accepted,
        )
        for step in result.steps
    ] == [
        (
            0,
            0,
            "n",
            20,
            4,
            4,
            1.4,
            EXPECTED_POINT_SEEDS[(4, 1.4)],
            19,
            0.475,
            True,
        ),
        (
            1,
            0,
            "sigma",
            1.4,
            0.1,
            4,
            0.1,
            EXPECTED_POINT_SEEDS[(4, 0.1)],
            32,
            0.8,
            False,
        ),
        (
            2,
            0,
            "sigma",
            1.4,
            0.4,
            4,
            0.4,
            EXPECTED_POINT_SEEDS[(4, 0.4)],
            15,
            0.375,
            True,
        ),
        (
            3,
            1,
            "sigma",
            0.4,
            0.1,
            4,
            0.1,
            EXPECTED_POINT_SEEDS[(4, 0.1)],
            32,
            0.8,
            False,
        ),
    ]

    assert result.final_point == _point(n=4, sigma=0.4)
    assert result.final_result == result.steps[2].result
    assert result.final_result.seed == EXPECTED_POINT_SEEDS[(4, 0.4)]
    assert not result.final_result.passed
    assert result.start_complexity == 6
    assert result.final_complexity == 1
    assert result.complexity_reduction == 5
    assert [step.parameter for step in result.accepted_steps] == [
        "n",
        "sigma",
    ]


def test_bootstrap_parameter_shrink_accepts_only_preserved_failures():
    result = shrink_counterexample(
        point=_point(n=20, sigma=1.4),
        plan=_plan(),
        evaluate=_evaluate,
        simulations=SIMULATIONS,
        root_seed=ROOT_SEED,
    )

    assert all(
        step.accepted == (not step.result.passed)
        for step in result.steps
    )
    assert result.criterion_name == "outside_tolerance"

    rejected = [step for step in result.steps if not step.accepted]
    assert len(rejected) == 2
    assert all(step.point == _point(n=4, sigma=0.1) for step in rejected)
    assert all(step.result.passed for step in rejected)
    assert rejected[0].seed == rejected[1].seed
    assert rejected[0].result == rejected[1].result


def test_bootstrap_parameter_shrink_finishes_at_plan_local_minimum():
    result = shrink_counterexample(
        point=_point(n=20, sigma=1.4),
        plan=_plan(),
        evaluate=_evaluate,
        simulations=SIMULATIONS,
        root_seed=ROOT_SEED,
    )

    assert result.final_point == _point(n=4, sigma=0.4)

    simpler_sigma = _evaluate(
        _point(n=4, sigma=0.1),
        EXPECTED_POINT_SEEDS[(4, 0.1)],
        SIMULATIONS,
    )
    assert simpler_sigma.passed
    assert simpler_sigma.empirical == METHOD.interval_level

    n_dimension, sigma_dimension = result.plan.dimensions
    assert n_dimension.index_of(result.final_point["n"]) == 0
    assert sigma_dimension.index_of(result.final_point["sigma"]) == 1


def test_bootstrap_parameter_shrink_is_exactly_deterministic():
    kwargs = {
        "point": _point(n=20, sigma=1.4),
        "plan": _plan(),
        "evaluate": _evaluate,
        "simulations": SIMULATIONS,
        "root_seed": ROOT_SEED,
    }

    first = shrink_counterexample(**kwargs)
    second = shrink_counterexample(**kwargs)

    assert first == second
    assert first.to_rows() == second.to_rows()
    assert first.final_result == second.final_result


def test_bootstrap_shrink_requires_seed_passthrough():
    def bad_evaluate(point, seed, simulations):
        return _evaluate(
            point,
            ROOT_SEED,
            simulations,
        )

    with pytest.raises(
        ValueError,
        match="statistical evaluation",
    ):
        shrink_counterexample(
            point=_point(n=20, sigma=1.4),
            plan=_plan(),
            evaluate=bad_evaluate,
            simulations=SIMULATIONS,
            root_seed=ROOT_SEED,
        )
