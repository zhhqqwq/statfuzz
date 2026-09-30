import pytest

from statfuzz import BootstrapCoverageResult, bootstrap_mean_coverage
from statfuzz.dgp import LogNormal, MixtureNormal, Normal, StudentT
from statfuzz.methods import BootstrapMeanPercentile
from statfuzz.search import (
    FamilyPoint,
    FamilyShrinkPlan,
    shrink_dgp_family,
)

ROOT_SEED = 1
SIMULATIONS = 40
N = 8
METHOD = BootstrapMeanPercentile(
    resamples=19,
    interval_level=0.8,
)
TOLERANCE = 0.10

EXPECTED_FAMILY_SEEDS = {
    "normal": 16896585173923650117,
    "lognormal": 9415301123600923416,
    "mixture_normal": 890689660519115215,
}


def _candidate(family, **parameters):
    return FamilyPoint.from_mapping(family, parameters)


def _plan():
    return FamilyShrinkPlan(
        (
            _candidate("normal", mean=0.0, sd=1.0),
            _candidate("lognormal", mean=0.0, sigma=1.0),
            _candidate(
                "student_t",
                df=5.0,
                mean=0.0,
                scale=1.0,
            ),
            _candidate(
                "mixture_normal",
                mean=0.0,
                mean1=0.0,
                mean2=0.0,
                sd1=1.0,
                sd2=5.0,
                weight=0.9,
            ),
        )
    )


def _build_dgp(point):
    if point.family == "normal":
        return Normal(
            mean=point.parameters["mean"],
            sd=point.parameters["sd"],
        )
    if point.family == "lognormal":
        return LogNormal(
            mean=point.parameters["mean"],
            sigma=point.parameters["sigma"],
        )
    if point.family == "student_t":
        return StudentT(
            df=point.parameters["df"],
            mean=point.parameters["mean"],
            scale=point.parameters["scale"],
        )
    if point.family == "mixture_normal":
        return MixtureNormal(
            weight=point.parameters["weight"],
            mean1=point.parameters["mean1"],
            sd1=point.parameters["sd1"],
            mean2=point.parameters["mean2"],
            sd2=point.parameters["sd2"],
            mean=point.parameters["mean"],
        )
    raise AssertionError(f"unexpected family {point.family!r}")


def _evaluate(point, seed, simulations):
    return bootstrap_mean_coverage(
        dgp=_build_dgp(point),
        n=N,
        simulations=simulations,
        method=METHOD,
        tolerance=TOLERANCE,
        seed=seed,
        batch_size=7,
        evidence_confidence_level=0.9,
    )


def test_bootstrap_family_shrink_locks_reference_trace():
    plan = _plan()
    result = shrink_dgp_family(
        start=plan.levels[-1],
        plan=plan,
        evaluate=_evaluate,
        simulations=SIMULATIONS,
        root_seed=ROOT_SEED,
    )

    assert isinstance(result.start_result, BootstrapCoverageResult)
    assert isinstance(result.final_result, BootstrapCoverageResult)

    assert result.start.family == "mixture_normal"
    assert result.start_result.seed == EXPECTED_FAMILY_SEEDS["mixture_normal"]
    assert result.start_result.coverage_count == 24
    assert result.start_result.empirical == 0.6
    assert not result.start_result.passed

    assert [
        (
            step.attempt,
            step.from_family,
            step.to_family,
            step.seed,
            step.result.coverage_count,
            step.result.empirical,
            step.accepted,
        )
        for step in result.steps
    ] == [
        (
            0,
            "mixture_normal",
            "normal",
            EXPECTED_FAMILY_SEEDS["normal"],
            31,
            0.775,
            False,
        ),
        (
            1,
            "mixture_normal",
            "lognormal",
            EXPECTED_FAMILY_SEEDS["lognormal"],
            16,
            0.4,
            True,
        ),
    ]

    assert result.final.family == "lognormal"
    assert result.final_result == result.steps[1].result
    assert result.final_result.seed == EXPECTED_FAMILY_SEEDS["lognormal"]
    assert not result.final_result.passed

    assert result.start_complexity == 3
    assert result.final_complexity == 1
    assert result.complexity_reduction == 2
    assert result.accepted_steps == (result.steps[1],)


def test_bootstrap_family_shrink_accepts_only_preserved_failures():
    plan = _plan()
    result = shrink_dgp_family(
        start=plan.levels[-1],
        plan=plan,
        evaluate=_evaluate,
        simulations=SIMULATIONS,
        root_seed=ROOT_SEED,
    )

    assert result.criterion_name == "outside_tolerance"
    assert all(
        step.accepted == (not step.result.passed)
        for step in result.steps
    )

    normal_step, lognormal_step = result.steps
    assert normal_step.result.passed
    assert not normal_step.accepted
    assert not lognormal_step.result.passed
    assert lognormal_step.accepted


def test_bootstrap_family_shrink_finishes_at_simplest_failing_family():
    plan = _plan()
    result = shrink_dgp_family(
        start=plan.levels[-1],
        plan=plan,
        evaluate=_evaluate,
        simulations=SIMULATIONS,
        root_seed=ROOT_SEED,
    )

    assert result.final == plan.levels[1]

    only_simpler = plan.levels[0]
    simpler_result = _evaluate(
        only_simpler,
        EXPECTED_FAMILY_SEEDS["normal"],
        SIMULATIONS,
    )
    assert simpler_result.passed
    assert simpler_result.coverage_count == 31
    assert plan.complexity(result.final) == 1
    assert plan.complexity(only_simpler) == 0


def test_bootstrap_family_shrink_is_exactly_deterministic():
    plan = _plan()
    kwargs = {
        "start": plan.levels[-1],
        "plan": plan,
        "evaluate": _evaluate,
        "simulations": SIMULATIONS,
        "root_seed": ROOT_SEED,
    }

    first = shrink_dgp_family(**kwargs)
    second = shrink_dgp_family(**kwargs)

    assert first == second
    assert first.to_rows() == second.to_rows()
    assert first.final_result == second.final_result


def test_bootstrap_family_shrink_requires_seed_passthrough():
    plan = _plan()

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
        shrink_dgp_family(
            start=plan.levels[-1],
            plan=plan,
            evaluate=bad_evaluate,
            simulations=SIMULATIONS,
            root_seed=ROOT_SEED,
        )
