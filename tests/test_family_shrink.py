import pytest

from statfuzz import StressTestResult
from statfuzz.dgp import LogNormal, MixtureNormal, Normal, StudentT
from statfuzz.search import (
    FamilyPoint,
    FamilyShrinkPlan,
    ObjectiveThresholdCriterion,
    shrink_dgp_family,
)


def _candidate(family, **parameters):
    return FamilyPoint.from_mapping(family, parameters)


def _plan():
    return FamilyShrinkPlan(
        (
            _candidate("normal", mean=0.0, sd=1.0),
            _candidate("lognormal", mean=0.0, sigma=1.0),
            _candidate("student_t", df=5.0, mean=0.0, scale=1.0),
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


def _fake_evaluate(point, seed, simulations):
    empirical_by_family = {
        "normal": 0.055,
        "lognormal": 0.080,
        "student_t": 0.085,
        "mixture_normal": 0.090,
    }
    return StressTestResult(
        method="fake",
        metric="type1_error",
        dgp1=point.family,
        dgp2=point.family,
        n1=20,
        n2=20,
        simulations=simulations,
        seed=seed,
        nominal=0.05,
        empirical=empirical_by_family[point.family],
        mcse=0.002,
        tolerance=0.01,
    )


def test_family_point_is_serializable_and_family_specific():
    point = _candidate("student_t", df=5.0, mean=0.0, scale=2.0)

    assert point.family == "student_t"
    assert point.parameters["df"] == 5.0
    assert point.parameters["scale"] == 2.0
    assert '"family":"student_t"' in point.to_json()


def test_family_shrink_selects_simplest_candidate_preserving_failure():
    result = shrink_dgp_family(
        start=_plan().levels[-1],
        plan=_plan(),
        evaluate=_fake_evaluate,
        simulations=5_000,
        root_seed=2029,
    )

    assert result.final.family == "lognormal"
    assert not result.final_result.passed
    assert result.changed
    assert result.start_complexity == 3
    assert result.final_complexity == 1
    assert result.complexity_reduction == 2


def test_family_shrink_trace_keeps_rejected_and_accepted_candidates():
    result = shrink_dgp_family(
        start=_plan().levels[-1],
        plan=_plan(),
        evaluate=_fake_evaluate,
        simulations=5_000,
        root_seed=2029,
    )

    assert [step.to_family for step in result.steps] == ["normal", "lognormal"]
    assert [step.accepted for step in result.steps] == [False, True]
    assert result.accepted_steps == (result.steps[1],)
    assert all(step.from_family == "mixture_normal" for step in result.steps)


def test_every_family_candidate_is_re_evaluated():
    calls = []

    def evaluate(point, seed, simulations):
        calls.append((point, seed, simulations))
        return _fake_evaluate(point, seed, simulations)

    result = shrink_dgp_family(
        start=_plan().levels[-1],
        plan=_plan(),
        evaluate=evaluate,
        simulations=5_000,
        root_seed=2029,
    )

    assert len(calls) == 1 + len(result.steps)
    assert calls[0][0] == result.start
    assert [call[0] for call in calls[1:]] == [
        step.candidate for step in result.steps
    ]


def test_family_shrink_is_reproducible():
    kwargs = {
        "start": _plan().levels[-1],
        "plan": _plan(),
        "evaluate": _fake_evaluate,
        "simulations": 5_000,
        "root_seed": 2029,
    }

    a = shrink_dgp_family(**kwargs)
    b = shrink_dgp_family(**kwargs)

    assert a.final == b.final
    assert a.final_result == b.final_result
    assert a.to_rows() == b.to_rows()


def test_family_plan_requires_unique_canonical_family():
    with pytest.raises(ValueError, match="one canonical"):
        FamilyShrinkPlan(
            (
                _candidate("normal", mean=0.0, sd=1.0),
                _candidate("normal", mean=0.0, sd=2.0),
            )
        )


def test_start_must_match_a_canonical_family_level():
    start = _candidate(
        "mixture_normal",
        mean=0.0,
        mean1=0.0,
        mean2=0.0,
        sd1=1.0,
        sd2=8.0,
        weight=0.95,
    )

    with pytest.raises(ValueError, match="exactly match"):
        shrink_dgp_family(
            start=start,
            plan=_plan(),
            evaluate=_fake_evaluate,
            simulations=5_000,
            root_seed=2029,
        )


def test_starting_family_must_preserve_failure():
    def evaluate(point, seed, simulations):
        result = _fake_evaluate(point, seed, simulations)
        return StressTestResult(
            method=result.method,
            metric=result.metric,
            dgp1=result.dgp1,
            dgp2=result.dgp2,
            n1=result.n1,
            n2=result.n2,
            simulations=result.simulations,
            seed=result.seed,
            nominal=result.nominal,
            empirical=0.055,
            mcse=result.mcse,
            tolerance=result.tolerance,
        )

    with pytest.raises(ValueError, match="starting family"):
        shrink_dgp_family(
            start=_plan().levels[-1],
            plan=_plan(),
            evaluate=evaluate,
            simulations=5_000,
            root_seed=2029,
        )


def test_objective_threshold_criterion_works_across_families():
    result = shrink_dgp_family(
        start=_plan().levels[-1],
        plan=_plan(),
        evaluate=_fake_evaluate,
        simulations=5_000,
        root_seed=2029,
        criterion=ObjectiveThresholdCriterion(
            objective="positive_deviation",
            minimum_score=0.025,
        ),
    )

    assert result.final.family == "lognormal"
    assert result.criterion_name == "positive_deviation>=0.025"


def test_family_evaluator_must_respect_seed_and_budget():
    def bad_seed(point, seed, simulations):
        result = _fake_evaluate(point, seed, simulations)
        return StressTestResult(
            method=result.method,
            metric=result.metric,
            dgp1=result.dgp1,
            dgp2=result.dgp2,
            n1=result.n1,
            n2=result.n2,
            simulations=result.simulations,
            seed=seed + 1,
            nominal=result.nominal,
            empirical=result.empirical,
            mcse=result.mcse,
            tolerance=result.tolerance,
        )

    with pytest.raises(ValueError, match="provided seed"):
        shrink_dgp_family(
            start=_plan().levels[-1],
            plan=_plan(),
            evaluate=bad_seed,
            simulations=5_000,
            root_seed=2029,
        )

    def bad_budget(point, seed, simulations):
        result = _fake_evaluate(point, seed, simulations)
        return StressTestResult(
            method=result.method,
            metric=result.metric,
            dgp1=result.dgp1,
            dgp2=result.dgp2,
            n1=result.n1,
            n2=result.n2,
            simulations=simulations + 1,
            seed=seed,
            nominal=result.nominal,
            empirical=result.empirical,
            mcse=result.mcse,
            tolerance=result.tolerance,
        )

    with pytest.raises(ValueError, match="expected"):
        shrink_dgp_family(
            start=_plan().levels[-1],
            plan=_plan(),
            evaluate=bad_budget,
            simulations=5_000,
            root_seed=2029,
        )


def test_builtin_dgp_families_can_be_constructed_from_family_points():
    builders = {
        "normal": lambda p: Normal(
            mean=p.parameters["mean"],
            sd=p.parameters["sd"],
        ),
        "lognormal": lambda p: LogNormal(
            mean=p.parameters["mean"],
            sigma=p.parameters["sigma"],
        ),
        "student_t": lambda p: StudentT(
            df=p.parameters["df"],
            mean=p.parameters["mean"],
            scale=p.parameters["scale"],
        ),
        "mixture_normal": lambda p: MixtureNormal(
            weight=p.parameters["weight"],
            mean1=p.parameters["mean1"],
            sd1=p.parameters["sd1"],
            mean2=p.parameters["mean2"],
            sd2=p.parameters["sd2"],
            mean=p.parameters["mean"],
        ),
    }

    for point in _plan().levels:
        dgp = builders[point.family](point)
        assert dgp.name
