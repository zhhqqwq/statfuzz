import pytest

from statfuzz import StressTestResult
from statfuzz.search import ParameterPoint
from statfuzz.search.shrink import (
    ObjectiveThresholdCriterion,
    ShrinkDimension,
    ShrinkPlan,
    shrink_counterexample,
)


def _point(**values):
    return ParameterPoint(tuple(sorted(values.items())))


def _evaluate(point, seed, simulations):
    n = point["n"]
    sigma = point["sigma"]
    failure = n >= 8 and sigma >= 1.0
    empirical = 0.08 if failure else 0.055
    return StressTestResult(
        method="fake",
        metric="type1_error",
        dgp1="fake",
        dgp2="fake",
        n1=n,
        n2=n,
        simulations=simulations,
        seed=seed,
        nominal=0.05,
        empirical=empirical,
        mcse=0.002,
        tolerance=0.01,
    )


def _plan():
    return ShrinkPlan(
        (
            ShrinkDimension("n", (4, 8, 12, 20)),
            ShrinkDimension("sigma", (0.4, 0.8, 1.0, 1.4)),
        )
    )


def test_shrink_counterexample_simplifies_and_preserves_failure():
    result = shrink_counterexample(
        point=_point(n=20, sigma=1.4),
        plan=_plan(),
        evaluate=_evaluate,
        simulations=1_000,
        root_seed=2026,
    )

    assert result.final_point == _point(n=8, sigma=1.0)
    assert not result.final_result.passed
    assert result.changed
    assert result.start_complexity == 6
    assert result.final_complexity == 3
    assert result.complexity_reduction == 3
    assert [step.parameter for step in result.accepted_steps] == ["n", "sigma"]


def test_shrink_trace_records_accepted_and_rejected_attempts():
    result = shrink_counterexample(
        point=_point(n=20, sigma=1.4),
        plan=_plan(),
        evaluate=_evaluate,
        simulations=1_000,
        root_seed=2026,
    )

    rows = result.to_rows()
    assert rows
    assert any(row["accepted"] for row in rows)
    assert any(not row["accepted"] for row in rows)
    assert all(row["simulations"] == 1_000 for row in rows)
    assert all(row["criterion"] == "outside_tolerance" for row in rows)


def test_every_shrink_attempt_is_re_evaluated():
    calls = []

    def evaluate(point, seed, simulations):
        calls.append((point, seed, simulations))
        return _evaluate(point, seed, simulations)

    result = shrink_counterexample(
        point=_point(n=20, sigma=1.4),
        plan=_plan(),
        evaluate=evaluate,
        simulations=1_000,
        root_seed=2026,
    )

    assert len(calls) == 1 + len(result.steps)
    assert calls[0][0] == result.start_point
    assert [call[0] for call in calls[1:]] == [step.point for step in result.steps]


def test_shrinking_is_reproducible_with_fixed_seed():
    kwargs = {
        "point": _point(n=20, sigma=1.4),
        "plan": _plan(),
        "evaluate": _evaluate,
        "simulations": 1_000,
        "root_seed": 2026,
    }

    a = shrink_counterexample(**kwargs)
    b = shrink_counterexample(**kwargs)

    assert a.final_point == b.final_point
    assert a.to_rows() == b.to_rows()
    assert a.final_result == b.final_result


def test_start_must_satisfy_failure_criterion():
    with pytest.raises(ValueError, match="starting point"):
        shrink_counterexample(
            point=_point(n=4, sigma=0.4),
            plan=_plan(),
            evaluate=_evaluate,
            simulations=1_000,
            root_seed=2026,
        )


def test_objective_threshold_criterion_can_define_failure():
    criterion = ObjectiveThresholdCriterion(
        objective="positive_deviation",
        minimum_score=0.02,
    )
    result = shrink_counterexample(
        point=_point(n=20, sigma=1.4),
        plan=_plan(),
        evaluate=_evaluate,
        simulations=1_000,
        root_seed=2026,
        criterion=criterion,
    )

    assert result.final_point == _point(n=8, sigma=1.0)
    assert result.criterion_name == "positive_deviation>=0.02"


def test_plan_mapping_preserves_explicit_dimension_order():
    plan = ShrinkPlan.from_mapping(
        {
            "sigma": [0.4, 0.8, 1.0, 1.4],
            "n": [4, 8, 12, 20],
        }
    )

    assert [dimension.name for dimension in plan.dimensions] == ["sigma", "n"]


def test_plan_requires_start_values_to_exist_in_levels():
    plan = ShrinkPlan(
        (
            ShrinkDimension("n", (4, 8, 12)),
            ShrinkDimension("sigma", (0.4, 0.8, 1.0, 1.4)),
        )
    )

    with pytest.raises(ValueError, match="not present"):
        shrink_counterexample(
            point=_point(n=20, sigma=1.4),
            plan=plan,
            evaluate=_evaluate,
            simulations=1_000,
            root_seed=2026,
        )


def test_shrink_evaluator_must_respect_seed_and_budget():
    def bad_seed(point, seed, simulations):
        result = _evaluate(point, seed, simulations)
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
        shrink_counterexample(
            point=_point(n=20, sigma=1.4),
            plan=_plan(),
            evaluate=bad_seed,
            simulations=1_000,
            root_seed=2026,
        )

    def bad_budget(point, seed, simulations):
        result = _evaluate(point, seed, simulations)
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
        shrink_counterexample(
            point=_point(n=20, sigma=1.4),
            plan=_plan(),
            evaluate=bad_budget,
            simulations=1_000,
            root_seed=2026,
        )


def test_accepted_steps_strictly_reduce_plan_complexity():
    result = shrink_counterexample(
        point=_point(n=20, sigma=1.4),
        plan=_plan(),
        evaluate=_evaluate,
        simulations=1_000,
        root_seed=2026,
    )

    previous = result.start_complexity
    for step in result.accepted_steps:
        current = result.plan.complexity(step.point)
        assert current < previous
        previous = current
