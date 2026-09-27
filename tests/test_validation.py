import pytest

from statfuzz import StressTestResult, stress_test
from statfuzz.dgp import LogNormal
from statfuzz.search import ParameterSpace, grid_search, validate_candidate


def _search_result(point, seed):
    empirical_by_x = {1: 0.06, 2: 0.10, 3: 0.03}
    empirical = empirical_by_x[point["x"]]
    return StressTestResult(
        method="fake",
        metric="type1_error",
        dgp1="fake",
        dgp2="fake",
        n1=10,
        n2=10,
        simulations=100,
        seed=seed,
        nominal=0.05,
        empirical=empirical,
        mcse=0.01,
        tolerance=0.01,
    )


def _validation_result(point, seed):
    empirical_by_x = {1: 0.055, 2: 0.082, 3: 0.045}
    empirical = empirical_by_x[point["x"]]
    return StressTestResult(
        method="fake",
        metric="type1_error",
        dgp1="fake",
        dgp2="fake",
        n1=10,
        n2=10,
        simulations=5000,
        seed=seed,
        nominal=0.05,
        empirical=empirical,
        mcse=0.004,
        tolerance=0.01,
    )


def test_validation_uses_same_candidate_with_new_seed():
    search = grid_search(
        space=ParameterSpace({"x": [1, 2, 3]}),
        evaluate=_search_result,
        seed=11,
    )

    validated = validate_candidate(
        search=search,
        evaluate=_validation_result,
        validation_root_seed=99,
    )

    assert validated.point["x"] == 2
    assert validated.candidate_rank == 0
    assert validated.search_seed != validated.validation_seed
    assert validated.search_root_seed == 11
    assert validated.validation_root_seed == 99


def test_validation_records_both_stages_and_budgets():
    search = grid_search(
        space=ParameterSpace({"x": [1, 2, 3]}),
        evaluate=_search_result,
        seed=11,
    )
    validated = validate_candidate(
        search=search,
        evaluate=_validation_result,
        validation_root_seed=99,
    )
    row = validated.as_row()

    assert row["param:x"] == 2
    assert row["search_simulations"] == 100
    assert row["validation_simulations"] == 5000
    assert row["search_empirical"] == 0.10
    assert row["validation_empirical"] == 0.082
    assert row["search_seed"] != row["validation_seed"]


def test_validation_can_select_non_top_rank():
    search = grid_search(
        space=ParameterSpace({"x": [1, 2, 3]}),
        evaluate=_search_result,
        seed=11,
    )
    validated = validate_candidate(
        search=search,
        evaluate=_validation_result,
        validation_root_seed=99,
        rank=1,
    )

    assert validated.point["x"] == 3


def test_validation_rejects_same_root_seed():
    search = grid_search(
        space=ParameterSpace({"x": [1]}),
        evaluate=_search_result,
        seed=42,
    )

    with pytest.raises(ValueError, match="must differ"):
        validate_candidate(
            search=search,
            evaluate=_validation_result,
            validation_root_seed=42,
        )


def test_validation_requires_seed_passthrough():
    search = grid_search(
        space=ParameterSpace({"x": [1]}),
        evaluate=_search_result,
        seed=42,
    )

    def bad_evaluator(point, seed):
        result = _validation_result(point, seed)
        return StressTestResult(
            method=result.method,
            metric=result.metric,
            dgp1=result.dgp1,
            dgp2=result.dgp2,
            n1=result.n1,
            n2=result.n2,
            simulations=result.simulations,
            seed=123,
            nominal=result.nominal,
            empirical=result.empirical,
            mcse=result.mcse,
            tolerance=result.tolerance,
        )

    with pytest.raises(ValueError, match="pass the provided seed"):
        validate_candidate(
            search=search,
            evaluate=bad_evaluator,
            validation_root_seed=99,
        )


def test_validation_integration_uses_larger_independent_budget():
    space = ParameterSpace({"n": [8, 12], "sigma": [0.8, 1.2]})

    def search_evaluate(point, seed):
        return stress_test(
            method="welch_ttest",
            metric="type1_error",
            dgp=LogNormal(sigma=point["sigma"]),
            n1=point["n"],
            n2=point["n"],
            simulations=100,
            seed=seed,
        )

    def validation_evaluate(point, seed):
        return stress_test(
            method="welch_ttest",
            metric="type1_error",
            dgp=LogNormal(sigma=point["sigma"]),
            n1=point["n"],
            n2=point["n"],
            simulations=500,
            seed=seed,
        )

    search = grid_search(space=space, evaluate=search_evaluate, seed=2026)
    validated = validate_candidate(
        search=search,
        evaluate=validation_evaluate,
        validation_root_seed=2027,
    )

    assert validated.search_result.simulations == 100
    assert validated.validation_result.simulations == 500
    assert validated.search_seed != validated.validation_seed
