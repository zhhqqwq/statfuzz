import pytest

from statfuzz import StressTestResult, stress_test
from statfuzz.dgp import LogNormal
from statfuzz.search import DiscoveryBudget, ParameterSpace, find_counterexample


def _search_result(point, seed, simulations):
    empirical_by_x = {1: 0.06, 2: 0.11, 3: 0.03}
    empirical = empirical_by_x[point["x"]]
    return StressTestResult(
        method="fake",
        metric="type1_error",
        dgp1="fake",
        dgp2="fake",
        n1=10,
        n2=10,
        simulations=simulations,
        seed=seed,
        nominal=0.05,
        empirical=empirical,
        mcse=0.01,
        tolerance=0.01,
    )


def _validation_result(point, seed, simulations):
    empirical_by_x = {1: 0.052, 2: 0.085, 3: 0.047}
    empirical = empirical_by_x[point["x"]]
    return StressTestResult(
        method="fake",
        metric="type1_error",
        dgp1="fake",
        dgp2="fake",
        n1=10,
        n2=10,
        simulations=simulations,
        seed=seed,
        nominal=0.05,
        empirical=empirical,
        mcse=0.004,
        tolerance=0.01,
    )


def _budget():
    return DiscoveryBudget(search_simulations=100, validation_simulations=5_000)


def test_find_counterexample_selects_top_search_candidate():
    result = find_counterexample(
        space=ParameterSpace({"x": [1, 2, 3]}),
        search_evaluate=_search_result,
        validation_evaluate=_validation_result,
        budget=_budget(),
        search_root_seed=11,
        validation_root_seed=99,
    )

    assert result.point["x"] == 2
    assert result.search_result.empirical == 0.11
    assert result.validation_result.empirical == 0.085


def test_find_counterexample_keeps_complete_search_table():
    result = find_counterexample(
        space=ParameterSpace({"x": [1, 2, 3]}),
        search_evaluate=_search_result,
        validation_evaluate=_validation_result,
        budget=_budget(),
        search_root_seed=11,
        validation_root_seed=99,
    )

    row = result.as_row()
    assert len(result.search.records) == 3
    assert [record.point["x"] for record in result.search.records] == [1, 2, 3]
    assert row["search_points"] == 3
    assert row["objective"] == "absolute_deviation"
    assert row["search_budget"] == 100
    assert row["validation_budget"] == 5_000


def test_find_counterexample_executes_stage_specific_budgets():
    calls = []

    def search_evaluate(point, seed, simulations):
        calls.append(("search", point["x"], simulations))
        return _search_result(point, seed, simulations)

    def validation_evaluate(point, seed, simulations):
        calls.append(("validation", point["x"], simulations))
        return _validation_result(point, seed, simulations)

    result = find_counterexample(
        space=ParameterSpace({"x": [1, 2, 3]}),
        search_evaluate=search_evaluate,
        validation_evaluate=validation_evaluate,
        budget=_budget(),
        search_root_seed=11,
        validation_root_seed=99,
    )

    assert [call[2] for call in calls if call[0] == "search"] == [100, 100, 100]
    assert [call[2] for call in calls if call[0] == "validation"] == [5_000]
    assert result.search_result.simulations == 100
    assert result.validation_result.simulations == 5_000


def test_find_counterexample_separates_search_and_validation_streams():
    result = find_counterexample(
        space=ParameterSpace({"x": [1, 2, 3]}),
        search_evaluate=_search_result,
        validation_evaluate=_validation_result,
        budget=_budget(),
        search_root_seed=11,
        validation_root_seed=99,
    )

    assert result.validation.search_root_seed == 11
    assert result.validation.validation_root_seed == 99
    assert result.validation.search_seed != result.validation.validation_seed


def test_find_counterexample_is_reproducible_with_fixed_seeds():
    kwargs = {
        "space": ParameterSpace({"x": [1, 2, 3]}),
        "search_evaluate": _search_result,
        "validation_evaluate": _validation_result,
        "budget": _budget(),
        "search_root_seed": 11,
        "validation_root_seed": 99,
    }

    a = find_counterexample(**kwargs)
    b = find_counterexample(**kwargs)

    assert a.as_row() == b.as_row()
    assert a.search.to_rows() == b.search.to_rows()


def test_find_counterexample_rejects_identical_root_seeds():
    with pytest.raises(ValueError, match="must differ"):
        find_counterexample(
            space=ParameterSpace({"x": [1]}),
            search_evaluate=_search_result,
            validation_evaluate=_validation_result,
            budget=_budget(),
            search_root_seed=42,
            validation_root_seed=42,
        )


def test_find_counterexample_rejects_budget_mismatch():
    def ignores_budget(point, seed, simulations):
        result = _search_result(point, seed, simulations)
        return StressTestResult(
            method=result.method,
            metric=result.metric,
            dgp1=result.dgp1,
            dgp2=result.dgp2,
            n1=result.n1,
            n2=result.n2,
            simulations=999,
            seed=result.seed,
            nominal=result.nominal,
            empirical=result.empirical,
            mcse=result.mcse,
            tolerance=result.tolerance,
        )

    with pytest.raises(ValueError, match="requested budget"):
        find_counterexample(
            space=ParameterSpace({"x": [1]}),
            search_evaluate=ignores_budget,
            validation_evaluate=_validation_result,
            budget=_budget(),
            search_root_seed=11,
            validation_root_seed=99,
        )


def test_find_counterexample_uses_selected_objective():
    result = find_counterexample(
        space=ParameterSpace({"x": [1, 2, 3]}),
        search_evaluate=_search_result,
        validation_evaluate=_validation_result,
        budget=_budget(),
        search_root_seed=11,
        validation_root_seed=99,
        objective="negative_deviation",
    )

    assert result.point["x"] == 3
    assert result.search.objective_name == "negative_deviation"


def test_discovery_budget_requires_positive_counts():
    with pytest.raises(ValueError, match="search_simulations"):
        DiscoveryBudget(search_simulations=0, validation_simulations=100)
    with pytest.raises(ValueError, match="validation_simulations"):
        DiscoveryBudget(search_simulations=100, validation_simulations=0)


def test_find_counterexample_integration_with_lognormal():
    space = ParameterSpace({"n": [8, 12], "sigma": [0.8, 1.2]})
    budget = DiscoveryBudget(search_simulations=100, validation_simulations=500)

    def search_evaluate(point, seed, simulations):
        return stress_test(
            method="welch_ttest",
            metric="type1_error",
            dgp=LogNormal(sigma=point["sigma"]),
            n1=point["n"],
            n2=point["n"],
            simulations=simulations,
            seed=seed,
        )

    def validation_evaluate(point, seed, simulations):
        return stress_test(
            method="welch_ttest",
            metric="type1_error",
            dgp=LogNormal(sigma=point["sigma"]),
            n1=point["n"],
            n2=point["n"],
            simulations=simulations,
            seed=seed,
        )

    result = find_counterexample(
        space=space,
        search_evaluate=search_evaluate,
        validation_evaluate=validation_evaluate,
        budget=budget,
        search_root_seed=2026,
        validation_root_seed=2027,
    )

    assert len(result.search.records) == 4
    assert result.point in tuple(record.point for record in result.search.records)
    assert result.search_result.simulations == 100
    assert result.validation_result.simulations == 500
    assert result.validation.search_seed != result.validation.validation_seed


def test_find_counterexample_can_use_random_search():
    result = find_counterexample(
        space=ParameterSpace({"x": list(range(20))}),
        search_evaluate=lambda point, seed, simulations: StressTestResult(
            method="fake",
            metric="type1_error",
            dgp1="fake",
            dgp2="fake",
            n1=10,
            n2=10,
            simulations=simulations,
            seed=seed,
            nominal=0.05,
            empirical=0.05 + 0.001 * point["x"],
            mcse=0.01,
            tolerance=0.01,
        ),
        validation_evaluate=lambda point, seed, simulations: StressTestResult(
            method="fake",
            metric="type1_error",
            dgp1="fake",
            dgp2="fake",
            n1=10,
            n2=10,
            simulations=simulations,
            seed=seed,
            nominal=0.05,
            empirical=0.05 + 0.001 * point["x"],
            mcse=0.004,
            tolerance=0.01,
        ),
        budget=DiscoveryBudget(search_simulations=100, validation_simulations=500),
        search_root_seed=123,
        validation_root_seed=999,
        search_draws=5,
    )

    assert len(result.search.records) == 5
    assert result.search_result.simulations == 100
    assert result.validation_result.simulations == 500
    assert result.validation.search_seed != result.validation.validation_seed


def test_find_counterexample_random_search_is_reproducible():
    def search_evaluate(point, seed, simulations):
        return StressTestResult(
            method="fake",
            metric="type1_error",
            dgp1="fake",
            dgp2="fake",
            n1=10,
            n2=10,
            simulations=simulations,
            seed=seed,
            nominal=0.05,
            empirical=0.05 + 0.001 * point["x"],
            mcse=0.01,
            tolerance=0.01,
        )

    def validation_evaluate(point, seed, simulations):
        return StressTestResult(
            method="fake",
            metric="type1_error",
            dgp1="fake",
            dgp2="fake",
            n1=10,
            n2=10,
            simulations=simulations,
            seed=seed,
            nominal=0.05,
            empirical=0.05 + 0.001 * point["x"],
            mcse=0.004,
            tolerance=0.01,
        )

    kwargs = {
        "space": ParameterSpace({"x": list(range(30))}),
        "search_evaluate": search_evaluate,
        "validation_evaluate": validation_evaluate,
        "budget": DiscoveryBudget(search_simulations=100, validation_simulations=500),
        "search_root_seed": 11,
        "validation_root_seed": 99,
        "search_draws": 3,
    }

    a = find_counterexample(**kwargs)
    b = find_counterexample(**kwargs)

    assert [record.point for record in a.search.records] == [
        record.point for record in b.search.records
    ]
    assert a.as_row() == b.as_row()
