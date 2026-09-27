import pytest

from statfuzz import StressTestResult, stress_test
from statfuzz.dgp import LogNormal
from statfuzz.search import ParameterSpace, find_counterexample


def _search_result(point, seed):
    empirical_by_x = {1: 0.06, 2: 0.11, 3: 0.03}
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
    empirical_by_x = {1: 0.052, 2: 0.085, 3: 0.047}
    empirical = empirical_by_x[point["x"]]
    return StressTestResult(
        method="fake",
        metric="type1_error",
        dgp1="fake",
        dgp2="fake",
        n1=10,
        n2=10,
        simulations=5_000,
        seed=seed,
        nominal=0.05,
        empirical=empirical,
        mcse=0.004,
        tolerance=0.01,
    )


def test_find_counterexample_selects_top_search_candidate():
    result = find_counterexample(
        space=ParameterSpace({"x": [1, 2, 3]}),
        search_evaluate=_search_result,
        validation_evaluate=_validation_result,
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
        search_root_seed=11,
        validation_root_seed=99,
    )

    assert len(result.search.records) == 3
    assert [record.point["x"] for record in result.search.records] == [1, 2, 3]
    assert result.as_row()["search_points"] == 3
    assert result.as_row()["objective"] == "absolute_deviation"


def test_find_counterexample_separates_search_and_validation_streams():
    result = find_counterexample(
        space=ParameterSpace({"x": [1, 2, 3]}),
        search_evaluate=_search_result,
        validation_evaluate=_validation_result,
        search_root_seed=11,
        validation_root_seed=99,
    )

    assert result.validation.search_root_seed == 11
    assert result.validation.validation_root_seed == 99
    assert result.validation.search_seed != result.validation.validation_seed
    assert result.search_result.simulations == 100
    assert result.validation_result.simulations == 5_000


def test_find_counterexample_is_reproducible_with_fixed_seeds():
    kwargs = {
        "space": ParameterSpace({"x": [1, 2, 3]}),
        "search_evaluate": _search_result,
        "validation_evaluate": _validation_result,
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
            search_root_seed=42,
            validation_root_seed=42,
        )


def test_find_counterexample_integration_with_lognormal():
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

    result = find_counterexample(
        space=space,
        search_evaluate=search_evaluate,
        validation_evaluate=validation_evaluate,
        search_root_seed=2026,
        validation_root_seed=2027,
    )

    assert len(result.search.records) == 4
    assert result.point in tuple(record.point for record in result.search.records)
    assert result.search_result.simulations == 100
    assert result.validation_result.simulations == 500
    assert result.validation.search_seed != result.validation.validation_seed
