from statfuzz import StressTestResult, stress_test
from statfuzz.dgp import LogNormal
from statfuzz.search import ParameterSpace, grid_search


def _fake_result(point, seed):
    empirical_by_x = {1: 0.06, 2: 0.09, 3: 0.01}
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


def test_grid_search_returns_every_point():
    space = ParameterSpace({"x": [1, 2, 3]})
    result = grid_search(space=space, evaluate=_fake_result, seed=42)

    assert len(result.records) == 3
    assert [record.point["x"] for record in result.records] == [1, 2, 3]


def test_grid_search_ranking_is_stable_and_documented():
    space = ParameterSpace({"x": [1, 2, 3]})
    result = grid_search(space=space, evaluate=_fake_result, seed=42)

    assert [record.point["x"] for record in result.ranked()] == [3, 2, 1]
    assert result.objective == "absolute_deviation"


def test_same_root_seed_reproduces_per_point_seeds():
    forward = ParameterSpace({"x": [1, 2, 3]})
    reverse = ParameterSpace({"x": [3, 2, 1]})

    a = grid_search(space=forward, evaluate=_fake_result, seed=123)
    b = grid_search(space=reverse, evaluate=_fake_result, seed=123)

    a_seeds = {record.point["x"]: record.seed for record in a.records}
    b_seeds = {record.point["x"]: record.seed for record in b.records}
    assert a_seeds == b_seeds
    assert len(set(a_seeds.values())) == 3


def test_rows_contain_statistical_search_columns():
    result = grid_search(
        space=ParameterSpace({"x": [1]}),
        evaluate=_fake_result,
        seed=7,
    )
    row = result.to_rows()[0]

    assert row["param:x"] == 1
    assert row["nominal"] == 0.05
    assert row["empirical"] == 0.06
    assert row["absolute_deviation"] == 0.01
    assert row["simulations"] == 100
    assert isinstance(row["seed"], int)


def test_integration_with_lognormal_stress_test():
    space = ParameterSpace({"n": [8], "sigma": [0.8, 1.2]})

    def evaluate(point, seed):
        return stress_test(
            method="welch_ttest",
            metric="type1_error",
            dgp=LogNormal(sigma=point["sigma"]),
            n1=point["n"],
            n2=point["n"],
            simulations=100,
            seed=seed,
        )

    result = grid_search(space=space, evaluate=evaluate, seed=2026)

    assert len(result.records) == 2
    assert all(record.result.simulations == 100 for record in result.records)
    assert all(record.result.seed == record.seed for record in result.records)
