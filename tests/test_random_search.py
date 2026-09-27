import pytest

from statfuzz import StressTestResult, stress_test
from statfuzz.dgp import LogNormal
from statfuzz.search import ParameterSpace, random_search, validate_candidate


def _fake_result(point, seed):
    empirical = 0.05 + 0.01 * point["x"]
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
    empirical = 0.05 + 0.008 * point["x"]
    return StressTestResult(
        method="fake",
        metric="type1_error",
        dgp1="fake",
        dgp2="fake",
        n1=10,
        n2=10,
        simulations=500,
        seed=seed,
        nominal=0.05,
        empirical=empirical,
        mcse=0.004,
        tolerance=0.01,
    )


def test_random_search_samples_without_replacement():
    space = ParameterSpace({"x": list(range(20))})
    result = random_search(
        space=space,
        evaluate=_fake_result,
        draws=12,
        seed=42,
    )

    assert len(result.records) == 12
    assert len(set(result.sampled_indices)) == 12
    assert len({record.point["x"] for record in result.records}) == 12


def test_random_search_is_reproducible_with_fixed_seed():
    space = ParameterSpace({"x": list(range(50))})

    a = random_search(space=space, evaluate=_fake_result, draws=10, seed=123)
    b = random_search(space=space, evaluate=_fake_result, draws=10, seed=123)

    assert a.sampled_indices == b.sampled_indices
    assert [record.point for record in a.records] == [record.point for record in b.records]
    assert [record.seed for record in a.records] == [record.seed for record in b.records]


def test_random_search_different_seed_changes_subset():
    space = ParameterSpace({"x": list(range(100))})

    a = random_search(space=space, evaluate=_fake_result, draws=10, seed=1)
    b = random_search(space=space, evaluate=_fake_result, draws=10, seed=2)

    assert a.sampled_indices != b.sampled_indices


def test_random_search_retains_shared_search_interface():
    space = ParameterSpace({"x": list(range(10))})
    result = random_search(
        space=space,
        evaluate=_fake_result,
        draws=5,
        seed=42,
        objective="positive_deviation",
    )

    assert result.objective_name == "positive_deviation"
    assert result.draws == 5
    assert result.space_size == 10
    assert result.coverage_fraction == pytest.approx(0.5)
    assert len(result.to_rows()) == 5
    assert len(result.ranked()) == 5
    assert result.to_markdown()


def test_random_search_respects_objective_ranking():
    space = ParameterSpace({"x": list(range(10))})
    result = random_search(
        space=space,
        evaluate=_fake_result,
        draws=10,
        seed=42,
        objective="positive_deviation",
    )

    assert result.ranked()[0].point["x"] == 9


def test_random_search_all_points_matches_full_coverage():
    space = ParameterSpace({"x": [1, 2, 3, 4]})
    result = random_search(
        space=space,
        evaluate=_fake_result,
        draws=len(space),
        seed=42,
    )

    assert set(result.sampled_indices) == set(range(len(space)))
    assert result.coverage_fraction == 1.0


def test_random_search_rejects_invalid_draw_count():
    space = ParameterSpace({"x": [1, 2, 3]})

    with pytest.raises(ValueError, match="positive"):
        random_search(space=space, evaluate=_fake_result, draws=0, seed=42)
    with pytest.raises(ValueError, match="exceeds"):
        random_search(space=space, evaluate=_fake_result, draws=4, seed=42)


def test_random_search_rejects_negative_seed():
    with pytest.raises(ValueError, match="non-negative"):
        random_search(
            space=ParameterSpace({"x": [1, 2, 3]}),
            evaluate=_fake_result,
            draws=2,
            seed=-1,
        )


def test_validation_accepts_random_search_result():
    search = random_search(
        space=ParameterSpace({"x": [1, 2, 3]}),
        evaluate=_fake_result,
        draws=3,
        seed=11,
        objective="positive_deviation",
    )

    validated = validate_candidate(
        search=search,
        evaluate=_validation_result,
        validation_root_seed=99,
    )

    assert validated.point["x"] == 3
    assert validated.search_seed != validated.validation_seed


def test_random_search_integration_with_lognormal():
    space = ParameterSpace(
        {
            "n": [8, 12, 20],
            "sigma": [0.6, 0.8, 1.0, 1.2, 1.4],
        }
    )

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

    result = random_search(
        space=space,
        evaluate=evaluate,
        draws=5,
        seed=2026,
    )

    assert result.draws == 5
    assert result.space_size == 15
    assert all(record.result.simulations == 100 for record in result.records)
    assert all(record.result.seed == record.seed for record in result.records)
