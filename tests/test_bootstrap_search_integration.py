import pytest

from statfuzz import BootstrapCoverageResult, bootstrap_mean_coverage
from statfuzz.dgp import LogNormal
from statfuzz.methods import BootstrapMeanPercentile
from statfuzz.search import (
    DiscoveryBudget,
    ParameterSpace,
    find_counterexample,
    grid_search,
    random_search,
    validate_candidate,
)

SEARCH_ROOT_SEED = 2026
VALIDATION_ROOT_SEED = 2027
METHOD = BootstrapMeanPercentile(resamples=19, interval_level=0.8)
SPACE = ParameterSpace(
    {
        "n": [8, 12],
        "sigma": [0.6, 1.0],
    }
)

SEARCH_POINT_SEEDS = {
    (8, 0.6): 18288191874956655351,
    (8, 1.0): 3446584563003053428,
    (12, 0.6): 8248403353295714311,
    (12, 1.0): 1398032075849990232,
}
VALIDATION_POINT_SEEDS = {
    (8, 0.6): 12650735208034868968,
    (8, 1.0): 7914576595547002326,
    (12, 0.6): 65652475395910325,
    (12, 1.0): 518234009642673046,
}


def _bootstrap_result(point, seed, simulations):
    return bootstrap_mean_coverage(
        dgp=LogNormal(
            sigma=point["sigma"],
            mean=0.0,
        ),
        n=point["n"],
        simulations=simulations,
        method=METHOD,
        tolerance=0.05,
        seed=seed,
        batch_size=7,
        evidence_confidence_level=0.9,
    )


def _point_key(point):
    return point["n"], point["sigma"]


def test_grid_search_accepts_real_bootstrap_coverage_results():
    first = grid_search(
        space=SPACE,
        evaluate=lambda point, seed: _bootstrap_result(
            point,
            seed,
            simulations=40,
        ),
        seed=SEARCH_ROOT_SEED,
    )
    second = grid_search(
        space=SPACE,
        evaluate=lambda point, seed: _bootstrap_result(
            point,
            seed,
            simulations=40,
        ),
        seed=SEARCH_ROOT_SEED,
    )

    assert len(first.records) == len(SPACE) == 4
    assert all(
        isinstance(record.result, BootstrapCoverageResult)
        for record in first.records
    )
    assert all(record.result.metric == "coverage" for record in first.records)

    observed_seeds = {
        _point_key(record.point): record.seed
        for record in first.records
    }
    assert observed_seeds == SEARCH_POINT_SEEDS

    assert first.records == second.records
    assert [record.point for record in first.ranked()] == [
        record.point for record in second.ranked()
    ]

    scores = [
        record.objective_score(first.objective)
        for record in first.ranked()
    ]
    assert scores == sorted(scores, reverse=True)


def test_random_search_keeps_deterministic_subset_and_bootstrap_point_seeds():
    first = random_search(
        space=SPACE,
        evaluate=lambda point, seed: _bootstrap_result(
            point,
            seed,
            simulations=40,
        ),
        draws=3,
        seed=SEARCH_ROOT_SEED,
    )
    second = random_search(
        space=SPACE,
        evaluate=lambda point, seed: _bootstrap_result(
            point,
            seed,
            simulations=40,
        ),
        draws=3,
        seed=SEARCH_ROOT_SEED,
    )

    assert first.sampled_indices == (2, 3, 1)
    assert first.sampled_indices == second.sampled_indices
    assert first.records == second.records
    assert [_point_key(record.point) for record in first.records] == [
        (12, 0.6),
        (12, 1.0),
        (8, 1.0),
    ]
    assert [record.seed for record in first.records] == [
        SEARCH_POINT_SEEDS[(12, 0.6)],
        SEARCH_POINT_SEEDS[(12, 1.0)],
        SEARCH_POINT_SEEDS[(8, 1.0)],
    ]
    assert all(
        isinstance(record.result, BootstrapCoverageResult)
        for record in first.records
    )


def test_validation_uses_independent_seed_for_bootstrap_candidate():
    search = grid_search(
        space=SPACE,
        evaluate=lambda point, seed: _bootstrap_result(
            point,
            seed,
            simulations=40,
        ),
        seed=SEARCH_ROOT_SEED,
    )

    validated = validate_candidate(
        search=search,
        evaluate=lambda point, seed: _bootstrap_result(
            point,
            seed,
            simulations=80,
        ),
        validation_root_seed=VALIDATION_ROOT_SEED,
    )

    key = _point_key(validated.point)
    assert validated.point == search.ranked()[0].point
    assert validated.search_seed == SEARCH_POINT_SEEDS[key]
    assert validated.validation_seed == VALIDATION_POINT_SEEDS[key]
    assert validated.search_seed != validated.validation_seed
    assert isinstance(validated.search_result, BootstrapCoverageResult)
    assert isinstance(validated.validation_result, BootstrapCoverageResult)
    assert validated.search_result.simulations == 40
    assert validated.validation_result.simulations == 80


def test_discovery_accepts_bootstrap_and_preserves_stage_budget_and_seed_separation():
    budget = DiscoveryBudget(
        search_simulations=30,
        validation_simulations=90,
    )
    calls = []

    def search_evaluate(point, seed, simulations):
        calls.append(("search", _point_key(point), seed, simulations))
        return _bootstrap_result(point, seed, simulations)

    def validation_evaluate(point, seed, simulations):
        calls.append(("validation", _point_key(point), seed, simulations))
        return _bootstrap_result(point, seed, simulations)

    first = find_counterexample(
        space=SPACE,
        search_evaluate=search_evaluate,
        validation_evaluate=validation_evaluate,
        budget=budget,
        search_root_seed=SEARCH_ROOT_SEED,
        validation_root_seed=VALIDATION_ROOT_SEED,
    )

    assert isinstance(first.search_result, BootstrapCoverageResult)
    assert isinstance(first.validation_result, BootstrapCoverageResult)
    assert first.search_result.simulations == 30
    assert first.validation_result.simulations == 90

    key = _point_key(first.point)
    assert first.validation.search_seed == SEARCH_POINT_SEEDS[key]
    assert first.validation.validation_seed == VALIDATION_POINT_SEEDS[key]
    assert first.validation.search_seed != first.validation.validation_seed

    search_calls = [call for call in calls if call[0] == "search"]
    validation_calls = [call for call in calls if call[0] == "validation"]
    assert len(search_calls) == len(SPACE)
    assert all(call[3] == 30 for call in search_calls)
    assert len(validation_calls) == 1
    assert validation_calls[0][3] == 90


def test_discovery_bootstrap_ranking_is_reproducible():
    kwargs = {
        "space": SPACE,
        "search_evaluate": _bootstrap_result,
        "validation_evaluate": _bootstrap_result,
        "budget": DiscoveryBudget(
            search_simulations=30,
            validation_simulations=90,
        ),
        "search_root_seed": SEARCH_ROOT_SEED,
        "validation_root_seed": VALIDATION_ROOT_SEED,
    }

    first = find_counterexample(**kwargs)
    second = find_counterexample(**kwargs)

    assert first.search.records == second.search.records
    assert first.search.ranked() == second.search.ranked()
    assert first.validation == second.validation
    assert first.as_row() == second.as_row()


def test_bootstrap_search_rejects_seed_not_passed_through():
    def bad_evaluate(point, seed):
        return _bootstrap_result(
            point,
            SEARCH_ROOT_SEED,
            simulations=20,
        )

    with pytest.raises(
        ValueError,
        match="statistical evaluation",
    ):
        grid_search(
            space=ParameterSpace({"n": [8], "sigma": [0.6]}),
            evaluate=bad_evaluate,
            seed=SEARCH_ROOT_SEED,
        )
