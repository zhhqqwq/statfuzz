import pytest

from statfuzz import StressTestResult
from statfuzz.search import (
    AbsoluteDeviationObjective,
    ParameterPoint,
    ParameterSpace,
    SearchRecord,
    SearchResult,
    grid_search,
    random_search,
    summarize_search_multiplicity,
    summarize_selection_effect,
    validate_candidate,
)


def _result(point, seed):
    x = int(point["x"])
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
        empirical=0.05 + 0.01 * x,
        mcse=0.005,
        tolerance=0.015,
    )


def _validation(point, seed):
    x = int(point["x"])
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
        empirical=0.05 + 0.0075 * x,
        mcse=0.002,
        tolerance=0.015,
    )


def test_multiplicity_summary_quantifies_unique_top_candidate():
    search = grid_search(
        space=ParameterSpace({"x": [1, 2, 3, 4]}),
        evaluate=_result,
        seed=42,
    )

    summary = summarize_search_multiplicity(search)

    assert summary.evaluated_points == 4
    assert summary.selection_opportunities == 4
    assert summary.candidate_rank == 0
    assert summary.candidate_rank_one_based == 1
    assert summary.candidate_score == pytest.approx(0.04)
    assert summary.tie_count == 1
    assert summary.empirical_percentile == 1.0
    assert summary.empirical_upper_tail_fraction == pytest.approx(0.25)
    assert summary.objective_min == pytest.approx(0.01)
    assert summary.objective_median == pytest.approx(0.025)
    assert summary.objective_p90 == pytest.approx(0.037)
    assert summary.objective_p95 == pytest.approx(0.0385)
    assert summary.objective_max == pytest.approx(0.04)
    assert summary.candidate_minus_median == pytest.approx(0.015)
    assert summary.candidate_minus_p95 == pytest.approx(0.0015)
    assert summary.outside_tolerance_count == 3
    assert summary.outside_tolerance_fraction == pytest.approx(0.75)


def test_multiplicity_summary_is_tie_aware():
    objective = AbsoluteDeviationObjective()
    records = tuple(
        SearchRecord(
            point=ParameterPoint((("x", x),)),
            result=StressTestResult(
                method="fake",
                metric="type1_error",
                dgp1="fake",
                dgp2="fake",
                n1=10,
                n2=10,
                simulations=100,
                seed=x,
                nominal=0.05,
                empirical=empirical,
                mcse=0.005,
                tolerance=0.01,
            ),
            seed=x,
        )
        for x, empirical in ((1, 0.07), (2, 0.07), (3, 0.06))
    )
    search = SearchResult(
        records=records,
        objective=objective,
        root_seed=42,
        parameter_names=("x",),
    )

    top = summarize_search_multiplicity(search, candidate_rank=0)
    tied_second = summarize_search_multiplicity(search, candidate_rank=1)

    assert top.tie_count == 2
    assert top.empirical_percentile == 1.0
    assert top.empirical_upper_tail_fraction == pytest.approx(2 / 3)
    assert tied_second.candidate_score == top.candidate_score
    assert tied_second.tie_count == 2
    assert tied_second.empirical_upper_tail_fraction == pytest.approx(2 / 3)


def test_candidate_rank_can_summarize_non_top_candidate():
    search = grid_search(
        space=ParameterSpace({"x": [1, 2, 3, 4]}),
        evaluate=_result,
        seed=42,
    )

    summary = summarize_search_multiplicity(search, candidate_rank=2)

    assert summary.candidate_rank_one_based == 3
    assert summary.candidate_score == pytest.approx(0.02)
    assert summary.empirical_percentile == pytest.approx(0.5)
    assert summary.empirical_upper_tail_fraction == pytest.approx(0.75)


def test_random_search_multiplicity_counts_evaluated_draws_not_full_space():
    search = random_search(
        space=ParameterSpace({"x": list(range(1, 11))}),
        evaluate=_result,
        draws=4,
        seed=42,
    )

    summary = summarize_search_multiplicity(search)

    assert summary.evaluated_points == 4
    assert summary.selection_opportunities == 4
    assert search.space_size == 10
    assert search.coverage_fraction == pytest.approx(0.4)
    assert summary.empirical_upper_tail_fraction == pytest.approx(0.25)


def test_multiplicity_rejects_empty_search_and_invalid_rank():
    empty = SearchResult(
        records=(),
        objective=AbsoluteDeviationObjective(),
        root_seed=42,
        parameter_names=("x",),
    )

    with pytest.raises(ValueError, match="empty search"):
        summarize_search_multiplicity(empty)

    search = grid_search(
        space=ParameterSpace({"x": [1, 2]}),
        evaluate=_result,
        seed=42,
    )
    with pytest.raises(ValueError, match="non-negative"):
        summarize_search_multiplicity(search, candidate_rank=-1)
    with pytest.raises(IndexError, match="outside"):
        summarize_search_multiplicity(search, candidate_rank=2)


def test_selection_effect_diagnostic_compares_search_and_validation_objective():
    search = grid_search(
        space=ParameterSpace({"x": [1, 2, 3, 4]}),
        evaluate=_result,
        seed=42,
    )
    validation = validate_candidate(
        search=search,
        evaluate=_validation,
        validation_root_seed=2026,
    )

    diagnostic = summarize_selection_effect(search, validation)

    assert diagnostic.evaluated_points == 4
    assert diagnostic.candidate_rank == 0
    assert diagnostic.search_score == pytest.approx(0.04)
    assert diagnostic.validation_score == pytest.approx(0.03)
    assert diagnostic.search_minus_validation_gap == pytest.approx(0.01)
    assert diagnostic.validation_minus_search_change == pytest.approx(-0.01)
    assert diagnostic.search_mcse == 0.005
    assert diagnostic.validation_mcse == 0.002
    assert diagnostic.search_simulations == 100
    assert diagnostic.validation_simulations == 500
    assert diagnostic.multiplicity.empirical_upper_tail_fraction == pytest.approx(0.25)


def test_selection_effect_rejects_mismatched_validation_object():
    search_a = grid_search(
        space=ParameterSpace({"x": [1, 2, 3]}),
        evaluate=_result,
        seed=42,
    )
    search_b = grid_search(
        space=ParameterSpace({"x": [1, 2, 4]}),
        evaluate=_result,
        seed=43,
    )
    validation = validate_candidate(
        search=search_a,
        evaluate=_validation,
        validation_root_seed=2026,
    )

    with pytest.raises(ValueError, match="does not match"):
        summarize_selection_effect(search_b, validation)
