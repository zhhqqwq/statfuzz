import json

import pytest

from statfuzz import StressTestResult
from statfuzz.report import (
    REPORT_SCHEMA_VERSION,
    build_report,
    failure_map_2d,
    render_html,
    write_html,
)
from statfuzz.search import (
    AbsoluteDeviationObjective,
    FamilyPoint,
    FamilyShrinkPlan,
    ParameterPoint,
    ParameterSpace,
    RandomSearchResult,
    SearchRecord,
    ShrinkDimension,
    ShrinkPlan,
    grid_search,
    shrink_counterexample,
    shrink_dgp_family,
    validate_candidate,
)


def _stress_result(*, seed, simulations, empirical, n=10):
    empirical_value = float(empirical)
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
        empirical=empirical_value,
        mcse=0.002,
        tolerance=0.01,
    )


def _grid_search():
    space = ParameterSpace(
        {
            "n": [8, 12],
            "sigma": [0.8, 1.2],
        }
    )

    def evaluate(point, seed):
        empirical = 0.05 + 0.02 * float(point["sigma"]) + 0.001 * int(point["n"])
        return _stress_result(
            seed=seed,
            simulations=100,
            empirical=empirical,
            n=int(point["n"]),
        )

    return grid_search(
        space=space,
        evaluate=evaluate,
        seed=42,
        objective="absolute_deviation",
    )


def _scalar_shrink():
    start = ParameterPoint((("n", 20), ("sigma", 1.4)))
    plan = ShrinkPlan(
        (
            ShrinkDimension("n", (4, 8, 20)),
            ShrinkDimension("sigma", (0.4, 1.0, 1.4)),
        )
    )

    def evaluate(point, seed, simulations):
        failure = int(point["n"]) >= 8 and float(point["sigma"]) >= 1.0
        return _stress_result(
            seed=seed,
            simulations=simulations,
            empirical=0.08 if failure else 0.055,
            n=int(point["n"]),
        )

    return shrink_counterexample(
        point=start,
        plan=plan,
        evaluate=evaluate,
        simulations=500,
        root_seed=77,
    )


def _family_shrink():
    plan = FamilyShrinkPlan(
        (
            FamilyPoint.from_mapping("normal", {"sd": 1.0}),
            FamilyPoint.from_mapping("lognormal", {"sigma": 1.0}),
            FamilyPoint.from_mapping("mixture", {"weight": 0.9}),
        )
    )

    def evaluate(point, seed, simulations):
        empirical = {
            "normal": 0.055,
            "lognormal": 0.08,
            "mixture": 0.09,
        }[point.family]
        return _stress_result(
            seed=seed,
            simulations=simulations,
            empirical=empirical,
        )

    return shrink_dgp_family(
        start=plan.levels[-1],
        plan=plan,
        evaluate=evaluate,
        simulations=500,
        root_seed=88,
    )


def test_report_json_is_deterministic_and_machine_readable(tmp_path):
    search = _grid_search()
    failure_map = failure_map_2d(
        search=search,
        x_parameter="n",
        y_parameter="sigma",
    )
    report = build_report(
        title="Deterministic report",
        search=search,
        failure_map=failure_map,
    )

    first = report.to_json()
    second = report.to_json()

    assert first == second
    parsed = json.loads(first)
    assert parsed["schema_version"] == REPORT_SCHEMA_VERSION
    assert parsed["search"]["strategy"] == "grid"
    assert parsed["search"]["multiplicity"]["evaluated_points"] == 4
    assert parsed["search"]["multiplicity"]["candidate_rank_one_based"] == 1
    assert parsed["search"]["multiplicity"]["empirical_upper_tail_fraction"] == 0.25
    assert parsed["failure_map"]["covered_cells"] == 4

    path = report.write_json(tmp_path / "report.json")
    assert path.read_text(encoding="utf-8") == first + "\n"


def test_report_can_snapshot_validation_scalar_and_family_shrinking():
    search = _grid_search()

    def validation_evaluate(point, seed):
        return _stress_result(
            seed=seed,
            simulations=1_000,
            empirical=0.09,
            n=int(point["n"]),
        )

    validation = validate_candidate(
        search=search,
        evaluate=validation_evaluate,
        validation_root_seed=2026,
    )

    report = build_report(
        title="Full report",
        search=search,
        validation=validation,
        scalar_shrink=_scalar_shrink(),
        family_shrink=_family_shrink(),
    )
    data = report.as_dict()

    assert data["validation"] is not None
    assert data["selection_effect"] is not None
    assert data["selection_effect"]["evaluated_points"] == 4
    assert data["selection_effect"]["candidate_rank_one_based"] == 1
    assert data["scalar_shrink"]["kind"] == "scalar"
    assert data["family_shrink"]["kind"] == "family"
    assert data["scalar_shrink"]["steps"]
    assert data["family_shrink"]["steps"]


def test_failure_map_has_sorted_axes_and_mcse_uncertainty():
    failure_map = failure_map_2d(
        search=_grid_search(),
        x_parameter="n",
        y_parameter="sigma",
    )

    assert failure_map.x_values == (8, 12)
    assert failure_map.y_values == (0.8, 1.2)
    assert failure_map.covered_cells == 4
    assert failure_map.missing_cells == 0
    assert failure_map.uncertainty == "mcse"
    assert all(
        cell is not None and cell.result.mcse == 0.002
        for row in failure_map.cells
        for cell in row
    )


def test_failure_map_requires_explicit_slice_for_extra_dimensions():
    space = ParameterSpace(
        {
            "n": [8],
            "sigma": [0.8, 1.2],
            "variance_ratio": [1.0, 2.0],
        }
    )

    def evaluate(point, seed):
        return _stress_result(
            seed=seed,
            simulations=100,
            empirical=0.08,
            n=int(point["n"]),
        )

    search = grid_search(space=space, evaluate=evaluate, seed=42)

    with pytest.raises(ValueError, match="fixed explicitly"):
        failure_map_2d(
            search=search,
            x_parameter="n",
            y_parameter="sigma",
        )

    failure_map = failure_map_2d(
        search=search,
        x_parameter="n",
        y_parameter="sigma",
        fixed={"variance_ratio": 2.0},
    )
    assert failure_map.fixed_parameters == {"variance_ratio": 2.0}
    assert failure_map.covered_cells == 2


def test_sparse_random_search_map_preserves_missing_cells():
    objective = AbsoluteDeviationObjective()
    records = (
        SearchRecord(
            point=ParameterPoint((("x", 1), ("y", 1))),
            result=_stress_result(seed=101, simulations=100, empirical=0.08),
            seed=101,
        ),
        SearchRecord(
            point=ParameterPoint((("x", 2), ("y", 2))),
            result=_stress_result(seed=202, simulations=100, empirical=0.09),
            seed=202,
        ),
    )
    search = RandomSearchResult(
        records=records,
        objective=objective,
        root_seed=42,
        parameter_names=("x", "y"),
        sampled_indices=(0, 3),
        space_size=4,
    )

    failure_map = failure_map_2d(
        search=search,
        x_parameter="x",
        y_parameter="y",
    )

    assert failure_map.total_cells == 4
    assert failure_map.covered_cells == 2
    assert failure_map.missing_cells == 2
    assert sum(
        cell is None
        for row in failure_map.cells
        for cell in row
    ) == 2


def test_failure_map_rejects_duplicate_coordinates():
    objective = AbsoluteDeviationObjective()
    point = ParameterPoint((("x", 1), ("y", 1)))
    records = (
        SearchRecord(
            point=point,
            result=_stress_result(seed=1, simulations=100, empirical=0.08),
            seed=1,
        ),
        SearchRecord(
            point=point,
            result=_stress_result(seed=2, simulations=100, empirical=0.09),
            seed=2,
        ),
    )
    search = RandomSearchResult(
        records=records,
        objective=objective,
        root_seed=42,
        parameter_names=("x", "y"),
        sampled_indices=(0, 0),
        space_size=1,
    )

    with pytest.raises(ValueError, match="same 2D coordinate"):
        failure_map_2d(
            search=search,
            x_parameter="x",
            y_parameter="y",
        )


def test_html_report_escapes_strings_and_renders_failure_map(tmp_path):
    search = _grid_search()
    report = build_report(
        title="<script>alert('x')</script>",
        search=search,
        failure_map=failure_map_2d(
            search=search,
            x_parameter="n",
            y_parameter="sigma",
        ),
    )

    rendered = render_html(report)

    assert "<script>alert" not in rendered
    assert "&lt;script&gt;" in rendered
    assert "Search Multiplicity" in rendered
    assert "not p-values" in rendered
    assert "Failure Map" in rendered
    assert "MCSE" in rendered
    assert "Search Records" in rendered

    path = write_html(report, tmp_path / "report.html")
    assert path.read_text(encoding="utf-8") == rendered


def test_html_renders_missing_cells_for_sparse_map():
    objective = AbsoluteDeviationObjective()
    records = (
        SearchRecord(
            point=ParameterPoint((("x", 1), ("y", 1))),
            result=_stress_result(seed=1, simulations=100, empirical=0.08),
            seed=1,
        ),
        SearchRecord(
            point=ParameterPoint((("x", 2), ("y", 2))),
            result=_stress_result(seed=2, simulations=100, empirical=0.09),
            seed=2,
        ),
    )
    search = RandomSearchResult(
        records=records,
        objective=objective,
        root_seed=42,
        parameter_names=("x", "y"),
        sampled_indices=(0, 3),
        space_size=4,
    )
    report = build_report(
        title="Sparse",
        search=search,
        failure_map=failure_map_2d(
            search=search,
            x_parameter="x",
            y_parameter="y",
        ),
    )

    assert "Missing" in render_html(report)


def test_html_renders_search_to_validation_selection_diagnostic():
    search = _grid_search()

    def validation_evaluate(point, seed):
        return _stress_result(
            seed=seed,
            simulations=1_000,
            empirical=0.075,
            n=int(point["n"]),
        )

    validation = validate_candidate(
        search=search,
        evaluate=validation_evaluate,
        validation_root_seed=2026,
    )
    report = build_report(
        title="Selection diagnostic",
        search=search,
        validation=validation,
    )

    rendered = render_html(report)

    assert "Search → validation diagnostic" in rendered
    assert "Search − validation gap" in rendered
    assert "not an unbiased estimate" in rendered
