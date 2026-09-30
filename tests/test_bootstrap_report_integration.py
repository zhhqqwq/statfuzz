import json

from statfuzz import StressTestResult, bootstrap_mean_coverage
from statfuzz.dgp import LogNormal, MixtureNormal, Normal, StudentT
from statfuzz.methods import BootstrapMeanPercentile
from statfuzz.report import (
    REPORT_SCHEMA_VERSION,
    build_report,
    failure_map_2d,
    render_html,
)
from statfuzz.search import (
    FamilyPoint,
    FamilyShrinkPlan,
    ParameterPoint,
    ParameterSpace,
    ShrinkDimension,
    ShrinkPlan,
    grid_search,
    shrink_counterexample,
    shrink_dgp_family,
    validate_candidate,
)

SEARCH_METHOD = BootstrapMeanPercentile(
    resamples=9,
    interval_level=0.8,
)
SHRINK_METHOD = BootstrapMeanPercentile(
    resamples=19,
    interval_level=0.8,
)


def _bootstrap_search_result(point, seed, simulations):
    return bootstrap_mean_coverage(
        dgp=LogNormal(
            sigma=point["sigma"],
            mean=0.0,
        ),
        n=point["n"],
        simulations=simulations,
        method=SEARCH_METHOD,
        tolerance=0.10,
        seed=seed,
        batch_size=7,
        evidence_confidence_level=0.9,
    )


def _bootstrap_search():
    return grid_search(
        space=ParameterSpace(
            {
                "n": [8, 12],
                "sigma": [0.6, 1.0],
            }
        ),
        evaluate=lambda point, seed: _bootstrap_search_result(
            point,
            seed,
            24,
        ),
        seed=2026,
    )


def _bootstrap_validation(search):
    return validate_candidate(
        search=search,
        evaluate=lambda point, seed: _bootstrap_search_result(
            point,
            seed,
            48,
        ),
        validation_root_seed=2027,
    )


def _bootstrap_scalar_shrink():
    plan = ShrinkPlan(
        (
            ShrinkDimension("n", (4, 8, 12, 20)),
            ShrinkDimension("sigma", (0.1, 0.4, 0.8, 1.4)),
        )
    )

    def evaluate(point, seed, simulations):
        return bootstrap_mean_coverage(
            dgp=LogNormal(
                sigma=point["sigma"],
                mean=0.0,
            ),
            n=point["n"],
            simulations=simulations,
            method=SHRINK_METHOD,
            tolerance=0.10,
            seed=seed,
            batch_size=7,
            evidence_confidence_level=0.9,
        )

    return shrink_counterexample(
        point=ParameterPoint(
            (
                ("n", 20),
                ("sigma", 1.4),
            )
        ),
        plan=plan,
        evaluate=evaluate,
        simulations=40,
        root_seed=23,
    )


def _family_candidate(family, **parameters):
    return FamilyPoint.from_mapping(family, parameters)


def _bootstrap_family_plan():
    return FamilyShrinkPlan(
        (
            _family_candidate("normal", mean=0.0, sd=1.0),
            _family_candidate("lognormal", mean=0.0, sigma=1.0),
            _family_candidate(
                "student_t",
                df=5.0,
                mean=0.0,
                scale=1.0,
            ),
            _family_candidate(
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


def _family_dgp(point):
    if point.family == "normal":
        return Normal(
            mean=point.parameters["mean"],
            sd=point.parameters["sd"],
        )
    if point.family == "lognormal":
        return LogNormal(
            mean=point.parameters["mean"],
            sigma=point.parameters["sigma"],
        )
    if point.family == "student_t":
        return StudentT(
            df=point.parameters["df"],
            mean=point.parameters["mean"],
            scale=point.parameters["scale"],
        )
    if point.family == "mixture_normal":
        return MixtureNormal(
            weight=point.parameters["weight"],
            mean1=point.parameters["mean1"],
            sd1=point.parameters["sd1"],
            mean2=point.parameters["mean2"],
            sd2=point.parameters["sd2"],
            mean=point.parameters["mean"],
        )
    raise AssertionError(f"unexpected family {point.family!r}")


def _bootstrap_family_shrink():
    plan = _bootstrap_family_plan()

    def evaluate(point, seed, simulations):
        return bootstrap_mean_coverage(
            dgp=_family_dgp(point),
            n=8,
            simulations=simulations,
            method=SHRINK_METHOD,
            tolerance=0.10,
            seed=seed,
            batch_size=7,
            evidence_confidence_level=0.9,
        )

    return shrink_dgp_family(
        start=plan.levels[-1],
        plan=plan,
        evaluate=evaluate,
        simulations=40,
        root_seed=1,
    )


def _full_bootstrap_report():
    search = _bootstrap_search()
    validation = _bootstrap_validation(search)
    return build_report(
        title="Bootstrap coverage report",
        search=search,
        validation=validation,
        scalar_shrink=_bootstrap_scalar_shrink(),
        family_shrink=_bootstrap_family_shrink(),
        failure_map=failure_map_2d(
            search=search,
            x_parameter="n",
            y_parameter="sigma",
        ),
    )


def test_build_report_accepts_complete_bootstrap_workflow():
    report = _full_bootstrap_report()
    data = report.as_dict()

    assert data["schema_version"] == REPORT_SCHEMA_VERSION == "1.3"

    search_result = data["search"]["records"][0]["result"]
    assert search_result["metric"] == "coverage"
    assert search_result["method"] == "bootstrap_mean_percentile"
    assert search_result["dgp_identity"]["family"] == "statfuzz.dgp.LogNormal"
    assert search_result["target_check"]["kind"] == "mean"
    assert search_result["target_check"]["source"] == "population_mean"
    assert isinstance(search_result["coverage_count"], int)
    assert search_result["bootstrap_method"] == SEARCH_METHOD.as_dict()
    assert search_result["evidence_interval"]["level"] == 0.9
    assert search_result["evidence_interval"]["method"] == "wilson"
    assert (
        search_result["evidence_interval"]["low"]
        <= search_result["empirical"]
        <= search_result["evidence_interval"]["high"]
    )

    assert data["validation"]["search_result"]["metric"] == "coverage"
    assert data["validation"]["validation_result"]["metric"] == "coverage"
    assert data["validation"]["validation_result"]["simulations"] == 48

    assert data["scalar_shrink"]["kind"] == "scalar"
    assert data["scalar_shrink"]["start_result"]["metric"] == "coverage"
    assert data["scalar_shrink"]["final_result"]["metric"] == "coverage"
    assert data["scalar_shrink"]["final"] == {
        "n": 4,
        "sigma": 0.4,
    }

    assert data["family_shrink"]["kind"] == "family"
    assert data["family_shrink"]["start_result"]["metric"] == "coverage"
    assert data["family_shrink"]["final_result"]["metric"] == "coverage"
    assert data["family_shrink"]["final"]["family"] == "lognormal"

    map_result = data["failure_map"]["cells"][0][0]["result"]
    assert map_result["metric"] == "coverage"
    assert "coverage_count" in map_result
    assert "rejection_count" not in map_result


def test_bootstrap_report_json_is_exactly_deterministic():
    first = _full_bootstrap_report()
    second = _full_bootstrap_report()

    assert first == second
    assert first.as_dict() == second.as_dict()
    assert first.to_json() == second.to_json()
    assert json.loads(first.to_json()) == first.as_dict()


def test_bootstrap_report_html_uses_coverage_language():
    rendered = render_html(_full_bootstrap_report())

    assert "Bootstrap coverage report" in rendered
    assert "Search Records" in rendered
    assert "Independent Validation" in rendered
    assert "Scalar Shrinking" in rendered
    assert "Family Shrinking" in rendered
    assert "Failure Map" in rendered
    assert "<th>Covered</th>" in rendered
    assert "<th>MC evidence interval</th>" in rendered
    assert "<dt>Covered</dt>" in rendered
    assert "<dt>Mean target</dt>" in rendered
    assert "90% wilson" in rendered
    assert "<th>Rejections</th>" not in rendered


def test_type_i_report_result_shape_and_html_remain_unchanged():
    def evaluate(point, seed):
        return StressTestResult(
            method="fake",
            metric="type1_error",
            dgp1="fake-one",
            dgp2="fake-two",
            n1=10,
            n2=10,
            simulations=100,
            seed=seed,
            nominal=0.05,
            empirical=0.08,
            mcse=0.002,
            tolerance=0.01,
        )

    search = grid_search(
        space=ParameterSpace({"n": [10]}),
        evaluate=evaluate,
        seed=42,
    )
    report = build_report(title="Type-I compatibility", search=search)
    result = report.as_dict()["search"]["records"][0]["result"]

    assert REPORT_SCHEMA_VERSION == "1.3"
    assert result == {
        "method": "fake",
        "metric": "type1_error",
        "dgp1": "fake-one",
        "dgp2": "fake-two",
        "dgp1_identity": None,
        "dgp2_identity": None,
        "null_check": None,
        "rejection_count": None,
        "confidence_interval": None,
        "n1": 10,
        "n2": 10,
        "simulations": 100,
        "seed": result["seed"],
        "nominal": 0.05,
        "empirical": 0.08,
        "mcse": 0.002,
        "tolerance": 0.01,
        "deviation": 0.03,
        "status": "OUTSIDE_TOLERANCE",
    }

    rendered = render_html(report)
    assert "<th>Rejections</th>" in rendered
    assert "<th>Binomial interval</th>" in rendered
    assert "<th>Covered</th>" not in rendered
