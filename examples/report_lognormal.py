from statfuzz import stress_test
from statfuzz.dgp import LogNormal
from statfuzz.report import build_report, failure_map_2d, write_html
from statfuzz.search import ParameterSpace, grid_search, validate_candidate

space = ParameterSpace(
    {
        "n": [8, 12, 20],
        "sigma": [0.6, 1.0, 1.4],
    }
)


def search_evaluate(point, seed):
    return stress_test(
        method="welch_ttest",
        metric="type1_error",
        dgp=LogNormal(sigma=point["sigma"]),
        n1=point["n"],
        n2=point["n"],
        simulations=2_000,
        seed=seed,
    )


def validation_evaluate(point, seed):
    return stress_test(
        method="welch_ttest",
        metric="type1_error",
        dgp=LogNormal(sigma=point["sigma"]),
        n1=point["n"],
        n2=point["n"],
        simulations=20_000,
        seed=seed,
    )


search = grid_search(
    space=space,
    evaluate=search_evaluate,
    seed=42,
    objective="absolute_deviation",
)

validation = validate_candidate(
    search=search,
    evaluate=validation_evaluate,
    validation_root_seed=2026,
)

failure_map = failure_map_2d(
    search=search,
    x_parameter="n",
    y_parameter="sigma",
)

report = build_report(
    title="Welch t-test finite-sample stress report",
    search=search,
    validation=validation,
    failure_map=failure_map,
)

report.write_json("statfuzz-report.json")
write_html(report, "statfuzz-report.html")
