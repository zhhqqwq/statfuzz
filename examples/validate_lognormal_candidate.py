from statfuzz import stress_test
from statfuzz.dgp import LogNormal
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
)

validated = validate_candidate(
    search=search,
    evaluate=validation_evaluate,
    validation_root_seed=2026,
)

print(validated.as_row())
