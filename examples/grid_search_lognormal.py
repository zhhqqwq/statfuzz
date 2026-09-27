from statfuzz import stress_test
from statfuzz.dgp import LogNormal
from statfuzz.search import ParameterSpace, grid_search

space = ParameterSpace(
    {
        "n": [8, 12, 20],
        "sigma": [0.6, 1.0, 1.4],
    }
)


def evaluate(point, seed):
    params = point.as_dict()
    return stress_test(
        method="welch_ttest",
        metric="type1_error",
        dgp=LogNormal(sigma=params["sigma"]),
        n1=params["n"],
        n2=params["n"],
        simulations=2_000,
        alpha=0.05,
        tolerance=0.01,
        seed=seed,
    )


result = grid_search(
    space=space,
    evaluate=evaluate,
    seed=42,
)

print(result.to_markdown())
