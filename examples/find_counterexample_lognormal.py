from statfuzz import stress_test
from statfuzz.dgp import LogNormal
from statfuzz.search import DiscoveryBudget, ParameterSpace, find_counterexample

space = ParameterSpace(
    {
        "n": [8, 12, 20],
        "sigma": [0.6, 1.0, 1.4],
    }
)

budget = DiscoveryBudget(
    search_simulations=2_000,
    validation_simulations=20_000,
)


def search_evaluate(point, seed, simulations):
    return stress_test(
        method="welch_ttest",
        metric="type1_error",
        dgp=LogNormal(sigma=point["sigma"]),
        n1=point["n"],
        n2=point["n"],
        simulations=simulations,
        seed=seed,
    )


def validation_evaluate(point, seed, simulations):
    return stress_test(
        method="welch_ttest",
        metric="type1_error",
        dgp=LogNormal(sigma=point["sigma"]),
        n1=point["n"],
        n2=point["n"],
        simulations=simulations,
        seed=seed,
    )


discovery = find_counterexample(
    space=space,
    search_evaluate=search_evaluate,
    validation_evaluate=validation_evaluate,
    budget=budget,
    search_root_seed=42,
    validation_root_seed=2026,
    objective="absolute_deviation",
)

print(discovery.as_row())
print(discovery.search.to_markdown())
