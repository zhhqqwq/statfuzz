from statfuzz import stress_test
from statfuzz.dgp import LogNormal
from statfuzz.search import (
    DiscoveryBudget,
    ParameterSpace,
    random_search,
    validate_candidate,
)

space = ParameterSpace(
    {
        "n": [8, 12, 20, 30],
        "sigma": [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6],
    }
)

budget = DiscoveryBudget(
    search_simulations=2_000,
    validation_simulations=20_000,
)


def search_evaluate(point, seed):
    return stress_test(
        method="welch_ttest",
        metric="type1_error",
        dgp=LogNormal(sigma=point["sigma"]),
        n1=point["n"],
        n2=point["n"],
        simulations=budget.search_simulations,
        seed=seed,
    )


def validation_evaluate(point, seed):
    return stress_test(
        method="welch_ttest",
        metric="type1_error",
        dgp=LogNormal(sigma=point["sigma"]),
        n1=point["n"],
        n2=point["n"],
        simulations=budget.validation_simulations,
        seed=seed,
    )


search = random_search(
    space=space,
    evaluate=search_evaluate,
    draws=8,
    seed=42,
    objective="absolute_deviation",
)

validated = validate_candidate(
    search=search,
    evaluate=validation_evaluate,
    validation_root_seed=2026,
)

print(search.to_markdown())
print(validated.as_row())
