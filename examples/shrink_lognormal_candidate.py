from statfuzz import stress_test
from statfuzz.dgp import LogNormal
from statfuzz.search import (
    ObjectiveThresholdCriterion,
    ParameterPoint,
    ShrinkDimension,
    ShrinkPlan,
    shrink_counterexample,
)

start = ParameterPoint(
    (
        ("n", 20),
        ("sigma", 1.4),
    )
)

plan = ShrinkPlan(
    (
        ShrinkDimension("n", (4, 8, 12, 20)),
        ShrinkDimension("sigma", (0.4, 0.8, 1.0, 1.4)),
    )
)


def evaluate(point, seed, simulations):
    return stress_test(
        method="welch_ttest",
        metric="type1_error",
        dgp=LogNormal(sigma=point["sigma"]),
        n1=point["n"],
        n2=point["n"],
        simulations=simulations,
        seed=seed,
    )


result = shrink_counterexample(
    point=start,
    plan=plan,
    evaluate=evaluate,
    simulations=10_000,
    root_seed=2028,
    criterion=ObjectiveThresholdCriterion(
        objective="absolute_deviation",
        minimum_score=0.015,
    ),
)

print("start:", result.start_point.as_dict())
print("final:", result.final_point.as_dict())
print("complexity reduction:", result.complexity_reduction)

for row in result.to_rows():
    print(row)
