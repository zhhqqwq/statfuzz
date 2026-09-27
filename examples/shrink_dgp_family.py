from statfuzz import stress_test
from statfuzz.dgp import LogNormal, MixtureNormal, Normal, StudentT
from statfuzz.search import (
    FamilyPoint,
    FamilyShrinkPlan,
    ObjectiveThresholdCriterion,
    shrink_dgp_family,
)

N = 8

plan = FamilyShrinkPlan(
    (
        FamilyPoint.from_mapping(
            "normal",
            {"mean": 0.0, "sd": 1.0},
        ),
        FamilyPoint.from_mapping(
            "lognormal",
            {"mean": 0.0, "sigma": 1.4},
        ),
        FamilyPoint.from_mapping(
            "student_t",
            {"df": 2.2, "mean": 0.0, "scale": 1.0},
        ),
        FamilyPoint.from_mapping(
            "mixture_normal",
            {
                "weight": 0.95,
                "mean1": 0.0,
                "sd1": 1.0,
                "mean2": 0.0,
                "sd2": 10.0,
                "mean": 0.0,
            },
        ),
    )
)


def build_dgp(point):
    p = point.parameters
    if point.family == "normal":
        return Normal(mean=p["mean"], sd=p["sd"])
    if point.family == "lognormal":
        return LogNormal(mean=p["mean"], sigma=p["sigma"])
    if point.family == "student_t":
        return StudentT(df=p["df"], mean=p["mean"], scale=p["scale"])
    if point.family == "mixture_normal":
        return MixtureNormal(
            weight=p["weight"],
            mean1=p["mean1"],
            sd1=p["sd1"],
            mean2=p["mean2"],
            sd2=p["sd2"],
            mean=p["mean"],
        )
    raise ValueError(f"unsupported family: {point.family}")


def evaluate(point, seed, simulations):
    return stress_test(
        method="welch_ttest",
        metric="type1_error",
        dgp=build_dgp(point),
        n1=N,
        n2=N,
        simulations=simulations,
        seed=seed,
    )


result = shrink_dgp_family(
    start=plan.levels[-1],
    plan=plan,
    evaluate=evaluate,
    simulations=20_000,
    root_seed=2030,
    criterion=ObjectiveThresholdCriterion(
        objective="absolute_deviation",
        minimum_score=0.01,
    ),
)

print("start:", result.start.as_dict())
print("final:", result.final.as_dict())

for row in result.to_rows():
    print(row)
