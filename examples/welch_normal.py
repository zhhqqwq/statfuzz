from statfuzz import stress_test
from statfuzz.dgp import Normal


result = stress_test(
    method="welch_ttest",
    metric="type1_error",
    dgp=Normal(),
    n1=30,
    n2=30,
    simulations=10_000,
    alpha=0.05,
    tolerance=0.01,
    seed=42,
)

print(result)
