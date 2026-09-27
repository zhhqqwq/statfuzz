from statfuzz import stress_test
from statfuzz.dgp import LogNormal


result = stress_test(
    method="welch_ttest",
    metric="type1_error",
    dgp=LogNormal(sigma=1.4),
    n1=12,
    n2=12,
    simulations=10_000,
    alpha=0.05,
    tolerance=0.01,
    seed=42,
)

print(result)
