from statfuzz import assert_property, stress_test
from statfuzz.dgp import Normal

result = stress_test(
    method="welch_ttest",
    metric="type1_error",
    dgp=Normal(),
    n1=30,
    n2=30,
    simulations=10_000,
    seed=42,
)

ci_result = assert_property(
    result,
    property="type1_error",
    target=0.05,
    tolerance=0.01,
)

print(ci_result.status)
print(ci_result.to_json())
