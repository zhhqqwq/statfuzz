from statfuzz import (
    StatCISuiteResult,
    assert_suite,
    check_property,
    stress_test,
    write_github_summary,
)
from statfuzz.dgp import Normal

first = stress_test(
    method="welch_ttest",
    metric="type1_error",
    dgp=Normal(),
    n1=20,
    n2=20,
    simulations=10_000,
    seed=42,
)

second = stress_test(
    method="welch_ttest",
    metric="type1_error",
    dgp=Normal(),
    n1=40,
    n2=40,
    simulations=10_000,
    seed=43,
)

suite = StatCISuiteResult.from_results(
    [
        check_property(
            first,
            property="type1_error",
            target=0.05,
            tolerance=0.01,
        ),
        check_property(
            second,
            property="type1_error",
            target=0.05,
            tolerance=0.01,
        ),
    ],
    name="Welch Type-I error checks",
)

suite.write_json("statci-suite.json")
write_github_summary(suite, "statci-summary.md")

# In GitHub Actions, omit the explicit path above so GITHUB_STEP_SUMMARY is used.
assert_suite(suite)
