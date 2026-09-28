from statfuzz import (
    RegressionPolicy,
    StatCISuiteResult,
    assert_regression_suite,
    compare_suites,
    write_regression_summary,
)

# Baseline suites can be preserved as CI artifacts or committed fixtures.
baseline = StatCISuiteResult.read_json("baseline-statci-suite.json")

# The current suite is produced by today's StatCI checks.
current = StatCISuiteResult.read_json("current-statci-suite.json")

policy = RegressionPolicy(
    minimum_worsening=0.002,
    uncertainty_multiplier=2.0,
    uncertainty_mode="conservative",
    strict_matching=True,
)

regression = compare_suites(
    baseline,
    current,
    policy=policy,
    name="StatCI baseline regression",
)

regression.write_json("statci-regression.json")
write_regression_summary(regression, "statci-regression-summary.md")

# In GitHub Actions, omit the explicit summary path above to append to
# GITHUB_STEP_SUMMARY. Gate only after preserving the comparison evidence.
assert_regression_suite(regression)
