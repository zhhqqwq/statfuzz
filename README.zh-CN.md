# StatFuzz

[English](README.md) | **简体中文**

**面向统计方法的性质驱动压力测试与自动反例发现工具。**

StatFuzz 不只是“再实现一遍统计检验”。它关注的是另一个问题：

> 在受控的数据生成机制下，一个统计方法在有限样本中什么时候开始偏离我们期望的统计性质？

## 发布状态

StatFuzz 当前是 **pre-1.0 alpha 软件**，第一个公开 package release candidate 为
**0.1.0**。

仓库 Roadmap 中的 v0.1～v0.6 是历史开发阶段编号，分别对应统计核心、搜索、自动反例
发现、shrinking、报告和 StatCI；它们**不是 Python 包版本号**。因此完成 Roadmap v0.6
并不意味着包版本必须叫 0.6.0。

发布相关文档：

- [Public API](docs/PUBLIC_API.md)
- [版本策略](docs/VERSIONING.md)
- [Changelog](CHANGELOG.md)
- [0.1.0 候选发布说明](docs/RELEASE_NOTES_0.1.0.md)
- [发布检查清单](docs/RELEASING.md)

## 当前能力

StatFuzz 已经提供：

- Welch 双样本 t 检验 I 类错误的可复现 Monte Carlo 压力测试；
- MCSE 与确定性随机种子；
- Normal、平移 LogNormal、Student-t、双成分 Normal mixture DGP；
- `ParameterSpace`、完整网格搜索和无放回随机搜索；
- Objective 驱动的自动反例发现与独立 validation；
- 面向搜索选择效应的 multiplicity-aware 描述性报告；
- 标量参数与跨分布族 counterexample shrinking；
- 2D Failure Map、确定性 JSON 与 standalone HTML report；
- StatCI assertion、suite、GitHub Summary、status/badge artifact 与 baseline regression comparison。

## 快速开始

从源码 checkout 安装：

```bash
python -m pip install .
```

开发环境：

```bash
python -m pip install -e ".[dev,release]"
```

### 单点压力测试

```python
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
    seed=42,
)

print(result)
```

### 参数网格搜索

```python
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
    p = point.as_dict()
    return stress_test(
        method="welch_ttest",
        metric="type1_error",
        dgp=LogNormal(sigma=p["sigma"]),
        n1=p["n"],
        n2=p["n"],
        simulations=2_000,
        alpha=0.05,
        seed=seed,
    )

search = grid_search(
    space=space,
    evaluate=evaluate,
    seed=42,
)

print(search.to_markdown())
```

`grid_search` 会保留**全部搜索点**，而不是只返回“最差的那个”。这样既方便画 failure map，也避免把搜索过程隐藏成一个黑箱。


### 随机搜索

当有限参数空间很大时，可以不用完整遍历笛卡尔积，而是在其中**无放回抽样**：

```python
from statfuzz.search import random_search

search = random_search(
    space=space,
    evaluate=evaluate,
    draws=100,
    seed=42,
    objective="absolute_deviation",
    # 可选：只随机搜索有限空间的一部分
    # search_draws=100,
)

print(search.coverage_fraction)
print(search.to_markdown())
```

`RandomSearchResult` 与 `GridSearchResult` 共享 `ranked()`、`to_rows()`、`to_markdown()` 等结果接口，并保留所有实际抽到的参数点和扁平索引。随机抽样采用确定性的稀疏 partial Fisher–Yates，因此辅助内存与 `draws` 成正比，不需要先构造完整参数空间。


### 独立验证候选反例

```python
from statfuzz.search import validate_candidate

validated = validate_candidate(
    search=search,
    evaluate=validation_evaluate,
    validation_root_seed=2026,
)
```

验证阶段会对搜索选中的**同一个 ParameterPoint**使用新的 root seed，并生成新的确定性子 seed。返回对象会同时保留搜索与验证阶段的经验估计、MCSE、simulation budget 和随机种子，而不会把两次结果合并成一个数字。


### 高层自动反例发现

```python
from statfuzz.search import DiscoveryBudget, find_counterexample

budget = DiscoveryBudget(
    search_simulations=2_000,
    validation_simulations=20_000,
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
```

该 API 会自动串联“完整网格或随机搜索 → Objective 排名 → 候选点选择 → 独立验证”，但仍保留完整搜索表和两阶段结果。`DiscoveryBudget` 会把搜索/验证 Monte Carlo 次数真正传给 evaluator，并核验返回结果使用了指定预算。这里的“counterexample”指**在明确搜索的 DGP 参数空间中，经搜索选中并由独立随机流重新估计的候选点**，不是对无限总体类别的数学证明。


### 反例简化（shrinking）

对于已经独立验证过的候选反例，可以显式声明“什么更简单”，然后逐步简化：

```python
from statfuzz.search import ShrinkDimension, ShrinkPlan, shrink_counterexample

plan = ShrinkPlan(
    (
        ShrinkDimension("n", (4, 8, 12, 20)),
        ShrinkDimension("sigma", (0.4, 0.8, 1.0, 1.4)),
    )
)

shrunk = shrink_counterexample(
    point=discovery.point,
    plan=plan,
    evaluate=shrink_evaluate,
    simulations=20_000,
    root_seed=2028,
)
```

每个维度都按“**最简单 → 最复杂**”排列。StatFuzz 会重新模拟每一个简化 proposal，只有仍满足 failure criterion 的候选才会被接受，同时保留完整的接受/拒绝 trace。当前算法是确定性的 greedy simplification，不声称得到数学意义上的全局最小反例。详见 `docs/SHRINKING.md`。


同时支持**跨分布族简化**。用户需要显式声明 canonical family hierarchy：

```python
from statfuzz.search import FamilyPoint, FamilyShrinkPlan, shrink_dgp_family

family_plan = FamilyShrinkPlan(
    (
        FamilyPoint.from_mapping("normal", {"mean": 0.0, "sd": 1.0}),
        FamilyPoint.from_mapping(
            "lognormal",
            {"mean": 0.0, "sigma": 1.0},
        ),
        FamilyPoint.from_mapping(
            "student_t",
            {"df": 5.0, "mean": 0.0, "scale": 1.0},
        ),
        FamilyPoint.from_mapping(
            "mixture_normal",
            {
                "weight": 0.9,
                "mean1": 0.0,
                "sd1": 1.0,
                "mean2": 0.0,
                "sd2": 5.0,
                "mean": 0.0,
            },
        ),
    )
)

family_shrink = shrink_dgp_family(
    start=family_plan.levels[-1],
    plan=family_plan,
    evaluate=family_evaluate,
    simulations=20_000,
    root_seed=2030,
)
```

StatFuzz 不会自行判断哪一种分布族“更简单”；顺序完全由用户显式声明。每一次跨 family proposal 都会重新模拟，并进入可审计 trace。


### Failure Map 与报告

搜索、独立验证和 shrinking 的结果可以冻结成一份确定性的统一报告：

```python
from statfuzz.report import build_report, failure_map_2d, write_html

failure_map = failure_map_2d(
    search=search,
    x_parameter="n",
    y_parameter="sigma",
)

report = build_report(
    title="Welch t-test stress report",
    search=search,
    validation=validated,
    failure_map=failure_map,
)

report.write_json("statfuzz-report.json")
write_html(report, "statfuzz-report.html")
```

JSON 与 HTML 都只消费同一份 `StatFuzzReport` 快照，不会重新运行 simulation。Failure Map 的每个已覆盖 cell 都包含经验估计、deviation、status 和 MCSE；随机搜索没有覆盖的位置会明确显示为 Missing。详见 `docs/REPORTING.md`。


### StatCI 统计断言

现在可以把 StatFuzz 的统计结果直接变成 pytest 风格的 CI contract：

```python
from statfuzz import assert_property

ci_result = assert_property(
    result,
    property="type1_error",
    target=0.05,
    tolerance=0.01,
)
```

第一版 StatCI 规则明确为：当且仅当 `abs(observed - target) <= tolerance` 时 PASS。MCSE 会进入证据字段，但不会偷偷改变阈值。失败时会抛出继承自 `AssertionError` 的 `StatisticalAssertionError`，异常中保留完整 `StatCIResult`；如果需要先收集结果再统一决定 CI 状态，可以使用不抛统计失败异常的 `check_property(...)`。结果支持确定性 JSON 导出。详见 `docs/STATCI.md`。


多个统计检查可以汇总成一个 CI suite：

```python
from statfuzz import StatCISuiteResult, assert_suite, write_github_summary

suite = StatCISuiteResult.from_results(
    [check_a, check_b, check_c],
    name="Statistical CI",
)

write_github_summary(suite)       # 追加写入 $GITHUB_STEP_SUMMARY
suite.write_json("statci-suite.json")
assert_suite(suite)               # 任意一个检查失败时让 job FAIL
```

Suite 会保留所有子结果并计算 overall PASS/FAIL；GitHub Actions Summary 会展示 target、observed、deviation、MCSE、simulation budget 和 seed。


还可以在不重新运行任何统计检查的前提下生成轻量 CI 状态和 badge artifact：

```python
from statfuzz import StatCIStatusArtifact

status = StatCIStatusArtifact.from_suite(suite)
status.write_json("statci-status.json")
status.write_badge_json("statci-badge.json")
```

Badge JSON 兼容 Shields endpoint badge：PASS 使用 `brightgreen`，FAIL 使用 `red`，message 会包含通过数量。若要在 README 中显示实时 badge，需要把该 JSON 发布到稳定的公开 URL；GitHub Actions artifact 本身主要用于 CI 保存和下载。


Baseline statistical regression comparison 不会把两次 Monte Carlo 数字简单相减后就判定“回归”。它会先按 assertion/实验身份严格匹配 baseline 与 current，再比较**离 target 的绝对偏离是否恶化**，并要求恶化量同时超过工程阈值和 MCSE uncertainty guard：

```python
from statfuzz import RegressionPolicy, compare_suites

baseline = StatCISuiteResult.read_json("baseline-statci-suite.json")

regression = compare_suites(
    baseline,
    current_suite,
    policy=RegressionPolicy(
        minimum_worsening=0.002,
        uncertainty_multiplier=2.0,
        uncertainty_mode="conservative",
    ),
)
```

PASS→FAIL 会单独记录，但不会绕过显式 regression policy。详细的 matching key、conservative / independent uncertainty 模式和 GitHub Actions gate 见 `docs/STATCI.md`。

## 为什么搜索层不直接创建 DGP？

这是 StatFuzz 的一个核心架构决定。

`ParameterSpace` 只负责描述：

```text
参数名
+
候选值
+
确定性笛卡尔积
```

它不知道 LogNormal、Welch t-test 或 Type-I error 是什么。

真正的统计实验由用户提供的 `evaluate(point, seed)` 定义：

```text
ParameterPoint
      |
      v
 user evaluator
      |
      v
stress_test(...)
      |
      v
StressTestResult
```

这样同一个搜索层以后可以用于：

- t-test 的 I 类错误；
- Bootstrap coverage；
- OLS 区间覆盖率；
- 回归系数 bias；
- power；
- calibration；
- 其他尚未加入 StatFuzz 的统计方法。

## 统计约定

### I 类错误实验必须真的处于零假设下

对于 v0.1 的 Welch t-test，目标零假设是两组总体均值相等。

因此内置的偏态、混合分布生成器都显式控制**算术均值**。例如平移 LogNormal 会先减去原始 LogNormal 的理论均值，再加上用户指定的目标均值。

### Monte Carlo 标准误

若进行了 (B) 次独立模拟，经验拒绝率为 (hat p)，则：

[
\operatorname{MCSE}(\hat p)
=
\sqrt{\frac{\hat p(1-\hat p)}{B}}.
]

### “搜索到最坏点”不等于“验证了反例”

网格搜索会比较许多候选点。如果直接报告搜索过程中最极端的那个结果，会受到选择效应和 Monte Carlo 噪声影响。

因此 StatFuzz 的计划流程是：

```text
search budget
    |
    v
candidate
    |
    v
independent validation budget
    |
    v
validated counterexample
```

这一层独立验证已经由 `validate_candidate()` 正式加入 API。

## 项目结构

```text
src/statfuzz/
├── dgp/          # 数据生成机制
├── methods/      # 统计方法
├── metrics/      # 被验证的统计性质
├── search/       # 参数空间与搜索算法
├── simulation.py
└── result.py
```

## Roadmap

v0.1～v0.6 的功能开发阶段已经全部完成。这里的 v0.x 是历史 Roadmap 阶段编号，
不是 package semantic version。

完整记录见 [ROADMAP.md](ROADMAP.md)；真正的包版本策略见
[docs/VERSIONING.md](docs/VERSIONING.md)。

## 开发

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
```

每个新的统计性质都应该同时说明：

1. 被检验/估计的统计量是什么；
2. DGP 满足什么条件；
3. 目标性质是什么；
4. Monte Carlo 不确定性如何报告；
5. 如何用固定随机种子复现。

## License

MIT


### 多重搜索 / 选择效应报告

当 StatFuzz 在很多带 Monte Carlo 噪声的参数点中挑出最极端候选时，winner 往往会受到选择效应影响。现在搜索结果会显式报告：实际评估了多少个点、候选的 tie-aware empirical rank / upper-tail fraction、objective 分布的 median / p90 / p95，以及 OUTSIDE_TOLERANCE 的比例。

```python
from statfuzz.search import (
    summarize_search_multiplicity,
    summarize_selection_effect,
)

multiplicity = summarize_search_multiplicity(search)
selection = summarize_selection_effect(search, validated)
```

这里的 empirical percentile / upper-tail fraction 只是**已评估搜索分数中的描述性排名**，不是 multiple-comparison corrected p-value，也不提供通用 FWER/FDR 控制。search → validation 的 objective gap 也只作为 selection diagnostic；独立 validation 仍然是确认候选的关键步骤。
