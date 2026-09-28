# StatFuzz

[CI](https://github.com/zhhqqwq/statfuzz/actions/workflows/ci.yml) · [文档](docs/PUBLIC_API.md) · [示例](examples) · [English](README.md)

面向统计方法的 Monte Carlo 压力测试工具。

StatFuzz 是一个 Python 库，用于研究统计方法在指定数据生成条件下的表现。它在有限参数空间中搜索偏离目标性质的候选点，使用独立随机流重新估计，并记录估计值、Monte Carlo 不确定性和随机种子。

当前内置实验评估 Welch 双样本 t 检验的 I 类错误率。你可以用它研究：当两组总体均值相等时，样本量和分布形状如何影响拒绝率；也可以在修改代码后，检查统计表现是否发生退化。

当前版本 0.1.0 处于 alpha 阶段，次版本更新可能调整公开接口，详见[版本策略](docs/VERSIONING.md)。

## 安装

需要 Python 3.10 或更高版本。安装时会自动安装 NumPy 和 SciPy。
CI 测试矩阵覆盖 Python 3.10、3.11 和 3.12。

从源码安装：

```bash
git clone https://github.com/zhhqqwq/statfuzz.git
cd statfuzz
python -m pip install .
```

## 快速开始

从同一个平移对数正态分布独立抽取两组样本，估计检验的拒绝率：

```python
from statfuzz import stress_test
from statfuzz.dgp import LogNormal

result = stress_test(
    method="welch_ttest",
    metric="type1_error",
    dgp=LogNormal(sigma=1.4, mean=0.0),
    n1=12,
    n2=12,
    simulations=10_000,
    alpha=0.05,
    tolerance=0.01,
    seed=42,
)

print(result)
```

结果包含经验拒绝率、Monte Carlo 标准误（MCSE）及其与 `alpha` 的偏差。
绝对偏差不超过 `tolerance` 时通过，否则状态为 `OUTSIDE_TOLERANCE`。

`LogNormal(mean=0.0)` 会平移生成的样本，使总体算术均值为零，因此两组满足均值相等的零假设。
如需使用不同的组间分布，可通过 `dgp2` 指定第二组，并将两组总体均值设为相同值。

## 搜索、验证和报告

下面的完整示例对九个参数组合各运行 2,000 次模拟，选出绝对偏差最大的候选点，再用独立种子运行 20,000 次模拟进行验证。

```python
from statfuzz import stress_test
from statfuzz.dgp import LogNormal
from statfuzz.report import build_report, failure_map_2d, write_html
from statfuzz.search import DiscoveryBudget, ParameterSpace, find_counterexample

space = ParameterSpace({
    "n": [8, 12, 20],
    "sigma": [0.6, 1.0, 1.4],
})


def evaluate(point, seed, simulations):
    return stress_test(
        method="welch_ttest",
        metric="type1_error",
        dgp=LogNormal(sigma=point["sigma"], mean=0.0),
        n1=point["n"],
        n2=point["n"],
        simulations=simulations,
        alpha=0.05,
        tolerance=0.01,
        seed=seed,
    )


discovery = find_counterexample(
    space=space,
    search_evaluate=evaluate,
    validation_evaluate=evaluate,
    budget=DiscoveryBudget(
        search_simulations=2_000,
        validation_simulations=20_000,
    ),
    search_root_seed=42,
    validation_root_seed=2026,
    objective="absolute_deviation",
)

print(discovery.point)
print(discovery.validation_result)

report = build_report(
    title="Welch t-test under shifted lognormal sampling",
    search=discovery.search,
    validation=discovery.validation,
    failure_map=failure_map_2d(
        search=discovery.search,
        x_parameter="n",
        y_parameter="sigma",
    ),
)
report.write_json("statfuzz-report.json")
write_html(report, "statfuzz-report.html")
```

在浏览器中打开 `statfuzz-report.html`，可以查看搜索结果、独立验证结果和二维失效图。
JSON 与 HTML 使用同一份结果快照。

即使验证结果在容差范围内，`find_counterexample` 也会返回排名最高的候选点。
判断是否失效时，应查看 `discovery.validation_result.status`。

参数空间较大时，可在 `find_counterexample` 中设置 `search_draws`，进行无放回随机抽样。
搜索层把参数点、种子和模拟预算传给 evaluator，由 evaluator 定义具体的统计实验。

## 在 CI 中使用统计检查

下面使用快速开始示例中的 `result`：

```python
from statfuzz import check_property

check = check_property(
    result,
    property="type1_error",
    target=0.05,
    tolerance=0.01,
)
print(check.status)
check.write_json("statci-result.json")
```

如需让检查失败时抛出异常，可用相同参数调用 `assert_property`。
它抛出的 `StatisticalAssertionError` 继承自 `AssertionError`，并保留完整结果。

StatCI 还支持汇总多个检查、写入 GitHub Actions 摘要，以及导出状态和徽章 JSON。
基线比较会先匹配实验身份，再判断相对目标值的绝对偏差是否恶化，并要求恶化量同时超过设定的工程阈值和 MCSE 不确定性阈值。
具体策略和 CI 示例见 [StatCI 文档](docs/STATCI.md)。

## 已支持的功能

| 模块 | 当前能力 |
| --- | --- |
| 统计实验 | Welch 双侧 t 检验；经验 I 类错误率与 MCSE |
| 数据生成 | 正态、平移对数正态、Student-t、双成分正态混合分布 |
| 参数搜索 | 有限笛卡尔积参数空间；网格搜索和无放回随机搜索 |
| 独立验证 | 分别设置搜索与验证预算，派生确定性子种子，分别保留估计结果 |
| 反例简化 | 按显式顺序贪心简化参数或分布族，保留完整尝试记录 |
| 结果报告 | 搜索选择效应诊断、二维失效图、JSON 和独立 HTML |
| StatCI | 断言、检查汇总、GitHub Actions 摘要、状态与徽章文件、基线比较 |

## 结果解释

I 类错误实验要求零假设成立。对于内置的 Welch 实验，两组总体均值必须相等。
调用者需要选择满足这一条件的数据生成机制。

MCSE 描述经验拒绝率的模拟不确定性。PASS/FAIL 使用明确的数值容差，MCSE 不会自动调整该阈值。
基线回归比较使用单独的策略，其中包含不确定性阈值。

从大量含有模拟噪声的估计值中选择极端结果，会产生选择效应。独立验证会重新估计该候选点。
报告中的搜索排名和百分位数是描述性指标，不提供校正后的 p 值或通用 FWER/FDR 控制。
候选反例的结论限于指定实验和模拟预算；反例简化依照用户给定的顺序执行，不保证找到全局最小反例。

复现实验时，应保留种子、参数、模拟预算和软件版本。自定义 evaluator 必须使用传入的种子和预算。
本版本的 `stress_test` 尚未内置 Bootstrap 覆盖率、OLS 推断、检验功效或校准评估。

## 文档

| 主题 | 入口 |
| --- | --- |
| 公开导入接口与兼容性 | [Public API](docs/PUBLIC_API.md) |
| 参数空间、搜索与验证 | [Search](docs/SEARCH.md) |
| 参数和分布族简化 | [Shrinking](docs/SHRINKING.md) |
| 失效图与报告导出 | [Reporting](docs/REPORTING.md) |
| 统计断言与回归比较 | [StatCI](docs/STATCI.md) |
| 架构设计 | [Design](docs/DESIGN.md) |
| 可运行脚本 | [Examples](examples) |
| 版本与发布 | [更新日志](CHANGELOG.md)、[0.1.0 发布说明](docs/RELEASE_NOTES_0.1.0.md)、[发布检查清单](docs/RELEASING.md) |

[Roadmap](ROADMAP.md) 记录已完成的开发阶段，其中 v0.1～v0.6 的阶段编号与软件包版本独立。

## 开发与贡献

在仓库根目录运行：

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check .
```

提交 Pull Request 前请阅读[贡献指南](CONTRIBUTING.md)。
涉及统计方法的贡献应说明目标性质与假设，提供可复现的种子，并报告 Monte Carlo 不确定性。

问题反馈和功能建议请提交至 [GitHub Issues](https://github.com/zhhqqwq/statfuzz/issues)。
反馈模拟结果时，请附上最小示例、依赖版本、种子和模拟预算。
项目由 [zhhqqwq](https://github.com/zhhqqwq) 维护。

## 许可证

[MIT](LICENSE)。
