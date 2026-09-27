# StatFuzz

[English](README.md) | **简体中文**

**面向统计方法的性质驱动压力测试与自动反例发现工具。**

StatFuzz 不只是“再实现一遍统计检验”。它关注的是另一个问题：

> 在受控的数据生成机制下，一个统计方法在有限样本中什么时候开始偏离我们期望的统计性质？

## 当前能力

v0.1 已经提供：

- Welch 双样本 t 检验；
- 经验 I 类错误率估计；
- Monte Carlo 标准误（MCSE）；
- 可复现随机种子；
- Normal、平移 LogNormal、Student-t、双成分 Normal mixture 四类 DGP；
- 与 SciPy 的数值交叉验证；
- GitHub Actions CI。

开发中的搜索层进一步加入：

- `ParameterSpace`：声明有限、可序列化、确定性枚举的参数空间；
- `grid_search`：遍历参数网格，保存每一个搜索点的完整统计结果；
- 稳定的 `absolute_deviation` 排序；
- 基于根随机种子和参数点内容生成的独立、可复现子种子。

## 快速开始

```bash
python -m pip install -e ".[dev]"
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

- **v0.1**：Welch t-test + Type-I error 压力测试核心
- **v0.2**：参数空间、网格搜索、随机搜索
- **v0.3**：`find_counterexample()` + 独立验证
- **v0.4**：counterexample shrinking
- **v0.5**：failure map + HTML report
- **v0.6**：StatCI / GitHub Actions

详细规划见 [ROADMAP.md](ROADMAP.md)。

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
