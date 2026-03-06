# 蒙特卡洛因子质检系统设计文档

**日期**: 2026-03-06
**状态**: 设计完成，待实现

---

## 1. 项目概述

### 1.1 目标

构建一个蒙特卡洛模拟驱动的**因子质检系统**，通过对比技术指标在真实市场 vs 随机游走中的触发概率，筛选出具有 Alpha 挖掘价值的真因子，过滤掉伪信号。

### 1.2 核心指标

- **Alpha Ratio** = $P_{actual} / P_{random}$
  - ≈ 1: 随机噪音，建议废弃
  - > 1.5: 具备 Alpha 价值，建议保留

### 1.3 适用范围

- **标的**: 全市场 A 股
- **数据**: 日线级别，前复权
- **模拟规模**: 10,000 条路径 × 252 个交易日

---

## 2. 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    FactorFilter (主控模块)                        │
│              串联四个阶段，输出报告和结构化数据                      │
└─────────────────────────────────────────────────────────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
        ▼                      ▼                      ▼
┌───────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ 阶段1: 参数估计 │    │ 阶段2: GBM 模拟引擎 │    │ 阶段3: 信号检测   │
│ParameterEstimator│   │  GBMSimulator    │    │  SignalDetector  │
│               │    │                  │    │                  │
│ • 读取A股数据  │    │ • 矩阵化生成路径  │    │ • 向量化指标计算  │
│ • 滚动估计μ,σ │───▶│ • 涨跌停约束     │───▶│ • 统计P_random   │
│ • 计算P_actual │    │ • 合成OHLC       │    │                  │
└───────────────┘    └──────────────────┘    └──────────────────┘
                                                       │
                              ┌──────────────────────┘
                              ▼
                   ┌──────────────────┐
                   │ 阶段4: 显著性检验  │
                   │ SignificanceTester│
                   │                  │
                   │ • Alpha Ratio    │
                   │ • 统计显著性      │
                   │ • 生成报告        │
                   └──────────────────┘
```

---

## 3. 模块详细设计

### 3.1 ParameterEstimator - 参数估计器

**职责**: 从真实 A 股数据中提取生成随机游走所需的统计特征

**核心方法**:

```python
class ParameterEstimator:
    def __init__(self, reader: DataReader, window: int = 252):
        """
        Args:
            reader: DataReader 实例，用于读取本地 DuckDB 数据
            window: 滚动窗口天数，默认 252 个交易日（约1年）
        """

    def estimate_parameters(
        self,
        ts_codes: List[str],
        start_date: str,
        end_date: str,
        factor: Factor
    ) -> pd.DataFrame:
        """
        返回每只股票的参数估计 DataFrame:
        - ts_code: 股票代码
        - mu: 年化对数收益率均值
        - sigma: 年化波动率
        - p_actual: 因子在真实数据上的触发概率

        计算公式:
        - $R_t = \ln(S_t / S_{t-1})$
        - $\mu = mean(R_t) \times 252$ (年化)
        - $\sigma = std(R_t) \times \sqrt{252}$ (年化)
        """

    def calculate_p_actual(self, df: pd.DataFrame, factor: Factor) -> float:
        """
        计算因子在真实历史数据上的触发概率:
        P_actual = 信号触发次数 / 总交易日数
        """
```

**数据清洗流程**:
1. 读取日线数据（前复权）
2. 剔除停牌日（成交量为0或价格无变化）
3. 剔除涨跌停日（可选，用于特殊测试场景）
4. 计算对数收益率序列

---

### 3.2 GBMSimulator - GBM 模拟引擎

**职责**: 构建高效的、符合 A 股机制的几何布朗运动生成器

**核心方法**:

```python
class GBMSimulator:
    def __init__(
        self,
        n_paths: int = 10000,
        n_steps: int = 252,
        limit_up: float = 0.10,      # 主板涨停 10%
        limit_down: float = -0.10,   # 主板跌停 10%
        dt: float = 1/252            # 时间步长（年化）
    ):
        """
        Args:
            n_paths: 模拟路径数量，建议 >= 10000 保证统计显著性
            n_steps: 每条路径的时间步长（交易日数）
            limit_up/down: 涨跌停限制，创业板/科创板可设为 0.20
        """

    def simulate(
        self,
        s0: float,
        mu: float,
        sigma: float,
        generate_ohlc: bool = False
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        批量生成模拟价格路径

        GBM 公式:
        $S_t = S_{t-1} \exp\left( \left(\mu - \frac{\sigma^2}{2}\right)\Delta t + \sigma \sqrt{\Delta t} Z \right)$

        Args:
            s0: 初始价格（标准化为 100）
            mu: 年化收益率
            sigma: 年化波动率
            generate_ohlc: 是否生成 OHLC 数据

        Returns:
            如果 generate_ohlc=False: 收盘价矩阵 (n_paths, n_steps)
            如果 generate_ohlc=True: {'open': ..., 'high': ..., 'low': ..., 'close': ...}
        """

    def _apply_price_limits(
        self,
        price_matrix: np.ndarray
    ) -> np.ndarray:
        """
        应用 A 股涨跌停约束

        实现逻辑:
        1. 计算单日收益率: returns = (S_t - S_{t-1}) / S_{t-1}
        2. 使用 np.clip 截断收益率: clipped = np.clip(returns, limit_down, limit_up)
        3. 还原价格: S_t = S_{t-1} * (1 + clipped)

        注意: 截断后需要重新标准化或记录，避免累计误差
        """
```

**合成 OHLC 数据**（可选）:
- 基于收盘价路径，使用随机波动率模型模拟日内波动
- Open ≈ 前一日 Close
- High = Close × (1 + random(0, intraday_vol))
- Low = Close × (1 - random(0, intraday_vol))

---

### 3.3 SignalDetector - 信号检测器

**职责**: 在数万条平行"合成宇宙"中，计算目标因子的随机触发概率

**核心方法**:

```python
class SignalDetector:
    def __init__(self, vectorized: bool = True):
        """
        Args:
            vectorized: 是否使用向量化计算（强烈推荐 True）
        """

    def detect_signals(
        self,
        price_matrix: np.ndarray,  # (n_paths, n_steps)
        factor: Factor
    ) -> np.ndarray:
        """
        在模拟价格矩阵上检测因子信号

        Args:
            price_matrix: 价格矩阵，形状 (n_paths, n_steps)
            factor: Factor 实例，定义了信号检测逻辑

        Returns:
            信号矩阵 (n_paths, n_steps)，True 表示该时刻触发信号
        """

    def calculate_p_random(self, signal_matrix: np.ndarray) -> float:
        """
        计算随机触发概率:
        P_random = 触发总次数 / (N × T)
        """

    def _macd_golden_cross_vectorized(
        self,
        price_matrix: np.ndarray,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> np.ndarray:
        """
        向量化 MACD 金叉检测（内置因子示例）

        实现要点:
        1. 使用卷积或累积加权实现 EMA 的向量化计算
        2. EMA_fast = ema(price, 12), EMA_slow = ema(price, 26)
        3. DIF = EMA_fast - EMA_slow
        4. DEA = ema(DIF, 9)
        5. 金叉 = (DIF > DEA) & (DIF.shift(1) <= DEA.shift(1))
        """
```

**向量化实现策略**:

```python
# EMA 向量化计算（使用指数加权移动平均）
def ema_vectorized(matrix: np.ndarray, span: int) -> np.ndarray:
    """
    对 (N, T) 维矩阵计算 EMA
    使用 pandas.DataFrame.ewm 或手动实现指数加权
    """
    alpha = 2 / (span + 1)
    # 使用累积乘积实现向量化 EMA
    ...

# RSI 向量化计算
def rsi_vectorized(matrix: np.ndarray, period: int = 14) -> np.ndarray:
    """
    对 (N, T) 维矩阵计算 RSI
    """
    ...
```

---

### 3.4 SignificanceTester - 显著性检验器

**职责**: 通过对比 $P_{actual}$ 和 $P_{random}$，决定因子是保留、优化还是废弃

**核心方法**:

```python
@dataclass
class TestResult:
    """单个标的的检验结果"""
    ts_code: str
    p_actual: float
    p_random: float
    alpha_ratio: float
    n_signals_actual: int
    n_signals_random: int
    p_value: float
    is_significant: bool
    recommendation: str  # "KEEP" | "OPTIMIZE" | "DISCARD"


class SignificanceTester:
    def __init__(
        self,
        alpha_threshold: float = 1.5,
        confidence_level: float = 0.95
    ):
        """
        Args:
            alpha_threshold: Alpha Ratio 阈值，超过认为显著
            confidence_level: 置信水平，用于计算 p-value
        """

    def test(
        self,
        p_actual: float,
        p_random: float,
        n_total: int
    ) -> TestResult:
        """
        统计假设检验:

        零假设 (H0): 该指标的触发纯粹是随机游走的结果
        备择假设 (H1): 真实市场中存在非随机力量促成该形态

        检验统计量: Alpha Ratio = P_actual / P_random

        Returns:
            TestResult 包含检验结果和建议
        """

    def generate_report(
        self,
        factor: Factor,
        results: List[TestResult]
    ) -> str:
        """
        生成 Markdown 格式报告
        """
```

**检验逻辑**:

```python
def evaluate_factor(alpha_ratio: float) -> str:
    if alpha_ratio < 1.2:
        return "DISCARD", "Alpha Ratio 接近1，随机噪音，建议废弃"
    elif alpha_ratio < 1.5:
        return "OPTIMIZE", "有一定信号，但未达显著阈值，建议优化"
    else:
        return "KEEP", "Alpha Ratio 显著大于1，具备 Alpha 挖掘价值"
```

---

### 3.5 Factor - 因子定义模块（混合模式）

**职责**: 支持配置化（YAML/JSON）和函数式两种因子定义方式

**数据结构**:

```python
from enum import Enum
from typing import Callable

class FactorType(Enum):
    BUILTIN = "builtin"      # 内置因子
    YAML = "yaml"            # YAML 配置
    FUNCTION = "function"    # Python 函数


@dataclass
class Factor:
    """因子定义"""
    name: str
    description: str
    definition: Union[str, Callable]  # YAML 字符串或 Python 函数
    type: FactorType
    parameters: Dict[str, Any] = None  # 因子参数（如周期）

    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        """
        在数据上评估因子，返回信号序列
        """
        if self.type == FactorType.FUNCTION:
            return self.definition(data)
        elif self.type == FactorType.YAML:
            return self._evaluate_yaml(data)
        else:
            raise NotImplementedError
```

**配置化定义示例**:

```yaml
# 放量突破因子配置
name: "volume_breakout"
description: "收盘价突破前高，且成交量放大1.5倍以上"
type: "yaml"
parameters:
  lookback: 20      # 前高回看周期
  volume_mult: 1.5  # 成交量倍数阈值
conditions:
  - indicator: close
    operator: ">"
    value: "rolling_high(high, lookback).shift(1)"
  - indicator: volume
    operator: ">"
    value: "sma(volume, 20) * volume_mult"
```

**函数式定义示例**:

```python
def macd_zero_cross_factor(df: pd.DataFrame) -> pd.Series:
    """
    MACD 零轴下金叉因子
    """
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9).mean()

    golden_cross = (dif > dea) & (dif.shift(1) <= dea.shift(1))
    below_zero = (dif < 0) & (dea < 0)

    return golden_cross & below_zero

# 注册因子
factor = FactorRegistry.create_from_function(
    macd_zero_cross_factor,
    name="macd_zero_cross",
    description="MACD 零轴下金叉"
)
```

**内置因子库**:

```python
class FactorRegistry:
    """因子注册表"""

    BUILTIN_FACTORS = {
        # 趋势指标
        "macd_golden_cross": Factor(...),
        "macd_death_cross": Factor(...),
        "golden_cross": Factor(...),  # 均线金叉
        "death_cross": Factor(...),

        # 动量指标
        "rsi_oversold": Factor(...),   # RSI < 30
        "rsi_overbought": Factor(...), # RSI > 70
        "kdj_golden_cross": Factor(...),

        # 波动率指标
        "bollinger_lower_break": Factor(...),
        "bollinger_upper_break": Factor(...),

        # 量价指标
        "volume_breakout": Factor(...),
        "volume_contraction": Factor(...),
    }

    @classmethod
    def list_builtin(cls) -> List[str]:
        """列出所有内置因子"""

    @classmethod
    def get(cls, name: str) -> Factor:
        """获取内置因子"""

    @classmethod
    def create_from_yaml(cls, yaml_str: str) -> Factor:
        """从 YAML 配置创建因子"""

    @classmethod
    def create_from_function(cls, func: Callable, name: str, **kwargs) -> Factor:
        """从 Python 函数创建因子"""
```

---

### 3.6 FactorFilter - 主控模块

**职责**: 串联四个阶段，提供统一的因子质检接口

**核心方法**:

```python
class FactorFilter:
    """蒙特卡洛因子质检系统主控"""

    def __init__(
        self,
        db_path: str = "tushare.db",
        n_simulations: int = 10000,
        simulation_days: int = 252,
        alpha_threshold: float = 1.5,
        window: int = 252
    ):
        """
        初始化所有组件
        """
        self.reader = DataReader(db_path)
        self.param_estimator = ParameterEstimator(self.reader, window)
        self.simulator = GBMSimulator(n_simulations, simulation_days)
        self.detector = SignalDetector()
        self.tester = SignificanceTester(alpha_threshold)

    def filter(
        self,
        factor: Union[str, Factor],
        ts_codes: List[str] = None,
        lookback_days: int = 504,
        parallel: bool = True
    ) -> FactorReport:
        """
        对因子进行完整质检流程

        Args:
            factor: 因子名称（内置）或 Factor 实例
            ts_codes: 股票代码列表，None 表示全市场
            lookback_days: 回看天数，用于参数估计
            parallel: 是否并行处理多只股票

        Returns:
            FactorReport 包含完整检验结果和报告
        """

    def batch_filter(
        self,
        factors: List[Union[str, Factor]],
        **kwargs
    ) -> List[FactorReport]:
        """
        批量质检多个因子
        """

    def benchmark_all_builtin(self, **kwargs) -> pd.DataFrame:
        """
        对系统内置的所有因子进行质检，输出对比表格
        """
```

**使用示例**:

```python
# 初始化质检系统
filter = FactorFilter(
    db_path="tushare.db",
    n_simulations=10000,
    alpha_threshold=1.5
)

# 质检单个内置因子
report = filter.filter("macd_golden_cross")
print(report.markdown)

# 质检自定义因子（配置化）
yaml_def = """
name: "my_factor"
conditions:
  - indicator: close
    operator: ">"
    value: "open * 1.03"
  - indicator: volume
    operator: ">"
    value: "volume.shift(1) * 1.5"
"""
factor = FactorRegistry.create_from_yaml(yaml_def)
report = filter.filter(factor)

# 质检自定义因子（函数式）
def my_factor(df):
    return (df.close > df.open * 1.03) & (df.volume > df.volume.shift(1) * 1.5)

factor = FactorRegistry.create_from_function(my_factor, "my_factor")
report = filter.filter(factor)

# 批量对比所有内置因子
results = filter.benchmark_all_builtin()
results.to_csv("factor_benchmark.csv")
```

---

## 4. 报告格式

### 4.1 单个因子报告

```markdown
# 因子质检报告

## 因子信息
| 属性 | 值 |
|------|-----|
| 名称 | MACD金叉 |
| 描述 | DIF上穿DEA，且DIF<0（零轴下金叉） |
| 类型 | 内置因子 |
| 测试标的数 | 5,123 |
| 回看天数 | 504 |
| 模拟路径数 | 10,000 |
| 模拟天数 | 252 |

## 统计结果汇总
| 统计项 | 中位数 | 均值 | 25分位 | 75分位 |
|--------|--------|------|--------|--------|
| P_actual | 2.34% | 2.41% | 1.87% | 2.89% |
| P_random | 1.87% | 1.92% | 1.45% | 2.34% |
| Alpha Ratio | 1.25 | 1.28 | 1.12 | 1.45 |
| 显著性 | - | 23% | - | - |

## 结论
**整体建议: 优化或废弃**

- Alpha Ratio 中位数为 1.25，未达到 1.5 的显著性阈值
- 仅 23% 的股票 Alpha Ratio > 1.5
- 该因子在真实市场中的触发与随机游走无显著差异

## 详细数据（Top 10）
| 股票代码 | 股票名称 | P_actual | P_random | Alpha Ratio | 建议 |
|----------|----------|----------|----------|-------------|------|
| 000001.SZ | 平安银行 | 3.12% | 1.45% | 2.15 | KEEP |
| 000002.SZ | 万科A | 2.89% | 1.67% | 1.73 | KEEP |
| ... | ... | ... | ... | ... | ... |

## 可视化
[Alpha Ratio 分布直方图]
[P_actual vs P_random 散点图]
```

### 4.2 批量对比报告

```markdown
# 因子批量质检对比报告

| 因子名称 | 类型 | P_actual | P_random | Alpha Ratio | 显著比例 | 建议 |
|----------|------|----------|----------|-------------|----------|------|
| 放量突破 | 量价 | 3.45% | 1.23% | 2.80 | 67% | 保留 |
| RSI超卖 | 动量 | 2.89% | 1.56% | 1.85 | 45% | 保留 |
| MACD金叉 | 趋势 | 2.34% | 1.87% | 1.25 | 23% | 优化 |
| 均线金叉 | 趋势 | 1.89% | 1.92% | 0.98 | 12% | 废弃 |

## 结论
- **推荐保留**: 放量突破、RSI超卖
- **建议优化**: MACD金叉（结合其他条件）
- **建议废弃**: 均线金叉（纯随机噪音）
```

---

## 5. 输出数据结构

```python
@dataclass
class FactorReport:
    """因子质检报告"""
    factor: Factor
    timestamp: datetime
    parameters: Dict[str, Any]  # 测试参数

    # 统计结果
    results: List[TestResult]

    # 汇总统计
    summary: Dict[str, float]  # p_actual_mean, p_random_mean, alpha_ratio_median, etc.

    # 报告
    markdown: str  # Markdown 格式报告
    dataframe: pd.DataFrame  # 详细数据表格

    def to_dict(self) -> Dict:
        """转换为字典，便于序列化"""

    def save(self, path: str):
        """保存报告到文件（Markdown + CSV）"""
```

---

## 6. 性能考虑

### 6.1 计算优化

| 优化点 | 策略 | 预期效果 |
|--------|------|----------|
| GBM 模拟 | NumPy 矩阵化，避免 Python 循环 | 10,000×252 矩阵 < 0.1s |
| 信号检测 | 向量化指标计算 | 比循环快 100x+ |
| 多股票处理 | ProcessPoolExecutor 并行 | 利用多核 CPU |
| 内存管理 | 分批次处理，避免全量加载 | 支持全市场 5000+ 股票 |

### 6.2 预估耗时

- 单股票（10,000 路径 × 252 天）: ~0.5s
- 全市场（5,000 股票）串行: ~40 分钟
- 全市场并行（8 核）: ~5 分钟

---

## 7. 接口集成

### 7.1 与 Qlib 集成

```python
# 作为 Qlib 的因子过滤器
from qlib.data.dataset.loader import QlibDataLoader

class QlibFactorFilter:
    """Qlib 集成的因子质检包装器"""

    def __init__(self, factor_filter: FactorFilter):
        self.filter = factor_filter

    def validate_features(self, feature_names: List[str]) -> List[str]:
        """
        对 Qlib 特征进行质检，返回通过检验的特征名
        """
        reports = self.filter.batch_filter(feature_names)
        return [r.factor.name for r in reports if r.summary['alpha_ratio_median'] > 1.5]
```

### 7.2 策略流水线接入

```python
# 策略流水线中的因子质检步骤
class FactorPipeline:
    def __init__(self):
        self.filter = FactorFilter()
        self.model = SomeMLModel()

    def train(self, factors: List[Factor]):
        # 步骤1: 因子质检
        reports = self.filter.batch_filter(factors)
        valid_factors = [r.factor for r in reports if r.is_significant]

        # 步骤2: 仅使用通过质检的因子训练模型
        self.model.train(valid_factors)
```

---

## 8. 待决策事项

1. **是否需要支持分钟级数据？** 当前设计仅支持日线，如需分钟级需要调整 GBM 时间步长
2. **是否需要支持多因子组合检验？** 当前设计为单因子检验，组合检验需要调整 P_actual 计算方式
3. **是否需要支持持仓周期收益检验？** 当前设计仅检验信号触发概率，可扩展至检验触发后的收益分布

---

## 9. 目录结构

```
src/tushare_db/
├── factor_validation/              # 新增: 因子质检模块
│   ├── __init__.py
│   ├── filter.py                   # FactorFilter 主控
│   ├── estimator.py                # ParameterEstimator
│   ├── simulator.py                # GBMSimulator
│   ├── detector.py                 # SignalDetector
│   ├── tester.py                   # SignificanceTester
│   ├── factor.py                   # Factor 定义和注册表
│   ├── builtin_factors.py          # 内置因子库
│   └── report.py                   # 报告生成
│
├── notebooks/                      # 新增: 示例笔记本
│   └── factor_validation_demo.ipynb
│
└── tests/
    └── factor_validation/          # 新增: 测试用例
        ├── test_simulator.py
        ├── test_detector.py
        └── test_integration.py
```

---

## 10. 下一步行动

1. **实现核心模块**: GBMSimulator → SignalDetector → FactorFilter
2. **添加内置因子**: MACD、RSI、布林带、均线等
3. **编写示例**: 展示如何检验自定义因子
4. **性能测试**: 全市场批量质检性能基准
5. **文档完善**: API 文档和使用指南
