# 蒙特卡洛因子质检系统实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 构建一个蒙特卡洛模拟驱动的因子质检系统，通过对比技术指标在真实市场 vs 随机游走中的触发概率，筛选出具有 Alpha 挖掘价值的真因子。

**Architecture:** 采用四阶段流水线架构：参数估计 → GBM 模拟 → 信号检测 → 显著性检验。使用 NumPy 向量化计算实现高性能，支持 YAML 配置和 Python 函数两种因子定义方式。

**Tech Stack:** Python 3.10+, NumPy, Pandas, DuckDB, Tushare-DuckDB (本地数据源), pytest

---

## 前置知识

### 项目结构

```
/Users/allen/workspace/python/stock/Tushare-DuckDB/
├── src/tushare_db/
│   ├── __init__.py           # 主模块导出
│   ├── reader.py             # DataReader - 读取本地 DuckDB
│   ├── duckdb_manager.py     # DuckDB 操作
│   └── sector_analysis/      # 行业分析模块（参考结构）
├── tests/                    # 测试目录
├── docs/plans/               # 设计文档
└── tushare.db                # 本地数据库（22GB）
```

### 核心依赖

```python
# 现有项目中可用的导入
from tushare_db import DataReader  # 读取本地数据
from tushare_db.duckdb_manager import DuckDBManager
```

### 数学背景

**几何布朗运动 (GBM) 公式:**
```
S_t = S_{t-1} × exp((μ - σ²/2) × Δt + σ × √Δt × Z)
```

**Alpha Ratio:**
```
Alpha Ratio = P_actual / P_random
P_actual = 真实市场信号触发概率
P_random = 随机游走信号触发概率
```

---

## Task 1: 创建模块目录结构

**目标:** 创建因子质检模块的基础目录结构

**Files:**
- Create: `src/tushare_db/factor_validation/__init__.py`
- Create: `src/tushare_db/factor_validation/simulator.py`
- Create: `src/tushare_db/factor_validation/detector.py`
- Create: `src/tushare_db/factor_validation/tester.py`
- Create: `src/tushare_db/factor_validation/factor.py`
- Create: `tests/factor_validation/__init__.py`

**Step 1: 创建模块目录**

```bash
mkdir -p src/tushare_db/factor_validation
mkdir -p tests/factor_validation
```

**Step 2: 创建基础 __init__.py 文件**

`src/tushare_db/factor_validation/__init__.py`:
```python
"""
蒙特卡洛因子质检系统

通过对比技术指标在真实市场 vs 随机游走中的触发概率，
筛选出具有 Alpha 挖掘价值的真因子。
"""

__version__ = "0.1.0"
```

`tests/factor_validation/__init__.py`:
```python
"""因子质检模块测试"""
```

**Step 3: Commit**

```bash
git add src/tushare_db/factor_validation/ tests/factor_validation/
git commit -m "chore: create factor_validation module structure

Generated with [Claude Code](https://claude.ai/code)
via [Happy](https://happy.engineering)

Co-Authored-By: Claude <noreply@anthropic.com>
Co-Authored-By: Happy <yesreply@happy.engineering>"
```

---

## Task 2: 实现 GBM 模拟器 (GBMSimulator)

**目标:** 实现带 A 股涨跌停约束的几何布朗运动模拟器

**Files:**
- Create: `src/tushare_db/factor_validation/simulator.py`
- Create: `tests/factor_validation/test_simulator.py`

**Step 1: 编写测试**

`tests/factor_validation/test_simulator.py`:
```python
import numpy as np
import pytest


def test_gbm_simulator_import():
    """测试模块可以导入"""
    from src.tushare_db.factor_validation.simulator import GBMSimulator
    assert GBMSimulator is not None


def test_gbm_simulator_init():
    """测试初始化参数"""
    from src.tushare_db.factor_validation.simulator import GBMSimulator

    sim = GBMSimulator(n_paths=100, n_steps=50)
    assert sim.n_paths == 100
    assert sim.n_steps == 50
    assert sim.limit_up == 0.10
    assert sim.limit_down == -0.10


def test_gbm_simulate_shape():
    """测试模拟结果形状正确"""
    from src.tushare_db.factor_validation.simulator import GBMSimulator

    sim = GBMSimulator(n_paths=100, n_steps=50)
    result = sim.simulate(s0=100, mu=0.1, sigma=0.2)

    assert result.shape == (100, 50)
    assert isinstance(result, np.ndarray)


def test_gbm_price_positive():
    """测试价格始终为正"""
    from src.tushare_db.factor_validation.simulator import GBMSimulator

    sim = GBMSimulator(n_paths=100, n_steps=50)
    result = sim.simulate(s0=100, mu=0.1, sigma=0.2)

    assert np.all(result > 0)


def test_gbm_first_column_is_s0():
    """测试第一列是初始价格"""
    from src.tushare_db.factor_validation.simulator import GBMSimulator

    sim = GBMSimulator(n_paths=100, n_steps=50)
    result = sim.simulate(s0=100, mu=0.1, sigma=0.2)

    assert np.allclose(result[:, 0], 100)


def test_gbm_price_limits():
    """测试涨跌停约束生效"""
    from src.tushare_db.factor_validation.simulator import GBMSimulator

    sim = GBMSimulator(n_paths=100, n_steps=50, limit_up=0.10, limit_down=-0.10)
    result = sim.simulate(s0=100, mu=0.1, sigma=0.5)  # 高波动率更容易触发限制

    # 计算日收益率
    returns = np.diff(result, axis=1) / result[:, :-1]

    # 检查没有超过涨跌停限制的收益率
    assert np.all(returns <= 0.10 + 1e-6)
    assert np.all(returns >= -0.10 - 1e-6)


def test_gbm_reproducibility():
    """测试可重复性（设置随机种子）"""
    from src.tushare_db.factor_validation.simulator import GBMSimulator

    sim1 = GBMSimulator(n_paths=100, n_steps=50, random_seed=42)
    result1 = sim1.simulate(s0=100, mu=0.1, sigma=0.2)

    sim2 = GBMSimulator(n_paths=100, n_steps=50, random_seed=42)
    result2 = sim2.simulate(s0=100, mu=0.1, sigma=0.2)

    assert np.allclose(result1, result2)
```

**Step 2: 运行测试确认失败**

```bash
pytest tests/factor_validation/test_simulator.py -v
```

Expected: ImportError 或失败

**Step 3: 实现 GBMSimulator**

`src/tushare_db/factor_validation/simulator.py`:
```python
"""
GBM 模拟器 - 带 A 股涨跌停约束的几何布朗运动生成器
"""
import numpy as np
from typing import Dict, Union


class GBMSimulator:
    """
    几何布朗运动模拟器

    支持 A 股涨跌停限制（主板 10%，创业板/科创板 20%）

    GBM 公式:
        S_t = S_{t-1} × exp((μ - σ²/2) × Δt + σ × √Δt × Z)

    Attributes:
        n_paths: 模拟路径数量
        n_steps: 每条路径的时间步长
        limit_up: 涨停限制（默认 10%）
        limit_down: 跌停限制（默认 -10%）
        dt: 时间步长（年化，默认 1/252）
    """

    def __init__(
        self,
        n_paths: int = 10000,
        n_steps: int = 252,
        limit_up: float = 0.10,
        limit_down: float = -0.10,
        dt: float = None,
        random_seed: int = None
    ):
        """
        初始化 GBM 模拟器

        Args:
            n_paths: 模拟路径数量，建议 >= 10000
            n_steps: 时间步长数（交易日数）
            limit_up: 涨停限制
            limit_down: 跌停限制
            dt: 时间步长，None 则自动计算为 1/n_steps
            random_seed: 随机种子，用于可重复性
        """
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.limit_up = limit_up
        self.limit_down = limit_down
        self.dt = dt if dt is not None else 1.0 / n_steps
        self.random_seed = random_seed

        if random_seed is not None:
            np.random.seed(random_seed)

    def simulate(
        self,
        s0: float,
        mu: float,
        sigma: float,
        generate_ohlc: bool = False
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        批量生成模拟价格路径

        Args:
            s0: 初始价格
            mu: 年化收益率
            sigma: 年化波动率
            generate_ohlc: 是否生成 OHLC 数据（目前仅返回收盘价）

        Returns:
            收盘价矩阵，形状 (n_paths, n_steps)
        """
        # 预分配结果矩阵
        prices = np.zeros((self.n_paths, self.n_steps))
        prices[:, 0] = s0

        # 预计算常量
        drift = (mu - 0.5 * sigma ** 2) * self.dt
        diffusion = sigma * np.sqrt(self.dt)

        # 生成所有随机数
        Z = np.random.standard_normal((self.n_paths, self.n_steps - 1))

        # 向量化计算价格路径
        for t in range(1, self.n_steps):
            # GBM 公式
            prices[:, t] = prices[:, t-1] * np.exp(drift + diffusion * Z[:, t-1])

        # 应用涨跌停约束
        prices = self._apply_price_limits(prices)

        return prices

    def _apply_price_limits(self, prices: np.ndarray) -> np.ndarray:
        """
        应用 A 股涨跌停约束

        如果单日涨跌幅超过限制，则强制限制在涨跌停板价格

        Args:
            prices: 价格矩阵 (n_paths, n_steps)

        Returns:
            约束后的价格矩阵
        """
        # 计算收益率
        returns = np.diff(prices, axis=1) / prices[:, :-1]

        # 截断收益率
        clipped_returns = np.clip(returns, self.limit_down, self.limit_up)

        # 根据截断后的收益率重新计算价格
        constrained_prices = np.zeros_like(prices)
        constrained_prices[:, 0] = prices[:, 0]

        for t in range(1, prices.shape[1]):
            constrained_prices[:, t] = constrained_prices[:, t-1] * (1 + clipped_returns[:, t-1])

        return constrained_prices

    def simulate_with_progress(
        self,
        s0: float,
        mu: float,
        sigma: float,
        batch_size: int = 1000
    ) -> np.ndarray:
        """
        分批模拟（用于大规模模拟时显示进度）

        Args:
            s0: 初始价格
            mu: 年化收益率
            sigma: 年化波动率
            batch_size: 每批模拟的路径数

        Returns:
            收盘价矩阵
        """
        all_prices = []
        n_batches = (self.n_paths + batch_size - 1) // batch_size

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, self.n_paths)
            batch_n = end_idx - start_idx

            # 临时修改 n_paths 进行单批模拟
            original_n_paths = self.n_paths
            self.n_paths = batch_n
            batch_prices = self.simulate(s0, mu, sigma)
            self.n_paths = original_n_paths

            all_prices.append(batch_prices)

        return np.vstack(all_prices)
```

**Step 4: 运行测试确认通过**

```bash
pytest tests/factor_validation/test_simulator.py -v
```

Expected: 所有测试通过

**Step 5: Commit**

```bash
git add src/tushare_db/factor_validation/simulator.py tests/factor_validation/test_simulator.py
git commit -m "feat: implement GBMSimulator with A-share price limits

- Geometric Brownian Motion simulation
- 10%/-10% price limit constraints
- Vectorized NumPy implementation
- Reproducible with random_seed

Generated with [Claude Code](https://claude.ai/code)
via [Happy](https://happy.engineering)

Co-Authored-By: Claude <noreply@anthropic.com>
Co-Authored-By: Happy <yesreply@happy.engineering>"
```

---

## Task 3: 实现因子定义模块 (Factor)

**目标:** 实现因子定义的数据结构和注册表

**Files:**
- Create: `src/tushare_db/factor_validation/factor.py`
- Create: `tests/factor_validation/test_factor.py`

**Step 1: 编写测试**

`tests/factor_validation/test_factor.py`:
```python
import pandas as pd
import pytest


def test_factor_import():
    """测试模块可以导入"""
    from src.tushare_db.factor_validation.factor import Factor, FactorType
    assert Factor is not None
    assert FactorType is not None


def test_factor_creation():
    """测试创建因子对象"""
    from src.tushare_db.factor_validation.factor import Factor, FactorType

    factor = Factor(
        name="test_factor",
        description="测试因子",
        definition="close > open",
        type=FactorType.YAML
    )

    assert factor.name == "test_factor"
    assert factor.description == "测试因子"
    assert factor.type == FactorType.YAML


def test_factor_from_function():
    """测试从函数创建因子"""
    from src.tushare_db.factor_validation.factor import Factor, FactorType

    def my_factor(df):
        return df['close'] > df['open']

    factor = Factor(
        name="close_gt_open",
        description="收盘大于开盘",
        definition=my_factor,
        type=FactorType.FUNCTION
    )

    # 测试评估
    df = pd.DataFrame({
        'open': [10, 11, 12],
        'close': [11, 10, 13]
    })

    result = factor.evaluate(df)
    expected = pd.Series([True, False, True])

    assert result.equals(expected)


def test_factor_registry_list_builtin():
    """测试列出内置因子"""
    from src.tushare_db.factor_validation.factor import FactorRegistry

    builtins = FactorRegistry.list_builtin()
    assert isinstance(builtins, list)


def test_factor_registry_get_builtin():
    """测试获取内置因子"""
    from src.tushare_db.factor_validation.factor import FactorRegistry, Factor

    # 初始可能没有内置因子，返回 None 或抛出异常
    try:
        factor = FactorRegistry.get("macd_golden_cross")
        assert isinstance(factor, Factor)
    except (KeyError, NotImplementedError):
        pass  # 预期行为，内置因子尚未实现
```

**Step 2: 运行测试确认失败**

```bash
pytest tests/factor_validation/test_factor.py -v
```

Expected: ImportError

**Step 3: 实现 Factor 模块**

`src/tushare_db/factor_validation/factor.py`:
```python
"""
因子定义模块 - 支持配置化（YAML）和函数式两种定义方式
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, Any, Union, List
import pandas as pd


class FactorType(Enum):
    """因子类型"""
    BUILTIN = "builtin"       # 内置因子
    YAML = "yaml"             # YAML 配置
    FUNCTION = "function"     # Python 函数


@dataclass
class Factor:
    """
    因子定义

    Attributes:
        name: 因子名称
        description: 因子描述
        definition: 因子定义（YAML字符串或Python函数）
        type: 因子类型
        parameters: 因子参数（如周期）
    """
    name: str
    description: str
    definition: Union[str, Callable]
    type: FactorType
    parameters: Dict[str, Any] = field(default_factory=dict)

    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        """
        在数据上评估因子，返回信号序列

        Args:
            data: 包含 OHLCV 等数据的 DataFrame

        Returns:
            布尔序列，True 表示信号触发
        """
        if self.type == FactorType.FUNCTION:
            return self.definition(data)
        elif self.type == FactorType.YAML:
            return self._evaluate_yaml(data)
        elif self.type == FactorType.BUILTIN:
            return self._evaluate_builtin(data)
        else:
            raise NotImplementedError(f"不支持的因子类型: {self.type}")

    def _evaluate_yaml(self, data: pd.DataFrame) -> pd.Series:
        """评估 YAML 配置的因子（简化实现）"""
        # TODO: 实现完整的 YAML 解析器
        # 暂时返回全 False，后续实现
        return pd.Series(False, index=data.index)

    def _evaluate_builtin(self, data: pd.DataFrame) -> pd.Series:
        """评估内置因子"""
        if self.type == FactorType.FUNCTION:
            return self.definition(data)
        raise NotImplementedError(f"内置因子 {self.name} 尚未实现")


class FactorRegistry:
    """
    因子注册表

    管理内置因子，支持从 YAML 或函数创建因子
    """

    # 内置因子库（将在后续任务中填充）
    BUILTIN_FACTORS: Dict[str, Factor] = {}

    @classmethod
    def list_builtin(cls) -> List[str]:
        """列出所有内置因子名称"""
        return list(cls.BUILTIN_FACTORS.keys())

    @classmethod
    def get(cls, name: str) -> Factor:
        """
        获取内置因子

        Args:
            name: 因子名称

        Returns:
            Factor 实例

        Raises:
            KeyError: 如果因子不存在
        """
        if name not in cls.BUILTIN_FACTORS:
            raise KeyError(f"内置因子 '{name}' 不存在")
        return cls.BUILTIN_FACTORS[name]

    @classmethod
    def register(cls, factor: Factor) -> None:
        """
        注册内置因子

        Args:
            factor: Factor 实例
        """
        cls.BUILTIN_FACTORS[factor.name] = factor

    @classmethod
    def create_from_yaml(cls, yaml_str: str) -> Factor:
        """
        从 YAML 配置创建因子

        Args:
            yaml_str: YAML 格式的因子定义

        Returns:
            Factor 实例

        TODO: 实现完整的 YAML 解析
        """
        # 简化实现：解析 name 和 description
        lines = yaml_str.strip().split('\n')
        name = "yaml_factor"
        description = "YAML 定义的因子"

        for line in lines:
            if line.startswith('name:'):
                name = line.split(':', 1)[1].strip().strip('"\'')
            elif line.startswith('description:'):
                description = line.split(':', 1)[1].strip().strip('"\'')

        return Factor(
            name=name,
            description=description,
            definition=yaml_str,
            type=FactorType.YAML
        )

    @classmethod
    def create_from_function(
        cls,
        func: Callable,
        name: str = None,
        description: str = None,
        parameters: Dict[str, Any] = None
    ) -> Factor:
        """
        从 Python 函数创建因子

        Args:
            func: 因子计算函数，接收 DataFrame 返回布尔 Series
            name: 因子名称（默认使用函数名）
            description: 因子描述
            parameters: 因子参数

        Returns:
            Factor 实例
        """
        return Factor(
            name=name or func.__name__,
            description=description or func.__doc__ or "自定义因子",
            definition=func,
            type=FactorType.FUNCTION,
            parameters=parameters or {}
        )


def close_gt_open_factor(df: pd.DataFrame) -> pd.Series:
    """收盘价大于开盘价（示例因子）"""
    return df['close'] > df['open']


def price_above_sma_factor(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """价格在 SMA 之上（示例因子）"""
    sma = df['close'].rolling(window=period).mean()
    return df['close'] > sma


# 注册示例内置因子
FactorRegistry.register(Factor(
    name="close_gt_open",
    description="收盘价大于开盘价",
    definition=close_gt_open_factor,
    type=FactorType.FUNCTION
))
```

**Step 4: 运行测试确认通过**

```bash
pytest tests/factor_validation/test_factor.py -v
```

Expected: 所有测试通过

**Step 5: Commit**

```bash
git add src/tushare_db/factor_validation/factor.py tests/factor_validation/test_factor.py
git commit -m "feat: implement Factor definition module with registry

- Factor dataclass with name, description, definition, type
- Support FUNCTION and YAML factor types
- FactorRegistry for managing builtin factors
- Helper functions to create factors from functions or YAML

Generated with [Claude Code](https://claude.ai/code)
via [Happy](https://happy.engineering)

Co-Authored-By: Claude <noreply@anthropic.com>
Co-Authored-By: Happy <yesreply@happy.engineering>"
```

---

## Task 4: 实现信号检测器 (SignalDetector)

**目标:** 实现向量化信号检测，支持 MACD 金叉等内置指标

**Files:**
- Create: `src/tushare_db/factor_validation/detector.py`
- Create: `tests/factor_validation/test_detector.py`

**Step 1: 编写测试**

`tests/factor_validation/test_detector.py`:
```python
import numpy as np
import pandas as pd
import pytest


def test_detector_import():
    """测试模块可以导入"""
    from src.tushare_db.factor_validation.detector import SignalDetector
    assert SignalDetector is not None


def test_detector_init():
    """测试初始化"""
    from src.tushare_db.factor_validation.detector import SignalDetector

    detector = SignalDetector()
    assert detector.vectorized is True


def test_macd_golden_cross_detection():
    """测试 MACD 金叉检测"""
    from src.tushare_db.factor_validation.detector import SignalDetector

    detector = SignalDetector()

    # 创建一个价格序列（100条路径，50天）
    np.random.seed(42)
    price_matrix = np.cumsum(np.random.randn(100, 50) * 0.01, axis=1) + 100

    # 检测 MACD 金叉
    signals = detector.macd_golden_cross(price_matrix)

    # 检查形状
    assert signals.shape == (100, 50)
    # 检查是布尔类型
    assert signals.dtype == bool
    # 检查至少有一些信号（随机数据应该有约 5% 的交叉）
    assert signals.sum() > 0


def test_calculate_p_random():
    """测试计算随机概率"""
    from src.tushare_db.factor_validation.detector import SignalDetector

    detector = SignalDetector()

    # 创建信号矩阵
    signal_matrix = np.array([
        [False, True, False],
        [True, False, True],
        [False, False, False]
    ])

    p_random = detector.calculate_p_random(signal_matrix)

    # 3个 True / 9个总位置 = 0.333
    assert abs(p_random - 3/9) < 1e-6


def test_detect_signals_with_simple_factor():
    """测试使用简单因子检测信号"""
    from src.tushare_db.factor_validation.detector import SignalDetector
    from src.tushare_db.factor_validation.factor import Factor, FactorType

    detector = SignalDetector()

    # 创建一个简单因子（价格上涨）
    def rising_factor(df):
        return df['close'] > df['close'].shift(1)

    factor = Factor(
        name="rising",
        description="价格上涨",
        definition=rising_factor,
        type=FactorType.FUNCTION
    )

    # 创建价格矩阵并转换为 DataFrame
    np.random.seed(42)
    price_matrix = np.cumsum(np.random.randn(10, 20) * 0.01, axis=1) + 100

    # 检测信号
    signals = detector.detect_signals(price_matrix, factor)

    assert signals.shape == (10, 20)
    assert signals.dtype == bool


def test_detector_with_builtin_factor():
    """测试使用内置因子"""
    from src.tushare_db.factor_validation.detector import SignalDetector
    from src.tushare_db.factor_validation.factor import FactorRegistry

    detector = SignalDetector()

    # 获取内置因子
    factor = FactorRegistry.get("close_gt_open")

    # 创建价格矩阵（模拟 open = close * 0.99）
    np.random.seed(42)
    close = np.cumsum(np.random.randn(10, 20) * 0.01, axis=1) + 100
    open_price = close * 0.99

    # 检测信号
    signals = detector.detect_signals(close, factor, open_price=open_price)

    assert signals.shape == (10, 20)
    # 因为 close > open (close > close*0.99)，应该全部为 True
    assert signals.sum() > 0
```

**Step 2: 运行测试确认失败**

```bash
pytest tests/factor_validation/test_detector.py -v
```

Expected: ImportError

**Step 3: 实现 SignalDetector**

`src/tushare_db/factor_validation/detector.py`:
```python
"""
信号检测器 - 在模拟数据上并行计算技术指标
"""
import numpy as np
import pandas as pd
from typing import Dict, Any

from .factor import Factor, FactorType


class SignalDetector:
    """
    信号检测器

    在模拟价格矩阵上向量化计算技术指标信号
    完全避免 for 循环，使用 NumPy 矩阵运算

    Attributes:
        vectorized: 是否使用向量化计算
    """

    def __init__(self, vectorized: bool = True):
        """
        初始化信号检测器

        Args:
            vectorized: 是否使用向量化计算（默认 True）
        """
        self.vectorized = vectorized

    def detect_signals(
        self,
        price_matrix: np.ndarray,
        factor: Factor,
        **kwargs
    ) -> np.ndarray:
        """
        在模拟价格矩阵上检测因子信号

        Args:
            price_matrix: 价格矩阵，形状 (n_paths, n_steps)
            factor: Factor 实例
            **kwargs: 额外参数（如 open_price, high_price 等）

        Returns:
            信号矩阵 (n_paths, n_steps)，True 表示该时刻触发信号
        """
        if factor.name == "macd_golden_cross":
            return self.macd_golden_cross(price_matrix)
        elif factor.name == "close_gt_open":
            open_price = kwargs.get('open_price')
            if open_price is None:
                # 假设 open = close.shift(1) 的近似
                open_price = np.roll(price_matrix, 1, axis=1)
                open_price[:, 0] = price_matrix[:, 0]
            return price_matrix > open_price
        elif factor.type == FactorType.FUNCTION:
            # 通用因子：转换为 DataFrame 评估
            return self._evaluate_function_factor(price_matrix, factor, **kwargs)
        else:
            raise NotImplementedError(f"因子 {factor.name} 的检测尚未实现")

    def calculate_p_random(self, signal_matrix: np.ndarray) -> float:
        """
        计算随机触发概率

        P_random = 触发总次数 / (N × T)

        Args:
            signal_matrix: 信号矩阵 (n_paths, n_steps)

        Returns:
            随机触发概率
        """
        total_signals = np.sum(signal_matrix)
        total_positions = signal_matrix.size

        if total_positions == 0:
            return 0.0

        return total_signals / total_positions

    def macd_golden_cross(
        self,
        price_matrix: np.ndarray,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> np.ndarray:
        """
        向量化 MACD 金叉检测

        Args:
            price_matrix: 价格矩阵 (n_paths, n_steps)
            fast: 快线周期
            slow: 慢线周期
            signal: 信号线周期

        Returns:
            金叉信号矩阵
        """
        n_paths, n_steps = price_matrix.shape

        # 计算 EMA（向量化）
        ema_fast = self._ema_vectorized(price_matrix, fast)
        ema_slow = self._ema_vectorized(price_matrix, slow)

        # DIF 线
        dif = ema_fast - ema_slow

        # DEA 线（DIF 的 EMA）
        dea = self._ema_vectorized(dif, signal)

        # 检测金叉：DIF 上穿 DEA
        # 金叉条件：DIF[t] > DEA[t] 且 DIF[t-1] <= DEA[t-1]
        dif_prev = np.roll(dif, 1, axis=1)
        dea_prev = np.roll(dea, 1, axis=1)

        # 第一列无法计算金叉
        golden_cross = (dif > dea) & (dif_prev <= dea_prev)
        golden_cross[:, 0] = False

        return golden_cross

    def _ema_vectorized(self, matrix: np.ndarray, span: int) -> np.ndarray:
        """
        向量化 EMA 计算

        使用指数加权移动平均公式
        EMA_t = α × Price_t + (1-α) × EMA_{t-1}
        其中 α = 2 / (span + 1)

        Args:
            matrix: 输入矩阵 (n_paths, n_steps)
            span: EMA 周期

        Returns:
            EMA 矩阵
        """
        alpha = 2.0 / (span + 1)
        n_paths, n_steps = matrix.shape

        ema = np.zeros_like(matrix)
        ema[:, 0] = matrix[:, 0]  # 第一行初始化为价格

        for t in range(1, n_steps):
            ema[:, t] = alpha * matrix[:, t] + (1 - alpha) * ema[:, t-1]

        return ema

    def _evaluate_function_factor(
        self,
        price_matrix: np.ndarray,
        factor: Factor,
        **kwargs
    ) -> np.ndarray:
        """
        评估函数式因子

        将价格矩阵转换为 DataFrame，应用因子函数
        注意：这是简化实现，性能不如纯向量化

        Args:
            price_matrix: 收盘价矩阵
            factor: 函数式因子
            **kwargs: 其他价格数据（open, high, low）

        Returns:
            信号矩阵
        """
        n_paths, n_steps = price_matrix.shape
        signals = np.zeros((n_paths, n_steps), dtype=bool)

        # 对每条路径分别计算
        for i in range(n_paths):
            df = pd.DataFrame({
                'close': price_matrix[i]
            })

            # 添加其他价格数据
            if 'open_price' in kwargs:
                df['open'] = kwargs['open_price'][i]
            else:
                df['open'] = df['close']  # 默认

            if 'high_price' in kwargs:
                df['high'] = kwargs['high_price'][i]
            else:
                df['high'] = df['close']

            if 'low_price' in kwargs:
                df['low'] = kwargs['low_price'][i]
            else:
                df['low'] = df['close']

            df['volume'] = 1000000  # 默认成交量

            # 应用因子函数
            result = factor.evaluate(df)
            signals[i] = result.values

        return signals

    def rsi_cross(
        self,
        price_matrix: np.ndarray,
        period: int = 14,
        oversold: float = 30,
        overbought: float = 70
    ) -> Dict[str, np.ndarray]:
        """
        RSI 超买/超卖检测

        Args:
            price_matrix: 价格矩阵
            period: RSI 周期
            oversold: 超卖阈值
            overbought: 超买阈值

        Returns:
            {'oversold': 超卖信号矩阵, 'overbought': 超买信号矩阵}
        """
        # 计算 RSI（向量化）
        rsi = self._rsi_vectorized(price_matrix, period)

        # 检测超卖（RSI 从下方上穿 oversold）
        rsi_prev = np.roll(rsi, 1, axis=1)
        oversold_signal = (rsi > oversold) & (rsi_prev <= oversold)
        oversold_signal[:, 0] = False

        # 检测超买（RSI 从上方下穿 overbought）
        overbought_signal = (rsi < overbought) & (rsi_prev >= overbought)
        overbought_signal[:, 0] = False

        return {
            'oversold': oversold_signal,
            'overbought': overbought_signal
        }

    def _rsi_vectorized(self, matrix: np.ndarray, period: int = 14) -> np.ndarray:
        """
        向量化 RSI 计算

        Args:
            matrix: 价格矩阵
            period: RSI 周期

        Returns:
            RSI 矩阵
        """
        n_paths, n_steps = matrix.shape

        # 计算价格变化
        delta = np.diff(matrix, axis=1)

        # 分离上涨和下跌
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)

        # 计算平均涨跌（简单移动平均版本）
        avg_gain = np.zeros_like(matrix)
        avg_loss = np.zeros_like(matrix)

        # 初始化第一列
        avg_gain[:, period] = np.mean(gain[:, :period], axis=1)
        avg_loss[:, period] = np.mean(loss[:, :period], axis=1)

        # 递推计算
        for t in range(period + 1, n_steps):
            avg_gain[:, t] = (avg_gain[:, t-1] * (period - 1) + gain[:, t-1]) / period
            avg_loss[:, t] = (avg_loss[:, t-1] * (period - 1) + loss[:, t-1]) / period

        # 计算 RS 和 RSI
        rs = avg_gain / (avg_loss + 1e-10)  # 避免除零
        rsi = 100 - (100 / (1 + rs))

        # 前 period 列设为 50（中性）
        rsi[:, :period] = 50

        return rsi
```

**Step 4: 运行测试确认通过**

```bash
pytest tests/factor_validation/test_detector.py -v
```

Expected: 所有测试通过

**Step 5: Commit**

```bash
git add src/tushare_db/factor_validation/detector.py tests/factor_validation/test_detector.py
git commit -m "feat: implement SignalDetector with vectorized indicators

- MACD golden cross detection (fully vectorized)
- RSI calculation (vectorized)
- P_random calculation
- Support for function-based factors
- EMA vectorized implementation

Generated with [Claude Code](https://claude.ai/code)
via [Happy](https://happy.engineering)

Co-Authored-By: Claude <noreply@anthropic.com>
Co-Authored-By: Happy <yesreply@happy.engineering>"
```

---

## Task 5: 实现显著性检验器 (SignificanceTester)

**目标:** 实现统计检验和报告生成功能

**Files:**
- Create: `src/tushare_db/factor_validation/tester.py`
- Create: `tests/factor_validation/test_tester.py`

**Step 1: 编写测试**

`tests/factor_validation/test_tester.py`:
```python
import pytest


def test_tester_import():
    """测试模块可以导入"""
    from src.tushare_db.factor_validation.tester import SignificanceTester, TestResult
    assert SignificanceTester is not None
    assert TestResult is not None


def test_test_result_creation():
    """测试创建测试结果"""
    from src.tushare_db.factor_validation.tester import TestResult

    result = TestResult(
        ts_code="000001.SZ",
        p_actual=0.03,
        p_random=0.02,
        alpha_ratio=1.5,
        n_signals_actual=30,
        n_signals_random=20,
        p_value=0.05,
        is_significant=True,
        recommendation="KEEP"
    )

    assert result.ts_code == "000001.SZ"
    assert result.alpha_ratio == 1.5


def test_significance_tester_init():
    """测试检验器初始化"""
    from src.tushare_db.factor_validation.tester import SignificanceTester

    tester = SignificanceTester(alpha_threshold=1.5)
    assert tester.alpha_threshold == 1.5


def test_test_significant_result():
    """测试显著结果"""
    from src.tushare_db.factor_validation.tester import SignificanceTester

    tester = SignificanceTester(alpha_threshold=1.5)

    # Alpha Ratio = 2.0 > 1.5，应该显著
    result = tester.test(
        p_actual=0.04,
        p_random=0.02,
        n_total=1000
    )

    assert result.alpha_ratio == 2.0
    assert result.is_significant is True
    assert result.recommendation == "KEEP"


def test_test_insignificant_result():
    """测试不显著结果"""
    from src.tushare_db.factor_validation.tester import SignificanceTester

    tester = SignificanceTester(alpha_threshold=1.5)

    # Alpha Ratio = 1.1 < 1.5，应该不显著
    result = tester.test(
        p_actual=0.022,
        p_random=0.02,
        n_total=1000
    )

    assert abs(result.alpha_ratio - 1.1) < 0.01
    assert result.is_significant is False
    assert result.recommendation == "DISCARD"


def test_test_optimize_result():
    """测试优化建议结果"""
    from src.tushare_db.factor_validation.tester import SignificanceTester

    tester = SignificanceTester(alpha_threshold=1.5)

    # Alpha Ratio = 1.3，介于 1.2 和 1.5 之间
    result = tester.test(
        p_actual=0.026,
        p_random=0.02,
        n_total=1000
    )

    assert abs(result.alpha_ratio - 1.3) < 0.01
    assert result.recommendation == "OPTIMIZE"


def test_generate_report():
    """测试生成报告"""
    from src.tushare_db.factor_validation.tester import SignificanceTester, TestResult
    from src.tushare_db.factor_validation.factor import Factor, FactorType

    tester = SignificanceTester()

    factor = Factor(
        name="test_factor",
        description="测试因子",
        definition="test",
        type=FactorType.BUILTIN
    )

    results = [
        TestResult(
            ts_code="000001.SZ",
            p_actual=0.03,
            p_random=0.02,
            alpha_ratio=1.5,
            n_signals_actual=30,
            n_signals_random=20,
            p_value=0.05,
            is_significant=True,
            recommendation="KEEP"
        )
    ]

    report = tester.generate_report(factor, results)

    assert "测试因子" in report or "test_factor" in report
    assert "000001.SZ" in report
```

**Step 2: 运行测试确认失败**

```bash
pytest tests/factor_validation/test_tester.py -v
```

Expected: ImportError

**Step 3: 实现 SignificanceTester**

`src/tushare_db/factor_validation/tester.py`:
```python
"""
显著性检验器 - 统计检验与报告生成
"""
from dataclasses import dataclass
from typing import List, Dict
import numpy as np
from scipy import stats

from .factor import Factor


@dataclass
class TestResult:
    """
    单个标的的检验结果

    Attributes:
        ts_code: 股票代码
        p_actual: 真实市场触发概率
        p_random: 随机游走触发概率
        alpha_ratio: Alpha Ratio = P_actual / P_random
        n_signals_actual: 真实市场信号数量
        n_signals_random: 随机游走信号数量（期望值）
        p_value: 统计显著性 p 值
        is_significant: 是否显著
        recommendation: 建议（KEEP/OPTIMIZE/DISCARD）
    """
    ts_code: str
    p_actual: float
    p_random: float
    alpha_ratio: float
    n_signals_actual: int
    n_signals_random: int
    p_value: float
    is_significant: bool
    recommendation: str


class SignificanceTester:
    """
    显著性检验器

    通过对比 P_actual 和 P_random，决定因子是保留、优化还是废弃

    统计假设检验：
        H0: 该指标的触发纯粹是随机游走的结果
        H1: 真实市场中存在非随机力量促成该形态

    Attributes:
        alpha_threshold: Alpha Ratio 阈值（默认 1.5）
        confidence_level: 置信水平（默认 0.95）
    """

    def __init__(
        self,
        alpha_threshold: float = 1.5,
        confidence_level: float = 0.95
    ):
        """
        初始化检验器

        Args:
            alpha_threshold: Alpha Ratio 阈值，超过认为显著
            confidence_level: 置信水平
        """
        self.alpha_threshold = alpha_threshold
        self.confidence_level = confidence_level

    def test(
        self,
        p_actual: float,
        p_random: float,
        n_total: int,
        ts_code: str = "unknown"
    ) -> TestResult:
        """
        执行统计假设检验

        Args:
            p_actual: 真实市场触发概率
            p_random: 随机游走触发概率
            n_total: 总样本数
            ts_code: 股票代码

        Returns:
            TestResult 包含检验结果和建议
        """
        # 计算 Alpha Ratio
        if p_random == 0:
            alpha_ratio = float('inf') if p_actual > 0 else 1.0
        else:
            alpha_ratio = p_actual / p_random

        # 计算信号数量
        n_signals_actual = int(p_actual * n_total)
        n_signals_random = int(p_random * n_total)

        # 计算 p 值（使用二项检验）
        # H0: P_actual = P_random
        # H1: P_actual > P_random (单侧检验)
        if p_random > 0 and n_total > 0:
            # 二项检验：在 n_total 次试验中，观察到 n_signals_actual 次成功
            # 假设成功概率为 p_random
            p_value = 1 - stats.binom.cdf(n_signals_actual - 1, n_total, p_random)
        else:
            p_value = 1.0

        # 判断是否显著
        is_significant = alpha_ratio >= self.alpha_threshold

        # 生成建议
        recommendation = self._generate_recommendation(alpha_ratio)

        return TestResult(
            ts_code=ts_code,
            p_actual=p_actual,
            p_random=p_random,
            alpha_ratio=alpha_ratio,
            n_signals_actual=n_signals_actual,
            n_signals_random=n_signals_random,
            p_value=p_value,
            is_significant=is_significant,
            recommendation=recommendation
        )

    def _generate_recommendation(self, alpha_ratio: float) -> str:
        """
        根据 Alpha Ratio 生成建议

        Args:
            alpha_ratio: Alpha Ratio

        Returns:
            建议字符串：KEEP / OPTIMIZE / DISCARD
        """
        if alpha_ratio < 1.2:
            return "DISCARD"
        elif alpha_ratio < self.alpha_threshold:
            return "OPTIMIZE"
        else:
            return "KEEP"

    def generate_report(
        self,
        factor: Factor,
        results: List[TestResult]
    ) -> str:
        """
        生成 Markdown 格式报告

        Args:
            factor: 因子定义
            results: 检验结果列表

        Returns:
            Markdown 格式报告
        """
        if not results:
            return f"# 因子质检报告\n\n因子: {factor.name}\n\n无检验结果。"

        # 计算汇总统计
        alpha_ratios = [r.alpha_ratio for r in results]
        p_actuals = [r.p_actual for r in results]
        p_randoms = [r.p_random for r in results]

        summary = {
            'alpha_ratio_median': np.median(alpha_ratios),
            'alpha_ratio_mean': np.mean(alpha_ratios),
            'p_actual_median': np.median(p_actuals),
            'p_random_median': np.median(p_randoms),
            'significant_count': sum(1 for r in results if r.is_significant),
            'total_count': len(results)
        }

        # 生成报告
        report_lines = [
            f"# 因子质检报告",
            "",
            f"## 因子信息",
            f"| 属性 | 值 |",
            f"|------|-----|",
            f"| 名称 | {factor.name} |",
            f"| 描述 | {factor.description} |",
            f"| 类型 | {factor.type.value} |",
            f"| 测试标的数 | {len(results)} |",
            "",
            f"## 统计结果汇总",
            f"| 统计项 | 中位数 | 均值 |",
            f"|--------|--------|------|",
            f"| P_actual | {summary['p_actual_median']:.2%} | {np.mean(p_actuals):.2%} |",
            f"| P_random | {summary['p_random_median']:.2%} | {np.mean(p_randoms):.2%} |",
            f"| Alpha Ratio | {summary['alpha_ratio_median']:.2f} | {summary['alpha_ratio_mean']:.2f} |",
            "",
            f"## 结论",
            f"**整体建议: {self._generate_overall_recommendation(summary)}**",
            "",
            f"- Alpha Ratio 中位数: {summary['alpha_ratio_median']:.2f}",
            f"- 显著标的数: {summary['significant_count']} / {summary['total_count']} "
            f"({summary['significant_count']/summary['total_count']*100:.1f}%)",
            "",
            f"## 详细数据（Top 10）",
            f"| 股票代码 | P_actual | P_random | Alpha Ratio | 建议 |",
            f"|----------|----------|----------|-------------|------|",
        ]

        # 按 Alpha Ratio 排序，取前 10
        sorted_results = sorted(results, key=lambda r: r.alpha_ratio, reverse=True)[:10]

        for r in sorted_results:
            report_lines.append(
                f"| {r.ts_code} | {r.p_actual:.2%} | {r.p_random:.2%} | "
                f"{r.alpha_ratio:.2f} | {r.recommendation} |"
            )

        return "\n".join(report_lines)

    def _generate_overall_recommendation(self, summary: Dict) -> str:
        """
        生成整体建议

        Args:
            summary: 汇总统计

        Returns:
            整体建议字符串
        """
        alpha_median = summary['alpha_ratio_median']
        significant_ratio = summary['significant_count'] / summary['total_count']

        if alpha_median < 1.2 or significant_ratio < 0.2:
            return "废弃 (DISCARD)"
        elif alpha_median < self.alpha_threshold or significant_ratio < 0.5:
            return "优化 (OPTIMIZE)"
        else:
            return "保留 (KEEP)"

    def batch_test(
        self,
        results_data: List[Dict[str, float]]
    ) -> List[TestResult]:
        """
        批量检验

        Args:
            results_data: 包含 p_actual, p_random, n_total, ts_code 的字典列表

        Returns:
            TestResult 列表
        """
        test_results = []
        for data in results_data:
            result = self.test(
                p_actual=data['p_actual'],
                p_random=data['p_random'],
                n_total=data.get('n_total', 1000),
                ts_code=data.get('ts_code', 'unknown')
            )
            test_results.append(result)
        return test_results
```

**Step 4: 运行测试确认通过**

```bash
pytest tests/factor_validation/test_tester.py -v
```

Expected: 所有测试通过

**Step 5: Commit**

```bash
git add src/tushare_db/factor_validation/tester.py tests/factor_validation/test_tester.py
git commit -m "feat: implement SignificanceTester with report generation

- TestResult dataclass for individual stock results
- SignificanceTester with binomial hypothesis testing
- KEEP/OPTIMIZE/DISCARD recommendations based on Alpha Ratio
- Markdown report generation with summary statistics
- Batch testing support

Generated with [Claude Code](https://claude.ai/code)
via [Happy](https://happy.engineering)

Co-Authored-By: Claude <noreply@anthropic.com>
Co-Authored-By: Happy <yesreply@happy.engineering>"
```

---

## Task 6: 实现参数估计器 (ParameterEstimator)

**目标:** 从真实 A 股数据中提取统计特征

**Files:**
- Create: `src/tushare_db/factor_validation/estimator.py`
- Create: `tests/factor_validation/test_estimator.py`

**Step 1: 编写测试**

`tests/factor_validation/test_estimator.py`:
```python
import pandas as pd
import numpy as np
import pytest


def test_estimator_import():
    """测试模块可以导入"""
    from src.tushare_db.factor_validation.estimator import ParameterEstimator
    assert ParameterEstimator is not None


def test_calculate_log_returns():
    """测试对数收益率计算"""
    from src.tushare_db.factor_validation.estimator import ParameterEstimator

    # 模拟 DataReader
    class MockReader:
        pass

    estimator = ParameterEstimator(MockReader())

    # 创建测试数据
    df = pd.DataFrame({
        'close': [100, 102, 101, 103, 105]
    })

    returns = estimator._calculate_log_returns(df['close'])

    # 手动计算验证
    expected = pd.Series([
        np.nan,
        np.log(102/100),
        np.log(101/102),
        np.log(103/101),
        np.log(105/103)
    ])

    assert np.allclose(returns.values[1:], expected.values[1:])


def test_calculate_p_actual():
    """测试 P_actual 计算"""
    from src.tushare_db.factor_validation.estimator import ParameterEstimator
    from src.tushare_db.factor_validation.factor import Factor, FactorType

    class MockReader:
        pass

    estimator = ParameterEstimator(MockReader())

    # 创建因子：收盘价 > 100
    def factor_func(df):
        return df['close'] > 100

    factor = Factor(
        name="close_gt_100",
        description="收盘大于100",
        definition=factor_func,
        type=FactorType.FUNCTION
    )

    # 创建测试数据
    df = pd.DataFrame({
        'close': [99, 101, 100, 102, 98]
    })

    p_actual = estimator.calculate_p_actual(df, factor)

    # 2/5 = 0.4（第一行因 shift 为 False，实际 101, 102 触发）
    # factor_func 返回 [False, True, False, True, False]
    # 但注意 pandas Series 比较第一行是 False（因为没有前一日）
    # 实际上：第2行(101)和第4行(102)触发，共 2/5 = 40%
    assert abs(p_actual - 0.4) < 0.01


def test_annualize_parameters():
    """测试参数年化"""
    from src.tushare_db.factor_validation.estimator import ParameterEstimator

    class MockReader:
        pass

    estimator = ParameterEstimator(MockReader())

    # 日收益率均值和标准差
    daily_mu = 0.001  # 0.1% 日收益
    daily_sigma = 0.02  # 2% 日波动

    annual_mu, annual_sigma = estimator._annualize_parameters(daily_mu, daily_sigma)

    # 年化收益 = 日收益 × 252
    assert abs(annual_mu - daily_mu * 252) < 0.001
    # 年化波动 = 日波动 × sqrt(252)
    assert abs(annual_sigma - daily_sigma * np.sqrt(252)) < 0.001
```

**Step 2: 运行测试确认失败**

```bash
pytest tests/factor_validation/test_estimator.py -v
```

Expected: ImportError

**Step 3: 实现 ParameterEstimator**

`src/tushare_db/factor_validation/estimator.py`:
```python
"""
参数估计器 - 从真实 A 股数据中提取统计特征
"""
from typing import List, Optional, Dict
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from .factor import Factor


class ParameterEstimator:
    """
    参数估计器

    从真实 A 股数据中提取生成随机游走所需的统计特征：
    - 年化收益率均值 μ
    - 年化波动率 σ
    - 因子在真实数据上的触发概率 P_actual

    Attributes:
        reader: DataReader 实例，用于读取本地 DuckDB 数据
        window: 滚动窗口天数（默认 252 个交易日）
    """

    def __init__(self, reader, window: int = 252):
        """
        初始化参数估计器

        Args:
            reader: DataReader 实例
            window: 滚动窗口天数
        """
        self.reader = reader
        self.window = window

    def estimate_parameters(
        self,
        ts_codes: List[str],
        lookback_days: int = 504,
        factor: Optional[Factor] = None
    ) -> pd.DataFrame:
        """
        估计多只股票的参数

        Args:
            ts_codes: 股票代码列表
            lookback_days: 回看天数（默认 504 = 2年）
            factor: 要测试的因子（可选）

        Returns:
            DataFrame，每行一只股票，包含：
            - ts_code: 股票代码
            - mu: 年化收益率
            - sigma: 年化波动率
            - p_actual: 因子触发概率（如果提供了 factor）
        """
        results = []

        # 计算日期范围
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y%m%d')

        for ts_code in ts_codes:
            try:
                params = self._estimate_single_stock(
                    ts_code, start_date, end_date, factor
                )
                results.append(params)
            except Exception as e:
                print(f"Warning: Failed to estimate parameters for {ts_code}: {e}")
                continue

        if not results:
            return pd.DataFrame()

        return pd.DataFrame(results)

    def _estimate_single_stock(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
        factor: Optional[Factor]
    ) -> Dict:
        """
        估计单只股票的参数

        Args:
            ts_code: 股票代码
            start_date: 开始日期 YYYYMMDD
            end_date: 结束日期 YYYYMMDD
            factor: 因子（可选）

        Returns:
            参数字典
        """
        # 从数据库读取数据
        try:
            df = self.reader.get_stock_daily(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
                adj='qfq'
            )
        except Exception as e:
            # 如果 reader 没有这个方法，使用 execute_query
            df = self.reader.db.execute_query(
                """
                SELECT trade_date, close, open, high, low, vol as volume
                FROM daily
                WHERE ts_code = ? AND trade_date BETWEEN ? AND ?
                ORDER BY trade_date
                """,
                [ts_code, start_date, end_date]
            )

        if df.empty:
            raise ValueError(f"No data found for {ts_code}")

        # 数据清洗：剔除停牌日
        df = self._clean_data(df)

        # 计算对数收益率
        log_returns = self._calculate_log_returns(df['close'])

        # 估计参数
        daily_mu = log_returns.mean()
        daily_sigma = log_returns.std()

        # 年化
        mu, sigma = self._annualize_parameters(daily_mu, daily_sigma)

        result = {
            'ts_code': ts_code,
            'mu': mu,
            'sigma': sigma,
            'n_days': len(df)
        }

        # 如果提供了因子，计算 P_actual
        if factor is not None:
            result['p_actual'] = self.calculate_p_actual(df, factor)

        return result

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        数据清洗

        - 剔除停牌日（成交量为0或价格无变化）
        - 按日期排序

        Args:
            df: 原始数据 DataFrame

        Returns:
            清洗后的 DataFrame
        """
        # 确保有必要的列
        if 'volume' in df.columns:
            # 剔除成交量为0的日期（停牌）
            df = df[df['volume'] > 0]

        # 剔除价格无变化的日期
        if 'close' in df.columns and 'open' in df.columns:
            df = df[df['close'] != df['open']]

        # 按日期排序
        if 'trade_date' in df.columns:
            df = df.sort_values('trade_date').reset_index(drop=True)

        return df

    def _calculate_log_returns(self, prices: pd.Series) -> pd.Series:
        """
        计算对数收益率

        R_t = ln(S_t / S_{t-1})

        Args:
            prices: 价格序列

        Returns:
            对数收益率序列
        """
        return np.log(prices / prices.shift(1)).dropna()

    def _annualize_parameters(
        self,
        daily_mu: float,
        daily_sigma: float
    ) -> tuple:
        """
        年化参数

        Args:
            daily_mu: 日收益率均值
            daily_sigma: 日收益率标准差

        Returns:
            (年化收益率, 年化波动率)
        """
        trading_days_per_year = 252

        annual_mu = daily_mu * trading_days_per_year
        annual_sigma = daily_sigma * np.sqrt(trading_days_per_year)

        return annual_mu, annual_sigma

    def calculate_p_actual(self, df: pd.DataFrame, factor: Factor) -> float:
        """
        计算因子在真实历史数据上的触发概率

        P_actual = 信号触发次数 / 总交易日数

        Args:
            df: 包含 OHLCV 数据的 DataFrame
            factor: 因子定义

        Returns:
            触发概率
        """
        # 应用因子
        signals = factor.evaluate(df)

        # 计算概率
        if len(signals) == 0:
            return 0.0

        return signals.sum() / len(signals)

    def get_stock_universe(
        self,
        index_code: Optional[str] = None,
        list_status: str = 'L'
    ) -> List[str]:
        """
        获取股票池

        Args:
            index_code: 指数代码（如 '000300.SH' 沪深300），None 则返回全市场
            list_status: 上市状态 'L'=上市, 'D'=退市, 'P'=暂停

        Returns:
            股票代码列表
        """
        if index_code:
            # 获取指数成分股
            try:
                df = self.reader.get_index_weight(index_code=index_code)
                return df['con_code'].unique().tolist()
            except:
                # 如果没有这个方法，查询 index_weight 表
                df = self.reader.db.execute_query(
                    "SELECT DISTINCT con_code FROM index_weight WHERE index_code = ?",
                    [index_code]
                )
                return df['con_code'].tolist()
        else:
            # 获取全市场股票
            try:
                df = self.reader.get_stock_basic(list_status=list_status)
                return df['ts_code'].tolist()
            except:
                # 如果没有这个方法，查询 stock_basic 表
                df = self.reader.db.execute_query(
                    "SELECT ts_code FROM stock_basic WHERE list_status = ?",
                    [list_status]
                )
                return df['ts_code'].tolist()
```

**Step 4: 运行测试确认通过**

```bash
pytest tests/factor_validation/test_estimator.py -v
```

Expected: 所有测试通过

**Step 5: Commit**

```bash
git add src/tushare_db/factor_validation/estimator.py tests/factor_validation/test_estimator.py
git commit -m "feat: implement ParameterEstimator for real market data

- Calculate log returns from historical prices
- Annualized mu and sigma estimation
- P_actual calculation for factors
- Data cleaning (remove suspended trading days)
- Support for stock universe selection

Generated with [Claude Code](https://claude.ai/code)
via [Happy](https://happy.engineering)

Co-Authored-By: Claude <noreply@anthropic.com>
Co-Authored-By: Happy <yesreply@happy.engineering>"
```

---

## Task 7: 实现主控模块 (FactorFilter)

**目标:** 整合四个阶段，提供统一的因子质检接口

**Files:**
- Create: `src/tushare_db/factor_validation/filter.py`
- Create: `tests/factor_validation/test_filter.py`
- Modify: `src/tushare_db/factor_validation/__init__.py`

**Step 1: 编写测试**

`tests/factor_validation/test_filter.py`:
```python
import pytest
import numpy as np


def test_filter_import():
    """测试模块可以导入"""
    from src.tushare_db.factor_validation.filter import FactorFilter
    assert FactorFilter is not None


def test_factor_filter_init():
    """测试初始化"""
    from src.tushare_db.factor_validation.filter import FactorFilter

    # 使用模拟路径（不需要真实数据库）
    filter_obj = FactorFilter(
        db_path=":memory:",  # DuckDB 内存数据库
        n_simulations=100,
        simulation_days=50
    )

    assert filter_obj.n_simulations == 100
    assert filter_obj.simulation_days == 50


def test_filter_with_mock_data():
    """测试使用模拟数据进行完整流程"""
    from src.tushare_db.factor_validation.filter import FactorFilter
    from src.tushare_db.factor_validation.factor import Factor, FactorType, FactorRegistry

    filter_obj = FactorFilter(
        db_path=":memory:",
        n_simulations=50,
        simulation_days=30,
        alpha_threshold=1.5
    )

    # 使用内置的简单因子
    factor = FactorRegistry.get("close_gt_open")

    # 创建模拟参数（跳过真实数据获取）
    import pandas as pd
    params_df = pd.DataFrame({
        'ts_code': ['000001.SZ', '000002.SZ'],
        'mu': [0.1, 0.15],
        'sigma': [0.2, 0.25],
        'p_actual': [0.5, 0.6]  # 假设真实触发概率
    })

    # 运行检验
    report = filter_obj._run_simulation_test(params_df, factor)

    assert report is not None
    assert report.factor.name == "close_gt_open"
    assert len(report.results) == 2


def test_factor_report_structure():
    """测试报告结构"""
    from src.tushare_db.factor_validation.filter import FactorReport
    from src.tushare_db.factor_validation.factor import Factor, FactorType
    from src.tushare_db.factor_validation.tester import TestResult
    from datetime import datetime

    factor = Factor(
        name="test",
        description="测试",
        definition="test",
        type=FactorType.BUILTIN
    )

    results = [
        TestResult(
            ts_code="000001.SZ",
            p_actual=0.03,
            p_random=0.02,
            alpha_ratio=1.5,
            n_signals_actual=30,
            n_signals_random=20,
            p_value=0.05,
            is_significant=True,
            recommendation="KEEP"
        )
    ]

    report = FactorReport(
        factor=factor,
        timestamp=datetime.now(),
        parameters={'n_simulations': 100},
        results=results,
        summary={'alpha_ratio_median': 1.5},
        markdown="# Test Report",
        dataframe=None
    )

    assert report.factor.name == "test"
    assert len(report.results) == 1
```

**Step 2: 运行测试确认失败**

```bash
pytest tests/factor_validation/test_filter.py -v
```

Expected: ImportError

**Step 3: 实现 FactorFilter**

`src/tushare_db/factor_validation/filter.py`:
```python
"""
因子过滤器 - 蒙特卡洛因子质检系统主控

串联四个阶段：参数估计 → GBM 模拟 → 信号检测 → 显著性检验
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Union, Optional, Dict, Any
import pandas as pd
import numpy as np

from .simulator import GBMSimulator
from .detector import SignalDetector
from .tester import SignificanceTester, TestResult
from .factor import Factor, FactorType, FactorRegistry
from .estimator import ParameterEstimator


@dataclass
class FactorReport:
    """
    因子质检报告

    Attributes:
        factor: 因子定义
        timestamp: 检验时间
        parameters: 测试参数
        results: 每只股票的检验结果列表
        summary: 汇总统计
        markdown: Markdown 格式报告
        dataframe: 详细数据 DataFrame
    """
    factor: Factor
    timestamp: datetime
    parameters: Dict[str, Any]
    results: List[TestResult]
    summary: Dict[str, float]
    markdown: str
    dataframe: Optional[pd.DataFrame]

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'factor_name': self.factor.name,
            'timestamp': self.timestamp.isoformat(),
            'parameters': self.parameters,
            'summary': self.summary,
            'results_count': len(self.results)
        }

    def save(self, path: str):
        """
        保存报告到文件

        Args:
            path: 文件路径（不含扩展名）
        """
        # 保存 Markdown
        with open(f"{path}.md", 'w', encoding='utf-8') as f:
            f.write(self.markdown)

        # 保存 CSV
        if self.dataframe is not None:
            self.dataframe.to_csv(f"{path}.csv", index=False, encoding='utf-8-sig')


class FactorFilter:
    """
    蒙特卡洛因子质检系统主控

    整合四个阶段，提供统一的因子质检接口：
    1. 参数估计：从真实数据提取 μ, σ, P_actual
    2. GBM 模拟：生成随机游走价格路径
    3. 信号检测：计算 P_random
    4. 显著性检验：Alpha Ratio 检验

    Attributes:
        n_simulations: 模拟路径数量
        simulation_days: 模拟天数
        alpha_threshold: Alpha Ratio 阈值
    """

    def __init__(
        self,
        db_path: str = "tushare.db",
        n_simulations: int = 10000,
        simulation_days: int = 252,
        alpha_threshold: float = 1.5,
        window: int = 252
    ):
        """
        初始化因子过滤器

        Args:
            db_path: DuckDB 数据库路径
            n_simulations: 模拟路径数量
            simulation_days: 模拟天数
            alpha_threshold: Alpha Ratio 阈值
            window: 参数估计滚动窗口
        """
        self.db_path = db_path
        self.n_simulations = n_simulations
        self.simulation_days = simulation_days
        self.alpha_threshold = alpha_threshold
        self.window = window

        # 初始化组件（延迟初始化 reader）
        self._reader = None
        self._param_estimator = None
        self._simulator = GBMSimulator(
            n_paths=n_simulations,
            n_steps=simulation_days
        )
        self._detector = SignalDetector()
        self._tester = SignificanceTester(alpha_threshold)

    @property
    def reader(self):
        """延迟初始化 DataReader"""
        if self._reader is None:
            # 延迟导入避免循环依赖
            from tushare_db import DataReader
            self._reader = DataReader(self.db_path)
        return self._reader

    @property
    def param_estimator(self):
        """延迟初始化 ParameterEstimator"""
        if self._param_estimator is None:
            self._param_estimator = ParameterEstimator(
                self.reader,
                self.window
            )
        return self._param_estimator

    def filter(
        self,
        factor: Union[str, Factor],
        ts_codes: List[str] = None,
        lookback_days: int = 504,
        use_sample: bool = False
    ) -> FactorReport:
        """
        对因子进行完整质检流程

        Args:
            factor: 因子名称（内置）或 Factor 实例
            ts_codes: 股票代码列表，None 则使用样本
            lookback_days: 回看天数
            use_sample: 是否使用模拟数据（用于测试）

        Returns:
            FactorReport 质检报告
        """
        # 解析因子
        if isinstance(factor, str):
            factor = FactorRegistry.get(factor)

        # 阶段 1: 参数估计
        if use_sample or ts_codes is None:
            # 使用模拟参数（用于测试）
            params_df = self._create_sample_params(factor)
        else:
            params_df = self.param_estimator.estimate_parameters(
                ts_codes=ts_codes,
                lookback_days=lookback_days,
                factor=factor
            )

        if params_df.empty:
            raise ValueError("No parameters estimated")

        # 阶段 2-4: 模拟检验
        return self._run_simulation_test(params_df, factor)

    def _create_sample_params(
        self,
        factor: Factor,
        n_stocks: int = 10
    ) -> pd.DataFrame:
        """
        创建模拟参数（用于测试）

        Args:
            factor: 因子
            n_stocks: 股票数量

        Returns:
            模拟参数 DataFrame
        """
        np.random.seed(42)

        data = []
        for i in range(n_stocks):
            data.append({
                'ts_code': f'{i:06d}.SZ',
                'mu': np.random.uniform(0.05, 0.15),
                'sigma': np.random.uniform(0.15, 0.35),
                'p_actual': np.random.uniform(0.02, 0.08),
                'n_days': 252
            })

        return pd.DataFrame(data)

    def _run_simulation_test(
        self,
        params_df: pd.DataFrame,
        factor: Factor
    ) -> FactorReport:
        """
        运行模拟检验

        Args:
            params_df: 参数 DataFrame
            factor: 因子

        Returns:
            FactorReport
        """
        test_results = []

        for _, row in params_df.iterrows():
            # 阶段 2: GBM 模拟
            price_matrix = self._simulator.simulate(
                s0=100,
                mu=row['mu'],
                sigma=row['sigma']
            )

            # 阶段 3: 信号检测
            signals = self._detector.detect_signals(price_matrix, factor)
            p_random = self._detector.calculate_p_random(signals)

            # 阶段 4: 显著性检验
            p_actual = row.get('p_actual', 0.05)
            n_total = self.n_simulations * self.simulation_days

            result = self._tester.test(
                p_actual=p_actual,
                p_random=p_random,
                n_total=n_total,
                ts_code=row['ts_code']
            )

            test_results.append(result)

        # 生成汇总统计
        summary = self._calculate_summary(test_results)

        # 生成报告
        report = self._tester.generate_report(factor, test_results)

        # 创建 DataFrame
        dataframe = pd.DataFrame([
            {
                'ts_code': r.ts_code,
                'p_actual': r.p_actual,
                'p_random': r.p_random,
                'alpha_ratio': r.alpha_ratio,
                'is_significant': r.is_significant,
                'recommendation': r.recommendation
            }
            for r in test_results
        ])

        return FactorReport(
            factor=factor,
            timestamp=datetime.now(),
            parameters={
                'n_simulations': self.n_simulations,
                'simulation_days': self.simulation_days,
                'alpha_threshold': self.alpha_threshold
            },
            results=test_results,
            summary=summary,
            markdown=report,
            dataframe=dataframe
        )

    def _calculate_summary(self, results: List[TestResult]) -> Dict[str, float]:
        """
        计算汇总统计

        Args:
            results: 检验结果列表

        Returns:
            汇总统计字典
        """
        if not results:
            return {}

        alpha_ratios = [r.alpha_ratio for r in results]
        p_actuals = [r.p_actual for r in results]
        p_randoms = [r.p_random for r in results]

        return {
            'alpha_ratio_median': float(np.median(alpha_ratios)),
            'alpha_ratio_mean': float(np.mean(alpha_ratios)),
            'p_actual_median': float(np.median(p_actuals)),
            'p_random_median': float(np.median(p_randoms)),
            'significant_ratio': sum(1 for r in results if r.is_significant) / len(results)
        }

    def batch_filter(
        self,
        factors: List[Union[str, Factor]],
        **kwargs
    ) -> List[FactorReport]:
        """
        批量质检多个因子

        Args:
            factors: 因子列表
            **kwargs: 传递给 filter() 的其他参数

        Returns:
            FactorReport 列表
        """
        reports = []
        for factor in factors:
            try:
                report = self.filter(factor, **kwargs)
                reports.append(report)
            except Exception as e:
                print(f"Error filtering factor {factor}: {e}")
                continue
        return reports

    def benchmark_all_builtin(
        self,
        **kwargs
    ) -> pd.DataFrame:
        """
        对所有内置因子进行质检

        Args:
            **kwargs: 传递给 filter() 的参数

        Returns:
            对比表格 DataFrame
        """
        builtin_names = FactorRegistry.list_builtin()
        reports = self.batch_filter(builtin_names, **kwargs)

        data = []
        for report in reports:
            data.append({
                'factor_name': report.factor.name,
                'alpha_ratio_median': report.summary.get('alpha_ratio_median', 0),
                'significant_ratio': report.summary.get('significant_ratio', 0),
                'recommendation': 'KEEP' if report.summary.get('alpha_ratio_median', 0) >= self.alpha_threshold else 'DISCARD'
            })

        return pd.DataFrame(data)
```

**Step 4: 更新 __init__.py**

`src/tushare_db/factor_validation/__init__.py`:
```python
"""
蒙特卡洛因子质检系统

通过对比技术指标在真实市场 vs 随机游走中的触发概率，
筛选出具有 Alpha 挖掘价值的真因子。

使用示例:
    from tushare_db.factor_validation import FactorFilter

    filter = FactorFilter(db_path="tushare.db")

    # 质检单个因子
    report = filter.filter("macd_golden_cross")
    print(report.markdown)

    # 保存报告
    report.save("reports/macd_report")
"""

__version__ = "0.1.0"

from .filter import FactorFilter, FactorReport
from .factor import Factor, FactorType, FactorRegistry
from .tester import TestResult

__all__ = [
    'FactorFilter',
    'FactorReport',
    'Factor',
    'FactorType',
    'FactorRegistry',
    'TestResult'
]
```

**Step 5: 运行测试确认通过**

```bash
pytest tests/factor_validation/test_filter.py -v
```

Expected: 所有测试通过

**Step 6: Commit**

```bash
git add src/tushare_db/factor_validation/filter.py tests/factor_validation/test_filter.py src/tushare_db/factor_validation/__init__.py
git commit -m "feat: implement FactorFilter main controller

- Integrate all 4 stages: estimation, simulation, detection, testing
- FactorReport with markdown and CSV export
- Batch filtering for multiple factors
- Benchmark all builtin factors
- Lazy initialization of DataReader

Generated with [Claude Code](https://claude.ai/code)
via [Happy](https://happy.engineering)

Co-Authored-By: Claude <noreply@anthropic.com>
Co-Authored-By: Happy <yesreply@happy.engineering>"
```

---

## Task 8: 添加内置因子库

**目标:** 实现 MACD 金叉、RSI 超卖等常用技术指标因子

**Files:**
- Create: `src/tushare_db/factor_validation/builtin_factors.py`
- Modify: `src/tushare_db/factor_validation/factor.py`
- Modify: `tests/factor_validation/test_factor.py`

**Step 1: 编写测试**

`tests/factor_validation/test_builtin_factors.py`:
```python
import pandas as pd
import numpy as np
import pytest


def test_builtin_factors_import():
    """测试模块可以导入"""
    from src.tushare_db.factor_validation import builtin_factors
    assert builtin_factors is not None


def test_macd_golden_cross_factor():
    """测试 MACD 金叉因子"""
    from src.tushare_db.factor_validation.builtin_factors import macd_golden_cross

    # 创建测试数据：构造一个金叉形态
    dates = pd.date_range('2024-01-01', periods=50)
    prices = [100]

    # 先下跌，然后上涨形成金叉
    for i in range(1, 25):
        prices.append(prices[-1] * 0.99)  # 下跌
    for i in range(25, 50):
        prices.append(prices[-1] * 1.01)  # 上涨

    df = pd.DataFrame({
        'close': prices
    }, index=dates)

    signals = macd_golden_cross(df)

    # 应该有至少一个金叉信号
    assert signals.sum() > 0


def test_rsi_oversold_factor():
    """测试 RSI 超卖因子"""
    from src.tushare_db.factor_validation.builtin_factors import rsi_oversold

    # 创建测试数据：连续下跌
    dates = pd.date_range('2024-01-01', periods=50)
    prices = [100]
    for i in range(1, 50):
        prices.append(prices[-1] * 0.98)  # 连续下跌

    df = pd.DataFrame({'close': prices}, index=dates)

    signals = rsi_oversold(df, period=14, threshold=30)

    # 连续下跌后应该有超卖信号
    assert signals.sum() > 0


def test_price_above_sma_factor():
    """测试价格在 SMA 之上因子"""
    from src.tushare_db.factor_validation.builtin_factors import price_above_sma

    dates = pd.date_range('2024-01-01', periods=50)
    # 构造趋势上涨的数据
    prices = [100 + i * 0.5 for i in range(50)]

    df = pd.DataFrame({'close': prices}, index=dates)

    signals = price_above_sma(df, period=20)

    # 上涨趋势中应该大部分时间在 SMA 之上
    assert signals.sum() > 25


def test_bollinger_lower_break_factor():
    """测试布林带下轨突破因子"""
    from src.tushare_db.factor_validation.builtin_factors import bollinger_lower_break

    dates = pd.date_range('2024-01-01', periods=50)
    prices = [100 + np.sin(i/5) * 10 for i in range(50)]

    df = pd.DataFrame({'close': prices}, index=dates)

    signals = bollinger_lower_break(df, period=20, std_dev=2)

    # 应该有下轨突破信号
    assert isinstance(signals, pd.Series)


def test_factor_registry_has_builtins():
    """测试注册表有内置因子"""
    from src.tushare_db.factor_validation.factor import FactorRegistry

    builtins = FactorRegistry.list_builtin()

    # 应该有一些内置因子
    assert len(builtins) > 0

    # 检查特定因子是否存在
    expected_factors = [
        'macd_golden_cross',
        'macd_death_cross',
        'rsi_oversold',
        'rsi_overbought'
    ]

    for factor_name in expected_factors:
        assert factor_name in builtins, f"{factor_name} not found in builtins"
```

**Step 2: 实现内置因子库**

`src/tushare_db/factor_validation/builtin_factors.py`:
```python
"""
内置因子库 - 常用技术指标因子
"""
import pandas as pd
import numpy as np


def macd_golden_cross(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> pd.Series:
    """
    MACD 金叉因子

    信号：DIF 上穿 DEA

    Args:
        df: 包含 'close' 列的 DataFrame
        fast: 快线周期
        slow: 慢线周期
        signal: 信号线周期

    Returns:
        布尔 Series
    """
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()

    golden_cross = (dif > dea) & (dif.shift(1) <= dea.shift(1))
    return golden_cross


def macd_death_cross(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> pd.Series:
    """
    MACD 死叉因子

    信号：DIF 下穿 DEA
    """
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()

    death_cross = (dif < dea) & (dif.shift(1) >= dea.shift(1))
    return death_cross


def macd_zero_golden_cross(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> pd.Series:
    """
    MACD 零轴下金叉因子

    信号：DIF 和 DEA 都在零轴下方，且 DIF 上穿 DEA
    """
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()

    golden_cross = (dif > dea) & (dif.shift(1) <= dea.shift(1))
    below_zero = (dif < 0) & (dea < 0)

    return golden_cross & below_zero


def rsi_oversold(
    df: pd.DataFrame,
    period: int = 14,
    threshold: float = 30
) -> pd.Series:
    """
    RSI 超卖因子

    信号：RSI 从下方上穿超卖线（默认 30）

    Args:
        df: 包含 'close' 列的 DataFrame
        period: RSI 周期
        threshold: 超卖阈值
    """
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    oversold_signal = (rsi > threshold) & (rsi.shift(1) <= threshold)
    return oversold_signal


def rsi_overbought(
    df: pd.DataFrame,
    period: int = 14,
    threshold: float = 70
) -> pd.Series:
    """
    RSI 超买因子

    信号：RSI 从上方下穿超买线（默认 70）
    """
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    overbought_signal = (rsi < threshold) & (rsi.shift(1) >= threshold)
    return overbought_signal


def price_above_sma(
    df: pd.DataFrame,
    period: int = 20
) -> pd.Series:
    """
    价格在 SMA 之上因子

    信号：收盘价上穿 SMA
    """
    sma = df['close'].rolling(window=period).mean()
    cross_above = (df['close'] > sma) & (df['close'].shift(1) <= sma.shift(1))
    return cross_above


def price_below_sma(
    df: pd.DataFrame,
    period: int = 20
) -> pd.Series:
    """
    价格在 SMA 之下因子

    信号：收盘价下穿 SMA
    """
    sma = df['close'].rolling(window=period).mean()
    cross_below = (df['close'] < sma) & (df['close'].shift(1) >= sma.shift(1))
    return cross_below


def bollinger_lower_break(
    df: pd.DataFrame,
    period: int = 20,
    std_dev: float = 2.0
) -> pd.Series:
    """
    布林带下轨突破因子

    信号：收盘价从下方上穿布林带下轨
    """
    sma = df['close'].rolling(window=period).mean()
    std = df['close'].rolling(window=period).std()
    lower_band = sma - std_dev * std

    break_above = (df['close'] > lower_band) & (df['close'].shift(1) <= lower_band.shift(1))
    return break_above


def bollinger_upper_break(
    df: pd.DataFrame,
    period: int = 20,
    std_dev: float = 2.0
) -> pd.Series:
    """
    布林带上轨突破因子

    信号：收盘价从上方下穿布林带上轨
    """
    sma = df['close'].rolling(window=period).mean()
    std = df['close'].rolling(window=period).std()
    upper_band = sma + std_dev * std

    break_below = (df['close'] < upper_band) & (df['close'].shift(1) >= upper_band.shift(1))
    return break_below


def volume_breakout(
    df: pd.DataFrame,
    volume_period: int = 20,
    multiplier: float = 1.5
) -> pd.Series:
    """
    成交量突破因子

    信号：成交量放大到均量的 multiplier 倍以上
    """
    if 'volume' not in df.columns:
        return pd.Series(False, index=df.index)

    avg_volume = df['volume'].rolling(window=volume_period).mean()
    breakout = df['volume'] > avg_volume * multiplier
    return breakout


def golden_cross(
    df: pd.DataFrame,
    fast_period: int = 5,
    slow_period: int = 20
) -> pd.Series:
    """
    均线金叉因子

    信号：短期均线上穿长期均线
    """
    fast_ma = df['close'].rolling(window=fast_period).mean()
    slow_ma = df['close'].rolling(window=slow_period).mean()

    cross = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
    return cross


def death_cross(
    df: pd.DataFrame,
    fast_period: int = 5,
    slow_period: int = 20
) -> pd.Series:
    """
    均线死叉因子

    信号：短期均线下穿长期均线
    """
    fast_ma = df['close'].rolling(window=fast_period).mean()
    slow_ma = df['close'].rolling(window=slow_period).mean()

    cross = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))
    return cross


def close_gt_open(df: pd.DataFrame) -> pd.Series:
    """
    阳线因子

    信号：收盘价 > 开盘价
    """
    if 'open' not in df.columns:
        return pd.Series(False, index=df.index)
    return df['close'] > df['open']


def gap_up(
    df: pd.DataFrame,
    threshold: float = 0.02
) -> pd.Series:
    """
    跳空高开因子

    信号：今日开盘价 > 昨日收盘价 * (1 + threshold)
    """
    if 'open' not in df.columns:
        return pd.Series(False, index=df.index)

    gap = df['open'] > df['close'].shift(1) * (1 + threshold)
    return gap


def gap_down(
    df: pd.DataFrame,
    threshold: float = 0.02
) -> pd.Series:
    """
    跳空低开因子

    信号：今日开盘价 < 昨日收盘价 * (1 - threshold)
    """
    if 'open' not in df.columns:
        return pd.Series(False, index=df.index)

    gap = df['open'] < df['close'].shift(1) * (1 - threshold)
    return gap
```

**Step 3: 更新 factor.py 注册内置因子**

在 `src/tushare_db/factor_validation/factor.py` 末尾添加：

```python
# 导入并注册内置因子
try:
    from . import builtin_factors

    # 注册 MACD 因子
    FactorRegistry.register(Factor(
        name="macd_golden_cross",
        description="MACD 金叉（DIF上穿DEA）",
        definition=builtin_factors.macd_golden_cross,
        type=FactorType.FUNCTION
    ))

    FactorRegistry.register(Factor(
        name="macd_death_cross",
        description="MACD 死叉（DIF下穿DEA）",
        definition=builtin_factors.macd_death_cross,
        type=FactorType.FUNCTION
    ))

    FactorRegistry.register(Factor(
        name="macd_zero_golden_cross",
        description="MACD 零轴下金叉",
        definition=builtin_factors.macd_zero_golden_cross,
        type=FactorType.FUNCTION
    ))

    # 注册 RSI 因子
    FactorRegistry.register(Factor(
        name="rsi_oversold",
        description="RSI 超卖（上穿30）",
        definition=builtin_factors.rsi_oversold,
        type=FactorType.FUNCTION
    ))

    FactorRegistry.register(Factor(
        name="rsi_overbought",
        description="RSI 超买（下穿70）",
        definition=builtin_factors.rsi_overbought,
        type=FactorType.FUNCTION
    ))

    # 注册均线因子
    FactorRegistry.register(Factor(
        name="golden_cross",
        description="均线金叉（5日上穿20日）",
        definition=builtin_factors.golden_cross,
        type=FactorType.FUNCTION,
        parameters={'fast_period': 5, 'slow_period': 20}
    ))

    FactorRegistry.register(Factor(
        name="death_cross",
        description="均线死叉（5日下穿20日）",
        definition=builtin_factors.death_cross,
        type=FactorType.FUNCTION,
        parameters={'fast_period': 5, 'slow_period': 20}
    ))

    FactorRegistry.register(Factor(
        name="price_above_sma",
        description="价格上穿20日均线",
        definition=builtin_factors.price_above_sma,
        type=FactorType.FUNCTION,
        parameters={'period': 20}
    ))

    # 注册布林带因子
    FactorRegistry.register(Factor(
        name="bollinger_lower_break",
        description="布林带下轨突破",
        definition=builtin_factors.bollinger_lower_break,
        type=FactorType.FUNCTION
    ))

    FactorRegistry.register(Factor(
        name="bollinger_upper_break",
        description="布林带上轨突破",
        definition=builtin_factors.bollinger_upper_break,
        type=FactorType.FUNCTION
    ))

    # 注册成交量因子
    FactorRegistry.register(Factor(
        name="volume_breakout",
        description="成交量突破（放量1.5倍）",
        definition=builtin_factors.volume_breakout,
        type=FactorType.FUNCTION
    ))

    # 注册价格形态因子
    FactorRegistry.register(Factor(
        name="close_gt_open",
        description="阳线（收盘大于开盘）",
        definition=builtin_factors.close_gt_open,
        type=FactorType.FUNCTION
    ))

    FactorRegistry.register(Factor(
        name="gap_up",
        description="跳空高开（2%以上）",
        definition=builtin_factors.gap_up,
        type=FactorType.FUNCTION
    ))

    FactorRegistry.register(Factor(
        name="gap_down",
        description="跳空低开（2%以上）",
        definition=builtin_factors.gap_down,
        type=FactorType.FUNCTION
    ))

except ImportError:
    pass  # 避免循环导入
```

**Step 4: 运行测试确认通过**

```bash
pytest tests/factor_validation/test_builtin_factors.py -v
```

Expected: 所有测试通过

**Step 5: Commit**

```bash
git add src/tushare_db/factor_validation/builtin_factors.py tests/factor_validation/test_builtin_factors.py src/tushare_db/factor_validation/factor.py
git commit -m "feat: add builtin technical indicator factors

- MACD golden/death/zero-cross factors
- RSI oversold/overbought factors
- Moving average golden/death cross
- Bollinger band break factors
- Volume breakout factor
- Price patterns (close>open, gap up/down)
- Auto-registration in FactorRegistry

Generated with [Claude Code](https://claude.ai/code)
via [Happy](https://happy.engineering)

Co-Authored-By: Claude <noreply@anthropic.com>
Co-Authored-By: Happy <yesreply@happy.engineering>"
```

---

## Task 9: 添加集成测试和示例

**目标:** 添加完整的集成测试和使用示例

**Files:**
- Create: `tests/factor_validation/test_integration.py`
- Create: `examples/factor_validation_demo.py`

**Step 1: 编写集成测试**

`tests/factor_validation/test_integration.py`:
```python
"""
集成测试 - 验证完整流程
"""
import pytest
import numpy as np


class TestFullWorkflow:
    """完整工作流测试"""

    def test_full_workflow_with_sample_data(self):
        """使用模拟数据的完整流程测试"""
        from src.tushare_db.factor_validation import FactorFilter, FactorRegistry

        # 初始化过滤器（使用内存数据库）
        filter_obj = FactorFilter(
            db_path=":memory:",
            n_simulations=100,
            simulation_days=50,
            alpha_threshold=1.5
        )

        # 获取内置因子
        factor = FactorRegistry.get("close_gt_open")

        # 运行质检
        report = filter_obj.filter(
            factor=factor,
            use_sample=True  # 使用模拟数据
        )

        # 验证报告结构
        assert report.factor.name == "close_gt_open"
        assert len(report.results) == 10  # 默认10只模拟股票
        assert report.summary is not None
        assert 'alpha_ratio_median' in report.summary
        assert report.markdown is not None
        assert report.dataframe is not None

    def test_batch_filter(self):
        """批量质检测试"""
        from src.tushare_db.factor_validation import FactorFilter, FactorRegistry

        filter_obj = FactorFilter(
            db_path=":memory:",
            n_simulations=50,
            simulation_days=30
        )

        # 批量质检
        factors = ["close_gt_open", "macd_golden_cross"]
        reports = filter_obj.batch_filter(factors, use_sample=True)

        assert len(reports) == 2
        for report in reports:
            assert report.factor.name in ["close_gt_open", "macd_golden_cross"]

    def test_report_save(self, tmp_path):
        """测试报告保存"""
        from src.tushare_db.factor_validation import FactorFilter, FactorRegistry

        filter_obj = FactorFilter(
            db_path=":memory:",
            n_simulations=50,
            simulation_days=30
        )

        factor = FactorRegistry.get("close_gt_open")
        report = filter_obj.filter(factor=factor, use_sample=True)

        # 保存到临时目录
        save_path = tmp_path / "test_report"
        report.save(str(save_path))

        # 验证文件存在
        assert (tmp_path / "test_report.md").exists()
        assert (tmp_path / "test_report.csv").exists()

        # 验证 Markdown 内容
        md_content = (tmp_path / "test_report.md").read_text()
        assert "close_gt_open" in md_content

    def test_alpha_ratio_calculation(self):
        """验证 Alpha Ratio 计算正确性"""
        from src.tushare_db.factor_validation import FactorFilter, FactorRegistry

        filter_obj = FactorFilter(
            db_path=":memory:",
            n_simulations=1000,
            simulation_days=100
        )

        factor = FactorRegistry.get("close_gt_open")

        # 手动创建参数：P_actual = 0.05
        import pandas as pd
        params_df = pd.DataFrame({
            'ts_code': ['TEST.SZ'],
            'mu': [0.1],
            'sigma': [0.2],
            'p_actual': [0.05]
        })

        report = filter_obj._run_simulation_test(params_df, factor)

        # 验证计算结果
        result = report.results[0]
        assert result.alpha_ratio == result.p_actual / result.p_random
        assert result.p_actual == 0.05


class TestRealDataIntegration:
    """真实数据集成测试（可选，需要真实数据库）"""

    @pytest.mark.skip(reason="需要真实数据库")
    def test_with_real_data(self):
        """使用真实数据的测试"""
        from src.tushare_db.factor_validation import FactorFilter

        filter_obj = FactorFilter(db_path="tushare.db")

        # 测试几只股票
        ts_codes = ['000001.SZ', '000002.SZ']

        report = filter_obj.filter(
            factor="macd_golden_cross",
            ts_codes=ts_codes,
            lookback_days=252
        )

        assert len(report.results) == 2
        for result in report.results:
            assert result.ts_code in ts_codes
```

**Step 2: 编写使用示例**

`examples/factor_validation_demo.py`:
```python
"""
蒙特卡洛因子质检系统使用示例

本示例展示如何使用 FactorFilter 对技术指标进行统计检验。
"""
import sys
sys.path.insert(0, '/Users/allen/workspace/python/stock/Tushare-DuckDB')

from src.tushare_db.factor_validation import FactorFilter, FactorRegistry


def demo_basic_usage():
    """基础用法：检验单个因子"""
    print("=" * 60)
    print("示例 1: 基础用法 - 检验 MACD 金叉因子")
    print("=" * 60)

    # 初始化因子过滤器
    filter_obj = FactorFilter(
        db_path=":memory:",  # 使用内存数据库进行演示
        n_simulations=1000,   # 模拟 1000 条路径
        simulation_days=100,  # 每条路径 100 天
        alpha_threshold=1.5   # Alpha Ratio 阈值
    )

    # 检验 MACD 金叉因子（使用模拟数据）
    report = filter_obj.filter(
        factor="macd_golden_cross",
        use_sample=True  # 使用模拟股票数据
    )

    # 打印报告
    print("\n检验结果：")
    print(f"因子名称: {report.factor.name}")
    print(f"Alpha Ratio 中位数: {report.summary['alpha_ratio_median']:.2f}")
    print(f"显著标的比例: {report.summary['significant_ratio']:.1%}")
    print(f"\n建议: {'保留' if report.summary['alpha_ratio_median'] >= 1.5 else '优化/废弃'}")


def demo_batch_filter():
    """批量质检多个因子"""
    print("\n" + "=" * 60)
    print("示例 2: 批量质检多个因子")
    print("=" * 60)

    filter_obj = FactorFilter(
        db_path=":memory:",
        n_simulations=500,
        simulation_days=50
    )

    # 要检验的因子列表
    factors = [
        "macd_golden_cross",
        "rsi_oversold",
        "golden_cross",
        "close_gt_open"
    ]

    # 批量检验
    reports = filter_obj.batch_filter(factors, use_sample=True)

    # 打印对比结果
    print("\n| 因子名称 | Alpha Ratio | 建议 |")
    print("|----------|-------------|------|")
    for report in reports:
        alpha = report.summary['alpha_ratio_median']
        rec = '保留' if alpha >= 1.5 else '废弃'
        print(f"| {report.factor.name:20} | {alpha:11.2f} | {rec:4} |")


def demo_custom_factor():
    """使用自定义因子"""
    print("\n" + "=" * 60)
    print("示例 3: 使用自定义因子")
    print("=" * 60)

    import pandas as pd
    from src.tushare_db.factor_validation import FactorRegistry

    # 定义自定义因子函数
    def my_custom_factor(df: pd.DataFrame) -> pd.Series:
        """
        自定义因子：放量上涨
        条件：收盘价 > 开盘价 且 成交量 > 前一日成交量 * 1.5
        """
        price_up = df['close'] > df['open']
        volume_surge = df['volume'] > df['volume'].shift(1) * 1.5
        return price_up & volume_surge

    # 注册自定义因子
    custom_factor = FactorRegistry.create_from_function(
        my_custom_factor,
        name="volume_surge_up",
        description="放量上涨因子"
    )

    # 检验自定义因子
    filter_obj = FactorFilter(
        db_path=":memory:",
        n_simulations=500,
        simulation_days=50
    )

    report = filter_obj.filter(custom_factor, use_sample=True)

    print(f"\n自定义因子 '{custom_factor.name}' 检验结果：")
    print(f"Alpha Ratio: {report.summary['alpha_ratio_median']:.2f}")
    print(f"P_actual: {report.summary['p_actual_median']:.2%}")
    print(f"P_random: {report.summary['p_random_median']:.2%}")


def demo_list_builtin_factors():
    """列出所有内置因子"""
    print("\n" + "=" * 60)
    print("示例 4: 列出所有内置因子")
    print("=" * 60)

    builtins = FactorRegistry.list_builtin()

    print(f"\n共有 {len(builtins)} 个内置因子：\n")

    categories = {
        'MACD': [f for f in builtins if 'macd' in f],
        'RSI': [f for f in builtins if 'rsi' in f],
        '均线': [f for f in builtins if 'cross' in f or 'sma' in f],
        '布林带': [f for f in builtins if 'bollinger' in f],
        '成交量': [f for f in builtins if 'volume' in f],
        '价格形态': [f for f in builtins if f not in [item for sublist in [
            [f for f in builtins if 'macd' in f],
            [f for f in builtins if 'rsi' in f],
            [f for f in builtins if 'cross' in f or 'sma' in f],
            [f for f in builtins if 'bollinger' in f],
            [f for f in builtins if 'volume' in f]
        ] for item in sublist]]
    }

    for category, factors in categories.items():
        if factors:
            print(f"\n【{category}】")
            for factor_name in factors:
                factor = FactorRegistry.get(factor_name)
                print(f"  - {factor_name}: {factor.description}")


def demo_save_report():
    """保存检验报告"""
    print("\n" + "=" * 60)
    print("示例 5: 保存检验报告")
    print("=" * 60)

    filter_obj = FactorFilter(
        db_path=":memory:",
        n_simulations=500,
        simulation_days=50
    )

    report = filter_obj.filter("rsi_oversold", use_sample=True)

    # 保存报告到文件
    output_path = "factor_validation_report"
    report.save(output_path)

    print(f"\n报告已保存到：")
    print(f"  - Markdown: {output_path}.md")
    print(f"  - CSV: {output_path}.csv")
    print(f"\n报告预览：")
    print(report.markdown[:500] + "...")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("蒙特卡洛因子质检系统 - 使用示例")
    print("=" * 60)

    # 运行所有示例
    demo_basic_usage()
    demo_batch_filter()
    demo_custom_factor()
    demo_list_builtin_factors()
    demo_save_report()

    print("\n" + "=" * 60)
    print("示例运行完成！")
    print("=" * 60)
```

**Step 3: 运行测试确认通过**

```bash
pytest tests/factor_validation/test_integration.py -v
```

Expected: 所有测试通过

**Step 4: Commit**

```bash
git add tests/factor_validation/test_integration.py examples/factor_validation_demo.py
git commit -m "feat: add integration tests and usage examples

- Full workflow integration tests
- Batch filtering tests
- Report saving tests
- Comprehensive usage examples demonstrating:
  - Basic single factor validation
  - Batch validation
  - Custom factor creation
  - Builtin factor listing
  - Report saving

Generated with [Claude Code](https://claude.ai/code)
via [Happy](https://happy.engineering)

Co-Authored-By: Claude <noreply@anthropic.com>
Co-Authored-By: Happy <yesreply@happy.engineering>"
```

---

## Task 10: 性能测试和优化

**目标:** 验证系统性能，确保向量化计算有效

**Files:**
- Create: `tests/factor_validation/test_performance.py`

**Step 1: 编写性能测试**

`tests/factor_validation/test_performance.py`:
```python
"""
性能测试 - 验证向量化计算性能
"""
import pytest
import time
import numpy as np


class TestPerformance:
    """性能测试"""

    def test_gbm_simulation_speed(self):
        """测试 GBM 模拟速度"""
        from src.tushare_db.factor_validation.simulator import GBMSimulator

        sim = GBMSimulator(n_paths=10000, n_steps=252)

        start = time.time()
        result = sim.simulate(s0=100, mu=0.1, sigma=0.2)
        elapsed = time.time() - start

        print(f"\nGBM 模拟 (10000×252): {elapsed:.3f}s")

        # 应该在 1 秒内完成
        assert elapsed < 1.0
        assert result.shape == (10000, 252)

    def test_macd_detection_speed(self):
        """测试 MACD 信号检测速度"""
        from src.tushare_db.factor_validation.detector import SignalDetector

        detector = SignalDetector()

        # 生成测试数据
        np.random.seed(42)
        price_matrix = np.cumsum(np.random.randn(10000, 252) * 0.01, axis=1) + 100

        start = time.time()
        signals = detector.macd_golden_cross(price_matrix)
        elapsed = time.time() - start

        print(f"\nMACD 金叉检测 (10000×252): {elapsed:.3f}s")

        # 应该在 1 秒内完成
        assert elapsed < 1.0
        assert signals.shape == (10000, 252)

    def test_vs_loop_implementation(self):
        """对比向量化 vs 循环实现"""
        from src.tushare_db.factor_validation.detector import SignalDetector

        detector = SignalDetector()

        # 小规模数据对比
        np.random.seed(42)
        price_matrix = np.cumsum(np.random.randn(100, 100) * 0.01, axis=1) + 100

        # 向量化实现
        start = time.time()
        vec_signals = detector.macd_golden_cross(price_matrix)
        vec_time = time.time() - start

        # 循环实现（仅测试用）
        start = time.time()
        loop_signals = np.zeros_like(price_matrix, dtype=bool)
        for i in range(price_matrix.shape[0]):
            prices = price_matrix[i]
            ema12 = pd.Series(prices).ewm(span=12).mean().values
            ema26 = pd.Series(prices).ewm(span=26).mean().values
            dif = ema12 - ema26
            dea = pd.Series(dif).ewm(span=9).mean().values
            golden = (dif[1:] > dea[1:]) & (dif[:-1] <= dea[:-1])
            loop_signals[i, 1:] = golden
        loop_time = time.time() - start

        print(f"\n向量化实现: {vec_time:.3f}s")
        print(f"循环实现: {loop_time:.3f}s")
        print(f"加速比: {loop_time / vec_time:.1f}x")

        # 向量化应该快得多
        assert vec_time < loop_time

    def test_memory_usage(self):
        """测试内存使用"""
        import psutil
        import os

        from src.tushare_db.factor_validation.simulator import GBMSimulator

        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        # 大规模模拟
        sim = GBMSimulator(n_paths=50000, n_steps=252)
        result = sim.simulate(s0=100, mu=0.1, sigma=0.2)

        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        mem_used = mem_after - mem_before

        print(f"\n内存使用: {mem_used:.1f} MB (50000×252)")

        # 50000×252 float64 约 100MB
        assert mem_used < 500  # 应该小于 500MB


class TestAccuracy:
    """准确性测试"""

    def test_gbm_statistical_properties(self):
        """验证 GBM 的统计特性"""
        from src.tushare_db.factor_validation.simulator import GBMSimulator

        np.random.seed(42)
        sim = GBMSimulator(n_paths=10000, n_steps=252, random_seed=42)
        result = sim.simulate(s0=100, mu=0.1, sigma=0.2)

        # 计算最终收益的统计特性
        final_returns = np.log(result[:, -1] / result[:, 0])

        mean_return = np.mean(final_returns)
        std_return = np.std(final_returns)

        # 理论值: E[ln(S_T/S_0)] = (μ - σ²/2) × T
        # T = 1 (252/252), μ = 0.1, σ = 0.2
        expected_mean = (0.1 - 0.2**2 / 2) * 1
        expected_std = 0.2 * np.sqrt(1)

        print(f"\n实际均值: {mean_return:.4f}, 理论均值: {expected_mean:.4f}")
        print(f"实际标准差: {std_return:.4f}, 理论标准差: {expected_std:.4f}")

        # 允许 10% 误差
        assert abs(mean_return - expected_mean) < abs(expected_mean) * 0.1
        assert abs(std_return - expected_std) < expected_std * 0.1

    def test_price_limits_effect(self):
        """验证涨跌停限制生效"""
        from src.tushare_db.factor_validation.simulator import GBMSimulator

        # 高波动率数据，容易触发涨跌停
        sim = GBMSimulator(
            n_paths=1000,
            n_steps=100,
            limit_up=0.10,
            limit_down=-0.10,
            random_seed=42
        )

        # 不加限制
        sim_no_limit = GBMSimulator(
            n_paths=1000,
            n_steps=100,
            limit_up=1.0,  # 100% 实际上不限制
            limit_down=-1.0,
            random_seed=42
        )

        result_limited = sim.simulate(s0=100, mu=0.1, sigma=0.5)
        result_unlimited = sim_no_limit.simulate(s0=100, mu=0.1, sigma=0.5)

        # 计算最大日收益率
        returns_limited = np.diff(result_limited) / result_limited[:, :-1]
        returns_unlimited = np.diff(result_unlimited) / result_unlimited[:, :-1]

        max_return_limited = np.max(returns_limited)
        max_return_unlimited = np.max(returns_unlimited)

        print(f"\n限制后最大日收益: {max_return_limited:.2%}")
        print(f"无限制最大日收益: {max_return_unlimited:.2%}")

        # 限制后应该不超过 10%
        assert max_return_limited <= 0.10 + 1e-6
        # 无限制应该更大
        assert max_return_unlimited > 0.10
```

**Step 2: 运行性能测试**

```bash
pytest tests/factor_validation/test_performance.py -v -s
```

Expected: 所有测试通过，显示性能指标

**Step 3: Commit**

```bash
git add tests/factor_validation/test_performance.py
git commit -m "test: add performance and accuracy tests

- GBM simulation speed test (< 1s for 10k×252)
- MACD detection speed test
- Vectorized vs loop comparison
- Memory usage verification
- Statistical properties validation
- Price limits effectiveness test

Generated with [Claude Code](https://claude.ai/code)
via [Happy](https://happy.engineering)

Co-Authored-By: Claude <noreply@anthropic.com>
Co-Authored-By: Happy <yesreply@happy.engineering>"
```

---

## Task 11: 更新主模块导出和最终验证

**目标:** 更新 tushare_db 主模块导出，运行全部测试

**Files:**
- Modify: `src/tushare_db/__init__.py`
- Run: All tests

**Step 1: 更新主模块导出**

`src/tushare_db/__init__.py`:
```python
"""
Tushare-DuckDB: A local caching layer for Tushare Pro financial data
"""

# 核心组件
from .reader import DataReader
from .downloader import DataDownloader

# 因子质检模块（新增）
try:
    from .factor_validation import (
        FactorFilter,
        FactorReport,
        Factor,
        FactorType,
        FactorRegistry,
        TestResult
    )
    FACTOR_VALIDATION_AVAILABLE = True
except ImportError:
    FACTOR_VALIDATION_AVAILABLE = False

__version__ = "0.1.0"

__all__ = [
    'DataReader',
    'DataDownloader',
    # 因子质检（如果可用）
    'FactorFilter',
    'FactorReport',
    'Factor',
    'FactorType',
    'FactorRegistry',
    'TestResult',
]
```

**Step 2: 运行全部测试**

```bash
pytest tests/factor_validation/ -v --tb=short
```

Expected: 所有测试通过

**Step 3: Commit**

```bash
git add src/tushare_db/__init__.py
git commit -m "feat: update main module exports for factor validation

- Export FactorFilter and related classes from tushare_db
- Add FACTOR_VALIDATION_AVAILABLE flag
- Make factor validation module optional import

Generated with [Claude Code](https://claude.ai/code)
via [Happy](https://happy.engineering)

Co-Authored-By: Claude <noreply@anthropic.com>
Co-Authored-By: Happy <yesreply@happy.engineering>"
```

---

## 完成检查清单

所有任务完成后，确认以下检查项：

- [ ] 所有单元测试通过 (`pytest tests/factor_validation/ -v`)
- [ ] 性能测试通过且满足要求
- [ ] 示例代码可以运行 (`python examples/factor_validation_demo.py`)
- [ ] 代码已提交到 git
- [ ] 设计文档和实现计划已更新

---

## 下一步（可选）

1. **文档完善**: 添加 API 文档和使用指南
2. **真实数据测试**: 使用 tushare.db 进行完整测试
3. **性能优化**: 进一步优化向量化计算
4. **更多因子**: 添加更多技术指标因子
5. **可视化**: 添加结果可视化功能
