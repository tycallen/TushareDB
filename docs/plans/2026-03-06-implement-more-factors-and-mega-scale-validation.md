# 扩展因子库与超大规模验证实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 实现20+新因子，并对所有50+因子进行100万路径级别的超大规模蒙特卡洛验证

**Architecture:** 分阶段实施：第一阶段添加20个新因子（动量、波动率、成交量、机器学习特征）；第二阶段实现向量化计算优化；第三阶段进行100万路径×252天=2.52亿样本的超大规模验证；第四阶段更新CLAUDE.md记录验证规模参数

**Tech Stack:** Python 3.10+, NumPy, Pandas, Tushare-DuckDB, pytest

---

## 前置准备

### 当前状态检查

```bash
# 当前已有因子
python -c "from src.tushare_db.factor_validation import FactorRegistry; print(f'当前因子数: {len(FactorRegistry.list_builtin())}')"

# 当前验证规模（应在50,000-1,000,000之间）
grep -A 5 "N_PATHS" src/tushare_db/factor_validation/filter.py
```

---

## Phase 1: 扩展因子库（添加20+新因子）

### Task 1: 添加动量因子

**Files:**
- Create: `tests/factor_validation/test_momentum_factors.py`
- Modify: `src/tushare_db/factor_validation/builtin_factors.py`

**Step 1: 编写测试**

Create `tests/factor_validation/test_momentum_factors.py`:
```python
import pandas as pd
import numpy as np
import pytest


def test_momentum_20_day_factor():
    """测试20日动量因子"""
    from src.tushare_db.factor_validation.builtin_factors import momentum_20_day

    df = pd.DataFrame({
        'close': [100, 102, 101, 105, 110, 108, 112, 115, 113, 118,
                  120, 122, 121, 125, 128, 126, 130, 132, 131, 135,
                  138, 140]
    })

    signals = momentum_20_day(df, period=20, threshold=0.05)

    # 最后几天应该有信号
    assert signals.sum() > 0
    assert signals.dtype == bool


def test_price_momentum_factor():
    """测试价格动量突破因子"""
    from src.tushare_db.factor_validation.builtin_factors import price_momentum_breakout

    df = pd.DataFrame({
        'close': [100] * 50 + [110, 112, 115]  # 长期横盘后突破
    })

    signals = price_momentum_breakout(df, lookback=20)

    # 突破时应该有信号
    assert signals.iloc[-1] == True


def test_acceleration_factor():
    """测试加速度因子（二阶动量）"""
    from src.tushare_db.factor_validation.builtin_factors import price_acceleration

    # 加速上涨的数据
    prices = [100]
    for i in range(20):
        prices.append(prices[-1] * (1.01 + i * 0.001))  # 涨幅越来越大

    df = pd.DataFrame({'close': prices})
    signals = price_acceleration(df)

    assert signals.sum() > 0
```

**Step 2: 运行测试确认失败**

```bash
cd /Users/allen/workspace/python/stock/Tushare-DuckDB/.worktrees/monte-carlo-factor-filter
pytest tests/factor_validation/test_momentum_factors.py -v
```
Expected: ImportError

**Step 3: 实现动量因子**

Add to `src/tushare_db/factor_validation/builtin_factors.py`:
```python
# =============================================================================
# Momentum Factors
# =============================================================================

def momentum_20_day(
    df: pd.DataFrame,
    period: int = 20,
    threshold: float = 0.05
) -> pd.Series:
    """
    20日动量因子：价格突破N日新高且动量超过阈值

    条件：
    1. 当前价格创20日新高
    2. (当前价 - 20日前价格) / 20日前价格 > threshold

    Args:
        df: DataFrame with 'close' column
        period: 动量计算周期
        threshold: 动量阈值

    Returns:
        Boolean Series indicating momentum breakout signals
    """
    if 'close' not in df.columns:
        raise ValueError("DataFrame must have 'close' column")

    # 计算N日最高价
    rolling_high = df['close'].rolling(window=period).max()

    # 当前价格等于N日最高价（创N日新高）
    is_new_high = df['close'] >= rolling_high

    # 计算动量
    price_n_periods_ago = df['close'].shift(period)
    momentum = (df['close'] - price_n_periods_ago) / price_n_periods_ago

    # 动量超过阈值
    has_momentum = momentum > threshold

    # 金叉：从下方上穿阈值
    momentum_prev = momentum.shift(1)
    momentum_cross = (momentum > threshold) & (momentum_prev <= threshold)

    return is_new_high & momentum_cross


def price_momentum_breakout(
    df: pd.DataFrame,
    lookback: int = 20,
    breakout_threshold: float = 0.03
) -> pd.Series:
    """
    价格动量突破：突破N日盘整区间且伴随动量放大

    识别长期横盘后的突破行情
    """
    if 'close' not in df.columns:
        raise ValueError("DataFrame must have 'close' column")

    # 计算N日最高最低价
    high_max = df['close'].rolling(window=lookback).max()
    low_min = df['close'].rolling(window=lookback).min()

    # 计算波动率（区间幅度）
    range_ratio = (high_max - low_min) / low_min

    # 突破：当前价格突破前高且幅度超过阈值
    breakout_up = (df['close'] > high_max.shift(1) * (1 + breakout_threshold))

    # 突破时应该有放量（如果有volume列）
    if 'vol' in df.columns:
        avg_volume = df['vol'].rolling(window=lookback).mean()
        volume_surge = df['vol'] > avg_volume * 1.5
    else:
        volume_surge = pd.Series(True, index=df.index)

    return breakout_up & volume_surge


def price_acceleration(
    df: pd.DataFrame,
    short_period: int = 5,
    long_period: int = 20
) -> pd.Series:
    """
    价格加速度因子：动量的变化率（二阶导数）

    识别加速上涨或加速下跌
    """
    if 'close' not in df.columns:
        raise ValueError("DataFrame must have 'close' column")

    # 计算短期和长期动量
    returns = df['close'].pct_change()
    momentum_short = returns.rolling(window=short_period).mean()
    momentum_long = returns.rolling(window=long_period).mean()

    # 加速度 = 短期动量 - 长期动量
    acceleration = momentum_short - momentum_long

    # 加速上涨信号
    acceleration_prev = acceleration.shift(1)
    accel_up_signal = (acceleration > 0) & (acceleration > acceleration_prev) & (acceleration_prev > 0)

    return accel_up_signal
```

**Step 4: 运行测试确认通过**

```bash
pytest tests/factor_validation/test_momentum_factors.py -v
```
Expected: 3 tests passed

**Step 5: Commit**

```bash
git add tests/factor_validation/test_momentum_factors.py src/tushare_db/factor_validation/builtin_factors.py
git commit -m "feat: add momentum factors (momentum_20_day, price_momentum_breakout, price_acceleration)

- 20-day momentum breakout with threshold
- Price breakout from consolidation with volume confirmation
- Price acceleration (second-order momentum)
- Full test coverage

Generated with [Claude Code](https://claude.ai/code)
via [Happy](https://happy.engineering)

Co-Authored-By: Claude <noreply@anthropic.com>
Co-Authored-By: Happy <yesreply@happy.engineering>"
```

---

### Task 2: 添加波动率因子

**Files:**
- Create: `tests/factor_validation/test_volatility_factors.py`
- Modify: `src/tushare_db/factor_validation/builtin_factors.py`

**Step 1-5:** 类似Task 1的结构

添加以下因子：
- `volatility_expansion` - 波动率扩张
- `volatility_contraction` - 波动率收缩（squeeze）
- `atr_percent_b` - ATR百分比通道
- `bollinger_squeeze` - 布林带收缩突破

---

### Task 3: 添加成交量因子

**Files:**
- Create: `tests/factor_validation/test_volume_factors.py`
- Modify: `src/tushare_db/factor_validation/builtin_factors.py`

添加以下因子：
- `volume_price_divergence` - 量价背离
- `on_balance_volume_breakout` - OBV突破
- `volume_weighted_momentum` - 成交量加权动量
- `accumulation_distribution_breakout` - 集散指标突破

---

### Task 4: 添加多时间框架因子

**Files:**
- Create: `tests/factor_validation/test_multitimeframe_factors.py`
- Modify: `src/tushare_db/factor_validation/builtin_factors.py`

添加以下因子：
- `multi_timeframe_alignment` - 多周期共振
- `higher_high_lower_low_sequence` - HHLL序列
- `support_resistance_breakout` - 支撑阻力突破

---

### Task 5: 注册所有新因子并更新测试

**Files:**
- Modify: `src/tushare_db/factor_validation/factor.py`

**Step:** 在factor.py中注册所有新因子

---

## Phase 2: 向量化计算优化

### Task 6: 实现向量化OHLC生成器

**Files:**
- Create: `src/tushare_db/factor_validation/vectorized_ohlc.py`
- Create: `tests/factor_validation/test_vectorized_ohlc.py`

**目标：** 为百万路径优化生成OHLC数据

```python
class VectorizedOHLCGenerator:
    """向量化生成OHLC数据，支持百万路径"""

    def generate(self, price_matrix: np.ndarray) -> Dict[str, np.ndarray]:
        """
        输入: (n_paths, n_steps) 价格矩阵
        输出: {'open': ..., 'high': ..., 'low': ..., 'close': ..., 'volume': ...}
        所有输出都是 (n_paths, n_steps) 矩阵
        """
        pass
```

---

## Phase 3: 超大规模验证（100万路径）

### Task 7: 创建百万路径验证脚本

**Files:**
- Create: `scripts/mega_scale_validation.py`

**要求：**
- 支持1,000,000路径
- 分块处理避免内存溢出
- 进度显示
- 结果保存到专用目录

### Task 8: 运行完整验证

**命令：**
```bash
python scripts/mega_scale_validation.py --n-paths 1000000 --n-steps 252 --all-factors
```

**输出：**
- CSV结果文件
- Markdown报告
- 更新到数据库

---

## Phase 4: 更新 CLAUDE.md

### Task 9: 记录验证规模参数

**Files:**
- Modify: `CLAUDE.md`

在CLAUDE.md中添加以下章节：

```markdown
## Factor Validation Scale

### Default Validation Parameters

```python
# Standard validation (daily use)
N_PATHS = 50_000        # 50,000 paths
N_STEPS = 252           # 252 trading days
TOTAL_SAMPLES = 12_600_000  # 12.6 million samples per factor

# Mega-scale validation (research grade)
N_PATHS = 1_000_000     # 1,000,000 paths
N_STEPS = 252
TOTAL_SAMPLES = 252_000_000  # 252 million samples per factor
```

### Current Factor Library Size

- **Total factors**: 50+ (30 builtin + 20 extended)
- **Validated at 1M paths**: All factors
- **Recommended for production**: 50K paths (balance of speed and accuracy)
- **Recommended for research**: 1M paths (maximum statistical power)

### Statistical Precision by Scale

| Paths | Relative Error | Time per Factor | Use Case |
|-------|---------------|-----------------|----------|
| 10,000 | ±0.7% | 0.2s | Quick screening |
| 50,000 | ±0.3% | 1.4s | Production validation |
| 100,000 | ±0.2% | 2.7s | High precision |
| 1,000,000 | ±0.1% | 33s | Research grade |
```

---

## 验收标准

### 功能验收

- [ ] 新增20+个因子全部实现并通过测试
- [ ] 总因子数达到50+
- [ ] 所有因子可以在100万路径下完成验证
- [ ] 验证耗时控制在30分钟内

### 性能验收

- [ ] 向量化计算速度 ≥ 50,000 paths/秒
- [ ] 内存使用 ≤ 16GB
- [ ] 百万路径验证不崩溃

### 文档验收

- [ ] CLAUDE.md更新验证规模参数
- [ ] 新增因子有完整文档
- [ ] 所有测试结果可重现

---

## 执行选项

**计划已完成并保存到 `docs/plans/2026-03-06-implement-more-factors-and-mega-scale-validation.md`**

**两个执行选项：**

1. **Subagent-Driven (this session)** - 我派遣新鲜子代理逐个任务执行，期间审查，快速迭代

2. **Parallel Session (separate)** - 你开启新会话使用 executing-plans，批量执行带检查点

**推荐：** 选择 **Subagent-Driven** 因为我可以在每个Phase完成后审查代码质量，确保向量化优化正确实现。

**预计总时间：** 4-6小时（包括百万路径验证运行时间）

**你想开始执行吗？**
