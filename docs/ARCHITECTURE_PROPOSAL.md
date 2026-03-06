# 蒙特卡洛模拟架构深度优化方案

## 当前问题分析

### 核心矛盾

```
GBM模拟器输出: 只有 Close 价格矩阵 (n_paths, n_steps)
                    ↓
因子需求: 完整的 OHLCV (Open, High, Low, Close, Volume)
                    ↓
当前做法: 只用 Path 0 估算 → 乘以 n_paths (严重错误!)
```

### 问题因子分类

| 类型 | 需要的数据 | 当前状态 | 问题 |
|------|-----------|---------|------|
| **趋势型** | Close only | ✅ 向量化 | 正常 |
| **动量型** | Close only | ✅ 向量化 | 正常 |
| **波动型** | High, Low, Close | ⚠️ 部分支持 | 简化估算 |
| **蜡烛图型** | Open, High, Low, Close | ❌ 单路径 | **严重错误** |
| **成交量型** | Volume (+OHLC) | ❌ 忽略 | 完全失效 |

---

## 方案对比

### 方案1: 多路径采样（快速修复）

```python
def calculate_p_random_sampling(factor, price_matrix, n_sample=5000):
    """随机采样多条路径估算 P_random"""
    n_paths, n_steps = price_matrix.shape
    sample_indices = np.random.choice(n_paths, size=min(n_sample, n_paths), replace=False)

    total_signals = 0
    for idx in sample_indices:
        # 为单条路径生成完整 OHLC
        ohlc = generate_ohlc_single_path(price_matrix[idx])
        signals = factor.evaluate(ohlc)
        total_signals += signals.sum()

    # 从样本推断总体
    p_sample = total_signals / (len(sample_indices) * n_steps)
    return p_sample
```

**优点:**
- 实现简单，2小时可完成
- 统计学上正确
- 计算量可控 (5000条路径 vs 100000条)

**缺点:**
- 仍有抽样误差
- 无法利用向量化加速
- 每个因子耗时 ~5-10秒

**适用场景:** 紧急修复，立即可用

---

### 方案2: 向量化OHLC生成（推荐方案）

```python
class VectorizedOHLCGenerator:
    """向量化生成 OHLC 数据"""

    def generate(self, price_matrix: np.ndarray) -> Dict[str, np.ndarray]:
        """
        输入: price_matrix (n_paths, n_steps) - Close prices
        输出: {
            'open': (n_paths, n_steps),
            'high': (n_paths, n_steps),
            'low': (n_paths, n_steps),
            'close': (n_paths, n_steps),
            'volume': (n_paths, n_steps)
        }
        """
        n_paths, n_steps = price_matrix.shape

        # Open = 前一日 Close (向量化实现)
        open_matrix = np.roll(price_matrix, 1, axis=1)
        open_matrix[:, 0] = price_matrix[:, 0] * 0.99

        # High/Low 基于随机波动 (向量化)
        np.random.seed(42)
        daily_vol = np.abs(np.random.randn(n_paths, n_steps)) * 0.01
        high_matrix = price_matrix * (1 + daily_vol)
        low_matrix = price_matrix * (1 - daily_vol)

        # Volume 随机生成
        volume_matrix = np.random.randint(1000000, 10000000, size=(n_paths, n_steps))

        return {
            'open': open_matrix,
            'high': high_matrix,
            'low': low_matrix,
            'close': price_matrix,
            'volume': volume_matrix
        }
```

**然后实现向量化因子:**

```python
def doji_vectorized(ohlc_dict: Dict[str, np.ndarray]) -> np.ndarray:
    """向量化十字星检测"""
    open_mat = ohlc_dict['open']
    high_mat = ohlc_dict['high']
    low_mat = ohlc_dict['low']
    close_mat = ohlc_dict['close']

    # 矩阵运算: body = |close - open|
    body = np.abs(close_mat - open_mat)
    range_size = high_mat - low_mat

    # 向量化条件判断
    is_doji = body < (range_size * 0.1)

    return is_doji  # (n_paths, n_steps) 布尔矩阵
```

**优点:**
- 真正的向量化，速度极快 (<0.1s)
- 利用全部 100K 路径，无抽样误差
- 代码优雅，易于维护

**缺点:**
- 开发工作量大 (~2-3天)
- 需要为每个非向量化因子重写
- 复杂逻辑（如KDJ的递归计算）较难向量化

**适用场景:** 生产环境，追求极致性能

---

### 方案3: 分层模拟引擎（终极方案）

```
Layer 1: GBM (Close)
    ↓
Layer 2: Stochastic Volatility Model (生成 High/Low)
    - 使用局部波动率
    - 考虑日内趋势
    - 保持与Close的一致性
    ↓
Layer 3: Market Microstructure (生成 Volume)
    - 基于波动率的成交量
    - 考虑价格变动方向
    ↓
Layer 4: Factor Calculation
    - 全向量化运算
```

**具体实现:**

```python
class HierarchicalSimulator:
    """分层模拟引擎"""

    def __init__(self, n_paths=100000, n_steps=252):
        self.n_paths = n_paths
        self.n_steps = n_steps

    def simulate(self, s0, mu, sigma) -> MarketDataCube:
        """
        生成完整的市场数据立方体

        Returns: MarketDataCube {
            'close': (n_paths, n_steps),
            'high': (n_paths, n_steps),
            'low': (n_paths, n_steps),
            'open': (n_paths, n_steps),
            'volume': (n_paths, n_steps),
            'volatility_regime': (n_paths, n_steps)  # 波动率状态
        }
        """
        # Layer 1: GBM for Close
        close = self._gbm_simulate(s0, mu, sigma)

        # Layer 2: Generate High/Low using Local Volatility
        high, low = self._generate_hl_with_local_vol(close, sigma)

        # Layer 3: Generate Open (with overnight gap model)
        open_price = self._generate_open_with_overnight_gap(close)

        # Layer 4: Generate Volume (correlated with volatility)
        volume = self._generate_volume_with_volatility_correlation(
            close, high, low
        )

        return MarketDataCube(close, high, low, open_price, volume)

    def _generate_hl_with_local_vol(self, close, base_sigma):
        """
        使用局部波动率生成 High/Low

        模型: High_t = Close_t * exp(|Z_1| * local_vol)
              Low_t = Close_t * exp(-|Z_2| * local_vol)

        local_vol 基于近期 realized volatility
        """
        # 计算局部波动率 (滚动窗口)
        log_returns = np.diff(np.log(close), axis=1, prepend=np.log(close[:, :1]))
        # 使用卷积计算滚动波动率
        window = 20
        local_var = np.convolve(log_returns**2, np.ones(window)/window, mode='same')
        local_vol = np.sqrt(local_var * 252)  # 年化

        # 生成 High/Low
        Z_high = np.abs(np.random.randn(self.n_paths, self.n_steps))
        Z_low = np.abs(np.random.randn(self.n_paths, self.n_steps))

        high = close * np.exp(Z_high * local_vol / np.sqrt(252))
        low = close * np.exp(-Z_low * local_vol / np.sqrt(252))

        return high, low

    def _generate_volume_with_volatility_correlation(
        self, close, high, low, correlation=0.6
    ):
        """
        生成与波动率相关的成交量

        模型: Volume ~ Exp(μ_vol + ρ * volatility + ε)
        """
        # 计算日内波动率
        intraday_vol = (high - low) / close

        # 基础成交量
        base_volume = 5000000

        # 波动率影响
        vol_effect = correlation * np.log(intraday_vol * 100 + 1)

        # 随机噪声
        noise = np.random.randn(self.n_paths, self.n_steps) * 0.3

        # 合成成交量
        log_volume = np.log(base_volume) + vol_effect + noise
        volume = np.exp(log_volume).astype(int)

        return np.clip(volume, 100000, 100000000)
```

**优点:**
- 生成的数据更真实（High/Low/Close关系合理）
- Volume 与波动率相关，更符合市场
- 为复杂因子提供准确输入
- 可扩展性强（未来可加入更多市场微观结构）

**缺点:**
- 开发复杂 (~1-2周)
- 计算量增加 ~30%
- 需要验证生成的数据是否保持统计特性

**适用场景:** 学术研究、对冲基金级应用

---

## 决策矩阵

| 维度 | 方案1 (采样) | 方案2 (向量化) | 方案3 (分层) |
|------|------------|--------------|------------|
| 开发时间 | 2小时 | 2-3天 | 1-2周 |
| 运行速度 | 5-10s/因子 | 0.1s/因子 | 0.2s/因子 |
| 精度 | 95% | 100% | 100%+更真实 |
| 维护难度 | 低 | 中 | 高 |
| 扩展性 | 差 | 好 | 极好 |
| 数据质量 | 一般 | 好 | 优秀 |

---

## 我的建议

### 短期（今天完成）: 方案1 + 方案2混合

```python
class AdaptiveSignalDetector:
    """自适应信号检测器"""

    def detect(self, price_matrix, factor):
        if factor.is_vectorized:
            # 方案2: 向量化
            return self._vectorized_detect(price_matrix, factor)
        elif factor.complexity == 'high':
            # 方案1: 采样 (KDJ等复杂因子)
            return self._sampled_detect(price_matrix, factor, n_sample=10000)
        else:
            # 方案2: 向量化OHLC生成 + 检测
            ohlc = self._generate_ohlc_vectorized(price_matrix)
            return self._vectorized_detect_with_ohlc(ohlc, factor)
```

### 中期（本周完成）: 完整方案2

为所有30个因子实现向量化版本。

### 长期（本月完成）: 方案3

实现分层模拟引擎，用于论文发表和生产环境。

---

## 立即实施的最佳实践

如果你想现在得到正确结果，使用这个修正版：

```python
def calculate_p_random_correct(factor, price_matrix, min_sample=10000):
    """
    正确计算 P_random 的方法

    策略:
    1. 对于简单因子: 向量化计算
    2. 对于复杂因子: 分层采样
    """
    n_paths, n_steps = price_matrix.shape

    # 确定采样数（至少 min_sample，最多 n_paths）
    n_sample = min(min_sample, n_paths)

    # 分层采样：确保覆盖不同路径特征
    # 按最终收益率排序，分层采样
    final_returns = price_matrix[:, -1] / price_matrix[:, 0]
    sorted_indices = np.argsort(final_returns)

    # 分层：上涨路径、下跌路径、震荡路径
    n_strata = 10
    strata_size = n_paths // n_strata
    sample_indices = []

    for i in range(n_strata):
        start = i * strata_size
        end = start + strata_size if i < n_strata - 1 else n_paths
        stratum_indices = sorted_indices[start:end]
        n_from_stratum = max(1, int(n_sample / n_strata))
        sample_indices.extend(
            np.random.choice(stratum_indices, size=min(n_from_stratum, len(stratum_indices)), replace=False)
        )

    # 在采样路径上计算信号
    total_signals = 0
    for idx in sample_indices:
        ohlc = generate_realistic_ohlc(price_matrix[idx])
        signals = factor.evaluate(ohlc)
        total_signals += signals.sum()

    # 推断总体
    p_strata = total_signals / (len(sample_indices) * n_steps)

    # 计算标准误
    se = np.sqrt(p_strata * (1 - p_strata) / (len(sample_indices) * n_steps))

    return p_strata, se
```

这样可以得到统计学上正确的结果，同时控制计算时间。

你想让我先实施哪个方案？
