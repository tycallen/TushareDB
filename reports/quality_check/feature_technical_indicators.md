# 技术指标特征体系文档

> 版本: v1.0
> 日期: 2025-01
> 数据源: Tushare daily 日线数据

---

## 概述

本文档定义了一套完整的技术指标特征体系，共计 **60+ 个技术指标特征**，分为 5 大类别：

| 类别 | 特征数量 | 说明 |
|------|----------|------|
| 趋势类指标 | 17 | 识别价格趋势方向和强度 |
| 动量类指标 | 10 | 衡量价格变动速度和力度 |
| 波动率指标 | 11 | 评估价格波动程度 |
| 成交量指标 | 5 | 分析资金流向和交易活跃度 |
| 价格结构特征 | 18 | 描述K线形态和价格位置 |

---

## 1. 趋势类指标

### 1.1 简单移动平均线 (MA)

**定义**: 过去 N 个交易日收盘价的算术平均值

**公式**:
```
MA(n) = Σ(close[i], i=0 to n-1) / n
```

**参数与特征**:
| 特征名 | 周期 | 用途 |
|--------|------|------|
| `ma_5` | 5日 | 超短期趋势，周线均线 |
| `ma_10` | 10日 | 短期趋势，半月线 |
| `ma_20` | 20日 | 短中期趋势，月线 |
| `ma_60` | 60日 | 中期趋势，季线 |
| `ma_120` | 120日 | 中长期趋势，半年线 |
| `ma_250` | 250日 | 长期趋势，年线 |

**交易信号**:
- 金叉 (短期MA上穿长期MA): 买入信号
- 死叉 (短期MA下穿长期MA): 卖出信号
- 价格突破年线: 牛熊分界

**Python 代码**:
```python
def calculate_ma(series: pd.Series, periods: List[int] = [5, 10, 20, 60, 120, 250]) -> pd.DataFrame:
    result = pd.DataFrame(index=series.index)
    for period in periods:
        result[f'ma_{period}'] = series.rolling(window=period, min_periods=1).mean()
    return result
```

---

### 1.2 指数移动平均线 (EMA)

**定义**: 对近期价格赋予更高权重的移动平均

**公式**:
```
EMA(t) = α * price(t) + (1-α) * EMA(t-1)
其中 α = 2 / (period + 1)
```

**特征**:
| 特征名 | 周期 | 用途 |
|--------|------|------|
| `ema_12` | 12日 | MACD 快线基础 |
| `ema_26` | 26日 | MACD 慢线基础 |

**与 MA 的区别**:
- EMA 对近期价格更敏感
- 适合短期交易者
- 滞后性比 MA 小

**Python 代码**:
```python
def calculate_ema(series: pd.Series, periods: List[int] = [12, 26]) -> pd.DataFrame:
    result = pd.DataFrame(index=series.index)
    for period in periods:
        result[f'ema_{period}'] = series.ewm(span=period, adjust=False).mean()
    return result
```

---

### 1.3 MACD (移动平均收敛发散)

**定义**: 通过两条 EMA 的差异来判断趋势

**公式**:
```
MACD Line = EMA(12) - EMA(26)
Signal Line = EMA(MACD, 9)
Histogram = MACD Line - Signal Line
```

**特征**:
| 特征名 | 说明 |
|--------|------|
| `macd` | MACD 差离值 (DIF) |
| `macd_signal` | 信号线 (DEA) |
| `macd_hist` | MACD 柱状图 (MACD Bar) |

**交易信号**:
- MACD 上穿 Signal: 买入
- MACD 下穿 Signal: 卖出
- Histogram 由负转正: 上涨动能增强
- 背离: 价格新高但 MACD 未新高 (顶背离)

**Python 代码**:
```python
def calculate_macd(series: pd.Series, fast=12, slow=26, signal=9) -> pd.DataFrame:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame({
        'macd': macd_line,
        'macd_signal': signal_line,
        'macd_hist': histogram
    })
```

---

### 1.4 ADX (平均趋向指数)

**定义**: 衡量趋势强度，不判断方向

**公式**:
```
+DM = high(t) - high(t-1)  (若为正且 > -DM)
-DM = low(t-1) - low(t)    (若为正且 > +DM)
TR = max(high-low, |high-close_prev|, |low-close_prev|)
+DI = 100 * smoothed(+DM) / smoothed(TR)
-DI = 100 * smoothed(-DM) / smoothed(TR)
DX = 100 * |+DI - -DI| / (+DI + -DI)
ADX = smoothed(DX)
```

**特征**:
| 特征名 | 说明 |
|--------|------|
| `adx` | 趋势强度 (0-100) |
| `plus_di` | 上升方向指标 |
| `minus_di` | 下降方向指标 |

**解读**:
- ADX < 20: 无明显趋势 (震荡市)
- ADX 20-40: 趋势形成中
- ADX > 40: 强趋势
- +DI > -DI: 上升趋势占优
- -DI > +DI: 下降趋势占优

---

### 1.5 Aroon 指标

**定义**: 判断趋势开始和结束的时机

**公式**:
```
Aroon Up = 100 * (period - days_since_highest_high) / period
Aroon Down = 100 * (period - days_since_lowest_low) / period
Aroon Oscillator = Aroon Up - Aroon Down
```

**特征**:
| 特征名 | 说明 |
|--------|------|
| `aroon_up` | 上升阿隆 (0-100) |
| `aroon_down` | 下降阿隆 (0-100) |
| `aroon_osc` | 阿隆振荡器 (-100 to 100) |

**解读**:
- Aroon Up > 70: 新高频繁，上升趋势
- Aroon Down > 70: 新低频繁，下降趋势
- Oscillator > 0: 上升趋势
- Oscillator < 0: 下降趋势

---

## 2. 动量类指标

### 2.1 RSI (相对强弱指数)

**定义**: 衡量价格上涨力度与下跌力度的比值

**公式**:
```
RS = avg_gain / avg_loss
RSI = 100 - (100 / (1 + RS))
```

**特征**:
| 特征名 | 周期 | 用途 |
|--------|------|------|
| `rsi_6` | 6日 | 短期超买超卖 |
| `rsi_14` | 14日 | 标准RSI |
| `rsi_24` | 24日 | 长周期RSI |

**解读**:
- RSI > 70: 超买区域，可能回调
- RSI < 30: 超卖区域，可能反弹
- RSI 50 为多空分界线
- 背离信号更有参考价值

**Python 代码**:
```python
def calculate_rsi(series: pd.Series, periods=[6, 14, 24]) -> pd.DataFrame:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    result = pd.DataFrame()
    for period in periods:
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        result[f'rsi_{period}'] = 100 - (100 / (1 + rs))
    return result
```

---

### 2.2 Stochastic Oscillator (随机指标)

**定义**: 收盘价在一定周期内价格区间的相对位置

**公式**:
```
%K = 100 * (close - lowest_low) / (highest_high - lowest_low)
%K_smooth = SMA(%K, 3)
%D = SMA(%K_smooth, 3)
```

**特征**:
| 特征名 | 说明 |
|--------|------|
| `stoch_k` | 快线 %K |
| `stoch_d` | 慢线 %D |

**参数**: (14, 3, 3) = K周期, K平滑, D平滑

**解读**:
- %K > 80: 超买
- %K < 20: 超卖
- %K 上穿 %D: 买入信号
- %K 下穿 %D: 卖出信号

---

### 2.3 Williams %R

**定义**: 类似 Stochastic，但反向表示

**公式**:
```
%R = -100 * (highest_high - close) / (highest_high - lowest_low)
```

**特征**: `williams_r`

**参数**: 14 日

**解读**:
- %R > -20: 超买
- %R < -80: 超卖
- 范围: -100 到 0

---

### 2.4 ROC (变动率)

**定义**: 当前价格相对于 N 日前价格的变化百分比

**公式**:
```
ROC = (close - close[n]) / close[n] * 100
```

**特征**: `roc`

**参数**: 12 日

**用途**:
- 衡量价格变动速度
- 正值表示上涨，负值表示下跌
- 可用于识别超买超卖

---

### 2.5 CCI (商品通道指数)

**定义**: 价格偏离统计平均值的程度

**公式**:
```
TP = (high + low + close) / 3
CCI = (TP - SMA(TP, 20)) / (0.015 * MeanDeviation)
```

**特征**: `cci`

**参数**: 20 日

**解读**:
- CCI > +100: 超买/强势
- CCI < -100: 超卖/弱势
- 0 为中轴线

---

### 2.6 MFI (资金流量指数)

**定义**: 考虑成交量的 RSI

**公式**:
```
TP = (high + low + close) / 3
Raw Money Flow = TP * volume
Money Ratio = Σ(positive MF) / Σ(negative MF)
MFI = 100 - (100 / (1 + Money Ratio))
```

**特征**: `mfi`

**参数**: 14 日

**解读**:
- MFI > 80: 超买
- MFI < 20: 超卖
- 比 RSI 多考虑了资金因素

---

## 3. 波动率指标

### 3.1 Bollinger Bands (布林带)

**定义**: 以移动平均线为中轨，上下各加减标准差形成通道

**公式**:
```
Middle Band = SMA(close, 20)
Upper Band = Middle + 2 * std(close, 20)
Lower Band = Middle - 2 * std(close, 20)
%B = (close - Lower) / (Upper - Lower)
Bandwidth = (Upper - Lower) / Middle
```

**特征**:
| 特征名 | 说明 |
|--------|------|
| `bb_middle` | 中轨 (20日均线) |
| `bb_upper` | 上轨 |
| `bb_lower` | 下轨 |
| `bb_percent_b` | 价格在带中的位置 (0-1) |
| `bb_bandwidth` | 带宽 (波动率衡量) |

**参数**: (20, 2) = 周期, 标准差倍数

**解读**:
- %B > 1: 突破上轨
- %B < 0: 跌破下轨
- Bandwidth 收窄: 波动率降低，可能即将突破

---

### 3.2 ATR (真实波动幅度)

**定义**: 衡量价格波动的绝对幅度

**公式**:
```
TR = max(high-low, |high-close_prev|, |low-close_prev|)
ATR = Wilder's smoothed average of TR
```

**特征**: `atr`

**参数**: 14 日

**用途**:
- 设置止损位 (如 2 倍 ATR)
- 仓位管理 (ATR 大时减仓)
- 识别波动率变化

**Python 代码**:
```python
def calculate_atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    return atr
```

---

### 3.3 Keltner Channel (肯特纳通道)

**定义**: 使用 ATR 代替标准差的通道指标

**公式**:
```
Middle = EMA(close, 20)
Upper = Middle + 2 * ATR(10)
Lower = Middle - 2 * ATR(10)
```

**特征**:
| 特征名 | 说明 |
|--------|------|
| `kc_middle` | 中轨 (EMA) |
| `kc_upper` | 上轨 |
| `kc_lower` | 下轨 |

**与布林带结合**:
- Squeeze: 布林带在肯特纳通道内收窄
- 突破: 低波动后的趋势启动信号

---

### 3.4 Historical Volatility (历史波动率)

**定义**: 基于对数收益率的年化标准差

**公式**:
```
log_return = ln(close / close_prev)
HV = std(log_return, n) * sqrt(252)
```

**特征**:
| 特征名 | 周期 | 用途 |
|--------|------|------|
| `hv_20` | 20日 | 短期波动率 |
| `hv_60` | 60日 | 中期波动率 |

**用途**:
- 期权定价
- 风险评估
- 波动率策略

---

## 4. 成交量指标

### 4.1 OBV (能量潮)

**定义**: 累计成交量，上涨日加成交量，下跌日减成交量

**公式**:
```
if close > close_prev: OBV = OBV_prev + volume
if close < close_prev: OBV = OBV_prev - volume
if close == close_prev: OBV = OBV_prev
```

**特征**: `obv`

**用途**:
- OBV 上升 + 价格上升: 趋势确认
- OBV 背离: 趋势可能反转
- 突破确认

---

### 4.2 Volume MA (成交量均线)

**特征**:
| 特征名 | 周期 |
|--------|------|
| `vol_ma_5` | 5日 |
| `vol_ma_20` | 20日 |

**用途**:
- 判断放量/缩量
- 量价配合分析

---

### 4.3 Volume Ratio (量比)

**定义**: 当日成交量与近期平均成交量的比值

**公式**:
```
VR = volume / avg_volume(5)
```

**特征**: `volume_ratio`

**解读**:
- VR > 2: 明显放量
- VR < 0.5: 明显缩量
- VR ≈ 1: 正常水平

---

### 4.4 VWAP (成交量加权平均价)

**定义**: 以成交量为权重的平均价格

**公式** (日线近似):
```
TP = (high + low + close) / 3
VWAP = Σ(TP * volume) / Σ(volume)
```

**特征**: `vwap`

**参数**: 20 日滚动窗口

**用途**:
- 机构常用的基准价格
- 价格在 VWAP 上方: 买盘占优

---

## 5. 价格结构特征

### 5.1 价格相对均线位置

**定义**: 当前价格偏离均线的百分比

**公式**:
```
pos_ma_n = (close / MA(n) - 1) * 100
```

**特征**:
| 特征名 | 含义 |
|--------|------|
| `pos_ma_5` | 相对5日均线位置 (%) |
| `pos_ma_10` | 相对10日均线位置 (%) |
| `pos_ma_20` | 相对20日均线位置 (%) |
| `pos_ma_60` | 相对60日均线位置 (%) |

**用途**:
- 正值: 价格在均线上方
- 负值: 价格在均线下方
- 绝对值越大，偏离越远

---

### 5.2 离高低点距离

**定义**: 当前价格与近期高低点的距离

**公式**:
```
dist_high_n = (highest_n - close) / highest_n * 100
dist_low_n = (close - lowest_n) / lowest_n * 100
```

**特征**:
| 特征名 | 含义 |
|--------|------|
| `dist_high_5/20/60` | 距离N日最高点 (%) |
| `dist_low_5/20/60` | 距离N日最低点 (%) |

**用途**:
- 判断价格在区间内的位置
- dist_high 接近 0: 接近高点
- dist_low 接近 0: 接近低点

---

### 5.3 K线形态特征

**特征**:
| 特征名 | 公式 | 含义 |
|--------|------|------|
| `body_ratio` | \|close-open\| / (high-low) | 实体占比 |
| `upper_shadow` | (high-max(open,close)) / (high-low) | 上影线比例 |
| `lower_shadow` | (min(open,close)-low) / (high-low) | 下影线比例 |
| `is_bullish` | close > open | 是否阳线 (0/1) |

**用途**:
- 识别 Doji (实体极小)
- 识别锤子线 (下影线长)
- 识别射击之星 (上影线长)

---

### 5.4 缺口检测

**定义**: 价格跳空形成的缺口

**公式**:
```
Gap Up: low > prev_high
Gap Down: high < prev_low
Gap Size: 缺口大小占前收盘价的百分比
```

**特征**:
| 特征名 | 含义 |
|--------|------|
| `gap_up` | 向上缺口 (0/1) |
| `gap_down` | 向下缺口 (0/1) |
| `gap_up_size` | 向上缺口大小 (%) |
| `gap_down_size` | 向下缺口大小 (%) |

**用途**:
- 突破缺口: 趋势启动
- 衰竭缺口: 趋势结束信号
- 岛形反转

---

## 6. 使用示例

### 6.1 完整计算示例

```python
import duckdb
import pandas as pd
from technical_indicators import calculate_all_features

# 连接数据库
conn = duckdb.connect('tushare.db', read_only=True)

# 获取数据 (带复权因子)
df = conn.execute("""
    SELECT d.*, a.adj_factor
    FROM daily d
    LEFT JOIN adj_factor a ON d.ts_code = a.ts_code AND d.trade_date = a.trade_date
    WHERE d.ts_code = '000001.SZ'
    ORDER BY d.trade_date
""").fetchdf()

# 计算所有特征
features = calculate_all_features(df, use_adj=True)

# 查看特征
print(features.tail())
```

### 6.2 单个指标计算示例

```python
from technical_indicators import calculate_macd, calculate_rsi, calculate_bollinger_bands

# MACD
macd_df = calculate_macd(df['close'], fast=12, slow=26, signal=9)

# RSI
rsi_df = calculate_rsi(df['close'], periods=[6, 14, 24])

# Bollinger Bands
bb_df = calculate_bollinger_bands(df['close'], period=20, std_dev=2)
```

### 6.3 交易信号生成示例

```python
# 金叉死叉信号
features['ma_cross'] = np.where(
    (features['ma_5'] > features['ma_20']) &
    (features['ma_5'].shift(1) <= features['ma_20'].shift(1)),
    1,  # 金叉
    np.where(
        (features['ma_5'] < features['ma_20']) &
        (features['ma_5'].shift(1) >= features['ma_20'].shift(1)),
        -1,  # 死叉
        0
    )
)

# RSI 超买超卖信号
features['rsi_signal'] = np.where(
    features['rsi_14'] > 70, -1,  # 超买
    np.where(features['rsi_14'] < 30, 1, 0)  # 超卖
)
```

---

## 7. 注意事项

### 7.1 复权处理

- **必须使用复权价格**计算技术指标
- 推荐使用**后复权** (当前价格不变，历史价格调整)
- 复权公式: `adj_price = price * adj_factor / latest_adj_factor`

### 7.2 边界情况

1. **新股**: 数据不足时，指标会有 NaN
2. **停牌**: 停牌期间数据可能重复或缺失
3. **涨跌停**: 可能导致指标异常值

### 7.3 参数选择

| 周期类型 | 日线 | 周线 | 月线 |
|----------|------|------|------|
| 短期 | 5-10 | 4 | 3 |
| 中期 | 20-60 | 10-13 | 6 |
| 长期 | 120-250 | 26-52 | 12 |

### 7.4 指标相关性

部分指标高度相关，需注意多重共线性:
- MA 系列之间
- RSI 不同周期之间
- Stochastic 与 Williams %R

---

## 8. 特征列表汇总

共计 **60 个特征**:

```
趋势类 (17):
  ma_5, ma_10, ma_20, ma_60, ma_120, ma_250
  ema_12, ema_26
  macd, macd_signal, macd_hist
  adx, plus_di, minus_di
  aroon_up, aroon_down, aroon_osc

动量类 (10):
  rsi_6, rsi_14, rsi_24
  stoch_k, stoch_d
  williams_r
  roc
  cci
  mfi

波动率 (11):
  bb_middle, bb_upper, bb_lower, bb_percent_b, bb_bandwidth
  atr
  kc_middle, kc_upper, kc_lower
  hv_20, hv_60

成交量 (5):
  obv
  vol_ma_5, vol_ma_20
  volume_ratio
  vwap

价格结构 (18):
  pos_ma_5, pos_ma_10, pos_ma_20, pos_ma_60
  dist_high_5, dist_high_20, dist_high_60
  dist_low_5, dist_low_20, dist_low_60
  body_ratio, upper_shadow, lower_shadow, is_bullish
  gap_up, gap_down, gap_up_size, gap_down_size
```

---

## 9. 特征统计分析结果

基于 10 只代表性股票 (平安银行、贵州茅台、五粮液、中国平安、美的集团、招商银行、海康威视、宁德时代、隆基绿能、京东方A) 的统计分析：

### 9.1 数据概览

| 指标 | 数值 |
|------|------|
| 股票数量 | 10 只 |
| 总数据量 | 45,943 条记录 |
| 特征数量 | 65 个 |
| 时间跨度 | 约 24 年 (2001-2025) |

### 9.2 缺失率统计

| 类别 | 平均缺失率 | 说明 |
|------|------------|------|
| 趋势类 | 0.63% | 主要来自均线计算需要的前期数据 |
| 动量类 | 0.98% | RSI/Stochastic 等需要一定预热期 |
| 波动率 | 1.32% | 历史波动率 (60日) 缺失率最高 (3.01%) |
| 成交量 | 0.60% | 缺失率最低 |
| 价格结构 | 1.11% | 主要来自高低点距离计算 |

**缺失原因分析**:
- 新股上市初期数据不足
- 滚动窗口计算需要前期数据
- 复权因子偶有缺失

### 9.3 异常值分析

异常值采用 IQR (四分位距) 方法检测: `outlier if x < Q1-1.5*IQR or x > Q3+1.5*IQR`

**高异常值率特征** (>10%):

| 特征 | 异常值率 | 原因分析 |
|------|----------|----------|
| macd_hist | 23.93% | 不同股票价格差异大导致 MACD 值分布差异 |
| macd_signal | 23.06% | 同上 |
| macd | 23.01% | 同上 |
| 价格类 (open/high/low/close) | ~15% | 不同股价量级差异 (茅台2000+ vs 平安10+) |
| 均线类 | ~13.7% | 同价格类 |

**低异常值率特征** (<1%):

| 特征 | 异常值率 | 说明 |
|------|----------|------|
| rsi_14 | 0.59% | 标准化到 0-100，分布良好 |
| bb_percent_b | 0.00% | 标准化特征 |
| aroon 系列 | 0.00% | 固定范围 0-100 |
| stochastic | 0.00% | 固定范围 0-100 |

### 9.4 关键特征分布

| 特征 | 均值 | 标准差 | 中位数 | Q25-Q75 |
|------|------|--------|--------|---------|
| rsi_14 | 51.27 | 12.57 | 50.85 | 42.6 - 59.7 |
| bb_percent_b | 0.52 | 0.33 | 0.53 | 0.26 - 0.78 |
| adx | 25.68 | 10.48 | 23.38 | 17.8 - 31.6 |
| volume_ratio | 1.06 | 0.76 | 0.90 | 0.64 - 1.24 |
| body_ratio | 0.56 | 0.27 | 0.58 | 0.35 - 0.78 |

### 9.5 数据分布特征

| 指标 | 数量 |
|------|------|
| 高偏度特征 (\|skew\| > 2) | 33 个 |
| 高峰度特征 (kurtosis > 10) | 35 个 |

**说明**: 高偏度/峰度主要来自价格相关特征，不同股票价格量级差异导致。建议在建模前进行：
- 对数变换 (价格、成交量)
- 标准化 (Z-score 或 Min-Max)
- 按股票分组标准化

### 9.6 特征质量建议

1. **推荐直接使用的特征** (标准化良好):
   - RSI 系列
   - Stochastic %K, %D
   - Williams %R
   - Bollinger %B
   - Aroon 系列
   - ADX 及 DI 指标

2. **建议标准化后使用的特征**:
   - 均线偏离度 (pos_ma_*)
   - 高低点距离 (dist_*)
   - K线形态特征

3. **建议按股票分组或变换后使用**:
   - 价格类指标 (MA, EMA, BB bands)
   - MACD 系列
   - ATR, OBV
   - 历史波动率

---

## 附录 A: 代码文件

完整代码实现见: `technical_indicators.py`

该文件包含:
- 所有指标的独立计算函数
- `calculate_all_features()` 综合计算函数
- `get_feature_list()` 特征分类列表
- 测试代码

---

## 附录 B: 统计分析文件

统计分析脚本: `analyze_feature_statistics.py`

输出文件:
- `feature_statistics.csv` - 完整特征统计表
- `missing_patterns.csv` - 缺失值模式分析
- `sample_features.csv` - 特征样例数据
