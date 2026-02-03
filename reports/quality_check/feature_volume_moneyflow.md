# 量价和资金流特征工程文档

## 概述

本文档详细描述了基于 DuckDB 数据库的量价特征和资金流特征的定义、计算公式和实现代码。

**数据源:**
- `daily`: 日线行情 (open, high, low, close, vol, amount)
- `daily_basic`: 每日指标 (turnover_rate, volume_ratio, pe, pb, ps, etc.)
- `moneyflow`: 资金流向 (buy_sm_vol, sell_sm_vol, buy_md_vol, sell_md_vol, buy_lg_vol, sell_lg_vol, buy_elg_vol, sell_elg_vol)
- `moneyflow_dc`: 东财资金流向
- `stk_factor_pro`: 技术因子 (包含预计算的技术指标)

**实现文件:**
- `/Users/allen/workspace/python/stock/Tushare-DuckDB/scripts/feature_volume_moneyflow.py`

---

## 一、成交量特征

### 1.1 量比 (Volume Ratio)

**定义:** 当日成交量与N日平均成交量的比值，衡量成交活跃程度。

**公式:**
```
量比 = 当日成交量 / MA(成交量, N)_shift(1)
```

**解读:**
- 量比 > 1: 放量，成交活跃
- 量比 < 1: 缩量，成交清淡
- 量比 > 2: 显著放量
- 量比 < 0.5: 显著缩量

**代码实现:**
```python
def calc_volume_ratio(vol: pd.Series, n: int = 5) -> pd.Series:
    """
    计算量比

    Args:
        vol: 成交量序列
        n: 平均天数，默认5日

    Returns:
        量比序列
    """
    ma_vol = vol.rolling(n).mean().shift(1)  # 使用前N日均量
    return vol / ma_vol
```

**使用示例:**
```python
df['vol_ratio_5d'] = calc_volume_ratio(df['vol'], 5)
df['vol_ratio_10d'] = calc_volume_ratio(df['vol'], 10)
df['vol_ratio_20d'] = calc_volume_ratio(df['vol'], 20)
```

---

### 1.2 换手率变化 (Turnover Rate Change)

**定义:** 当日换手率与N日平均换手率的比值。

**公式:**
```
换手率变化 = 当日换手率 / MA(换手率, N)_shift(1)
```

**数据来源:** `daily_basic.turnover_rate`

**代码实现:**
```python
def calc_turnover_change(turnover_rate: pd.Series, n: int = 5) -> pd.Series:
    """
    计算换手率变化
    """
    ma_turnover = turnover_rate.rolling(n).mean().shift(1)
    return turnover_rate / ma_turnover
```

---

### 1.3 成交量突变检测 (Volume Spike Detection)

**定义:** 检测成交量是否出现异常放大。

**公式:**
```
Volume Spike = 1  if 当日成交量 > N日均量 * threshold
             = 0  otherwise
```

**默认参数:** N=20, threshold=2.0

**代码实现:**
```python
def detect_volume_spike(vol: pd.Series, n: int = 20, threshold: float = 2.0) -> pd.Series:
    """
    检测成交量突变
    """
    ma_vol = vol.rolling(n).mean().shift(1)
    spike = vol > (ma_vol * threshold)
    return spike.astype(int)
```

---

### 1.4 量价背离 (Volume-Price Divergence)

**定义:** 检测价格与成交量的背离现象。

**类型:**
- **顶背离 (+1):** 价格创N日新高，但成交量萎缩 (< 70% 均量)
- **底背离 (-1):** 价格创N日新低，但成交量放大 (> 130% 均量)
- **无背离 (0):** 其他情况

**公式:**
```
顶背离: close == MAX(close, N) AND vol < MA(vol, N) * 0.7
底背离: close == MIN(close, N) AND vol > MA(vol, N) * 1.3
```

**代码实现:**
```python
def calc_volume_price_divergence(close: pd.Series, vol: pd.Series, n: int = 20) -> pd.Series:
    """
    计算量价背离

    Returns:
        1: 顶背离, -1: 底背离, 0: 无背离
    """
    price_high = close == close.rolling(n).max()
    price_low = close == close.rolling(n).min()
    vol_ma = vol.rolling(n).mean()
    vol_shrink = vol < vol_ma * 0.7
    vol_expand = vol > vol_ma * 1.3

    divergence = pd.Series(0, index=close.index)
    divergence[price_high & vol_shrink] = 1    # 顶背离
    divergence[price_low & vol_expand] = -1    # 底背离
    return divergence
```

---

### 1.5 连续放量/缩量天数 (Consecutive Volume Days)

**定义:** 成交量连续增加或减少的天数。

**公式:**
```
放量: vol[t] > vol[t-1]
缩量: vol[t] < vol[t-1]
```

**代码实现:**
```python
def calc_consecutive_volume_days(vol: pd.Series, n: int = 5) -> Tuple[pd.Series, pd.Series]:
    """
    计算连续放量/缩量天数

    Returns:
        (连续放量天数, 连续缩量天数)
    """
    vol_change = vol.diff()
    increase = (vol_change > 0).astype(int)
    decrease = (vol_change < 0).astype(int)

    def count_consecutive(series):
        result = []
        count = 0
        for val in series:
            if val == 1:
                count += 1
            else:
                count = 0
            result.append(count)
        return pd.Series(result, index=series.index)

    return count_consecutive(increase), count_consecutive(decrease)
```

---

## 二、价格特征

### 2.1 振幅 (Amplitude)

**定义:** 当日价格波动幅度占前收盘价的百分比。

**公式:**
```
振幅 = (最高价 - 最低价) / 前收盘价 × 100%
```

**代码实现:**
```python
def calc_amplitude(high: pd.Series, low: pd.Series, pre_close: pd.Series) -> pd.Series:
    """
    计算振幅 (百分比)
    """
    return (high - low) / pre_close * 100
```

---

### 2.2 连涨/连跌天数 (Consecutive Up/Down Days)

**定义:** 连续上涨或下跌的交易日数量。

**公式:**
```
连涨: pct_chg > 0 的连续天数
连跌: pct_chg < 0 的连续天数
```

**代码实现:**
```python
def calc_consecutive_days(pct_chg: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    计算连涨/连跌天数

    Returns:
        (连涨天数, 连跌天数)
    """
    up = (pct_chg > 0).astype(int)
    down = (pct_chg < 0).astype(int)

    def count_consecutive(series):
        result = []
        count = 0
        for val in series:
            count = count + 1 if val == 1 else 0
            result.append(count)
        return pd.Series(result, index=series.index)

    return count_consecutive(up), count_consecutive(down)
```

---

### 2.3 价格动量 (Price Momentum)

**定义:** 不同周期的收益率，衡量价格趋势强度。

**公式:**
```
动量_N = (当前价格 / N日前价格 - 1) × 100%
```

**常用周期:** 5日、10日、20日、60日

**代码实现:**
```python
def calc_momentum(close: pd.Series, periods: List[int] = [5, 10, 20, 60]) -> pd.DataFrame:
    """
    计算价格动量 (收益率)
    """
    result = pd.DataFrame(index=close.index)
    for n in periods:
        result[f'momentum_{n}d'] = (close / close.shift(n) - 1) * 100
    return result
```

---

### 2.4 相对强度 (Relative Strength)

**定义:** 个股相对于指数的超额收益。

**公式:**
```
相对强度 = 个股N日涨幅 - 指数N日涨幅
```

**代码实现:**
```python
def calc_relative_strength(stock_close: pd.Series, index_close: pd.Series, n: int = 20) -> pd.Series:
    """
    计算相对强度 (相对指数)
    """
    stock_ret = (stock_close / stock_close.shift(n) - 1) * 100
    index_ret = (index_close / index_close.shift(n) - 1) * 100
    return stock_ret - index_ret
```

---

## 三、估值特征

### 3.1 估值历史分位数 (Valuation Percentile)

**定义:** 当前估值在历史区间内的相对位置。

**公式:**
```
分位数 = COUNT(历史估值 < 当前估值) / (窗口长度 - 1)
```

**数据来源:** `daily_basic.pe_ttm`, `daily_basic.pb`, `daily_basic.ps_ttm`

**常用窗口:** 250日 (约一年)

**代码实现:**
```python
def calc_valuation_percentile(series: pd.Series, window: int = 250) -> pd.Series:
    """
    计算估值指标的历史分位数 (0-1)
    """
    def rolling_percentile(x):
        if len(x) < window:
            return np.nan
        return (x.iloc[-1] > x.iloc[:-1]).sum() / (len(x) - 1)

    return series.rolling(window, min_periods=window).apply(rolling_percentile, raw=False)
```

**解读:**
- 分位数接近 1: 估值处于历史高位
- 分位数接近 0: 估值处于历史低位
- 分位数约 0.5: 估值处于历史中位

---

## 四、主力资金特征

### 4.1 主力净流入 (Main Net Inflow)

**定义:** 大单和特大单的净买入金额/量。

**公式:**
```
主力买入 = 大单买入 + 特大单买入
主力卖出 = 大单卖出 + 特大单卖出
主力净流入 = 主力买入 - 主力卖出
```

**数据来源:** `moneyflow.buy_lg_vol/amount`, `moneyflow.buy_elg_vol/amount`

**代码实现:**
```python
def calc_main_net_inflow(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算主力资金净流入
    """
    result = df.copy()

    # 主力净流入量
    result['main_buy_vol'] = df['buy_lg_vol'].fillna(0) + df['buy_elg_vol'].fillna(0)
    result['main_sell_vol'] = df['sell_lg_vol'].fillna(0) + df['sell_elg_vol'].fillna(0)
    result['main_net_vol'] = result['main_buy_vol'] - result['main_sell_vol']

    # 主力净流入金额 (万元)
    result['main_buy_amount'] = df['buy_lg_amount'].fillna(0) + df['buy_elg_amount'].fillna(0)
    result['main_sell_amount'] = df['sell_lg_amount'].fillna(0) + df['sell_elg_amount'].fillna(0)
    result['main_net_amount'] = result['main_buy_amount'] - result['main_sell_amount']

    return result
```

---

### 4.2 主力净流入率 (Main Net Inflow Rate)

**定义:** 主力净流入占总成交额的比例。

**公式:**
```
主力净流入率 = 主力净流入金额 / 总成交额 × 100%
```

**代码实现:**
```python
def calc_main_net_inflow_rate(df: pd.DataFrame) -> pd.Series:
    """
    计算主力净流入率 (百分比)
    """
    main_net = (df['buy_lg_amount'].fillna(0) + df['buy_elg_amount'].fillna(0) -
               df['sell_lg_amount'].fillna(0) - df['sell_elg_amount'].fillna(0))
    total_amount = df['amount'].fillna(0) / 10  # 单位转换
    return main_net / total_amount * 100
```

---

### 4.3 连续流入天数 (Consecutive Inflow Days)

**定义:** 主力资金连续净流入的天数。

**规则:**
- 正数: 连续净流入天数
- 负数: 连续净流出天数
- 0: 当日无净流入/流出

**代码实现:**
```python
def calc_consecutive_inflow_days(net_inflow: pd.Series) -> pd.Series:
    """
    计算连续流入天数 (负数表示连续流出天数)
    """
    result = []
    count = 0
    prev_sign = 0

    for val in net_inflow:
        if pd.isna(val):
            result.append(0)
            count = 0
            prev_sign = 0
            continue

        current_sign = 1 if val > 0 else (-1 if val < 0 else 0)

        if current_sign == prev_sign and current_sign != 0:
            count = count + current_sign
        elif current_sign != 0:
            count = current_sign
        else:
            count = 0

        result.append(count)
        prev_sign = current_sign

    return pd.Series(result, index=net_inflow.index)
```

---

### 4.4 累计净流入 (Cumulative Net Inflow)

**定义:** N日内主力资金净流入的累计值。

**公式:**
```
累计净流入_N = SUM(主力净流入, N)
```

**常用周期:** 5日、10日、20日

**代码实现:**
```python
def calc_cumulative_inflow(net_inflow: pd.Series, periods: List[int] = [5, 10, 20]) -> pd.DataFrame:
    """
    计算累计净流入
    """
    result = pd.DataFrame(index=net_inflow.index)
    for n in periods:
        result[f'cum_inflow_{n}d'] = net_inflow.rolling(n).sum()
    return result
```

---

## 五、大中小单分布特征

### 5.1 各类型单占比 (Order Type Ratio)

**定义:** 各类型成交单在总成交中的占比。

**单类型定义 (Tushare):**
- 小单 (sm): 小于等于2万股或4万元
- 中单 (md): 2-10万股或4-20万元
- 大单 (lg): 10-50万股或20-100万元
- 特大单 (elg): 大于50万股或100万元

**公式:**
```
小单占比 = (小单买入量 + 小单卖出量) / 总成交量 × 100%
中单占比 = (中单买入量 + 中单卖出量) / 总成交量 × 100%
大单占比 = (大单买入量 + 大单卖出量) / 总成交量 × 100%
特大单占比 = (特大单买入量 + 特大单卖出量) / 总成交量 × 100%
主力占比 = 大单占比 + 特大单占比
```

**代码实现:**
```python
def calc_order_type_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算各类型单占比
    """
    result = df.copy()

    # 总成交量
    total_vol = (df['buy_sm_vol'].fillna(0) + df['sell_sm_vol'].fillna(0) +
                df['buy_md_vol'].fillna(0) + df['sell_md_vol'].fillna(0) +
                df['buy_lg_vol'].fillna(0) + df['sell_lg_vol'].fillna(0) +
                df['buy_elg_vol'].fillna(0) + df['sell_elg_vol'].fillna(0))

    result['sm_vol_ratio'] = (df['buy_sm_vol'].fillna(0) + df['sell_sm_vol'].fillna(0)) / total_vol * 100
    result['md_vol_ratio'] = (df['buy_md_vol'].fillna(0) + df['sell_md_vol'].fillna(0)) / total_vol * 100
    result['lg_vol_ratio'] = (df['buy_lg_vol'].fillna(0) + df['sell_lg_vol'].fillna(0)) / total_vol * 100
    result['elg_vol_ratio'] = (df['buy_elg_vol'].fillna(0) + df['sell_elg_vol'].fillna(0)) / total_vol * 100
    result['main_vol_ratio'] = result['lg_vol_ratio'] + result['elg_vol_ratio']

    return result
```

---

### 5.2 各类型净流入 (Net Inflow by Type)

**定义:** 分别计算各类型单的净买入。

**公式:**
```
小单净流入 = 小单买入 - 小单卖出
中单净流入 = 中单买入 - 中单卖出
大单净流入 = 大单买入 - 大单卖出
特大单净流入 = 特大单买入 - 特大单卖出
```

**代码实现:**
```python
def calc_net_inflow_by_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算各类型单净流入
    """
    result = df.copy()

    # 各类型净流入量
    result['sm_net_vol'] = df['buy_sm_vol'].fillna(0) - df['sell_sm_vol'].fillna(0)
    result['md_net_vol'] = df['buy_md_vol'].fillna(0) - df['sell_md_vol'].fillna(0)
    result['lg_net_vol'] = df['buy_lg_vol'].fillna(0) - df['sell_lg_vol'].fillna(0)
    result['elg_net_vol'] = df['buy_elg_vol'].fillna(0) - df['sell_elg_vol'].fillna(0)

    # 各类型净流入金额
    result['sm_net_amount'] = df['buy_sm_amount'].fillna(0) - df['sell_sm_amount'].fillna(0)
    result['md_net_amount'] = df['buy_md_amount'].fillna(0) - df['sell_md_amount'].fillna(0)
    result['lg_net_amount'] = df['buy_lg_amount'].fillna(0) - df['sell_lg_amount'].fillna(0)
    result['elg_net_amount'] = df['buy_elg_amount'].fillna(0) - df['sell_elg_amount'].fillna(0)

    return result
```

---

## 六、资金流动量特征

### 6.1 资金流强度 (Moneyflow Intensity)

**定义:** 净流入金额占总成交额的比例。

**公式:**
```
资金流强度 = 净流入金额 / 总成交额 × 100%
```

**代码实现:**
```python
def calc_moneyflow_intensity(net_amount: pd.Series, total_amount: pd.Series) -> pd.Series:
    """
    计算资金流强度 (百分比)
    """
    return net_amount / total_amount * 100
```

---

### 6.2 资金流动量 (Moneyflow Momentum)

**定义:** N日净流入的累计值，反映资金流动趋势。

**公式:**
```
资金流动量_N = SUM(净流入, N)
```

**代码实现:**
```python
def calc_moneyflow_momentum(net_inflow: pd.Series, n: int = 5) -> pd.Series:
    """
    计算资金流动量
    """
    return net_inflow.rolling(n).sum()
```

---

### 6.3 资金流转折点 (Moneyflow Turning Point)

**定义:** 检测资金流方向的转变。

**规则:**
- +1: 从净流出转为净流入 (N日累计从负转正)
- -1: 从净流入转为净流出 (N日累计从正转负)
- 0: 无转折

**代码实现:**
```python
def detect_moneyflow_turning_point(net_inflow: pd.Series, n: int = 5) -> pd.Series:
    """
    检测资金流转折点
    """
    cum_flow = net_inflow.rolling(n).sum()
    cum_flow_prev = cum_flow.shift(1)

    turning = pd.Series(0, index=net_inflow.index)
    turning[(cum_flow_prev < 0) & (cum_flow > 0)] = 1   # 从负转正
    turning[(cum_flow_prev > 0) & (cum_flow < 0)] = -1  # 从正转负

    return turning
```

---

## 七、使用示例

### 7.1 完整特征计算

```python
from scripts.feature_volume_moneyflow import VolumeMoneyflowFeatures

db_path = "/Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db"

with VolumeMoneyflowFeatures(db_path) as calc:
    # 计算单只股票的所有特征
    df = calc.calculate_all_features('000001.SZ', '20240101', '20260130')

    # 查看特征列
    print(df.columns.tolist())

    # 查看最新数据
    print(df.tail())
```

### 7.2 自定义特征计算

```python
with VolumeMoneyflowFeatures(db_path) as calc:
    # 加载原始数据
    df = calc.load_merged_data('000001.SZ', '20240101')

    # 计算特定特征
    df['vol_ratio'] = calc.calc_volume_ratio(df['vol'], 5)
    df['amplitude'] = calc.calc_amplitude(df['high'], df['low'], df['pre_close'])

    # 主力资金特征
    df = calc.calc_main_net_inflow(df)
```

### 7.3 批量计算

```python
import duckdb

conn = duckdb.connect(db_path, read_only=True)

# 获取所有有资金流数据的股票
stocks = conn.execute("""
    SELECT DISTINCT ts_code FROM moneyflow
    WHERE buy_lg_vol IS NOT NULL
""").fetchdf()['ts_code'].tolist()

with VolumeMoneyflowFeatures(db_path) as calc:
    for ts_code in stocks[:100]:
        df = calc.calculate_all_features(ts_code, '20240101')
        # 保存或进一步处理
        df.to_parquet(f'features/{ts_code}.parquet')
```

---

## 八、特征统计分布

### 8.1 数据概览

基于 2024-01-01 至 2026-01-30 期间的数据统计:

| 数据表 | 记录数 | 股票数 | 起始日期 | 结束日期 |
|-------|--------|-------|---------|---------|
| daily | 16,486,059 | 5,795 | 20000104 | 20260130 |
| daily_basic | 15,415,921 | 5,482 | 19960605 | 20260130 |
| moneyflow | 13,239,498 | 5,635 | 20100104 | 20260130 |

### 8.2 成交量特征分布

**量比分布 (使用 volume_ratio)**

| 量比区间 | 记录数 | 占比(%) | 市场含义 |
|---------|--------|---------|---------|
| < 0.5 | 100,569 | 3.90 | 极度缩量 |
| 0.5-0.8 | 790,565 | 30.66 | 缩量 |
| 0.8-1.2 | 1,023,063 | 39.68 | 正常 |
| 1.2-2.0 | 517,780 | 20.08 | 温和放量 |
| 2.0-3.0 | 102,542 | 3.98 | 明显放量 |
| > 3.0 | 44,043 | 1.71 | 大幅放量 |

### 8.3 价格特征分布

**振幅分布**

| 振幅区间 | 记录数 | 占比(%) |
|---------|--------|---------|
| 0-2% | 495,695 | 18.25 |
| 2-4% | 1,198,377 | 44.11 |
| 4-6% | 557,005 | 20.50 |
| 6-8% | 223,652 | 8.23 |
| 8-10% | 111,445 | 4.10 |
| 10%+ | 130,565 | 4.81 |

**连涨连跌统计 (基于 stk_factor_pro)**

| 类型 | 平均天数 | 最大天数 | 连续5天以上出现次数 |
|-----|---------|---------|------------------|
| 连涨 | 0.96 | 27 | 58,224 |
| 连跌 | 0.95 | 31 | 54,027 |

### 8.4 估值特征分布 (最新截面数据)

**PE_TTM 分位数**

| P10 | P25 | P50 | P75 | P90 |
|-----|-----|-----|-----|-----|
| 16.30 | 25.81 | 45.64 | 90.24 | 177.26 |

**PB 分位数**

| P10 | P25 | P50 | P75 | P90 |
|-----|-----|-----|-----|-----|
| 1.31 | 2.00 | 3.20 | 5.20 | 8.88 |

### 8.5 资金流特征分布

**按板块统计 (2024年以来)**

| 板块 | 记录数 | 平均主力净流入(万) | 标准差 | 平均主力占比(%) |
|-----|--------|-----------------|-------|---------------|
| 上海 | 1,134,552 | -245.53 | 4,313.69 | 19.23 |
| 深圳主板 | 747,072 | -633.58 | 7,448.80 | 25.82 |
| 创业板 | 685,733 | -661.87 | 7,075.26 | 24.08 |

**主力资金流入统计**

- 平均流入天数占比: **39.29%** (即约4/10个交易日为净流入)
- 占比标准差: 5.68%
- 占比范围: 19.11% ~ 64.16%
- 统计股票数量: 5,201只

### 8.6 单只股票特征分布 (000001.SZ 平安银行示例)

| 特征名称 | 平均值 | 标准差 | 最小值 | 最大值 | 10%分位 | 90%分位 |
|---------|-------|-------|-------|-------|--------|--------|
| vol_ratio_5d | 1.04 | 0.42 | 0.30 | 3.72 | 0.63 | 1.56 |
| amplitude | 1.67 | 0.87 | 0.25 | 10.49 | 0.84 | 2.63 |
| consecutive_up_days | 0.98 | 1.58 | 0 | 10 | 0 | 3 |
| consecutive_down_days | 0.96 | 1.42 | 0 | 7 | 0 | 3 |
| momentum_5d | 0.22 | 2.97 | -11.3 | 24.2 | -2.87 | 3.06 |
| main_net_inflow_rate | -2.37 | 7.89 | -31.7 | 28.7 | -12.9 | 8.4 |
| consecutive_inflow_days | -0.78 | 2.31 | -9 | 9 | -4 | 2 |
| main_vol_ratio | 53.93 | 5.33 | 37.9 | 70.5 | 48.6 | 59.2 |

---

## 九、注意事项

### 9.1 数据对齐

- `daily`, `daily_basic`, `moneyflow` 三表通过 `ts_code` + `trade_date` 连接
- 注意处理缺失值: `moneyflow` 表可能不包含所有交易日
- 资金流数据通常从 2014 年开始

### 9.2 数据延迟

- `moneyflow` 数据通常 T+1 更新
- 交易日判断使用 `trade_cal` 表

### 9.3 特殊情况处理

- 新股上市初期可能无资金流数据
- 停牌期间无数据
- ST股票可能有特殊交易规则

### 9.4 单位说明

| 字段 | 单位 |
|-----|------|
| vol | 手 (100股) |
| amount | 千元 (daily) / 万元 (moneyflow) |
| turnover_rate | % |
| total_mv/circ_mv | 万元 |

---

## 十、扩展方向

1. **行业相对特征**
   - 相对行业的估值分位
   - 行业内资金流强度排名

2. **时序特征**
   - 特征的LSTM/Transformer嵌入
   - 滚动窗口特征

3. **交叉特征**
   - 量价组合特征
   - 资金流-价格联动

4. **分钟级特征**
   - 日内资金流节奏
   - 尾盘主力动向

---

---

## 附录A: 完整特征列表

计算后生成的完整特征列表 (共85列):

**原始数据列:**
- ts_code, trade_date, open, high, low, close, pre_close, change, pct_chg, vol, amount

**估值指标 (from daily_basic):**
- turnover_rate, turnover_rate_f, volume_ratio, pe, pe_ttm, pb, ps, ps_ttm, total_mv, circ_mv

**资金流原始数据 (from moneyflow):**
- buy_sm_vol, buy_sm_amount, sell_sm_vol, sell_sm_amount
- buy_md_vol, buy_md_amount, sell_md_vol, sell_md_amount
- buy_lg_vol, buy_lg_amount, sell_lg_vol, sell_lg_amount
- buy_elg_vol, buy_elg_amount, sell_elg_vol, sell_elg_amount
- net_mf_vol, net_mf_amount

**成交量特征:**
- vol_ratio_5d, vol_ratio_10d, vol_ratio_20d (量比)
- turnover_change_5d (换手率变化)
- vol_spike (量突变, 0/1)
- vol_price_divergence (量价背离, -1/0/1)
- consecutive_vol_up, consecutive_vol_down (连续放量/缩量天数)

**价格特征:**
- amplitude (振幅, %)
- consecutive_up_days, consecutive_down_days (连涨/连跌天数)
- momentum_5d, momentum_10d, momentum_20d, momentum_60d (动量, %)

**估值特征:**
- pe_percentile_250d, pb_percentile_250d, ps_percentile_250d (估值历史分位数)

**主力资金特征:**
- main_buy_vol, main_sell_vol, main_net_vol (主力成交量)
- main_buy_amount, main_sell_amount, main_net_amount (主力成交金额)
- main_net_inflow_rate (主力净流入率, %)
- consecutive_inflow_days (连续流入天数)
- cum_inflow_5d, cum_inflow_10d, cum_inflow_20d (累计净流入)

**大中小单分布特征:**
- sm_vol_ratio, md_vol_ratio, lg_vol_ratio, elg_vol_ratio, main_vol_ratio (各类型占比, %)
- sm_net_vol, md_net_vol, lg_net_vol, elg_net_vol (各类型净流入量)
- sm_net_amount, md_net_amount, lg_net_amount, elg_net_amount (各类型净流入金额)

**资金流动量特征:**
- mf_intensity (资金流强度, %)
- mf_momentum_5d, mf_momentum_10d (资金流动量)
- mf_turning_point (资金流转折点, -1/0/1)

---

## 附录B: SQL查询示例

### B.1 计算主力资金特征 (纯SQL实现)

```sql
WITH stock_features AS (
    SELECT
        m.ts_code,
        m.trade_date,
        -- 主力净流入
        (m.buy_lg_amount + m.buy_elg_amount - m.sell_lg_amount - m.sell_elg_amount) as main_net_amount,
        -- 主力占比
        CASE WHEN (m.buy_sm_vol + m.sell_sm_vol + m.buy_md_vol + m.sell_md_vol +
                   m.buy_lg_vol + m.sell_lg_vol + m.buy_elg_vol + m.sell_elg_vol) > 0
             THEN (m.buy_lg_vol + m.sell_lg_vol + m.buy_elg_vol + m.sell_elg_vol) * 100.0 /
                  (m.buy_sm_vol + m.sell_sm_vol + m.buy_md_vol + m.sell_md_vol +
                   m.buy_lg_vol + m.sell_lg_vol + m.buy_elg_vol + m.sell_elg_vol)
             ELSE NULL END as main_vol_ratio,
        -- 振幅
        CASE WHEN d.pre_close > 0
             THEN (d.high - d.low) / d.pre_close * 100
             ELSE NULL END as amplitude,
        d.pct_chg
    FROM moneyflow m
    JOIN daily d ON m.ts_code = d.ts_code AND m.trade_date = d.trade_date
    WHERE m.trade_date >= '20240101'
        AND m.buy_lg_vol IS NOT NULL AND m.buy_lg_vol > 0
)
SELECT * FROM stock_features;
```

### B.2 查找连续资金流入的股票

```sql
WITH daily_flow AS (
    SELECT
        ts_code,
        trade_date,
        (buy_lg_amount + buy_elg_amount - sell_lg_amount - sell_elg_amount) as main_net,
        CASE WHEN (buy_lg_amount + buy_elg_amount - sell_lg_amount - sell_elg_amount) > 0
             THEN 1 ELSE 0 END as is_inflow
    FROM moneyflow
    WHERE trade_date >= '20260101' AND buy_lg_vol IS NOT NULL
),
consecutive AS (
    SELECT
        ts_code,
        trade_date,
        main_net,
        SUM(CASE WHEN is_inflow = 0 THEN 1 ELSE 0 END)
            OVER (PARTITION BY ts_code ORDER BY trade_date) as grp
    FROM daily_flow
),
streaks AS (
    SELECT
        ts_code,
        MIN(trade_date) as start_date,
        MAX(trade_date) as end_date,
        COUNT(*) as streak_days,
        SUM(main_net) as total_inflow
    FROM consecutive
    WHERE main_net > 0
    GROUP BY ts_code, grp
    HAVING COUNT(*) >= 5
)
SELECT * FROM streaks ORDER BY streak_days DESC LIMIT 20;
```

---

*文档生成时间: 2025-01-31*
*数据库版本: tushare.db*
*实现文件: /Users/allen/workspace/python/stock/Tushare-DuckDB/scripts/feature_volume_moneyflow.py*
