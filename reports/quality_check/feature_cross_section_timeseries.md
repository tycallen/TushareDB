# 横截面与时序特征工程文档

## 概述

本文档定义了量化投资中常用的横截面特征和时序特征，包括完整的定义、计算公式和实现代码。

- **数据库**: `/Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db`
- **数据范围**: 2000-01-04 至 2026-01-30
- **主要数据表**: daily, daily_basic, index_member_all, cyq_perf, moneyflow

---

## 一、横截面特征 (Cross-sectional Features)

横截面特征是在同一时间点上，比较不同股票之间的相对关系。

### 1.1 市场排名特征

#### 1.1.1 涨幅排名百分位 (pct_chg_rank)

**定义**: 当日涨幅在全市场的排名百分位，值越大表示涨幅越高。

**公式**:
```
pct_chg_rank = PERCENT_RANK() OVER (PARTITION BY trade_date ORDER BY pct_chg)
```

**取值范围**: [0, 1]

**SQL实现**:
```sql
SELECT
    ts_code,
    trade_date,
    pct_chg,
    PERCENT_RANK() OVER (PARTITION BY trade_date ORDER BY pct_chg) AS pct_chg_rank
FROM daily
WHERE trade_date = '20260130';
```

**使用场景**:
- 识别市场强势股/弱势股
- 构建动量因子
- 筛选涨幅领先个股

---

#### 1.1.2 成交量排名百分位 (vol_rank)

**定义**: 当日成交量在全市场的排名百分位。

**公式**:
```
vol_rank = PERCENT_RANK() OVER (PARTITION BY trade_date ORDER BY vol)
```

**SQL实现**:
```sql
SELECT
    ts_code,
    trade_date,
    vol,
    PERCENT_RANK() OVER (PARTITION BY trade_date ORDER BY vol) AS vol_rank
FROM daily
WHERE trade_date = '20260130';
```

**使用场景**:
- 量价配合分析
- 识别放量突破
- 流动性筛选

---

#### 1.1.3 换手率排名 (turnover_rank)

**定义**: 当日换手率在全市场的排名百分位。

**公式**:
```
turnover_rank = PERCENT_RANK() OVER (PARTITION BY trade_date ORDER BY turnover_rate)
```

**SQL实现**:
```sql
SELECT
    d.ts_code,
    d.trade_date,
    db.turnover_rate,
    PERCENT_RANK() OVER (PARTITION BY d.trade_date ORDER BY db.turnover_rate) AS turnover_rank
FROM daily d
JOIN daily_basic db ON d.ts_code = db.ts_code AND d.trade_date = db.trade_date
WHERE d.trade_date = '20260130';
```

**使用场景**:
- 市场活跃度分析
- 热点板块识别
- 短线交易筛选

---

#### 1.1.4 市值排名 (mktcap_rank)

**定义**: 当日总市值在全市场的排名百分位。

**公式**:
```
mktcap_rank = PERCENT_RANK() OVER (PARTITION BY trade_date ORDER BY total_mv)
```

**SQL实现**:
```sql
SELECT
    ts_code,
    trade_date,
    total_mv,
    PERCENT_RANK() OVER (PARTITION BY trade_date ORDER BY total_mv) AS mktcap_rank
FROM daily_basic
WHERE trade_date = '20260130';
```

**使用场景**:
- 大小盘风格判断
- 市值因子构建
- 组合风格分析

---

### 1.2 行业相对特征

#### 1.2.1 个股涨幅 vs 行业涨幅 (excess_return_industry)

**定义**: 个股涨幅相对于所属行业平均涨幅的超额收益。

**公式**:
```
industry_avg_pct_chg = AVG(pct_chg) OVER (PARTITION BY trade_date, l1_code)
excess_return_industry = pct_chg - industry_avg_pct_chg
```

**SQL实现**:
```sql
WITH industry_mapping AS (
    SELECT ts_code, l1_code, l1_name
    FROM index_member_all
    WHERE out_date IS NULL OR out_date > '20260130'
),
daily_with_industry AS (
    SELECT
        d.ts_code,
        d.trade_date,
        d.pct_chg,
        im.l1_code,
        im.l1_name
    FROM daily d
    JOIN industry_mapping im ON d.ts_code = im.ts_code
    WHERE d.trade_date = '20260130'
)
SELECT
    ts_code,
    trade_date,
    pct_chg,
    l1_name,
    AVG(pct_chg) OVER (PARTITION BY trade_date, l1_code) AS industry_avg_pct_chg,
    pct_chg - AVG(pct_chg) OVER (PARTITION BY trade_date, l1_code) AS excess_return_industry
FROM daily_with_industry;
```

**使用场景**:
- 行业内选股
- Alpha因子构建
- 行业中性策略

---

#### 1.2.2 个股换手率 vs 行业换手率 (relative_turnover_industry)

**定义**: 个股换手率相对于行业平均换手率的比值。

**公式**:
```
industry_avg_turnover = AVG(turnover_rate) OVER (PARTITION BY trade_date, l1_code)
relative_turnover_industry = turnover_rate / industry_avg_turnover
```

**SQL实现**:
```sql
WITH industry_mapping AS (
    SELECT ts_code, l1_code, l1_name
    FROM index_member_all
    WHERE out_date IS NULL OR out_date > '20260130'
),
daily_with_industry AS (
    SELECT
        db.ts_code,
        db.trade_date,
        db.turnover_rate,
        im.l1_code,
        im.l1_name
    FROM daily_basic db
    JOIN industry_mapping im ON db.ts_code = im.ts_code
    WHERE db.trade_date = '20260130'
)
SELECT
    ts_code,
    trade_date,
    turnover_rate,
    l1_name,
    AVG(turnover_rate) OVER (PARTITION BY trade_date, l1_code) AS industry_avg_turnover,
    turnover_rate / NULLIF(AVG(turnover_rate) OVER (PARTITION BY trade_date, l1_code), 0)
        AS relative_turnover_industry
FROM daily_with_industry;
```

**使用场景**:
- 行业内活跃度对比
- 资金关注度分析
- 板块轮动判断

---

#### 1.2.3 行业内排名 (rank_in_industry)

**定义**: 个股涨幅在所属行业内的排名百分位。

**公式**:
```
rank_in_industry = PERCENT_RANK() OVER (PARTITION BY trade_date, l1_code ORDER BY pct_chg)
```

**SQL实现**:
```sql
WITH industry_mapping AS (
    SELECT ts_code, l1_code, l1_name
    FROM index_member_all
    WHERE out_date IS NULL OR out_date > '20260130'
),
daily_with_industry AS (
    SELECT
        d.ts_code,
        d.trade_date,
        d.pct_chg,
        im.l1_code,
        im.l1_name
    FROM daily d
    JOIN industry_mapping im ON d.ts_code = im.ts_code
    WHERE d.trade_date = '20260130'
)
SELECT
    ts_code,
    trade_date,
    pct_chg,
    l1_name,
    PERCENT_RANK() OVER (PARTITION BY trade_date, l1_code ORDER BY pct_chg) AS rank_in_industry,
    COUNT(*) OVER (PARTITION BY trade_date, l1_code) AS industry_stock_count
FROM daily_with_industry;
```

**使用场景**:
- 行业龙头识别
- 行业内相对强弱
- 选股排序

---

#### 1.2.4 行业强弱度 (industry_strength)

**定义**: 行业平均涨幅相对于全市场平均涨幅的差值。

**公式**:
```
market_avg_pct_chg = AVG(pct_chg) OVER (PARTITION BY trade_date)
industry_avg_pct_chg = AVG(pct_chg) OVER (PARTITION BY trade_date, l1_code)
industry_strength = industry_avg_pct_chg - market_avg_pct_chg
```

**SQL实现**:
```sql
WITH industry_mapping AS (
    SELECT ts_code, l1_code, l1_name
    FROM index_member_all
    WHERE out_date IS NULL OR out_date > '20260130'
),
daily_with_industry AS (
    SELECT
        d.ts_code,
        d.trade_date,
        d.pct_chg,
        im.l1_code,
        im.l1_name
    FROM daily d
    JOIN industry_mapping im ON d.ts_code = im.ts_code
    WHERE d.trade_date = '20260130'
)
SELECT DISTINCT
    l1_code,
    l1_name,
    trade_date,
    AVG(pct_chg) OVER (PARTITION BY trade_date, l1_code) AS industry_avg_pct_chg,
    AVG(pct_chg) OVER (PARTITION BY trade_date) AS market_avg_pct_chg,
    AVG(pct_chg) OVER (PARTITION BY trade_date, l1_code) -
        AVG(pct_chg) OVER (PARTITION BY trade_date) AS industry_strength
FROM daily_with_industry
ORDER BY industry_strength DESC;
```

**使用场景**:
- 板块轮动分析
- 行业配置决策
- 热点追踪

---

### 1.3 板块特征

#### 1.3.1 龙头股效应 (is_industry_leader)

**定义**: 是否为当日行业领涨股（涨幅排名第一）。

**公式**:
```
is_industry_leader = CASE WHEN RANK() OVER (PARTITION BY trade_date, l1_code ORDER BY pct_chg DESC) = 1
                     THEN 1 ELSE 0 END
```

**SQL实现**:
```sql
WITH industry_mapping AS (
    SELECT ts_code, l1_code, l1_name
    FROM index_member_all
    WHERE out_date IS NULL OR out_date > '20260130'
),
daily_with_industry AS (
    SELECT
        d.ts_code,
        d.trade_date,
        d.pct_chg,
        im.l1_code,
        im.l1_name
    FROM daily d
    JOIN industry_mapping im ON d.ts_code = im.ts_code
    WHERE d.trade_date = '20260130'
)
SELECT
    ts_code,
    trade_date,
    pct_chg,
    l1_name,
    RANK() OVER (PARTITION BY trade_date, l1_code ORDER BY pct_chg DESC) AS industry_rank,
    CASE WHEN RANK() OVER (PARTITION BY trade_date, l1_code ORDER BY pct_chg DESC) = 1
         THEN 1 ELSE 0 END AS is_industry_leader
FROM daily_with_industry;
```

**使用场景**:
- 龙头股策略
- 板块启动信号
- 资金聚集效应分析

---

#### 1.3.2 行业集中度 (industry_concentration)

**定义**: 行业内涨幅标准差，反映行业内个股表现的分化程度。

**公式**:
```
industry_concentration = STDDEV(pct_chg) OVER (PARTITION BY trade_date, l1_code)
```

**SQL实现**:
```sql
WITH industry_mapping AS (
    SELECT ts_code, l1_code, l1_name
    FROM index_member_all
    WHERE out_date IS NULL OR out_date > '20260130'
),
daily_with_industry AS (
    SELECT
        d.ts_code,
        d.trade_date,
        d.pct_chg,
        im.l1_code,
        im.l1_name
    FROM daily d
    JOIN industry_mapping im ON d.ts_code = im.ts_code
    WHERE d.trade_date = '20260130'
)
SELECT DISTINCT
    l1_code,
    l1_name,
    trade_date,
    STDDEV(pct_chg) OVER (PARTITION BY trade_date, l1_code) AS industry_volatility,
    COUNT(*) OVER (PARTITION BY trade_date, l1_code) AS stock_count
FROM daily_with_industry
ORDER BY industry_volatility DESC;
```

**使用场景**:
- 行业分化程度分析
- 选股难度评估
- 行业一致性判断

---

### 1.4 市场情绪特征

#### 1.4.1 涨停/跌停数量 (limit_up_count / limit_down_count)

**定义**: 全市场涨停和跌停股票数量。

**公式**:
```
limit_up_count = COUNT(*) WHERE pct_chg >= 9.5 (普通股) OR pct_chg >= 19.5 (创业板/科创板)
limit_down_count = COUNT(*) WHERE pct_chg <= -9.5 (普通股) OR pct_chg <= -19.5 (创业板/科创板)
```

**SQL实现**:
```sql
SELECT
    trade_date,
    COUNT(CASE WHEN (ts_code LIKE '30%' OR ts_code LIKE '68%') AND pct_chg >= 19.5 THEN 1
               WHEN pct_chg >= 9.5 THEN 1 END) AS limit_up_count,
    COUNT(CASE WHEN (ts_code LIKE '30%' OR ts_code LIKE '68%') AND pct_chg <= -19.5 THEN 1
               WHEN pct_chg <= -9.5 THEN 1 END) AS limit_down_count,
    COUNT(*) AS total_stocks
FROM daily
WHERE trade_date = '20260130'
GROUP BY trade_date;
```

**使用场景**:
- 市场情绪判断
- 风险预警
- 择时信号

---

#### 1.4.2 上涨/下跌家数比 (advance_decline_ratio)

**定义**: 上涨股票数量与下跌股票数量的比值。

**公式**:
```
advance_decline_ratio = COUNT(pct_chg > 0) / COUNT(pct_chg < 0)
```

**SQL实现**:
```sql
SELECT
    trade_date,
    COUNT(CASE WHEN pct_chg > 0 THEN 1 END) AS advance_count,
    COUNT(CASE WHEN pct_chg < 0 THEN 1 END) AS decline_count,
    COUNT(CASE WHEN pct_chg = 0 THEN 1 END) AS unchanged_count,
    COUNT(CASE WHEN pct_chg > 0 THEN 1 END) * 1.0 /
        NULLIF(COUNT(CASE WHEN pct_chg < 0 THEN 1 END), 0) AS advance_decline_ratio
FROM daily
WHERE trade_date = '20260130'
GROUP BY trade_date;
```

**使用场景**:
- 市场宽度分析
- 牛熊市场判断
- 指数背离分析

---

#### 1.4.3 市场宽度 (market_breadth)

**定义**: 上涨家数减去下跌家数，反映市场整体强弱。

**公式**:
```
market_breadth = COUNT(pct_chg > 0) - COUNT(pct_chg < 0)
market_breadth_pct = (advance_count - decline_count) / total_stocks
```

**SQL实现**:
```sql
SELECT
    trade_date,
    COUNT(CASE WHEN pct_chg > 0 THEN 1 END) AS advance_count,
    COUNT(CASE WHEN pct_chg < 0 THEN 1 END) AS decline_count,
    COUNT(*) AS total_stocks,
    COUNT(CASE WHEN pct_chg > 0 THEN 1 END) - COUNT(CASE WHEN pct_chg < 0 THEN 1 END) AS market_breadth,
    (COUNT(CASE WHEN pct_chg > 0 THEN 1 END) - COUNT(CASE WHEN pct_chg < 0 THEN 1 END)) * 1.0 /
        COUNT(*) AS market_breadth_pct
FROM daily
WHERE trade_date = '20260130'
GROUP BY trade_date;
```

**使用场景**:
- 市场参与度分析
- 指数真实强弱
- 资金效应判断

---

#### 1.4.4 主力资金净流入 (net_main_inflow)

**定义**: 大单和超大单净流入金额。

**公式**:
```
net_main_inflow = (buy_lg_amount + buy_elg_amount) - (sell_lg_amount + sell_elg_amount)
```

**SQL实现**:
```sql
SELECT
    ts_code,
    trade_date,
    (buy_lg_amount + COALESCE(buy_elg_amount, 0)) AS main_buy_amount,
    (sell_lg_amount + COALESCE(sell_elg_amount, 0)) AS main_sell_amount,
    (buy_lg_amount + COALESCE(buy_elg_amount, 0)) -
        (sell_lg_amount + COALESCE(sell_elg_amount, 0)) AS net_main_inflow
FROM moneyflow
WHERE trade_date = '20260130'
ORDER BY net_main_inflow DESC
LIMIT 20;
```

**使用场景**:
- 主力动向追踪
- 资金流向分析
- 龙虎榜关联

---

## 二、时序特征 (Time-series Features)

时序特征是基于单只股票历史数据计算的特征。

### 2.1 动量特征

#### 2.1.1 不同周期收益率 (return_Nd)

**定义**: 过去N日的累计收益率。

**公式**:
```
return_1d = (close - pre_close) / pre_close
return_5d = (close - close_5d_ago) / close_5d_ago
return_Nd = close / LAG(close, N) - 1
```

**SQL实现**:
```sql
WITH price_history AS (
    SELECT
        ts_code,
        trade_date,
        close,
        LAG(close, 1) OVER (PARTITION BY ts_code ORDER BY trade_date) AS close_1d_ago,
        LAG(close, 5) OVER (PARTITION BY ts_code ORDER BY trade_date) AS close_5d_ago,
        LAG(close, 10) OVER (PARTITION BY ts_code ORDER BY trade_date) AS close_10d_ago,
        LAG(close, 20) OVER (PARTITION BY ts_code ORDER BY trade_date) AS close_20d_ago,
        LAG(close, 60) OVER (PARTITION BY ts_code ORDER BY trade_date) AS close_60d_ago,
        LAG(close, 120) OVER (PARTITION BY ts_code ORDER BY trade_date) AS close_120d_ago
    FROM daily
    WHERE ts_code = '000001.SZ'
)
SELECT
    ts_code,
    trade_date,
    close,
    (close / close_1d_ago - 1) * 100 AS return_1d,
    (close / close_5d_ago - 1) * 100 AS return_5d,
    (close / close_10d_ago - 1) * 100 AS return_10d,
    (close / close_20d_ago - 1) * 100 AS return_20d,
    (close / close_60d_ago - 1) * 100 AS return_60d,
    (close / close_120d_ago - 1) * 100 AS return_120d
FROM price_history
WHERE trade_date >= '20260101'
ORDER BY trade_date DESC;
```

**使用场景**:
- 动量因子构建
- 趋势跟踪策略
- 风险监控

---

#### 2.1.2 动量反转 (momentum_reversal)

**定义**: 短期动量与长期动量的差值，用于捕捉反转信号。

**公式**:
```
momentum_reversal = return_5d - return_20d
momentum_reversal_ratio = return_5d / return_20d
```

**SQL实现**:
```sql
WITH price_history AS (
    SELECT
        ts_code,
        trade_date,
        close,
        LAG(close, 5) OVER (PARTITION BY ts_code ORDER BY trade_date) AS close_5d_ago,
        LAG(close, 20) OVER (PARTITION BY ts_code ORDER BY trade_date) AS close_20d_ago
    FROM daily
    WHERE ts_code = '000001.SZ'
)
SELECT
    ts_code,
    trade_date,
    close,
    (close / close_5d_ago - 1) * 100 AS return_5d,
    (close / close_20d_ago - 1) * 100 AS return_20d,
    ((close / close_5d_ago) - (close / close_20d_ago)) * 100 AS momentum_reversal
FROM price_history
WHERE trade_date >= '20260101'
ORDER BY trade_date DESC;
```

**使用场景**:
- 反转策略
- 超买超卖判断
- 均值回归策略

---

#### 2.1.3 累计超额收益 (cumulative_excess_return)

**定义**: 相对于基准指数的累计超额收益。

**公式**:
```
cumulative_excess_return = SUM(stock_return - benchmark_return) OVER (ORDER BY trade_date)
```

**SQL实现**:
```sql
WITH stock_returns AS (
    SELECT
        ts_code,
        trade_date,
        pct_chg AS stock_return
    FROM daily
    WHERE ts_code = '000001.SZ' AND trade_date >= '20260101'
),
benchmark_returns AS (
    -- 假设使用沪深300作为基准
    SELECT
        trade_date,
        pct_chg AS benchmark_return
    FROM daily
    WHERE ts_code = '000300.SH' AND trade_date >= '20260101'
)
SELECT
    s.ts_code,
    s.trade_date,
    s.stock_return,
    b.benchmark_return,
    s.stock_return - COALESCE(b.benchmark_return, 0) AS daily_excess_return,
    SUM(s.stock_return - COALESCE(b.benchmark_return, 0))
        OVER (PARTITION BY s.ts_code ORDER BY s.trade_date) AS cumulative_excess_return
FROM stock_returns s
LEFT JOIN benchmark_returns b ON s.trade_date = b.trade_date
ORDER BY s.trade_date;
```

**使用场景**:
- Alpha评估
- 绩效归因
- 主动管理能力分析

---

### 2.2 波动率特征

#### 2.2.1 历史波动率 (historical_volatility)

**定义**: 过去N日收益率的标准差（年化）。

**公式**:
```
daily_volatility_N = STDDEV(pct_chg) OVER (PARTITION BY ts_code ORDER BY trade_date ROWS N-1 PRECEDING)
annualized_volatility = daily_volatility_N * SQRT(252)
```

**SQL实现**:
```sql
WITH volatility_calc AS (
    SELECT
        ts_code,
        trade_date,
        pct_chg,
        STDDEV(pct_chg) OVER (PARTITION BY ts_code ORDER BY trade_date ROWS 4 PRECEDING) AS vol_5d,
        STDDEV(pct_chg) OVER (PARTITION BY ts_code ORDER BY trade_date ROWS 9 PRECEDING) AS vol_10d,
        STDDEV(pct_chg) OVER (PARTITION BY ts_code ORDER BY trade_date ROWS 19 PRECEDING) AS vol_20d,
        STDDEV(pct_chg) OVER (PARTITION BY ts_code ORDER BY trade_date ROWS 59 PRECEDING) AS vol_60d
    FROM daily
    WHERE ts_code = '000001.SZ'
)
SELECT
    ts_code,
    trade_date,
    pct_chg,
    vol_5d,
    vol_10d,
    vol_20d,
    vol_60d,
    vol_20d * SQRT(252) AS annualized_vol_20d
FROM volatility_calc
WHERE trade_date >= '20260101'
ORDER BY trade_date DESC;
```

**使用场景**:
- 风险管理
- 期权定价
- 波动率交易策略

---

#### 2.2.2 波动率变化率 (volatility_change)

**定义**: 当前波动率相对于过去波动率的变化。

**公式**:
```
volatility_change = vol_5d / vol_20d - 1
volatility_ratio = vol_5d / vol_60d
```

**SQL实现**:
```sql
WITH volatility_calc AS (
    SELECT
        ts_code,
        trade_date,
        pct_chg,
        STDDEV(pct_chg) OVER (PARTITION BY ts_code ORDER BY trade_date ROWS 4 PRECEDING) AS vol_5d,
        STDDEV(pct_chg) OVER (PARTITION BY ts_code ORDER BY trade_date ROWS 19 PRECEDING) AS vol_20d,
        STDDEV(pct_chg) OVER (PARTITION BY ts_code ORDER BY trade_date ROWS 59 PRECEDING) AS vol_60d
    FROM daily
    WHERE ts_code = '000001.SZ'
)
SELECT
    ts_code,
    trade_date,
    vol_5d,
    vol_20d,
    vol_60d,
    (vol_5d / NULLIF(vol_20d, 0) - 1) * 100 AS vol_change_5d_20d,
    vol_5d / NULLIF(vol_60d, 0) AS vol_ratio_5d_60d
FROM volatility_calc
WHERE trade_date >= '20260101'
ORDER BY trade_date DESC;
```

**使用场景**:
- 波动率突破信号
- 市场恐慌/平静判断
- 动态仓位管理

---

#### 2.2.3 波动率偏度 (volatility_skewness)

**定义**: 收益率分布的偏度，反映尾部风险。

**公式**:
```
skewness = E[(X - μ)³] / σ³
```

**SQL实现**:
```sql
WITH return_stats AS (
    SELECT
        ts_code,
        trade_date,
        pct_chg,
        AVG(pct_chg) OVER (PARTITION BY ts_code ORDER BY trade_date ROWS 19 PRECEDING) AS mean_20d,
        STDDEV(pct_chg) OVER (PARTITION BY ts_code ORDER BY trade_date ROWS 19 PRECEDING) AS std_20d
    FROM daily
    WHERE ts_code = '000001.SZ'
),
skewness_calc AS (
    SELECT
        ts_code,
        trade_date,
        pct_chg,
        mean_20d,
        std_20d,
        POWER((pct_chg - mean_20d) / NULLIF(std_20d, 0), 3) AS cubed_zscore
    FROM return_stats
    WHERE std_20d > 0
)
SELECT
    ts_code,
    trade_date,
    AVG(cubed_zscore) OVER (PARTITION BY ts_code ORDER BY trade_date ROWS 19 PRECEDING) AS skewness_20d
FROM skewness_calc
WHERE trade_date >= '20260101'
ORDER BY trade_date DESC;
```

**使用场景**:
- 下行风险评估
- 尾部风险管理
- 风险偏好分析

---

### 2.3 自相关特征

#### 2.3.1 收益率自相关 (return_autocorr)

**定义**: 当前收益率与滞后N期收益率的相关系数。

**公式**:
```
autocorr_lag_n = CORR(return_t, return_{t-n})
```

**SQL实现**:
```sql
WITH lagged_returns AS (
    SELECT
        ts_code,
        trade_date,
        pct_chg AS return_t,
        LAG(pct_chg, 1) OVER (PARTITION BY ts_code ORDER BY trade_date) AS return_lag1,
        LAG(pct_chg, 2) OVER (PARTITION BY ts_code ORDER BY trade_date) AS return_lag2,
        LAG(pct_chg, 3) OVER (PARTITION BY ts_code ORDER BY trade_date) AS return_lag3,
        LAG(pct_chg, 5) OVER (PARTITION BY ts_code ORDER BY trade_date) AS return_lag5
    FROM daily
    WHERE ts_code = '000001.SZ'
)
SELECT
    ts_code,
    MAX(trade_date) AS as_of_date,
    CORR(return_t, return_lag1) AS autocorr_lag1,
    CORR(return_t, return_lag2) AS autocorr_lag2,
    CORR(return_t, return_lag3) AS autocorr_lag3,
    CORR(return_t, return_lag5) AS autocorr_lag5
FROM lagged_returns
WHERE trade_date >= '20250101'
GROUP BY ts_code;
```

**使用场景**:
- 市场效率检验
- 趋势延续性分析
- 均值回归策略

---

#### 2.3.2 成交量自相关 (volume_autocorr)

**定义**: 成交量的自相关性，反映量能的持续性。

**公式**:
```
vol_autocorr = CORR(vol_t, vol_{t-1})
```

**SQL实现**:
```sql
WITH lagged_volume AS (
    SELECT
        ts_code,
        trade_date,
        vol AS vol_t,
        LAG(vol, 1) OVER (PARTITION BY ts_code ORDER BY trade_date) AS vol_lag1,
        LAG(vol, 5) OVER (PARTITION BY ts_code ORDER BY trade_date) AS vol_lag5
    FROM daily
    WHERE ts_code = '000001.SZ'
)
SELECT
    ts_code,
    MAX(trade_date) AS as_of_date,
    CORR(vol_t, vol_lag1) AS vol_autocorr_lag1,
    CORR(vol_t, vol_lag5) AS vol_autocorr_lag5
FROM lagged_volume
WHERE trade_date >= '20250101'
GROUP BY ts_code;
```

**使用场景**:
- 量能持续性判断
- 放量信号确认
- 趋势强度分析

---

#### 2.3.3 波动率聚类 (volatility_clustering)

**定义**: 高波动率后往往跟随高波动率的特性。

**公式**:
```
abs_return_t = |pct_chg|
vol_clustering = CORR(abs_return_t, abs_return_{t-1})
```

**SQL实现**:
```sql
WITH abs_returns AS (
    SELECT
        ts_code,
        trade_date,
        ABS(pct_chg) AS abs_return_t,
        LAG(ABS(pct_chg), 1) OVER (PARTITION BY ts_code ORDER BY trade_date) AS abs_return_lag1
    FROM daily
    WHERE ts_code = '000001.SZ'
)
SELECT
    ts_code,
    MAX(trade_date) AS as_of_date,
    CORR(abs_return_t, abs_return_lag1) AS volatility_clustering
FROM abs_returns
WHERE trade_date >= '20250101'
GROUP BY ts_code;
```

**使用场景**:
- GARCH模型输入
- 风险预测
- 波动率交易

---

### 2.4 状态特征

#### 2.4.1 创新高/新低天数 (days_since_high_low)

**定义**: 距离N日最高价/最低价的天数。

**公式**:
```
days_since_high = 当前日期 - N日内最高价出现日期
days_since_low = 当前日期 - N日内最低价出现日期
```

**SQL实现**:
```sql
WITH price_with_high_low AS (
    SELECT
        ts_code,
        trade_date,
        close,
        high,
        low,
        MAX(high) OVER (PARTITION BY ts_code ORDER BY trade_date ROWS 19 PRECEDING) AS high_20d,
        MIN(low) OVER (PARTITION BY ts_code ORDER BY trade_date ROWS 19 PRECEDING) AS low_20d,
        MAX(high) OVER (PARTITION BY ts_code ORDER BY trade_date ROWS 59 PRECEDING) AS high_60d,
        MIN(low) OVER (PARTITION BY ts_code ORDER BY trade_date ROWS 59 PRECEDING) AS low_60d
    FROM daily
    WHERE ts_code = '000001.SZ'
)
SELECT
    ts_code,
    trade_date,
    close,
    high_20d,
    low_20d,
    CASE WHEN close >= high_20d THEN 1 ELSE 0 END AS at_20d_high,
    CASE WHEN close <= low_20d THEN 1 ELSE 0 END AS at_20d_low,
    (close - low_20d) / NULLIF(high_20d - low_20d, 0) AS position_in_20d_range
FROM price_with_high_low
WHERE trade_date >= '20260101'
ORDER BY trade_date DESC;
```

**使用场景**:
- 突破策略
- 支撑阻力分析
- 趋势强度判断

---

#### 2.4.2 均线多头/空头排列 (ma_alignment)

**定义**: 不同周期均线的排列状态。

**公式**:
```
ma_bull_alignment = CASE WHEN ma5 > ma10 > ma20 > ma60 THEN 1 ELSE 0 END
ma_bear_alignment = CASE WHEN ma5 < ma10 < ma20 < ma60 THEN 1 ELSE 0 END
ma_score = (ma5 > ma10) + (ma10 > ma20) + (ma20 > ma60) 的计数
```

**SQL实现**:
```sql
WITH ma_calc AS (
    SELECT
        ts_code,
        trade_date,
        close,
        AVG(close) OVER (PARTITION BY ts_code ORDER BY trade_date ROWS 4 PRECEDING) AS ma5,
        AVG(close) OVER (PARTITION BY ts_code ORDER BY trade_date ROWS 9 PRECEDING) AS ma10,
        AVG(close) OVER (PARTITION BY ts_code ORDER BY trade_date ROWS 19 PRECEDING) AS ma20,
        AVG(close) OVER (PARTITION BY ts_code ORDER BY trade_date ROWS 59 PRECEDING) AS ma60
    FROM daily
    WHERE ts_code = '000001.SZ'
)
SELECT
    ts_code,
    trade_date,
    close,
    ma5, ma10, ma20, ma60,
    CASE WHEN ma5 > ma10 AND ma10 > ma20 AND ma20 > ma60 THEN 1 ELSE 0 END AS ma_bull_alignment,
    CASE WHEN ma5 < ma10 AND ma10 < ma20 AND ma20 < ma60 THEN 1 ELSE 0 END AS ma_bear_alignment,
    (CASE WHEN ma5 > ma10 THEN 1 ELSE 0 END +
     CASE WHEN ma10 > ma20 THEN 1 ELSE 0 END +
     CASE WHEN ma20 > ma60 THEN 1 ELSE 0 END) AS ma_score,
    (close / ma20 - 1) * 100 AS price_ma20_deviation
FROM ma_calc
WHERE trade_date >= '20260101'
ORDER BY trade_date DESC;
```

**使用场景**:
- 趋势确认
- 入场时机判断
- 风格因子构建

---

#### 2.4.3 趋势强度 ADX (adx)

**定义**: 平均趋向指数，衡量趋势的强弱程度。

**公式**:
```
TR = MAX(high - low, |high - pre_close|, |low - pre_close|)
+DM = high - pre_high (if > 0 and > pre_low - low)
-DM = pre_low - low (if > 0 and > high - pre_high)
+DI = 100 * EMA(+DM, 14) / ATR(14)
-DI = 100 * EMA(-DM, 14) / ATR(14)
DX = 100 * |+DI - -DI| / (+DI + -DI)
ADX = EMA(DX, 14)
```

**SQL实现** (简化版):
```sql
WITH price_data AS (
    SELECT
        ts_code,
        trade_date,
        high,
        low,
        close,
        LAG(high, 1) OVER (PARTITION BY ts_code ORDER BY trade_date) AS pre_high,
        LAG(low, 1) OVER (PARTITION BY ts_code ORDER BY trade_date) AS pre_low,
        LAG(close, 1) OVER (PARTITION BY ts_code ORDER BY trade_date) AS pre_close
    FROM daily
    WHERE ts_code = '000001.SZ'
),
tr_dm_calc AS (
    SELECT
        ts_code,
        trade_date,
        GREATEST(high - low, ABS(high - pre_close), ABS(low - pre_close)) AS tr,
        CASE WHEN high - pre_high > pre_low - low AND high - pre_high > 0
             THEN high - pre_high ELSE 0 END AS plus_dm,
        CASE WHEN pre_low - low > high - pre_high AND pre_low - low > 0
             THEN pre_low - low ELSE 0 END AS minus_dm
    FROM price_data
    WHERE pre_close IS NOT NULL
),
smoothed AS (
    SELECT
        ts_code,
        trade_date,
        AVG(tr) OVER (PARTITION BY ts_code ORDER BY trade_date ROWS 13 PRECEDING) AS atr_14,
        AVG(plus_dm) OVER (PARTITION BY ts_code ORDER BY trade_date ROWS 13 PRECEDING) AS plus_dm_14,
        AVG(minus_dm) OVER (PARTITION BY ts_code ORDER BY trade_date ROWS 13 PRECEDING) AS minus_dm_14
    FROM tr_dm_calc
),
di_calc AS (
    SELECT
        ts_code,
        trade_date,
        100 * plus_dm_14 / NULLIF(atr_14, 0) AS plus_di,
        100 * minus_dm_14 / NULLIF(atr_14, 0) AS minus_di
    FROM smoothed
),
dx_calc AS (
    SELECT
        ts_code,
        trade_date,
        plus_di,
        minus_di,
        100 * ABS(plus_di - minus_di) / NULLIF(plus_di + minus_di, 0) AS dx
    FROM di_calc
)
SELECT
    ts_code,
    trade_date,
    plus_di,
    minus_di,
    dx,
    AVG(dx) OVER (PARTITION BY ts_code ORDER BY trade_date ROWS 13 PRECEDING) AS adx
FROM dx_calc
WHERE trade_date >= '20260101'
ORDER BY trade_date DESC;
```

**使用场景**:
- 趋势交易策略
- 震荡/趋势市场识别
- 策略选择依据

---

#### 2.4.4 布林带位置 (bollinger_position)

**定义**: 当前价格在布林带中的相对位置。

**公式**:
```
middle_band = MA(close, 20)
upper_band = middle_band + 2 * STDDEV(close, 20)
lower_band = middle_band - 2 * STDDEV(close, 20)
bollinger_position = (close - lower_band) / (upper_band - lower_band)
bollinger_width = (upper_band - lower_band) / middle_band
```

**SQL实现**:
```sql
WITH bollinger_calc AS (
    SELECT
        ts_code,
        trade_date,
        close,
        AVG(close) OVER (PARTITION BY ts_code ORDER BY trade_date ROWS 19 PRECEDING) AS middle_band,
        STDDEV(close) OVER (PARTITION BY ts_code ORDER BY trade_date ROWS 19 PRECEDING) AS std_20
    FROM daily
    WHERE ts_code = '000001.SZ'
)
SELECT
    ts_code,
    trade_date,
    close,
    middle_band,
    middle_band + 2 * std_20 AS upper_band,
    middle_band - 2 * std_20 AS lower_band,
    (close - (middle_band - 2 * std_20)) / NULLIF(4 * std_20, 0) AS bollinger_position,
    4 * std_20 / NULLIF(middle_band, 0) AS bollinger_width,
    (close - middle_band) / NULLIF(std_20, 0) AS z_score
FROM bollinger_calc
WHERE trade_date >= '20260101'
ORDER BY trade_date DESC;
```

**使用场景**:
- 超买超卖判断
- 波动率突破
- 均值回归交易

---

## 三、筹码特征 (Chip Features)

基于 cyq_perf 表的筹码分布特征。

### 3.1 获利比例特征

#### 3.1.1 获利盘比例 (winner_rate)

**定义**: 当前价格下处于盈利状态的筹码比例。

**字段**: winner_rate

**SQL实现**:
```sql
SELECT
    ts_code,
    trade_date,
    winner_rate,
    winner_rate - LAG(winner_rate, 1) OVER (PARTITION BY ts_code ORDER BY trade_date) AS winner_rate_change_1d,
    winner_rate - LAG(winner_rate, 5) OVER (PARTITION BY ts_code ORDER BY trade_date) AS winner_rate_change_5d
FROM cyq_perf
WHERE ts_code = '000001.SZ' AND trade_date >= '20260101'
ORDER BY trade_date DESC;
```

**使用场景**:
- 套牢盘压力分析
- 获利了结风险评估
- 支撑位判断

---

#### 3.1.2 套牢盘比例 (loser_rate)

**定义**: 当前价格下处于亏损状态的筹码比例。

**公式**:
```
loser_rate = 1 - winner_rate
```

**SQL实现**:
```sql
SELECT
    ts_code,
    trade_date,
    winner_rate,
    (1 - winner_rate) AS loser_rate,
    CASE
        WHEN winner_rate > 0.9 THEN '获利盘极高'
        WHEN winner_rate > 0.7 THEN '获利盘较高'
        WHEN winner_rate > 0.5 THEN '获利盘适中'
        WHEN winner_rate > 0.3 THEN '套牢盘较多'
        ELSE '套牢盘极多'
    END AS chip_status
FROM cyq_perf
WHERE ts_code = '000001.SZ' AND trade_date >= '20260101'
ORDER BY trade_date DESC;
```

**使用场景**:
- 压力位识别
- 洗盘判断
- 底部特征分析

---

### 3.2 成本分布特征

#### 3.2.1 平均成本 (weight_avg)

**定义**: 筹码的加权平均成本价格。

**字段**: weight_avg

**SQL实现**:
```sql
SELECT
    c.ts_code,
    c.trade_date,
    d.close,
    c.weight_avg,
    (d.close / c.weight_avg - 1) * 100 AS price_vs_avg_cost,
    c.weight_avg - LAG(c.weight_avg, 5) OVER (PARTITION BY c.ts_code ORDER BY c.trade_date) AS cost_change_5d
FROM cyq_perf c
JOIN daily d ON c.ts_code = d.ts_code AND c.trade_date = d.trade_date
WHERE c.ts_code = '000001.SZ' AND c.trade_date >= '20260101'
ORDER BY c.trade_date DESC;
```

**使用场景**:
- 成本线支撑/压力
- 主力成本估计
- 盈亏平衡分析

---

#### 3.2.2 90%筹码集中度 (chip_concentration)

**定义**: 90%筹码分布的价格区间，反映筹码集中程度。

**公式**:
```
chip_concentration = (cost_95pct - cost_5pct) / weight_avg
```

**SQL实现**:
```sql
SELECT
    c.ts_code,
    c.trade_date,
    d.close,
    c.cost_5pct,
    c.cost_95pct,
    c.weight_avg,
    (c.cost_95pct - c.cost_5pct) AS chip_range,
    (c.cost_95pct - c.cost_5pct) / NULLIF(c.weight_avg, 0) AS chip_concentration,
    CASE
        WHEN (c.cost_95pct - c.cost_5pct) / NULLIF(c.weight_avg, 0) < 0.1 THEN '极度集中'
        WHEN (c.cost_95pct - c.cost_5pct) / NULLIF(c.weight_avg, 0) < 0.2 THEN '较为集中'
        WHEN (c.cost_95pct - c.cost_5pct) / NULLIF(c.weight_avg, 0) < 0.3 THEN '适度分散'
        ELSE '较为分散'
    END AS concentration_level
FROM cyq_perf c
JOIN daily d ON c.ts_code = d.ts_code AND c.trade_date = d.trade_date
WHERE c.ts_code = '000001.SZ' AND c.trade_date >= '20260101'
ORDER BY c.trade_date DESC;
```

**使用场景**:
- 主力控盘度判断
- 筹码锁定程度
- 突破可能性评估

---

#### 3.2.3 筹码峰位置 (chip_peak_position)

**定义**: 主要成本区间相对于当前价格的位置。

**公式**:
```
chip_peak_position = (close - cost_50pct) / close
```

**SQL实现**:
```sql
SELECT
    c.ts_code,
    c.trade_date,
    d.close,
    c.cost_50pct AS median_cost,
    c.cost_15pct,
    c.cost_85pct,
    (d.close - c.cost_50pct) / NULLIF(d.close, 0) AS peak_position,
    CASE
        WHEN d.close > c.cost_85pct THEN '价格在筹码上方'
        WHEN d.close > c.cost_50pct THEN '价格在中位成本上方'
        WHEN d.close > c.cost_15pct THEN '价格在中位成本下方'
        ELSE '价格在筹码下方'
    END AS price_chip_relation
FROM cyq_perf c
JOIN daily d ON c.ts_code = d.ts_code AND c.trade_date = d.trade_date
WHERE c.ts_code = '000001.SZ' AND c.trade_date >= '20260101'
ORDER BY c.trade_date DESC;
```

**使用场景**:
- 支撑阻力分析
- 突破确认
- 趋势判断

---

## 四、综合特征计算 (Python实现)

### 4.1 完整特征计算类

```python
"""
横截面与时序特征工程实现
数据库: /Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db
"""

import duckdb
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from datetime import datetime, timedelta


class FeatureEngineer:
    """横截面和时序特征计算器"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = duckdb.connect(db_path, read_only=True)

    def close(self):
        self.conn.close()

    # ==================== 横截面特征 ====================

    def compute_market_rank_features(self, trade_date: str) -> pd.DataFrame:
        """
        计算市场排名特征
        - pct_chg_rank: 涨幅排名百分位
        - vol_rank: 成交量排名百分位
        - turnover_rank: 换手率排名百分位
        - mktcap_rank: 市值排名百分位
        """
        query = """
        SELECT
            d.ts_code,
            d.trade_date,
            d.pct_chg,
            d.vol,
            db.turnover_rate,
            db.total_mv,
            PERCENT_RANK() OVER (ORDER BY d.pct_chg) AS pct_chg_rank,
            PERCENT_RANK() OVER (ORDER BY d.vol) AS vol_rank,
            PERCENT_RANK() OVER (ORDER BY db.turnover_rate) AS turnover_rank,
            PERCENT_RANK() OVER (ORDER BY db.total_mv) AS mktcap_rank
        FROM daily d
        JOIN daily_basic db ON d.ts_code = db.ts_code AND d.trade_date = db.trade_date
        WHERE d.trade_date = ?
        """
        return self.conn.execute(query, [trade_date]).fetchdf()

    def compute_industry_relative_features(self, trade_date: str) -> pd.DataFrame:
        """
        计算行业相对特征
        - excess_return_industry: 相对行业超额收益
        - relative_turnover_industry: 相对行业换手率
        - rank_in_industry: 行业内排名
        - industry_strength: 行业强弱度
        """
        query = """
        WITH industry_mapping AS (
            SELECT ts_code, l1_code, l1_name
            FROM index_member_all
            WHERE out_date IS NULL OR out_date > ?
        ),
        daily_with_industry AS (
            SELECT
                d.ts_code,
                d.trade_date,
                d.pct_chg,
                db.turnover_rate,
                im.l1_code,
                im.l1_name
            FROM daily d
            JOIN daily_basic db ON d.ts_code = db.ts_code AND d.trade_date = db.trade_date
            JOIN industry_mapping im ON d.ts_code = im.ts_code
            WHERE d.trade_date = ?
        )
        SELECT
            ts_code,
            trade_date,
            pct_chg,
            turnover_rate,
            l1_code,
            l1_name,
            AVG(pct_chg) OVER (PARTITION BY l1_code) AS industry_avg_pct_chg,
            pct_chg - AVG(pct_chg) OVER (PARTITION BY l1_code) AS excess_return_industry,
            turnover_rate / NULLIF(AVG(turnover_rate) OVER (PARTITION BY l1_code), 0) AS relative_turnover_industry,
            PERCENT_RANK() OVER (PARTITION BY l1_code ORDER BY pct_chg) AS rank_in_industry,
            AVG(pct_chg) OVER () AS market_avg_pct_chg,
            AVG(pct_chg) OVER (PARTITION BY l1_code) - AVG(pct_chg) OVER () AS industry_strength
        FROM daily_with_industry
        """
        return self.conn.execute(query, [trade_date, trade_date]).fetchdf()

    def compute_market_sentiment_features(self, trade_date: str) -> pd.DataFrame:
        """
        计算市场情绪特征
        - limit_up_count: 涨停数量
        - limit_down_count: 跌停数量
        - advance_count: 上涨家数
        - decline_count: 下跌家数
        - advance_decline_ratio: 涨跌比
        - market_breadth: 市场宽度
        """
        query = """
        SELECT
            trade_date,
            COUNT(CASE WHEN (ts_code LIKE '30%' OR ts_code LIKE '68%') AND pct_chg >= 19.5 THEN 1
                       WHEN pct_chg >= 9.5 THEN 1 END) AS limit_up_count,
            COUNT(CASE WHEN (ts_code LIKE '30%' OR ts_code LIKE '68%') AND pct_chg <= -19.5 THEN 1
                       WHEN pct_chg <= -9.5 THEN 1 END) AS limit_down_count,
            COUNT(CASE WHEN pct_chg > 0 THEN 1 END) AS advance_count,
            COUNT(CASE WHEN pct_chg < 0 THEN 1 END) AS decline_count,
            COUNT(CASE WHEN pct_chg = 0 THEN 1 END) AS unchanged_count,
            COUNT(*) AS total_stocks,
            COUNT(CASE WHEN pct_chg > 0 THEN 1 END) * 1.0 /
                NULLIF(COUNT(CASE WHEN pct_chg < 0 THEN 1 END), 0) AS advance_decline_ratio,
            (COUNT(CASE WHEN pct_chg > 0 THEN 1 END) -
                COUNT(CASE WHEN pct_chg < 0 THEN 1 END)) * 1.0 / COUNT(*) AS market_breadth_pct
        FROM daily
        WHERE trade_date = ?
        GROUP BY trade_date
        """
        return self.conn.execute(query, [trade_date]).fetchdf()

    # ==================== 时序特征 ====================

    def compute_momentum_features(self, ts_code: str, end_date: str, lookback: int = 252) -> pd.DataFrame:
        """
        计算动量特征
        - return_1d/5d/10d/20d/60d/120d: 不同周期收益率
        - momentum_reversal: 动量反转信号
        """
        query = """
        WITH price_history AS (
            SELECT
                ts_code,
                trade_date,
                close,
                LAG(close, 1) OVER (ORDER BY trade_date) AS close_1d,
                LAG(close, 5) OVER (ORDER BY trade_date) AS close_5d,
                LAG(close, 10) OVER (ORDER BY trade_date) AS close_10d,
                LAG(close, 20) OVER (ORDER BY trade_date) AS close_20d,
                LAG(close, 60) OVER (ORDER BY trade_date) AS close_60d,
                LAG(close, 120) OVER (ORDER BY trade_date) AS close_120d
            FROM daily
            WHERE ts_code = ?
            ORDER BY trade_date
        )
        SELECT
            ts_code,
            trade_date,
            close,
            (close / NULLIF(close_1d, 0) - 1) * 100 AS return_1d,
            (close / NULLIF(close_5d, 0) - 1) * 100 AS return_5d,
            (close / NULLIF(close_10d, 0) - 1) * 100 AS return_10d,
            (close / NULLIF(close_20d, 0) - 1) * 100 AS return_20d,
            (close / NULLIF(close_60d, 0) - 1) * 100 AS return_60d,
            (close / NULLIF(close_120d, 0) - 1) * 100 AS return_120d,
            ((close / NULLIF(close_5d, 0)) - (close / NULLIF(close_20d, 0))) * 100 AS momentum_reversal
        FROM price_history
        WHERE trade_date <= ?
        ORDER BY trade_date DESC
        LIMIT ?
        """
        return self.conn.execute(query, [ts_code, end_date, lookback]).fetchdf()

    def compute_volatility_features(self, ts_code: str, end_date: str, lookback: int = 252) -> pd.DataFrame:
        """
        计算波动率特征
        - vol_5d/10d/20d/60d: 不同周期波动率
        - vol_change: 波动率变化
        - annualized_vol: 年化波动率
        """
        query = """
        WITH volatility_calc AS (
            SELECT
                ts_code,
                trade_date,
                pct_chg,
                STDDEV(pct_chg) OVER (ORDER BY trade_date ROWS 4 PRECEDING) AS vol_5d,
                STDDEV(pct_chg) OVER (ORDER BY trade_date ROWS 9 PRECEDING) AS vol_10d,
                STDDEV(pct_chg) OVER (ORDER BY trade_date ROWS 19 PRECEDING) AS vol_20d,
                STDDEV(pct_chg) OVER (ORDER BY trade_date ROWS 59 PRECEDING) AS vol_60d
            FROM daily
            WHERE ts_code = ?
            ORDER BY trade_date
        )
        SELECT
            ts_code,
            trade_date,
            pct_chg,
            vol_5d,
            vol_10d,
            vol_20d,
            vol_60d,
            vol_20d * SQRT(252) AS annualized_vol_20d,
            (vol_5d / NULLIF(vol_20d, 0) - 1) AS vol_change_ratio
        FROM volatility_calc
        WHERE trade_date <= ?
        ORDER BY trade_date DESC
        LIMIT ?
        """
        return self.conn.execute(query, [ts_code, end_date, lookback]).fetchdf()

    def compute_state_features(self, ts_code: str, end_date: str, lookback: int = 252) -> pd.DataFrame:
        """
        计算状态特征
        - ma5/10/20/60: 均线
        - ma_bull_alignment: 多头排列
        - bollinger_position: 布林带位置
        - position_in_range: 价格在区间内位置
        """
        query = """
        WITH state_calc AS (
            SELECT
                ts_code,
                trade_date,
                close,
                high,
                low,
                AVG(close) OVER (ORDER BY trade_date ROWS 4 PRECEDING) AS ma5,
                AVG(close) OVER (ORDER BY trade_date ROWS 9 PRECEDING) AS ma10,
                AVG(close) OVER (ORDER BY trade_date ROWS 19 PRECEDING) AS ma20,
                AVG(close) OVER (ORDER BY trade_date ROWS 59 PRECEDING) AS ma60,
                STDDEV(close) OVER (ORDER BY trade_date ROWS 19 PRECEDING) AS std_20,
                MAX(high) OVER (ORDER BY trade_date ROWS 19 PRECEDING) AS high_20d,
                MIN(low) OVER (ORDER BY trade_date ROWS 19 PRECEDING) AS low_20d
            FROM daily
            WHERE ts_code = ?
            ORDER BY trade_date
        )
        SELECT
            ts_code,
            trade_date,
            close,
            ma5, ma10, ma20, ma60,
            CASE WHEN ma5 > ma10 AND ma10 > ma20 AND ma20 > ma60 THEN 1 ELSE 0 END AS ma_bull_alignment,
            CASE WHEN ma5 < ma10 AND ma10 < ma20 AND ma20 < ma60 THEN 1 ELSE 0 END AS ma_bear_alignment,
            (close - low_20d) / NULLIF(high_20d - low_20d, 0) AS position_in_20d_range,
            (close - (ma20 - 2 * std_20)) / NULLIF(4 * std_20, 0) AS bollinger_position,
            (close - ma20) / NULLIF(std_20, 0) AS z_score,
            (close / ma20 - 1) * 100 AS price_ma20_deviation
        FROM state_calc
        WHERE trade_date <= ?
        ORDER BY trade_date DESC
        LIMIT ?
        """
        return self.conn.execute(query, [ts_code, end_date, lookback]).fetchdf()

    # ==================== 筹码特征 ====================

    def compute_chip_features(self, ts_code: str, end_date: str, lookback: int = 252) -> pd.DataFrame:
        """
        计算筹码特征
        - winner_rate: 获利盘比例
        - chip_concentration: 筹码集中度
        - price_vs_avg_cost: 价格相对平均成本
        """
        query = """
        SELECT
            c.ts_code,
            c.trade_date,
            d.close,
            c.winner_rate,
            (1 - c.winner_rate) AS loser_rate,
            c.weight_avg,
            (d.close / c.weight_avg - 1) * 100 AS price_vs_avg_cost,
            c.cost_5pct,
            c.cost_95pct,
            (c.cost_95pct - c.cost_5pct) / NULLIF(c.weight_avg, 0) AS chip_concentration,
            c.cost_50pct,
            (d.close - c.cost_50pct) / NULLIF(d.close, 0) AS peak_position,
            c.winner_rate - LAG(c.winner_rate, 1) OVER (ORDER BY c.trade_date) AS winner_rate_change_1d,
            c.winner_rate - LAG(c.winner_rate, 5) OVER (ORDER BY c.trade_date) AS winner_rate_change_5d
        FROM cyq_perf c
        JOIN daily d ON c.ts_code = d.ts_code AND c.trade_date = d.trade_date
        WHERE c.ts_code = ? AND c.trade_date <= ?
        ORDER BY c.trade_date DESC
        LIMIT ?
        """
        return self.conn.execute(query, [ts_code, end_date, lookback]).fetchdf()

    # ==================== 批量计算 ====================

    def compute_all_cross_sectional_features(self, trade_date: str) -> pd.DataFrame:
        """计算某一天所有横截面特征"""
        # 市场排名特征
        rank_features = self.compute_market_rank_features(trade_date)

        # 行业相对特征
        industry_features = self.compute_industry_relative_features(trade_date)

        # 合并
        result = rank_features.merge(
            industry_features[['ts_code', 'l1_code', 'l1_name', 'excess_return_industry',
                              'relative_turnover_industry', 'rank_in_industry', 'industry_strength']],
            on='ts_code',
            how='left'
        )

        return result

    def compute_all_timeseries_features(self, ts_code: str, end_date: str) -> pd.DataFrame:
        """计算某只股票所有时序特征"""
        # 动量特征
        momentum = self.compute_momentum_features(ts_code, end_date)

        # 波动率特征
        volatility = self.compute_volatility_features(ts_code, end_date)

        # 状态特征
        state = self.compute_state_features(ts_code, end_date)

        # 筹码特征
        chip = self.compute_chip_features(ts_code, end_date)

        # 合并
        result = momentum.merge(volatility[['ts_code', 'trade_date', 'vol_5d', 'vol_20d',
                                            'annualized_vol_20d', 'vol_change_ratio']],
                               on=['ts_code', 'trade_date'], how='left')

        result = result.merge(state[['ts_code', 'trade_date', 'ma5', 'ma20', 'ma60',
                                     'ma_bull_alignment', 'bollinger_position', 'z_score']],
                             on=['ts_code', 'trade_date'], how='left')

        result = result.merge(chip[['ts_code', 'trade_date', 'winner_rate', 'chip_concentration',
                                    'price_vs_avg_cost']],
                             on=['ts_code', 'trade_date'], how='left')

        return result


# ==================== 使用示例 ====================

if __name__ == "__main__":
    db_path = "/Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db"
    fe = FeatureEngineer(db_path)

    # 1. 计算横截面特征
    print("=== 横截面特征 ===")
    cs_features = fe.compute_all_cross_sectional_features('20260130')
    print(cs_features.head(10))

    # 2. 计算市场情绪
    print("\n=== 市场情绪 ===")
    sentiment = fe.compute_market_sentiment_features('20260130')
    print(sentiment)

    # 3. 计算时序特征
    print("\n=== 时序特征 ===")
    ts_features = fe.compute_all_timeseries_features('000001.SZ', '20260130')
    print(ts_features.head(10))

    fe.close()
```

---

## 五、特征汇总表

| 类别 | 特征名 | 类型 | 计算复杂度 | 使用场景 |
|------|--------|------|------------|----------|
| **横截面-排名** | pct_chg_rank | 百分位 | O(n log n) | 动量选股 |
| | vol_rank | 百分位 | O(n log n) | 量能筛选 |
| | turnover_rank | 百分位 | O(n log n) | 活跃度筛选 |
| | mktcap_rank | 百分位 | O(n log n) | 市值风格 |
| **横截面-行业** | excess_return_industry | 数值 | O(n) | Alpha构建 |
| | rank_in_industry | 百分位 | O(n log n) | 行业内选股 |
| | industry_strength | 数值 | O(n) | 板块轮动 |
| **横截面-情绪** | limit_up_count | 计数 | O(n) | 市场情绪 |
| | advance_decline_ratio | 比值 | O(n) | 市场宽度 |
| **时序-动量** | return_Nd | 百分比 | O(1) | 趋势跟踪 |
| | momentum_reversal | 数值 | O(1) | 反转策略 |
| **时序-波动** | vol_Nd | 数值 | O(n) | 风险管理 |
| | vol_change_ratio | 比值 | O(1) | 波动率交易 |
| **时序-状态** | ma_bull_alignment | 二值 | O(1) | 趋势确认 |
| | bollinger_position | [0,1] | O(n) | 超买超卖 |
| **筹码** | winner_rate | [0,1] | O(1) | 套牢盘分析 |
| | chip_concentration | 比值 | O(1) | 控盘度判断 |

---

## 六、性能优化建议

### 6.1 数据库层面
```sql
-- 创建复合索引加速查询
CREATE INDEX IF NOT EXISTS idx_daily_date_code ON daily(trade_date, ts_code);
CREATE INDEX IF NOT EXISTS idx_daily_basic_date_code ON daily_basic(trade_date, ts_code);
CREATE INDEX IF NOT EXISTS idx_cyq_perf_date_code ON cyq_perf(trade_date, ts_code);
CREATE INDEX IF NOT EXISTS idx_index_member_code ON index_member_all(ts_code, l1_code);
```

### 6.2 计算层面
1. **分批计算**: 对于全市场特征，按日期分批处理
2. **缓存中间结果**: 行业映射等静态数据可缓存
3. **并行计算**: 使用多进程处理不同股票的时序特征
4. **增量更新**: 对于历史数据，只计算新增部分

### 6.3 存储层面
1. **特征表**: 预计算的特征存入独立表
2. **分区存储**: 按日期分区存储横截面特征
3. **压缩存储**: 使用Parquet格式存储特征数据

---

## 七、注意事项

1. **数据对齐**: 确保所有数据按相同交易日对齐
2. **缺失值处理**: 新股或停牌股票的特征需特殊处理
3. **前视偏差**: 避免使用未来数据计算特征
4. **极值处理**: 对极端值进行截断或标准化
5. **行业变更**: 注意股票行业分类可能发生变化

---

## 八、验证结果示例

### 8.1 横截面特征验证 (2026-01-30)

**市场排名特征 (涨幅前10):**
```
  ts_code  pct_chg  pct_chg_rank  vol_rank  turnover_rank  mktcap_rank
920119.BJ 161.4613      1.000000  0.474089       1.000000     0.542941
688583.SH  20.0037      0.999817  0.044680       0.656107     0.664897
688025.SH  20.0000      0.999634  0.378868       0.878227     0.746933
301486.SZ  19.9988      0.999451  0.429958       0.949277     0.846548
300731.SZ  19.9931      0.999268  0.691998       0.983703     0.583959
```

**行业领涨股:**
```
  ts_code l1_name  pct_chg  excess_return_industry  industry_strength
920119.BJ    电力设备 161.4613              161.781739          -0.047455
688583.SH    机械设备  20.0037               19.623681           0.653003
301486.SZ      电子  19.9988               19.349734           0.922050
300731.SZ    基础化工  19.9931               19.535796           0.730288
300805.SZ      传媒  19.9847               19.929319           0.328365
```

**市场情绪:**
```
涨停数量: 51
跌停数量: 63
上涨家数: 2453
下跌家数: 2896
涨跌比: 0.8470
市场宽度: -0.0811
```

### 8.2 时序特征验证 (000001.SZ 平安银行)

**动量特征 (最近5天):**
```
trade_date  close  return_1d  return_5d  return_20d  momentum_reversal
  20260130  10.83    -1.1861  -1.455869   -5.083260           3.627391
  20260129  10.96     1.1070  -0.993677   -4.529617           3.535940
  20260128  10.84    -0.9141  -2.077687   -6.228374           4.150686
  20260127  10.94    -0.1825  -1.971326   -5.199307           3.227981
  20260126  10.96    -0.2730  -1.438849   -5.190311           3.751462
```

**波动率特征:**
```
trade_date   vol_5d  vol_20d  annualized_vol_20d  vol_ratio_5_20
  20260130 0.888250 0.744306           11.815498        1.193392
  20260129 0.790094 0.717140           11.384243        1.101729
```

**状态特征:**
```
trade_date  close  ma_score  bollinger_position   z_score  position_in_20d_range
  20260130  10.83         0            0.131987 -1.472054               0.146552
  20260129  10.96         0            0.210601 -1.157597               0.258621
```

**筹码特征:**
```
trade_date  close  winner_rate  chip_concentration  price_vs_avg_cost
  20260130  10.83        14.86            0.208153          -6.071119
  20260129  10.96        14.49            0.208153          -4.943625
```

**自相关特征:**
```
return_autocorr_lag1: 0.0281
volume_autocorr_lag1: 0.8394
volatility_clustering: 0.1790
```

### 8.3 综合特征视图 (000001.SZ 2026-01-30)

| 特征类别 | 特征名 | 值 |
|---------|--------|------|
| 排名 | pct_chg_rank | 0.3347 |
| 排名 | vol_rank | 0.9425 |
| 排名 | mktcap_rank | 0.9879 |
| 行业 | l1_name | 银行 |
| 行业 | excess_return_industry | -0.7774 |
| 行业 | industry_strength | -0.1357 |
| 动量 | return_5d | -1.4559 |
| 动量 | return_20d | -5.0833 |
| 波动 | annualized_vol_20d | 11.8155 |
| 状态 | ma_bull_alignment | 0 |
| 状态 | bollinger_position | 0.1320 |
| 状态 | z_score | -1.4721 |
| 筹码 | winner_rate | 14.86 |
| 筹码 | chip_concentration | 0.2082 |

---

*文档生成时间: 2026-01-31*
*数据库版本: DuckDB*
*Python实现: feature_engineer.py*
