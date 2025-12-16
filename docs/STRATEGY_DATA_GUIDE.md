# 策略数据获取指南 (Strategy Data Access Guide)

本文档详细说明如何获取您策略所需的关键数据指标。

## 策略指标与数据表映射

| 策略指标 | 对应的数据库字段 | 来源表 | 说明 |
| :--- | :--- | :--- | :--- |
| **涨跌幅** | `pct_chg` | `pro_bar` | 单位为%，例如 +5.0 表示涨5% |
| **换手率** | `turnover_rate` | `pro_bar` | 单位为%，衡量股票流动性 |
| **获利筹码比例** | `winner_rate` | `cyq_perf` | 单位为%，例如 62 表示 62% 的筹码获利 |
| **股票名称(ST)** | `name` | `stock_basic` | 用于判断是否包含 "ST" |
| **涨停判断** | `pct_chg` | `pro_bar` | 通常 > 9.5% 或 19.5% (创业板/科创板) |

---

## 如何获取数据

### 1. 初始化 DataReader

首先，在您的 Python 脚本中引入并初始化 `DataReader`：

```python
import sys
import os
# 确保 src 在路径中
sys.path.append('src')

from tushare_db import DataReader

reader = DataReader()
```

### 2. 获取每日行情 (涨幅 & 换手率)

使用 `get_stock_daily` 方法获取 `pro_bar` 表数据：

```python
# 获取某只股票在特定日期范围的行情
df_price = reader.get_stock_daily(
    ts_code='000001.SZ', 
    start_date='20250101', 
    end_date='20250131'
)

# 查看所需字段
print(df_price[['trade_date', 'ts_code', 'pct_chg', 'turnover_rate']])
```

### 3. 获取筹码分布 (获利筹码)

目前 `DataReader` 没有直接封装 `get_cyq_perf` 方法，建议使用 SQL 查询 `cyq_perf` 表：

```python
# 获取某只股票的筹码获利比例
df_chips = reader.query(
    "SELECT trade_date, ts_code, winner_rate FROM cyq_perf WHERE ts_code='000001.SZ' AND trade_date BETWEEN '20250101' AND '20250131'"
)
print(df_chips)
```

### 4. 获取股票基本信息 (ST 判断)

使用 `get_stock_basic` 方法：

```python
df_basic = reader.get_stock_basic()

# 筛选非 ST 股票
non_st_stocks = df_basic[~df_basic['name'].str.contains('ST')]
print(non_st_stocks.head())
```

---

## 综合查询示例 (SQL)

为了高效验证策略，您可以直接编写 SQL 语句关联这些表。以下是一个查询示例，用于找出**某一天**满足 "涨幅<5%" 且 "获利筹码>62%" 的股票：

```python
target_date = '20251208'

sql = f"""
SELECT 
    p.ts_code, 
    p.pct_chg, 
    p.turnover_rate, 
    c.winner_rate,
    s.name
FROM pro_bar p
JOIN cyq_perf c ON p.ts_code = c.ts_code AND p.trade_date = c.trade_date
JOIN stock_basic s ON p.ts_code = s.ts_code
WHERE p.trade_date = '{target_date}'
  AND p.pct_chg <= 5            -- 涨幅 <= 5%
  AND c.winner_rate > 62        -- 获利筹码 > 62%
  AND s.name NOT LIKE '%ST%'    -- 排除 ST
LIMIT 10;
"""

df_result = reader.query(sql)
print(df_result)
```

## 注意事项

1.  **数据更新**：请确保每日运行 `python scripts/update_daily.py` 以更新 `pro_bar` 和 `cyq_perf` 数据。
2.  **筹码数据滞后**：交易所不直接公布筹码数据，`cyq_perf` 是基于 Tushare 算法计算的，每日更新可能需要一定计算时间。

---

*文档生成时间: 2025-12-09*
