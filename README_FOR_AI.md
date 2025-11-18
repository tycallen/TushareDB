# Tushare-DuckDB - AI 快速参考

> 项目路径: `/Users/allen/workspace/python/stock/Tushare-DuckDB`
> 这是一个高性能的中国A股数据查询库，使用DuckDB存储，查询速度比网络API快50-100倍

## 核心概念

**两个主要类：**
- `DataDownloader` - 下载数据到本地数据库（一次性/定期任务）
- `DataReader` - 查询本地数据（回测/分析，推荐使用）

**数据已准备好** - 16GB数据库已包含2020-2025年全部A股数据，无需下载

## 最常用的 5 个操作

### 1. 查询股票日线数据（最常用）
```python
from tushare_db import DataReader

reader = DataReader()

# 获取前复权日线数据（推荐用于计算收益率）
df = reader.get_stock_daily('000001.SZ', '20230101', '20231231', adj='qfq')
# 返回列: ts_code, trade_date, open, high, low, close, vol, amount

reader.close()
```

### 2. 查询多只股票数据（批量查询）
```python
reader = DataReader()

codes = ['000001.SZ', '000002.SZ', '600000.SH']
df = reader.get_multiple_stocks_daily(codes, '20230101', '20230131', adj='qfq')

reader.close()
```

### 3. 查询股票列表
```python
reader = DataReader()

# 获取所有上市股票
stocks = reader.get_stock_basic(list_status='L')
# 返回列: ts_code, name, industry, list_date, market

# 查找特定股票
banks = reader.query("SELECT * FROM stock_basic WHERE name LIKE '%银行%'")

reader.close()
```

### 4. 查询交易日历
```python
reader = DataReader()

# 获取某时间段的所有交易日
trading_days = reader.get_trade_calendar('20230101', '20231231', is_open='1')
dates_list = trading_days['cal_date'].tolist()

reader.close()
```

### 5. 自定义SQL查询（最灵活）
```python
reader = DataReader()

# 查找涨停板
df = reader.query("""
    SELECT ts_code, trade_date, close, pct_chg
    FROM pro_bar
    WHERE trade_date >= ? AND pct_chg > 9.9
    ORDER BY pct_chg DESC
""", ['20230101'])

reader.close()
```

## 重要的表名（供 SQL 查询）

| 表名 | 说明 | 主要字段 |
|------|------|---------|
| `stock_basic` | 股票列表 | ts_code, name, industry, list_date |
| `pro_bar` | 日线数据（未复权） | ts_code, trade_date, open, high, low, close, vol |
| `adj_factor` | 复权因子 | ts_code, trade_date, adj_factor |
| `daily_basic` | 每日指标 | ts_code, trade_date, pe, pb, total_mv, circ_mv |
| `trade_cal` | 交易日历 | cal_date, is_open, pretrade_date |

## 常见代码格式

- **深圳股票**: `000001.SZ`, `000002.SZ` (中小板), `300XXX.SZ` (创业板)
- **上海股票**: `600000.SH`, `601XXX.SH` (大盘股), `688XXX.SH` (科创板)
- **日期格式**: `YYYYMMDD`，如 `20230101`

## 性能提示

**回测场景推荐做法：**
```python
reader = DataReader()

# ✅ 好：一次性加载所有数据
all_data = reader.get_stock_daily('000001.SZ', '20200101', '20231231', adj='qfq')

# 然后在内存中过滤
for date in trading_dates:
    daily = all_data[all_data['trade_date'] == date]
    # 你的策略逻辑...

reader.close()

# ❌ 差：每天查询一次（慢100倍）
# for date in trading_dates:
#     daily = reader.get_stock_daily('000001.SZ', date, date)  # 避免这样
```

## Web API（如需HTTP访问）

```bash
# 启动服务器
uvicorn src.tushare_db.web_server:app --host 0.0.0.0 --port 8000

# API文档
http://localhost:8000/docs

# 示例请求
curl "http://localhost:8000/api/pro_bar?ts_code=000001.SZ&start_date=20230101&end_date=20230131&adjfactor=true"
```

## 完整文档

- 详细API: `API_REFERENCE_FOR_LLM.md`
- 迁移指南: `MIGRATION_GUIDE.md`
- 示例代码: `scripts/example_new_architecture.py`

## 错误处理

```python
from tushare_db import DataReader

reader = DataReader()

try:
    df = reader.get_stock_daily('000001.SZ', '20230101', '20230131')
    if df.empty:
        print("没有数据，可能需要先下载")
except Exception as e:
    print(f"查询失败: {e}")
finally:
    reader.close()
```

## 架构说明（供理解）

```
┌─────────────────┐      ┌──────────────┐
│ DataDownloader  │──>  │  DuckDB      │  <──┐
│ (写数据)        │      │  (16GB)      │     │
└─────────────────┘      └──────────────┘     │
                                               │
┌─────────────────┐                           │
│  DataReader     │───── 纯SQL查询 ────────────┘
│  (读数据)       │       (1-3ms/次)
└─────────────────┘
```

**关键特点：**
- 读写分离：下载和查询完全解耦
- 高性能：本地查询，无网络开销
- 可复现：静态数据，回测结果一致
