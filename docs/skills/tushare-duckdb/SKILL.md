---
name: tushare-duckdb
description: Query Chinese stock data from local DuckDB cache. Use when user wants local data access instead of direct Tushare API calls.
---

<!--
INSTALLATION:
  Global:  cp -r docs/skills/tushare-duckdb ~/.claude/skills/
  Project: cp -r docs/skills/tushare-duckdb .claude/skills/
-->

# tushare-duckdb

Local DuckDB cache layer for Tushare Pro data. For API parameter details, use `/tushare-finance <api_name>`.

## Quick Start

### Read Data (no network, concurrent-safe)

```python
from tushare_db import DataReader

reader = DataReader(db_path="tushare.db")

# Single stock with forward adjustment
df = reader.get_stock_daily('000001.SZ', '20240101', adj='qfq')

# Multiple stocks
df = reader.get_multiple_stocks_daily(['000001.SZ', '600519.SH'], '20240101')

# Trade calendar
df = reader.get_trade_calendar('20240101', '20241231')

# Stock list
df = reader.get_stock_basic(list_status='L')

# Custom SQL
df = reader.query("SELECT * FROM daily WHERE ts_code = ?", ['000001.SZ'])

reader.close()
```

### Download Data (requires TUSHARE_TOKEN env var)

```python
from tushare_db import DataDownloader

downloader = DataDownloader(db_path="tushare.db", rate_limit_profile="standard")

# Basic data
downloader.download_trade_calendar()
downloader.download_stock_basic()

# Daily data for all stocks
downloader.download_all_stocks_daily(start_date='20200101')

# Daily update (single date)
downloader.download_daily_data_by_date('20241218')

downloader.close()
```

## Key Notes

- **Date format**: `YYYYMMDD` string (e.g., `'20240101'`)
- **Adjustment types**:
  - `adj_factor` table stores raw Tushare adjustment factors
  - `qfq` (forward): `price * (factor / latest_factor)` — latest price = market price
  - `hfq` (backward): `price * factor` — historical prices stay fixed
  - DataReader handles calculation automatically via `adj='qfq'` or `adj='hfq'`
- **Rate limit profiles**: `'trial'`, `'standard'`, `'pro'`

## Implemented Tables

### 基础信息表（静态数据）

| Table | Tushare API | Primary Keys | 说明 |
|-------|-------------|--------------|------|
| `trade_cal` | trade_cal | exchange, cal_date | 交易日历，exchange='SSE' |
| `stock_basic` | stock_basic | ts_code | 股票列表 |
| `stock_company` | stock_company | ts_code | 上市公司信息 |
| `index_basic` | index_basic | ts_code | 指数列表 |
| `index_classify` | index_classify | industry_code | 申万行业分类，src='SW2021' |
| `index_member_all` | index_member_all | ts_code, l3_code, in_date | 申万行业成分股 |
| `hs_const` | hs_const | ts_code, in_date | 沪深港通成分 |
| `ths_index` | ths_index | ts_code | 同花顺板块列表 (见下方说明) |
| `ths_member` | ths_member | ts_code, con_code | 同花顺板块成分股 |

### 日频时间序列表

| Table | Tushare API | Primary Keys | 官方最早日期 | 说明 |
|-------|-------------|--------------|-------------|------|
| `daily` | daily | ts_code, trade_date | 19910404 | 日线行情 |
| `adj_factor` | adj_factor | ts_code, trade_date | 19910403 | 复权因子 |
| `daily_basic` | daily_basic | ts_code, trade_date | 19910404 | 每日指标（PE/PB/市值） |
| `index_daily` | index_daily | ts_code, trade_date | 19901219 | 指数日线 |
| `moneyflow` | moneyflow | ts_code, trade_date | **20100104** | 个股资金流向 |
| `cyq_perf` | cyq_perf | ts_code, trade_date | **20180102** | 筹码分布绩效 |
| `stk_factor_pro` | stk_factor_pro | ts_code, trade_date | **20050104** | 技术因子（MACD/KDJ等） |
| `limit_list_d` | limit_list_d | ts_code, trade_date, limit | **20200102** | 涨跌停统计 (U/D/Z) |
| `margin_detail` | margin_detail | ts_code, trade_date | **20100331** | 融资融券明细 |
| `sw_daily` | sw_daily | ts_code, trade_date | **20210104** | 申万指数日线 (SW2021版) |
| `ths_daily` | ths_daily | ts_code, trade_date | **20180102** | 同花顺板块日行情 |
| `moneyflow_dc` | moneyflow_dc | ts_code, trade_date | **20230911** | 个股资金流(东财) |
| `moneyflow_ind_dc` | moneyflow_ind_dc | trade_date, ts_code | **20230912** | 行业资金流(东财) |
| `dc_index` | dc_index | ts_code, trade_date | **20241220** | 龙虎榜个股明细 |
| `dc_member` | dc_member | ts_code, trade_date, con_code | **20241220** | 龙虎榜机构席位 |
| `kpl_concept` | kpl_concept | ts_code, trade_date | **20241014** | 开盘啦题材列表 |
| `kpl_concept_cons` | kpl_concept_cons | ts_code, con_code, trade_date | **20251001** | 开盘啦题材成分 |
| `index_weight` | index_weight | index_code, trade_date, con_code | **20050930** | 指数成分权重（月度） |

### 财务数据表（季度更新）

| Table | Tushare API | Primary Keys | 官方最早日期 | 说明 |
|-------|-------------|--------------|-------------|------|
| `fina_indicator_vip` | fina_indicator_vip | ts_code, end_date | 19901231 | 财务指标 |
| `income` | income_vip | ts_code, end_date, report_type | 19941231 | 利润表 |
| `balancesheet` | balancesheet_vip | ts_code, end_date, report_type | 19891231 | 资产负债表 |
| `cashflow` | cashflow_vip | ts_code, end_date, report_type | 19980331 | 现金流量表 |
| `dividend` | dividend | ts_code, end_date | 19901231 | 分红送股 |

### 基金数据表

| Table | Tushare API | Primary Keys | 说明 |
|-------|-------------|--------------|------|
| `fund_basic` | fund_basic | ts_code | 基金列表 (E=场内 O=场外) |
| `fund_daily` | fund_daily | ts_code, trade_date | 场内基金日线行情 |
| `fund_nav` | fund_nav | ts_code, nav_date | 基金净值 (单位/累计/复权) |
| `fund_div` | fund_div | ts_code, ann_date | 基金分红 |
| `fund_portfolio` | fund_portfolio | ts_code, end_date, symbol | 基金持仓 (十大重仓股) |
| `fund_share` | fund_share | ts_code, trade_date | 基金份额变动 |
| `fund_manager` | fund_manager | ts_code, begin_date, name | 基金经理 |
| `fund_adj` | fund_adj | ts_code, trade_date | 基金复权因子 |

### 沪深港通数据表

| Table | Tushare API | Primary Keys | 说明 |
|-------|-------------|--------------|------|
| `moneyflow_hsgt` | moneyflow_hsgt | trade_date | 沪深港通资金流向 (北向/南向) |
| `hsgt_top10` | hsgt_top10 | trade_date, ts_code, market_type | 沪深股通十大成交股 |
| `ggt_top10` | ggt_top10 | trade_date, ts_code, market_type | 港股通十大成交股 |
| `ggt_daily` | ggt_daily | trade_date | 港股通每日成交统计 |
| `hk_hold` | hk_hold | code, trade_date, exchange | 沪深港通持股明细 |

### 同花顺板块说明

`ths_index` 表包含以下类型的板块指数，本项目**仅实现部分类型**：

| 类型 | 说明 | 数量 | 实现状态 |
|------|------|------|----------|
| **I** | 行业板块 | 1077 | ✅ 已实现 |
| **N** | 概念板块 | 411 | ✅ 已实现 |
| **R** | 地域板块 | 33 | ✅ 已实现 |
| **BB** | 宽基指数 | 46 | ✅ 已实现 |
| S | 特色指数 | 126 | ❌ 未实现 (技术面筛选，使用少) |
| ST | 风格指数 | 21 | ❌ 未实现 (可用其他方式构建) |
| TH | 同花顺特色 | 10 | ❌ 未实现 (专有指数，可替代性强) |

For column details and parameter meanings, invoke `/tushare-finance <api_name>`.
