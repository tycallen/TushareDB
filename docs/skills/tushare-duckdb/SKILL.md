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

| Table | Tushare API | Primary Keys | Fixed/Default Params |
|-------|-------------|--------------|----------------------|
| `trade_cal` | trade_cal | exchange, cal_date | exchange='SSE' |
| `stock_basic` | stock_basic | ts_code | — |
| `stock_company` | stock_company | ts_code | — |
| `daily` | daily | ts_code, trade_date | asset='E' (stocks), 'I' (index) |
| `index_daily` | index_daily | ts_code, trade_date | 主要指数日线数据 |
| `adj_factor` | adj_factor | ts_code, trade_date | — |
| `daily_basic` | daily_basic | ts_code, trade_date | — |
| `index_basic` | index_basic | ts_code | — |
| `index_classify` | index_classify | industry_code | src='SW2021' |
| `index_member_all` | index_member_all | ts_code, l3_code, in_date | — |
| `index_weight` | index_weight | index_code, trade_date, con_code | — |
| `cyq_perf` | cyq_perf | ts_code, trade_date | — |
| `stk_factor_pro` | stk_factor_pro | ts_code, trade_date | — |
| `moneyflow` | moneyflow | ts_code, trade_date | — |
| `moneyflow_dc` | moneyflow_dc | ts_code, trade_date | — |
| `moneyflow_ind_dc` | moneyflow_ind_dc | trade_date, ts_code | — |
| `dc_index` | dc_index | ts_code, trade_date | 龙虎榜个股明细 |
| `dc_member` | dc_member | ts_code, trade_date, con_code | 龙虎榜机构席位 |
| `limit_list_d` | limit_list_d | ts_code, trade_date, limit | 涨跌停炸板 (U/D/Z) |
| `fina_indicator_vip` | fina_indicator_vip | ts_code, end_date | — |
| `income` | income / income_vip | ts_code, end_date, report_type | — |
| `balancesheet` | balancesheet / balancesheet_vip | ts_code, end_date, report_type | — |
| `cashflow` | cashflow / cashflow_vip | ts_code, end_date, report_type | — |
| `dividend` | dividend | ts_code, end_date | — |
| `margin_detail` | margin_detail | ts_code, trade_date | — |
| `hs_const` | hs_const | ts_code, in_date | — |
| `sw_daily` | sw_daily | ts_code, trade_date | 申万2021版 |
| `kpl_concept` | kpl_concept | ts_code, trade_date | 开盘啦题材列表 |
| `kpl_concept_cons` | kpl_concept_cons | ts_code, con_code, trade_date | 开盘啦题材成分 |

For column details and parameter meanings, invoke `/tushare-finance <api_name>`.
