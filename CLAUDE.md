# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Tushare-DuckDB is a Python library that provides a local caching layer for [Tushare Pro](https://tushare.pro/) financial data using [DuckDB](https://duckdb.org/). It reduces API calls, speeds up data access, and enables local quantitative analysis for Chinese stock market data (A-shares, indices, ETFs).

## Common Commands

### Installation
```bash
pip install -e .
```

### Data Operations
```bash
# Initialize historical data (full sync)
python scripts/init_data.py --trade-cal --stock-basic --pro-bar

# Daily incremental update
python scripts/update_daily.py
```

### Running Tests
```bash
pytest                    # Run all tests
pytest tests/             # Run tests in tests directory
```

### Environment Setup
Copy `.env.example` to `.env` and set `TUSHARE_TOKEN`:
```bash
cp .env.example .env
# Edit .env with your Tushare Pro API token
```

## Architecture

The codebase uses a **separation of concerns** pattern with two main components:

### DataDownloader (`src/tushare_db/downloader.py`)
- **Write-only**: Downloads data from Tushare API and stores in DuckDB
- Used for initialization scripts and daily updates
- No query functionality, simple fetch -> write flow
- Requires `TUSHARE_TOKEN` environment variable

### DataReader (`src/tushare_db/reader.py`)
- **Read-only**: Queries data from local DuckDB database
- Zero network requests, millisecond response times
- Opens database in `read_only=True` mode for concurrent access
- Used for backtesting, web APIs, data analysis

### Key Internal Components

- **DuckDBManager** (`duckdb_manager.py`): Handles database operations with UPSERT support
- **TushareFetcher** (`tushare_fetcher.py`): Wraps Tushare API with rate limiting
- **Rate Limit Config** (`rate_limit_config.py`): Pre-configured profiles for different subscription levels (trial/standard/pro)

### Data Flow
```
Tushare API → TushareFetcher → DataDownloader → DuckDB
                                                  ↓
                              DataReader ← DuckDB (read_only)
```

## Usage Patterns

### Download Data
```python
from tushare_db import DataDownloader

downloader = DataDownloader(db_path="tushare.db")
downloader.download_trade_calendar()
downloader.download_stock_basic(list_status='L')
downloader.download_all_stocks_daily(start_date='20200101')
downloader.close()
```

### Query Data
```python
from tushare_db import DataReader

reader = DataReader(db_path="tushare.db")
df = reader.get_stock_daily('000001.SZ', '20240101', adj='qfq')
reader.close()
```

## Database Schema

Primary keys are defined in `TABLE_PRIMARY_KEYS` dict in `duckdb_manager.py`. Key tables:

| Table | Primary Keys | Description |
|-------|-------------|-------------|
| `daily` | ts_code, trade_date | Daily OHLCV (stocks and indices) |
| `adj_factor` | ts_code, trade_date | Price adjustment factors |
| `daily_basic` | ts_code, trade_date | Daily PE/PB/市值等 |
| `stock_basic` | ts_code | Stock metadata |
| `trade_cal` | exchange, cal_date | Trading calendar |
| `index_classify` | index_code | 申万行业分类 |
| `index_member_all` | ts_code, l3_code, in_date | 申万行业成分（PIT支持） |
| `index_weight` | index_code, con_code, trade_date | 指数成分权重 |
| `moneyflow` | ts_code, trade_date | 个股资金流向 |
| `cyq_perf` | ts_code, trade_date | 筹码分布 |
| `dc_member` | ts_code, trade_date, exalter | 龙虎榜机构明细 |
| `fina_indicator_vip` | ts_code, end_date | 财务指标（VIP） |

## Important Notes

### Adjustment Factors
- **Forward adjustment (qfq)**: `price × (adj_factor / latest_factor)` - Most recent price equals market price
- **Backward adjustment (hfq)**: `price × adj_factor` - Historical prices stay fixed
- `pro_bar(adj='hfq')` returns `price × adj_factor` (not divided by latest factor)

### Rate Limiting
The library enforces Tushare API rate limits based on subscription level. Configure via `rate_limit_profile` parameter: `'trial'`, `'standard'`, or `'pro'`.

### Date Format
All dates use `YYYYMMDD` string format (e.g., `'20240101'`).

### Point-in-Time (PIT) Data Support

`index_member_all` table now supports Point-in-Time queries for backtesting:

```python
from tushare_db import DataReader

reader = DataReader(db_path="tushare.db")

# PIT query: Get sector constituents on a specific historical date
df = reader.get_index_member_all(
    l1_code='801010.SI',  # 农林牧渔
    trade_date='20230115'  # Query as of 2023-01-15
)
# Returns only stocks that were in the sector on that date
# (in_date <= trade_date AND (out_date IS NULL OR out_date > trade_date))
```

**Data Structure:**
- `is_new='Y'`: Current members (out_date is NULL)
- `is_new='N'`: Historical members (out_date has value)

**Important:** Always use `trade_date` parameter for PIT queries. Using only `is_new='Y'` gives current snapshot which causes look-ahead bias in backtests.

**Manual Data Backfill:**
```bash
# If your existing database lacks PIT data, run:
python scripts/backfill_index_member_pit.py
```

## Factor Validation (Monte Carlo)

The project includes a Monte Carlo factor validation system for testing technical indicators.

### Quick Start

```python
from tushare_db import EnhancedFactorFilter

# Create filter with auto-save
filter_obj = EnhancedFactorFilter(db_path="tushare.db")

# Run validation
report = filter_obj.filter(
    factor="macd_golden_cross",
    ts_codes=['000001.SZ']
)

# View results
print(report.markdown)
print(f"Alpha Ratio: {report.summary['alpha_ratio_median']}")

# Query history
history = filter_obj.get_factor_history("macd_golden_cross")
stats = filter_obj.get_statistics(days=30)
```

### Builtin Factors (30 total)

**Trend Indicators:**
- `macd_golden_cross`, `macd_death_cross`, `macd_zero_golden_cross`
- `golden_cross`, `death_cross` (MA cross)
- `price_above_sma`, `price_below_sma`

**Momentum Indicators:**
- `rsi_oversold`, `rsi_overbought`
- `kdj_golden_cross`, `kdj_death_cross`
- `williams_r_oversold`, `williams_r_overbought`
- `cci_oversold`, `cci_overbought`

**Volatility Indicators:**
- `bollinger_lower_break`, `bollinger_upper_break`
- `atr_breakout`, `atr_breakdown`

**Volume Indicators:**
- `volume_breakout`

**Candlestick Patterns:**
- `bullish_engulfing`, `bearish_engulfing`
- `hammer`, `shooting_star`, `doji`
- `three_white_soldiers`, `three_black_crows`

**Price Patterns:**
- `close_gt_open`, `gap_up`, `gap_down`

### Report Management

Validation results are automatically saved to `~/.factor_validation/reports.db`:

```python
from src.tushare_db.factor_validation.report_manager import ReportManager

manager = ReportManager()

# Query records
df = manager.query_records(
    factor_name="macd_golden_cross",
    recommendation="KEEP"
)

# Compare over time
df_time = manager.compare_factor_over_time(
    factor_name="macd_golden_cross",
    ts_code="000001.SZ"
)

# Export
manager.export_to_csv("reports.csv")
manager.export_to_excel("reports.xlsx")
```

See `REPORT_MANAGER_GUIDE.md` for detailed documentation.
## Skill Maintenance

This project provides a Claude Code skill at `docs/skills/tushare-duckdb/SKILL.md`.

**IMPORTANT**: When adding or removing tables in `TABLE_PRIMARY_KEYS` (in `duckdb_manager.py`), you MUST update the skill file to match. Run validation:

```bash
python scripts/validate_skill_sync.py
```

The script checks that all tables in `TABLE_PRIMARY_KEYS` are documented in the skill file.

Users can install the skill by copying to `~/.claude/skills/` or `.claude/skills/`.
