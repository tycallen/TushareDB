# Tushare-DuckDB API Reference (For LLM/Agent Use)

> **Purpose**: This document is designed for LLMs and AI agents to understand how to use the Tushare-DuckDB library programmatically.

## 🎯 Quick Start

```python
# Install (if published to PyPI)
pip install tushare-db

# Or use directly from source
import sys
sys.path.append('/path/to/Tushare-DuckDB/src')
```

---

## 📚 Core Architecture

Tushare-DuckDB follows a **strict separation of concerns**:

```
┌────────────────────┐      ┌────────────────────┐
│  DataDownloader    │      │    DataReader      │
│  (Write to DB)     │      │  (Read from DB)    │
├────────────────────┤      ├────────────────────┤
│ • download_*()     │      │ • get_*()          │
│ • validate_*()     │      │ • query()          │
└────────┬───────────┘      └─────────┬──────────┘
         │                            │
         └──────────┬─────────────────┘
                    ▼
            ┌───────────────┐
            │   DuckDB      │
            │  (tushare.db) │
            └───────────────┘
```

**Key Principle**:
- Use `DataDownloader` for **data acquisition** (one-time or scheduled tasks)
- Use `DataReader` for **data queries** (backtesting, analysis, web APIs)

---

## 🔧 API Usage

### 1. Data Download (DataDownloader)

```python
from tushare_db import DataDownloader

# Initialize
downloader = DataDownloader(
    tushare_token='YOUR_TOKEN',  # Optional, reads from env TUSHARE_TOKEN
    db_path='tushare.db',
    rate_limit_profile='standard'  # 'trial', 'standard', 'pro'
)

# === Basic Data ===
# Download trading calendar (1990-2030)
downloader.download_trade_calendar('19900101', '20301231')

# Download stock list
downloader.download_stock_basic('L')  # L=listed, D=delisted, P=paused

# === Single Stock Data ===
# Download daily bars (unadjusted)
rows = downloader.download_stock_daily('000001.SZ', '20200101', '20231231')

# Download adjustment factors
rows = downloader.download_adj_factor('000001.SZ', '20200101', '20231231')

# Download daily fundamentals
rows = downloader.download_daily_basic('000001.SZ', '20200101', '20231231')

# === Batch Download ===
# Download all stocks (with progress bar)
downloader.download_all_stocks_daily('20200101', '20231231', list_status='L')
downloader.download_all_adj_factors('20200101', '20231231', list_status='L')

# === Daily Update (for scheduled tasks) ===
# Download data by specific date
downloader.download_daily_data_by_date('20240118')
# This downloads: pro_bar + adj_factor + daily_basic for all stocks

# === Data Validation ===
result = downloader.validate_data_integrity('20200101', '20231231', sample_size=10)
print(result['is_valid'])  # True/False
print(result['sample_stocks'])  # List of validation results

# Close connection
downloader.close()
```

### 2. Data Query (DataReader)

```python
from tushare_db import DataReader

# Initialize (read-only mode)
reader = DataReader(
    db_path='tushare.db',
    strict_mode=False  # If True, raises exception when data not found
)

# === Basic Info ===
# Get stock list
stocks = reader.get_stock_basic(list_status='L')
print(stocks[['ts_code', 'name', 'industry']])

# Get trading calendar
cal = reader.get_trade_calendar('20230101', '20231231', is_open='1')
print(f"Trading days: {len(cal)}")

# === Stock Data ===
# Get daily bars (unadjusted)
df = reader.get_stock_daily('000001.SZ', '20230101', '20231231')

# Get daily bars (qfq=forward adjusted, hfq=backward adjusted)
df_qfq = reader.get_stock_daily('000001.SZ', '20230101', '20231231', adj='qfq')
df_hfq = reader.get_stock_daily('000001.SZ', '20230101', '20231231', adj='hfq')

# Batch query multiple stocks
codes = ['000001.SZ', '000002.SZ', '600000.SH']
df_batch = reader.get_multiple_stocks_daily(codes, '20230101', '20230131', adj='qfq')

# === Other Data ===
# Daily fundamentals (PE, PB, market cap, etc.)
df_basic = reader.get_daily_basic('000001.SZ', '20230101', '20231231')

# Adjustment factors
df_adj = reader.get_adj_factor('000001.SZ', '20230101', '20231231')

# Company info
company = reader.get_stock_company('000001.SZ')

# Technical factors
factors = reader.get_stk_factor_pro('000001.SZ', '20230101', '20231231')

# === Custom SQL Query ===
# Direct SQL access for advanced queries
df = reader.query(
    "SELECT * FROM pro_bar WHERE ts_code = ? AND trade_date >= ?",
    ['000001.SZ', '20230101']
)

# Find stocks by name
banks = reader.query(
    "SELECT ts_code, name FROM stock_basic WHERE name LIKE ?",
    ['%银行%']
)

# Close connection
reader.close()
```

---

## 🌐 Web API (FastAPI)

The project includes a FastAPI web server for HTTP access:

```bash
# Start server
uvicorn src.tushare_db.web_server:app --host 0.0.0.0 --port 8000

# Or in Python
python -c "from src.tushare_db.web_server import app; import uvicorn; uvicorn.run(app, host='0.0.0.0', port=8000)"
```

**API Endpoints**:
```
GET /                              - Health check
GET /api/stock_basic               - Stock list
GET /api/trade_cal                 - Trading calendar
GET /api/pro_bar                   - Daily bars
GET /api/adj_factor                - Adjustment factors
GET /api/daily_basic               - Daily fundamentals
GET /api/stock_company             - Company info
GET /api/cyq_perf                  - Chip cost
GET /api/stk_factor_pro            - Technical factors
... (17 endpoints total)
```

**Example HTTP Request**:
```bash
# Get stock daily data
curl "http://localhost:8000/api/pro_bar?ts_code=000001.SZ&start_date=20230101&end_date=20230131"

# Response: JSON array of OHLCV data
[
  {
    "ts_code": "000001.SZ",
    "trade_date": "20230103",
    "open": 13.50,
    "high": 13.80,
    "low": 13.40,
    "close": 13.70,
    "vol": 12345678,
    ...
  },
  ...
]
```

**Interactive API Docs**:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

## 🔍 Common Use Cases

### Use Case 1: Initialize Database
```python
from tushare_db import DataDownloader

downloader = DataDownloader()

# Step 1: Download basic data
downloader.download_trade_calendar()
downloader.download_stock_basic('L')

# Step 2: Download historical data (example: 5 stocks)
stocks = ['000001.SZ', '000002.SZ', '600000.SH', '600519.SH', '000858.SZ']
for code in stocks:
    downloader.download_stock_daily(code, '20200101', '20231231')
    downloader.download_adj_factor(code, '20200101', '20231231')

# Step 3: Validate
result = downloader.validate_data_integrity('20200101', '20231231')
print(f"Data complete: {result['is_valid']}")

downloader.close()
```

### Use Case 2: Daily Update (Scheduled Task)
```python
from tushare_db import DataDownloader
from datetime import datetime, timedelta

downloader = DataDownloader()

# Get yesterday's date
yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')

# Download all data for that date
downloader.download_daily_data_by_date(yesterday)

downloader.close()
```

### Use Case 3: Backtesting
```python
from tushare_db import DataReader
import pandas as pd

reader = DataReader()

# 1. Get trading calendar
trading_dates = reader.get_trade_calendar('20200101', '20231231', is_open='1')['cal_date'].tolist()

# 2. Load all data once (recommended for performance)
data = reader.get_stock_daily('000001.SZ', '20200101', '20231231', adj='qfq')

# 3. Backtest loop (pure in-memory operations, very fast!)
for date in trading_dates:
    daily_data = data[data['trade_date'] == date]
    # ... your strategy logic ...

reader.close()
```

### Use Case 4: Data Analysis
```python
from tushare_db import DataReader

reader = DataReader()

# Get all banks
banks = reader.query(
    "SELECT ts_code, name FROM stock_basic WHERE industry = '银行'",
)

# Load their data
all_data = []
for ts_code in banks['ts_code']:
    df = reader.get_stock_daily(ts_code, '20230101', '20231231', adj='qfq')
    all_data.append(df)

combined = pd.concat(all_data, ignore_index=True)

# Analysis
print(combined.groupby('ts_code')['close'].agg(['mean', 'std']))

reader.close()
```

---

## 📊 Data Schema

### Table: stock_basic
| Column | Type | Description |
|--------|------|-------------|
| ts_code | VARCHAR | Stock code (e.g., '000001.SZ') |
| symbol | VARCHAR | Stock symbol (e.g., '000001') |
| name | VARCHAR | Stock name (e.g., '平安银行') |
| industry | VARCHAR | Industry (e.g., '银行') |
| list_status | VARCHAR | Status ('L'=listed, 'D'=delisted) |
| list_date | VARCHAR | List date (YYYYMMDD) |
| market | VARCHAR | Market (e.g., '主板', '创业板') |

### Table: pro_bar (daily bars, unadjusted)
| Column | Type | Description |
|--------|------|-------------|
| ts_code | VARCHAR | Stock code |
| trade_date | VARCHAR | Date (YYYYMMDD) |
| open | DOUBLE | Open price |
| high | DOUBLE | High price |
| low | DOUBLE | Low price |
| close | DOUBLE | Close price |
| vol | DOUBLE | Volume (shares) |
| amount | DOUBLE | Amount (CNY) |

### Table: adj_factor (adjustment factors)
| Column | Type | Description |
|--------|------|-------------|
| ts_code | VARCHAR | Stock code |
| trade_date | VARCHAR | Date (YYYYMMDD) |
| adj_factor | DOUBLE | Adjustment factor |

**Adjustment Calculation**:
```python
# Forward adjustment (qfq)
adjusted_close = close * adj_factor

# Backward adjustment (hfq)
adjusted_close = close * (adj_factor / latest_adj_factor)
```

### Table: trade_cal
| Column | Type | Description |
|--------|------|-------------|
| exchange | VARCHAR | Exchange (SSE/SZSE) |
| cal_date | VARCHAR | Date (YYYYMMDD) |
| is_open | INTEGER | Trading day (1=yes, 0=no) |
| pretrade_date | VARCHAR | Previous trading date |

---

## ⚡ Performance Tips

### 1. Pre-load Data for Backtesting
```python
# Bad: Query for each day (slow!)
for date in dates:
    df = reader.get_stock_daily(code, date, date)  # 1000+ queries!

# Good: Load once, filter in memory (fast!)
df_all = reader.get_stock_daily(code, start_date, end_date)
for date in dates:
    daily = df_all[df_all['trade_date'] == date]  # In-memory filter
```

### 2. Batch Query Multiple Stocks
```python
# Bad: Loop one by one
for code in codes:
    df = reader.get_stock_daily(code, start, end)  # N queries

# Good: Batch query
df = reader.get_multiple_stocks_daily(codes, start, end)  # 1 query!
```

### 3. Use Custom SQL for Complex Queries
```python
# Complex filter with SQL (very fast!)
df = reader.query("""
    SELECT * FROM pro_bar
    WHERE trade_date BETWEEN ? AND ?
    AND close > open * 1.05  -- Up > 5%
    AND vol > 100000000      -- Volume > 100M
""", [start_date, end_date])
```

---

## 🛠️ Environment Setup

### Method 1: Using Environment Variables
```bash
export TUSHARE_TOKEN='your_token_here'

python your_script.py
```

### Method 2: Pass Token Directly
```python
from tushare_db import DataDownloader

downloader = DataDownloader(tushare_token='your_token_here')
```

### Method 3: .env File (with python-dotenv)
```bash
# .env file
TUSHARE_TOKEN=your_token_here
```

```python
from dotenv import load_dotenv
load_dotenv()

# Token automatically loaded from environment
from tushare_db import DataDownloader
downloader = DataDownloader()
```

---

## 🐛 Error Handling

```python
from tushare_db import DataReader, DataReaderError

reader = DataReader(strict_mode=True)

try:
    df = reader.get_stock_daily('000001.SZ', '20230101', '20230131')
except DataReaderError as e:
    print(f"Data not found: {e}")
    print("Hint: Please download data first using DataDownloader")
```

---

## 📦 Package Installation (If Published)

```bash
# From PyPI (future)
pip install tushare-db

# From source
git clone https://github.com/your-repo/Tushare-DuckDB.git
cd Tushare-DuckDB
pip install -e .
```

---

## 🤖 LLM Integration Tips

### For Code Generation
When generating code using this library:

1. **Always separate download and query**:
   - Use `DataDownloader` in initialization/update scripts
   - Use `DataReader` in analysis/backtesting code

2. **Check data existence before querying**:
   ```python
   if reader.table_exists('pro_bar'):
       df = reader.get_stock_daily(...)
   else:
       print("Data not downloaded yet")
   ```

3. **Use context managers** (optional, but recommended):
   ```python
   with DataReader() as reader:
       df = reader.get_stock_daily(...)
   # Auto-closes connection
   ```

### For Tool Calling (Function Calling)
Define tools like this:

```json
{
  "name": "query_stock_data",
  "description": "Query stock daily OHLCV data from Tushare-DuckDB",
  "parameters": {
    "type": "object",
    "properties": {
      "ts_code": {"type": "string", "description": "Stock code (e.g., '000001.SZ')"},
      "start_date": {"type": "string", "description": "Start date (YYYYMMDD)"},
      "end_date": {"type": "string", "description": "End date (YYYYMMDD)"},
      "adj": {"type": "string", "enum": [null, "qfq", "hfq"], "description": "Adjustment type"}
    },
    "required": ["ts_code", "start_date", "end_date"]
  }
}
```

Implementation:
```python
def query_stock_data(ts_code, start_date, end_date, adj=None):
    from tushare_db import DataReader
    reader = DataReader()
    df = reader.get_stock_daily(ts_code, start_date, end_date, adj=adj)
    reader.close()
    return df.to_dict('records')
```

---

## 📞 Support

- Documentation: `/docs` (this file + MIGRATION_GUIDE.md)
- Examples: `/scripts/example_new_architecture.py`
- Tests: `/scripts/test_new_architecture.py`
- Web API Docs: http://localhost:8000/docs (when server running)

---

**Version**: 2.0 (New Architecture)
**Last Updated**: 2024-11-18
**License**: MIT (or your license)
