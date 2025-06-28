# TushareDB

A Python library for efficiently fetching and managing Tushare financial data with a local DuckDB cache.

## Overview

TushareDB provides a robust and easy-to-use interface to interact with the Tushare Pro API, intelligently caching data in a local DuckDB database. This allows for faster subsequent data retrieval, reduces API calls, and simplifies local financial data analysis.

## Features

*   **Unified Interface**: A single `TushareDBClient` to fetch various Tushare API data.
*   **Transparent Caching**: Automatically stores fetched data in a local DuckDB file.
*   **Smart Update Strategies**: Supports both incremental updates (for time-series data like daily prices) and full refreshes with configurable Time-To-Live (TTL) for static data (like stock basic information).
*   **API Rate Limiting**: Built-in thread-safe mechanism to strictly adhere to Tushare Pro API call frequency limits.
*   **Efficient Local Storage**: Leverages DuckDB for high-performance analytical queries on your local data.

## Installation

To install TushareDB, you can use pip:

```bash
pip install tusharedb # (Placeholder - replace with actual package name when published)
```

For now, you can clone the repository and install it locally:

```bash
git clone https://github.com/your-repo/Tushare-DuckDB.git # Replace with actual repo URL
cd Tushare-DuckDB
pip install -e .
```

## Configuration

TushareDB requires your Tushare Pro API token. You can provide it in two ways:

1.  **Environment Variable (Recommended)**: Set the `TUSHARE_TOKEN` environment variable.

    ```bash
    export TUSHARE_TOKEN="YOUR_TUSHARE_PRO_TOKEN"
    ```

2.  **Directly in Code**: Pass the token directly to the `TushareDBClient` constructor.

    ```python
    from tushare_db import TushareDBClient
    client = TushareDBClient(tushare_token="YOUR_TUSHARE_PRO_TOKEN")
    ```

## Quick Start

Here's a quick example demonstrating how to use `TushareDBClient` to fetch data with caching.

```python
import os
from tushare_db import TushareDBClient

# Ensure your TUSHARE_TOKEN environment variable is set
# export TUSHARE_TOKEN="YOUR_TUSHARE_PRO_TOKEN"

# Initialize the client
# The database will be created as 'tushare.db' in the current directory
# You can specify a different path: db_path='data/my_tushare.db'
client = TushareDBClient()

# --- Example 1: Fetching daily stock data (Incremental Update Policy) ---
print("\n--- Fetching daily stock data (ts_code='000001.SZ') ---")
daily_df = client.get_data('daily', ts_code='000001.SZ', start_date='20230101', end_date='20230131')
print(f"First fetch of daily data (000001.SZ): {len(daily_df)} rows")
print(daily_df.head())

# Second fetch for the same data - should load from cache (incremental update)
print("\n--- Second fetch of daily stock data (should be from cache) ---")
daily_df_cached = client.get_data('daily', ts_code='000001.SZ', start_date='20230101', end_date='20230131')
print(f"Second fetch of daily data (000001.SZ): {len(daily_df_cached)} rows (from cache)")

# --- Example 2: Fetching stock basic information (Full Update Policy with TTL) ---
print("\n--- Fetching stock basic information ---")
stock_basic_df = client.get_data('stock_basic', exchange='SSE', list_status='L')
print(f"First fetch of stock_basic data: {len(stock_basic_df)} rows")
print(stock_basic_df.head())

# Second fetch for stock_basic - should load from cache if within TTL
print("\n--- Second fetch of stock basic information (should be from cache) ---")
stock_basic_df_cached = client.get_data('stock_basic', exchange='SSE', list_status='L')
print(f"Second fetch of stock_basic data: {len(stock_basic_df_cached)} rows (from cache)")

# Close the database connection when done
client.close()
print("\nDatabase connection closed.")
```

## Contributing

Contributions are welcome! Please refer to the `CONTRIBUTING.md` for guidelines.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.