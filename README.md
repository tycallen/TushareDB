# Tushare-DuckDB

A Python library for efficiently fetching and managing Tushare financial data with a local DuckDB cache.

## Overview

Tushare-DuckDB provides a robust and easy-to-use interface to interact with the Tushare Pro API, intelligently caching data in a local DuckDB database. This allows for faster subsequent data retrieval, reduces API calls, and simplifies local financial data analysis.

## Features

*   **统一接口**: 单一的 `TushareDBClient` 用于获取各种 Tushare API 数据。
*   **透明缓存**: 自动将获取的数据存储在本地 DuckDB 文件中。
*   **智能更新策略**: 支持增量更新（针对日线数据等时间序列数据）和带可配置 TTL（Time-To-Live）的完全刷新（针对股票基本信息等静态数据）。
*   **API 频率限制**: 内置线程安全机制，严格遵守 Tushare Pro API 调用频率限制。
*   **高效本地存储**: 利用 DuckDB 在本地数据上进行高性能分析查询。

## Installation

目前，您可以克隆此仓库并在本地安装：

```bash
git clone https://github.com/your-repo/Tushare-DuckDB.git # 请替换为实际的仓库 URL
cd Tushare-DuckDB
pip install -e .
```

## Configuration

Tushare-DuckDB 需要您的 Tushare Pro API Token。您可以通过两种方式提供：

1.  **环境变量 (推荐)**: 设置 `TUSHARE_TOKEN` 环境变量。

    ```bash
    export TUSHARE_TOKEN="YOUR_TUSHARE_PRO_TOKEN"
    ```

2.  **直接在代码中**: 将 Token 直接传递给 `TushareDBClient` 构造函数。

    ```python
    from tushare_db import TushareDBClient
    client = TushareDBClient(tushare_token="YOUR_TUSHARE_PRO_TOKEN")
    ```

## Quick Start

以下是一个快速示例，演示如何使用 `TushareDBClient` 进行数据获取和缓存。

```python
import os
from tushare_db import TushareDBClient, ProBarAsset, ProBarAdj, ProBarFreq

# 确保您的 TUSHARE_TOKEN 环境变量已设置
# export TUSHARE_TOKEN="YOUR_TUSHARE_PRO_TOKEN"

# 初始化客户端
# 数据库文件将创建在当前目录下的 'tushare.db'
# 您可以指定不同的路径: db_path='data/my_tushare.db'
client = TushareDBClient()

# --- 示例 1: 获取股票日线数据 (增量更新策略) ---
print("\n--- 获取股票日线数据 (ts_code='000001.SZ') ---")
daily_df = client.get_data('daily', ts_code='000001.SZ', start_date='20230101', end_date='20230131')
print(f"首次获取日线数据 (000001.SZ): {len(daily_df)} 行")
print(daily_df.head())

# 再次获取相���数据 - 应从缓存加载 (增量更新)
print("\n--- 再次获取股票日线数据 (应从缓存加载) ---")
daily_df_cached = client.get_data('daily', ts_code='000001.SZ', start_date='20230101', end_date='20230131')
print(f"再次获取日线数据 (000001.SZ): {len(daily_df_cached)} 行 (来自缓存)")

# --- 示例 2: 获取股票基本信息 (带 TTL 的完全更新策略) ---
print("\n--- 获取股票基本信息 ---")
stock_basic_df = client.get_data('stock_basic', exchange='SSE', list_status='L')
print(f"首次获取 stock_basic 数据: {len(stock_basic_df)} 行")
print(stock_basic_df.head())

# 再次获取 stock_basic - 如果在 TTL 内，应从缓存加载
print("\n--- 再次获取股票基本信息 (应从缓存加载) ---")
stock_basic_df_cached = client.get_data('stock_basic', exchange='SSE', list_status='L')
print(f"再次获取 stock_basic 数据: {len(stock_basic_df_cached)} 行 (来自缓存)")

# --- 示例 3: 使用 pro_bar 接口获取前复权行情 ---
print("\n--- 使用 pro_bar 接口获取前复权行情 (000001.SZ) ---")
pro_bar_qfq_df = client.get_data(
    'pro_bar',
    ts_code='000001.SZ',
    start_date='20230101',
    end_date='20230131',
    adj=ProBarAdj.QFQ, # 前复权
    freq=ProBarFreq.DAILY, # 日线数据
    asset=ProBarAsset.STOCK # 股票资产
)
print(f"获��� 000001.SZ 前复权行情: {len(pro_bar_qfq_df)} 行")
print(pro_bar_qfq_df.head())

# --- 示例 4: 使用 pro_bar 接口获取上证指数行情 ---
print("\n--- 使用 pro_bar 接口获取上证指数行情 (000001.SH) ---")
pro_bar_index_df = client.get_data(
    'pro_bar',
    ts_code='000001.SH',
    start_date='20230101',
    end_date='20230131',
    asset=ProBarAsset.INDEX # 指数资产
)
print(f"获取 000001.SH 指数行情: {len(pro_bar_index_df)} 行")
print(pro_bar_index_df.head())

# 关闭数据库连接
client.close()
print("\n数据库连接已关闭。")

## Contributing

欢迎贡献！请参考 `CONTRIBUTING.md` 获取贡献指南。

## License

本项目采用 MIT 许可证 - 详情请参阅 `LICENSE` 文件。
