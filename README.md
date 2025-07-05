# Tushare-DuckDB

一个用于高效获取和管理 Tushare 金融数据，并使用本地 DuckDB 进行缓存的 Python 库。

## 概述

Tushare-DuckDB 提供了一个强大且易于使用的接口，用于与 Tushare Pro API 交互，并智能地将数据缓存在本地 DuckDB 数据库中。这使得后续数据检索更快，减少了 API 调用次数，并简化了本地金融数据分析。

## 特性

*   **统一接口**: 单一的 `TushareDBClient` 用于获取各种 Tushare API 数据。
*   **透明缓存**: 自动将获取的数据存储在本地 DuckDB 文件中。
*   **智能更新策略**: 支持增量更新（针对日线数据等时间序列数据）和带可配置 TTL（Time-To-Live）的完全刷新（针对股票基本信息等静态数据）。
*   **API 频率限制**: 内置线程安全机制，严格遵守 Tushare Pro API 调用频率限制。
*   **高效本地存储**: 利用 DuckDB 在本地数据上进行高性能分析查询。

## 安装

目前，您可以克隆此仓库并在本地安装：

```bash
git clone https://github.com/your-repo/Tushare-DuckDB.git # 请替换为实际的仓库 URL
cd Tushare-DuckDB
pip install -e .
```

## 配置

Tushare-DuckDB 需要您的 Tushare Pro API 令牌。您可以通过两种方式提供：

1.  **环境变量 (推荐)**: 设置 `TUSHARE_TOKEN` 环境变量。

    ```bash
    export TUSHARE_TOKEN="YOUR_TUSHARE_PRO_TOKEN"
    ```

2.  **直接在代码中**: 将令牌直接传递给 `TushareDBClient` 构造函数。

    ```python
    from tushare_db import TushareDBClient
    client = TushareDBClient(tushare_token="YOUR_TUSHARE_PRO_TOKEN")
    ```

## 快速入门

以下是一个快速示例，演示如何使用 `TushareDBClient` 进行数据获取和缓存。

## 使用方法

您可以通过两种主要方式从 Tushare 获取数据：

### 1. 通用 `get_data` 方法

这是最灵活的访问方式，允许您调用任何 Tushare Pro API 接口。您只需提供接口名称（例如 `'daily'`、`'stock_basic'`）和相应的参数即可。`TushareDBClient` 会自动处理缓存和数据更新。

```python
# 使用通用方法获取日线数据
daily_data = client.get_data('daily', ts_code='000001.SZ', start_date='20230101')

# 获取备用列表
备用_data = client.get_data('bak_basic', trade_date='20231120')
```

### 2. 使用预定义的 API 函数

为了方便起见，`tushare_db.api` 模块中预定义了许多常用的 Tushare 接口。这些函数提供了更好的代码提示和参数类型检查，使您的代码更具可读性和健壮性。

这些预定义的函数最终也会通过 `TushareDBClient` 来执行，因此同样享受透明缓存和智能更新带来的所有好处。

```python
from tushare_db import api

# 使用预定义的函数获取日线数据
daily_data_api = api.daily(client, ts_code='000001.SZ', start_date='20230101')

# 获取交易日历
trade_cal_data = api.trade_cal(client, exchange='SSE', start_date='20230101', end_date='20231231')
```

```python
import os
from tushare_db import TushareDBClient, ProBarAsset, ProBarAdj, ProBarFreq

# 确保您的 TUSHARE_TOKEN 环境变量已设置
# export TUSHARE_TOKEN="YOUR_TUSHARE_PRO_TOKEN"

# 初始化客户端
# 数据库文件将默认在当前目录下创建，文件名为 'tushare.db'
# 您可以指定不同的路径: db_path='data/my_tushare.db'
client = TushareDBClient()

# --- 示例 1: 获取股票日线数据 (增量更新策略) ---
print("\n--- 获取股票日线数据 (ts_code='000001.SZ') ---")
daily_df = client.get_data('daily', ts_code='000001.SZ', start_date='20230101', end_date='20230131')
print(f"首次获取日线数据 (000001.SZ): {len(daily_df)} 行")
print(daily_df.head())

# 再次获取相同数据 - 应从缓存加载 (增量更新)
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
print(f"获取 000001.SZ 前复权行情: {len(pro_bar_qfq_df)} 行")
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

## 贡献

欢迎贡献！请参考 `CONTRIBUTING.md` 获取贡献指南。

## 许可证

本项目采用 MIT 许可证 - 详情请参阅 `LICENSE` 文件。
