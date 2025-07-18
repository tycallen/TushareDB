# Tushare-DuckDB：本地化 Tushare 数据缓存方案

一个基于 Python 的工具库，旨在利用 [DuckDB](https://duckdb.org/) 为 [Tushare Pro](https://tushare.pro/) 金融数据提供一个高效、本地化的缓存层。本库可以显著减少重复的网络请求，加快数据获取速度，并为本地量化分析提供便利。

## 核心功能

- **智能缓存**: 自动将从 Tushare API 获取的数据缓存到本地 DuckDB 数据库中，后续请求优先从本地读取，大幅提升数据访问效率。
- **自动建表**: 无需预先定义数据库表结构。在首次写入数据时，系统会自动根据数据内容创建相应的表。
- **`pro_bar` 接口优化**: 针对 `pro_bar` 接口获取多只股票行情的需求进行了特殊优化。当传入多只股票代码时，会自动拆分为单个请求循环获取，并将结果合并，对上层调用完全透明。
- **智能查询参数**: 在构建数据库查询语句时，会自动过滤掉非数据表列的 API 参数（如 `fields`），避免了因参数误用导致的数据库查询错误。
- **易用的 API**: 提供了简洁的 `Client` 和一系列封装好的 API ���数，让数据获取和缓存管理变得简单直观。
- **抑制第三方库警告**: 自动屏蔽 Tushare 库内部因依赖旧版 `pandas` 功能而产生的 `FutureWarning`，保持控制台输出整洁。

## 安装

1.  克隆本项目到本地：
    ```bash
    git clone https://github.com/your-repo/Tushare-DuckDB.git # 请替换为您的仓库 URL
    cd Tushare-DuckDB
    ```

2.  使用 pip 进行安装。推荐使用可编辑模式（`-e`），便于后续开发和调试：
    ```bash
    pip install -e .
    ```

## 配置

项目需要您的 Tushare Pro API 令牌才能正常工作。

1.  将配置文件模板 `.env.example` 复制为 `.env`：
    ```bash
    cp .env.example .env
    ```

2.  编辑 `.env` 文件，填入您的个人 Tushare Token：
    ```
    TUSHARE_TOKEN='your_actual_tushare_token_here'
    ```
    程序启动时会自动加载此文件中的配置。

## 使用说明

您可以通过两种主要方式来使用本工具。

### 1. 运行初始化脚本

对于首次使用或需要大批量初始化数据的场景，可以直接运行 `scripts` 目录下的脚本。

- **初始化基础数据**:
  ```bash
  python scripts/init_data.py
  ```
  该脚本会执行一些预设的任务，例如下载所有股票的基本信息、交易日历以及历史行情数据。您可���根据需要自行修改此脚本。

- **每日数据更新**:
  ```bash
  python scripts/update_daily.py
  ```
  该脚本用于获取最近一个交易日的日线行情等增量数据。建议配置定时任务（如 `cron`）来自动执行。

### 2. 作为库在代码中调用

您可以方便地在自己的 Python 代码中引入本库，进行更灵活的数据操作。

```python
from tushare_db import TushareDBClient, api
from datetime import datetime

# 1. 初始化客户端
# 默认会在项目根目录下创建或连接 tushare.db 数据库文件
client = TushareDBClient()

# 2. 使用封装好的 API 函数获取数据
# 优先从本地数据库读取，如果数据不存在或不完整，则通过 Tushare API 获取并存入本地数据库
try:
    # 获取交易日历
    trade_cal_df = api.trade_cal(client, start_date='20240101', end_date='20240715')
    print("交易日历:")
    print(trade_cal_df.head())

    # 获取单只股票的前复权日线行情
    # pro_bar 接口经过优化，即使一次传入多个 ts_code 也能正常工作
    pro_bar_df = api.pro_bar(
        client,
        ts_code='000001.SZ',
        adj='qfq',
        start_date='20240701',
        end_date=datetime.now().strftime('%Y%m%d')
    )
    print("\n平安银行前复权行情:")
    print(pro_bar_df.head())

    # 3. 使用通用 fetch 方法获取数据
    # 对于 api.py 中未封装的接口，可以使用此方法
    stock_basic_df = client.fetch(api_name='stock_basic', list_status='L', fields='ts_code,name,industry')
    print("\n部分上市股票列表:")
    print(stock_basic_df.head())

finally:
    # 4. 关闭数据库连接
    client.close()
    print("\n数据库连接已关闭。")

```

## 项目结构

```
.
├───scripts/              # 存放可直接运行的数据初始化、更新脚本
│   ├───init_data.py      # 全量数据初始化脚本
│   └───update_daily.py   # 每日增量更新脚本
├───src/
│   └───tushare_db/       # 核心库代码
│       ├───api.py        # 封装的 Tushare 常用接口
│       ├───client.py     # 核心客户端，负责调度和数据获取
│       ├───duckdb_manager.py # DuckDB 数据库操作封装
│       ├───cache_policies.py # 缓存策略逻辑
│       └───tushare_fetcher.py # Tushare API 数据拉取逻辑
└───tests/                # 测试用例
```

## 贡献

欢迎对本项目进行贡献。如果您有任何改进建议或发现了 Bug，请随时提交 Pull Request 或创建 Issue。

## 许可证

本项目采用 MIT 许可证。详情请参阅 `LICENSE` 文件。