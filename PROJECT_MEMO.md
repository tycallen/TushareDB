# Tushare-DuckDB 项目备忘录 (Project Memo)

本文档为 `Tushare-DuckDB` 项目的使用者（包括开发者和AI助手）提供核心功能、使用方法和API参考。

## 1. 项目概述 (Project Overview)

`Tushare-DuckDB` 是一个Python库，它为 [Tushare Pro](https://tushare.pro/) 金融数据接口提供了一个高效、持久化的本地缓存层。通过将API数据缓存到本地的 [DuckDB](https://duckdb.org/) 数据库中，实现数据的快速查询和增量更新，从而显著提升数据获取效率，并节省API积分。

## 2. 核心特性 (Key Features)

- **透明缓存**: 自动缓存Tushare API的请求结果。后续相同的请求将直接从本地数据库读取，速度极快。
- **智能增量更新**: 对时间序列数据（如日线行情、财务指标），能自动检测本地最新数据点，并仅拉取此日期之后的新数据。
- **开箱即用**: 提供了对Tushare常用接口的直接封装，无需关心缓存策略和数据库操作。
- **数据初始化**: 提供 `scripts/init_data.py` 脚本，用于批量下载历史数据，方便快速搭建本地金融数据库。
- **数据更新**: 提供 `scripts/update_daily.py` 脚本，用于每日更新所有时间序列数据。

## 3. 使用方法 (Usage)

### a. 初始化客户端

```python
import os
import tushare_db

# 建议从环境变量获取Token
tushare_token = os.getenv("TUSHARE_TOKEN")

# 初始化客户端，指定数据库文件路径
client = tushare_db.TushareDBClient(
    tushare_token=tushare_token,
    db_path="/path/to/your/tushare.db"
)
```

### b. 调用API

直接从 `tushare_db.api` 模块导入并调用封装好的函数。

```python
from tushare_db import TushareDBClient, api

# 初始化客户端
client = TushareDBClient(tushare_token="YOUR_TOKEN", db_path="tushare.db")

# 获取某只股票的每日基本面指标
df = api.daily_basic(client, ts_code='000001.SZ', start_date='20240101')

print(df.head())

# 使用完毕后建议关闭连接
client.close()
```

## 4. API 接口参考 (API Reference)

所有API函数都接收一个 `TushareDBClient` 实例作为第一个参数，并返回一个 `pandas.DataFrame`。

---

### `stock_basic`

获取股票基础信息数据。

- **参数**:
  - `ts_code` (str, optional): TS股票代码。
  - `name` (str, optional): 股票名称 (支持模糊匹配)。
  - `market` (str, optional): 市场类别 (主板/创业板/科创板/CDR/北交所)。
  - `list_status` (str, optional): 上市状态, `L`上市 `D`退市 `P`暂停上市。默认 `L`。
  - `exchange` (str, optional): 交易所, `SSE`上交所 `SZSE`深交所 `BSE`北交所。
  - `fields` (str, optional): 需要返回的字段，逗号分隔。

- **返回字段 (部分)**: `ts_code`, `symbol`, `name`, `area`, `industry`, `market`, `list_date`, `list_status`。

---

### `trade_cal`

获取各大交易所交易日历数据。

- **参数**:
  - `exchange` (str, optional): 交易所代码, 默认 `SSE`。
  - `start_date` (str, optional): 开始日期 (YYYYMMDD)。
  - `end_date` (str, optional): 结束日期 (YYYYMMDD)。
  - `is_open` (str, optional): 是否交易, `0`休市 `1`交易。

- **返回字段**: `exchange`, `cal_date`, `is_open`, `pretrade_date`。

---

### `daily_basic`

获取每日重要的基本面指标。

- **参数**:
  - `ts_code` (str | List[str], optional): 股票代码, 单个或列表。
  - `trade_date` (str, optional): 交易日期 (YYYYMMDD)。
  - `start_date` (str, optional): 开始日期 (YYYYMMDD)。
  - `end_date` (str, optional): 结束日期 (YYYYMMDD)。

- **返回字段 (部分)**: `ts_code`, `trade_date`, `close`, `turnover_rate`, `volume_ratio`, `pe`, `pe_ttm`, `pb`, `total_share`, `total_mv`, `circ_mv`。

---

### `pro_bar`

获取复权或未复权的行情数据。

- **参数**:
  - `ts_code` (str | List[str], optional): 证券代码, 单个或列表。
  - `start_date` (str, optional): 开始日期 (YYYYMMDD)。
  - `end_date` (str, optional): 结束日期 (YYYYMMDD)。
  - `asset` (str, optional): 资产类别, `E`股票 `I`指数 `FD`基金等。默认 `E`。
  - `adj` (str, optional): 复权类型, `qfq`前复权 `hfq`后复权。默认不复权。
  - `freq` (str, optional): 数据频度, `D`日 `W`周 `M`月。默认 `D`。

- **返回字段 (部分)**: `ts_code`, `trade_date`, `open`, `high`, `low`, `close`, `pre_close`, `change`, `pct_chg`, `vol`, `amount`。

---

### `index_basic`

获取指数基础信息。

- **参数**:
  - `ts_code` (str, optional): 指数代码。
  - `market` (str, optional): 市场, 如 `SSE`, `SZSE`, `CSI` 等。
  - `publisher` (str, optional): 发布方。
  - `category` (str, optional): 指数类别。

- **返回字段 (部分)**: `ts_code`, `name`, `fullname`, `market`, `publisher`, `list_date`。

---

### `index_weight`

获取指数成分和权重（月度数据）。

- **参数**:
  - `index_code` (str, required): 指数代码。
  - `year` (int, required): 年份。
  - `month` (int, required): 月份。

- **返回字段**: `index_code`, `con_code`, `trade_date`, `weight`。

---

### `cyq_perf`

获取A股每日筹码平均成本和胜率。

- **参数**:
  - `ts_code` (str | List[str], optional): 股票代码。
  - `trade_date` (str, optional): 交易日期 (YYYYMMDD)。
  - `start_date` (str, optional): 开始日期 (YYYYMMDD)。
  - `end_date` (str, optional): 结束日期 (YYYYMMDD)。

- **返回字段 (部分)**: `ts_code`, `trade_date`, `his_low`, `his_high`, `cost_50pct`, `winner_rate`。

---

### `stk_factor_pro`

获取股票每日技术面因子数据。

- **参数**:
  - `ts_code` (str | List[str], optional): 股票代码。
  - `trade_date` (str, optional): 交易日期 (YYYYMMDD)。
  - `start_date` (str, optional): 开始日期 (YYYYMMDD)。
  - `end_date` (str, optional): 结束日期 (YYYYMMDD)。

- **返回字段 (部分)**: `ts_code`, `trade_date`, `close`, `open`, `high`, `low`, `pe`, `pb`, `ps`, `dv_ratio`, `macd_dif`, `kdj_k`, `rsi_6`, `boll_upper`。 (包含大量技术指标，详情请参考Tushare文档)。

---

### `fina_indicator_vip`

获取A股上市公司财务指标数据 (VIP接口)。

- **参数**:
  - `period` (str, required): 报告期 (YYYYMMDD), 例如 '20231231'。
  - `ann_date` (str, optional): 公告日期 (YYYYMMDD)。
  - `start_date` (str, optional): 报告期开始日期 (YYYYMMDD)。
  - `end_date` (str, optional): 报告期结束日期 (YYYYMMDD)。

- **返回字段 (部分)**: `ts_code`, `ann_date`, `end_date`, `eps`, `roe`, `pb`, `netprofit_yoy`, `assets_yoy`, `debt_to_assets`。 (包含大量财务指标，详情请参考Tushare文档)。

---
*注：更多接口及其详细参数可查阅 `src/tushare_db/api.py` 文件。*
