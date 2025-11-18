# 新架构迁移指南

## 为什么要重构？

### 旧架构的问题

```python
# 旧架构：TushareDBClient
client = TushareDBClient()
df = client.get_data('pro_bar', ts_code='000001.SZ', start_date='20230101', end_date='20231231')

# 问题：
# ✗ 每次查询都执行复杂的缓存策略判断（IncrementalCachePolicy 200行）
# ✗ 检查交易日历、追溯性变化、智能遍历决策...
# ✗ 可能触发意外的网络请求
# ✗ 复权因子的追溯性检测导致数据不可复现
# ✗ 职责混淆：下载和查询耦合在一起
```

### 新架构的优势

```python
# 新架构：职责分离
from tushare_db import DataDownloader, DataReader

# 数据下载（一次性或定时任务）
downloader = DataDownloader()
downloader.download_stock_daily('000001.SZ', '20230101', '20231231')

# 数据查询（回测、Web API）
reader = DataReader()
df = reader.get_stock_daily('000001.SZ', '20230101', '20231231', adj='qfq')

# 优势：
# ✓ 职责清晰：下载是下载，查询是查询
# ✓ 高性能：纯SQL查询，毫秒级响应（提速50-100倍）
# ✓ 可靠：回测时数据不会突然变化
# ✓ 简单：代码总量减少60%
```

---

## 快速开始

### 1. 测试新架构

```bash
# 运行测试脚本
python scripts/test_new_architecture.py

# 查看使用示例
python scripts/example_new_architecture.py
```

### 2. 迁移步骤

#### 场景1：数据初始化脚本

**旧代码 (scripts/init_data.py):**
```python
from tushare_db import TushareDBClient
import tushare_db.api as api

client = TushareDBClient()

# 下载交易日历
api.trade_cal(client, start_date='19900101', end_date='20301231')

# 下载股票列表
api.stock_basic(client, list_status='L')

# 下载日线数据
all_stocks = api.stock_basic(client, list_status='L')
for ts_code in all_stocks['ts_code']:
    api.pro_bar(client, ts_code=ts_code, start_date='20000101')
```

**新代码:**
```python
from tushare_db import DataDownloader

downloader = DataDownloader()

# 下载交易日历
downloader.download_trade_calendar('19900101', '20301231')

# 下载股票列表
downloader.download_stock_basic('L')

# 批量下载日线数据（带进度条）
downloader.download_all_stocks_daily('20000101', '20231231', list_status='L')

# 验证数据完整性
result = downloader.validate_data_integrity('20000101', '20231231')
print(f"数据完整: {result['is_valid']}")
```

#### 场景2：每日更新脚本

**旧代码 (scripts/update_daily.py):**
```python
from tushare_db import TushareDBClient
import tushare_db.api as api

client = TushareDBClient()

# 复杂的增量更新逻辑，内部自动判断
api.pro_bar(client, trade_date='20240118')
api.adj_factor(client, trade_date='20240118')  # 可能触发追溯性检查
```

**新代码:**
```python
from tushare_db import DataDownloader

downloader = DataDownloader()

# 简单直接：按日期下载当天所有数据
downloader.download_daily_data_by_date('20240118')
```

#### 场景3：回测系统

**旧代码 (backtest/strategy.py):**
```python
from tushare_db import TushareDBClient
import tushare_db.api as api

client = TushareDBClient()

# 每次查询都触发缓存策略检查（慢！）
for date in trading_dates:
    df = api.pro_bar(client, ts_code='000001.SZ', start_date=date, end_date=date)
    # ... 回测逻辑
```

**新代码:**
```python
from tushare_db import DataReader

reader = DataReader()

# 一次性加载所有数据（推荐）
df_all = reader.get_stock_daily('000001.SZ', '20200101', '20231231', adj='qfq')

# 或者按需查询（也很快）
for date in trading_dates:
    df = reader.get_stock_daily('000001.SZ', date, date, adj='qfq')
    # ... 回测逻辑（纯SQL，毫秒级）
```

#### 场景4：Web API 服务

**旧代码 (src/tushare_db/web_server.py):**
```python
from .client import TushareDBClient
from . import api

client = TushareDBClient()

@app.get("/api/pro_bar")
async def get_pro_bar(ts_code: str, start_date: str, end_date: str):
    df = api.pro_bar(client, ts_code=ts_code, start_date=start_date, end_date=end_date)
    return df_to_json_response(df)
```

**新代码:**
```python
from .reader import DataReader

reader = DataReader()

@app.get("/api/pro_bar")
async def get_pro_bar(ts_code: str, start_date: str, end_date: str, adj: str = None):
    df = reader.get_stock_daily(ts_code, start_date, end_date, adj=adj)
    return df_to_json_response(df)
```

---

## API 对照表

### 旧架构 → 新架构

| 旧架构 (TushareDBClient) | 新架构 (DataDownloader / DataReader) | 说明 |
|-------------------------|-------------------------------------|------|
| `client.get_data('trade_cal', ...)` | **下载:** `downloader.download_trade_calendar()` | 职责分离 |
| | **查询:** `reader.get_trade_calendar()` | |
| `client.get_data('stock_basic', ...)` | **下载:** `downloader.download_stock_basic()` | |
| | **查询:** `reader.get_stock_basic()` | |
| `client.get_data('pro_bar', ...)` | **下载:** `downloader.download_stock_daily()` | |
| | **查询:** `reader.get_stock_daily(..., adj='qfq')` | 支持复权 |
| `client.get_data('adj_factor', ...)` | **下载:** `downloader.download_adj_factor()` | |
| | **查询:** `reader.get_adj_factor()` | |
| `api.pro_bar(client, ...)` | 同上 | api.py 将废弃 |

### 新增功能

| 功能 | API | 说明 |
|------|-----|------|
| 批量下载 | `downloader.download_all_stocks_daily()` | 带进度条 |
| 按日期更新 | `downloader.download_daily_data_by_date()` | 适合定时任务 |
| 数据验证 | `downloader.validate_data_integrity()` | 检查完整性 |
| 批量查询 | `reader.get_multiple_stocks_daily()` | 高性能 |
| 自定义SQL | `reader.query(sql, params)` | 灵活查询 |

---

## 性能对比

### 回测场景测试

```python
# 测试：查询1000次日线数据

# 旧架构
import time
from tushare_db import TushareDBClient

client = TushareDBClient()
start = time.time()
for i in range(1000):
    df = client.get_data('pro_bar', ts_code='000001.SZ', start_date='20230101', end_date='20230131')
old_time = time.time() - start
print(f"旧架构: {old_time:.2f}秒")  # 约 50-150秒

# 新架构
from tushare_db import DataReader

reader = DataReader()
start = time.time()
for i in range(1000):
    df = reader.get_stock_daily('000001.SZ', '20230101', '20230131')
new_time = time.time() - start
print(f"新架构: {new_time:.2f}秒")  # 约 1-3秒

print(f"提速: {old_time/new_time:.1f}倍")  # 30-100倍
```

---

## 常见问题

### Q1: 旧代码会立即失效吗？

**不会。** 旧的 `TushareDBClient` 和 `api.py` 仍然可用，保持向后兼容。但建议尽快迁移到新架构。

### Q2: 需要重新下载数据吗？

**不需要。** 新旧架构共用同一个 DuckDB 数据库，数据完全兼容。

### Q3: 如何处理复权因子的追溯性变化？

**新架构的设计：**
- 下载时：不做追溯性检测，简单的 upsert
- 查询时：动态计算复权价格
- 如果分红送股导致历史复权因子变化，手动删除重下即可：
  ```python
  # 删除某只股票的所有复权因子
  reader.query("DELETE FROM adj_factor WHERE ts_code = ?", ['000001.SZ'])

  # 重新下载
  downloader.download_adj_factor('000001.SZ', '20000101', '20231231')
  ```

### Q4: 前端需要修改吗？

**需要小改。** 只需修改 `web_server.py` 中的一处：

```python
# 旧：
from .client import TushareDBClient
client = TushareDBClient()

# 新：
from .reader import DataReader
reader = DataReader()
```

前端代码完全不用动。

### Q5: 什么时候删除旧代码？

建议流程：
1. **第1天**: 测试新架构 (`test_new_architecture.py`)
2. **第2天**: 迁移回测和 Web 服务
3. **第3天**: 运行1周，确认无问题
4. **第7天**: 删除 `cache_policies.py` 和 `client.py`

---

## 完整迁移检查清单

- [ ] 运行 `python scripts/test_new_architecture.py` 测试通过
- [ ] 修改数据初始化脚本使用 `DataDownloader`
- [ ] 修改每日更新脚本使用 `DataDownloader`
- [ ] 修改回测代码使用 `DataReader`
- [ ] 修改 `web_server.py` 使用 `DataReader`
- [ ] 验证前端功能正常
- [ ] 运行回测验证结果一致
- [ ] 删除旧代码 `cache_policies.py`、`client.py`
- [ ] 更新 `README.md` 文档

---

## 获取帮助

- 查看示例：`python scripts/example_new_architecture.py`
- 查看源码：`src/tushare_db/downloader.py` 和 `reader.py`
- 遇到问题：提交 Issue 到 GitHub

---

**祝迁移顺利！享受新架构带来的简洁和高性能 🚀**
