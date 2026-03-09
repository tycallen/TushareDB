# 代码质量问题修复实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 修复代码审查中发现的所有质量问题，包括废弃 API 引用、重复导入、SQL 注入风险和日志问题。

**Architecture:** 保持现有架构不变，仅修复具体代码缺陷。所有修复都应保持向后兼容，不改变公共 API 签名。

**Tech Stack:** Python 3.8+, DuckDB, Tushare, pytest

---

## 任务概览

| 优先级 | 任务 | 文件 | 估计时间 |
|--------|------|------|----------|
| Critical | 修复废弃 API 引用 | `scripts/init_data.py` | 15 min |
| Important | 修复重复导入 | `src/tushare_db/tushare_fetcher.py` | 2 min |
| Important | 修复 SQL 注入风险 | `src/tushare_db/duckdb_manager.py` | 5 min |
| Important | 修复日志级别 | `src/tushare_db/duckdb_manager.py` | 2 min |
| Medium | 脱敏 Token 日志 | `src/tushare_db/tushare_fetcher.py` | 5 min |

---

## Task 1: 修复 `init_data.py` 废弃 API 引用（Critical）

**问题：** `scripts/init_data.py` 引用了不存在的 `client` 变量和 `tushare_db.api` 模块。

**Files:**
- Modify: `scripts/init_data.py:71-172`

**Step 1: 分析当前问题代码**

读取文件确认问题范围：

```bash
grep -n "tushare_db.api\|client" scripts/init_data.py | head -30
```

Expected output: 多行显示使用了 `tushare_db.api.xxx(client, ...)` 模式

**Step 2: 修复 `init_fina_indicator_vip` 函数**

修改代码，使用 `DataReader` 替代废弃的 API：

```python
def init_fina_indicator_vip():
    """
    初始化所有股票的财务指标数据（VIP接口）
    从2000年至今，按季度获取。
    """
    from tushare_db import DataReader
    reader = DataReader()

    print("开始初始化所有股票的财务指标数据...")
    current_year = datetime.now().year
    for year in range(current_year + 1, 1990, -1):
        for quarter in ['0331', '0630', '0930', '1231']:
            period = f"{year}{quarter}"
            # 如果计算出的报告期在未来，则跳过
            if datetime.strptime(period, '%Y%m%d') > datetime.now():
                continue

            print(f"正在获取 {period} 的财务指标数据...")
            try:
                # 使用 DataReader 查询本地数据
                df = reader.db.execute_query(
                    "SELECT * FROM fina_indicator_vip WHERE end_date = ?",
                    [period]
                )
                if df.empty:
                    print(f"  本地无 {period} 数据，请使用 DataDownloader 下载")
                else:
                    print(f"  获取到 {len(df)} 条记录")
                    print(df.head())
            except Exception as e:
                print(f"获取 {period} 财务指标数据时出错: {e}")
    reader.close()
    print("财务指标数据初始化完成。")
```

**Step 3: 修复 `init_index_basic` 函数**

```python
def init_index_basic():
    """初始化所有指数的基本信息"""
    print("开始初始化所有指数的基本信息...")
    markets = ['MSCI', 'CSI', 'SSE', 'SZSE', 'CICC', 'SW', 'OTH']
    for market in markets:
        print(f"正在获取 {market} 的指数基本信息...")
        try:
            # 使用 DataReader 查询
            from tushare_db import DataReader
            reader = DataReader()
            df = reader.get_index_basic(market=market)
            print(f"  获取到 {len(df)} 条记录")
            print(df.head())
            reader.close()
        except Exception as e:
            print(f"获取 {market} 指数基本信息时出错: {e}")
    print("所有指数的基本信息初始化完成。")
```

**Step 4: 修复 `init_index_weight` 函数**

```python
def init_index_weight():
    """初始化主要指数的权重"""
    print("开始初始化主要指数的权重...")
    today = datetime.today()
    # 常见指数列表
    common_indices = [
        '000001.SH',  # 上证指数
        '000300.SH',  # 沪深300
        '000016.SH',  # 上证50
        '399001.SZ',  # 深圳成指
        '000905.SH',  # 中证500
        '399006.SZ',  # 创业板指
        '000852.SH',  # 中证1000
        '399303.SZ',  # 国证2000
        '000688.SH',  # 科创50
        '399102.SZ',  # 创业板综合
        '399005.SZ',  # 中小板指数
        '399101.SZ',  # 中小板综合
    ]
    MONTHs = 144

    from tushare_db import DataReader
    reader = DataReader()

    for index_code in common_indices:
        print(f"正在获取指数 {index_code} 的权重...")
        # 获取过去144个月的数据
        for i in range(MONTHs):
            target_date = today - timedelta(days=(MONTHs - i) * 30)
            year = target_date.year
            month = target_date.month
            print(f"  - 获取 {year}年{month}月 的数据...")
            try:
                df = reader.get_index_weight(index_code=index_code, year=year, month=month)
                print(f"    获取到 {len(df)} 条记录")
                print(df.head())
            except Exception as e:
                print(f"获取 {index_code} 在 {year}-{month} 的权重数据时出错: {e}")

    reader.close()
    print("主要指数的权重初始化完成。")
```

**Step 5: 修复 `init_daily_basic` 函数**

```python
def init_daily_basic():
    """初始化所有股票的每日基本面指标"""
    print("开始初始化所有股票的每日基本面指标...")

    from tushare_db import DataReader
    reader = DataReader()

    # 获取所有A股上市公司列表
    all_stocks = reader.get_stock_basic(list_status='L')
    # 合并所有市场
    for status in ['D', 'P']:
        try:
            df = reader.get_stock_basic(list_status=status)
            all_stocks = pd.concat([all_stocks, df])
        except:
            pass

    for _, stock in all_stocks.iterrows():
        ts_code = stock['ts_code']
        list_date = stock['list_date']
        try:
            df = reader.get_daily_basic(ts_code=ts_code, start_date=list_date, end_date=today)
            print(f"获取 {ts_code} 每日基本面: {len(df)} 条记录")
            if not df.empty:
                print(df.head())
        except Exception as e:
            print(f"获取 {ts_code} 数据时出错: {e}")

    reader.close()
    print("所有股票的每日基本面指标初始化完成。")
```

**Step 6: 修复 `init_adj_factor_data` 函数**

```python
def init_adj_factor_data():
    """初始化所有股票的历史复权因子数据"""
    print("开始初始化所有股票的历史复权因子数据...")

    from tushare_db import DataReader
    reader = DataReader()

    # 获取所有股票代码
    all_stocks = reader.get_stock_basic(list_status='L')
    for status in ['D', 'P']:
        try:
            df = reader.get_stock_basic(list_status=status)
            all_stocks = pd.concat([all_stocks, df])
        except:
            pass

    ts_codes = all_stocks["ts_code"].unique().tolist()

    for ts_code in tqdm(ts_codes, desc="正在初始化复权因子"):
        try:
            df = reader.get_adj_factor(ts_code=ts_code, start_date='20000101', end_date=today)
            print(f"获取 {ts_code} 复权因子数据: {len(df)} 条记录")
        except Exception as e:
            print(f"获取 {ts_code} 复权因子数据时出错: {e}")

    reader.close()
    print("所有股票的历史复权因子数据初始化完成。")
```

**Step 7: 修复 `init_cyq_chips` 函数**

```python
def init_cyq_chips():
    """初始化所有股票的历史筹码分布数据"""
    print("开始初始化所有股票的历史筹码分布数据...")

    from tushare_db import DataReader
    reader = DataReader()

    # 获取所有股票代码
    all_stocks = reader.get_stock_basic(list_status='L')
    for status in ['D', 'P']:
        try:
            df = reader.get_stock_basic(list_status=status)
            all_stocks = pd.concat([all_stocks, df])
        except:
            pass

    ts_codes = all_stocks["ts_code"].unique().tolist()

    for ts_code in tqdm(ts_codes, desc="正在初始化筹码分布"):
        try:
            df = reader.get_cyq_chips(ts_code=ts_code, start_date='20000101', end_date=today)
            print(f"获取 {ts_code} 筹码分布数据: {len(df)} 条记录")
        except Exception as e:
            print(f"获取 {ts_code} 筹码分布数据时出错: {e}")

    reader.close()
    print("所有股票的历史筹码分布数据初始化完成。")
```

**Step 8: 修复资金流向函数**

```python
def init_moneyflow_cnt_ths(start_date: str, end_date: str):
    """
    初始化同花顺概念板块资金流向数据。
    按天循环获取指定日期范围内的数据。
    """
    print(f"开始初始化同花顺概念板块资金流向数据，从 {start_date} 到 {end_date}...")

    from tushare_db import DataReader
    reader = DataReader()

    # 1. 获取指定范围内的所有交易日
    try:
        trade_cal_df = reader.get_trade_calendar(start_date=start_date, end_date=end_date, is_open='1')
        trade_dates = trade_cal_df['cal_date'].tolist()
        if not trade_dates:
            print("指定日期范围内没有交易日，任务结束。")
            reader.close()
            return
    except Exception as e:
        print(f"获取交易日历失败: {e}")
        reader.close()
        return

    # 2. 遍历每个交易日，查询本地数据
    for trade_date in tqdm(trade_dates, desc="正在查询同花顺概念资金流向"):
        try:
            df = reader.db.execute_query(
                "SELECT * FROM moneyflow_cnt_ths WHERE trade_date = ?",
                [trade_date]
            )
            print(f"查询 {trade_date} 的本地数据: {len(df)} 条记录")
        except Exception as e:
            print(f"查询 {trade_date} 数据时出错: {e}")

    reader.close()
    print("同花顺概念板块资金流向数据查询完成。")
```

```python
def init_moneyflow_ind_dc(start_date: str, end_date: str):
    """
    初始化东方财富概念及行业板块资金流向数据。
    """
    print(f"开始查询东方财富概念及行业板块资金流向数据，从 {start_date} 到 {end_date}...")

    from tushare_db import DataReader
    reader = DataReader()

    # 获取交易日
    try:
        trade_cal_df = reader.get_trade_calendar(start_date=start_date, end_date=end_date, is_open='1')
        trade_dates = trade_cal_df['cal_date'].tolist()
        if not trade_dates:
            print("指定日期范围内没有交易日，任务结束。")
            reader.close()
            return
    except Exception as e:
        print(f"获取交易日历失败: {e}")
        reader.close()
        return

    # 查询本地数据
    for trade_date in tqdm(trade_dates, desc="正在查询东方财富概念资金流向"):
        try:
            df = reader.db.execute_query(
                "SELECT * FROM moneyflow_ind_dc WHERE trade_date = ?",
                [trade_date]
            )
            print(f"查询 {trade_date} 的本地数据: {len(df)} 条记录")
        except Exception as e:
            print(f"查询 {trade_date} 数据时出错: {e}")

    reader.close()
    print("东方财富概念及行业板块资金流向数据查询完成。")
```

**Step 9: 验证修复**

```bash
python -c "import scripts.init_data; print('Import successful')"
```

Expected: 成功导入，无错误

**Step 10: Commit**

```bash
git add scripts/init_data.py
git commit -m "fix: replace deprecated API calls with DataReader in init_data.py

- Replace tushare_db.api.*(client, ...) with DataReader methods
- Add proper DataReader initialization and cleanup
- Fix undefined 'client' variable references

Fixes Critical issue from code review"
```

---

## Task 2: 修复重复导入（Important）

**问题：** `tushare_fetcher.py` 第 5 行和第 9 行重复导入 `time` 模块。

**Files:**
- Modify: `src/tushare_db/tushare_fetcher.py:5-9`

**Step 1: 读取当前导入语句**

```bash
head -15 src/tushare_db/tushare_fetcher.py
```

**Step 2: 删除重复导入**

删除第 9 行的 `import time`。

修改前：
```python
import time
import threading
import collections
import logging
import time  # 重复
```

修改后：
```python
import time
import threading
import collections
import logging
```

**Step 3: 验证修复**

```bash
python -c "from src.tushare_db.tushare_fetcher import TushareFetcher; print('Import successful')"
```

**Step 4: Commit**

```bash
git add src/tushare_db/tushare_fetcher.py
git commit -m "fix: remove duplicate 'import time' statement

Removes redundant import on line 9.

Fixes Important issue from code review"
```

---

## Task 3: 修复 SQL 注入风险（Important）

**问题：** `duckdb_manager.py` 第 436 行直接拼接 SQL 字符串。

**Files:**
- Modify: `src/tushare_db/duckdb_manager.py:432-441`

**Step 1: 查看当前代码**

```bash
sed -n '432,441p' src/tushare_db/duckdb_manager.py
```

**Step 2: 修改为参数化查询**

修改 `get_cache_metadata` 方法：

```python
def get_cache_metadata(self, table_name: str) -> Optional[float]:
    """Retrieves the last updated timestamp for a given table."""
    try:
        with self._lock:
            # 使用参数化查询防止 SQL 注入
            result = self.con.execute(
                "SELECT last_updated_timestamp FROM _tushare_cache_metadata WHERE table_name = ?",
                [table_name]
            ).fetchone()
        return float(result[0]) if result and result[0] is not None else None
    except Exception as e:
        logging.error(f"Error getting cache metadata for {table_name}: {e}")
        raise DuckDBManagerError(f"Failed to get cache metadata: {e}") from e
```

**Step 3: 验证修复**

```bash
python -c "
from src.tushare_db.duckdb_manager import DuckDBManager
# 创建测试数据库
import tempfile
import os
with tempfile.TemporaryDirectory() as tmpdir:
    db_path = os.path.join(tmpdir, 'test.db')
    db = DuckDBManager(db_path)
    # 测试方法存在且工作
    result = db.get_cache_metadata('test_table')
    print(f'get_cache_metadata works: returned {result}')
    db.close()
"
```

**Step 4: Commit**

```bash
git add src/tushare_db/duckdb_manager.py
git commit -m "fix: use parameterized query in get_cache_metadata

Replace f-string SQL construction with proper parameterized query
to prevent potential SQL injection.

Fixes Important issue from code review"
```

---

## Task 4: 修复日志级别错误（Important）

**问题：** `duckdb_manager.py` 第 236 行在创建表时使用 `logging.error` 而不是 `logging.info`。

**Files:**
- Modify: `src/tushare_db/duckdb_manager.py:236`

**Step 1: 查看当前代码**

```bash
sed -n '230,240p' src/tushare_db/duckdb_manager.py
```

**Step 2: 修复日志级别**

修改第 236 行：

```python
# 修改前
logging.error(f"Executing create SQL: {create_sql}")

# 修改后
logging.info(f"Executing create SQL: {create_sql}")
```

**Step 3: Commit**

```bash
git add src/tushare_db/duckdb_manager.py
git commit -m "fix: correct log level from error to info

Creating a table is not an error condition, changed from
logging.error to logging.info.

Fixes Important issue from code review"
```

---

## Task 5: 脱敏 Token 日志（Medium）

**问题：** `tushare_fetcher.py` 第 47 行将 Token 输出到日志。

**Files:**
- Modify: `src/tushare_db/tushare_fetcher.py:45-48`

**Step 1: 查看当前代码**

```bash
sed -n '40,50p' src/tushare_db/tushare_fetcher.py
```

**Step 2: 脱敏 Token**

修改日志输出，只显示 Token 的部分信息：

```python
# 脱敏 Token 用于日志显示
masked_token = self.token[:4] + "****" + self.token[-4:] if len(self.token) > 8 else "****"
logging.info(f"User with token {masked_token}, credit info: {df}")
```

**Step 3: 验证修复**

```bash
python -c "
token = '1234567890abcdef'
masked = token[:4] + '****' + token[-4:] if len(token) > 8 else '****'
print(f'Masked token: {masked}')
assert masked == '1234****cdef'
print('Token masking works correctly')
"
```

**Step 4: Commit**

```bash
git add src/tushare_db/tushare_fetcher.py
git commit -m "security: mask token in log output

Replace full token with masked version (first 4 + **** + last 4 chars)
to prevent sensitive credential leakage in logs.

Fixes Medium issue from code review"
```

---

## Task 6: 添加核心模块测试（Follow-up）

**Files:**
- Create: `tests/test_duckdb_manager.py`

**Step 1: 创建基础测试**

```python
import pytest
import tempfile
import os
import pandas as pd
from tushare_db.duckdb_manager import DuckDBManager, DuckDBManagerError


class TestDuckDBManager:
    """Test DuckDBManager core functionality"""

    @pytest.fixture
    def db(self):
        """Create a temporary database for testing"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            db = DuckDBManager(db_path)
            yield db
            db.close()

    def test_table_exists(self, db):
        """Test table_exists method"""
        # Create a test table
        db.con.execute("CREATE TABLE test_table (id INTEGER, name VARCHAR)")
        assert db.table_exists('test_table') is True
        assert db.table_exists('non_existent') is False

    def test_write_and_read_dataframe(self, db):
        """Test writing and reading DataFrame"""
        df = pd.DataFrame({
            'ts_code': ['000001.SZ', '000002.SZ'],
            'trade_date': ['20240101', '20240102'],
            'close': [10.5, 20.3]
        })

        db.write_dataframe(df, 'daily', mode='append')

        # Verify data was written
        result = db.execute_query("SELECT * FROM daily")
        assert len(result) == 2
        assert list(result['ts_code']) == ['000001.SZ', '000002.SZ']

    def test_get_cache_metadata_parameterized(self, db):
        """Test get_cache_metadata uses parameterized queries"""
        # This test ensures the SQL injection fix works
        db.con.execute("""
            CREATE TABLE _tushare_cache_metadata (
                table_name VARCHAR PRIMARY KEY,
                last_updated_timestamp DOUBLE
            )
        """)
        db.con.execute("""
            INSERT INTO _tushare_cache_metadata VALUES ('test_table', 12345.0)
        """)

        # Test with normal table name
        result = db.get_cache_metadata('test_table')
        assert result == 12345.0

        # Test with potentially malicious input (should not cause injection)
        result = db.get_cache_metadata("test' OR '1'='1")
        assert result is None  # Should return None, not raise error or return data


class TestRateLimiter:
    """Test TushareFetcher rate limiting"""

    def test_rate_limit_config_parsing(self):
        """Test rate limit configuration is parsed correctly"""
        from tushare_db.tushare_fetcher import TushareFetcher

        config = {
            "default": {"limit": 100, "period": "minute"},
            "daily": {"limit": 200, "period": "day"}
        }

        # Mock token for initialization
        fetcher = TushareFetcher("mock_token_for_testing", config)

        assert fetcher.rate_limit_config["default"]["limit"] == 100
        assert fetcher.rate_limit_config["default"]["period_seconds"] == 60
        assert fetcher.rate_limit_config["daily"]["period_seconds"] == 86400
```

**Step 2: 运行测试**

```bash
pytest tests/test_duckdb_manager.py -v
```

Expected: 所有测试通过

**Step 3: Commit**

```bash
git add tests/test_duckdb_manager.py
git commit -m "test: add core DuckDBManager tests

- Test table_exists functionality
- Test DataFrame write/read
- Test parameterized query safety in get_cache_metadata
- Test rate limit config parsing

Addresses test coverage gap from code review"
```

---

## 总结

完成以上所有任务后，代码审查中发现的问题将全部修复：

| 问题 | 状态 | 修复位置 |
|------|------|----------|
| 废弃 API 引用 | ✅ Task 1 | `scripts/init_data.py` |
| 重复导入 | ✅ Task 2 | `src/tushare_db/tushare_fetcher.py` |
| SQL 注入风险 | ✅ Task 3 | `src/tushare_db/duckdb_manager.py` |
| 日志级别错误 | ✅ Task 4 | `src/tushare_db/duckdb_manager.py` |
| Token 泄露 | ✅ Task 5 | `src/tushare_db/tushare_fetcher.py` |
| 测试覆盖不足 | ✅ Task 6 | `tests/test_duckdb_manager.py` |

---

**执行选择：**

1. **Subagent-Driven（当前会话）** - 我为每个任务分配新的 subagent，任务间审查
2. **Parallel Session（独立会话）** - 开启新会话使用 executing-plans skill 批量执行

请选择执行方式：
