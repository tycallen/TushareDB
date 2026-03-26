# Tushare-DuckDB：本地化金融数据缓存方案

一个基于 Python 的工具库，利用 [DuckDB](https://duckdb.org/) 为 [Tushare Pro](https://tushare.pro/) 金融数据提供高效、本地化的缓存层。同时集成 [jquant_data_sync](https://github.com/tycallen/jquant_data_sync) A股概念板块数据，为量化分析提供完整的数据支持。

## 核心特性

- **双层架构**：
  - `DataDownloader`：数据下载（写操作）
  - `DataReader`：数据查询（读操作，零网络请求）
- **智能缓存**：自动缓存 Tushare API 数据到本地 DuckDB，毫秒级查询响应
- **PIT 支持**：申万行业成分、概念板块等数据支持 Point-in-Time 查询，避免未来函数
- **概念板块**：集成聚宽概念板块数据（GitHub 每日自动更新）
- **自动建表**：首次写入自动创建表结构，无需预定义
- **复权计算**：内置前复权/后复权价格计算

## 安装

```bash
git clone https://github.com/your-repo/Tushare-DuckDB.git
cd Tushare-DuckDB
pip install -e .
```

## 配置

复制环境变量模板并填入 Tushare Token：

```bash
cp .env.example .env
# 编辑 .env，填入你的 TushARE_TOKEN
```

## 快速开始

### 1. 查询数据（DataReader - 零网络请求）

```python
from tushare_db import DataReader

reader = DataReader(db_path="tushare.db")

# 股票日线（支持复权）
df = reader.get_stock_daily('000001.SZ', '20240101', '20241231', adj='qfq')

# 申万行业日线
df = reader.get_sw_daily(ts_code='801010.SI', start_date='20240101')

# 概念板块成分股（来自 jquant_data_sync）
df = reader.get_concept_stocks('20240115', concept_name='人工智能')

# 某股票的概念列表
df = reader.get_stock_concepts('20240115', ts_code='000001.SZ')

reader.close()
```

### 2. 下载数据（DataDownloader）

```python
from tushare_db import DataDownloader

downloader = DataDownloader(db_path="tushare.db")

# 基础数据
downloader.download_trade_calendar()
downloader.download_stock_basic()

# 批量下载所有股票日线
downloader.download_all_stocks_daily(start_date='20200101')

# 单日增量更新
downloader.download_daily_data_by_date('20241218')

downloader.close()
```

## 核心概念

### DataReader vs DataDownloader

| 特性 | DataReader | DataDownloader |
|------|-----------|----------------|
| 职责 | 只读查询 | 只写下载 |
| 网络请求 | ❌ 无 | ✅ 有 |
| 并发安全 | ✅ 只读模式支持多进程 | ❌ 独占写入 |
| 适用场景 | 回测、API 服务、分析 | 初始化、定时更新 |

### Point-in-Time (PIT) 查询

对于行业成分、概念板块等数据，始终使用 PIT 查询避免未来函数：

```python
from tushare_db import DataReader

reader = DataReader(db_path="tushare.db")

# ❌ 错误：当前快照（引入未来函数）
df = reader.get_index_member_all(l1_code='801010.SI', is_new='Y')

# ✅ 正确：PIT 查询（历史某日的真实成分）
df = reader.get_index_member_all(
    l1_code='801010.SI',
    trade_date='20230115'  # 查询2023年1月15日的成分
)

# ✅ 概念板块 PIT 查询
df = reader.get_concept_stocks('20230115', concept_name='人工智能')
```

## 功能模块

### 股票数据

```python
# 日线行情（支持复权）
df = reader.get_stock_daily('000001.SZ', '20240101', adj='qfq')

# 批量查询
df = reader.get_multiple_stocks_daily(['000001.SZ', '600519.SH'], '20240101')

# 每日指标（PE/PB/市值）
df = reader.get_daily_basic('000001.SZ', '20240101')

# 资金流向
df = reader.get_moneyflow(ts_code='000001.SZ', start_date='20240101')
```

### 指数与行业

```python
# 申万行业日线
df = reader.get_sw_daily(ts_code='801010.SI')

# 申万行业成分（PIT）
df = reader.get_index_member_all(l1_code='801010.SI', trade_date='20240115')

# 指数成分权重（月度）
df = reader.get_index_weight(index_code='000300.SH', trade_date='20240131')
```

### 概念板块（jquant_data_sync）

```python
# 获取某天某概念的成分股
df = reader.get_concept_stocks('20240115', concept_name='人工智能')

# 获取某股票在某天的所有概念
df = reader.get_stock_concepts('20240115', ts_code='000001.SZ')

# 获取某天完整截面数据
df = reader.get_concept_cross_section('20240115')

# 搜索概念板块
df = reader.search_concepts('芯片')

# 获取所有概念列表
df = reader.get_all_concepts()

# 强制刷新数据（重新下载）
reader.refresh_concept_data()
```

概念板块数据特点：
- **来源**：[jquant_data_sync](https://github.com/tycallen/jquant_data_sync) GitHub Release
- **更新**：每日自动从 GitHub 拉取（带本地缓存）
- **格式**：SCD (缓慢变化维度) 区间生效结构
- **缓存**：`.concept_cache/all_concepts_pit_scd.csv`

### 财务数据

```python
# 财务指标（季度）
df = reader.query("""
    SELECT * FROM fina_indicator_vip
    WHERE ts_code = '000001.SZ' AND end_date = '20231231'
""")

# 利润表
df = reader.query("""
    SELECT * FROM income
    WHERE ts_code = '000001.SZ' AND end_date = '20231231'
""")
```

### 基金数据

```python
# 基金列表
df = reader.get_fund_basic()

# 场内基金日线
df = reader.get_fund_daily(ts_code='510050.SH')

# 基金净值
df = reader.get_fund_nav(ts_code='110022.OF')
```

## 项目结构

```
.
├── scripts/                    # 数据初始化与更新脚本
│   ├── init_data.py           # 全量数据初始化
│   ├── update_daily.py        # 每日增量更新
│   └── backfill_index_member_pit.py  # PIT 数据回填
├── src/tushare_db/
│   ├── reader.py              # DataReader：数据查询
│   ├── downloader.py          # DataDownloader：数据下载
│   ├── concept_manager.py     # 概念板块数据管理
│   ├── duckdb_manager.py      # DuckDB 操作封装
│   └── tushare_fetcher.py     # Tushare API 封装
├── tests/                     # 测试用例
└── docs/
│   ├── CONCEPT_MANAGER.md     # 概念板块使用指南
│   └── skills/                # Claude Code Skills
```

## Claude Code Integration

本项目提供 Claude Code Skill，方便在 AI 助手对话中使用：

```bash
# 全局安装（所有项目可用）
cp -r docs/skills/tushare-duckdb ~/.claude/skills/

# 或项目本地安装
mkdir -p .claude/skills
cp -r docs/skills/tushare-duckdb .claude/skills/
```

安装后在 Claude Code 中使用：
```
/tushare-duckdb          # 查看可用 API
/tushare-finance daily   # 查看 Tushare API 参数详情
```

## 复权因子说明

正确处理复权价格是量化分析的关键。本库自动处理复权计算：

```python
# DataReader 自动处理复权
df = reader.get_stock_daily('000001.SZ', '20240101', adj='qfq')  # 前复权
df = reader.get_stock_daily('000001.SZ', '20240101', adj='hfq')  # 后复权
```

复权原理：
- **前复权 (qfq)**：`price × (factor / latest_factor)` — 最新价格等于市场价
- **后复权 (hfq)**：`price × factor` — 历史价格固定

## 贡献

欢迎提交 Pull Request 或创建 Issue。

## 许可证

MIT License
