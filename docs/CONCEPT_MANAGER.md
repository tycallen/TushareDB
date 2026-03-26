# 概念板块数据查询指南

A股概念板块数据接入 https://github.com/tycallen/jquant_data_sync

## 统一 API（推荐）

概念板块数据已集成到 `DataReader`，使用方式与其他数据一致：

```python
from tushare_db import DataReader

reader = DataReader(db_path="tushare.db")

# 1. 获取某天某概念板块的所有股票
stocks = reader.get_concept_stocks('20240115', concept_name='人工智能')
print(stocks['ts_code'].tolist())

# 2. 获取某天某股票的所有概念
concepts = reader.get_stock_concepts('20240115', ts_code='000001.SZ')
print(concepts['concept_name'].tolist())

# 3. 获取某天完整截面数据（所有概念-股票关系）
df = reader.get_concept_cross_section('20240115')

# 4. 获取所有概念板块列表
all_concepts = reader.get_all_concepts()

# 5. 搜索概念板块
chips = reader.search_concepts("芯片")
```

## DataReader 概念数据 API 详解

### get_concept_stocks(trade_date, concept_name=None, concept_code=None)
获取指定日期某概念板块的所有成分股。

```python
# 通过概念名称查询
stocks = reader.get_concept_stocks('20240115', concept_name='人工智能')

# 通过概念代码查询
stocks = reader.get_concept_stocks('20240115', concept_code='GN036')

# 返回：DataFrame，列: ts_code, concept_name, in_date, out_date
```

### get_stock_concepts(trade_date, ts_code)
获取指定日期某股票所属的所有概念板块。

```python
concepts = reader.get_stock_concepts('20240115', ts_code='000001.SZ')
# 返回：DataFrame，列: concept_code, concept_name, in_date, out_date
```

### get_concept_cross_section(trade_date)
获取指定日期的完整概念板块截面数据。

```python
df = reader.get_concept_cross_section('20240115')
# 返回：DataFrame，列: concept_code, concept_name, stock_code, ts_code, in_date, out_date
```

### get_all_concepts()
获取所有概念板块列表。

```python
concepts = reader.get_all_concepts()
# 返回：DataFrame，列: concept_code, concept_name
```

### search_concepts(keyword)
搜索概念板块。

```python
chips = reader.search_concepts("芯片")
```

### get_concept_cache_info()
获取概念数据缓存信息。

```python
info = reader.get_concept_cache_info()
print(info)
# {
#   'cache_dir': '/path/to/.concept_cache',
#   'cache_file': '/path/to/all_concepts_pit_scd.csv',
#   'exists': True,
#   'valid_today': True,
#   'total_records': 500000,
#   ...
# }
```

### refresh_concept_data()
强制刷新概念数据（重新下载）。

```python
reader.refresh_concept_data()  # 强制重新下载
```

## 底层实现（ConceptDataManager）

如需直接使用底层管理器（不推荐）：

```python
from tushare_db import ConceptDataManager

manager = ConceptDataManager(db_path="tushare.db")
manager.pull_data()

# API 与 DataReader 中的概念查询方法一致
stocks = manager.get_concept_stocks('20240115', concept_name='人工智能')
```

## 数据结构

原始 CSV 文件 `all_concepts_pit_scd.csv` 格式：

| 字段 | 说明 | 示例 |
|------|------|------|
| concept_code | 概念代码 | GN036 |
| concept_name | 概念名称 | 人工智能 |
| stock_code | 股票代码（聚宽格式） | 000001.XSHE |
| in_date | 纳入日期 | 2023-05-12 |
| out_date | 剔除日期 | 2099-12-31（代表至今有效）|

DataReader 返回的 DataFrame 会额外包含：
- `ts_code`: Tushare 格式股票代码（如 `000001.SZ`）
- 日期统一为 `YYYYMMDD` 格式

## 缓存机制

1. **位置**：与 `tushare.db` 同级目录下的 `.concept_cache/all_concepts_pit_scd.csv`
2. **策略**：
   - 每日首次查询时自动检查缓存
   - 文件修改日期 = 今日：直接使用缓存
   - 文件修改日期 < 今日：尝试下载最新数据
   - 下载失败：自动回退到历史缓存
3. **强制刷新**：`reader.refresh_concept_data()`

## PIT (Point-in-Time) 查询

数据采用 SCD (缓慢变化维度) 区间生效结构，支持历史回测：

```python
# 查询某股票在不同时间点的概念归属
for date in ['20230115', '20230615', '20240115']:
    concepts = reader.get_stock_concepts(date, '000001.SZ')
    print(f"{date}: {len(concepts)} 个概念")
```

避免了使用快照数据导致的未来函数问题。

## 完整示例

```python
from tushare_db import DataReader

reader = DataReader(db_path="tushare.db")

# 示例1：获取某概念板块的历史成分股变化
concept_name = "人工智能"
for date in ['20230115', '20240115']:
    stocks = reader.get_concept_stocks(date, concept_name=concept_name)
    print(f"{date} {concept_name}板块: {len(stocks)} 只股票")

# 示例2：获取某股票的概念漂移
for date in ['20230115', '20230615', '20240115']:
    concepts = reader.get_stock_concepts(date, ts_code='000001.SZ')
    print(f"{date} 000001.SZ 所属概念: {len(concepts)} 个")

# 示例3：概念板块轮动分析
df = reader.get_concept_cross_section('20240115')
# 统计每只股票的所属概念数量
concept_counts = df.groupby('ts_code').size()
print(concept_counts.describe())

reader.close()
```
