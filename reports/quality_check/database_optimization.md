# DuckDB 数据库优化报告

**数据库路径**: `/Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db`
**数据库大小**: 19.1 GiB
**DuckDB版本**: v1.3.1 (Ossivalis)
**分析日期**: 2026-01-31

---

## 1. 当前状态诊断

### 1.1 数据库概览

| 指标 | 值 |
|------|-----|
| 总大小 | 19.1 GiB |
| 块大小 | 262,144 bytes (256 KB) |
| 总块数 | 78,380 |
| 已用块数 | 69,961 |
| 空闲块数 | 8,419 (10.7%) |
| 表数量 | 29 |

### 1.2 表存储分析 (按大小排序)

| 表名 | 行数 | 列数 | 估计大小 | 有索引 | 问题 |
|------|------|------|----------|--------|------|
| cyq_chips | 37,635,784 | 4 | ~36 MB | **无** | 缺少主键和索引 |
| daily | 16,486,059 | 11 | ~16 MB | 有 | 正常 |
| adj_factor | 16,479,372 | 3 | ~16 MB | 有 | 正常 |
| daily_basic | 15,415,921 | 18 | ~15 MB | 有 | NULL值较多 |
| stk_factor_pro | 14,870,710 | 261 | ~14 MB | **无** | 缺少主键，列数过多 |
| moneyflow | 13,239,498 | 20 | ~13 MB | 有 | NULL值存在 |
| cyq_perf | 8,629,539 | 11 | ~8 MB | 有 | 正常 |
| margin_detail | 6,169,938 | 10 | ~6 MB | 有 | 正常 |
| dc_member | 3,457,347 | 4 | ~3 MB | 有 | 正常 |
| moneyflow_dc | 3,324,050 | 15 | ~3 MB | 有 | 正常 |

### 1.3 数据类型分析

**发现的问题**:

1. **日期存储为VARCHAR**: `trade_date`、`end_date` 等日期字段使用 VARCHAR 类型存储 (格式: YYYYMMDD)
   - 影响: 日期比较效率低，无法使用日期函数优化
   - 示例: `trade_date = '20250120'`

2. **股票代码使用VARCHAR**: `ts_code` 字段为 VARCHAR(9)
   - 当前状态: 使用字典压缩，效率尚可
   - 建议: 可考虑使用 ENUM 类型或字典编码

3. **数值类型统一使用DOUBLE**: 所有价格、成交量等使用 DOUBLE
   - 成交量 (vol) 实际为整数，使用 DOUBLE 浪费空间
   - 价格精度固定为2位小数，可使用 DECIMAL

### 1.4 压缩情况分析

**daily 表压缩分布**:
| 压缩类型 | 段数 | 占比 |
|----------|------|------|
| ALP (自适应) | 1,680 | 48% |
| Constant | 1,473 | 42% |
| Dictionary | 311 | 9% |
| FSST | 27 | <1% |
| Uncompressed | 12 | <1% |

**stk_factor_pro 表压缩分布** (问题表):
| 压缩类型 | 段数 | 占比 |
|----------|------|------|
| ALP | 55,624 | 62.4% |
| **Uncompressed** | **18,057** | **20.3%** |
| Constant | 13,799 | 15.5% |
| RLE | 1,343 | 1.5% |

> **警告**: stk_factor_pro 表有 20% 的数据未压缩，是主要的优化目标。

### 1.5 索引和约束分析

**缺少主键的表** (需要添加):
- `cyq_chips` - 建议主键: (ts_code, trade_date, price)
- `stk_factor_pro` - 建议主键: (ts_code, trade_date)
- `moneyflow_cnt_ths` - 建议主键: (ts_code, trade_date)
- `stock_basic_backup` - 备份表，可忽略

**现有主键约束**:
- `daily`: (ts_code, trade_date)
- `daily_basic`: (ts_code, trade_date)
- `adj_factor`: (ts_code, trade_date)
- `moneyflow`: (ts_code, trade_date)

### 1.6 数据分布分析

**daily 表年度数据分布**:
```
年份      行数         股票数
2000     146,861        974
...
2020     963,888      4,363
2021   1,085,202      4,839
2022   1,178,830      5,181
2023   1,258,502      5,380
2024   1,293,651      5,432
2025   1,313,872      5,499
2026     109,216      5,478
```

**特点**:
- 数据量逐年增长，2025年后增长加速
- 每行组包含多年数据，未按时间有序存储
- Zone Map 效果有限

---

## 2. 查询性能分析

### 2.1 基准测试结果

| 查询类型 | 执行时间 | 扫描行数 | 评估 |
|----------|----------|----------|------|
| 按股票代码查询 (单股) | 0.006s | 6,010 | 优秀 |
| 按日期查询 (单日全市场) | **0.302s** | 5,376 | **需优化** |
| 按股票+日期范围查询 | 0.007s | 13 | 优秀 |
| 多表JOIN查询 | 0.029s | ~500 | 良好 |
| 聚合查询 (年度统计) | **0.104s** | 1,423,088 | 可优化 |
| cyq_chips 大表扫描 | **0.079s** | 36,916,863 | 需关注 |

### 2.2 慢查询模式识别

**问题1: 按日期查询效率低**
```sql
-- 问题查询 (0.3s)
SELECT * FROM daily WHERE trade_date = '20250120';
```
原因: 数据按股票代码聚集，不是按日期聚集，需要全表扫描。

**问题2: 全表聚合查询**
```sql
-- 需扫描大量数据 (0.1s)
SELECT ts_code, AVG(close) FROM daily
WHERE trade_date >= '20250101' GROUP BY ts_code;
```

### 2.3 执行计划分析

当前所有查询都使用 **Sequential Scan**，没有使用索引：
- DuckDB 是列式存储，主要依赖 Zone Maps 和压缩
- 对于分析型查询，列式扫描通常已足够高效
- 但按特定日期查询仍有优化空间

---

## 3. 存储优化建议

### 3.1 数据类型优化

#### 3.1.1 日期类型转换 [高优先级]

**当前**: VARCHAR(8) 存储 '20250120' 格式
**建议**: 转换为 DATE 类型

```sql
-- 创建新表时使用 DATE 类型
CREATE TABLE daily_optimized AS
SELECT
    ts_code,
    CAST(trade_date AS DATE) AS trade_date,  -- 需要格式转换函数
    open, high, low, close, pre_close, change, pct_chg, vol, amount
FROM daily;

-- 或者使用自定义转换
ALTER TABLE daily ADD COLUMN trade_date_dt DATE;
UPDATE daily SET trade_date_dt =
    strptime(trade_date, '%Y%m%d')::DATE;
```

**预期收益**:
- 存储空间减少约 50% (8 bytes VARCHAR -> 4 bytes DATE)
- 日期比较和范围查询性能提升 30-50%
- 支持日期函数和时间序列操作

#### 3.1.2 数值类型优化 [中优先级]

```sql
-- 成交量使用 BIGINT
vol BIGINT        -- 原 DOUBLE

-- 价格使用 DECIMAL(10,2)
open DECIMAL(10,2)   -- 精度足够
close DECIMAL(10,2)
```

**预期收益**: 存储空间减少 10-15%

### 3.2 数据压缩策略

#### 3.2.1 重新压缩 stk_factor_pro 表 [高优先级]

```sql
-- 导出并重新导入以触发压缩
COPY stk_factor_pro TO 'stk_factor_pro_backup.parquet' (FORMAT PARQUET);
DROP TABLE stk_factor_pro;
CREATE TABLE stk_factor_pro AS SELECT * FROM 'stk_factor_pro_backup.parquet';

-- 或使用 VACUUM
VACUUM ANALYZE stk_factor_pro;
```

**预期收益**: 存储空间减少 15-20%

#### 3.2.2 整库压缩优化

```sql
-- 设置更激进的压缩
PRAGMA force_compression='auto';  -- 默认已启用

-- 执行 CHECKPOINT 确保压缩生效
CHECKPOINT;
```

### 3.3 分区方案建议

#### 方案A: 按年份分区 [推荐]

```sql
-- 创建按年分区的视图
CREATE VIEW daily_partitioned AS
SELECT *,
    CAST(LEFT(trade_date, 4) AS INTEGER) AS year
FROM daily;

-- 或使用 Hive 分区格式导出
COPY (SELECT * FROM daily WHERE trade_date >= '20240101' AND trade_date < '20250101')
TO 'data/daily/year=2024/' (FORMAT PARQUET, PARTITION_BY (ts_code));
```

#### 方案B: 冷热数据分离

```sql
-- 热数据表 (最近2年)
CREATE TABLE daily_hot AS
SELECT * FROM daily WHERE trade_date >= '20240101';

-- 冷数据表 (历史数据)
CREATE TABLE daily_cold AS
SELECT * FROM daily WHERE trade_date < '20240101';

-- 创建联合视图
CREATE VIEW daily_all AS
SELECT * FROM daily_hot
UNION ALL
SELECT * FROM daily_cold;
```

**预期收益**:
- 热数据查询速度提升 40-60%
- 冷数据可使用更高压缩率

### 3.4 数据清理建议

```sql
-- 识别可能的重复数据（虽然测试未发现）
SELECT ts_code, trade_date, COUNT(*)
FROM daily GROUP BY 1, 2 HAVING COUNT(*) > 1;

-- 删除 backup 表
DROP TABLE IF EXISTS stock_basic_backup;
```

---

## 4. 索引优化建议

### 4.1 需要添加的索引

#### 4.1.1 cyq_chips 表 [高优先级]

```sql
-- 添加主键约束
ALTER TABLE cyq_chips ADD PRIMARY KEY (ts_code, trade_date, price);

-- 如果主键已存在数据，先去重
CREATE TABLE cyq_chips_new AS
SELECT DISTINCT ON (ts_code, trade_date, price) *
FROM cyq_chips;
DROP TABLE cyq_chips;
ALTER TABLE cyq_chips_new RENAME TO cyq_chips;
ALTER TABLE cyq_chips ADD PRIMARY KEY (ts_code, trade_date, price);
```

#### 4.1.2 stk_factor_pro 表 [高优先级]

```sql
-- 添加主键
ALTER TABLE stk_factor_pro ADD PRIMARY KEY (ts_code, trade_date);
```

#### 4.1.3 日期索引 (可选)

DuckDB 的列式存储通常不需要额外索引，但如果按日期查询频繁：

```sql
-- 创建 ART 索引（实验性）
CREATE INDEX idx_daily_trade_date ON daily(trade_date);
```

**注意**: DuckDB 索引主要用于点查询，对范围查询效果有限。

### 4.2 复合索引设计

当前主键 (ts_code, trade_date) 已经是最优的复合索引：
- 支持按股票查询 (利用前缀)
- 支持按股票+日期组合查询

### 4.3 Zone Map 优化

Zone Maps 是 DuckDB 内置的"索引"，通过数据有序存储来优化：

```sql
-- 按 trade_date 排序重建表，优化日期查询
CREATE TABLE daily_sorted AS
SELECT * FROM daily ORDER BY trade_date, ts_code;
DROP TABLE daily;
ALTER TABLE daily_sorted RENAME TO daily;
```

**预期收益**: 按日期查询性能提升 50-80%

---

## 5. 查询优化建议

### 5.1 常用查询优化写法

#### 5.1.1 单股历史数据查询 (已优化)

```sql
-- 当前写法已最优
SELECT * FROM daily
WHERE ts_code = '000001.SZ'
ORDER BY trade_date DESC
LIMIT 100;
```

#### 5.1.2 全市场单日查询 (需优化)

```sql
-- 原查询
SELECT * FROM daily WHERE trade_date = '20250120';

-- 优化1: 只选择需要的列
SELECT ts_code, open, high, low, close, vol
FROM daily WHERE trade_date = '20250120';

-- 优化2: 如果表已按日期排序
SELECT * FROM daily
WHERE trade_date = '20250120'
USING SAMPLE 100%;  -- 强制顺序扫描（仅测试用）
```

#### 5.1.3 多表JOIN优化

```sql
-- 原查询 (多次扫描)
SELECT d.*, db.pe, af.adj_factor
FROM daily d
LEFT JOIN daily_basic db ON d.ts_code = db.ts_code AND d.trade_date = db.trade_date
LEFT JOIN adj_factor af ON d.ts_code = af.ts_code AND d.trade_date = af.trade_date
WHERE d.ts_code = '000001.SZ';

-- 优化: 使用 CTE 减少扫描
WITH base AS (
    SELECT ts_code, trade_date FROM daily
    WHERE ts_code = '000001.SZ' AND trade_date >= '20240101'
)
SELECT d.*, db.pe, af.adj_factor
FROM daily d
JOIN base b USING (ts_code, trade_date)
LEFT JOIN daily_basic db USING (ts_code, trade_date)
LEFT JOIN adj_factor af USING (ts_code, trade_date);
```

#### 5.1.4 聚合查询优化

```sql
-- 使用 QUALIFY 替代子查询
SELECT ts_code, trade_date, close,
    AVG(close) OVER (PARTITION BY ts_code ORDER BY trade_date ROWS 20 PRECEDING) as ma20
FROM daily
WHERE trade_date >= '20240101'
QUALIFY ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY trade_date DESC) = 1;
```

### 5.2 批量查询策略

```sql
-- 批量获取多只股票数据
SELECT * FROM daily
WHERE ts_code IN ('000001.SZ', '000002.SZ', '600000.SH')
  AND trade_date >= '20240101';

-- 使用 UNION ALL 并行化
SELECT * FROM daily WHERE ts_code = '000001.SZ' AND trade_date >= '20240101'
UNION ALL
SELECT * FROM daily WHERE ts_code = '000002.SZ' AND trade_date >= '20240101';
```

### 5.3 缓存策略

```python
# Python 应用层缓存示例
import duckdb
from functools import lru_cache

@lru_cache(maxsize=100)
def get_stock_data(ts_code: str, start_date: str, end_date: str):
    """缓存常用股票数据查询"""
    conn = duckdb.connect('tushare.db', read_only=True)
    return conn.execute("""
        SELECT * FROM daily
        WHERE ts_code = ? AND trade_date BETWEEN ? AND ?
        ORDER BY trade_date
    """, [ts_code, start_date, end_date]).fetchdf()
```

---

## 6. 架构优化建议

### 6.1 表设计改进

#### 6.1.1 stk_factor_pro 表拆分 [高优先级]

当前 261 列过多，建议拆分为多个主题表：

```sql
-- 价格指标表
CREATE TABLE stk_price AS
SELECT ts_code, trade_date,
    open, open_hfq, open_qfq,
    high, high_hfq, high_qfq,
    low, low_hfq, low_qfq,
    close, close_hfq, close_qfq,
    pre_close, change, pct_chg, vol, amount
FROM stk_factor_pro;

-- 技术指标表 - MACD
CREATE TABLE stk_macd AS
SELECT ts_code, trade_date,
    macd_bfq, macd_hfq, macd_qfq,
    macd_dif_bfq, macd_dif_hfq, macd_dif_qfq,
    macd_dea_bfq, macd_dea_hfq, macd_dea_qfq
FROM stk_factor_pro;

-- 技术指标表 - KDJ
CREATE TABLE stk_kdj AS
SELECT ts_code, trade_date,
    kdj_k_bfq, kdj_k_hfq, kdj_k_qfq,
    kdj_d_bfq, kdj_d_hfq, kdj_d_qfq,
    kdj_j_bfq, kdj_j_hfq, kdj_j_qfq
FROM stk_factor_pro;

-- ... 其他指标类似
```

**预期收益**:
- 查询只需要部分指标时，减少 I/O
- 压缩效率提升
- 维护更灵活

#### 6.1.2 cyq_chips 表优化

```sql
-- 当前结构 (37M 行)
-- ts_code, trade_date, price, percent

-- 建议: 使用更紧凑的存储
CREATE TABLE cyq_chips_optimized (
    ts_code VARCHAR NOT NULL,
    trade_date DATE NOT NULL,
    price DECIMAL(10,2) NOT NULL,
    percent DECIMAL(8,6) NOT NULL,
    PRIMARY KEY (ts_code, trade_date, price)
);
```

### 6.2 数据更新策略

#### 6.2.1 增量更新机制

```sql
-- 使用 _tushare_cache_metadata 表跟踪更新
CREATE TABLE IF NOT EXISTS update_log (
    table_name VARCHAR PRIMARY KEY,
    last_update_date VARCHAR,
    last_update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    rows_updated BIGINT
);

-- 增量更新示例
INSERT INTO daily
SELECT * FROM staging_daily s
WHERE NOT EXISTS (
    SELECT 1 FROM daily d
    WHERE d.ts_code = s.ts_code AND d.trade_date = s.trade_date
);

-- 或使用 INSERT OR REPLACE
INSERT OR REPLACE INTO daily SELECT * FROM staging_daily;
```

#### 6.2.2 ETL 管道设计

```python
def incremental_update(table_name: str, source_df: pd.DataFrame):
    """增量更新数据表"""
    conn = duckdb.connect('tushare.db')

    # 1. 创建临时表
    conn.execute(f"CREATE TEMP TABLE staging AS SELECT * FROM {table_name} LIMIT 0")
    conn.register('staging_data', source_df)
    conn.execute("INSERT INTO staging SELECT * FROM staging_data")

    # 2. 合并数据 (UPSERT)
    conn.execute(f"""
        INSERT OR REPLACE INTO {table_name}
        SELECT * FROM staging
    """)

    # 3. 更新元数据
    conn.execute(f"""
        INSERT OR REPLACE INTO update_log
        VALUES ('{table_name}', CURRENT_DATE, CURRENT_TIMESTAMP, {len(source_df)})
    """)

    conn.close()
```

### 6.3 多数据库架构 (可选)

如果数据继续增长，考虑分库：

```
tushare_daily.db      # 日线数据 (~10 GB)
tushare_factor.db     # 因子数据 (~5 GB)
tushare_financial.db  # 财务数据 (~2 GB)
tushare_misc.db       # 其他数据 (~2 GB)
```

```python
# 使用 ATTACH 连接多个数据库
conn = duckdb.connect()
conn.execute("ATTACH 'tushare_daily.db' AS daily_db")
conn.execute("ATTACH 'tushare_factor.db' AS factor_db")

# 跨库查询
result = conn.execute("""
    SELECT d.*, f.macd_bfq
    FROM daily_db.daily d
    JOIN factor_db.stk_macd f USING (ts_code, trade_date)
""")
```

---

## 7. 实施计划

### 阶段1: 快速优化 (1-2小时) [立即执行]

1. **添加缺失的主键**
```sql
ALTER TABLE stk_factor_pro ADD PRIMARY KEY (ts_code, trade_date);
-- cyq_chips 可能需要先检查数据唯一性
```

2. **执行 VACUUM**
```sql
VACUUM ANALYZE;
CHECKPOINT;
```

3. **删除无用表**
```sql
DROP TABLE IF EXISTS stock_basic_backup;
```

**预期收益**: 存储减少 5-10%, 查询稳定性提升

### 阶段2: 数据类型优化 (2-4小时)

1. **日期类型转换**
2. **数值类型优化**
3. **重新压缩数据**

**预期收益**: 存储减少 15-25%, 日期查询提升 30%

### 阶段3: 结构优化 (4-8小时)

1. **按日期排序重建 daily 表**
2. **拆分 stk_factor_pro 表**
3. **建立冷热数据分离**

**预期收益**: 日期查询提升 50-80%, 整体查询提升 20-30%

### 阶段4: 长期优化 (持续)

1. 建立增量更新机制
2. 监控查询性能
3. 根据使用模式调整架构

---

## 8. 预期性能提升总结

| 优化项 | 存储节省 | 查询提升 | 实施难度 |
|--------|----------|----------|----------|
| 添加主键 | 0% | 5-10% | 低 |
| VACUUM 压缩 | 5-10% | 0% | 低 |
| 日期类型转换 | 10-15% | 30-50% | 中 |
| 按日期排序 | 0% | 50-80% | 中 |
| 表拆分 | 5-10% | 20-30% | 高 |
| 冷热分离 | 0% | 40-60% | 中 |

**整体预期**:
- 存储空间: 从 19GB 减少到 ~14-15GB (节省 20-25%)
- 按日期查询: 从 0.3s 减少到 <0.05s (提升 6x)
- 按股票查询: 保持 <0.01s (已优化)
- 多表JOIN: 从 0.03s 减少到 <0.02s (提升 1.5x)

---

## 9. 优化脚本

### 完整优化脚本

```sql
-- ============================================
-- DuckDB 数据库优化脚本
-- 执行前请备份数据库!
-- ============================================

-- 阶段1: 基础优化
-- ------------------------------------------

-- 1.1 添加缺失的主键
ALTER TABLE stk_factor_pro ADD PRIMARY KEY (ts_code, trade_date);
ALTER TABLE moneyflow_cnt_ths ADD PRIMARY KEY (ts_code, trade_date);

-- 1.2 删除无用表
DROP TABLE IF EXISTS stock_basic_backup;

-- 1.3 执行压缩
VACUUM ANALYZE;
CHECKPOINT;

-- 阶段2: 日期优化 (需要重建表)
-- ------------------------------------------

-- 2.1 创建日期转换函数视图
CREATE OR REPLACE VIEW daily_with_date AS
SELECT
    ts_code,
    strptime(trade_date, '%Y%m%d')::DATE AS trade_date_dt,
    trade_date AS trade_date_str,
    open, high, low, close, pre_close, change, pct_chg, vol, amount
FROM daily;

-- 2.2 按日期排序重建 (可选，需要临时空间)
-- CREATE TABLE daily_new AS
-- SELECT * FROM daily ORDER BY trade_date, ts_code;
-- DROP TABLE daily;
-- ALTER TABLE daily_new RENAME TO daily;
-- ALTER TABLE daily ADD PRIMARY KEY (ts_code, trade_date);

-- 阶段3: 查询优化视图
-- ------------------------------------------

-- 3.1 最近N天数据视图
CREATE OR REPLACE VIEW daily_recent AS
SELECT * FROM daily
WHERE trade_date >= strftime(CURRENT_DATE - INTERVAL '30 days', '%Y%m%d');

-- 3.2 完整数据宽表视图
CREATE OR REPLACE VIEW daily_full AS
SELECT
    d.*,
    db.pe, db.pb, db.total_mv, db.circ_mv, db.turnover_rate,
    af.adj_factor,
    mf.net_mf_amount, mf.buy_elg_amount, mf.sell_elg_amount
FROM daily d
LEFT JOIN daily_basic db USING (ts_code, trade_date)
LEFT JOIN adj_factor af USING (ts_code, trade_date)
LEFT JOIN moneyflow mf USING (ts_code, trade_date);

-- 完成
SELECT 'Optimization completed!' AS status;
```

---

## 10. 监控建议

### 定期检查脚本

```sql
-- 检查表大小变化
SELECT
    table_name,
    estimated_size / 1024 / 1024 AS size_mb,
    column_count,
    index_count
FROM duckdb_tables()
ORDER BY estimated_size DESC;

-- 检查压缩效率
SELECT
    compression,
    COUNT(*) as segments,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as pct
FROM pragma_storage_info('daily')
GROUP BY compression
ORDER BY segments DESC;

-- 检查查询性能
EXPLAIN ANALYZE SELECT * FROM daily WHERE trade_date = '20250120';
```

---

*报告生成时间: 2026-01-31*
*工具版本: DuckDB v1.3.1*
