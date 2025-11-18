# 数据库Schema与API文档一致性检查报告

> 生成时间: 2024-11-18
> 检查范围: 数据库表结构 vs API文档 vs Web端点

---

## ✅ 总体检查结果

**数据库表总数**: 18个
**DataReader方法数**: 14个
**Web API端点数**: 17个

### 一致性状态

- ✅ 所有文档中提到的表都存在于数据库
- ⚠️ 发现部分字段描述不准确
- ⚠️ 2个数据库表未在文档中说明

---

## 📊 数据库表清单

### 核心业务表（16个）

| 表名 | 字段数 | 用途 | 文档状态 |
|------|--------|------|----------|
| `stock_basic` | 4 | 股票基础信息 | ✅ 已文档化 |
| `pro_bar` | 13 | 日线数据（未复权） | ✅ 已文档化 |
| `adj_factor` | 3 | 复权因子 | ✅ 已文档化 |
| `daily_basic` | 18 | 每日指标（PE、PB等） | ✅ 已文档化 |
| `trade_cal` | 4 | 交易日历 | ✅ 已文档化 |
| `stock_company` | 15 | 上市公司信息 | ✅ 已文档化 |
| `cyq_perf` | 11 | 筹码分布绩效 | ✅ 已文档化 |
| `cyq_chips` | 4 | 筹码成本分布 | ✅ 已文档化 |
| `stk_factor_pro` | 261 | 技术因子（超大表） | ✅ 已文档化 |
| `dc_member` | 4 | 董财板块成分股 | ✅ 已文档化 |
| `dc_index` | 11 | 董财板块指数 | ✅ 已文档化 |
| `index_basic` | 8 | 指数基础信息 | ✅ 已文档化 |
| `index_weight` | 4 | 指数成分权重 | ✅ 已文档化 |
| `hs_const` | 5 | 沪深港通成分 | ✅ 已文档化 |
| `fina_indicator_vip` | 109 | 财务指标（超大表） | ✅ 已文档化 |
| `moneyflow_ind_dc` | 18 | 董财行业资金流向 | ✅ 已文档化 |

### 未文档化的表（2个）

| 表名 | 字段数 | 说明 | 建议 |
|------|--------|------|------|
| `_tushare_cache_metadata` | 2 | 缓存元数据（内部表） | 无需暴露 |
| `moneyflow_cnt_ths` | 12 | 同花顺版资金流向 | 可选择暴露 |

---

## ⚠️ 发现的不一致问题

### 问题 1: stock_basic 表字段描述错误

**文档中的描述**:
```
返回列: ts_code, name, industry, list_date, market
```

**实际数据库字段**:
```sql
ts_code      VARCHAR  NOT NULL
list_date    VARCHAR  NULL
market       VARCHAR  NULL
name         VARCHAR  NULL
```

**问题**:
- ❌ 文档中提到的 `industry` 字段不存在
- ❌ 文档中提到的 `list_status` 字段不存在

**影响范围**:
- `README_FOR_AI.md` - 第45行、第83行
- `API_REFERENCE_FOR_LLM.md` - 可能多处

**修复方案**:
1. 更新文档，移除 `industry` 字段的提及
2. 说明 `list_status` 参数在当前数据库版本中不可用（已在代码中做兼容处理）

### 问题 2: 表字段数量差异

部分超大表的字段数量：
- `stk_factor_pro`: 261个字段（技术因子）
- `fina_indicator_vip`: 109个字段（财务指标）

这些表字段极多，文档中无法一一列举，建议使用示例字段或字段分组说明。

---

## 📋 常用表详细字段

### stock_basic（股票基础信息）

```sql
CREATE TABLE stock_basic (
    ts_code      VARCHAR  NOT NULL,  -- 股票代码
    list_date    VARCHAR  NULL,      -- 上市日期
    market       VARCHAR  NULL,      -- 市场类别
    name         VARCHAR  NULL       -- 股票名称
);
```

**注意**: 无 `industry`（行业）和 `list_status`（上市状态）字段！

### pro_bar（日线数据，未复权）

```sql
CREATE TABLE pro_bar (
    trade_date     VARCHAR  NOT NULL,  -- 交易日期
    ts_code        VARCHAR  NOT NULL,  -- 股票代码
    open           DOUBLE   NULL,      -- 开盘价
    high           DOUBLE   NULL,      -- 最高价
    low            DOUBLE   NULL,      -- 最低价
    close          DOUBLE   NULL,      -- 收盘价
    pre_close      DOUBLE   NULL,      -- 昨收价
    change         DOUBLE   NULL,      -- 涨跌额
    pct_chg        DOUBLE   NULL,      -- 涨跌幅(%)
    vol            DOUBLE   NULL,      -- 成交量(手)
    amount         DOUBLE   NULL,      -- 成交额(千元)
    turnover_rate  DOUBLE   NULL,      -- 换手率(%)
    volume_ratio   DOUBLE   NULL       -- 量比
);
```

### daily_basic（每日指标）

```sql
CREATE TABLE daily_basic (
    ts_code          VARCHAR  NOT NULL,  -- 股票代码
    trade_date       VARCHAR  NOT NULL,  -- 交易日期
    close            DOUBLE   NULL,      -- 收盘价
    turnover_rate    DOUBLE   NULL,      -- 换手率
    turnover_rate_f  DOUBLE   NULL,      -- 换手率(自由流通股)
    volume_ratio     DOUBLE   NULL,      -- 量比
    pe               DOUBLE   NULL,      -- 市盈率(总股本)
    pe_ttm           DOUBLE   NULL,      -- 市盈率(TTM)
    pb               DOUBLE   NULL,      -- 市净率
    ps               DOUBLE   NULL,      -- 市销率
    ps_ttm           DOUBLE   NULL,      -- 市销率(TTM)
    dv_ratio         DOUBLE   NULL,      -- 股息率
    dv_ttm           DOUBLE   NULL,      -- 股息率(TTM)
    total_share      DOUBLE   NULL,      -- 总股本(万股)
    float_share      DOUBLE   NULL,      -- 流通股本(万股)
    free_share       DOUBLE   NULL,      -- 自由流通股本(万股)
    total_mv         DOUBLE   NULL,      -- 总市值(万元)
    circ_mv          DOUBLE   NULL       -- 流通市值(万元)
);
```

### adj_factor（复权因子）

```sql
CREATE TABLE adj_factor (
    ts_code      VARCHAR  NOT NULL,  -- 股票代码
    trade_date   VARCHAR  NOT NULL,  -- 交易日期
    adj_factor   DOUBLE   NULL       -- 复权因子
);
```

**复权计算公式**:
```python
# 前复权
adjusted_price = close * adj_factor

# 后复权
adjusted_price = close * (adj_factor / latest_adj_factor)
```

### trade_cal（交易日历）

```sql
CREATE TABLE trade_cal (
    exchange      VARCHAR  NULL,  -- 交易所(SSE/SZSE)
    cal_date      VARCHAR  NULL,  -- 日期
    is_open       BIGINT   NULL,  -- 是否交易日(1=是,0=否)
    pretrade_date VARCHAR  NULL   -- 上一交易日
);
```

---

## 🔧 DataReader 方法覆盖情况

### 已实现的方法（14个）

| 方法名 | 对应表 | 状态 |
|--------|--------|------|
| `get_stock_basic()` | stock_basic | ✅ |
| `get_stock_daily()` | pro_bar + adj_factor | ✅ |
| `get_multiple_stocks_daily()` | pro_bar + adj_factor | ✅ |
| `get_trade_calendar()` | trade_cal | ✅ |
| `get_daily_basic()` | daily_basic | ✅ |
| `get_adj_factor()` | adj_factor | ✅ |
| `get_stock_company()` | stock_company | ✅ |
| `get_cyq_perf()` | cyq_perf | ✅ |
| `get_stk_factor_pro()` | stk_factor_pro | ✅ |
| `get_moneyflow_ind_dc()` | moneyflow_ind_dc | ✅ |
| `query()` | 所有表 | ✅ 通用查询 |
| `table_exists()` | 元数据 | ✅ 工具方法 |
| `get_table_info()` | 元数据 | ✅ 工具方法 |
| `close()` | - | ✅ 资源管理 |

### 未实现专用方法的表

这些表可以通过 `reader.query()` 自定义SQL查询：
- `cyq_chips` - 筹码成本分布
- `dc_member` - 板块成分
- `dc_index` - 板块指数
- `index_basic` - 指数基础信息
- `index_weight` - 指数权重
- `hs_const` - 沪深港通成分
- `fina_indicator_vip` - 财务指标
- `moneyflow_cnt_ths` - 同花顺资金流向

---

## 🌐 Web API 端点覆盖情况

### 已实现端点（17个）

| 路径 | 对应表/功能 | 状态 |
|------|------------|------|
| `/api/stock_basic` | stock_basic | ✅ |
| `/api/pro_bar` | pro_bar + 复权计算 | ✅ |
| `/api/daily_basic` | daily_basic | ✅ |
| `/api/adj_factor` | adj_factor | ✅ |
| `/api/trade_cal` | trade_cal | ✅ |
| `/api/stock_company` | stock_company | ✅ |
| `/api/cyq_chips` | cyq_chips | ✅ |
| `/api/cyq_perf` | cyq_perf | ✅ |
| `/api/stk_factor_pro` | stk_factor_pro | ✅ |
| `/api/dc_member` | dc_member | ✅ |
| `/api/dc_index` | dc_index | ✅ |
| `/api/index_basic` | index_basic | ✅ |
| `/api/index_weight` | index_weight | ✅ |
| `/api/hs_const` | hs_const | ✅ |
| `/api/fina_indicator_vip` | fina_indicator_vip | ✅ |
| `/api/get_top_n_sector_members` | 复杂查询 | ⚠️ 需检查实现 |

---

## 🔍 建议修正

### 立即修正（高优先级）

1. **修正 README_FOR_AI.md**
   - 第45行：移除 `industry` 字段
   - 第83行：更新表结构说明

2. **修正 API_REFERENCE_FOR_LLM.md**
   - 检查所有提到 `stock_basic` 的地方
   - 更新字段列表为: `ts_code, name, list_date, market`

### 可选改进（低优先级）

1. **添加 moneyflow_cnt_ths 表的文档和API**
   - 同花顺版本的资金流向数据
   - 可能对某些用户有用

2. **为超大表添加字段分组说明**
   - `stk_factor_pro` (261字段)
   - `fina_indicator_vip` (109字段)
   - 按功能分组列举常用字段

3. **添加字段注释到文档**
   - 特别是单位说明（万股、万元、%等）
   - 提高文档可读性

---

## 📄 附件

详细Schema已导出到: `database_schema.json`

查看完整字段列表:
```bash
cat database_schema.json | jq '.tables.stock_basic'
```

---

**检查工具**: `check_schema.py`
**下次检查**: 数据库结构变更后
