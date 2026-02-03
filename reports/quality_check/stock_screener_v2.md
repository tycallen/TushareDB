# 股票基本面筛选系统设计文档 V2

## 1. 系统概述

### 1.1 设计目标
基于 DuckDB 数据库构建一套完整的股票基本面筛选系统，支持多维度因子计算和多种投资策略筛选。

### 1.2 数据库信息
- **数据库路径**: `/Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db`
- **当前股票数量**: 5,477 只上市股票
- **行情数据截止**: 2026-01-30
- **财务数据截止**: 2025-12-31

### 1.3 核心数据表

| 表名 | 用途 | 主键 |
|------|------|------|
| `stock_basic` | 股票基本信息 | ts_code |
| `daily_basic` | 每日估值指标 | ts_code, trade_date |
| `fina_indicator_vip` | 财务指标（核心） | ts_code, end_date |
| `income` | 利润表 | ts_code, end_date, report_type |
| `balancesheet` | 资产负债表 | ts_code, end_date, report_type |
| `cashflow` | 现金流量表 | ts_code, end_date, report_type |
| `dividend` | 分红数据 | ts_code, end_date |
| `index_member_all` | 行业分类 | l3_code, ts_code, in_date |

---

## 2. 因子体系设计

### 2.1 估值因子 (Valuation Factors)

#### 2.1.1 基础估值指标
| 因子名称 | 数据来源 | 字段 | 计算方法 |
|----------|----------|------|----------|
| PE_TTM | daily_basic | pe_ttm | 直接取值 |
| PB | daily_basic | pb | 直接取值 |
| PS_TTM | daily_basic | ps_ttm | 直接取值 |
| 股息率TTM | daily_basic | dv_ttm | 直接取值 |
| 总市值 | daily_basic | total_mv | 直接取值(万元) |
| 流通市值 | daily_basic | circ_mv | 直接取值(万元) |

#### 2.1.2 历史分位数计算
```sql
-- PE历史分位数 (过去N年)
WITH pe_history AS (
    SELECT
        ts_code,
        pe_ttm,
        PERCENT_RANK() OVER (
            PARTITION BY ts_code
            ORDER BY pe_ttm
        ) as pe_percentile
    FROM daily_basic
    WHERE trade_date >= strftime('%Y%m%d', current_date - INTERVAL '3 year')
      AND pe_ttm > 0 AND pe_ttm < 500
)
SELECT ts_code, pe_ttm, pe_percentile
FROM pe_history
WHERE trade_date = (SELECT MAX(trade_date) FROM daily_basic);
```

#### 2.1.3 PEG计算
```sql
-- PEG = PE / 预期盈利增长率
SELECT
    d.ts_code,
    d.pe_ttm,
    f.netprofit_yoy as profit_growth,
    CASE
        WHEN f.netprofit_yoy > 0 AND d.pe_ttm > 0
        THEN d.pe_ttm / f.netprofit_yoy
        ELSE NULL
    END as peg
FROM daily_basic d
JOIN (
    SELECT ts_code, netprofit_yoy,
           ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY end_date DESC) as rn
    FROM fina_indicator_vip
    WHERE end_date LIKE '%1231'  -- 年报
) f ON d.ts_code = f.ts_code AND f.rn = 1
WHERE d.trade_date = (SELECT MAX(trade_date) FROM daily_basic);
```

#### 2.1.4 EV/EBITDA计算
```sql
-- EV/EBITDA 企业价值倍数
WITH enterprise_value AS (
    SELECT
        d.ts_code,
        d.total_mv * 10000 as market_cap,  -- 转换为元
        COALESCE(b.st_borr, 0) + COALESCE(b.lt_borr, 0) as total_debt,
        COALESCE(b.money_cap, 0) as cash,
        d.total_mv * 10000 + COALESCE(b.st_borr, 0) + COALESCE(b.lt_borr, 0) - COALESCE(b.money_cap, 0) as ev
    FROM daily_basic d
    LEFT JOIN (
        SELECT ts_code, st_borr, lt_borr, money_cap,
               ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY end_date DESC) as rn
        FROM balancesheet
        WHERE report_type = '1'
    ) b ON d.ts_code = b.ts_code AND b.rn = 1
    WHERE d.trade_date = (SELECT MAX(trade_date) FROM daily_basic)
),
ebitda_calc AS (
    SELECT
        ts_code,
        ebitda
    FROM (
        SELECT ts_code, ebitda,
               ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY end_date DESC) as rn
        FROM fina_indicator_vip
        WHERE ebitda IS NOT NULL
    ) t WHERE rn = 1
)
SELECT
    ev.ts_code,
    ev.ev,
    e.ebitda,
    CASE WHEN e.ebitda > 0 THEN ev.ev / e.ebitda ELSE NULL END as ev_ebitda
FROM enterprise_value ev
LEFT JOIN ebitda_calc e ON ev.ts_code = e.ts_code;
```

---

### 2.2 盈利因子 (Profitability Factors)

#### 2.2.1 核心盈利指标
| 因子名称 | 数据来源 | 字段 | 说明 |
|----------|----------|------|------|
| ROE | fina_indicator_vip | roe | 净资产收益率 |
| ROE_加权 | fina_indicator_vip | roe_waa | 加权净资产收益率 |
| ROE_扣非 | fina_indicator_vip | roe_dt | 扣非净资产收益率 |
| ROA | fina_indicator_vip | roa | 总资产收益率 |
| ROIC | fina_indicator_vip | roic | 投资资本回报率 |
| 毛利率 | fina_indicator_vip | gross_margin | 销售毛利率 |
| 净利率 | fina_indicator_vip | netprofit_margin | 销售净利率 |
| 营业利润率 | fina_indicator_vip | op_of_gr | 营业利润/营业总收入 |

#### 2.2.2 盈利因子提取SQL
```sql
-- 获取最新财报盈利指标
WITH latest_fina AS (
    SELECT
        ts_code,
        end_date,
        roe,
        roe_waa,
        roe_dt,
        roa,
        roic,
        gross_margin,
        netprofit_margin,
        op_of_gr as operating_profit_margin,
        ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY end_date DESC) as rn
    FROM fina_indicator_vip
)
SELECT
    ts_code,
    end_date,
    roe,
    roe_waa,
    roe_dt,
    roa,
    roic,
    gross_margin,
    netprofit_margin,
    operating_profit_margin
FROM latest_fina
WHERE rn = 1;
```

#### 2.2.3 单季度盈利指标
```sql
-- 单季度ROE和净利率
SELECT
    ts_code,
    end_date,
    q_roe,        -- 单季度ROE
    q_dt_roe,     -- 单季度扣非ROE
    q_npta,       -- 单季度净利润/总资产
    q_ocf_to_sales -- 单季度经营现金流/营收
FROM fina_indicator_vip
WHERE end_date = (SELECT MAX(end_date) FROM fina_indicator_vip);
```

---

### 2.3 成长因子 (Growth Factors)

#### 2.3.1 核心成长指标
| 因子名称 | 数据来源 | 字段 | 说明 |
|----------|----------|------|------|
| 营收增长率(YoY) | fina_indicator_vip | tr_yoy | 营业总收入同比 |
| 营业收入增长率 | fina_indicator_vip | or_yoy | 营业收入同比 |
| 净利润增长率 | fina_indicator_vip | netprofit_yoy | 归母净利润同比 |
| 扣非净利润增长率 | fina_indicator_vip | dt_netprofit_yoy | 扣非净利润同比 |
| 经营现金流增长率 | fina_indicator_vip | ocf_yoy | 经营现金流同比 |
| 基本EPS增长率 | fina_indicator_vip | basic_eps_yoy | 基本每股收益同比 |
| 资产增长率 | fina_indicator_vip | assets_yoy | 总资产同比增长 |
| 净资产增长率 | fina_indicator_vip | eqt_yoy | 净资产同比增长 |

#### 2.3.2 成长因子提取SQL
```sql
-- 获取成长指标
WITH growth_data AS (
    SELECT
        ts_code,
        end_date,
        tr_yoy as revenue_growth,
        or_yoy as operating_revenue_growth,
        netprofit_yoy as net_profit_growth,
        dt_netprofit_yoy as dt_net_profit_growth,
        ocf_yoy as ocf_growth,
        basic_eps_yoy as eps_growth,
        assets_yoy as asset_growth,
        eqt_yoy as equity_growth,
        ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY end_date DESC) as rn
    FROM fina_indicator_vip
)
SELECT * FROM growth_data WHERE rn = 1;
```

#### 2.3.3 多期成长稳定性
```sql
-- 计算过去4个财报期的成长稳定性
WITH growth_periods AS (
    SELECT
        ts_code,
        end_date,
        netprofit_yoy,
        tr_yoy,
        ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY end_date DESC) as period_rank
    FROM fina_indicator_vip
    WHERE end_date LIKE '%1231' OR end_date LIKE '%0630'  -- 年报和半年报
),
growth_stats AS (
    SELECT
        ts_code,
        AVG(netprofit_yoy) as avg_profit_growth,
        STDDEV(netprofit_yoy) as std_profit_growth,
        AVG(tr_yoy) as avg_revenue_growth,
        STDDEV(tr_yoy) as std_revenue_growth,
        MIN(netprofit_yoy) as min_profit_growth,
        COUNT(*) as periods
    FROM growth_periods
    WHERE period_rank <= 4
    GROUP BY ts_code
)
SELECT
    ts_code,
    avg_profit_growth,
    std_profit_growth,
    avg_revenue_growth,
    std_revenue_growth,
    -- 成长稳定性得分 (平均增长/波动)
    CASE WHEN std_profit_growth > 0
         THEN avg_profit_growth / std_profit_growth
         ELSE NULL
    END as growth_stability_score
FROM growth_stats
WHERE periods >= 3;
```

---

### 2.4 财务健康因子 (Financial Health Factors)

#### 2.4.1 偿债能力指标
| 因子名称 | 数据来源 | 字段/计算 | 说明 |
|----------|----------|-----------|------|
| 流动比率 | fina_indicator_vip | current_ratio | 流动资产/流动负债 |
| 速动比率 | fina_indicator_vip | quick_ratio | (流动资产-存货)/流动负债 |
| 现金比率 | fina_indicator_vip | cash_ratio | 现金/流动负债 |
| 资产负债率 | fina_indicator_vip | debt_to_assets | 总负债/总资产 |
| 产权比率 | fina_indicator_vip | debt_to_eqt | 总负债/所有者权益 |

#### 2.4.2 资产负债表健康度
```sql
-- 财务健康指标
WITH balance_health AS (
    SELECT
        b.ts_code,
        b.end_date,
        b.total_assets,
        b.total_liab,
        b.total_hldr_eqy_exc_min_int as equity,
        b.total_cur_assets,
        b.total_cur_liab,
        b.money_cap as cash,
        b.st_borr as short_term_debt,
        b.lt_borr as long_term_debt,
        -- 资产负债率
        CASE WHEN b.total_assets > 0
             THEN b.total_liab / b.total_assets * 100
             ELSE NULL
        END as debt_ratio,
        -- 流动比率 (非金融企业)
        CASE WHEN b.total_cur_liab > 0
             THEN b.total_cur_assets / b.total_cur_liab
             ELSE NULL
        END as current_ratio,
        ROW_NUMBER() OVER (PARTITION BY b.ts_code ORDER BY b.end_date DESC) as rn
    FROM balancesheet b
    WHERE b.report_type = '1'
)
SELECT * FROM balance_health WHERE rn = 1;
```

#### 2.4.3 现金流健康度
```sql
-- 现金流健康指标
WITH cashflow_health AS (
    SELECT
        c.ts_code,
        c.end_date,
        c.n_cashflow_act as operating_cashflow,
        c.n_cashflow_inv_act as investing_cashflow,
        c.n_cash_flows_fnc_act as financing_cashflow,
        -- 计算自由现金流 (简化版)
        c.n_cashflow_act + c.n_cashflow_inv_act as free_cashflow,
        ROW_NUMBER() OVER (PARTITION BY c.ts_code ORDER BY c.end_date DESC) as rn
    FROM cashflow c
    WHERE c.report_type = '1'
),
fina_data AS (
    SELECT
        ts_code,
        end_date,
        ocfps,  -- 每股经营现金流
        fcff,   -- 企业自由现金流
        fcfe,   -- 股权自由现金流
        q_ocf_to_sales,  -- 单季度经营现金流/营收
        ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY end_date DESC) as rn
    FROM fina_indicator_vip
)
SELECT
    ch.ts_code,
    ch.operating_cashflow,
    ch.investing_cashflow,
    ch.financing_cashflow,
    ch.free_cashflow,
    fd.ocfps,
    fd.fcff,
    fd.q_ocf_to_sales
FROM cashflow_health ch
LEFT JOIN fina_data fd ON ch.ts_code = fd.ts_code AND fd.rn = 1
WHERE ch.rn = 1;
```

#### 2.4.4 综合财务健康评分
```sql
-- Altman Z-Score (适用于制造业)
-- Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
WITH zscore_calc AS (
    SELECT
        b.ts_code,
        b.total_assets,
        b.total_cur_assets,
        b.total_cur_liab,
        b.total_liab,
        b.total_hldr_eqy_exc_min_int as equity,
        i.total_revenue,
        i.ebit,
        d.total_mv * 10000 as market_cap,
        -- X1: 营运资本/总资产
        CASE WHEN b.total_assets > 0
             THEN (b.total_cur_assets - b.total_cur_liab) / b.total_assets
             ELSE NULL END as x1,
        -- X2: 留存收益/总资产 (用未分配利润近似)
        CASE WHEN b.total_assets > 0
             THEN b.undistr_porfit / b.total_assets
             ELSE NULL END as x2,
        -- X3: EBIT/总资产
        CASE WHEN b.total_assets > 0 AND i.ebit IS NOT NULL
             THEN i.ebit / b.total_assets
             ELSE NULL END as x3,
        -- X4: 市值/总负债
        CASE WHEN b.total_liab > 0
             THEN d.total_mv * 10000 / b.total_liab
             ELSE NULL END as x4,
        -- X5: 营收/总资产
        CASE WHEN b.total_assets > 0
             THEN i.total_revenue / b.total_assets
             ELSE NULL END as x5
    FROM (
        SELECT *, ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY end_date DESC) as rn
        FROM balancesheet WHERE report_type = '1'
    ) b
    LEFT JOIN (
        SELECT *, ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY end_date DESC) as rn
        FROM income WHERE report_type = '1'
    ) i ON b.ts_code = i.ts_code AND i.rn = 1
    LEFT JOIN (
        SELECT *, ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY trade_date DESC) as rn
        FROM daily_basic
    ) d ON b.ts_code = d.ts_code AND d.rn = 1
    WHERE b.rn = 1
)
SELECT
    ts_code,
    x1, x2, x3, x4, x5,
    1.2 * COALESCE(x1, 0) +
    1.4 * COALESCE(x2, 0) +
    3.3 * COALESCE(x3, 0) +
    0.6 * COALESCE(x4, 0) +
    1.0 * COALESCE(x5, 0) as z_score
FROM zscore_calc;
```

---

### 2.5 分红因子 (Dividend Factors)

#### 2.5.1 股息相关指标
| 因子名称 | 数据来源 | 字段/计算 | 说明 |
|----------|----------|-----------|------|
| 股息率(TTM) | daily_basic | dv_ttm | 近12个月股息率 |
| 股息率(LYR) | daily_basic | dv_ratio | 近一年股息率 |
| 现金分红 | dividend | cash_div | 每10股派现(元) |
| 股票股利 | dividend | stk_div | 每10股送股(股) |

**注意**: 当前数据库中 `dividend` 表的 `cash_div` 数据可能需要更新，建议使用 `daily_basic` 中的 `dv_ttm` 作为主要股息率来源。

#### 2.5.2 分红连续性计算
```sql
-- 分红连续性分析 (基于daily_basic的股息率)
WITH yearly_dividend AS (
    SELECT
        ts_code,
        SUBSTR(trade_date, 1, 4) as year,
        MAX(dv_ttm) as max_dividend_yield
    FROM daily_basic
    WHERE dv_ttm > 0
    GROUP BY ts_code, SUBSTR(trade_date, 1, 4)
),
dividend_continuity AS (
    SELECT
        ts_code,
        COUNT(DISTINCT year) as dividend_years,
        MIN(year) as first_dividend_year,
        MAX(year) as last_dividend_year,
        AVG(max_dividend_yield) as avg_dividend_yield
    FROM yearly_dividend
    WHERE year >= '2020'
    GROUP BY ts_code
)
SELECT
    ts_code,
    dividend_years,
    first_dividend_year,
    last_dividend_year,
    avg_dividend_yield,
    -- 连续性评分: 近5年分红年数
    CASE
        WHEN dividend_years >= 5 THEN '优秀'
        WHEN dividend_years >= 3 THEN '良好'
        WHEN dividend_years >= 1 THEN '一般'
        ELSE '无分红'
    END as dividend_continuity_rating
FROM dividend_continuity;
```

#### 2.5.3 分红派息率
```sql
-- 派息率 = 每股股利 / 每股收益
WITH dividend_payout AS (
    SELECT
        f.ts_code,
        f.end_date,
        f.eps,
        d.dv_ttm,
        d.close,
        -- 派息率计算: 股息率 * 股价 / EPS
        CASE WHEN f.eps > 0 AND d.close > 0
             THEN (d.dv_ttm / 100 * d.close) / f.eps * 100
             ELSE NULL
        END as payout_ratio
    FROM (
        SELECT ts_code, end_date, eps,
               ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY end_date DESC) as rn
        FROM fina_indicator_vip
        WHERE end_date LIKE '%1231'  -- 年报EPS
    ) f
    JOIN (
        SELECT ts_code, dv_ttm, close,
               ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY trade_date DESC) as rn
        FROM daily_basic
    ) d ON f.ts_code = d.ts_code AND d.rn = 1
    WHERE f.rn = 1
)
SELECT * FROM dividend_payout WHERE payout_ratio IS NOT NULL;
```

---

## 3. 筛选策略

### 3.1 价值策略 (Value Strategy)

寻找被低估的优质股票。

```sql
-- 价值投资策略筛选
WITH latest_data AS (
    SELECT
        d.ts_code,
        s.name,
        s.industry,
        d.pe_ttm,
        d.pb,
        d.ps_ttm,
        d.dv_ttm,
        d.total_mv / 10000 as market_cap_billion,  -- 亿元
        f.roe,
        f.netprofit_yoy
    FROM daily_basic d
    JOIN stock_basic s ON d.ts_code = s.ts_code
    LEFT JOIN (
        SELECT ts_code, roe, netprofit_yoy,
               ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY end_date DESC) as rn
        FROM fina_indicator_vip
    ) f ON d.ts_code = f.ts_code AND f.rn = 1
    WHERE d.trade_date = (SELECT MAX(trade_date) FROM daily_basic)
      AND s.list_status = 'L'
)
SELECT
    ts_code,
    name,
    industry,
    pe_ttm,
    pb,
    ps_ttm,
    dv_ttm,
    roe,
    market_cap_billion
FROM latest_data
WHERE
    -- 估值条件
    pe_ttm > 0 AND pe_ttm < 15           -- PE < 15
    AND pb > 0 AND pb < 2                 -- PB < 2
    AND dv_ttm > 2                        -- 股息率 > 2%
    -- 质量条件
    AND roe > 10                          -- ROE > 10%
    AND netprofit_yoy > 0                 -- 净利润正增长
    -- 规模条件
    AND market_cap_billion > 50           -- 市值 > 50亿
ORDER BY dv_ttm DESC, pe_ttm ASC
LIMIT 50;
```

### 3.2 成长策略 (Growth Strategy)

寻找高成长潜力股票。

```sql
-- 成长投资策略筛选
WITH growth_stocks AS (
    SELECT
        d.ts_code,
        s.name,
        s.industry,
        d.pe_ttm,
        d.total_mv / 10000 as market_cap_billion,
        f.tr_yoy as revenue_growth,
        f.netprofit_yoy as profit_growth,
        f.roe,
        f.gross_margin,
        -- PEG计算
        CASE WHEN f.netprofit_yoy > 5 AND d.pe_ttm > 0
             THEN d.pe_ttm / f.netprofit_yoy
             ELSE NULL
        END as peg
    FROM daily_basic d
    JOIN stock_basic s ON d.ts_code = s.ts_code
    LEFT JOIN (
        SELECT ts_code, tr_yoy, netprofit_yoy, roe, gross_margin,
               ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY end_date DESC) as rn
        FROM fina_indicator_vip
    ) f ON d.ts_code = f.ts_code AND f.rn = 1
    WHERE d.trade_date = (SELECT MAX(trade_date) FROM daily_basic)
      AND s.list_status = 'L'
)
SELECT
    ts_code,
    name,
    industry,
    pe_ttm,
    revenue_growth,
    profit_growth,
    peg,
    roe,
    gross_margin,
    market_cap_billion
FROM growth_stocks
WHERE
    -- 成长条件
    revenue_growth > 20                    -- 营收增长 > 20%
    AND profit_growth > 25                 -- 净利润增长 > 25%
    -- 质量条件
    AND roe > 12                           -- ROE > 12%
    AND gross_margin > 30                  -- 毛利率 > 30%
    -- 估值合理
    AND peg > 0 AND peg < 1.5              -- PEG < 1.5
    AND pe_ttm > 0 AND pe_ttm < 60         -- PE 合理区间
    -- 规模条件
    AND market_cap_billion > 30            -- 市值 > 30亿
ORDER BY profit_growth DESC, peg ASC
LIMIT 50;
```

### 3.3 GARP策略 (Growth at Reasonable Price)

平衡成长与估值，寻找价格合理的成长股。

```sql
-- GARP策略筛选
WITH garp_candidates AS (
    SELECT
        d.ts_code,
        s.name,
        s.industry,
        d.pe_ttm,
        d.pb,
        d.dv_ttm,
        d.total_mv / 10000 as market_cap_billion,
        f.tr_yoy as revenue_growth,
        f.netprofit_yoy as profit_growth,
        f.roe,
        f.roe_waa,
        f.gross_margin,
        f.netprofit_margin,
        -- PEG
        CASE WHEN f.netprofit_yoy > 0 AND d.pe_ttm > 0
             THEN d.pe_ttm / f.netprofit_yoy
             ELSE NULL
        END as peg,
        -- 综合评分
        (COALESCE(f.roe, 0) * 0.3 +
         COALESCE(f.netprofit_yoy, 0) * 0.3 +
         COALESCE(f.tr_yoy, 0) * 0.2 +
         COALESCE(d.dv_ttm, 0) * 2) as quality_score
    FROM daily_basic d
    JOIN stock_basic s ON d.ts_code = s.ts_code
    LEFT JOIN (
        SELECT ts_code, tr_yoy, netprofit_yoy, roe, roe_waa,
               gross_margin, netprofit_margin,
               ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY end_date DESC) as rn
        FROM fina_indicator_vip
    ) f ON d.ts_code = f.ts_code AND f.rn = 1
    WHERE d.trade_date = (SELECT MAX(trade_date) FROM daily_basic)
      AND s.list_status = 'L'
)
SELECT
    ts_code,
    name,
    industry,
    pe_ttm,
    pb,
    dv_ttm,
    revenue_growth,
    profit_growth,
    peg,
    roe,
    gross_margin,
    quality_score,
    market_cap_billion
FROM garp_candidates
WHERE
    -- GARP核心条件
    peg > 0 AND peg <= 1                   -- PEG <= 1 (核心)
    AND pe_ttm > 5 AND pe_ttm < 30         -- PE在合理区间
    -- 成长要求
    AND profit_growth >= 15                 -- 适度增长 >= 15%
    AND revenue_growth >= 10                -- 营收增长 >= 10%
    -- 质量要求
    AND roe >= 12                           -- ROE >= 12%
    AND gross_margin >= 25                  -- 毛利率 >= 25%
    -- 规模要求
    AND market_cap_billion >= 50            -- 市值 >= 50亿
ORDER BY peg ASC, quality_score DESC
LIMIT 50;
```

### 3.4 质量策略 (Quality Strategy)

寻找财务稳健、盈利能力强的高质量公司。

```sql
-- 质量投资策略筛选
WITH quality_stocks AS (
    SELECT
        d.ts_code,
        s.name,
        s.industry,
        d.pe_ttm,
        d.pb,
        d.dv_ttm,
        d.total_mv / 10000 as market_cap_billion,
        f.roe,
        f.roe_waa,
        f.roe_dt,
        f.roa,
        f.gross_margin,
        f.netprofit_margin,
        f.current_ratio,
        f.debt_to_assets,
        f.ocfps,
        f.netprofit_yoy,
        f.tr_yoy
    FROM daily_basic d
    JOIN stock_basic s ON d.ts_code = s.ts_code
    LEFT JOIN (
        SELECT ts_code, roe, roe_waa, roe_dt, roa, gross_margin,
               netprofit_margin, current_ratio, debt_to_assets,
               ocfps, netprofit_yoy, tr_yoy,
               ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY end_date DESC) as rn
        FROM fina_indicator_vip
    ) f ON d.ts_code = f.ts_code AND f.rn = 1
    WHERE d.trade_date = (SELECT MAX(trade_date) FROM daily_basic)
      AND s.list_status = 'L'
),
-- 计算ROE稳定性 (近4期)
roe_stability AS (
    SELECT
        ts_code,
        AVG(roe) as avg_roe,
        STDDEV(roe) as std_roe,
        MIN(roe) as min_roe,
        COUNT(*) as periods
    FROM (
        SELECT ts_code, roe, end_date,
               ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY end_date DESC) as rn
        FROM fina_indicator_vip
        WHERE roe IS NOT NULL
    ) t
    WHERE rn <= 4
    GROUP BY ts_code
)
SELECT
    q.ts_code,
    q.name,
    q.industry,
    q.pe_ttm,
    q.pb,
    q.dv_ttm,
    q.roe,
    q.roe_dt,
    q.roa,
    q.gross_margin,
    q.netprofit_margin,
    q.debt_to_assets,
    q.current_ratio,
    q.netprofit_yoy,
    r.avg_roe,
    r.min_roe,
    q.market_cap_billion
FROM quality_stocks q
LEFT JOIN roe_stability r ON q.ts_code = r.ts_code
WHERE
    -- 盈利质量
    q.roe >= 15                            -- ROE >= 15%
    AND q.roe_dt >= 12                     -- 扣非ROE >= 12%
    AND r.min_roe >= 10                    -- 历史最低ROE >= 10%
    AND q.gross_margin >= 30               -- 毛利率 >= 30%
    AND q.netprofit_margin >= 10           -- 净利率 >= 10%
    -- 财务健康
    AND q.debt_to_assets < 60              -- 资产负债率 < 60%
    AND q.ocfps > 0                        -- 经营现金流为正
    -- 稳定增长
    AND q.netprofit_yoy > -10              -- 净利润不大幅下滑
    -- 规模要求
    AND q.market_cap_billion >= 100        -- 市值 >= 100亿
ORDER BY q.roe DESC, q.gross_margin DESC
LIMIT 50;
```

---

## 4. 综合筛选器

### 4.1 多因子评分系统

```sql
-- 综合多因子评分
WITH factor_data AS (
    SELECT
        d.ts_code,
        s.name,
        s.industry,
        im.l1_name as sector,
        d.pe_ttm,
        d.pb,
        d.ps_ttm,
        d.dv_ttm,
        d.total_mv / 10000 as market_cap_billion,
        f.roe,
        f.roe_dt,
        f.gross_margin,
        f.netprofit_margin,
        f.netprofit_yoy,
        f.tr_yoy,
        f.debt_to_assets,
        f.current_ratio,
        f.ocfps
    FROM daily_basic d
    JOIN stock_basic s ON d.ts_code = s.ts_code
    LEFT JOIN (
        SELECT ts_code, l1_name,
               ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY in_date DESC) as rn
        FROM index_member_all
        WHERE out_date IS NULL
    ) im ON d.ts_code = im.ts_code AND im.rn = 1
    LEFT JOIN (
        SELECT *,
               ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY end_date DESC) as rn
        FROM fina_indicator_vip
    ) f ON d.ts_code = f.ts_code AND f.rn = 1
    WHERE d.trade_date = (SELECT MAX(trade_date) FROM daily_basic)
      AND s.list_status = 'L'
),
factor_scores AS (
    SELECT
        ts_code,
        name,
        industry,
        sector,
        pe_ttm,
        pb,
        dv_ttm,
        roe,
        netprofit_yoy,
        tr_yoy,
        debt_to_assets,
        market_cap_billion,

        -- 估值得分 (0-100)
        CASE
            WHEN pe_ttm <= 0 OR pe_ttm > 100 THEN 0
            WHEN pe_ttm <= 10 THEN 100
            WHEN pe_ttm <= 15 THEN 80
            WHEN pe_ttm <= 20 THEN 60
            WHEN pe_ttm <= 30 THEN 40
            ELSE 20
        END as pe_score,

        CASE
            WHEN pb <= 0 OR pb > 10 THEN 0
            WHEN pb <= 1 THEN 100
            WHEN pb <= 1.5 THEN 80
            WHEN pb <= 2 THEN 60
            WHEN pb <= 3 THEN 40
            ELSE 20
        END as pb_score,

        -- 盈利得分 (0-100)
        CASE
            WHEN roe IS NULL OR roe <= 0 THEN 0
            WHEN roe >= 25 THEN 100
            WHEN roe >= 20 THEN 80
            WHEN roe >= 15 THEN 60
            WHEN roe >= 10 THEN 40
            ELSE 20
        END as roe_score,

        -- 成长得分 (0-100)
        CASE
            WHEN netprofit_yoy IS NULL THEN 50
            WHEN netprofit_yoy >= 50 THEN 100
            WHEN netprofit_yoy >= 30 THEN 80
            WHEN netprofit_yoy >= 15 THEN 60
            WHEN netprofit_yoy >= 0 THEN 40
            ELSE 20
        END as growth_score,

        -- 财务健康得分 (0-100)
        CASE
            WHEN debt_to_assets IS NULL THEN 50
            WHEN debt_to_assets <= 30 THEN 100
            WHEN debt_to_assets <= 45 THEN 80
            WHEN debt_to_assets <= 60 THEN 60
            WHEN debt_to_assets <= 75 THEN 40
            ELSE 20
        END as health_score,

        -- 分红得分 (0-100)
        CASE
            WHEN dv_ttm IS NULL OR dv_ttm <= 0 THEN 0
            WHEN dv_ttm >= 5 THEN 100
            WHEN dv_ttm >= 3 THEN 80
            WHEN dv_ttm >= 2 THEN 60
            WHEN dv_ttm >= 1 THEN 40
            ELSE 20
        END as dividend_score

    FROM factor_data
)
SELECT
    ts_code,
    name,
    industry,
    sector,
    pe_ttm,
    pb,
    dv_ttm,
    roe,
    netprofit_yoy,
    market_cap_billion,

    pe_score,
    pb_score,
    roe_score,
    growth_score,
    health_score,
    dividend_score,

    -- 综合得分 (可调整权重)
    ROUND(
        pe_score * 0.15 +
        pb_score * 0.10 +
        roe_score * 0.25 +
        growth_score * 0.20 +
        health_score * 0.15 +
        dividend_score * 0.15
    , 2) as total_score

FROM factor_scores
WHERE market_cap_billion >= 30  -- 最低市值要求
ORDER BY total_score DESC
LIMIT 100;
```

### 4.2 行业中性筛选

```sql
-- 行业中性选股 (各行业选Top N)
WITH industry_ranked AS (
    SELECT
        d.ts_code,
        s.name,
        s.industry,
        d.pe_ttm,
        d.pb,
        d.dv_ttm,
        f.roe,
        f.netprofit_yoy,
        d.total_mv / 10000 as market_cap_billion,
        -- 行业内排名 (按ROE)
        ROW_NUMBER() OVER (
            PARTITION BY s.industry
            ORDER BY f.roe DESC NULLS LAST
        ) as industry_rank,
        -- 行业内PE分位
        PERCENT_RANK() OVER (
            PARTITION BY s.industry
            ORDER BY d.pe_ttm
        ) as pe_percentile_in_industry
    FROM daily_basic d
    JOIN stock_basic s ON d.ts_code = s.ts_code
    LEFT JOIN (
        SELECT ts_code, roe, netprofit_yoy,
               ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY end_date DESC) as rn
        FROM fina_indicator_vip
    ) f ON d.ts_code = f.ts_code AND f.rn = 1
    WHERE d.trade_date = (SELECT MAX(trade_date) FROM daily_basic)
      AND s.list_status = 'L'
      AND d.pe_ttm > 0 AND d.pe_ttm < 200
      AND f.roe > 0
)
SELECT
    ts_code,
    name,
    industry,
    pe_ttm,
    pb,
    roe,
    netprofit_yoy,
    pe_percentile_in_industry,
    industry_rank,
    market_cap_billion
FROM industry_ranked
WHERE
    industry_rank <= 5                     -- 各行业前5名
    AND pe_percentile_in_industry < 0.5    -- 行业内PE中位数以下
ORDER BY industry, industry_rank;
```

---

## 5. 视图定义

为便于日常使用，建议创建以下视图。

### 5.1 最新估值视图
```sql
CREATE OR REPLACE VIEW v_latest_valuation AS
SELECT
    d.ts_code,
    s.name,
    s.industry,
    d.trade_date,
    d.close,
    d.pe_ttm,
    d.pb,
    d.ps_ttm,
    d.dv_ttm,
    d.total_mv / 10000 as market_cap_billion,
    d.circ_mv / 10000 as float_cap_billion
FROM daily_basic d
JOIN stock_basic s ON d.ts_code = s.ts_code
WHERE d.trade_date = (SELECT MAX(trade_date) FROM daily_basic)
  AND s.list_status = 'L';
```

### 5.2 最新财务指标视图
```sql
CREATE OR REPLACE VIEW v_latest_financials AS
SELECT
    f.ts_code,
    s.name,
    f.end_date,
    f.roe,
    f.roe_waa,
    f.roe_dt,
    f.roa,
    f.gross_margin,
    f.netprofit_margin,
    f.netprofit_yoy,
    f.tr_yoy,
    f.debt_to_assets,
    f.current_ratio,
    f.quick_ratio,
    f.ocfps
FROM (
    SELECT *,
           ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY end_date DESC) as rn
    FROM fina_indicator_vip
) f
JOIN stock_basic s ON f.ts_code = s.ts_code
WHERE f.rn = 1
  AND s.list_status = 'L';
```

### 5.3 综合筛选视图
```sql
CREATE OR REPLACE VIEW v_stock_screener AS
SELECT
    v.ts_code,
    v.name,
    v.industry,
    v.close,
    v.pe_ttm,
    v.pb,
    v.dv_ttm,
    v.market_cap_billion,
    f.roe,
    f.netprofit_margin,
    f.netprofit_yoy,
    f.tr_yoy,
    f.debt_to_assets,
    -- PEG
    CASE WHEN f.netprofit_yoy > 0 AND v.pe_ttm > 0
         THEN ROUND(v.pe_ttm / f.netprofit_yoy, 2)
         ELSE NULL
    END as peg
FROM v_latest_valuation v
LEFT JOIN v_latest_financials f ON v.ts_code = f.ts_code;
```

---

## 6. 使用示例

### 6.1 快速筛选低估值高股息股
```sql
SELECT * FROM v_stock_screener
WHERE pe_ttm > 0 AND pe_ttm < 12
  AND pb < 1.5
  AND dv_ttm > 3
  AND roe > 10
  AND market_cap_billion > 50
ORDER BY dv_ttm DESC;
```

### 6.2 快速筛选成长股
```sql
SELECT * FROM v_stock_screener
WHERE netprofit_yoy > 30
  AND tr_yoy > 20
  AND roe > 15
  AND peg > 0 AND peg < 1
ORDER BY netprofit_yoy DESC;
```

### 6.3 快速筛选质量股
```sql
SELECT * FROM v_stock_screener
WHERE roe > 20
  AND netprofit_margin > 15
  AND debt_to_assets < 50
  AND market_cap_billion > 100
ORDER BY roe DESC;
```

---

## 7. 注意事项

### 7.1 数据质量
1. **财务数据时效性**: `fina_indicator_vip` 数据以财报披露为准，存在1-4个月的滞后
2. **行业分类**: 使用申万行业分类(`index_member_all`)，注意股票可能调入调出
3. **估值指标**: 负PE/PB需特殊处理，通常表示亏损或净资产为负
4. **分红数据**: `dividend`表可能需要更新，建议优先使用`daily_basic.dv_ttm`

### 7.2 筛选建议
1. **避免单一指标**: 多因子组合筛选更可靠
2. **行业差异**: 不同行业的合理估值区间不同，建议行业内比较
3. **周期性行业**: PE/PB在周期底部可能失真，需结合PB和周期位置判断
4. **成长陷阱**: 高增长需结合质量指标，避免业绩不可持续的公司

### 7.3 策略适用场景
| 策略 | 适用市场环境 | 适合投资者 |
|------|-------------|-----------|
| 价值策略 | 熊市/震荡市 | 保守型/长期投资者 |
| 成长策略 | 牛市/结构性行情 | 积极型投资者 |
| GARP策略 | 任何市场 | 平衡型投资者 |
| 质量策略 | 不确定性高时期 | 稳健型投资者 |

---

## 8. 附录：关键字段映射

### 8.1 fina_indicator_vip 核心字段
| 字段 | 含义 | 单位 |
|------|------|------|
| roe | 净资产收益率 | % |
| roe_waa | 加权净资产收益率 | % |
| roe_dt | 扣非净资产收益率 | % |
| roa | 总资产收益率 | % |
| gross_margin | 销售毛利率 | % |
| netprofit_margin | 销售净利率 | % |
| netprofit_yoy | 归母净利润同比增长 | % |
| tr_yoy | 营业总收入同比增长 | % |
| debt_to_assets | 资产负债率 | % |
| current_ratio | 流动比率 | 倍 |
| ocfps | 每股经营现金流 | 元 |

### 8.2 daily_basic 核心字段
| 字段 | 含义 | 单位 |
|------|------|------|
| pe_ttm | 滚动市盈率 | 倍 |
| pb | 市净率 | 倍 |
| ps_ttm | 滚动市销率 | 倍 |
| dv_ttm | 滚动股息率 | % |
| total_mv | 总市值 | 万元 |
| circ_mv | 流通市值 | 万元 |
| turnover_rate_f | 换手率(自由流通) | % |

---

*文档版本: V2.0*
*生成日期: 2026-01-31*
*数据库: Tushare-DuckDB*
