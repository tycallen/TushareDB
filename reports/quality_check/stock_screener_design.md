# 股票基本面筛选系统设计文档

## 一、系统概述

本系统基于 DuckDB 数据库设计，提供全面的股票基本面筛选功能。通过多维度因子分析，帮助投资者从估值、盈利能力、成长性、财务健康和分红等角度筛选优质股票。

### 1.1 数据源

- **数据库**: `/Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db`
- **数据范围**: 1991年至今，覆盖5477只上市股票
- **核心数据表**:
  - `stock_basic`: 股票基本信息
  - `daily_basic`: 每日指标 (PE, PB, PS, 市值等)
  - `fina_indicator_vip`: 财务指标 (ROE, 毛利率等)
  - `income`: 利润表
  - `balancesheet`: 资产负债表
  - `cashflow`: 现金流量表
  - `dividend`: 分红数据
  - `index_member_all`: 申万行业分类

---

## 二、因子体系设计

### 2.1 估值因子

| 因子名称 | 计算公式 | 说明 |
|---------|---------|------|
| PE_TTM | daily_basic.pe_ttm | 市盈率(TTM) |
| PB | daily_basic.pb | 市净率 |
| PS_TTM | daily_basic.ps_ttm | 市销率(TTM) |
| PCF | total_mv / n_cashflow_act | 市现率 |
| PE_Percentile | PERCENT_RANK() OVER (PARTITION BY ts_code ORDER BY pe_ttm) | PE历史分位数 |
| PB_Percentile | PERCENT_RANK() OVER (PARTITION BY ts_code ORDER BY pb) | PB历史分位数 |
| PEG | pe_ttm / netprofit_yoy | 市盈率相对增长比率 |
| EV/EBITDA | (total_mv + total_liab - money_cap) / EBITDA | 企业价值倍数 |

### 2.2 盈利能力因子

| 因子名称 | 计算公式 | 说明 |
|---------|---------|------|
| ROE | fina_indicator_vip.roe | 净资产收益率(%) |
| ROE_TTM | fina_indicator_vip.roe_yearly | 年化ROE |
| ROA | fina_indicator_vip.roa | 总资产收益率(%) |
| ROIC | fina_indicator_vip.roic | 投入资本回报率(%) |
| Gross_Margin | fina_indicator_vip.gross_margin | 毛利率(%) |
| Net_Profit_Margin | fina_indicator_vip.netprofit_margin | 净利率(%) |
| Expense_Ratio | (sell_exp + admin_exp) / revenue | 费用率 |
| ROE_Stability | STDDEV(roe) OVER 3Y | ROE标准差(稳定性) |

### 2.3 成长因子

| 因子名称 | 计算公式 | 说明 |
|---------|---------|------|
| Revenue_YoY | fina_indicator_vip.or_yoy / fina_indicator_vip.tr_yoy | 营收同比增长率(%) |
| NetProfit_YoY | fina_indicator_vip.netprofit_yoy | 净利润同比增长率(%) |
| Revenue_CAGR_3Y | (revenue_t / revenue_t-3)^(1/3) - 1 | 3年营收复合增长率 |
| NetProfit_CAGR_3Y | (netprofit_t / netprofit_t-3)^(1/3) - 1 | 3年净利润复合增长率 |
| Growth_Acceleration | netprofit_yoy_t - netprofit_yoy_t-1 | 增速加速度 |
| Q_Sales_YoY | fina_indicator_vip.q_sales_yoy | 单季度营收同比(%) |

### 2.4 财务健康因子

| 因子名称 | 计算公式 | 说明 |
|---------|---------|------|
| Debt_to_Assets | fina_indicator_vip.debt_to_assets | 资产负债率(%) |
| Current_Ratio | fina_indicator_vip.current_ratio | 流动比率 |
| Quick_Ratio | fina_indicator_vip.quick_ratio | 速动比率 |
| Cash_Ratio | fina_indicator_vip.cash_ratio | 现金比率 |
| Interest_Coverage | EBIT / int_exp | 利息覆盖倍数 |
| OCF_to_NetProfit | n_cashflow_act / n_income | 经营现金流/净利润 |
| Debt_to_Equity | fina_indicator_vip.debt_to_eqt | 产权比率 |

### 2.5 分红因子

| 因子名称 | 计算公式 | 说明 |
|---------|---------|------|
| Dividend_Yield | daily_basic.dv_ttm | 股息率(TTM) |
| Dividend_Payout | cash_div / EPS | 股利支付率 |
| Dividend_Continuity | COUNT(DISTINCT year with dividend) | 连续分红年数 |
| Dividend_Growth | (div_t / div_t-3)^(1/3) - 1 | 3年股息复合增长率 |

---

## 三、SQL 实现

### 3.1 基础因子计算 SQL

```sql
-- =====================================================
-- 创建基础因子视图
-- =====================================================

-- 1. 获取最新财务指标
CREATE OR REPLACE VIEW v_latest_fina AS
SELECT
    f.*,
    ROW_NUMBER() OVER (PARTITION BY f.ts_code ORDER BY f.end_date DESC) as rn
FROM fina_indicator_vip f
WHERE f.end_date >= '20230101'  -- 只取最近2年数据
  AND f.update_flag = '1';      -- 只取最新更新的记录

-- 2. 获取最新每日指标
CREATE OR REPLACE VIEW v_latest_daily AS
SELECT
    d.*,
    ROW_NUMBER() OVER (PARTITION BY d.ts_code ORDER BY d.trade_date DESC) as rn
FROM daily_basic d
WHERE d.trade_date >= '20250101';

-- 3. 估值因子计算
CREATE OR REPLACE VIEW v_valuation_factors AS
WITH latest_fina AS (
    SELECT * FROM v_latest_fina WHERE rn = 1
),
latest_daily AS (
    SELECT * FROM v_latest_daily WHERE rn = 1
),
pe_history AS (
    SELECT
        ts_code,
        pe_ttm,
        PERCENT_RANK() OVER (PARTITION BY ts_code ORDER BY pe_ttm) as pe_percentile
    FROM daily_basic
    WHERE pe_ttm > 0 AND pe_ttm < 500
      AND trade_date >= '20200101'
),
pb_history AS (
    SELECT
        ts_code,
        pb,
        PERCENT_RANK() OVER (PARTITION BY ts_code ORDER BY pb) as pb_percentile
    FROM daily_basic
    WHERE pb > 0 AND pb < 50
      AND trade_date >= '20200101'
)
SELECT
    d.ts_code,
    d.trade_date,
    d.pe_ttm,
    d.pb,
    d.ps_ttm,
    d.total_mv,
    d.circ_mv,
    d.dv_ttm as dividend_yield,
    -- PEG: PE / 盈利增长率 (增长率为正时)
    CASE
        WHEN f.netprofit_yoy > 0 AND d.pe_ttm > 0
        THEN d.pe_ttm / f.netprofit_yoy
        ELSE NULL
    END as peg,
    -- PE历史分位数
    pe_h.pe_percentile,
    -- PB历史分位数
    pb_h.pb_percentile
FROM latest_daily d
LEFT JOIN latest_fina f ON d.ts_code = f.ts_code
LEFT JOIN (SELECT DISTINCT ON (ts_code) * FROM pe_history ORDER BY ts_code, pe_ttm DESC) pe_h
    ON d.ts_code = pe_h.ts_code
LEFT JOIN (SELECT DISTINCT ON (ts_code) * FROM pb_history ORDER BY ts_code, pb DESC) pb_h
    ON d.ts_code = pb_h.ts_code
WHERE d.rn = 1;

-- 4. 盈利能力因子计算
CREATE OR REPLACE VIEW v_profitability_factors AS
WITH latest_fina AS (
    SELECT * FROM v_latest_fina WHERE rn = 1
),
roe_stability AS (
    SELECT
        ts_code,
        STDDEV(roe) as roe_std,
        AVG(roe) as roe_avg
    FROM fina_indicator_vip
    WHERE end_date >= '20210101'
      AND roe IS NOT NULL
      AND roe BETWEEN -100 AND 100
    GROUP BY ts_code
    HAVING COUNT(*) >= 4
)
SELECT
    f.ts_code,
    f.end_date,
    f.roe,
    f.roe_yearly as roe_ttm,
    f.roa,
    f.roa_yearly,
    f.roic,
    f.gross_margin,
    f.netprofit_margin,
    f.expense_of_sales as expense_ratio,
    f.op_of_gr as operating_margin,
    s.roe_std,
    s.roe_avg,
    -- ROE变异系数 (越小越稳定)
    CASE WHEN s.roe_avg > 0 THEN s.roe_std / s.roe_avg ELSE NULL END as roe_cv
FROM latest_fina f
LEFT JOIN roe_stability s ON f.ts_code = s.ts_code
WHERE f.rn = 1;

-- 5. 成长因子计算
CREATE OR REPLACE VIEW v_growth_factors AS
WITH latest_fina AS (
    SELECT * FROM v_latest_fina WHERE rn = 1
),
fina_3y_ago AS (
    SELECT
        f.*,
        ROW_NUMBER() OVER (PARTITION BY f.ts_code ORDER BY f.end_date DESC) as rn
    FROM fina_indicator_vip f
    WHERE f.end_date BETWEEN '20210101' AND '20211231'
),
growth_accel AS (
    SELECT
        a.ts_code,
        a.netprofit_yoy as growth_rate_current,
        b.netprofit_yoy as growth_rate_prev,
        a.netprofit_yoy - b.netprofit_yoy as growth_acceleration
    FROM (SELECT * FROM v_latest_fina WHERE rn = 1) a
    LEFT JOIN (SELECT * FROM v_latest_fina WHERE rn = 2) b ON a.ts_code = b.ts_code
)
SELECT
    f.ts_code,
    f.end_date,
    f.or_yoy as revenue_yoy,
    f.tr_yoy as total_revenue_yoy,
    f.netprofit_yoy,
    f.dt_netprofit_yoy as deducted_netprofit_yoy,
    f.q_sales_yoy,
    f.basic_eps_yoy,
    -- 3年营收CAGR (需要3年前数据)
    CASE
        WHEN f3.revenue_ps > 0 AND f.revenue_ps > 0
        THEN POWER(f.revenue_ps / f3.revenue_ps, 1.0/3) - 1
        ELSE NULL
    END as revenue_cagr_3y,
    -- 增速加速度
    g.growth_acceleration
FROM latest_fina f
LEFT JOIN (SELECT * FROM fina_3y_ago WHERE rn = 1) f3 ON f.ts_code = f3.ts_code
LEFT JOIN growth_accel g ON f.ts_code = g.ts_code
WHERE f.rn = 1;

-- 6. 财务健康因子计算
CREATE OR REPLACE VIEW v_financial_health AS
WITH latest_fina AS (
    SELECT * FROM v_latest_fina WHERE rn = 1
),
latest_cashflow AS (
    SELECT
        c.*,
        ROW_NUMBER() OVER (PARTITION BY c.ts_code ORDER BY c.end_date DESC) as rn
    FROM cashflow c
    WHERE c.report_type = '1'
      AND c.end_date >= '20230101'
),
latest_income AS (
    SELECT
        i.*,
        ROW_NUMBER() OVER (PARTITION BY i.ts_code ORDER BY i.end_date DESC) as rn
    FROM income i
    WHERE i.report_type = '1'
      AND i.end_date >= '20230101'
)
SELECT
    f.ts_code,
    f.end_date,
    f.debt_to_assets,
    f.current_ratio,
    f.quick_ratio,
    f.cash_ratio,
    f.debt_to_eqt as debt_to_equity,
    f.ocf_to_debt,
    -- 经营现金流/净利润
    CASE
        WHEN i.n_income > 0 THEN c.n_cashflow_act / i.n_income
        ELSE NULL
    END as ocf_to_netprofit,
    -- 利息覆盖倍数 (近似计算)
    CASE
        WHEN f.finaexp_of_gr > 0 THEN f.ebit_of_gr / f.finaexp_of_gr
        ELSE NULL
    END as interest_coverage
FROM latest_fina f
LEFT JOIN (SELECT * FROM latest_cashflow WHERE rn = 1) c ON f.ts_code = c.ts_code
LEFT JOIN (SELECT * FROM latest_income WHERE rn = 1) i ON f.ts_code = i.ts_code
WHERE f.rn = 1;

-- 7. 分红因子计算
CREATE OR REPLACE VIEW v_dividend_factors AS
WITH latest_daily AS (
    SELECT * FROM v_latest_daily WHERE rn = 1
),
dividend_history AS (
    SELECT
        ts_code,
        SUBSTR(end_date, 1, 4) as div_year,
        SUM(cash_div) as annual_cash_div,
        COUNT(*) as div_count
    FROM dividend
    WHERE cash_div > 0
      AND div_proc = '实施'
    GROUP BY ts_code, SUBSTR(end_date, 1, 4)
),
dividend_continuity AS (
    SELECT
        ts_code,
        COUNT(DISTINCT div_year) as continuous_years,
        MAX(div_year) as last_div_year
    FROM dividend_history
    WHERE div_year >= '2019'
    GROUP BY ts_code
),
dividend_cagr AS (
    SELECT
        a.ts_code,
        CASE
            WHEN b.annual_cash_div > 0
            THEN POWER(a.annual_cash_div / b.annual_cash_div, 1.0/3) - 1
            ELSE NULL
        END as div_cagr_3y
    FROM (SELECT * FROM dividend_history WHERE div_year = '2024') a
    LEFT JOIN (SELECT * FROM dividend_history WHERE div_year = '2021') b
        ON a.ts_code = b.ts_code
)
SELECT
    d.ts_code,
    d.dv_ttm as dividend_yield,
    dc.continuous_years,
    dc.last_div_year,
    dcagr.div_cagr_3y
FROM latest_daily d
LEFT JOIN dividend_continuity dc ON d.ts_code = dc.ts_code
LEFT JOIN dividend_cagr dcagr ON d.ts_code = dcagr.ts_code
WHERE d.rn = 1;
```

### 3.2 行业中性化处理

```sql
-- =====================================================
-- 行业中性化因子计算
-- =====================================================

-- 获取行业分类
CREATE OR REPLACE VIEW v_industry AS
SELECT DISTINCT ON (ts_code)
    ts_code,
    l1_name as industry_l1,
    l2_name as industry_l2,
    l3_name as industry_l3
FROM index_member_all
WHERE out_date IS NULL  -- 当前有效的行业分类
ORDER BY ts_code, in_date DESC;

-- 行业中性化PE
CREATE OR REPLACE VIEW v_pe_industry_neutral AS
WITH pe_data AS (
    SELECT
        d.ts_code,
        d.pe_ttm,
        i.industry_l1
    FROM (SELECT * FROM v_latest_daily WHERE rn = 1) d
    JOIN v_industry i ON d.ts_code = i.ts_code
    WHERE d.pe_ttm > 0 AND d.pe_ttm < 500
),
industry_stats AS (
    SELECT
        industry_l1,
        AVG(pe_ttm) as industry_pe_avg,
        STDDEV(pe_ttm) as industry_pe_std
    FROM pe_data
    GROUP BY industry_l1
    HAVING COUNT(*) >= 10
)
SELECT
    p.ts_code,
    p.pe_ttm,
    p.industry_l1,
    s.industry_pe_avg,
    s.industry_pe_std,
    -- Z-Score标准化
    CASE
        WHEN s.industry_pe_std > 0
        THEN (p.pe_ttm - s.industry_pe_avg) / s.industry_pe_std
        ELSE 0
    END as pe_zscore,
    -- 行业内排名分位
    PERCENT_RANK() OVER (
        PARTITION BY p.industry_l1
        ORDER BY p.pe_ttm
    ) as pe_industry_percentile
FROM pe_data p
JOIN industry_stats s ON p.industry_l1 = s.industry_l1;

-- 行业中性化ROE
CREATE OR REPLACE VIEW v_roe_industry_neutral AS
WITH roe_data AS (
    SELECT
        f.ts_code,
        f.roe,
        i.industry_l1
    FROM (SELECT * FROM v_latest_fina WHERE rn = 1) f
    JOIN v_industry i ON f.ts_code = i.ts_code
    WHERE f.roe IS NOT NULL AND f.roe BETWEEN -50 AND 100
),
industry_stats AS (
    SELECT
        industry_l1,
        AVG(roe) as industry_roe_avg,
        STDDEV(roe) as industry_roe_std
    FROM roe_data
    GROUP BY industry_l1
    HAVING COUNT(*) >= 10
)
SELECT
    r.ts_code,
    r.roe,
    r.industry_l1,
    s.industry_roe_avg,
    s.industry_roe_std,
    -- Z-Score标准化
    CASE
        WHEN s.industry_roe_std > 0
        THEN (r.roe - s.industry_roe_avg) / s.industry_roe_std
        ELSE 0
    END as roe_zscore,
    -- 行业内排名分位 (ROE越高越好)
    PERCENT_RANK() OVER (
        PARTITION BY r.industry_l1
        ORDER BY r.roe DESC
    ) as roe_industry_rank
FROM roe_data r
JOIN industry_stats s ON r.industry_l1 = s.industry_l1;
```

---

## 四、筛选策略实现

### 4.1 价值投资策略 (低PE低PB高股息)

```sql
-- =====================================================
-- 策略1: 价值投资策略
-- 条件: 低PE + 低PB + 高股息 + ROE稳定
-- =====================================================

WITH stock_info AS (
    SELECT ts_code, name, industry FROM stock_basic WHERE list_status = 'L'
),
latest_daily AS (
    SELECT d.*, ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY trade_date DESC) as rn
    FROM daily_basic d WHERE trade_date >= '20250101'
),
latest_fina AS (
    SELECT f.*, ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY end_date DESC) as rn
    FROM fina_indicator_vip f WHERE end_date >= '20240101'
),
industry_pe AS (
    SELECT
        i.l1_name as industry,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY d.pe_ttm) as median_pe
    FROM latest_daily d
    JOIN index_member_all i ON d.ts_code = i.ts_code AND i.out_date IS NULL
    WHERE d.rn = 1 AND d.pe_ttm > 0 AND d.pe_ttm < 200
    GROUP BY i.l1_name
)
SELECT
    s.ts_code,
    s.name,
    s.industry,
    d.pe_ttm,
    d.pb,
    d.dv_ttm as dividend_yield,
    d.total_mv / 10000 as market_cap_yi,
    f.roe,
    f.debt_to_assets,
    f.netprofit_yoy as profit_growth,
    ip.median_pe as industry_median_pe,
    -- 综合评分: 估值因子权重50% + 股息30% + 盈利稳定性20%
    ROUND(
        (1 - LEAST(d.pe_ttm, 50) / 50) * 0.25 +  -- PE越低越好
        (1 - LEAST(d.pb, 5) / 5) * 0.25 +         -- PB越低越好
        LEAST(d.dv_ttm, 10) / 10 * 0.30 +          -- 股息率越高越好
        LEAST(f.roe, 30) / 30 * 0.20               -- ROE适中即可
    , 4) as value_score
FROM stock_info s
JOIN (SELECT * FROM latest_daily WHERE rn = 1) d ON s.ts_code = d.ts_code
JOIN (SELECT * FROM latest_fina WHERE rn = 1) f ON s.ts_code = f.ts_code
LEFT JOIN industry_pe ip ON s.industry = ip.industry
WHERE
    -- 基础过滤
    d.total_mv >= 5000000  -- 市值>=50亿
    -- 估值条件
    AND d.pe_ttm > 0 AND d.pe_ttm < 20           -- PE < 20
    AND d.pb > 0 AND d.pb < 3                     -- PB < 3
    AND d.dv_ttm >= 2                             -- 股息率 >= 2%
    -- 盈利条件
    AND f.roe > 8                                 -- ROE > 8%
    AND f.debt_to_assets < 70                     -- 资产负债率 < 70%
    -- 排除亏损
    AND f.netprofit_yoy > -30                     -- 净利润同比 > -30%
ORDER BY value_score DESC
LIMIT 50;
```

### 4.2 成长投资策略 (高增长高ROE)

```sql
-- =====================================================
-- 策略2: 成长投资策略
-- 条件: 高增长 + 高ROE + 适度估值
-- =====================================================

WITH stock_info AS (
    SELECT ts_code, name, industry FROM stock_basic WHERE list_status = 'L'
),
latest_daily AS (
    SELECT d.*, ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY trade_date DESC) as rn
    FROM daily_basic d WHERE trade_date >= '20250101'
),
latest_fina AS (
    SELECT f.*, ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY end_date DESC) as rn
    FROM fina_indicator_vip f WHERE end_date >= '20240101'
),
prev_fina AS (
    SELECT f.*, ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY end_date DESC) as rn
    FROM fina_indicator_vip f
    WHERE end_date >= '20230101' AND end_date < '20240101'
)
SELECT
    s.ts_code,
    s.name,
    s.industry,
    d.pe_ttm,
    d.total_mv / 10000 as market_cap_yi,
    f.roe,
    f.netprofit_yoy as profit_growth,
    f.or_yoy as revenue_growth,
    f.q_sales_yoy as quarterly_revenue_growth,
    -- PEG
    CASE WHEN f.netprofit_yoy > 0 THEN ROUND(d.pe_ttm / f.netprofit_yoy, 2) ELSE NULL END as peg,
    -- 增速加速度
    f.netprofit_yoy - COALESCE(pf.netprofit_yoy, 0) as growth_acceleration,
    -- 综合评分: 成长因子权重60% + 盈利质量30% + 估值合理性10%
    ROUND(
        LEAST(f.netprofit_yoy, 100) / 100 * 0.30 +     -- 净利润增速
        LEAST(f.or_yoy, 100) / 100 * 0.15 +            -- 营收增速
        LEAST(f.roe, 40) / 40 * 0.30 +                  -- ROE
        (1 - LEAST(d.pe_ttm, 100) / 100) * 0.15 +       -- PE越低越好
        LEAST(f.gross_margin, 60) / 60 * 0.10           -- 毛利率
    , 4) as growth_score
FROM stock_info s
JOIN (SELECT * FROM latest_daily WHERE rn = 1) d ON s.ts_code = d.ts_code
JOIN (SELECT * FROM latest_fina WHERE rn = 1) f ON s.ts_code = f.ts_code
LEFT JOIN (SELECT * FROM prev_fina WHERE rn = 1) pf ON s.ts_code = pf.ts_code
WHERE
    -- 基础过滤
    d.total_mv >= 3000000  -- 市值>=30亿
    -- 成长条件
    AND f.netprofit_yoy >= 20                     -- 净利润增速 >= 20%
    AND f.or_yoy >= 15                            -- 营收增速 >= 15%
    -- 盈利质量
    AND f.roe >= 15                               -- ROE >= 15%
    AND f.gross_margin >= 20                      -- 毛利率 >= 20%
    -- 估值约束
    AND d.pe_ttm > 0 AND d.pe_ttm < 80            -- PE < 80 (成长股估值可以略高)
    -- PEG约束
    AND d.pe_ttm / NULLIF(f.netprofit_yoy, 0) < 2  -- PEG < 2
ORDER BY growth_score DESC
LIMIT 50;
```

### 4.3 GARP策略 (成长+估值平衡)

```sql
-- =====================================================
-- 策略3: GARP策略 (Growth at Reasonable Price)
-- 条件: 合理增长 + 合理估值 + PEG<1
-- =====================================================

WITH stock_info AS (
    SELECT ts_code, name, industry FROM stock_basic WHERE list_status = 'L'
),
latest_daily AS (
    SELECT d.*, ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY trade_date DESC) as rn
    FROM daily_basic d WHERE trade_date >= '20250101'
),
latest_fina AS (
    SELECT f.*, ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY end_date DESC) as rn
    FROM fina_indicator_vip f WHERE end_date >= '20240101'
)
SELECT
    s.ts_code,
    s.name,
    s.industry,
    d.pe_ttm,
    d.pb,
    d.total_mv / 10000 as market_cap_yi,
    f.roe,
    f.netprofit_yoy as profit_growth,
    f.or_yoy as revenue_growth,
    -- PEG (核心指标)
    ROUND(d.pe_ttm / NULLIF(f.netprofit_yoy, 0), 2) as peg,
    -- GARP得分: PEG越低越好,同时考虑ROE和增长稳定性
    ROUND(
        (1 - LEAST(d.pe_ttm / NULLIF(f.netprofit_yoy, 0), 2) / 2) * 0.40 +  -- PEG权重40%
        LEAST(f.roe, 30) / 30 * 0.25 +                                       -- ROE权重25%
        LEAST(f.netprofit_yoy, 50) / 50 * 0.20 +                             -- 增长率权重20%
        (1 - LEAST(f.debt_to_assets, 80) / 80) * 0.15                        -- 财务健康15%
    , 4) as garp_score
FROM stock_info s
JOIN (SELECT * FROM latest_daily WHERE rn = 1) d ON s.ts_code = d.ts_code
JOIN (SELECT * FROM latest_fina WHERE rn = 1) f ON s.ts_code = f.ts_code
WHERE
    -- 基础过滤
    d.total_mv >= 5000000  -- 市值>=50亿
    -- 成长条件 (温和增长)
    AND f.netprofit_yoy >= 10 AND f.netprofit_yoy <= 50  -- 10-50%适度增长
    AND f.or_yoy >= 5                                     -- 营收增长>=5%
    -- 估值条件
    AND d.pe_ttm > 0 AND d.pe_ttm < 40                    -- PE < 40
    -- PEG条件 (核心)
    AND d.pe_ttm / NULLIF(f.netprofit_yoy, 0) > 0
    AND d.pe_ttm / NULLIF(f.netprofit_yoy, 0) < 1.2       -- PEG < 1.2
    -- 盈利质量
    AND f.roe >= 12                                        -- ROE >= 12%
    AND f.debt_to_assets < 60                              -- 资产负债率 < 60%
ORDER BY garp_score DESC
LIMIT 50;
```

### 4.4 质量因子策略 (高ROE稳定增长)

```sql
-- =====================================================
-- 策略4: 质量因子策略
-- 条件: 高ROE + 稳定盈利 + 健康现金流 + 低负债
-- =====================================================

WITH stock_info AS (
    SELECT ts_code, name, industry FROM stock_basic WHERE list_status = 'L'
),
latest_daily AS (
    SELECT d.*, ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY trade_date DESC) as rn
    FROM daily_basic d WHERE trade_date >= '20250101'
),
latest_fina AS (
    SELECT f.*, ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY end_date DESC) as rn
    FROM fina_indicator_vip f WHERE end_date >= '20240101'
),
-- ROE稳定性 (过去3年)
roe_stability AS (
    SELECT
        ts_code,
        AVG(roe) as avg_roe,
        STDDEV(roe) as std_roe,
        MIN(roe) as min_roe,
        COUNT(*) as roe_count
    FROM fina_indicator_vip
    WHERE end_date >= '20210101'
      AND end_date LIKE '%1231'  -- 只取年报
      AND roe IS NOT NULL
    GROUP BY ts_code
    HAVING COUNT(*) >= 3
),
-- 获取最新现金流数据
latest_cashflow AS (
    SELECT
        c.ts_code,
        c.n_cashflow_act,
        ROW_NUMBER() OVER (PARTITION BY c.ts_code ORDER BY c.end_date DESC) as rn
    FROM cashflow c
    WHERE c.report_type = '1' AND c.end_date >= '20240101'
),
latest_income AS (
    SELECT
        i.ts_code,
        i.n_income_attr_p as net_profit,
        ROW_NUMBER() OVER (PARTITION BY i.ts_code ORDER BY i.end_date DESC) as rn
    FROM income i
    WHERE i.report_type = '1' AND i.end_date >= '20240101'
)
SELECT
    s.ts_code,
    s.name,
    s.industry,
    d.pe_ttm,
    d.pb,
    d.total_mv / 10000 as market_cap_yi,
    f.roe,
    rs.avg_roe as roe_3y_avg,
    rs.std_roe as roe_3y_std,
    ROUND(rs.std_roe / NULLIF(rs.avg_roe, 0), 2) as roe_cv,  -- 变异系数
    f.gross_margin,
    f.debt_to_assets,
    -- 经营现金流/净利润
    ROUND(cf.n_cashflow_act / NULLIF(inc.net_profit, 0), 2) as ocf_ratio,
    -- 质量得分
    ROUND(
        LEAST(f.roe, 35) / 35 * 0.30 +                                    -- ROE
        (1 - LEAST(rs.std_roe / NULLIF(rs.avg_roe, 0), 1)) * 0.20 +        -- ROE稳定性
        LEAST(f.gross_margin, 50) / 50 * 0.15 +                            -- 毛利率
        (1 - LEAST(f.debt_to_assets, 70) / 70) * 0.15 +                    -- 低负债
        LEAST(cf.n_cashflow_act / NULLIF(inc.net_profit, 0), 2) / 2 * 0.20 -- 现金流质量
    , 4) as quality_score
FROM stock_info s
JOIN (SELECT * FROM latest_daily WHERE rn = 1) d ON s.ts_code = d.ts_code
JOIN (SELECT * FROM latest_fina WHERE rn = 1) f ON s.ts_code = f.ts_code
JOIN roe_stability rs ON s.ts_code = rs.ts_code
LEFT JOIN (SELECT * FROM latest_cashflow WHERE rn = 1) cf ON s.ts_code = cf.ts_code
LEFT JOIN (SELECT * FROM latest_income WHERE rn = 1) inc ON s.ts_code = inc.ts_code
WHERE
    -- 基础过滤
    d.total_mv >= 5000000  -- 市值>=50亿
    -- 质量条件
    AND f.roe >= 15                               -- 当期ROE >= 15%
    AND rs.avg_roe >= 12                          -- 3年平均ROE >= 12%
    AND rs.min_roe >= 8                           -- 3年最低ROE >= 8%
    AND rs.std_roe / NULLIF(rs.avg_roe, 0) < 0.5  -- ROE变异系数 < 0.5 (稳定)
    -- 盈利质量
    AND f.gross_margin >= 25                      -- 毛利率 >= 25%
    -- 财务健康
    AND f.debt_to_assets < 55                     -- 资产负债率 < 55%
    -- 现金流质量
    AND cf.n_cashflow_act > 0                     -- 经营现金流为正
    AND cf.n_cashflow_act / NULLIF(inc.net_profit, 0) >= 0.7  -- 现金流/净利润 >= 0.7
    -- 估值适度
    AND d.pe_ttm > 0 AND d.pe_ttm < 50
ORDER BY quality_score DESC
LIMIT 50;
```

### 4.5 困境反转策略

```sql
-- =====================================================
-- 策略5: 困境反转策略
-- 条件: 业绩触底反弹 + 估值低位 + 行业地位稳固
-- =====================================================

WITH stock_info AS (
    SELECT ts_code, name, industry FROM stock_basic WHERE list_status = 'L'
),
latest_daily AS (
    SELECT d.*, ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY trade_date DESC) as rn
    FROM daily_basic d WHERE trade_date >= '20250101'
),
latest_fina AS (
    SELECT f.*, ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY end_date DESC) as rn
    FROM fina_indicator_vip f WHERE end_date >= '20240101'
),
prev_fina AS (
    SELECT f.*, ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY end_date DESC) as rn
    FROM fina_indicator_vip f
    WHERE end_date >= '20230101' AND end_date < '20240101'
),
-- PE历史分位数
pe_percentile AS (
    SELECT
        ts_code,
        pe_ttm,
        PERCENT_RANK() OVER (PARTITION BY ts_code ORDER BY pe_ttm) as pe_pct
    FROM daily_basic
    WHERE trade_date >= '20200101' AND pe_ttm > 0 AND pe_ttm < 200
),
latest_pe_pct AS (
    SELECT ts_code, pe_pct
    FROM (
        SELECT *, ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY pe_ttm DESC) as rn
        FROM pe_percentile
    ) t WHERE rn = 1
)
SELECT
    s.ts_code,
    s.name,
    s.industry,
    d.pe_ttm,
    d.pb,
    d.total_mv / 10000 as market_cap_yi,
    pp.pe_pct as pe_percentile,  -- PE历史分位
    f.roe,
    pf.roe as prev_roe,
    f.netprofit_yoy as profit_growth,
    pf.netprofit_yoy as prev_profit_growth,
    -- 业绩反转信号
    CASE
        WHEN f.netprofit_yoy > 0 AND pf.netprofit_yoy < 0 THEN '扭亏为盈'
        WHEN f.netprofit_yoy > pf.netprofit_yoy + 20 THEN '业绩大幅改善'
        WHEN f.roe > pf.roe + 5 THEN 'ROE显著提升'
        ELSE '温和改善'
    END as turnaround_signal,
    -- 困境反转得分
    ROUND(
        (1 - COALESCE(pp.pe_pct, 0.5)) * 0.25 +                              -- PE低位
        (1 - LEAST(d.pb, 5) / 5) * 0.15 +                                     -- PB低
        GREATEST(f.netprofit_yoy - COALESCE(pf.netprofit_yoy, 0), 0) / 100 * 0.30 +  -- 业绩改善
        GREATEST(f.roe - COALESCE(pf.roe, 0), 0) / 20 * 0.20 +                -- ROE改善
        (1 - LEAST(f.debt_to_assets, 80) / 80) * 0.10                         -- 财务安全
    , 4) as turnaround_score
FROM stock_info s
JOIN (SELECT * FROM latest_daily WHERE rn = 1) d ON s.ts_code = d.ts_code
JOIN (SELECT * FROM latest_fina WHERE rn = 1) f ON s.ts_code = f.ts_code
LEFT JOIN (SELECT * FROM prev_fina WHERE rn = 1) pf ON s.ts_code = pf.ts_code
LEFT JOIN latest_pe_pct pp ON s.ts_code = pp.ts_code
WHERE
    -- 基础过滤
    d.total_mv >= 3000000  -- 市值>=30亿
    -- 困境条件 (曾经困难)
    AND (
        pf.netprofit_yoy < 0                    -- 去年利润负增长
        OR pf.roe < 8                           -- 去年ROE较低
        OR pp.pe_pct < 0.3                      -- PE处于历史低位
    )
    -- 反转信号
    AND (
        f.netprofit_yoy > COALESCE(pf.netprofit_yoy, 0) + 15  -- 利润增速大幅改善
        OR (f.netprofit_yoy > 0 AND COALESCE(pf.netprofit_yoy, 0) < 0)  -- 扭亏
        OR f.roe > COALESCE(pf.roe, 0) + 3                     -- ROE改善
    )
    -- 估值安全边际
    AND d.pe_ttm > 0 AND d.pe_ttm < 30
    AND d.pb > 0 AND d.pb < 4
    -- 财务底线
    AND f.debt_to_assets < 70
ORDER BY turnaround_score DESC
LIMIT 50;
```

---

## 五、Python 实现

### 5.1 因子计算引擎

```python
"""
股票基本面筛选系统 - Python 实现
"""

import duckdb
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class ScreenerStrategy(Enum):
    """筛选策略枚举"""
    VALUE = "value"          # 价值投资
    GROWTH = "growth"        # 成长投资
    GARP = "garp"            # GARP策略
    QUALITY = "quality"      # 质量因子
    TURNAROUND = "turnaround" # 困境反转


@dataclass
class ScreenerConfig:
    """筛选配置"""
    min_market_cap: float = 5000000  # 最小市值(万元)
    max_pe: float = 50               # 最大PE
    min_roe: float = 8               # 最小ROE
    max_debt_ratio: float = 70       # 最大资产负债率
    min_dividend_yield: float = 0    # 最小股息率
    top_n: int = 50                  # 返回数量


class StockScreener:
    """股票筛选器"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = duckdb.connect(db_path, read_only=True)

    def __del__(self):
        if hasattr(self, 'conn'):
            self.conn.close()

    def get_latest_trade_date(self) -> str:
        """获取最新交易日"""
        sql = "SELECT MAX(trade_date) FROM daily_basic"
        return self.conn.execute(sql).fetchone()[0]

    def get_latest_report_date(self) -> str:
        """获取最新报告期"""
        sql = "SELECT MAX(end_date) FROM fina_indicator_vip"
        return self.conn.execute(sql).fetchone()[0]

    def calculate_valuation_factors(self, trade_date: Optional[str] = None) -> pd.DataFrame:
        """计算估值因子"""
        if trade_date is None:
            trade_date = self.get_latest_trade_date()

        sql = f"""
        WITH latest_daily AS (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY trade_date DESC) as rn
            FROM daily_basic WHERE trade_date <= '{trade_date}'
        ),
        latest_fina AS (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY end_date DESC) as rn
            FROM fina_indicator_vip WHERE end_date <= '{trade_date}'
        ),
        pe_percentile AS (
            SELECT
                ts_code,
                PERCENT_RANK() OVER (PARTITION BY ts_code ORDER BY pe_ttm) as pe_pct
            FROM daily_basic
            WHERE pe_ttm > 0 AND pe_ttm < 500
              AND trade_date BETWEEN strftime('{trade_date}'::DATE - INTERVAL '3 years', '%Y%m%d') AND '{trade_date}'
        )
        SELECT
            d.ts_code,
            s.name,
            s.industry,
            d.trade_date,
            d.pe_ttm,
            d.pb,
            d.ps_ttm,
            d.total_mv / 10000 as market_cap_yi,
            d.dv_ttm as dividend_yield,
            f.netprofit_yoy,
            CASE WHEN f.netprofit_yoy > 0 THEN d.pe_ttm / f.netprofit_yoy ELSE NULL END as peg,
            pp.pe_pct as pe_percentile
        FROM (SELECT * FROM latest_daily WHERE rn = 1) d
        JOIN stock_basic s ON d.ts_code = s.ts_code AND s.list_status = 'L'
        LEFT JOIN (SELECT * FROM latest_fina WHERE rn = 1) f ON d.ts_code = f.ts_code
        LEFT JOIN (SELECT DISTINCT ON (ts_code) * FROM pe_percentile ORDER BY ts_code, pe_pct DESC) pp
            ON d.ts_code = pp.ts_code
        WHERE d.pe_ttm > 0 AND d.total_mv > 0
        """
        return self.conn.execute(sql).df()

    def calculate_profitability_factors(self) -> pd.DataFrame:
        """计算盈利能力因子"""
        sql = """
        WITH latest_fina AS (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY end_date DESC) as rn
            FROM fina_indicator_vip WHERE end_date >= '20240101'
        ),
        roe_stability AS (
            SELECT
                ts_code,
                AVG(roe) as roe_avg,
                STDDEV(roe) as roe_std,
                MIN(roe) as roe_min,
                MAX(roe) as roe_max
            FROM fina_indicator_vip
            WHERE end_date >= '20210101' AND roe BETWEEN -50 AND 100
            GROUP BY ts_code
            HAVING COUNT(*) >= 4
        )
        SELECT
            f.ts_code,
            s.name,
            f.end_date,
            f.roe,
            f.roe_yearly as roe_ttm,
            f.roa,
            f.roic,
            f.gross_margin,
            f.netprofit_margin,
            f.expense_of_sales as expense_ratio,
            rs.roe_avg,
            rs.roe_std,
            rs.roe_min,
            rs.roe_max,
            CASE WHEN rs.roe_avg > 0 THEN rs.roe_std / rs.roe_avg ELSE NULL END as roe_cv
        FROM (SELECT * FROM latest_fina WHERE rn = 1) f
        JOIN stock_basic s ON f.ts_code = s.ts_code AND s.list_status = 'L'
        LEFT JOIN roe_stability rs ON f.ts_code = rs.ts_code
        """
        return self.conn.execute(sql).df()

    def calculate_growth_factors(self) -> pd.DataFrame:
        """计算成长因子"""
        sql = """
        WITH latest_fina AS (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY end_date DESC) as rn
            FROM fina_indicator_vip WHERE end_date >= '20240101'
        ),
        prev_fina AS (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY end_date DESC) as rn
            FROM fina_indicator_vip WHERE end_date >= '20230101' AND end_date < '20240101'
        )
        SELECT
            f.ts_code,
            s.name,
            f.end_date,
            f.or_yoy as revenue_yoy,
            f.netprofit_yoy,
            f.dt_netprofit_yoy as deducted_profit_yoy,
            f.q_sales_yoy,
            f.basic_eps_yoy,
            pf.netprofit_yoy as prev_profit_yoy,
            f.netprofit_yoy - COALESCE(pf.netprofit_yoy, 0) as growth_acceleration
        FROM (SELECT * FROM latest_fina WHERE rn = 1) f
        JOIN stock_basic s ON f.ts_code = s.ts_code AND s.list_status = 'L'
        LEFT JOIN (SELECT * FROM prev_fina WHERE rn = 1) pf ON f.ts_code = pf.ts_code
        """
        return self.conn.execute(sql).df()

    def calculate_financial_health_factors(self) -> pd.DataFrame:
        """计算财务健康因子"""
        sql = """
        WITH latest_fina AS (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY end_date DESC) as rn
            FROM fina_indicator_vip WHERE end_date >= '20240101'
        ),
        latest_cf AS (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY end_date DESC) as rn
            FROM cashflow WHERE report_type = '1' AND end_date >= '20240101'
        ),
        latest_inc AS (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY end_date DESC) as rn
            FROM income WHERE report_type = '1' AND end_date >= '20240101'
        )
        SELECT
            f.ts_code,
            s.name,
            f.end_date,
            f.debt_to_assets,
            f.current_ratio,
            f.quick_ratio,
            f.cash_ratio,
            f.debt_to_eqt as debt_to_equity,
            cf.n_cashflow_act,
            inc.n_income_attr_p as net_profit,
            CASE WHEN inc.n_income_attr_p > 0
                THEN cf.n_cashflow_act / inc.n_income_attr_p
                ELSE NULL END as ocf_to_profit
        FROM (SELECT * FROM latest_fina WHERE rn = 1) f
        JOIN stock_basic s ON f.ts_code = s.ts_code AND s.list_status = 'L'
        LEFT JOIN (SELECT * FROM latest_cf WHERE rn = 1) cf ON f.ts_code = cf.ts_code
        LEFT JOIN (SELECT * FROM latest_inc WHERE rn = 1) inc ON f.ts_code = inc.ts_code
        """
        return self.conn.execute(sql).df()

    def screen_value_stocks(self, config: Optional[ScreenerConfig] = None) -> pd.DataFrame:
        """价值投资策略筛选"""
        if config is None:
            config = ScreenerConfig()

        sql = f"""
        WITH latest_daily AS (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY trade_date DESC) as rn
            FROM daily_basic WHERE trade_date >= '20250101'
        ),
        latest_fina AS (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY end_date DESC) as rn
            FROM fina_indicator_vip WHERE end_date >= '20240101'
        )
        SELECT
            s.ts_code,
            s.name,
            s.industry,
            d.trade_date,
            d.pe_ttm,
            d.pb,
            d.dv_ttm as dividend_yield,
            d.total_mv / 10000 as market_cap_yi,
            f.roe,
            f.debt_to_assets,
            f.netprofit_yoy as profit_growth,
            -- 价值评分
            ROUND(
                (1 - LEAST(d.pe_ttm, 50) / 50) * 0.25 +
                (1 - LEAST(d.pb, 5) / 5) * 0.25 +
                LEAST(d.dv_ttm, 10) / 10 * 0.30 +
                LEAST(f.roe, 30) / 30 * 0.20
            , 4) as value_score
        FROM stock_basic s
        JOIN (SELECT * FROM latest_daily WHERE rn = 1) d ON s.ts_code = d.ts_code
        JOIN (SELECT * FROM latest_fina WHERE rn = 1) f ON s.ts_code = f.ts_code
        WHERE
            s.list_status = 'L'
            AND d.total_mv >= {config.min_market_cap}
            AND d.pe_ttm > 0 AND d.pe_ttm < 20
            AND d.pb > 0 AND d.pb < 3
            AND d.dv_ttm >= {config.min_dividend_yield}
            AND f.roe > {config.min_roe}
            AND f.debt_to_assets < {config.max_debt_ratio}
            AND f.netprofit_yoy > -30
        ORDER BY value_score DESC
        LIMIT {config.top_n}
        """
        return self.conn.execute(sql).df()

    def screen_growth_stocks(self, config: Optional[ScreenerConfig] = None) -> pd.DataFrame:
        """成长投资策略筛选"""
        if config is None:
            config = ScreenerConfig(max_pe=80, min_roe=15)

        sql = f"""
        WITH latest_daily AS (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY trade_date DESC) as rn
            FROM daily_basic WHERE trade_date >= '20250101'
        ),
        latest_fina AS (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY end_date DESC) as rn
            FROM fina_indicator_vip WHERE end_date >= '20240101'
        )
        SELECT
            s.ts_code,
            s.name,
            s.industry,
            d.pe_ttm,
            d.total_mv / 10000 as market_cap_yi,
            f.roe,
            f.netprofit_yoy as profit_growth,
            f.or_yoy as revenue_growth,
            f.gross_margin,
            CASE WHEN f.netprofit_yoy > 0 THEN ROUND(d.pe_ttm / f.netprofit_yoy, 2) ELSE NULL END as peg,
            -- 成长评分
            ROUND(
                LEAST(f.netprofit_yoy, 100) / 100 * 0.30 +
                LEAST(f.or_yoy, 100) / 100 * 0.15 +
                LEAST(f.roe, 40) / 40 * 0.30 +
                (1 - LEAST(d.pe_ttm, 100) / 100) * 0.15 +
                LEAST(f.gross_margin, 60) / 60 * 0.10
            , 4) as growth_score
        FROM stock_basic s
        JOIN (SELECT * FROM latest_daily WHERE rn = 1) d ON s.ts_code = d.ts_code
        JOIN (SELECT * FROM latest_fina WHERE rn = 1) f ON s.ts_code = f.ts_code
        WHERE
            s.list_status = 'L'
            AND d.total_mv >= 3000000
            AND f.netprofit_yoy >= 20
            AND f.or_yoy >= 15
            AND f.roe >= {config.min_roe}
            AND f.gross_margin >= 20
            AND d.pe_ttm > 0 AND d.pe_ttm < {config.max_pe}
            AND d.pe_ttm / NULLIF(f.netprofit_yoy, 0) < 2
        ORDER BY growth_score DESC
        LIMIT {config.top_n}
        """
        return self.conn.execute(sql).df()

    def screen_garp_stocks(self, config: Optional[ScreenerConfig] = None) -> pd.DataFrame:
        """GARP策略筛选"""
        if config is None:
            config = ScreenerConfig(max_pe=40, min_roe=12)

        sql = f"""
        WITH latest_daily AS (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY trade_date DESC) as rn
            FROM daily_basic WHERE trade_date >= '20250101'
        ),
        latest_fina AS (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY end_date DESC) as rn
            FROM fina_indicator_vip WHERE end_date >= '20240101'
        )
        SELECT
            s.ts_code,
            s.name,
            s.industry,
            d.pe_ttm,
            d.pb,
            d.total_mv / 10000 as market_cap_yi,
            f.roe,
            f.netprofit_yoy as profit_growth,
            f.or_yoy as revenue_growth,
            f.debt_to_assets,
            ROUND(d.pe_ttm / NULLIF(f.netprofit_yoy, 0), 2) as peg,
            -- GARP评分
            ROUND(
                (1 - LEAST(d.pe_ttm / NULLIF(f.netprofit_yoy, 0), 2) / 2) * 0.40 +
                LEAST(f.roe, 30) / 30 * 0.25 +
                LEAST(f.netprofit_yoy, 50) / 50 * 0.20 +
                (1 - LEAST(f.debt_to_assets, 80) / 80) * 0.15
            , 4) as garp_score
        FROM stock_basic s
        JOIN (SELECT * FROM latest_daily WHERE rn = 1) d ON s.ts_code = d.ts_code
        JOIN (SELECT * FROM latest_fina WHERE rn = 1) f ON s.ts_code = f.ts_code
        WHERE
            s.list_status = 'L'
            AND d.total_mv >= {config.min_market_cap}
            AND f.netprofit_yoy >= 10 AND f.netprofit_yoy <= 50
            AND f.or_yoy >= 5
            AND d.pe_ttm > 0 AND d.pe_ttm < {config.max_pe}
            AND d.pe_ttm / NULLIF(f.netprofit_yoy, 0) > 0
            AND d.pe_ttm / NULLIF(f.netprofit_yoy, 0) < 1.2
            AND f.roe >= {config.min_roe}
            AND f.debt_to_assets < 60
        ORDER BY garp_score DESC
        LIMIT {config.top_n}
        """
        return self.conn.execute(sql).df()

    def screen_quality_stocks(self, config: Optional[ScreenerConfig] = None) -> pd.DataFrame:
        """质量因子策略筛选"""
        if config is None:
            config = ScreenerConfig(min_roe=15, max_debt_ratio=55)

        sql = f"""
        WITH latest_daily AS (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY trade_date DESC) as rn
            FROM daily_basic WHERE trade_date >= '20250101'
        ),
        latest_fina AS (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY end_date DESC) as rn
            FROM fina_indicator_vip WHERE end_date >= '20240101'
        ),
        roe_stability AS (
            SELECT
                ts_code,
                AVG(roe) as avg_roe,
                STDDEV(roe) as std_roe,
                MIN(roe) as min_roe
            FROM fina_indicator_vip
            WHERE end_date >= '20210101' AND end_date LIKE '%1231'
              AND roe IS NOT NULL
            GROUP BY ts_code
            HAVING COUNT(*) >= 3
        ),
        latest_cf AS (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY end_date DESC) as rn
            FROM cashflow WHERE report_type = '1' AND end_date >= '20240101'
        ),
        latest_inc AS (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY end_date DESC) as rn
            FROM income WHERE report_type = '1' AND end_date >= '20240101'
        )
        SELECT
            s.ts_code,
            s.name,
            s.industry,
            d.pe_ttm,
            d.pb,
            d.total_mv / 10000 as market_cap_yi,
            f.roe,
            rs.avg_roe as roe_3y_avg,
            rs.std_roe as roe_3y_std,
            ROUND(rs.std_roe / NULLIF(rs.avg_roe, 0), 2) as roe_cv,
            f.gross_margin,
            f.debt_to_assets,
            ROUND(cf.n_cashflow_act / NULLIF(inc.n_income_attr_p, 0), 2) as ocf_ratio,
            -- 质量评分
            ROUND(
                LEAST(f.roe, 35) / 35 * 0.30 +
                (1 - LEAST(rs.std_roe / NULLIF(rs.avg_roe, 0), 1)) * 0.20 +
                LEAST(f.gross_margin, 50) / 50 * 0.15 +
                (1 - LEAST(f.debt_to_assets, 70) / 70) * 0.15 +
                LEAST(cf.n_cashflow_act / NULLIF(inc.n_income_attr_p, 0), 2) / 2 * 0.20
            , 4) as quality_score
        FROM stock_basic s
        JOIN (SELECT * FROM latest_daily WHERE rn = 1) d ON s.ts_code = d.ts_code
        JOIN (SELECT * FROM latest_fina WHERE rn = 1) f ON s.ts_code = f.ts_code
        JOIN roe_stability rs ON s.ts_code = rs.ts_code
        LEFT JOIN (SELECT * FROM latest_cf WHERE rn = 1) cf ON s.ts_code = cf.ts_code
        LEFT JOIN (SELECT * FROM latest_inc WHERE rn = 1) inc ON s.ts_code = inc.ts_code
        WHERE
            s.list_status = 'L'
            AND d.total_mv >= {config.min_market_cap}
            AND f.roe >= {config.min_roe}
            AND rs.avg_roe >= 12
            AND rs.min_roe >= 8
            AND rs.std_roe / NULLIF(rs.avg_roe, 0) < 0.5
            AND f.gross_margin >= 25
            AND f.debt_to_assets < {config.max_debt_ratio}
            AND cf.n_cashflow_act > 0
            AND cf.n_cashflow_act / NULLIF(inc.n_income_attr_p, 0) >= 0.7
            AND d.pe_ttm > 0 AND d.pe_ttm < {config.max_pe}
        ORDER BY quality_score DESC
        LIMIT {config.top_n}
        """
        return self.conn.execute(sql).df()

    def screen_turnaround_stocks(self, config: Optional[ScreenerConfig] = None) -> pd.DataFrame:
        """困境反转策略筛选"""
        if config is None:
            config = ScreenerConfig(min_market_cap=3000000, max_pe=30)

        sql = f"""
        WITH latest_daily AS (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY trade_date DESC) as rn
            FROM daily_basic WHERE trade_date >= '20250101'
        ),
        latest_fina AS (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY end_date DESC) as rn
            FROM fina_indicator_vip WHERE end_date >= '20240101'
        ),
        prev_fina AS (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY end_date DESC) as rn
            FROM fina_indicator_vip WHERE end_date >= '20230101' AND end_date < '20240101'
        ),
        pe_pct AS (
            SELECT
                ts_code,
                PERCENT_RANK() OVER (PARTITION BY ts_code ORDER BY pe_ttm) as pe_percentile
            FROM daily_basic
            WHERE trade_date >= '20200101' AND pe_ttm > 0 AND pe_ttm < 200
        )
        SELECT
            s.ts_code,
            s.name,
            s.industry,
            d.pe_ttm,
            d.pb,
            d.total_mv / 10000 as market_cap_yi,
            pp.pe_percentile,
            f.roe,
            pf.roe as prev_roe,
            f.netprofit_yoy as profit_growth,
            pf.netprofit_yoy as prev_profit_growth,
            f.debt_to_assets,
            CASE
                WHEN f.netprofit_yoy > 0 AND pf.netprofit_yoy < 0 THEN '扭亏为盈'
                WHEN f.netprofit_yoy > pf.netprofit_yoy + 20 THEN '业绩大幅改善'
                WHEN f.roe > pf.roe + 5 THEN 'ROE显著提升'
                ELSE '温和改善'
            END as turnaround_signal,
            -- 困境反转评分
            ROUND(
                (1 - COALESCE(pp.pe_percentile, 0.5)) * 0.25 +
                (1 - LEAST(d.pb, 5) / 5) * 0.15 +
                GREATEST(f.netprofit_yoy - COALESCE(pf.netprofit_yoy, 0), 0) / 100 * 0.30 +
                GREATEST(f.roe - COALESCE(pf.roe, 0), 0) / 20 * 0.20 +
                (1 - LEAST(f.debt_to_assets, 80) / 80) * 0.10
            , 4) as turnaround_score
        FROM stock_basic s
        JOIN (SELECT * FROM latest_daily WHERE rn = 1) d ON s.ts_code = d.ts_code
        JOIN (SELECT * FROM latest_fina WHERE rn = 1) f ON s.ts_code = f.ts_code
        LEFT JOIN (SELECT * FROM prev_fina WHERE rn = 1) pf ON s.ts_code = pf.ts_code
        LEFT JOIN (SELECT DISTINCT ON (ts_code) * FROM pe_pct ORDER BY ts_code, pe_percentile DESC) pp
            ON s.ts_code = pp.ts_code
        WHERE
            s.list_status = 'L'
            AND d.total_mv >= {config.min_market_cap}
            AND (pf.netprofit_yoy < 0 OR pf.roe < 8 OR pp.pe_percentile < 0.3)
            AND (
                f.netprofit_yoy > COALESCE(pf.netprofit_yoy, 0) + 15
                OR (f.netprofit_yoy > 0 AND COALESCE(pf.netprofit_yoy, 0) < 0)
                OR f.roe > COALESCE(pf.roe, 0) + 3
            )
            AND d.pe_ttm > 0 AND d.pe_ttm < {config.max_pe}
            AND d.pb > 0 AND d.pb < 4
            AND f.debt_to_assets < {config.max_debt_ratio}
        ORDER BY turnaround_score DESC
        LIMIT {config.top_n}
        """
        return self.conn.execute(sql).df()

    def screen(self, strategy: ScreenerStrategy, config: Optional[ScreenerConfig] = None) -> pd.DataFrame:
        """统一筛选接口"""
        strategy_methods = {
            ScreenerStrategy.VALUE: self.screen_value_stocks,
            ScreenerStrategy.GROWTH: self.screen_growth_stocks,
            ScreenerStrategy.GARP: self.screen_garp_stocks,
            ScreenerStrategy.QUALITY: self.screen_quality_stocks,
            ScreenerStrategy.TURNAROUND: self.screen_turnaround_stocks,
        }
        return strategy_methods[strategy](config)

    def get_industry_stats(self) -> pd.DataFrame:
        """获取行业统计信息"""
        sql = """
        SELECT
            i.l1_name as industry,
            COUNT(DISTINCT s.ts_code) as stock_count,
            AVG(d.pe_ttm) as avg_pe,
            AVG(d.pb) as avg_pb,
            AVG(f.roe) as avg_roe,
            AVG(f.netprofit_yoy) as avg_profit_growth
        FROM stock_basic s
        JOIN index_member_all i ON s.ts_code = i.ts_code AND i.out_date IS NULL
        JOIN (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY trade_date DESC) as rn
            FROM daily_basic WHERE trade_date >= '20250101'
        ) d ON s.ts_code = d.ts_code AND d.rn = 1
        JOIN (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY end_date DESC) as rn
            FROM fina_indicator_vip WHERE end_date >= '20240101'
        ) f ON s.ts_code = f.ts_code AND f.rn = 1
        WHERE s.list_status = 'L'
          AND d.pe_ttm > 0 AND d.pe_ttm < 200
          AND f.roe BETWEEN -50 AND 100
        GROUP BY i.l1_name
        HAVING COUNT(*) >= 10
        ORDER BY stock_count DESC
        """
        return self.conn.execute(sql).df()


# 使用示例
if __name__ == "__main__":
    DB_PATH = "/Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db"

    screener = StockScreener(DB_PATH)

    # 价值投资筛选
    print("=" * 60)
    print("价值投资策略筛选结果:")
    print("=" * 60)
    value_stocks = screener.screen(ScreenerStrategy.VALUE)
    print(value_stocks.head(20).to_string())

    # 成长投资筛选
    print("\n" + "=" * 60)
    print("成长投资策略筛选结果:")
    print("=" * 60)
    growth_stocks = screener.screen(ScreenerStrategy.GROWTH)
    print(growth_stocks.head(20).to_string())

    # GARP策略筛选
    print("\n" + "=" * 60)
    print("GARP策略筛选结果:")
    print("=" * 60)
    garp_stocks = screener.screen(ScreenerStrategy.GARP)
    print(garp_stocks.head(20).to_string())
```

---

## 六、回测框架建议

### 6.1 回测框架设计

```python
"""
股票筛选策略回测框架
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import duckdb


class BacktestEngine:
    """回测引擎"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = duckdb.connect(db_path, read_only=True)

    def get_rebalance_dates(
        self,
        start_date: str,
        end_date: str,
        frequency: str = 'Q'  # Q=季度, M=月度, Y=年度
    ) -> List[str]:
        """获取调仓日期"""
        sql = f"""
        SELECT DISTINCT trade_date
        FROM daily_basic
        WHERE trade_date BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY trade_date
        """
        all_dates = self.conn.execute(sql).df()['trade_date'].tolist()

        if frequency == 'Q':
            # 每季度末调仓
            rebalance_dates = [d for d in all_dates if d[4:6] in ['03', '06', '09', '12'] and d[6:8] >= '25']
        elif frequency == 'M':
            # 每月末调仓
            rebalance_dates = [d for d in all_dates if d[6:8] >= '25']
        else:
            # 每年末调仓
            rebalance_dates = [d for d in all_dates if d[4:8] >= '1225']

        return rebalance_dates

    def get_portfolio_returns(
        self,
        stocks: List[str],
        start_date: str,
        end_date: str,
        weights: List[float] = None
    ) -> pd.DataFrame:
        """计算组合收益"""
        if weights is None:
            weights = [1.0 / len(stocks)] * len(stocks)

        stock_list = "'" + "','".join(stocks) + "'"
        sql = f"""
        SELECT
            trade_date,
            ts_code,
            close,
            LAG(close) OVER (PARTITION BY ts_code ORDER BY trade_date) as prev_close
        FROM daily
        WHERE ts_code IN ({stock_list})
          AND trade_date BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY trade_date, ts_code
        """
        df = self.conn.execute(sql).df()
        df['daily_return'] = (df['close'] - df['prev_close']) / df['prev_close']

        # 计算组合收益
        portfolio_returns = df.pivot_table(
            index='trade_date',
            columns='ts_code',
            values='daily_return'
        )

        # 加权平均
        weights_dict = dict(zip(stocks, weights))
        portfolio_returns['portfolio'] = sum(
            portfolio_returns.get(s, 0) * weights_dict.get(s, 0)
            for s in stocks
        )

        return portfolio_returns[['portfolio']].dropna()

    def run_backtest(
        self,
        screener_func,  # 筛选函数
        start_date: str = '20200101',
        end_date: str = '20241231',
        rebalance_freq: str = 'Q',
        top_n: int = 30,
        initial_capital: float = 1000000
    ) -> Dict:
        """运行回测"""

        rebalance_dates = self.get_rebalance_dates(start_date, end_date, rebalance_freq)

        results = {
            'dates': [],
            'portfolio_value': [initial_capital],
            'holdings': [],
            'returns': []
        }

        current_value = initial_capital

        for i, date in enumerate(rebalance_dates[:-1]):
            # 在调仓日筛选股票
            selected_stocks = screener_func(date, top_n)

            if len(selected_stocks) == 0:
                continue

            # 计算到下一个调仓日的收益
            next_date = rebalance_dates[i + 1]
            portfolio_returns = self.get_portfolio_returns(
                selected_stocks, date, next_date
            )

            if len(portfolio_returns) > 0:
                period_return = (1 + portfolio_returns['portfolio']).prod() - 1
                current_value *= (1 + period_return)

                results['dates'].append(next_date)
                results['portfolio_value'].append(current_value)
                results['holdings'].append(selected_stocks)
                results['returns'].append(period_return)

        # 计算回测指标
        returns_series = pd.Series(results['returns'])

        results['metrics'] = {
            'total_return': (current_value / initial_capital - 1) * 100,
            'annual_return': ((current_value / initial_capital) ** (252 / len(results['dates'])) - 1) * 100 if len(results['dates']) > 0 else 0,
            'sharpe_ratio': returns_series.mean() / returns_series.std() * np.sqrt(4) if returns_series.std() > 0 else 0,  # 季度调仓
            'max_drawdown': self._calculate_max_drawdown(results['portfolio_value']),
            'win_rate': (returns_series > 0).mean() * 100
        }

        return results

    def _calculate_max_drawdown(self, values: List[float]) -> float:
        """计算最大回撤"""
        values = np.array(values)
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / peak
        return drawdown.min() * 100

    def compare_strategies(
        self,
        strategies: Dict[str, callable],
        start_date: str = '20200101',
        end_date: str = '20241231'
    ) -> pd.DataFrame:
        """比较多个策略"""
        results = {}

        for name, screener_func in strategies.items():
            backtest_result = self.run_backtest(
                screener_func, start_date, end_date
            )
            results[name] = backtest_result['metrics']

        return pd.DataFrame(results).T


# 回测使用示例
def create_value_screener(db_path: str):
    """创建价值投资筛选函数"""
    def screener(date: str, top_n: int) -> List[str]:
        conn = duckdb.connect(db_path, read_only=True)
        sql = f"""
        WITH latest_daily AS (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY trade_date DESC) as rn
            FROM daily_basic WHERE trade_date <= '{date}'
        ),
        latest_fina AS (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY end_date DESC) as rn
            FROM fina_indicator_vip WHERE end_date <= '{date}'
        )
        SELECT s.ts_code
        FROM stock_basic s
        JOIN (SELECT * FROM latest_daily WHERE rn = 1) d ON s.ts_code = d.ts_code
        JOIN (SELECT * FROM latest_fina WHERE rn = 1) f ON s.ts_code = f.ts_code
        WHERE
            s.list_status = 'L'
            AND d.pe_ttm > 0 AND d.pe_ttm < 15
            AND d.pb > 0 AND d.pb < 2
            AND d.dv_ttm >= 3
            AND f.roe > 10
        ORDER BY d.dv_ttm DESC
        LIMIT {top_n}
        """
        result = conn.execute(sql).df()
        conn.close()
        return result['ts_code'].tolist()
    return screener
```

### 6.2 回测指标说明

| 指标 | 计算公式 | 说明 |
|------|---------|------|
| 总收益率 | (期末净值 / 期初净值 - 1) * 100% | 累计收益 |
| 年化收益率 | (1 + 总收益率)^(252/交易日数) - 1 | 折算年化 |
| 夏普比率 | (组合收益率 - 无风险收益率) / 收益率标准差 | 风险调整收益 |
| 最大回撤 | max(累计最高值 - 当前值) / 累计最高值 | 最大亏损幅度 |
| 胜率 | 盈利调仓期数 / 总调仓期数 | 盈利概率 |
| 信息比率 | (组合收益 - 基准收益) / 跟踪误差 | 主动管理能力 |
| Calmar比率 | 年化收益率 / 最大回撤 | 风险调整收益 |

---

## 七、数据质量处理

### 7.1 缺失值处理

```sql
-- 财务数据缺失填充策略
-- 1. 使用最近期数据填充
-- 2. 行业均值填充
-- 3. 标记为NULL不参与计算

WITH filled_data AS (
    SELECT
        ts_code,
        end_date,
        -- ROE: 使用最近有效值
        COALESCE(
            roe,
            LAG(roe, 1) OVER (PARTITION BY ts_code ORDER BY end_date),
            LAG(roe, 2) OVER (PARTITION BY ts_code ORDER BY end_date)
        ) as roe_filled,
        -- 毛利率: 使用行业均值
        COALESCE(
            gross_margin,
            AVG(gross_margin) OVER (PARTITION BY ts_code)
        ) as gross_margin_filled
    FROM fina_indicator_vip
)
SELECT * FROM filled_data;
```

### 7.2 异常值处理

```sql
-- 异常值识别与处理
-- 使用3倍标准差法或行业分位数法

WITH stats AS (
    SELECT
        ts_code,
        roe,
        AVG(roe) OVER () as mean_roe,
        STDDEV(roe) OVER () as std_roe
    FROM fina_indicator_vip
    WHERE roe IS NOT NULL
),
cleaned AS (
    SELECT
        ts_code,
        CASE
            WHEN ABS(roe - mean_roe) > 3 * std_roe THEN NULL  -- 3sigma外视为异常
            WHEN roe < -100 OR roe > 200 THEN NULL            -- 绝对边界
            ELSE roe
        END as roe_cleaned
    FROM stats
)
SELECT * FROM cleaned;
```

---

## 八、使用建议

### 8.1 策略选择指南

| 投资风格 | 推荐策略 | 适用场景 |
|---------|---------|---------|
| 保守型 | 价值投资 | 熊市末期、震荡市 |
| 稳健型 | GARP | 任何市场环境 |
| 积极型 | 成长投资 | 牛市初中期 |
| 机构型 | 质量因子 | 长期配置 |
| 博弈型 | 困境反转 | 行业周期底部 |

### 8.2 风险提示

1. **数据滞后**: 财务数据有1-2个月滞后，需考虑信息时效性
2. **行业差异**: 不同行业估值标准差异大，需行业中性化处理
3. **市值偏差**: 小市值股票流动性风险大，需设置市值门槛
4. **极端值影响**: 部分指标受异常值影响大，需做异常值处理
5. **样本偏差**: 历史回测存在幸存者偏差，实际收益可能低于回测

### 8.3 最佳实践

1. 多因子组合优于单因子
2. 定期调仓(季度/月度)优于长期持有
3. 设置止损线控制风险
4. 结合技术面信号提高择时
5. 分散投资,单只股票仓位不超过10%

---

## 附录: 完整字段参考

### fina_indicator_vip 主要字段

| 字段名 | 中文名 | 说明 |
|-------|-------|------|
| roe | 净资产收益率 | % |
| roa | 总资产收益率 | % |
| roic | 投入资本回报率 | % |
| gross_margin | 毛利率 | % |
| netprofit_margin | 净利率 | % |
| debt_to_assets | 资产负债率 | % |
| current_ratio | 流动比率 | 倍 |
| quick_ratio | 速动比率 | 倍 |
| netprofit_yoy | 净利润同比增长 | % |
| or_yoy | 营业收入同比增长 | % |
| q_sales_yoy | 单季度营收同比 | % |

### daily_basic 主要字段

| 字段名 | 中文名 | 说明 |
|-------|-------|------|
| pe_ttm | 市盈率TTM | 滚动12月 |
| pb | 市净率 | |
| ps_ttm | 市销率TTM | |
| dv_ttm | 股息率TTM | % |
| total_mv | 总市值 | 万元 |
| circ_mv | 流通市值 | 万元 |

---

*文档版本: v1.0*
*创建日期: 2026-01-31*
*作者: 量化投资系统*
