-- =====================================================
-- 主力资金流因子分析SQL脚本
-- 用于重现研究报告中的分析结果
-- =====================================================

-- 1. 数据概览
-- =====================================================

-- 1.1 moneyflow表基本统计
SELECT
    'moneyflow' as source,
    COUNT(*) as total_records,
    COUNT(DISTINCT ts_code) as stock_cnt,
    COUNT(DISTINCT trade_date) as date_cnt,
    MIN(trade_date) as min_date,
    MAX(trade_date) as max_date
FROM moneyflow;

-- 1.2 moneyflow_dc表基本统计
SELECT
    'moneyflow_dc' as source,
    COUNT(*) as total_records,
    COUNT(DISTINCT ts_code) as stock_cnt,
    COUNT(DISTINCT trade_date) as date_cnt,
    MIN(trade_date) as min_date,
    MAX(trade_date) as max_date
FROM moneyflow_dc;

-- 1.3 两个数据源对比
SELECT
    m.ts_code,
    m.trade_date,
    -- moneyflow 计算主力净流入 (超大单+大单)
    (m.buy_elg_amount - m.sell_elg_amount) + (m.buy_lg_amount - m.sell_lg_amount) as mf_main_net,
    -- moneyflow_dc 的主力净流入
    d.net_amount as dc_main_net,
    -- 差异
    ABS(((m.buy_elg_amount - m.sell_elg_amount) + (m.buy_lg_amount - m.sell_lg_amount)) - d.net_amount) as diff
FROM moneyflow m
JOIN moneyflow_dc d ON m.ts_code = d.ts_code AND m.trade_date = d.trade_date
WHERE m.ts_code = '000001.SZ'
LIMIT 20;


-- 2. 主力行为分析
-- =====================================================

-- 2.1 主力资金流向与涨跌关系
SELECT
    CASE
        WHEN d.pct_change >= 9.5 THEN '涨停'
        WHEN d.pct_change >= 5 THEN '大涨(5-9.5%)'
        WHEN d.pct_change >= 2 THEN '中涨(2-5%)'
        WHEN d.pct_change >= 0 THEN '小涨(0-2%)'
        WHEN d.pct_change >= -2 THEN '小跌(0-2%)'
        WHEN d.pct_change >= -5 THEN '中跌(2-5%)'
        WHEN d.pct_change >= -9.5 THEN '大跌(5-9.5%)'
        ELSE '跌停'
    END as price_range,
    COUNT(*) as cnt,
    ROUND(AVG(net_amount), 2) as avg_main_net,
    ROUND(AVG(buy_elg_amount), 2) as avg_elg_net,
    ROUND(AVG(buy_lg_amount), 2) as avg_lg_net,
    ROUND(SUM(CASE WHEN net_amount > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as pct_positive
FROM moneyflow_dc d
WHERE d.pct_change IS NOT NULL
GROUP BY price_range
ORDER BY
    CASE price_range
        WHEN '涨停' THEN 1 WHEN '大涨(5-9.5%)' THEN 2 WHEN '中涨(2-5%)' THEN 3
        WHEN '小涨(0-2%)' THEN 4 WHEN '小跌(0-2%)' THEN 5 WHEN '中跌(2-5%)' THEN 6
        WHEN '大跌(5-9.5%)' THEN 7 WHEN '跌停' THEN 8
    END;

-- 2.2 连续资金流模式分析
WITH daily_flow AS (
    SELECT
        ts_code, trade_date, net_amount, pct_change,
        LAG(net_amount, 1) OVER (PARTITION BY ts_code ORDER BY trade_date) as prev_1_net,
        LAG(net_amount, 2) OVER (PARTITION BY ts_code ORDER BY trade_date) as prev_2_net,
        LAG(net_amount, 3) OVER (PARTITION BY ts_code ORDER BY trade_date) as prev_3_net,
        LAG(net_amount, 4) OVER (PARTITION BY ts_code ORDER BY trade_date) as prev_4_net,
        LEAD(pct_change, 1) OVER (PARTITION BY ts_code ORDER BY trade_date) as next_1_ret,
        LEAD(pct_change, 3) OVER (PARTITION BY ts_code ORDER BY trade_date) as next_3_ret,
        LEAD(pct_change, 5) OVER (PARTITION BY ts_code ORDER BY trade_date) as next_5_ret
    FROM moneyflow_dc
),
consecutive_inflow AS (
    SELECT *,
        CASE
            WHEN net_amount > 0 AND prev_1_net > 0 AND prev_2_net > 0 AND prev_3_net > 0 AND prev_4_net > 0 THEN '连续5日净流入'
            WHEN net_amount > 0 AND prev_1_net > 0 AND prev_2_net > 0 THEN '连续3日净流入'
            WHEN net_amount < 0 AND prev_1_net < 0 AND prev_2_net < 0 AND prev_3_net < 0 AND prev_4_net < 0 THEN '连续5日净流出'
            WHEN net_amount < 0 AND prev_1_net < 0 AND prev_2_net < 0 THEN '连续3日净流出'
            ELSE '其他'
        END as flow_pattern
    FROM daily_flow
    WHERE prev_4_net IS NOT NULL
)
SELECT
    flow_pattern,
    COUNT(*) as cnt,
    ROUND(AVG(next_1_ret), 4) as avg_next_1_ret,
    ROUND(AVG(next_3_ret), 4) as avg_next_3_ret,
    ROUND(AVG(next_5_ret), 4) as avg_next_5_ret,
    ROUND(SUM(CASE WHEN next_1_ret > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as win_rate_1d
FROM consecutive_inflow
WHERE flow_pattern != '其他'
GROUP BY flow_pattern;

-- 2.3 主力控盘程度分析
WITH control_analysis AS (
    SELECT
        m.ts_code, m.trade_date, d.pct_change, d.net_amount,
        (m.buy_elg_amount + m.buy_lg_amount) /
            NULLIF(m.buy_elg_amount + m.buy_lg_amount + m.buy_md_amount + m.buy_sm_amount, 0) as main_buy_ratio
    FROM moneyflow m
    JOIN moneyflow_dc d ON m.ts_code = d.ts_code AND m.trade_date = d.trade_date
    WHERE m.buy_elg_amount + m.buy_lg_amount + m.buy_md_amount + m.buy_sm_amount > 0
)
SELECT
    CASE
        WHEN main_buy_ratio >= 0.6 THEN '高控盘(>=60%)'
        WHEN main_buy_ratio >= 0.4 THEN '中控盘(40-60%)'
        WHEN main_buy_ratio >= 0.2 THEN '低控盘(20-40%)'
        ELSE '散户主导(<20%)'
    END as control_level,
    COUNT(*) as cnt,
    ROUND(AVG(pct_change), 4) as avg_return,
    ROUND(AVG(net_amount), 2) as avg_main_net,
    ROUND(SUM(CASE WHEN pct_change > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as win_rate
FROM control_analysis
WHERE main_buy_ratio IS NOT NULL
GROUP BY control_level
ORDER BY control_level;


-- 3. 因子有效性检验
-- =====================================================

-- 3.1 主力净流入因子分层收益
WITH factor_data AS (
    SELECT
        d.ts_code, d.trade_date, d.pct_change, d.close,
        d.net_amount / NULLIF(d.close * 10000, 0) as main_net_factor,
        LEAD(d.pct_change, 1) OVER (PARTITION BY d.ts_code ORDER BY d.trade_date) as next_1_ret,
        LEAD(d.pct_change, 5) OVER (PARTITION BY d.ts_code ORDER BY d.trade_date) as next_5_ret
    FROM moneyflow_dc d
    WHERE d.pct_change IS NOT NULL AND d.close > 0
),
factor_quintile AS (
    SELECT *,
        NTILE(5) OVER (PARTITION BY trade_date ORDER BY main_net_factor) as quintile
    FROM factor_data
    WHERE main_net_factor IS NOT NULL
)
SELECT
    '主力净流入因子' as factor_name,
    quintile,
    COUNT(*) as cnt,
    ROUND(AVG(next_1_ret), 4) as avg_next_1_ret,
    ROUND(AVG(next_5_ret), 4) as avg_next_5_ret,
    ROUND(SUM(CASE WHEN next_1_ret > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as win_rate
FROM factor_quintile
WHERE next_1_ret IS NOT NULL
GROUP BY quintile
ORDER BY quintile;

-- 3.2 IC/IR计算
WITH factor_returns AS (
    SELECT
        d.trade_date, d.ts_code,
        d.net_amount / NULLIF(d.close * 10000, 0) as main_net_factor,
        d.buy_elg_amount / NULLIF(d.close * 10000, 0) as elg_net_factor,
        d.net_amount_rate as net_rate_factor,
        LEAD(d.pct_change, 1) OVER (PARTITION BY d.ts_code ORDER BY d.trade_date) as next_1_ret
    FROM moneyflow_dc d
    WHERE d.pct_change IS NOT NULL AND d.close > 0
),
daily_ic AS (
    SELECT
        trade_date,
        CORR(main_net_factor, next_1_ret) as main_net_ic,
        CORR(elg_net_factor, next_1_ret) as elg_net_ic,
        CORR(net_rate_factor, next_1_ret) as net_rate_ic
    FROM factor_returns
    WHERE next_1_ret IS NOT NULL
    GROUP BY trade_date
    HAVING COUNT(*) > 100
)
SELECT
    '主力净流入因子' as factor,
    ROUND(AVG(main_net_ic), 6) as avg_ic,
    ROUND(STDDEV(main_net_ic), 6) as std_ic,
    ROUND(AVG(main_net_ic) / NULLIF(STDDEV(main_net_ic), 0), 4) as ir,
    ROUND(SUM(CASE WHEN main_net_ic > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as ic_positive_pct
FROM daily_ic
UNION ALL
SELECT
    '超大单净流入因子',
    ROUND(AVG(elg_net_ic), 6),
    ROUND(STDDEV(elg_net_ic), 6),
    ROUND(AVG(elg_net_ic) / NULLIF(STDDEV(elg_net_ic), 0), 4),
    ROUND(SUM(CASE WHEN elg_net_ic > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2)
FROM daily_ic
UNION ALL
SELECT
    '净流入占比因子',
    ROUND(AVG(net_rate_ic), 6),
    ROUND(STDDEV(net_rate_ic), 6),
    ROUND(AVG(net_rate_ic) / NULLIF(STDDEV(net_rate_ic), 0), 4),
    ROUND(SUM(CASE WHEN net_rate_ic > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2)
FROM daily_ic;


-- 4. 策略回测
-- =====================================================

-- 4.1 资金流跟踪策略
WITH strategy_signal AS (
    SELECT
        d.ts_code, d.name, d.trade_date, d.pct_change, d.close,
        d.net_amount, d.buy_elg_amount,
        LAG(d.net_amount, 1) OVER (PARTITION BY d.ts_code ORDER BY d.trade_date) as prev_1_net,
        LAG(d.net_amount, 2) OVER (PARTITION BY d.ts_code ORDER BY d.trade_date) as prev_2_net,
        LEAD(d.pct_change, 1) OVER (PARTITION BY d.ts_code ORDER BY d.trade_date) as next_1_ret,
        LEAD(d.pct_change, 3) OVER (PARTITION BY d.ts_code ORDER BY d.trade_date) as next_3_ret,
        LEAD(d.pct_change, 5) OVER (PARTITION BY d.ts_code ORDER BY d.trade_date) as next_5_ret
    FROM moneyflow_dc d
),
buy_signals AS (
    SELECT * FROM strategy_signal
    WHERE net_amount > 0 AND prev_1_net > 0 AND prev_2_net > 0
        AND pct_change BETWEEN -5 AND 3
        AND buy_elg_amount > 0
)
SELECT
    '资金流跟踪策略' as strategy,
    COUNT(*) as signal_cnt,
    ROUND(AVG(next_1_ret), 4) as avg_1d_ret,
    ROUND(AVG(next_3_ret), 4) as avg_3d_ret,
    ROUND(AVG(next_5_ret), 4) as avg_5d_ret,
    ROUND(SUM(CASE WHEN next_1_ret > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as win_rate_1d,
    ROUND(SUM(CASE WHEN next_5_ret > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as win_rate_5d
FROM buy_signals
WHERE next_5_ret IS NOT NULL;

-- 4.2 资金反转策略
WITH reversal_signal AS (
    SELECT
        d.ts_code, d.name, d.trade_date, d.pct_change, d.close, d.net_amount,
        LAG(d.net_amount, 1) OVER (PARTITION BY d.ts_code ORDER BY d.trade_date) as prev_1_net,
        LAG(d.net_amount, 2) OVER (PARTITION BY d.ts_code ORDER BY d.trade_date) as prev_2_net,
        LAG(d.net_amount, 3) OVER (PARTITION BY d.ts_code ORDER BY d.trade_date) as prev_3_net,
        LAG(d.pct_change, 1) OVER (PARTITION BY d.ts_code ORDER BY d.trade_date) as prev_1_ret,
        LAG(d.pct_change, 2) OVER (PARTITION BY d.ts_code ORDER BY d.trade_date) as prev_2_ret,
        LEAD(d.pct_change, 1) OVER (PARTITION BY d.ts_code ORDER BY d.trade_date) as next_1_ret,
        LEAD(d.pct_change, 5) OVER (PARTITION BY d.ts_code ORDER BY d.trade_date) as next_5_ret
    FROM moneyflow_dc d
),
reversal_signals AS (
    SELECT * FROM reversal_signal
    WHERE prev_1_net < 0 AND prev_2_net < 0 AND prev_3_net < 0
        AND net_amount > 500
        AND prev_1_ret < 0 AND prev_2_ret < 0
        AND pct_change BETWEEN -1 AND 3
)
SELECT
    '资金反转策略' as strategy,
    COUNT(*) as signal_cnt,
    ROUND(AVG(next_1_ret), 4) as avg_1d_ret,
    ROUND(AVG(next_5_ret), 4) as avg_5d_ret,
    ROUND(SUM(CASE WHEN next_1_ret > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as win_rate_1d,
    ROUND(SUM(CASE WHEN next_5_ret > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as win_rate_5d
FROM reversal_signals
WHERE next_5_ret IS NOT NULL;

-- 4.3 主力吸筹识别
WITH accumulation_signal AS (
    SELECT
        d.ts_code, d.name, d.trade_date, d.pct_change, d.close,
        d.net_amount, d.buy_elg_amount,
        db.turnover_rate,
        LAG(d.net_amount, 1) OVER (PARTITION BY d.ts_code ORDER BY d.trade_date) as prev_1_net,
        LAG(d.net_amount, 2) OVER (PARTITION BY d.ts_code ORDER BY d.trade_date) as prev_2_net,
        LAG(d.net_amount, 3) OVER (PARTITION BY d.ts_code ORDER BY d.trade_date) as prev_3_net,
        LAG(d.net_amount, 4) OVER (PARTITION BY d.ts_code ORDER BY d.trade_date) as prev_4_net,
        AVG(d.pct_change) OVER (PARTITION BY d.ts_code ORDER BY d.trade_date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) as avg_5d_ret,
        AVG(db.turnover_rate) OVER (PARTITION BY d.ts_code ORDER BY d.trade_date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) as avg_5d_turnover,
        LEAD(d.pct_change, 5) OVER (PARTITION BY d.ts_code ORDER BY d.trade_date) as next_5_ret,
        LEAD(d.pct_change, 10) OVER (PARTITION BY d.ts_code ORDER BY d.trade_date) as next_10_ret
    FROM moneyflow_dc d
    JOIN daily_basic db ON d.ts_code = db.ts_code AND d.trade_date = db.trade_date
),
accumulation_signals AS (
    SELECT * FROM accumulation_signal
    WHERE net_amount > 0 AND prev_1_net > 0 AND prev_2_net > 0 AND prev_3_net > 0 AND prev_4_net > 0
        AND avg_5d_ret BETWEEN 0 AND 1
        AND avg_5d_turnover < 3
)
SELECT
    '主力吸筹识别' as pattern,
    COUNT(*) as signal_cnt,
    ROUND(AVG(next_5_ret), 4) as avg_5d_ret,
    ROUND(AVG(next_10_ret), 4) as avg_10d_ret,
    ROUND(SUM(CASE WHEN next_5_ret > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as win_rate_5d,
    ROUND(SUM(CASE WHEN next_10_ret > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as win_rate_10d
FROM accumulation_signals
WHERE next_10_ret IS NOT NULL;

-- 4.4 主力出货识别
WITH distribution_signal AS (
    SELECT
        d.ts_code, d.name, d.trade_date, d.pct_change, d.close,
        d.net_amount, d.buy_elg_amount,
        db.turnover_rate,
        MAX(daily.close) OVER (PARTITION BY d.ts_code ORDER BY d.trade_date ROWS BETWEEN 59 PRECEDING AND 1 PRECEDING) as high_60d,
        AVG(db.turnover_rate) OVER (PARTITION BY d.ts_code ORDER BY d.trade_date ROWS BETWEEN 19 PRECEDING AND 1 PRECEDING) as avg_20d_turnover,
        LEAD(d.pct_change, 5) OVER (PARTITION BY d.ts_code ORDER BY d.trade_date) as next_5_ret,
        LEAD(d.pct_change, 10) OVER (PARTITION BY d.ts_code ORDER BY d.trade_date) as next_10_ret
    FROM moneyflow_dc d
    JOIN daily_basic db ON d.ts_code = db.ts_code AND d.trade_date = db.trade_date
    JOIN daily ON d.ts_code = daily.ts_code AND d.trade_date = daily.trade_date
),
distribution_signals AS (
    SELECT * FROM distribution_signal
    WHERE close >= high_60d * 0.95
        AND net_amount < -2000
        AND turnover_rate > avg_20d_turnover * 2
        AND pct_change < 0
)
SELECT
    '主力出货识别' as pattern,
    COUNT(*) as signal_cnt,
    ROUND(AVG(next_5_ret), 4) as avg_5d_ret,
    ROUND(AVG(next_10_ret), 4) as avg_10d_ret,
    ROUND(SUM(CASE WHEN next_5_ret < 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as down_rate_5d,
    ROUND(SUM(CASE WHEN next_10_ret < 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as down_rate_10d
FROM distribution_signals
WHERE next_10_ret IS NOT NULL;


-- 5. 今日信号生成 (可用于实盘)
-- =====================================================

-- 5.1 今日资金流跟踪信号
WITH recent_flow AS (
    SELECT
        d.ts_code, d.name, d.trade_date, d.pct_change, d.close,
        d.net_amount, d.buy_elg_amount, d.net_amount_rate,
        LAG(d.net_amount, 1) OVER (PARTITION BY d.ts_code ORDER BY d.trade_date) as prev_1_net,
        LAG(d.net_amount, 2) OVER (PARTITION BY d.ts_code ORDER BY d.trade_date) as prev_2_net,
        ROW_NUMBER() OVER (PARTITION BY d.ts_code ORDER BY d.trade_date DESC) as rn
    FROM moneyflow_dc d
)
SELECT
    ts_code, name, trade_date,
    ROUND(pct_change, 2) as pct_change,
    ROUND(close, 2) as close,
    ROUND(net_amount, 2) as net_amount,
    ROUND(buy_elg_amount, 2) as elg_net,
    ROUND(net_amount_rate, 2) as net_rate
FROM recent_flow
WHERE rn = 1
    AND net_amount > 0 AND prev_1_net > 0 AND prev_2_net > 0
    AND pct_change BETWEEN -5 AND 3
    AND buy_elg_amount > 0
ORDER BY net_amount DESC
LIMIT 20;

-- 5.2 今日资金反转信号
WITH recent_reversal AS (
    SELECT
        d.ts_code, d.name, d.trade_date, d.pct_change, d.close,
        d.net_amount, d.buy_elg_amount,
        LAG(d.net_amount, 1) OVER (PARTITION BY d.ts_code ORDER BY d.trade_date) as prev_1_net,
        LAG(d.net_amount, 2) OVER (PARTITION BY d.ts_code ORDER BY d.trade_date) as prev_2_net,
        LAG(d.net_amount, 3) OVER (PARTITION BY d.ts_code ORDER BY d.trade_date) as prev_3_net,
        LAG(d.pct_change, 1) OVER (PARTITION BY d.ts_code ORDER BY d.trade_date) as prev_1_ret,
        LAG(d.pct_change, 2) OVER (PARTITION BY d.ts_code ORDER BY d.trade_date) as prev_2_ret,
        ROW_NUMBER() OVER (PARTITION BY d.ts_code ORDER BY d.trade_date DESC) as rn
    FROM moneyflow_dc d
)
SELECT
    ts_code, name, trade_date,
    ROUND(pct_change, 2) as pct_change,
    ROUND(close, 2) as close,
    ROUND(net_amount, 2) as net_amount,
    ROUND(prev_1_net + prev_2_net + prev_3_net, 2) as prev_3d_outflow
FROM recent_reversal
WHERE rn = 1
    AND prev_1_net < 0 AND prev_2_net < 0 AND prev_3_net < 0
    AND net_amount > 500
    AND prev_1_ret < 0 AND prev_2_ret < 0
    AND pct_change BETWEEN -1 AND 3
ORDER BY net_amount DESC
LIMIT 20;
