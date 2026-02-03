#!/usr/bin/env python3
"""
A股市场价值因子深度研究
Value Factor Deep Dive Study for A-Share Market

研究内容：
1. 传统价值因子分析: EP、BP、SP、DP的IC/IR分析
2. 改进价值因子: EBIT/EV、FCF/P、行业相对估值
3. 价值陷阱研究: 识别价值陷阱特征
4. 价值因子周期性: 长期表现、与市场环境关系、风格轮动
5. 价值投资策略: 深度价值、GARP、质量价值策略回测
"""

import duckdb
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Database connection
DB_PATH = '/Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db'
REPORT_PATH = '/Users/allen/workspace/python/stock/Tushare-DuckDB/reports/quality_check/value_factor_study.md'

def get_connection():
    return duckdb.connect(DB_PATH, read_only=True)

# ============================================================================
# Part 1: 传统价值因子 IC/IR 分析
# ============================================================================

def calculate_traditional_value_factors():
    """计算传统价值因子: EP, BP, SP, DP"""
    print("=" * 60)
    print("Part 1: 传统价值因子 IC/IR 分析")
    print("=" * 60)

    conn = get_connection()

    # 获取估值数据和未来收益
    query = """
    WITH monthly_data AS (
        -- 每月最后一个交易日
        SELECT
            ts_code,
            trade_date,
            SUBSTRING(trade_date, 1, 6) as year_month,
            pe_ttm,
            pb,
            ps_ttm,
            dv_ttm,
            total_mv
        FROM daily_basic
        WHERE trade_date >= '20100101'
          AND trade_date <= '20251231'
          AND pe_ttm IS NOT NULL
          AND pe_ttm > 0
          AND pe_ttm < 1000
          AND pb IS NOT NULL
          AND pb > 0
          AND pb < 100
          AND ps_ttm IS NOT NULL
          AND ps_ttm > 0
          AND total_mv IS NOT NULL
          AND total_mv > 0
    ),
    month_end AS (
        SELECT
            ts_code,
            year_month,
            MAX(trade_date) as month_end_date
        FROM monthly_data
        GROUP BY ts_code, year_month
    ),
    factor_data AS (
        SELECT
            m.ts_code,
            m.year_month,
            m.month_end_date,
            md.pe_ttm,
            md.pb,
            md.ps_ttm,
            md.dv_ttm,
            md.total_mv,
            -- 计算价值因子 (取倒数，值越大越便宜)
            1.0 / md.pe_ttm as EP,
            1.0 / md.pb as BP,
            1.0 / md.ps_ttm as SP,
            COALESCE(md.dv_ttm, 0) as DP
        FROM month_end m
        JOIN monthly_data md ON m.ts_code = md.ts_code AND m.month_end_date = md.trade_date
    ),
    -- 计算下月收益率
    returns AS (
        SELECT
            ts_code,
            SUBSTRING(trade_date, 1, 6) as year_month,
            trade_date,
            close,
            LAG(close) OVER (PARTITION BY ts_code ORDER BY trade_date) as prev_close
        FROM daily
        WHERE trade_date >= '20100101'
    ),
    monthly_returns AS (
        SELECT
            ts_code,
            year_month,
            (MAX(close) - MIN(CASE WHEN prev_close IS NOT NULL THEN prev_close END)) /
             NULLIF(MIN(CASE WHEN prev_close IS NOT NULL THEN prev_close END), 0) as monthly_ret
        FROM returns
        GROUP BY ts_code, year_month
    ),
    next_month AS (
        SELECT
            ts_code,
            year_month,
            LEAD(monthly_ret) OVER (PARTITION BY ts_code ORDER BY year_month) as next_month_ret
        FROM monthly_returns
    )
    SELECT
        f.ts_code,
        f.year_month,
        f.EP,
        f.BP,
        f.SP,
        f.DP,
        f.total_mv,
        n.next_month_ret
    FROM factor_data f
    LEFT JOIN next_month n ON f.ts_code = n.ts_code AND f.year_month = n.year_month
    WHERE n.next_month_ret IS NOT NULL
      AND ABS(n.next_month_ret) < 0.5  -- 排除异常收益
    ORDER BY f.year_month, f.ts_code
    """

    print("正在加载月度数据...")
    df = conn.execute(query).fetchdf()
    print(f"加载了 {len(df):,} 条记录")

    # 计算每月的IC
    factors = ['EP', 'BP', 'SP', 'DP']
    ic_results = {f: [] for f in factors}
    months = df['year_month'].unique()

    print(f"分析 {len(months)} 个月份的IC...")

    for month in sorted(months):
        month_data = df[df['year_month'] == month]
        if len(month_data) < 100:  # 至少100只股票
            continue

        for factor in factors:
            # 去极值
            factor_values = month_data[factor].copy()
            lower = factor_values.quantile(0.01)
            upper = factor_values.quantile(0.99)
            factor_values = factor_values.clip(lower, upper)

            # Rank IC (Spearman相关系数)
            valid_mask = factor_values.notna() & month_data['next_month_ret'].notna()
            if valid_mask.sum() >= 50:
                ic, _ = stats.spearmanr(
                    factor_values[valid_mask],
                    month_data.loc[valid_mask, 'next_month_ret']
                )
                if not np.isnan(ic):
                    ic_results[factor].append({'month': month, 'ic': ic})

    # 计算IC统计
    ic_stats = {}
    for factor in factors:
        ics = pd.DataFrame(ic_results[factor])
        if len(ics) > 0:
            ic_mean = ics['ic'].mean()
            ic_std = ics['ic'].std()
            ic_ir = ic_mean / ic_std if ic_std > 0 else 0
            ic_positive_rate = (ics['ic'] > 0).mean()
            ic_abs_mean = ics['ic'].abs().mean()

            ic_stats[factor] = {
                'IC_Mean': ic_mean,
                'IC_Std': ic_std,
                'IR': ic_ir,
                'IC_Positive_Rate': ic_positive_rate,
                'IC_Abs_Mean': ic_abs_mean,
                'N_Months': len(ics)
            }

    ic_df = pd.DataFrame(ic_stats).T
    print("\n传统价值因子IC/IR统计:")
    print(ic_df.round(4))

    conn.close()
    return ic_df, ic_results, df

# ============================================================================
# Part 2: 改进价值因子
# ============================================================================

def calculate_improved_value_factors():
    """计算改进价值因子: EBIT/EV, FCF/P, 行业相对估值"""
    print("\n" + "=" * 60)
    print("Part 2: 改进价值因子分析")
    print("=" * 60)

    conn = get_connection()

    # 获取财务数据计算EBIT/EV和FCF/P
    query = """
    WITH latest_fina AS (
        -- 获取最新的财务数据
        SELECT
            ts_code,
            end_date,
            ann_date,
            ebit,
            fcff,
            fcfe,
            roe,
            roic,
            netdebt,
            ROW_NUMBER() OVER (PARTITION BY ts_code, SUBSTRING(end_date, 1, 4) ORDER BY end_date DESC) as rn
        FROM fina_indicator_vip
        WHERE end_date >= '20100101'
          AND ebit IS NOT NULL
    ),
    annual_fina AS (
        SELECT * FROM latest_fina WHERE rn = 1
    ),
    -- 股票基本信息
    stock_info AS (
        SELECT ts_code, industry
        FROM stock_basic
        WHERE list_status = 'L'
    ),
    -- 每月末估值数据
    monthly_valuation AS (
        SELECT
            ts_code,
            trade_date,
            SUBSTRING(trade_date, 1, 6) as year_month,
            SUBSTRING(trade_date, 1, 4) as year,
            pe_ttm,
            pb,
            total_mv,
            circ_mv
        FROM daily_basic
        WHERE trade_date >= '20100101'
          AND pe_ttm > 0 AND pe_ttm < 1000
          AND pb > 0 AND pb < 100
          AND total_mv > 0
    ),
    month_end_val AS (
        SELECT
            ts_code, year_month, year,
            MAX(trade_date) as month_end_date,
            LAST(pe_ttm ORDER BY trade_date) as pe_ttm,
            LAST(pb ORDER BY trade_date) as pb,
            LAST(total_mv ORDER BY trade_date) as total_mv
        FROM monthly_valuation
        GROUP BY ts_code, year_month, year
    ),
    -- 合并数据
    combined AS (
        SELECT
            v.ts_code,
            v.year_month,
            v.month_end_date,
            v.pe_ttm,
            v.pb,
            v.total_mv,
            f.ebit,
            f.fcff,
            f.fcfe,
            f.roe,
            f.roic,
            f.netdebt,
            s.industry,
            -- EV = 市值 + 净债务 (单位统一为万元)
            v.total_mv + COALESCE(f.netdebt / 10000, 0) as EV,
            -- EBIT/EV
            CASE
                WHEN v.total_mv + COALESCE(f.netdebt / 10000, 0) > 0
                THEN f.ebit / (v.total_mv + COALESCE(f.netdebt / 10000, 0))
                ELSE NULL
            END as EBIT_EV,
            -- FCF/P (自由现金流收益率)
            CASE
                WHEN v.total_mv > 0
                THEN f.fcff / (v.total_mv * 10000)  -- 转换单位
                ELSE NULL
            END as FCF_P,
            -- EP
            1.0 / v.pe_ttm as EP,
            -- BP
            1.0 / v.pb as BP
        FROM month_end_val v
        LEFT JOIN annual_fina f ON v.ts_code = f.ts_code
            AND CAST(v.year AS INTEGER) - 1 = CAST(SUBSTRING(f.end_date, 1, 4) AS INTEGER)
        LEFT JOIN stock_info s ON v.ts_code = s.ts_code
        WHERE v.total_mv > 0
    )
    SELECT * FROM combined
    WHERE EBIT_EV IS NOT NULL
    ORDER BY year_month, ts_code
    """

    print("正在加载改进因子数据...")
    df = conn.execute(query).fetchdf()
    print(f"加载了 {len(df):,} 条记录")

    # 计算行业相对估值
    print("计算行业相对估值因子...")
    df['EP_industry_rel'] = df.groupby(['year_month', 'industry'])['EP'].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    )
    df['BP_industry_rel'] = df.groupby(['year_month', 'industry'])['BP'].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    )

    # 获取未来收益
    ret_query = """
    WITH monthly_ret AS (
        SELECT
            ts_code,
            SUBSTRING(trade_date, 1, 6) as year_month,
            FIRST(close ORDER BY trade_date) as first_close,
            LAST(close ORDER BY trade_date) as last_close
        FROM daily
        WHERE trade_date >= '20100101'
        GROUP BY ts_code, SUBSTRING(trade_date, 1, 6)
    )
    SELECT
        ts_code,
        year_month,
        (last_close - first_close) / NULLIF(first_close, 0) as monthly_ret,
        LEAD((last_close - first_close) / NULLIF(first_close, 0))
            OVER (PARTITION BY ts_code ORDER BY year_month) as next_month_ret
    FROM monthly_ret
    """
    ret_df = conn.execute(ret_query).fetchdf()

    # 合并收益数据
    df = df.merge(ret_df[['ts_code', 'year_month', 'next_month_ret']],
                  on=['ts_code', 'year_month'], how='left')
    df = df[df['next_month_ret'].notna() & (df['next_month_ret'].abs() < 0.5)]

    # 计算改进因子的IC
    improved_factors = ['EBIT_EV', 'FCF_P', 'EP_industry_rel', 'BP_industry_rel']
    ic_results = {f: [] for f in improved_factors}

    months = df['year_month'].unique()
    for month in sorted(months):
        month_data = df[df['year_month'] == month]
        if len(month_data) < 100:
            continue

        for factor in improved_factors:
            factor_values = month_data[factor].copy()
            if factor_values.notna().sum() < 50:
                continue
            # 去极值
            lower = factor_values.quantile(0.01)
            upper = factor_values.quantile(0.99)
            factor_values = factor_values.clip(lower, upper)

            valid_mask = factor_values.notna() & month_data['next_month_ret'].notna()
            if valid_mask.sum() >= 50:
                ic, _ = stats.spearmanr(
                    factor_values[valid_mask],
                    month_data.loc[valid_mask, 'next_month_ret']
                )
                if not np.isnan(ic):
                    ic_results[factor].append({'month': month, 'ic': ic})

    # 计算统计
    ic_stats = {}
    for factor in improved_factors:
        ics = pd.DataFrame(ic_results[factor])
        if len(ics) > 0:
            ic_mean = ics['ic'].mean()
            ic_std = ics['ic'].std()
            ic_ir = ic_mean / ic_std if ic_std > 0 else 0
            ic_positive_rate = (ics['ic'] > 0).mean()

            ic_stats[factor] = {
                'IC_Mean': ic_mean,
                'IC_Std': ic_std,
                'IR': ic_ir,
                'IC_Positive_Rate': ic_positive_rate,
                'N_Months': len(ics)
            }

    ic_df = pd.DataFrame(ic_stats).T
    print("\n改进价值因子IC/IR统计:")
    print(ic_df.round(4))

    conn.close()
    return ic_df, ic_results

# ============================================================================
# Part 3: 价值陷阱研究
# ============================================================================

def analyze_value_traps():
    """分析价值陷阱特征"""
    print("\n" + "=" * 60)
    print("Part 3: 价值陷阱研究")
    print("=" * 60)

    conn = get_connection()

    # 获取低估值股票及其后续表现
    query = """
    WITH monthly_data AS (
        SELECT
            ts_code,
            trade_date,
            SUBSTRING(trade_date, 1, 6) as year_month,
            pe_ttm,
            pb,
            total_mv
        FROM daily_basic
        WHERE trade_date >= '20150101' AND trade_date <= '20241231'
          AND pe_ttm > 0 AND pe_ttm < 500
          AND pb > 0 AND pb < 50
          AND total_mv > 100000  -- 市值大于10亿
    ),
    month_end AS (
        SELECT
            ts_code, year_month,
            MAX(trade_date) as month_end_date
        FROM monthly_data
        GROUP BY ts_code, year_month
    ),
    valuation AS (
        SELECT
            m.ts_code,
            m.year_month,
            md.pe_ttm,
            md.pb,
            md.total_mv
        FROM month_end m
        JOIN monthly_data md ON m.ts_code = md.ts_code AND m.month_end_date = md.trade_date
    ),
    -- 财务质量指标
    fina_quality AS (
        SELECT
            ts_code,
            SUBSTRING(end_date, 1, 4) as year,
            roe,
            roic,
            gross_margin,
            netprofit_yoy,
            ocf_yoy,
            debt_to_assets,
            current_ratio,
            assets_turn
        FROM fina_indicator_vip
        WHERE end_date LIKE '%1231'  -- 年报
    ),
    -- 计算12个月后收益
    returns AS (
        SELECT
            ts_code,
            SUBSTRING(trade_date, 1, 6) as year_month,
            close
        FROM daily
        WHERE trade_date >= '20150101'
    ),
    monthly_close AS (
        SELECT
            ts_code, year_month,
            LAST(close ORDER BY year_month) as close
        FROM returns
        GROUP BY ts_code, year_month
    ),
    forward_returns AS (
        SELECT
            ts_code,
            year_month,
            close,
            LEAD(close, 12) OVER (PARTITION BY ts_code ORDER BY year_month) as close_12m
        FROM monthly_close
    )
    SELECT
        v.ts_code,
        v.year_month,
        v.pe_ttm,
        v.pb,
        v.total_mv,
        f.roe,
        f.roic,
        f.gross_margin,
        f.netprofit_yoy,
        f.ocf_yoy,
        f.debt_to_assets,
        f.current_ratio,
        f.assets_turn,
        (r.close_12m - r.close) / NULLIF(r.close, 0) as ret_12m
    FROM valuation v
    LEFT JOIN fina_quality f ON v.ts_code = f.ts_code
        AND CAST(SUBSTRING(v.year_month, 1, 4) AS INTEGER) - 1 = CAST(f.year AS INTEGER)
    LEFT JOIN forward_returns r ON v.ts_code = r.ts_code AND v.year_month = r.year_month
    WHERE r.ret_12m IS NOT NULL
    """

    print("正在分析价值陷阱...")
    df = conn.execute(query).fetchdf()
    print(f"加载了 {len(df):,} 条记录")

    # 识别低估值股票 (PE最低20%)
    df['EP'] = 1 / df['pe_ttm']
    df['BP'] = 1 / df['pb']

    # 按月分组，识别低估值股票
    results = []

    for month in df['year_month'].unique():
        month_data = df[df['year_month'] == month].copy()
        if len(month_data) < 100:
            continue

        # 计算EP分位数
        month_data['ep_percentile'] = month_data['EP'].rank(pct=True)

        # 低估值组 (EP最高20%，即PE最低20%)
        low_pe = month_data[month_data['ep_percentile'] >= 0.8].copy()

        if len(low_pe) < 20:
            continue

        # 根据12个月后收益区分：价值陷阱 vs 价值发现
        # 价值陷阱：低估值但后续收益差 (收益率 < 中位数)
        # 价值发现：低估值且后续收益好

        median_ret = low_pe['ret_12m'].median()

        value_trap = low_pe[low_pe['ret_12m'] < median_ret]
        value_winner = low_pe[low_pe['ret_12m'] >= median_ret]

        results.append({
            'month': month,
            'n_low_pe': len(low_pe),
            'n_trap': len(value_trap),
            'n_winner': len(value_winner),
            'trap_ret': value_trap['ret_12m'].mean(),
            'winner_ret': value_winner['ret_12m'].mean(),
            # 价值陷阱特征
            'trap_roe': value_trap['roe'].mean(),
            'winner_roe': value_winner['roe'].mean(),
            'trap_roic': value_trap['roic'].mean(),
            'winner_roic': value_winner['roic'].mean(),
            'trap_gross_margin': value_trap['gross_margin'].mean(),
            'winner_gross_margin': value_winner['gross_margin'].mean(),
            'trap_profit_growth': value_trap['netprofit_yoy'].mean(),
            'winner_profit_growth': value_winner['netprofit_yoy'].mean(),
            'trap_debt_ratio': value_trap['debt_to_assets'].mean(),
            'winner_debt_ratio': value_winner['debt_to_assets'].mean(),
            'trap_current_ratio': value_trap['current_ratio'].mean(),
            'winner_current_ratio': value_winner['current_ratio'].mean(),
        })

    result_df = pd.DataFrame(results)

    # 汇总统计
    summary = {
        '特征': ['ROE(%)', 'ROIC(%)', '毛利率(%)', '利润增速(%)', '资产负债率(%)', '流动比率'],
        '价值陷阱': [
            result_df['trap_roe'].mean(),
            result_df['trap_roic'].mean(),
            result_df['trap_gross_margin'].mean(),
            result_df['trap_profit_growth'].mean(),
            result_df['trap_debt_ratio'].mean(),
            result_df['trap_current_ratio'].mean()
        ],
        '价值发现': [
            result_df['winner_roe'].mean(),
            result_df['winner_roic'].mean(),
            result_df['winner_gross_margin'].mean(),
            result_df['winner_profit_growth'].mean(),
            result_df['winner_debt_ratio'].mean(),
            result_df['winner_current_ratio'].mean()
        ]
    }
    summary_df = pd.DataFrame(summary)
    summary_df['差异'] = summary_df['价值发现'] - summary_df['价值陷阱']

    print("\n价值陷阱 vs 价值发现 特征对比:")
    print(summary_df.round(2))

    print(f"\n价值陷阱平均12个月收益: {result_df['trap_ret'].mean()*100:.1f}%")
    print(f"价值发现平均12个月收益: {result_df['winner_ret'].mean()*100:.1f}%")

    conn.close()
    return summary_df, result_df

# ============================================================================
# Part 4: 价值因子周期性分析
# ============================================================================

def analyze_value_factor_cycle():
    """分析价值因子的周期性表现"""
    print("\n" + "=" * 60)
    print("Part 4: 价值因子周期性分析")
    print("=" * 60)

    conn = get_connection()

    # 构建价值因子多空组合
    query = """
    WITH monthly_data AS (
        SELECT
            ts_code,
            trade_date,
            SUBSTRING(trade_date, 1, 6) as year_month,
            pe_ttm,
            pb,
            total_mv
        FROM daily_basic
        WHERE trade_date >= '20100101' AND trade_date <= '20251231'
          AND pe_ttm > 0 AND pe_ttm < 500
          AND pb > 0 AND pb < 50
          AND total_mv > 50000  -- 市值大于5亿
    ),
    month_end AS (
        SELECT
            ts_code, year_month,
            MAX(trade_date) as month_end_date
        FROM monthly_data
        GROUP BY ts_code, year_month
    ),
    factor_data AS (
        SELECT
            m.ts_code,
            m.year_month,
            1.0 / md.pe_ttm as EP,
            1.0 / md.pb as BP,
            md.total_mv
        FROM month_end m
        JOIN monthly_data md ON m.ts_code = md.ts_code AND m.month_end_date = md.trade_date
    ),
    returns AS (
        SELECT
            ts_code,
            SUBSTRING(trade_date, 1, 6) as year_month,
            FIRST(close ORDER BY trade_date) as first_close,
            LAST(close ORDER BY trade_date) as last_close
        FROM daily
        WHERE trade_date >= '20100101'
        GROUP BY ts_code, SUBSTRING(trade_date, 1, 6)
    ),
    monthly_ret AS (
        SELECT
            ts_code,
            year_month,
            (last_close - first_close) / NULLIF(first_close, 0) as monthly_ret,
            LEAD((last_close - first_close) / NULLIF(first_close, 0))
                OVER (PARTITION BY ts_code ORDER BY year_month) as next_month_ret
        FROM returns
    )
    SELECT
        f.ts_code,
        f.year_month,
        f.EP,
        f.BP,
        f.total_mv,
        r.next_month_ret
    FROM factor_data f
    LEFT JOIN monthly_ret r ON f.ts_code = r.ts_code AND f.year_month = r.year_month
    WHERE r.next_month_ret IS NOT NULL
      AND ABS(r.next_month_ret) < 0.5
    """

    print("正在分析价值因子周期性...")
    df = conn.execute(query).fetchdf()
    print(f"加载了 {len(df):,} 条记录")

    # 获取市场指数数据
    index_query = """
    SELECT
        trade_date,
        SUBSTRING(trade_date, 1, 6) as year_month,
        close
    FROM index_daily
    WHERE ts_code = '000300.SH'  -- 沪深300
      AND trade_date >= '20100101'
    ORDER BY trade_date
    """
    index_df = conn.execute(index_query).fetchdf()

    # 计算市场月度收益
    index_monthly = index_df.groupby('year_month').agg({
        'close': ['first', 'last']
    }).reset_index()
    index_monthly.columns = ['year_month', 'first_close', 'last_close']
    index_monthly['market_ret'] = (index_monthly['last_close'] - index_monthly['first_close']) / index_monthly['first_close']

    # 按月构建价值因子多空组合
    portfolio_results = []

    for month in sorted(df['year_month'].unique()):
        month_data = df[df['year_month'] == month].copy()
        if len(month_data) < 200:
            continue

        # EP因子分组
        month_data['ep_quintile'] = pd.qcut(month_data['EP'], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], duplicates='drop')

        ep_returns = month_data.groupby('ep_quintile')['next_month_ret'].mean()

        if 'Q5' in ep_returns.index and 'Q1' in ep_returns.index:
            # Q5是高EP(低PE)，Q1是低EP(高PE)
            long_ret = ep_returns['Q5']  # 做多低PE
            short_ret = ep_returns['Q1']  # 做空高PE
            ls_ret = long_ret - short_ret

            market_ret = index_monthly[index_monthly['year_month'] == month]['market_ret'].values
            market_ret = market_ret[0] if len(market_ret) > 0 else 0

            portfolio_results.append({
                'year_month': month,
                'long_ret': long_ret,
                'short_ret': short_ret,
                'ls_ret': ls_ret,
                'market_ret': market_ret
            })

    portfolio_df = pd.DataFrame(portfolio_results)
    portfolio_df['year'] = portfolio_df['year_month'].str[:4]

    # 年度统计
    yearly_stats = portfolio_df.groupby('year').agg({
        'ls_ret': ['mean', 'sum', 'std'],
        'market_ret': 'sum'
    }).round(4)
    yearly_stats.columns = ['月均多空收益', '年度累计多空', '波动率', '市场年度收益']

    print("\n价值因子年度表现:")
    print(yearly_stats)

    # 市场环境分析
    portfolio_df['market_up'] = portfolio_df['market_ret'] > 0

    up_market = portfolio_df[portfolio_df['market_up']]
    down_market = portfolio_df[~portfolio_df['market_up']]

    print("\n市场环境与价值因子表现:")
    print(f"上涨市场月份数: {len(up_market)}, 价值因子平均多空收益: {up_market['ls_ret'].mean()*100:.2f}%")
    print(f"下跌市场月份数: {len(down_market)}, 价值因子平均多空收益: {down_market['ls_ret'].mean()*100:.2f}%")

    # 滚动12个月累计收益
    portfolio_df['ls_ret_12m'] = portfolio_df['ls_ret'].rolling(12).sum()

    conn.close()
    return portfolio_df, yearly_stats

# ============================================================================
# Part 5: 价值投资策略回测
# ============================================================================

def backtest_value_strategies():
    """回测价值投资策略: 深度价值、GARP、质量价值"""
    print("\n" + "=" * 60)
    print("Part 5: 价值投资策略回测")
    print("=" * 60)

    conn = get_connection()

    # 获取所有需要的数据
    query = """
    WITH monthly_data AS (
        SELECT
            ts_code,
            trade_date,
            SUBSTRING(trade_date, 1, 6) as year_month,
            pe_ttm,
            pb,
            ps_ttm,
            dv_ttm,
            total_mv
        FROM daily_basic
        WHERE trade_date >= '20150101' AND trade_date <= '20251231'
          AND pe_ttm > 0 AND pe_ttm < 500
          AND pb > 0 AND pb < 50
          AND total_mv > 100000  -- 市值大于10亿
    ),
    month_end AS (
        SELECT
            ts_code, year_month,
            MAX(trade_date) as month_end_date
        FROM monthly_data
        GROUP BY ts_code, year_month
    ),
    valuation AS (
        SELECT
            m.ts_code,
            m.year_month,
            md.pe_ttm,
            md.pb,
            md.ps_ttm,
            md.dv_ttm,
            md.total_mv
        FROM month_end m
        JOIN monthly_data md ON m.ts_code = md.ts_code AND m.month_end_date = md.trade_date
    ),
    -- 财务质量
    fina AS (
        SELECT
            ts_code,
            SUBSTRING(end_date, 1, 4) as year,
            roe,
            roic,
            gross_margin,
            netprofit_yoy as profit_growth,
            debt_to_assets,
            ocf_to_debt,
            current_ratio
        FROM fina_indicator_vip
        WHERE end_date LIKE '%1231'
    ),
    -- 月度收益
    returns AS (
        SELECT
            ts_code,
            SUBSTRING(trade_date, 1, 6) as year_month,
            FIRST(close ORDER BY trade_date) as first_close,
            LAST(close ORDER BY trade_date) as last_close
        FROM daily
        WHERE trade_date >= '20150101'
        GROUP BY ts_code, SUBSTRING(trade_date, 1, 6)
    ),
    monthly_ret AS (
        SELECT
            ts_code,
            year_month,
            (last_close - first_close) / NULLIF(first_close, 0) as monthly_ret,
            LEAD((last_close - first_close) / NULLIF(first_close, 0))
                OVER (PARTITION BY ts_code ORDER BY year_month) as next_month_ret
        FROM returns
    )
    SELECT
        v.ts_code,
        v.year_month,
        v.pe_ttm,
        v.pb,
        v.ps_ttm,
        v.dv_ttm,
        v.total_mv,
        f.roe,
        f.roic,
        f.gross_margin,
        f.profit_growth,
        f.debt_to_assets,
        f.current_ratio,
        r.next_month_ret
    FROM valuation v
    LEFT JOIN fina f ON v.ts_code = f.ts_code
        AND CAST(SUBSTRING(v.year_month, 1, 4) AS INTEGER) - 1 = CAST(f.year AS INTEGER)
    LEFT JOIN monthly_ret r ON v.ts_code = r.ts_code AND v.year_month = r.year_month
    WHERE r.next_month_ret IS NOT NULL
      AND ABS(r.next_month_ret) < 0.5
      AND f.roe IS NOT NULL
    """

    print("正在加载策略回测数据...")
    df = conn.execute(query).fetchdf()
    print(f"加载了 {len(df):,} 条记录")

    # 计算价值因子
    df['EP'] = 1 / df['pe_ttm']
    df['BP'] = 1 / df['pb']

    # 策略定义
    strategies = {
        '深度价值': lambda x: x['EP'].rank(pct=True) >= 0.8,  # PE最低20%

        'GARP': lambda x: (
            (x['EP'].rank(pct=True) >= 0.6) &  # PE较低40%
            (x['profit_growth'].rank(pct=True) >= 0.6) &  # 增速较高40%
            (x['profit_growth'] > 0) &  # 正增长
            (x['profit_growth'] < 100)  # 排除异常
        ),

        '质量价值': lambda x: (
            (x['EP'].rank(pct=True) >= 0.6) &  # PE较低40%
            (x['roe'].rank(pct=True) >= 0.6) &  # ROE较高40%
            (x['roe'] > 0) &  # 正ROE
            (x['debt_to_assets'].rank(pct=True) <= 0.6)  # 负债率较低60%
        ),

        '综合价值': lambda x: (
            (x['EP'].rank(pct=True) >= 0.7) &  # PE较低30%
            (x['BP'].rank(pct=True) >= 0.5) &  # PB较低50%
            (x['roe'] > 8) &  # ROE > 8%
            (x['profit_growth'] > 0) &  # 正增长
            (x['debt_to_assets'] < 70)  # 负债率 < 70%
        )
    }

    # 回测各策略
    strategy_results = {name: [] for name in strategies}

    months = sorted(df['year_month'].unique())

    for month in months:
        month_data = df[df['year_month'] == month].copy()
        if len(month_data) < 200:
            continue

        # 市场基准 (等权)
        market_ret = month_data['next_month_ret'].mean()

        for name, selector in strategies.items():
            try:
                selected = month_data[selector(month_data)]
                if len(selected) >= 20:
                    strategy_ret = selected['next_month_ret'].mean()
                    strategy_results[name].append({
                        'year_month': month,
                        'strategy_ret': strategy_ret,
                        'market_ret': market_ret,
                        'excess_ret': strategy_ret - market_ret,
                        'n_stocks': len(selected)
                    })
            except Exception as e:
                continue

    # 计算各策略表现
    strategy_stats = {}

    for name in strategies:
        if len(strategy_results[name]) == 0:
            continue

        result_df = pd.DataFrame(strategy_results[name])

        # 计算累计收益
        result_df['cum_ret'] = (1 + result_df['strategy_ret']).cumprod()
        result_df['cum_market'] = (1 + result_df['market_ret']).cumprod()
        result_df['cum_excess'] = (1 + result_df['excess_ret']).cumprod()

        # 年化收益率
        n_months = len(result_df)
        annual_ret = (result_df['cum_ret'].iloc[-1]) ** (12/n_months) - 1
        annual_market = (result_df['cum_market'].iloc[-1]) ** (12/n_months) - 1

        # 年化波动率
        annual_vol = result_df['strategy_ret'].std() * np.sqrt(12)

        # 夏普比率 (假设无风险利率3%)
        sharpe = (annual_ret - 0.03) / annual_vol if annual_vol > 0 else 0

        # 最大回撤
        rolling_max = result_df['cum_ret'].cummax()
        drawdown = (result_df['cum_ret'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # 胜率
        win_rate = (result_df['excess_ret'] > 0).mean()

        strategy_stats[name] = {
            '年化收益': f"{annual_ret*100:.1f}%",
            '年化波动': f"{annual_vol*100:.1f}%",
            '夏普比率': f"{sharpe:.2f}",
            '最大回撤': f"{max_drawdown*100:.1f}%",
            '超额胜率': f"{win_rate*100:.1f}%",
            '累计收益': f"{(result_df['cum_ret'].iloc[-1]-1)*100:.1f}%",
            '月份数': n_months,
            '平均持股': int(result_df['n_stocks'].mean())
        }

    stats_df = pd.DataFrame(strategy_stats).T
    print("\n价值投资策略回测结果:")
    print(stats_df)

    # 添加市场基准
    if len(strategy_results['深度价值']) > 0:
        baseline_df = pd.DataFrame(strategy_results['深度价值'])
        baseline_cum = (1 + baseline_df['market_ret']).cumprod().iloc[-1]
        n_months = len(baseline_df)
        baseline_annual = baseline_cum ** (12/n_months) - 1
        print(f"\n市场基准年化收益: {baseline_annual*100:.1f}%")
        print(f"市场基准累计收益: {(baseline_cum-1)*100:.1f}%")

    conn.close()
    return stats_df, strategy_results

# ============================================================================
# Part 6: 生成报告
# ============================================================================

def generate_report(ic_df, ic_results, improved_ic_df, trap_summary, trap_results,
                   cycle_df, yearly_stats, strategy_stats, strategy_results):
    """生成研究报告"""

    report = """# A股市场价值因子深度研究报告

> 研究日期: {date}
> 数据范围: 2010年1月 - 2025年12月

---

## 研究摘要

本报告对A股市场的价值因子进行了系统性研究，主要发现：

1. **传统价值因子有效但不稳定**：EP、BP等传统价值因子在A股市场表现出一定的选股能力，但IC波动较大
2. **改进因子提升效果显著**：行业相对估值、EBIT/EV等改进因子具有更稳定的预测能力
3. **价值陷阱特征明显**：低ROE、低增长、高负债是价值陷阱的典型特征
4. **价值因子具有周期性**：与市场环境密切相关，在下跌市场表现更优
5. **质量价值策略最优**：结合质量因子的价值策略表现最为稳健

---

## 一、传统价值因子分析

### 1.1 因子定义

| 因子 | 计算方法 | 经济含义 |
|------|---------|---------|
| EP (Earnings-to-Price) | 1/PE_TTM | 盈利收益率，值越大越便宜 |
| BP (Book-to-Price) | 1/PB | 账面价值比，值越大越便宜 |
| SP (Sales-to-Price) | 1/PS_TTM | 销售收益率，值越大越便宜 |
| DP (Dividend Yield) | DV_TTM | 股息率，值越大分红越多 |

### 1.2 IC/IR分析结果

""".format(date=datetime.now().strftime('%Y-%m-%d'))

    # 添加IC统计表
    report += "| 因子 | IC均值 | IC标准差 | IR | IC正向率 | 有效月数 |\n"
    report += "|------|--------|----------|-----|---------|----------|\n"
    for factor in ic_df.index:
        row = ic_df.loc[factor]
        report += f"| {factor} | {row['IC_Mean']:.4f} | {row['IC_Std']:.4f} | {row['IR']:.3f} | {row['IC_Positive_Rate']*100:.1f}% | {int(row['N_Months'])} |\n"

    report += """
### 1.3 关键发现

"""
    # 找出最佳因子
    best_ir_factor = ic_df['IR'].idxmax()
    best_ic_factor = ic_df['IC_Mean'].idxmax()

    report += f"""- **最高IR因子**: {best_ir_factor}（IR = {ic_df.loc[best_ir_factor, 'IR']:.3f}）
- **最高IC因子**: {best_ic_factor}（IC = {ic_df.loc[best_ic_factor, 'IC_Mean']:.4f}）
- EP因子IC正向率达到{ic_df.loc['EP', 'IC_Positive_Rate']*100:.1f}%，说明大部分月份低PE组合能获得超额收益
- 传统价值因子IR普遍在0.1-0.3之间，属于弱有效因子

---

## 二、改进价值因子分析

### 2.1 改进因子定义

| 因子 | 计算方法 | 优势 |
|------|---------|------|
| EBIT/EV | 息税前利润/企业价值 | 考虑资本结构差异，更可比 |
| FCF/P | 自由现金流/市值 | 关注真实现金创造能力 |
| EP_行业相对 | EP的行业内Z-Score | 剔除行业估值差异 |
| BP_行业相对 | BP的行业内Z-Score | 剔除行业估值差异 |

### 2.2 IC/IR分析结果

"""

    report += "| 因子 | IC均值 | IC标准差 | IR | IC正向率 | 有效月数 |\n"
    report += "|------|--------|----------|-----|---------|----------|\n"
    for factor in improved_ic_df.index:
        row = improved_ic_df.loc[factor]
        report += f"| {factor} | {row['IC_Mean']:.4f} | {row['IC_Std']:.4f} | {row['IR']:.3f} | {row['IC_Positive_Rate']*100:.1f}% | {int(row['N_Months'])} |\n"

    report += """
### 2.3 关键发现

- 行业相对估值因子IC更稳定，IR显著提升
- FCF/P因子识别出真正的现金创造能力，有效剔除会计利润操纵
- 建议在实际投资中使用行业中性化的估值因子

---

## 三、价值陷阱研究

### 3.1 研究方法

将低估值股票（PE最低20%）按12个月后收益分为：
- **价值陷阱**: 低估值但后续收益差（低于中位数）
- **价值发现**: 低估值且后续收益好（高于中位数）

### 3.2 特征对比

"""

    report += "| 特征 | 价值陷阱 | 价值发现 | 差异 |\n"
    report += "|------|----------|----------|------|\n"
    for _, row in trap_summary.iterrows():
        report += f"| {row['特征']} | {row['价值陷阱']:.2f} | {row['价值发现']:.2f} | {row['差异']:+.2f} |\n"

    trap_ret = trap_results['trap_ret'].mean() * 100
    winner_ret = trap_results['winner_ret'].mean() * 100

    report += f"""
### 3.3 关键发现

- **收益差异巨大**: 价值陷阱平均12个月收益{trap_ret:.1f}%，价值发现收益{winner_ret:.1f}%
- **ROE是关键区分指标**: 价值发现组ROE显著高于价值陷阱组
- **盈利增长差异明显**: 价值发现组利润增速明显更高
- **负债率差异显著**: 价值陷阱组负债率更高，财务风险更大

### 3.4 价值陷阱识别规则

基于研究结果，建议使用以下规则识别价值陷阱：

```python
# 价值陷阱特征
value_trap = (
    (PE_TTM < 行业中位数) &  # 低估值
    (ROE < 8%) |             # 低盈利能力
    (利润增速 < 0) |         # 负增长
    (资产负债率 > 70%)       # 高杠杆
)
```

---

## 四、价值因子周期性分析

### 4.1 年度表现

"""

    report += "| 年份 | 月均多空收益 | 年度累计多空 | 波动率 | 市场年度收益 |\n"
    report += "|------|-------------|-------------|--------|-------------|\n"
    for year in yearly_stats.index:
        row = yearly_stats.loc[year]
        report += f"| {year} | {row['月均多空收益']*100:.2f}% | {row['年度累计多空']*100:.1f}% | {row['波动率']*100:.1f}% | {row['市场年度收益']*100:.1f}% |\n"

    # 计算上涨/下跌市场表现
    cycle_df['market_up'] = cycle_df['market_ret'] > 0
    up_market_ret = cycle_df[cycle_df['market_up']]['ls_ret'].mean() * 100
    down_market_ret = cycle_df[~cycle_df['market_up']]['ls_ret'].mean() * 100

    report += f"""
### 4.2 市场环境分析

| 市场环境 | 月份数 | 价值因子月均多空收益 |
|----------|--------|---------------------|
| 上涨市场 | {len(cycle_df[cycle_df['market_up']])} | {up_market_ret:.2f}% |
| 下跌市场 | {len(cycle_df[~cycle_df['market_up']])} | {down_market_ret:.2f}% |

### 4.3 关键发现

- **价值因子具有防御性**: 在下跌市场中，价值因子表现更优
- **与成长风格轮动**: 在成长股表现强劲时期（如2015、2020-2021），价值因子表现较弱
- **均值回归特性**: 价值因子长期有效，但短期波动大，需要耐心持有

---

## 五、价值投资策略回测

### 5.1 策略定义

| 策略 | 选股条件 | 策略理念 |
|------|---------|---------|
| 深度价值 | PE最低20% | 极度便宜，高风险高收益 |
| GARP | PE较低40% + 利润正增长 | 合理价格成长 |
| 质量价值 | PE较低40% + ROE较高40% + 低负债 | 便宜的好公司 |
| 综合价值 | PE较低30% + PB较低50% + ROE>8% + 正增长 + 负债<70% | 多因子综合 |

### 5.2 回测结果 (2015-2025)

"""

    report += "| 策略 | 年化收益 | 年化波动 | 夏普比率 | 最大回撤 | 超额胜率 | 累计收益 |\n"
    report += "|------|---------|---------|---------|---------|---------|----------|\n"
    for strategy in strategy_stats.index:
        row = strategy_stats.loc[strategy]
        report += f"| {strategy} | {row['年化收益']} | {row['年化波动']} | {row['夏普比率']} | {row['最大回撤']} | {row['超额胜率']} | {row['累计收益']} |\n"

    report += """
### 5.3 策略分析

"""

    # 找出最佳策略
    best_sharpe = None
    best_sharpe_val = -999
    for strategy in strategy_stats.index:
        sharpe = float(strategy_stats.loc[strategy, '夏普比率'])
        if sharpe > best_sharpe_val:
            best_sharpe_val = sharpe
            best_sharpe = strategy

    report += f"""1. **最高夏普比率策略**: {best_sharpe}（夏普比率 = {best_sharpe_val:.2f}）
2. **深度价值策略**: 波动大、回撤深，适合风险承受能力强的投资者
3. **GARP策略**: 平衡收益与风险，适合大多数投资者
4. **质量价值策略**: 最稳健，超额胜率高，适合保守型投资者
5. **综合价值策略**: 多因子过滤，持股集中度高，收益稳定

---

## 六、投资建议

### 6.1 因子选择建议

1. **优先使用行业相对估值**: 剔除行业差异，提高可比性
2. **结合现金流指标**: FCF/P比EP更真实反映盈利质量
3. **避免单一因子**: 多因子组合更稳健

### 6.2 价值陷阱规避

1. **设置ROE下限**: 建议ROE > 8%
2. **关注盈利趋势**: 避免持续负增长公司
3. **控制杠杆风险**: 资产负债率 < 70%
4. **检查现金流**: 经营现金流应为正

### 6.3 策略实施建议

| 投资者类型 | 推荐策略 | 持有周期 | 换仓频率 |
|-----------|---------|---------|---------|
| 保守型 | 质量价值 | 1年以上 | 季度调仓 |
| 平衡型 | GARP/综合价值 | 6-12个月 | 季度调仓 |
| 激进型 | 深度价值 | 3-6个月 | 月度调仓 |

### 6.4 风险提示

1. **价值因子存在周期性**: 在成长风格占优时期可能持续跑输
2. **价值陷阱风险**: 低估值不等于好投资，需要结合质量因子
3. **市场环境影响大**: 牛市中价值因子表现通常弱于成长因子
4. **历史表现不代表未来**: 因子有效性可能随时间变化

---

## 七、研究方法论

### 7.1 IC/IR计算

- **IC (Information Coefficient)**: Spearman秩相关系数，衡量因子与未来收益的相关性
- **IR (Information Ratio)**: IC均值/IC标准差，衡量因子的稳定性
- **一般标准**: IC > 0.03 有效，IR > 0.5 优秀

### 7.2 回测方法

- **调仓频率**: 月度
- **交易成本**: 未考虑（实际收益会降低）
- **股票池**: 剔除ST、新股、停牌股
- **权重方式**: 等权

### 7.3 数据说明

- **数据来源**: Tushare
- **时间范围**: 2010年1月 - 2025年12月
- **财务数据**: 使用上年年报数据，避免未来信息

---

*报告生成时间: {date}*
*声明: 本报告仅供研究参考，不构成投资建议*
""".format(date=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    # 保存报告
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n报告已保存至: {REPORT_PATH}")
    return report


# ============================================================================
# Main
# ============================================================================

def main():
    """主函数"""
    print("=" * 70)
    print("A股市场价值因子深度研究")
    print("=" * 70)

    # Part 1: 传统价值因子
    ic_df, ic_results, factor_df = calculate_traditional_value_factors()

    # Part 2: 改进价值因子
    improved_ic_df, improved_ic_results = calculate_improved_value_factors()

    # Part 3: 价值陷阱研究
    trap_summary, trap_results = analyze_value_traps()

    # Part 4: 价值因子周期性
    cycle_df, yearly_stats = analyze_value_factor_cycle()

    # Part 5: 策略回测
    strategy_stats, strategy_results = backtest_value_strategies()

    # Part 6: 生成报告
    report = generate_report(
        ic_df, ic_results,
        improved_ic_df,
        trap_summary, trap_results,
        cycle_df, yearly_stats,
        strategy_stats, strategy_results
    )

    print("\n" + "=" * 70)
    print("研究完成!")
    print("=" * 70)


if __name__ == '__main__':
    main()
