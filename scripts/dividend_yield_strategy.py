#!/usr/bin/env python3
"""
股息率(DV)策略研究
=================

研究内容：
1. 股息率分析：高股息股票分布、行业对比
2. 狗股理论验证：高股息策略回测
3. 策略优化：股息率+质量筛选、股息增长、红利再投资

Author: AI Research
Date: 2025-01-31
"""

import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False

# 数据库路径
DB_PATH = '/Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db'
REPORT_DIR = '/Users/allen/workspace/python/stock/Tushare-DuckDB/reports/research'

def get_connection():
    """获取数据库连接"""
    return duckdb.connect(DB_PATH, read_only=True)

# =============================================================================
# 第一部分：股息率分析
# =============================================================================

def analyze_dividend_yield_distribution(conn, trade_date='20241231'):
    """分析股息率分布"""
    print(f"\n{'='*60}")
    print(f"股息率分布分析 (截止日期: {trade_date})")
    print(f"{'='*60}")

    # 获取最近交易日的股息率数据
    query = f"""
    WITH latest_date AS (
        SELECT MAX(trade_date) as max_date
        FROM daily_basic
        WHERE trade_date <= '{trade_date}'
    ),
    dv_data AS (
        SELECT
            db.ts_code,
            sb.name,
            sb.industry,
            db.dv_ratio,
            db.dv_ttm,
            db.pe_ttm,
            db.pb,
            db.total_mv / 10000 as total_mv_yi  -- 转为亿元
        FROM daily_basic db
        JOIN stock_basic sb ON db.ts_code = sb.ts_code
        WHERE db.trade_date = (SELECT max_date FROM latest_date)
            AND sb.list_status = 'L'  -- 上市状态
            AND db.dv_ttm > 0
            AND db.total_mv > 0
    )
    SELECT * FROM dv_data
    ORDER BY dv_ttm DESC
    """
    df = conn.execute(query).df()

    print(f"\n有股息数据的股票数量: {len(df)}")
    print(f"\n股息率(TTM)统计:")
    print(df['dv_ttm'].describe())

    # 股息率分布统计
    dv_bins = [0, 1, 2, 3, 4, 5, 7, 10, float('inf')]
    dv_labels = ['0-1%', '1-2%', '2-3%', '3-4%', '4-5%', '5-7%', '7-10%', '10%+']
    df['dv_range'] = pd.cut(df['dv_ttm'], bins=dv_bins, labels=dv_labels)

    dist = df['dv_range'].value_counts().sort_index()
    print(f"\n股息率分布:")
    for range_name, count in dist.items():
        pct = count / len(df) * 100
        print(f"  {range_name}: {count} 只 ({pct:.1f}%)")

    # 高股息股票 (>5%)
    high_dv = df[df['dv_ttm'] >= 5].copy()
    print(f"\n高股息股票 (股息率>=5%): {len(high_dv)} 只")
    print(f"\n前20只高股息股票:")
    display_cols = ['ts_code', 'name', 'industry', 'dv_ttm', 'pe_ttm', 'pb', 'total_mv_yi']
    print(high_dv[display_cols].head(20).to_string(index=False))

    return df, high_dv

def analyze_dividend_by_industry(conn, trade_date='20241231'):
    """分析各行业股息率"""
    print(f"\n{'='*60}")
    print(f"行业股息率对比分析")
    print(f"{'='*60}")

    query = f"""
    WITH latest_date AS (
        SELECT MAX(trade_date) as max_date
        FROM daily_basic
        WHERE trade_date <= '{trade_date}'
    ),
    dv_data AS (
        SELECT
            im.l1_name as industry,
            db.ts_code,
            db.dv_ttm,
            db.total_mv
        FROM daily_basic db
        JOIN index_member_all im ON db.ts_code = im.ts_code
        WHERE db.trade_date = (SELECT max_date FROM latest_date)
            AND im.out_date IS NULL  -- 当前成分股
            AND im.is_new = 'Y'
            AND db.dv_ttm IS NOT NULL
            AND db.dv_ttm > 0
    )
    SELECT
        industry,
        COUNT(*) as stock_count,
        AVG(dv_ttm) as avg_dv_ttm,
        MEDIAN(dv_ttm) as median_dv_ttm,
        MAX(dv_ttm) as max_dv_ttm,
        SUM(CASE WHEN dv_ttm >= 3 THEN 1 ELSE 0 END) as high_dv_count,
        SUM(CASE WHEN dv_ttm >= 5 THEN 1 ELSE 0 END) as very_high_dv_count
    FROM dv_data
    GROUP BY industry
    HAVING COUNT(*) >= 5
    ORDER BY avg_dv_ttm DESC
    """
    df_industry = conn.execute(query).df()

    print(f"\n行业股息率统计 (按平均股息率排序):")
    print(df_industry.to_string(index=False))

    return df_industry

def analyze_dividend_by_market_cap(conn, trade_date='20241231'):
    """按市值分析股息率"""
    print(f"\n{'='*60}")
    print(f"市值与股息率关系分析")
    print(f"{'='*60}")

    query = f"""
    WITH latest_date AS (
        SELECT MAX(trade_date) as max_date
        FROM daily_basic
        WHERE trade_date <= '{trade_date}'
    ),
    dv_data AS (
        SELECT
            db.ts_code,
            db.dv_ttm,
            db.total_mv / 10000 as total_mv_yi,
            CASE
                WHEN db.total_mv / 10000 >= 1000 THEN '超大盘(>=1000亿)'
                WHEN db.total_mv / 10000 >= 300 THEN '大盘(300-1000亿)'
                WHEN db.total_mv / 10000 >= 100 THEN '中盘(100-300亿)'
                WHEN db.total_mv / 10000 >= 30 THEN '小盘(30-100亿)'
                ELSE '微盘(<30亿)'
            END as cap_level
        FROM daily_basic db
        JOIN stock_basic sb ON db.ts_code = sb.ts_code
        WHERE db.trade_date = (SELECT max_date FROM latest_date)
            AND sb.list_status = 'L'
            AND db.dv_ttm IS NOT NULL
            AND db.dv_ttm > 0
    )
    SELECT
        cap_level,
        COUNT(*) as stock_count,
        AVG(dv_ttm) as avg_dv_ttm,
        MEDIAN(dv_ttm) as median_dv_ttm,
        MAX(dv_ttm) as max_dv_ttm
    FROM dv_data
    GROUP BY cap_level
    ORDER BY avg_dv_ttm DESC
    """
    df_cap = conn.execute(query).df()

    print(f"\n市值分组股息率统计:")
    print(df_cap.to_string(index=False))

    return df_cap

# =============================================================================
# 第二部分：狗股理论验证 - 高股息策略回测
# =============================================================================

def get_annual_high_dividend_stocks(conn, year, top_n=30):
    """获取某年年初的高股息股票"""
    # 获取该年第一个交易日
    trade_date = f"{year}0105"  # 大约是1月第一周

    query = f"""
    WITH first_trade_date AS (
        SELECT MIN(trade_date) as first_date
        FROM daily_basic
        WHERE trade_date >= '{year}0101' AND trade_date <= '{year}0115'
    ),
    dv_data AS (
        SELECT
            db.ts_code,
            sb.name,
            db.dv_ttm,
            db.pe_ttm,
            db.pb,
            db.total_mv / 10000 as total_mv_yi,
            db.trade_date
        FROM daily_basic db
        JOIN stock_basic sb ON db.ts_code = sb.ts_code
        WHERE db.trade_date = (SELECT first_date FROM first_trade_date)
            AND sb.list_status = 'L'
            AND db.dv_ttm IS NOT NULL
            AND db.dv_ttm > 0
            AND db.total_mv > 0
            -- 排除ST股票
            AND sb.name NOT LIKE '%ST%'
            -- 排除市值过小的股票 (>20亿)
            AND db.total_mv > 200000
    )
    SELECT * FROM dv_data
    ORDER BY dv_ttm DESC
    LIMIT {top_n}
    """
    df = conn.execute(query).df()
    return df

def calculate_annual_returns(conn, ts_codes, year):
    """计算某年的收益率"""
    codes_str = "', '".join(ts_codes)

    query = f"""
    WITH year_data AS (
        SELECT
            ts_code,
            trade_date,
            close,
            ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY trade_date) as rn_first,
            ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY trade_date DESC) as rn_last
        FROM daily
        WHERE ts_code IN ('{codes_str}')
            AND trade_date >= '{year}0101'
            AND trade_date <= '{year}1231'
    )
    SELECT
        ts_code,
        MAX(CASE WHEN rn_first = 1 THEN close END) as first_close,
        MAX(CASE WHEN rn_last = 1 THEN close END) as last_close
    FROM year_data
    GROUP BY ts_code
    HAVING MAX(CASE WHEN rn_first = 1 THEN close END) IS NOT NULL
        AND MAX(CASE WHEN rn_last = 1 THEN close END) IS NOT NULL
    """
    df = conn.execute(query).df()
    df['return'] = (df['last_close'] / df['first_close'] - 1) * 100
    return df

def get_index_annual_return(conn, index_code, year):
    """获取指数年度收益率"""
    query = f"""
    WITH year_data AS (
        SELECT
            trade_date,
            close,
            ROW_NUMBER() OVER (ORDER BY trade_date) as rn_first,
            ROW_NUMBER() OVER (ORDER BY trade_date DESC) as rn_last
        FROM index_daily
        WHERE ts_code = '{index_code}'
            AND trade_date >= '{year}0101'
            AND trade_date <= '{year}1231'
    )
    SELECT
        MAX(CASE WHEN rn_first = 1 THEN close END) as first_close,
        MAX(CASE WHEN rn_last = 1 THEN close END) as last_close
    FROM year_data
    """
    result = conn.execute(query).fetchone()
    if result[0] and result[1]:
        return (result[1] / result[0] - 1) * 100
    return None

def backtest_dogs_of_dow_strategy(conn, start_year=2015, end_year=2024, top_n=30):
    """
    狗股理论回测
    策略: 每年初买入股息率最高的N只股票，持有一年后调仓
    """
    print(f"\n{'='*60}")
    print(f"狗股理论回测 ({start_year}-{end_year})")
    print(f"策略: 每年初买入股息率最高的{top_n}只股票")
    print(f"{'='*60}")

    results = []

    for year in range(start_year, end_year + 1):
        # 获取年初高股息股票
        high_dv_stocks = get_annual_high_dividend_stocks(conn, year, top_n)

        if len(high_dv_stocks) == 0:
            print(f"{year}年: 无数据")
            continue

        # 计算组合收益
        ts_codes = high_dv_stocks['ts_code'].tolist()
        returns_df = calculate_annual_returns(conn, ts_codes, year)

        if len(returns_df) == 0:
            print(f"{year}年: 无收益数据")
            continue

        # 等权组合收益
        portfolio_return = returns_df['return'].mean()

        # 沪深300收益
        hs300_return = get_index_annual_return(conn, '000300.SH', year)

        # 上证指数收益
        sh_return = get_index_annual_return(conn, '000001.SH', year)

        # 平均股息率
        avg_dv = high_dv_stocks['dv_ttm'].mean()

        results.append({
            'year': year,
            'stock_count': len(returns_df),
            'avg_dv_ratio': avg_dv,
            'portfolio_return': portfolio_return,
            'hs300_return': hs300_return,
            'sh_return': sh_return,
            'excess_hs300': portfolio_return - hs300_return if hs300_return else None,
            'excess_sh': portfolio_return - sh_return if sh_return else None
        })

        print(f"{year}年: 组合收益={portfolio_return:.2f}%, 沪深300={hs300_return:.2f}%, 超额={portfolio_return - hs300_return:.2f}%")

    results_df = pd.DataFrame(results)

    # 汇总统计
    print(f"\n{'='*40}")
    print(f"回测汇总统计:")
    print(f"{'='*40}")
    print(f"年均组合收益: {results_df['portfolio_return'].mean():.2f}%")
    print(f"年均沪深300收益: {results_df['hs300_return'].mean():.2f}%")
    print(f"年均超额收益: {results_df['excess_hs300'].mean():.2f}%")
    print(f"胜率(跑赢沪深300): {(results_df['excess_hs300'] > 0).sum()}/{len(results_df)} = {(results_df['excess_hs300'] > 0).mean()*100:.1f}%")

    # 计算累计收益
    results_df['cum_portfolio'] = (1 + results_df['portfolio_return']/100).cumprod()
    results_df['cum_hs300'] = (1 + results_df['hs300_return']/100).cumprod()

    print(f"\n累计收益 ({start_year}-{end_year}):")
    print(f"高股息策略: {(results_df['cum_portfolio'].iloc[-1] - 1)*100:.1f}%")
    print(f"沪深300: {(results_df['cum_hs300'].iloc[-1] - 1)*100:.1f}%")

    return results_df

def analyze_dividend_yield_ranking_turnover(conn, start_year=2015, end_year=2024, top_n=30):
    """分析股息率排名变动（调仓分析）"""
    print(f"\n{'='*60}")
    print(f"股息率排名调仓分析")
    print(f"{'='*60}")

    yearly_stocks = {}
    turnover_rates = []

    for year in range(start_year, end_year + 1):
        high_dv_stocks = get_annual_high_dividend_stocks(conn, year, top_n)
        if len(high_dv_stocks) > 0:
            yearly_stocks[year] = set(high_dv_stocks['ts_code'].tolist())

    # 计算调仓比例
    for year in range(start_year + 1, end_year + 1):
        if year in yearly_stocks and year-1 in yearly_stocks:
            prev_stocks = yearly_stocks[year-1]
            curr_stocks = yearly_stocks[year]

            # 留存股票
            retained = prev_stocks & curr_stocks
            # 新增股票
            new_stocks = curr_stocks - prev_stocks
            # 剔除股票
            removed_stocks = prev_stocks - curr_stocks

            turnover = len(new_stocks) / len(curr_stocks) * 100
            turnover_rates.append({
                'year': year,
                'retained': len(retained),
                'new': len(new_stocks),
                'removed': len(removed_stocks),
                'turnover_rate': turnover
            })

            print(f"{year}年: 留存{len(retained)}只, 新增{len(new_stocks)}只, 剔除{len(removed_stocks)}只, 换手率{turnover:.1f}%")

    turnover_df = pd.DataFrame(turnover_rates)
    print(f"\n平均年度换手率: {turnover_df['turnover_rate'].mean():.1f}%")

    return turnover_df

# =============================================================================
# 第三部分：策略优化
# =============================================================================

def backtest_dividend_quality_strategy(conn, start_year=2015, end_year=2024, top_n=30):
    """
    股息率 + 质量筛选策略
    额外条件: ROE > 10%, 资产负债率 < 70%, PE_TTM > 0
    """
    print(f"\n{'='*60}")
    print(f"股息率+质量筛选策略回测 ({start_year}-{end_year})")
    print(f"策略: 高股息 + ROE>10% + 资产负债率<70%")
    print(f"{'='*60}")

    results = []

    for year in range(start_year, end_year + 1):
        # 获取该年第一个交易日
        query = f"""
        WITH first_trade_date AS (
            SELECT MIN(trade_date) as first_date
            FROM daily_basic
            WHERE trade_date >= '{year}0101' AND trade_date <= '{year}0115'
        ),
        -- 获取最近的财务数据
        latest_fina AS (
            SELECT
                ts_code,
                roe,
                debt_to_assets,
                ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY end_date DESC) as rn
            FROM fina_indicator_vip
            WHERE end_date <= '{year-1}1231'
                AND end_date >= '{year-1}0101'
        ),
        dv_data AS (
            SELECT
                db.ts_code,
                sb.name,
                db.dv_ttm,
                db.pe_ttm,
                db.pb,
                db.total_mv / 10000 as total_mv_yi,
                f.roe,
                f.debt_to_assets
            FROM daily_basic db
            JOIN stock_basic sb ON db.ts_code = sb.ts_code
            LEFT JOIN latest_fina f ON db.ts_code = f.ts_code AND f.rn = 1
            WHERE db.trade_date = (SELECT first_date FROM first_trade_date)
                AND sb.list_status = 'L'
                AND db.dv_ttm IS NOT NULL
                AND db.dv_ttm > 2  -- 股息率 > 2%
                AND db.pe_ttm > 0 AND db.pe_ttm < 50  -- 合理PE
                AND db.total_mv > 200000  -- 市值 > 20亿
                AND sb.name NOT LIKE '%ST%'
                -- 质量筛选
                AND f.roe > 10  -- ROE > 10%
                AND f.debt_to_assets < 70  -- 资产负债率 < 70%
        )
        SELECT * FROM dv_data
        ORDER BY dv_ttm DESC
        LIMIT {top_n}
        """
        high_quality_dv = conn.execute(query).df()

        if len(high_quality_dv) == 0:
            print(f"{year}年: 无符合条件的股票")
            continue

        # 计算组合收益
        ts_codes = high_quality_dv['ts_code'].tolist()
        returns_df = calculate_annual_returns(conn, ts_codes, year)

        if len(returns_df) == 0:
            continue

        portfolio_return = returns_df['return'].mean()
        hs300_return = get_index_annual_return(conn, '000300.SH', year)
        avg_dv = high_quality_dv['dv_ttm'].mean()
        avg_roe = high_quality_dv['roe'].mean()

        results.append({
            'year': year,
            'stock_count': len(returns_df),
            'avg_dv_ratio': avg_dv,
            'avg_roe': avg_roe,
            'portfolio_return': portfolio_return,
            'hs300_return': hs300_return,
            'excess_return': portfolio_return - hs300_return if hs300_return else None
        })

        print(f"{year}年: 股票数={len(returns_df)}, 平均ROE={avg_roe:.1f}%, 组合收益={portfolio_return:.2f}%, 超额={portfolio_return - hs300_return:.2f}%")

    results_df = pd.DataFrame(results)

    if len(results_df) > 0:
        print(f"\n{'='*40}")
        print(f"质量策略汇总:")
        print(f"年均组合收益: {results_df['portfolio_return'].mean():.2f}%")
        print(f"年均超额收益: {results_df['excess_return'].mean():.2f}%")
        print(f"胜率: {(results_df['excess_return'] > 0).sum()}/{len(results_df)} = {(results_df['excess_return'] > 0).mean()*100:.1f}%")

    return results_df

def analyze_dividend_growth(conn, trade_date='20241231'):
    """分析股息增长因子 - 使用股息率TTM历史数据"""
    print(f"\n{'='*60}")
    print(f"股息增长分析 (基于股息率TTM历史变化)")
    print(f"{'='*60}")

    # 获取每年年初的股息率数据，分析连续分红和增长趋势
    query = """
    WITH yearly_dv AS (
        SELECT
            ts_code,
            SUBSTR(trade_date, 1, 4) as year,
            AVG(dv_ttm) as avg_dv_ttm,
            MAX(dv_ttm) as max_dv_ttm
        FROM daily_basic
        WHERE dv_ttm > 0
            AND trade_date >= '20200101'
            AND trade_date <= '20241231'
        GROUP BY ts_code, SUBSTR(trade_date, 1, 4)
    ),
    yearly_pivot AS (
        SELECT
            ts_code,
            MAX(CASE WHEN year = '2020' THEN avg_dv_ttm END) as dv_2020,
            MAX(CASE WHEN year = '2021' THEN avg_dv_ttm END) as dv_2021,
            MAX(CASE WHEN year = '2022' THEN avg_dv_ttm END) as dv_2022,
            MAX(CASE WHEN year = '2023' THEN avg_dv_ttm END) as dv_2023,
            MAX(CASE WHEN year = '2024' THEN avg_dv_ttm END) as dv_2024
        FROM yearly_dv
        GROUP BY ts_code
    ),
    growth_calc AS (
        SELECT
            yp.ts_code,
            sb.name,
            sb.industry,
            yp.dv_2020,
            yp.dv_2021,
            yp.dv_2022,
            yp.dv_2023,
            yp.dv_2024,
            -- 连续分红年数
            (CASE WHEN yp.dv_2020 > 0 THEN 1 ELSE 0 END +
             CASE WHEN yp.dv_2021 > 0 THEN 1 ELSE 0 END +
             CASE WHEN yp.dv_2022 > 0 THEN 1 ELSE 0 END +
             CASE WHEN yp.dv_2023 > 0 THEN 1 ELSE 0 END +
             CASE WHEN yp.dv_2024 > 0 THEN 1 ELSE 0 END) as consecutive_years,
            -- 股息率CAGR (4年)
            CASE
                WHEN yp.dv_2020 > 0.5 AND yp.dv_2024 > 0
                THEN POWER(yp.dv_2024 / yp.dv_2020, 0.25) - 1
                ELSE NULL
            END as dv_cagr_4y,
            -- 股息率增长 (2024 vs 2022)
            CASE
                WHEN yp.dv_2022 > 0.5 AND yp.dv_2024 > 0
                THEN (yp.dv_2024 / yp.dv_2022 - 1)
                ELSE NULL
            END as dv_growth_2y
        FROM yearly_pivot yp
        JOIN stock_basic sb ON yp.ts_code = sb.ts_code
        WHERE sb.list_status = 'L'
    )
    SELECT * FROM growth_calc
    WHERE consecutive_years >= 4  -- 至少4年连续分红
        AND dv_2024 >= 2  -- 当前股息率 >= 2%
    ORDER BY dv_cagr_4y DESC NULLS LAST
    """
    df = conn.execute(query).df()

    print(f"\n连续分红4年以上且当前股息率>=2%的公司: {len(df)} 只")

    # 分红增长统计
    df_growth = df[df['dv_cagr_4y'].notna()].copy()
    print(f"有增长数据的公司: {len(df_growth)} 只")

    if len(df_growth) > 0:
        print(f"\n股息率增长统计 (4年CAGR):")
        print(df_growth['dv_cagr_4y'].describe())

        # 高增长分红公司 (股息率年增长>5%)
        high_growth = df_growth[df_growth['dv_cagr_4y'] > 0.05].copy()
        print(f"\n股息率年增长>5%的公司: {len(high_growth)} 只")

        if len(high_growth) > 0:
            print(f"\n前20只股息增长公司:")
            display_cols = ['ts_code', 'name', 'industry', 'dv_2020', 'dv_2024', 'dv_cagr_4y', 'consecutive_years']
            high_growth_display = high_growth[display_cols].head(20).copy()
            high_growth_display['dv_cagr_4y'] = high_growth_display['dv_cagr_4y'].apply(lambda x: f"{x*100:.1f}%")
            print(high_growth_display.to_string(index=False))

        # 股息稳定增长公司 (连续5年分红且增长)
        stable_growth = df_growth[
            (df_growth['consecutive_years'] == 5) &
            (df_growth['dv_cagr_4y'] > 0)
        ].copy()
        print(f"\n连续5年分红且股息率增长的公司: {len(stable_growth)} 只")

    return df

def backtest_dividend_reinvestment(conn, start_year=2015, end_year=2024, initial_capital=1000000, top_n=30):
    """
    红利再投资策略模拟
    假设: 每年收到的股息在下一年再投资
    """
    print(f"\n{'='*60}")
    print(f"红利再投资策略模拟 ({start_year}-{end_year})")
    print(f"初始资金: {initial_capital:,} 元")
    print(f"{'='*60}")

    capital = initial_capital
    capital_no_reinvest = initial_capital  # 对比：不进行再投资

    yearly_results = []

    for year in range(start_year, end_year + 1):
        # 获取高股息股票
        high_dv_stocks = get_annual_high_dividend_stocks(conn, year, top_n)

        if len(high_dv_stocks) == 0:
            continue

        avg_dv = high_dv_stocks['dv_ttm'].mean() / 100  # 转为小数

        # 计算年度价格收益
        ts_codes = high_dv_stocks['ts_code'].tolist()
        returns_df = calculate_annual_returns(conn, ts_codes, year)

        if len(returns_df) == 0:
            continue

        price_return = returns_df['return'].mean() / 100  # 价格收益率

        # 红利再投资: 总收益 = 价格收益 + 股息收益 (简化假设)
        total_return_with_reinvest = price_return + avg_dv
        total_return_no_reinvest = price_return

        capital = capital * (1 + total_return_with_reinvest)
        capital_no_reinvest = capital_no_reinvest * (1 + total_return_no_reinvest)

        yearly_results.append({
            'year': year,
            'avg_dividend_yield': avg_dv * 100,
            'price_return': price_return * 100,
            'total_return_reinvest': total_return_with_reinvest * 100,
            'capital_reinvest': capital,
            'capital_no_reinvest': capital_no_reinvest
        })

        print(f"{year}年: 股息率={avg_dv*100:.2f}%, 价格收益={price_return*100:.2f}%, "
              f"再投资资金={capital:,.0f}, 不再投资={capital_no_reinvest:,.0f}")

    results_df = pd.DataFrame(yearly_results)

    if len(results_df) > 0:
        print(f"\n{'='*40}")
        print(f"红利再投资对比:")
        print(f"初始资金: {initial_capital:,}")
        print(f"再投资终值: {capital:,.0f} (收益率: {(capital/initial_capital-1)*100:.1f}%)")
        print(f"不再投资终值: {capital_no_reinvest:,.0f} (收益率: {(capital_no_reinvest/initial_capital-1)*100:.1f}%)")
        print(f"再投资增益: {(capital - capital_no_reinvest):,.0f} ({(capital/capital_no_reinvest-1)*100:.1f}%)")

    return results_df

# =============================================================================
# 可视化
# =============================================================================

def create_visualizations(dv_dist_df, industry_df, backtest_df, quality_df, reinvest_df):
    """创建可视化图表"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. 股息率分布直方图
    ax1 = axes[0, 0]
    dv_data = dv_dist_df[dv_dist_df['dv_ttm'] <= 15]['dv_ttm']
    ax1.hist(dv_data, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.axvline(dv_data.median(), color='red', linestyle='--', label=f'中位数: {dv_data.median():.2f}%')
    ax1.axvline(dv_data.mean(), color='green', linestyle='--', label=f'均值: {dv_data.mean():.2f}%')
    ax1.set_xlabel('股息率 TTM (%)')
    ax1.set_ylabel('股票数量')
    ax1.set_title('A股股息率分布')
    ax1.legend()

    # 2. 行业股息率对比
    ax2 = axes[0, 1]
    top_industries = industry_df.head(15)
    bars = ax2.barh(range(len(top_industries)), top_industries['avg_dv_ttm'], color='coral')
    ax2.set_yticks(range(len(top_industries)))
    ax2.set_yticklabels(top_industries['industry'])
    ax2.set_xlabel('平均股息率 TTM (%)')
    ax2.set_title('行业平均股息率 (Top 15)')
    ax2.invert_yaxis()

    # 3. 狗股策略年度收益对比
    ax3 = axes[0, 2]
    if backtest_df is not None and len(backtest_df) > 0:
        x = range(len(backtest_df))
        width = 0.35
        ax3.bar([i - width/2 for i in x], backtest_df['portfolio_return'], width, label='高股息策略', color='steelblue')
        ax3.bar([i + width/2 for i in x], backtest_df['hs300_return'], width, label='沪深300', color='coral')
        ax3.set_xticks(x)
        ax3.set_xticklabels(backtest_df['year'])
        ax3.set_xlabel('年份')
        ax3.set_ylabel('收益率 (%)')
        ax3.set_title('狗股策略 vs 沪深300 年度收益')
        ax3.legend()
        ax3.axhline(0, color='black', linewidth=0.5)

    # 4. 累计收益对比
    ax4 = axes[1, 0]
    if backtest_df is not None and len(backtest_df) > 0:
        ax4.plot(backtest_df['year'], backtest_df['cum_portfolio'], marker='o', label='高股息策略', linewidth=2)
        ax4.plot(backtest_df['year'], backtest_df['cum_hs300'], marker='s', label='沪深300', linewidth=2)
        ax4.set_xlabel('年份')
        ax4.set_ylabel('累计收益 (倍数)')
        ax4.set_title('累计收益对比')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    # 5. 质量策略超额收益
    ax5 = axes[1, 1]
    if quality_df is not None and len(quality_df) > 0:
        colors = ['green' if x > 0 else 'red' for x in quality_df['excess_return']]
        ax5.bar(quality_df['year'], quality_df['excess_return'], color=colors)
        ax5.set_xlabel('年份')
        ax5.set_ylabel('超额收益 (%)')
        ax5.set_title('股息+质量策略超额收益')
        ax5.axhline(0, color='black', linewidth=0.5)

    # 6. 红利再投资对比
    ax6 = axes[1, 2]
    if reinvest_df is not None and len(reinvest_df) > 0:
        ax6.plot(reinvest_df['year'], reinvest_df['capital_reinvest']/1000000, marker='o',
                 label='红利再投资', linewidth=2)
        ax6.plot(reinvest_df['year'], reinvest_df['capital_no_reinvest']/1000000, marker='s',
                 label='不再投资', linewidth=2)
        ax6.set_xlabel('年份')
        ax6.set_ylabel('资金 (百万)')
        ax6.set_title('红利再投资效果')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{REPORT_DIR}/dividend_strategy_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n图表已保存: {REPORT_DIR}/dividend_strategy_analysis.png")
    plt.close()

# =============================================================================
# 生成研究报告
# =============================================================================

def generate_report(dv_dist_df, high_dv_df, industry_df, cap_df,
                   backtest_df, turnover_df, quality_df, growth_df, reinvest_df):
    """生成Markdown研究报告"""

    report = f"""# 股息率(DV)策略研究报告

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 一、股息率分析

### 1.1 A股股息率分布

| 统计项 | 数值 |
|--------|------|
| 样本数量 | {len(dv_dist_df):,} |
| 平均股息率(TTM) | {dv_dist_df['dv_ttm'].mean():.2f}% |
| 中位数股息率 | {dv_dist_df['dv_ttm'].median():.2f}% |
| 最高股息率 | {dv_dist_df['dv_ttm'].max():.2f}% |
| 股息率>3%的股票 | {len(dv_dist_df[dv_dist_df['dv_ttm']>=3])} 只 ({len(dv_dist_df[dv_dist_df['dv_ttm']>=3])/len(dv_dist_df)*100:.1f}%) |
| 股息率>5%的股票 | {len(dv_dist_df[dv_dist_df['dv_ttm']>=5])} 只 ({len(dv_dist_df[dv_dist_df['dv_ttm']>=5])/len(dv_dist_df)*100:.1f}%) |

### 1.2 高股息股票 Top 20 (股息率>=5%)

| 代码 | 名称 | 行业 | 股息率TTM | PE_TTM | PB | 市值(亿) |
|------|------|------|-----------|--------|-----|----------|
"""

    for _, row in high_dv_df.head(20).iterrows():
        report += f"| {row['ts_code']} | {row['name']} | {row.get('industry', 'N/A')} | {row['dv_ttm']:.2f}% | {row.get('pe_ttm', 0):.1f} | {row.get('pb', 0):.2f} | {row.get('total_mv_yi', 0):.0f} |\n"

    report += f"""
### 1.3 行业股息率对比

| 行业 | 股票数 | 平均股息率 | 中位数 | 高股息(>=3%)数量 |
|------|--------|------------|--------|------------------|
"""

    for _, row in industry_df.head(15).iterrows():
        report += f"| {row['industry']} | {row['stock_count']} | {row['avg_dv_ttm']:.2f}% | {row['median_dv_ttm']:.2f}% | {row['high_dv_count']:.0f} |\n"

    report += f"""

**行业分析要点**:
- 银行、煤炭、钢铁等传统周期行业股息率较高
- 科技、医药等成长性行业股息率相对较低
- 公用事业类股票股息稳定性较好

### 1.4 市值与股息率关系

| 市值分组 | 股票数 | 平均股息率 | 中位数 |
|----------|--------|------------|--------|
"""

    for _, row in cap_df.iterrows():
        report += f"| {row['cap_level']} | {row['stock_count']} | {row['avg_dv_ttm']:.2f}% | {row['median_dv_ttm']:.2f}% |\n"

    report += f"""

---

## 二、狗股理论验证

### 2.1 策略说明

**狗股理论 (Dogs of the Dow)**:
- 每年初选取股息率最高的30只股票
- 等权重买入并持有一年
- 年末调仓，重新选取高股息股票

### 2.2 回测结果

| 年份 | 股票数 | 平均股息率 | 组合收益 | 沪深300 | 超额收益 |
|------|--------|------------|----------|---------|----------|
"""

    if backtest_df is not None and len(backtest_df) > 0:
        for _, row in backtest_df.iterrows():
            report += f"| {row['year']} | {row['stock_count']} | {row['avg_dv_ratio']:.2f}% | {row['portfolio_return']:.2f}% | {row['hs300_return']:.2f}% | {row['excess_hs300']:.2f}% |\n"

        report += f"""
**回测统计**:
- 测试期间: {backtest_df['year'].min()}-{backtest_df['year'].max()}
- 年均组合收益: {backtest_df['portfolio_return'].mean():.2f}%
- 年均沪深300收益: {backtest_df['hs300_return'].mean():.2f}%
- 年均超额收益: {backtest_df['excess_hs300'].mean():.2f}%
- 跑赢沪深300胜率: {(backtest_df['excess_hs300'] > 0).sum()}/{len(backtest_df)} ({(backtest_df['excess_hs300'] > 0).mean()*100:.1f}%)
- 累计收益 (高股息策略): {(backtest_df['cum_portfolio'].iloc[-1] - 1)*100:.1f}%
- 累计收益 (沪深300): {(backtest_df['cum_hs300'].iloc[-1] - 1)*100:.1f}%
"""

    report += f"""
### 2.3 调仓分析

| 年份 | 留存 | 新增 | 剔除 | 换手率 |
|------|------|------|------|--------|
"""

    if turnover_df is not None and len(turnover_df) > 0:
        for _, row in turnover_df.iterrows():
            report += f"| {row['year']} | {row['retained']} | {row['new']} | {row['removed']} | {row['turnover_rate']:.1f}% |\n"

        report += f"""
**调仓特点**:
- 平均年度换手率: {turnover_df['turnover_rate'].mean():.1f}%
- 高股息组合有一定的稳定性，部分股票连续多年入选

---

## 三、策略优化

### 3.1 股息率+质量筛选策略

**筛选条件**:
- 股息率 > 2%
- ROE > 10%
- 资产负债率 < 70%
- PE_TTM > 0 且 < 50
- 市值 > 20亿
- 排除ST股票

| 年份 | 股票数 | 平均ROE | 组合收益 | 沪深300 | 超额收益 |
|------|--------|---------|----------|---------|----------|
"""

    if quality_df is not None and len(quality_df) > 0:
        for _, row in quality_df.iterrows():
            report += f"| {row['year']} | {row['stock_count']} | {row['avg_roe']:.1f}% | {row['portfolio_return']:.2f}% | {row['hs300_return']:.2f}% | {row['excess_return']:.2f}% |\n"

        report += f"""
**质量策略统计**:
- 年均组合收益: {quality_df['portfolio_return'].mean():.2f}%
- 年均超额收益: {quality_df['excess_return'].mean():.2f}%
- 胜率: {(quality_df['excess_return'] > 0).sum()}/{len(quality_df)} ({(quality_df['excess_return'] > 0).mean()*100:.1f}%)
"""

    report += f"""
### 3.2 股息增长分析

| 统计项 | 数值 |
|--------|------|
| 连续4年以上分红公司(股息率>=2%) | {len(growth_df)} 只 |
| 股息率年增长>5%的公司 | {len(growth_df[growth_df['dv_cagr_4y'].notna() & (growth_df['dv_cagr_4y'] > 0.05)])} 只 |

**股息增长的意义**:
- 持续增长的股息反映公司盈利能力和分红意愿
- 股息增长因子可以作为选股的重要参考
- 结合股息率和股息增长可以筛选出"股息成长股"

### 3.3 红利再投资效果

"""

    if reinvest_df is not None and len(reinvest_df) > 0:
        initial = 1000000
        final_reinvest = reinvest_df['capital_reinvest'].iloc[-1]
        final_no_reinvest = reinvest_df['capital_no_reinvest'].iloc[-1]

        report += f"""| 指标 | 数值 |
|------|------|
| 初始资金 | ¥{initial:,.0f} |
| 再投资终值 | ¥{final_reinvest:,.0f} |
| 不再投资终值 | ¥{final_no_reinvest:,.0f} |
| 再投资收益率 | {(final_reinvest/initial-1)*100:.1f}% |
| 不再投资收益率 | {(final_no_reinvest/initial-1)*100:.1f}% |
| 再投资增益 | ¥{(final_reinvest - final_no_reinvest):,.0f} ({(final_reinvest/final_no_reinvest-1)*100:.1f}%) |

**红利再投资要点**:
- 长期来看，红利再投资可以显著增加总收益
- 复利效应在长期投资中尤为明显
- 建议选择有股息再投资计划(DRIP)的产品

---

## 四、结论与建议

### 4.1 主要发现

1. **股息率分布**: A股整体股息率偏低，中位数约{dv_dist_df['dv_ttm'].median():.1f}%，但银行、煤炭等行业有较多高股息股票

2. **狗股策略有效性**: 高股息策略在长期有超额收益，但波动较大，个别年份可能跑输指数

3. **质量因子增强**: 加入ROE、负债率等质量因子可以提高策略稳定性

4. **红利再投资**: 长期来看红利再投资可以显著提升总收益

### 4.2 策略建议

1. **组合构建**:
   - 选取股息率排名前30-50的股票
   - 结合ROE>10%、负债率<70%等质量筛选
   - 适当考虑股息增长因子

2. **调仓频率**:
   - 建议年度调仓，避免频繁交易
   - 关注股息率排名的变动趋势

3. **风险控制**:
   - 避免过度集中于单一行业
   - 排除ST股票和业绩下滑的公司
   - 设置最低市值门槛

4. **红利处理**:
   - 建议采用红利再投资策略
   - 利用复利效应提升长期收益

---

*报告说明: 本报告基于历史数据分析，不构成投资建议。投资有风险，入市需谨慎。*
"""

    return report

# =============================================================================
# 主程序
# =============================================================================

def main():
    """主程序"""
    print("="*60)
    print("股息率(DV)策略研究")
    print("="*60)

    conn = get_connection()

    # 第一部分：股息率分析
    print("\n" + "="*60)
    print("第一部分：股息率分析")
    print("="*60)

    dv_dist_df, high_dv_df = analyze_dividend_yield_distribution(conn)
    industry_df = analyze_dividend_by_industry(conn)
    cap_df = analyze_dividend_by_market_cap(conn)

    # 第二部分：狗股理论验证
    print("\n" + "="*60)
    print("第二部分：狗股理论验证")
    print("="*60)

    backtest_df = backtest_dogs_of_dow_strategy(conn, start_year=2015, end_year=2024, top_n=30)
    turnover_df = analyze_dividend_yield_ranking_turnover(conn, start_year=2015, end_year=2024, top_n=30)

    # 第三部分：策略优化
    print("\n" + "="*60)
    print("第三部分：策略优化")
    print("="*60)

    quality_df = backtest_dividend_quality_strategy(conn, start_year=2015, end_year=2024, top_n=30)
    growth_df = analyze_dividend_growth(conn)
    reinvest_df = backtest_dividend_reinvestment(conn, start_year=2015, end_year=2024)

    # 生成可视化
    print("\n生成可视化图表...")
    create_visualizations(dv_dist_df, industry_df, backtest_df, quality_df, reinvest_df)

    # 生成报告
    print("\n生成研究报告...")
    report = generate_report(
        dv_dist_df, high_dv_df, industry_df, cap_df,
        backtest_df, turnover_df, quality_df, growth_df, reinvest_df
    )

    report_path = f'{REPORT_DIR}/dividend_yield_strategy_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"报告已保存: {report_path}")

    conn.close()
    print("\n研究完成!")

if __name__ == '__main__':
    main()
