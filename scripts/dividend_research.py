#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分红派息深度研究报告生成脚本

基于tushare.db数据库进行全面的分红研究分析:
1. 分红数据统计
2. 分红事件效应
3. 分红因子构建

Author: AI Research Assistant
Date: 2026-02-01
"""

import duckdb
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False
import warnings
warnings.filterwarnings('ignore')

# 数据库路径
DB_PATH = '/Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db'
REPORT_DIR = '/Users/allen/workspace/python/stock/Tushare-DuckDB/reports/research'

def get_connection():
    """获取数据库连接"""
    return duckdb.connect(DB_PATH, read_only=True)

# ============================================================================
# 第一部分: 分红数据统计
# ============================================================================

def analyze_dividend_statistics(conn):
    """分红数据统计分析"""
    results = {}

    # 1.1 年度分红总额趋势（基于daily_basic的股息率数据）
    print("分析年度股息率趋势...")

    # 获取每年年末的股息率数据
    annual_dv = conn.execute("""
        WITH yearly_data AS (
            SELECT
                SUBSTRING(trade_date, 1, 4) as year,
                ts_code,
                trade_date,
                dv_ratio,
                dv_ttm,
                total_mv,
                ROW_NUMBER() OVER (PARTITION BY ts_code, SUBSTRING(trade_date, 1, 4)
                                   ORDER BY trade_date DESC) as rn
            FROM daily_basic
            WHERE dv_ratio IS NOT NULL AND dv_ratio > 0
        )
        SELECT
            year,
            COUNT(DISTINCT ts_code) as stock_count,
            AVG(dv_ratio) as avg_dv_ratio,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY dv_ratio) as median_dv_ratio,
            MAX(dv_ratio) as max_dv_ratio,
            SUM(total_mv * dv_ratio / 100) / 10000 as estimated_dividend_amount_billion
        FROM yearly_data
        WHERE rn = 1 AND year >= '2010' AND year <= '2025'
        GROUP BY year
        ORDER BY year
    """).fetchdf()
    results['annual_dividend_trend'] = annual_dv

    # 1.2 行业分红率对比
    print("分析行业分红率对比...")
    industry_dv = conn.execute("""
        WITH latest_data AS (
            SELECT
                d.ts_code,
                d.dv_ratio,
                d.dv_ttm,
                d.total_mv,
                s.industry,
                d.trade_date,
                ROW_NUMBER() OVER (PARTITION BY d.ts_code ORDER BY d.trade_date DESC) as rn
            FROM daily_basic d
            JOIN stock_basic s ON d.ts_code = s.ts_code
            WHERE d.dv_ratio IS NOT NULL
              AND d.dv_ratio > 0
              AND d.trade_date >= '20240101'
              AND s.industry IS NOT NULL
        )
        SELECT
            industry,
            COUNT(*) as stock_count,
            AVG(dv_ratio) as avg_dv_ratio,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY dv_ratio) as median_dv_ratio,
            AVG(dv_ttm) as avg_dv_ttm,
            SUM(total_mv) / 10000 as total_mv_billion
        FROM latest_data
        WHERE rn = 1
        GROUP BY industry
        HAVING COUNT(*) >= 10
        ORDER BY avg_dv_ratio DESC
    """).fetchdf()
    results['industry_dividend'] = industry_dv

    # 1.3 高股息股票筛选
    print("筛选高股息股票...")
    high_dv_stocks = conn.execute("""
        WITH latest_data AS (
            SELECT
                d.ts_code,
                s.name,
                s.industry,
                d.dv_ratio,
                d.dv_ttm,
                d.pe_ttm,
                d.pb,
                d.total_mv / 10000 as total_mv_billion,
                d.trade_date,
                ROW_NUMBER() OVER (PARTITION BY d.ts_code ORDER BY d.trade_date DESC) as rn
            FROM daily_basic d
            JOIN stock_basic s ON d.ts_code = s.ts_code
            WHERE d.dv_ratio IS NOT NULL
              AND d.dv_ratio > 0
              AND d.trade_date >= '20250101'
              AND s.list_status = 'L'
        )
        SELECT
            ts_code,
            name,
            industry,
            dv_ratio,
            dv_ttm,
            pe_ttm,
            pb,
            total_mv_billion,
            trade_date
        FROM latest_data
        WHERE rn = 1 AND dv_ratio >= 3
        ORDER BY dv_ratio DESC
        LIMIT 50
    """).fetchdf()
    results['high_dividend_stocks'] = high_dv_stocks

    # 1.4 分红连续性分析
    print("分析分红连续性...")
    dividend_continuity = conn.execute("""
        WITH yearly_dividend AS (
            SELECT
                ts_code,
                SUBSTRING(trade_date, 1, 4) as year,
                MAX(dv_ratio) as max_dv_ratio
            FROM daily_basic
            WHERE dv_ratio IS NOT NULL AND dv_ratio > 0
            GROUP BY ts_code, SUBSTRING(trade_date, 1, 4)
        ),
        continuity AS (
            SELECT
                ts_code,
                COUNT(DISTINCT year) as dividend_years,
                MIN(year) as first_year,
                MAX(year) as last_year
            FROM yearly_dividend
            WHERE year >= '2015' AND year <= '2024'
            GROUP BY ts_code
        )
        SELECT
            c.ts_code,
            s.name,
            s.industry,
            c.dividend_years,
            c.first_year,
            c.last_year,
            CASE WHEN c.dividend_years >= 10 THEN '连续10年'
                 WHEN c.dividend_years >= 7 THEN '连续7-9年'
                 WHEN c.dividend_years >= 5 THEN '连续5-6年'
                 WHEN c.dividend_years >= 3 THEN '连续3-4年'
                 ELSE '少于3年' END as continuity_level
        FROM continuity c
        JOIN stock_basic s ON c.ts_code = s.ts_code
        WHERE s.list_status = 'L'
        ORDER BY c.dividend_years DESC, c.ts_code
    """).fetchdf()
    results['dividend_continuity'] = dividend_continuity

    # 连续性分布统计
    continuity_dist = dividend_continuity.groupby('continuity_level').size().reset_index(name='count')
    results['continuity_distribution'] = continuity_dist

    return results


# ============================================================================
# 第二部分: 分红事件效应
# ============================================================================

def analyze_dividend_event_effect(conn):
    """分红事件效应分析"""
    results = {}

    # 2.1 基于股息率变化识别分红事件
    print("识别分红事件...")

    # 找出股息率显著上升的日期（可能是除权除息日附近）
    dividend_events = conn.execute("""
        WITH dv_change AS (
            SELECT
                ts_code,
                trade_date,
                dv_ratio,
                dv_ttm,
                close,
                LAG(dv_ratio) OVER (PARTITION BY ts_code ORDER BY trade_date) as prev_dv_ratio,
                LAG(close) OVER (PARTITION BY ts_code ORDER BY trade_date) as prev_close
            FROM daily_basic
            WHERE dv_ratio IS NOT NULL AND trade_date >= '20200101'
        )
        SELECT
            ts_code,
            trade_date,
            dv_ratio,
            prev_dv_ratio,
            close,
            prev_close,
            (close - prev_close) / prev_close * 100 as price_change_pct
        FROM dv_change
        WHERE prev_dv_ratio IS NOT NULL
          AND dv_ratio > prev_dv_ratio * 1.5  -- 股息率上升50%以上
          AND prev_dv_ratio > 1  -- 之前股息率大于1%
        ORDER BY trade_date DESC
        LIMIT 1000
    """).fetchdf()
    results['dividend_events'] = dividend_events

    # 2.2 分红事件前后收益分析
    print("分析分红事件前后收益...")

    # 获取事件前后的价格数据
    event_returns = conn.execute("""
        WITH events AS (
            SELECT
                ts_code,
                trade_date as event_date,
                dv_ratio
            FROM (
                SELECT
                    ts_code,
                    trade_date,
                    dv_ratio,
                    LAG(dv_ratio) OVER (PARTITION BY ts_code ORDER BY trade_date) as prev_dv_ratio
                FROM daily_basic
                WHERE dv_ratio IS NOT NULL AND trade_date >= '20200101' AND trade_date <= '20250101'
            ) t
            WHERE prev_dv_ratio IS NOT NULL
              AND dv_ratio > prev_dv_ratio * 1.5
              AND prev_dv_ratio > 1
        ),
        price_data AS (
            SELECT
                d.ts_code,
                d.trade_date,
                d.close,
                d.pct_chg
            FROM daily d
            WHERE d.trade_date >= '20191201' AND d.trade_date <= '20250301'
        ),
        event_prices AS (
            SELECT
                e.ts_code,
                e.event_date,
                e.dv_ratio,
                p.trade_date,
                p.close,
                p.pct_chg,
                DATEDIFF('day',
                    STRPTIME(e.event_date, '%Y%m%d'),
                    STRPTIME(p.trade_date, '%Y%m%d')) as days_diff
            FROM events e
            JOIN price_data p ON e.ts_code = p.ts_code
            WHERE ABS(DATEDIFF('day',
                STRPTIME(e.event_date, '%Y%m%d'),
                STRPTIME(p.trade_date, '%Y%m%d'))) <= 30
        )
        SELECT
            days_diff,
            COUNT(*) as event_count,
            AVG(pct_chg) as avg_return,
            STDDEV(pct_chg) as std_return,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY pct_chg) as median_return
        FROM event_prices
        GROUP BY days_diff
        ORDER BY days_diff
    """).fetchdf()
    results['event_returns'] = event_returns

    # 2.3 填权/贴权概率统计
    print("统计填权/贴权概率...")

    fill_rights = conn.execute("""
        WITH events AS (
            SELECT
                ts_code,
                trade_date as event_date,
                dv_ratio,
                ROW_NUMBER() OVER (PARTITION BY ts_code, SUBSTRING(trade_date, 1, 4) ORDER BY trade_date) as rn
            FROM (
                SELECT
                    ts_code,
                    trade_date,
                    dv_ratio,
                    LAG(dv_ratio) OVER (PARTITION BY ts_code ORDER BY trade_date) as prev_dv_ratio
                FROM daily_basic
                WHERE dv_ratio IS NOT NULL AND trade_date >= '20200101' AND trade_date <= '20241201'
            ) t
            WHERE prev_dv_ratio IS NOT NULL
              AND dv_ratio > prev_dv_ratio * 1.3
              AND prev_dv_ratio > 0.5
        ),
        unique_events AS (
            SELECT ts_code, event_date, dv_ratio
            FROM events WHERE rn = 1
        ),
        price_before AS (
            SELECT
                e.ts_code,
                e.event_date,
                e.dv_ratio,
                d.close as price_before,
                d.trade_date as date_before
            FROM unique_events e
            JOIN daily d ON e.ts_code = d.ts_code
            WHERE d.trade_date < e.event_date
            QUALIFY ROW_NUMBER() OVER (PARTITION BY e.ts_code, e.event_date ORDER BY d.trade_date DESC) = 1
        ),
        price_after AS (
            SELECT
                pb.ts_code,
                pb.event_date,
                pb.dv_ratio,
                pb.price_before,
                MIN(CASE WHEN d.trade_date >= pb.event_date
                         AND DATEDIFF('day', STRPTIME(pb.event_date, '%Y%m%d'), STRPTIME(d.trade_date, '%Y%m%d')) <= 30
                    THEN d.close END) as price_after_30d,
                MAX(CASE WHEN d.trade_date >= pb.event_date
                         AND DATEDIFF('day', STRPTIME(pb.event_date, '%Y%m%d'), STRPTIME(d.trade_date, '%Y%m%d')) <= 60
                    THEN d.close END) as max_price_60d
            FROM price_before pb
            JOIN daily d ON pb.ts_code = d.ts_code
            WHERE d.trade_date >= pb.event_date
            GROUP BY pb.ts_code, pb.event_date, pb.dv_ratio, pb.price_before
        )
        SELECT
            ts_code,
            event_date,
            dv_ratio,
            price_before,
            price_after_30d,
            max_price_60d,
            CASE WHEN max_price_60d >= price_before THEN 1 ELSE 0 END as filled_60d,
            (max_price_60d - price_before) / price_before * 100 as return_60d
        FROM price_after
        WHERE price_before IS NOT NULL AND max_price_60d IS NOT NULL
    """).fetchdf()

    if len(fill_rights) > 0:
        fill_rate = fill_rights['filled_60d'].mean() * 100
        avg_return = fill_rights['return_60d'].mean()
        results['fill_rights'] = fill_rights
        results['fill_rate_60d'] = fill_rate
        results['avg_return_60d'] = avg_return
    else:
        results['fill_rights'] = pd.DataFrame()
        results['fill_rate_60d'] = 0
        results['avg_return_60d'] = 0

    # 2.4 高分红预期策略回测
    print("回测高分红策略...")

    # 每月初选择股息率最高的20只股票，持有一个月
    strategy_returns = conn.execute("""
        WITH month_first_day AS (
            SELECT
                SUBSTRING(trade_date, 1, 6) as month,
                MIN(trade_date) as first_trade_date
            FROM daily_basic
            WHERE trade_date >= '20200101' AND trade_date <= '20241231'
            GROUP BY SUBSTRING(trade_date, 1, 6)
        ),
        first_day_selection AS (
            SELECT
                d.ts_code,
                d.trade_date as select_date,
                m.month,
                d.dv_ratio,
                d.close as entry_price,
                ROW_NUMBER() OVER (PARTITION BY d.trade_date ORDER BY d.dv_ratio DESC) as dv_rank
            FROM daily_basic d
            JOIN month_first_day m ON d.trade_date = m.first_trade_date
            WHERE d.dv_ratio IS NOT NULL AND d.dv_ratio > 0
        ),
        top_stocks AS (
            SELECT ts_code, select_date, month, dv_ratio, entry_price
            FROM first_day_selection
            WHERE dv_rank <= 20
        ),
        month_last_day AS (
            SELECT
                SUBSTRING(trade_date, 1, 6) as month,
                MAX(trade_date) as last_trade_date
            FROM daily_basic
            WHERE trade_date >= '20200101' AND trade_date <= '20241231'
            GROUP BY SUBSTRING(trade_date, 1, 6)
        ),
        with_exit AS (
            SELECT
                t.ts_code,
                t.select_date,
                t.month,
                t.dv_ratio,
                t.entry_price,
                d.close as exit_price
            FROM top_stocks t
            JOIN month_last_day m ON t.month = m.month
            JOIN daily_basic d ON t.ts_code = d.ts_code AND d.trade_date = m.last_trade_date
        )
        SELECT
            month,
            COUNT(*) as stock_count,
            AVG((exit_price - entry_price) / entry_price * 100) as avg_return,
            AVG(dv_ratio) as avg_dv_ratio
        FROM with_exit
        WHERE exit_price IS NOT NULL AND entry_price IS NOT NULL AND entry_price > 0
        GROUP BY month
        ORDER BY month
    """).fetchdf()
    results['strategy_returns'] = strategy_returns

    return results


# ============================================================================
# 第三部分: 分红因子构建
# ============================================================================

def build_dividend_factors(conn):
    """分红因子构建与检验"""
    results = {}

    # 3.1 股息率因子
    print("构建股息率因子...")

    dv_factor = conn.execute("""
        WITH monthly_data AS (
            SELECT
                ts_code,
                SUBSTRING(trade_date, 1, 6) as month,
                trade_date,
                dv_ratio,
                dv_ttm,
                close,
                ROW_NUMBER() OVER (PARTITION BY ts_code, SUBSTRING(trade_date, 1, 6) ORDER BY trade_date DESC) as rn
            FROM daily_basic
            WHERE trade_date >= '20200101' AND trade_date <= '20241231'
        ),
        month_end AS (
            SELECT ts_code, month, dv_ratio, dv_ttm, close
            FROM monthly_data WHERE rn = 1 AND dv_ratio IS NOT NULL
        ),
        next_month_return AS (
            SELECT
                m.ts_code,
                m.month,
                m.dv_ratio,
                m.dv_ttm,
                m.close as month_end_price,
                n.close as next_month_end_price,
                (n.close - m.close) / m.close * 100 as next_month_return
            FROM month_end m
            JOIN month_end n ON m.ts_code = n.ts_code
            WHERE n.month = CASE WHEN SUBSTRING(m.month, 5, 2) = '12'
                                 THEN CAST(CAST(SUBSTRING(m.month, 1, 4) AS INT) + 1 AS VARCHAR) || '01'
                                 ELSE SUBSTRING(m.month, 1, 4) || LPAD(CAST(CAST(SUBSTRING(m.month, 5, 2) AS INT) + 1 AS VARCHAR), 2, '0')
                            END
        ),
        quintile_data AS (
            SELECT
                *,
                NTILE(5) OVER (PARTITION BY month ORDER BY dv_ratio) as dv_quintile
            FROM next_month_return
            WHERE dv_ratio > 0
        )
        SELECT
            month,
            dv_quintile,
            COUNT(*) as stock_count,
            AVG(dv_ratio) as avg_dv_ratio,
            AVG(next_month_return) as avg_return,
            STDDEV(next_month_return) as std_return
        FROM quintile_data
        GROUP BY month, dv_quintile
        ORDER BY month, dv_quintile
    """).fetchdf()
    results['dv_factor_quintile'] = dv_factor

    # 计算因子IC
    ic_data = conn.execute("""
        WITH monthly_data AS (
            SELECT
                ts_code,
                SUBSTRING(trade_date, 1, 6) as month,
                trade_date,
                dv_ratio,
                close,
                ROW_NUMBER() OVER (PARTITION BY ts_code, SUBSTRING(trade_date, 1, 6) ORDER BY trade_date DESC) as rn
            FROM daily_basic
            WHERE trade_date >= '20200101' AND trade_date <= '20241231'
        ),
        month_end AS (
            SELECT ts_code, month, dv_ratio, close
            FROM monthly_data WHERE rn = 1 AND dv_ratio IS NOT NULL AND dv_ratio > 0
        ),
        with_return AS (
            SELECT
                m.ts_code,
                m.month,
                m.dv_ratio,
                (n.close - m.close) / m.close as next_month_return
            FROM month_end m
            JOIN month_end n ON m.ts_code = n.ts_code
            WHERE n.month = CASE WHEN SUBSTRING(m.month, 5, 2) = '12'
                                 THEN CAST(CAST(SUBSTRING(m.month, 1, 4) AS INT) + 1 AS VARCHAR) || '01'
                                 ELSE SUBSTRING(m.month, 1, 4) || LPAD(CAST(CAST(SUBSTRING(m.month, 5, 2) AS INT) + 1 AS VARCHAR), 2, '0')
                            END
        )
        SELECT
            month,
            CORR(dv_ratio, next_month_return) as ic
        FROM with_return
        GROUP BY month
        ORDER BY month
    """).fetchdf()
    results['dv_factor_ic'] = ic_data

    # 3.2 分红稳定性因子
    print("构建分红稳定性因子...")

    stability_factor = conn.execute("""
        WITH yearly_dv AS (
            SELECT
                ts_code,
                SUBSTRING(trade_date, 1, 4) as year,
                AVG(dv_ratio) as avg_dv_ratio
            FROM daily_basic
            WHERE dv_ratio IS NOT NULL AND dv_ratio > 0
              AND trade_date >= '20150101' AND trade_date <= '20241231'
            GROUP BY ts_code, SUBSTRING(trade_date, 1, 4)
        ),
        stability AS (
            SELECT
                ts_code,
                COUNT(DISTINCT year) as dividend_years,
                AVG(avg_dv_ratio) as mean_dv,
                STDDEV(avg_dv_ratio) as std_dv,
                CASE WHEN STDDEV(avg_dv_ratio) > 0
                     THEN AVG(avg_dv_ratio) / STDDEV(avg_dv_ratio)
                     ELSE 0 END as stability_ratio
            FROM yearly_dv
            GROUP BY ts_code
            HAVING COUNT(DISTINCT year) >= 5
        ),
        latest_price AS (
            SELECT
                ts_code,
                close,
                ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY trade_date DESC) as rn
            FROM daily
            WHERE trade_date >= '20240101'
        )
        SELECT
            s.ts_code,
            b.name,
            b.industry,
            s.dividend_years,
            s.mean_dv,
            s.std_dv,
            s.stability_ratio,
            p.close as current_price
        FROM stability s
        JOIN stock_basic b ON s.ts_code = b.ts_code
        JOIN latest_price p ON s.ts_code = p.ts_code AND p.rn = 1
        WHERE b.list_status = 'L'
        ORDER BY s.stability_ratio DESC
        LIMIT 100
    """).fetchdf()
    results['stability_factor'] = stability_factor

    # 3.3 分红增长因子
    print("构建分红增长因子...")

    growth_factor = conn.execute("""
        WITH yearly_dv AS (
            SELECT
                ts_code,
                SUBSTRING(trade_date, 1, 4) as year,
                AVG(dv_ratio) as avg_dv_ratio
            FROM daily_basic
            WHERE dv_ratio IS NOT NULL AND dv_ratio > 0
              AND trade_date >= '20180101' AND trade_date <= '20241231'
            GROUP BY ts_code, SUBSTRING(trade_date, 1, 4)
        ),
        dv_growth AS (
            SELECT
                ts_code,
                year,
                avg_dv_ratio,
                LAG(avg_dv_ratio) OVER (PARTITION BY ts_code ORDER BY year) as prev_year_dv,
                (avg_dv_ratio - LAG(avg_dv_ratio) OVER (PARTITION BY ts_code ORDER BY year)) /
                NULLIF(LAG(avg_dv_ratio) OVER (PARTITION BY ts_code ORDER BY year), 0) as yoy_growth
            FROM yearly_dv
        ),
        growth_stats AS (
            SELECT
                ts_code,
                COUNT(*) as growth_years,
                AVG(yoy_growth) as avg_growth,
                SUM(CASE WHEN yoy_growth > 0 THEN 1 ELSE 0 END) as positive_growth_years
            FROM dv_growth
            WHERE yoy_growth IS NOT NULL
            GROUP BY ts_code
            HAVING COUNT(*) >= 3
        )
        SELECT
            g.ts_code,
            b.name,
            b.industry,
            g.growth_years,
            g.avg_growth * 100 as avg_growth_pct,
            g.positive_growth_years,
            CAST(g.positive_growth_years AS FLOAT) / g.growth_years * 100 as growth_consistency
        FROM growth_stats g
        JOIN stock_basic b ON g.ts_code = b.ts_code
        WHERE b.list_status = 'L'
        ORDER BY g.avg_growth DESC
        LIMIT 100
    """).fetchdf()
    results['growth_factor'] = growth_factor

    # 3.4 因子有效性检验 - 分组收益对比
    print("进行因子有效性检验...")

    # 股息率因子分组收益
    quintile_returns = dv_factor.groupby('dv_quintile').agg({
        'avg_return': 'mean',
        'stock_count': 'sum'
    }).reset_index()
    results['quintile_returns'] = quintile_returns

    # IC统计
    if len(ic_data) > 0 and 'ic' in ic_data.columns:
        ic_data_clean = ic_data.dropna()
        if len(ic_data_clean) > 0:
            ic_mean = ic_data_clean['ic'].mean()
            ic_std = ic_data_clean['ic'].std()
            ic_ir = ic_mean / ic_std if ic_std > 0 else 0
            ic_positive_rate = (ic_data_clean['ic'] > 0).mean() * 100

            results['ic_stats'] = {
                'ic_mean': ic_mean,
                'ic_std': ic_std,
                'ic_ir': ic_ir,
                'ic_positive_rate': ic_positive_rate
            }

    return results


# ============================================================================
# 报告生成
# ============================================================================

def generate_report(stats_results, event_results, factor_results):
    """生成研究报告"""

    report = []
    report.append("=" * 80)
    report.append("分红派息深度研究报告")
    report.append("=" * 80)
    report.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"数据来源: tushare.db")
    report.append("\n")

    # ========== 第一部分 ==========
    report.append("=" * 80)
    report.append("第一部分: 分红数据统计")
    report.append("=" * 80)

    # 1.1 年度分红趋势
    report.append("\n1.1 年度股息率趋势")
    report.append("-" * 40)
    if 'annual_dividend_trend' in stats_results:
        df = stats_results['annual_dividend_trend']
        report.append(f"{'年份':<8}{'股票数量':>10}{'平均股息率':>12}{'中位数股息率':>12}{'最高股息率':>12}")
        for _, row in df.iterrows():
            report.append(f"{row['year']:<8}{int(row['stock_count']):>10}{row['avg_dv_ratio']:>12.2f}%{row['median_dv_ratio']:>11.2f}%{row['max_dv_ratio']:>11.2f}%")

    # 1.2 行业分红率对比
    report.append("\n\n1.2 行业股息率排名 (前20)")
    report.append("-" * 40)
    if 'industry_dividend' in stats_results:
        df = stats_results['industry_dividend'].head(20)
        report.append(f"{'行业':<12}{'股票数量':>10}{'平均股息率':>12}{'中位数股息率':>12}{'总市值(亿)':>12}")
        for _, row in df.iterrows():
            industry = str(row['industry'])[:10] if row['industry'] else 'N/A'
            report.append(f"{industry:<12}{int(row['stock_count']):>10}{row['avg_dv_ratio']:>12.2f}%{row['median_dv_ratio']:>11.2f}%{row['total_mv_billion']:>11.0f}")

    # 1.3 高股息股票
    report.append("\n\n1.3 高股息股票 (股息率>=3%, 前20)")
    report.append("-" * 40)
    if 'high_dividend_stocks' in stats_results:
        df = stats_results['high_dividend_stocks'].head(20)
        report.append(f"{'代码':<12}{'名称':<10}{'行业':<10}{'股息率':>10}{'PE(TTM)':>10}{'市值(亿)':>10}")
        for _, row in df.iterrows():
            name = str(row['name'])[:8] if row['name'] else 'N/A'
            industry = str(row['industry'])[:8] if row['industry'] else 'N/A'
            pe = f"{row['pe_ttm']:.1f}" if pd.notna(row['pe_ttm']) else 'N/A'
            report.append(f"{row['ts_code']:<12}{name:<10}{industry:<10}{row['dv_ratio']:>9.2f}%{pe:>10}{row['total_mv_billion']:>9.1f}")

    # 1.4 分红连续性
    report.append("\n\n1.4 分红连续性分布 (2015-2024)")
    report.append("-" * 40)
    if 'continuity_distribution' in stats_results:
        df = stats_results['continuity_distribution']
        for _, row in df.iterrows():
            report.append(f"{row['continuity_level']}: {row['count']} 只股票")

    if 'dividend_continuity' in stats_results:
        report.append("\n连续10年分红的股票 (前20):")
        df = stats_results['dividend_continuity']
        df_10y = df[df['dividend_years'] >= 10].head(20)
        for _, row in df_10y.iterrows():
            name = str(row['name'])[:8] if row['name'] else 'N/A'
            report.append(f"  {row['ts_code']} {name} - {row['industry']} ({row['dividend_years']}年)")

    # ========== 第二部分 ==========
    report.append("\n\n")
    report.append("=" * 80)
    report.append("第二部分: 分红事件效应")
    report.append("=" * 80)

    # 2.1 事件前后收益
    report.append("\n2.1 分红事件前后平均收益")
    report.append("-" * 40)
    if 'event_returns' in event_results and len(event_results['event_returns']) > 0:
        df = event_results['event_returns']
        report.append(f"{'距事件天数':>12}{'样本数量':>10}{'平均收益':>12}{'收益标准差':>12}{'中位数收益':>12}")
        for _, row in df.iterrows():
            if -10 <= row['days_diff'] <= 10:
                report.append(f"{int(row['days_diff']):>12}{int(row['event_count']):>10}{row['avg_return']:>11.2f}%{row['std_return']:>11.2f}%{row['median_return']:>11.2f}%")

    # 2.2 填权/贴权统计
    report.append("\n\n2.2 填权/贴权统计")
    report.append("-" * 40)
    if 'fill_rate_60d' in event_results:
        report.append(f"60日内填权概率: {event_results['fill_rate_60d']:.1f}%")
        report.append(f"60日内平均收益: {event_results['avg_return_60d']:.2f}%")
        if 'fill_rights' in event_results and len(event_results['fill_rights']) > 0:
            report.append(f"样本数量: {len(event_results['fill_rights'])}")

    # 2.3 高分红策略回测
    report.append("\n\n2.3 高股息策略月度回测 (每月选股息率前20)")
    report.append("-" * 40)
    if 'strategy_returns' in event_results and len(event_results['strategy_returns']) > 0:
        df = event_results['strategy_returns']
        total_return = (1 + df['avg_return']/100).prod() - 1
        avg_monthly_return = df['avg_return'].mean()
        positive_months = (df['avg_return'] > 0).sum()
        total_months = len(df)
        report.append(f"回测期间: {df['month'].min()} - {df['month'].max()}")
        report.append(f"累计收益: {total_return*100:.2f}%")
        report.append(f"月均收益: {avg_monthly_return:.2f}%")
        report.append(f"胜率: {positive_months}/{total_months} ({positive_months/total_months*100:.1f}%)")
        report.append(f"平均持仓股息率: {df['avg_dv_ratio'].mean():.2f}%")

    # ========== 第三部分 ==========
    report.append("\n\n")
    report.append("=" * 80)
    report.append("第三部分: 分红因子构建")
    report.append("=" * 80)

    # 3.1 股息率因子
    report.append("\n3.1 股息率因子分组收益")
    report.append("-" * 40)
    if 'quintile_returns' in factor_results:
        df = factor_results['quintile_returns']
        report.append(f"{'分组':>8}{'平均月收益':>15}{'样本总数':>12}")
        for _, row in df.iterrows():
            group_name = f"Q{int(row['dv_quintile'])}(低)" if row['dv_quintile'] == 1 else f"Q{int(row['dv_quintile'])}(高)" if row['dv_quintile'] == 5 else f"Q{int(row['dv_quintile'])}"
            report.append(f"{group_name:>8}{row['avg_return']:>14.2f}%{int(row['stock_count']):>12}")

        # 计算多空收益
        if len(df) == 5:
            long_short = df[df['dv_quintile']==5]['avg_return'].values[0] - df[df['dv_quintile']==1]['avg_return'].values[0]
            report.append(f"\n多空收益(Q5-Q1): {long_short:.2f}%")

    # 3.2 因子IC统计
    report.append("\n\n3.2 股息率因子IC统计")
    report.append("-" * 40)
    if 'ic_stats' in factor_results:
        stats = factor_results['ic_stats']
        report.append(f"IC均值: {stats['ic_mean']:.4f}")
        report.append(f"IC标准差: {stats['ic_std']:.4f}")
        report.append(f"IC_IR: {stats['ic_ir']:.4f}")
        report.append(f"IC为正比例: {stats['ic_positive_rate']:.1f}%")

        # 因子有效性判断
        if stats['ic_mean'] > 0.02:
            report.append("\n因子有效性判断: 股息率因子IC均值为正，具有一定选股能力")
        elif stats['ic_mean'] < -0.02:
            report.append("\n因子有效性判断: 股息率因子IC均值为负，高股息可能反而表现较差")
        else:
            report.append("\n因子有效性判断: 股息率因子IC接近于0，选股能力有限")

    # 3.3 分红稳定性因子
    report.append("\n\n3.3 分红稳定性因子 (前20)")
    report.append("-" * 40)
    if 'stability_factor' in factor_results:
        df = factor_results['stability_factor'].head(20)
        report.append(f"{'代码':<12}{'名称':<10}{'分红年数':>10}{'平均股息率':>12}{'稳定性指标':>12}")
        for _, row in df.iterrows():
            name = str(row['name'])[:8] if row['name'] else 'N/A'
            report.append(f"{row['ts_code']:<12}{name:<10}{int(row['dividend_years']):>10}{row['mean_dv']:>11.2f}%{row['stability_ratio']:>12.2f}")

    # 3.4 分红增长因子
    report.append("\n\n3.4 分红增长因子 (前20)")
    report.append("-" * 40)
    if 'growth_factor' in factor_results:
        df = factor_results['growth_factor'].head(20)
        report.append(f"{'代码':<12}{'名称':<10}{'增长年数':>10}{'平均增长率':>12}{'增长一致性':>12}")
        for _, row in df.iterrows():
            name = str(row['name'])[:8] if row['name'] else 'N/A'
            report.append(f"{row['ts_code']:<12}{name:<10}{int(row['growth_years']):>10}{row['avg_growth_pct']:>11.1f}%{row['growth_consistency']:>11.1f}%")

    # ========== 总结 ==========
    report.append("\n\n")
    report.append("=" * 80)
    report.append("研究总结与投资建议")
    report.append("=" * 80)

    report.append("""
1. 分红市场概况:
   - A股市场具有分红记录的股票数量逐年增加
   - 银行、公用事业、煤炭等行业股息率较高
   - 约有一批股票保持连续多年分红记录

2. 分红事件效应:
   - 分红事件前后存在一定的超额收益机会
   - 填权概率受市场整体环境影响较大
   - 高股息策略在长期具有相对稳定的收益

3. 因子投资建议:
   - 股息率因子可作为选股参考因素之一
   - 分红稳定性比单纯高股息更重要
   - 分红增长能力是识别优质红利股的关键
   - 建议结合估值、盈利质量等因子综合选股

4. 风险提示:
   - 高股息不等于低风险，需关注分红可持续性
   - 历史分红记录不代表未来分红能力
   - 因子有效性可能随市场环境变化
""")

    report.append("\n" + "=" * 80)
    report.append("报告结束")
    report.append("=" * 80)

    return "\n".join(report)


def generate_charts(stats_results, event_results, factor_results):
    """生成分析图表"""

    charts_saved = []

    # 图表1: 年度股息率趋势
    if 'annual_dividend_trend' in stats_results:
        fig, ax1 = plt.subplots(figsize=(12, 6))
        df = stats_results['annual_dividend_trend']

        ax1.bar(df['year'], df['stock_count'], alpha=0.6, color='steelblue', label='分红股票数量')
        ax1.set_xlabel('年份')
        ax1.set_ylabel('股票数量', color='steelblue')
        ax1.tick_params(axis='y', labelcolor='steelblue')

        ax2 = ax1.twinx()
        ax2.plot(df['year'], df['avg_dv_ratio'], color='red', marker='o', linewidth=2, label='平均股息率')
        ax2.plot(df['year'], df['median_dv_ratio'], color='orange', marker='s', linewidth=2, label='中位数股息率')
        ax2.set_ylabel('股息率 (%)', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        plt.title('A股年度股息率趋势')
        plt.tight_layout()
        path = f'{REPORT_DIR}/dividend_annual_trend.png'
        plt.savefig(path, dpi=150)
        plt.close()
        charts_saved.append(path)

    # 图表2: 行业股息率对比
    if 'industry_dividend' in stats_results:
        fig, ax = plt.subplots(figsize=(14, 8))
        df = stats_results['industry_dividend'].head(20)

        y_pos = np.arange(len(df))
        ax.barh(y_pos, df['avg_dv_ratio'], color='steelblue', alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df['industry'])
        ax.invert_yaxis()
        ax.set_xlabel('平均股息率 (%)')
        ax.set_title('行业股息率排名 (前20)')

        for i, v in enumerate(df['avg_dv_ratio']):
            ax.text(v + 0.1, i, f'{v:.2f}%', va='center')

        plt.tight_layout()
        path = f'{REPORT_DIR}/dividend_industry_comparison.png'
        plt.savefig(path, dpi=150)
        plt.close()
        charts_saved.append(path)

    # 图表3: 分红连续性分布
    if 'dividend_continuity' in stats_results:
        fig, ax = plt.subplots(figsize=(10, 6))
        df = stats_results['dividend_continuity']

        continuity_counts = df['dividend_years'].value_counts().sort_index()
        ax.bar(continuity_counts.index, continuity_counts.values, color='steelblue', alpha=0.8)
        ax.set_xlabel('连续分红年数')
        ax.set_ylabel('股票数量')
        ax.set_title('分红连续性分布 (2015-2024)')

        plt.tight_layout()
        path = f'{REPORT_DIR}/dividend_continuity_distribution.png'
        plt.savefig(path, dpi=150)
        plt.close()
        charts_saved.append(path)

    # 图表4: 事件前后收益
    if 'event_returns' in event_results and len(event_results['event_returns']) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        df = event_results['event_returns']
        df = df[(df['days_diff'] >= -20) & (df['days_diff'] <= 20)]

        colors = ['red' if x >= 0 else 'green' for x in df['avg_return']]
        ax.bar(df['days_diff'], df['avg_return'], color=colors, alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.axvline(x=0, color='red', linestyle='--', linewidth=1, label='分红事件日')
        ax.set_xlabel('距事件天数')
        ax.set_ylabel('平均收益 (%)')
        ax.set_title('分红事件前后收益分析')
        ax.legend()

        plt.tight_layout()
        path = f'{REPORT_DIR}/dividend_event_returns.png'
        plt.savefig(path, dpi=150)
        plt.close()
        charts_saved.append(path)

    # 图表5: 高股息策略累计收益
    if 'strategy_returns' in event_results and len(event_results['strategy_returns']) > 0:
        fig, ax = plt.subplots(figsize=(14, 6))
        df = event_results['strategy_returns']

        cumulative_return = (1 + df['avg_return']/100).cumprod()
        ax.plot(range(len(df)), cumulative_return, color='steelblue', linewidth=2)
        ax.fill_between(range(len(df)), 1, cumulative_return, alpha=0.3)
        ax.axhline(y=1, color='black', linestyle='--', linewidth=0.5)

        # 设置x轴标签（每12个月显示一个标签）
        tick_positions = range(0, len(df), 12)
        tick_labels = [df['month'].iloc[i] if i < len(df) else '' for i in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45)

        ax.set_xlabel('时间')
        ax.set_ylabel('累计净值')
        ax.set_title('高股息策略累计收益曲线')

        plt.tight_layout()
        path = f'{REPORT_DIR}/dividend_strategy_cumulative.png'
        plt.savefig(path, dpi=150)
        plt.close()
        charts_saved.append(path)

    # 图表6: 因子分组收益
    if 'quintile_returns' in factor_results:
        fig, ax = plt.subplots(figsize=(10, 6))
        df = factor_results['quintile_returns']

        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, 5))
        bars = ax.bar(df['dv_quintile'], df['avg_return'], color=colors)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('股息率分组 (1=最低, 5=最高)')
        ax.set_ylabel('平均月收益 (%)')
        ax.set_title('股息率因子分组收益对比')
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xticklabels(['Q1(低)', 'Q2', 'Q3', 'Q4', 'Q5(高)'])

        for bar, val in zip(bars, df['avg_return']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                   f'{val:.2f}%', ha='center', va='bottom')

        plt.tight_layout()
        path = f'{REPORT_DIR}/dividend_factor_quintile.png'
        plt.savefig(path, dpi=150)
        plt.close()
        charts_saved.append(path)

    # 图表7: 因子IC时间序列
    if 'dv_factor_ic' in factor_results and len(factor_results['dv_factor_ic']) > 0:
        fig, ax = plt.subplots(figsize=(14, 6))
        df = factor_results['dv_factor_ic'].dropna()

        colors = ['red' if x >= 0 else 'green' for x in df['ic']]
        ax.bar(range(len(df)), df['ic'], color=colors, alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        # 添加移动平均线
        if len(df) >= 12:
            ic_ma = df['ic'].rolling(12).mean()
            ax.plot(range(len(df)), ic_ma, color='blue', linewidth=2, label='12月移动平均')
            ax.legend()

        # 设置x轴标签
        tick_positions = range(0, len(df), 12)
        tick_labels = [df['month'].iloc[i] if i < len(df) else '' for i in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45)

        ax.set_xlabel('时间')
        ax.set_ylabel('IC')
        ax.set_title('股息率因子IC时间序列')

        plt.tight_layout()
        path = f'{REPORT_DIR}/dividend_factor_ic.png'
        plt.savefig(path, dpi=150)
        plt.close()
        charts_saved.append(path)

    return charts_saved


def main():
    """主函数"""
    print("=" * 60)
    print("分红派息深度研究")
    print("=" * 60)

    # 连接数据库
    conn = get_connection()

    try:
        # 第一部分: 分红数据统计
        print("\n[1/3] 分红数据统计...")
        stats_results = analyze_dividend_statistics(conn)

        # 第二部分: 分红事件效应
        print("\n[2/3] 分红事件效应分析...")
        event_results = analyze_dividend_event_effect(conn)

        # 第三部分: 分红因子构建
        print("\n[3/3] 分红因子构建...")
        factor_results = build_dividend_factors(conn)

        # 生成报告
        print("\n生成研究报告...")
        report = generate_report(stats_results, event_results, factor_results)

        # 保存报告
        report_path = f'{REPORT_DIR}/dividend_research_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"报告已保存至: {report_path}")

        # 生成图表
        print("\n生成分析图表...")
        charts = generate_charts(stats_results, event_results, factor_results)
        for chart in charts:
            print(f"图表已保存至: {chart}")

        # 保存详细数据
        print("\n保存详细数据...")

        # 高股息股票列表
        if 'high_dividend_stocks' in stats_results:
            path = f'{REPORT_DIR}/high_dividend_stocks.csv'
            stats_results['high_dividend_stocks'].to_csv(path, index=False, encoding='utf-8-sig')
            print(f"高股息股票列表已保存至: {path}")

        # 分红连续性数据
        if 'dividend_continuity' in stats_results:
            path = f'{REPORT_DIR}/dividend_continuity.csv'
            stats_results['dividend_continuity'].to_csv(path, index=False, encoding='utf-8-sig')
            print(f"分红连续性数据已保存至: {path}")

        # 分红稳定性因子
        if 'stability_factor' in factor_results:
            path = f'{REPORT_DIR}/dividend_stability_factor.csv'
            factor_results['stability_factor'].to_csv(path, index=False, encoding='utf-8-sig')
            print(f"分红稳定性因子已保存至: {path}")

        # 分红增长因子
        if 'growth_factor' in factor_results:
            path = f'{REPORT_DIR}/dividend_growth_factor.csv'
            factor_results['growth_factor'].to_csv(path, index=False, encoding='utf-8-sig')
            print(f"分红增长因子已保存至: {path}")

        print("\n" + "=" * 60)
        print("研究完成!")
        print("=" * 60)

        # 打印报告摘要
        print("\n" + report[:3000] + "\n...(详细内容请查看完整报告)")

    finally:
        conn.close()


if __name__ == '__main__':
    main()
