#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创业板股票特征研究分析脚本

功能：
1. 数据统计：股票分布、行业分布、地区分布
2. 交易特征：波动率、估值、成长性
3. 投资策略：因子分析、PEG选股

使用方法：
    python gem_analysis.py

依赖：
    pip install duckdb pandas matplotlib
"""

import duckdb
import pandas as pd
from pathlib import Path
from datetime import datetime

# 数据库路径
DB_PATH = Path(__file__).parent.parent.parent / "tushare.db"
OUTPUT_DIR = Path(__file__).parent


def get_connection():
    """获取数据库连接（只读模式）"""
    return duckdb.connect(str(DB_PATH), read_only=True)


def basic_statistics(conn):
    """创业板基本统计"""
    print("\n" + "="*60)
    print("一、创业板基本统计")
    print("="*60)

    # 总体统计
    sql = """
    SELECT
        COUNT(*) as total_count,
        SUM(CASE WHEN list_status = 'L' THEN 1 ELSE 0 END) as listed_count,
        SUM(CASE WHEN list_status = 'D' THEN 1 ELSE 0 END) as delisted_count,
        MIN(list_date) as earliest_list_date,
        MAX(list_date) as latest_list_date
    FROM stock_basic
    WHERE ts_code LIKE '300%' OR ts_code LIKE '301%'
    """
    df = conn.execute(sql).df()
    print("\n【总体统计】")
    print(f"历史累计股票数: {df['total_count'].values[0]}")
    print(f"当前上市股票数: {df['listed_count'].values[0]}")
    print(f"退市股票数: {df['delisted_count'].values[0]}")
    print(f"最早上市日期: {df['earliest_list_date'].values[0]}")
    print(f"最新上市日期: {df['latest_list_date'].values[0]}")

    # 上市时间分布
    sql = """
    SELECT
        SUBSTR(list_date, 1, 4) as year,
        COUNT(*) as count
    FROM stock_basic
    WHERE (ts_code LIKE '300%' OR ts_code LIKE '301%') AND list_status = 'L'
    GROUP BY SUBSTR(list_date, 1, 4)
    ORDER BY year
    """
    df = conn.execute(sql).df()
    print("\n【上市时间分布】")
    print(df.to_string(index=False))

    # 行业分布
    sql = """
    SELECT
        industry,
        COUNT(*) as count
    FROM stock_basic
    WHERE (ts_code LIKE '300%' OR ts_code LIKE '301%') AND list_status = 'L'
    GROUP BY industry
    ORDER BY count DESC
    LIMIT 15
    """
    df = conn.execute(sql).df()
    print("\n【行业分布 TOP15】")
    print(df.to_string(index=False))

    return df


def trading_features(conn):
    """交易特征分析"""
    print("\n" + "="*60)
    print("二、交易特征分析")
    print("="*60)

    # 波动率对比
    sql = """
    WITH gem_data AS (
        SELECT
            d.trade_date,
            AVG(ABS(d.pct_chg)) as avg_abs_chg,
            STDDEV(d.pct_chg) as std_chg,
            AVG(d.pct_chg) as avg_chg
        FROM daily d
        JOIN stock_basic s ON d.ts_code = s.ts_code
        WHERE (s.ts_code LIKE '300%' OR s.ts_code LIKE '301%')
            AND s.list_status = 'L'
            AND d.trade_date >= '20240101'
        GROUP BY d.trade_date
    ),
    main_data AS (
        SELECT
            d.trade_date,
            AVG(ABS(d.pct_chg)) as avg_abs_chg,
            STDDEV(d.pct_chg) as std_chg,
            AVG(d.pct_chg) as avg_chg
        FROM daily d
        JOIN stock_basic s ON d.ts_code = s.ts_code
        WHERE (s.ts_code LIKE '60%' OR s.ts_code LIKE '00%')
            AND NOT (s.ts_code LIKE '300%' OR s.ts_code LIKE '301%')
            AND s.list_status = 'L'
            AND d.trade_date >= '20240101'
        GROUP BY d.trade_date
    )
    SELECT
        '创业板' as market,
        ROUND(AVG(avg_abs_chg), 4) as avg_daily_abs_change,
        ROUND(AVG(std_chg), 4) as avg_daily_std,
        ROUND(AVG(avg_chg), 4) as avg_daily_return
    FROM gem_data
    UNION ALL
    SELECT
        '主板' as market,
        ROUND(AVG(avg_abs_chg), 4) as avg_daily_abs_change,
        ROUND(AVG(std_chg), 4) as avg_daily_std,
        ROUND(AVG(avg_chg), 4) as avg_daily_return
    FROM main_data
    """
    df = conn.execute(sql).df()
    print("\n【波动率对比(2024年)】")
    print(df.to_string(index=False))

    # 估值对比
    sql = """
    WITH latest_date AS (
        SELECT MAX(trade_date) as max_date
        FROM daily_basic
        WHERE trade_date >= '20250101'
    ),
    gem_valuation AS (
        SELECT
            db.ts_code,
            db.pe_ttm,
            db.pb,
            db.turnover_rate
        FROM daily_basic db
        JOIN stock_basic s ON db.ts_code = s.ts_code
        CROSS JOIN latest_date l
        WHERE (s.ts_code LIKE '300%' OR s.ts_code LIKE '301%')
            AND s.list_status = 'L'
            AND db.trade_date = l.max_date
            AND db.pe_ttm > 0 AND db.pe_ttm < 1000
            AND db.pb > 0 AND db.pb < 100
    ),
    main_valuation AS (
        SELECT
            db.ts_code,
            db.pe_ttm,
            db.pb,
            db.turnover_rate
        FROM daily_basic db
        JOIN stock_basic s ON db.ts_code = s.ts_code
        CROSS JOIN latest_date l
        WHERE (s.ts_code LIKE '60%' OR s.ts_code LIKE '00%')
            AND NOT (s.ts_code LIKE '300%' OR s.ts_code LIKE '301%')
            AND s.list_status = 'L'
            AND db.trade_date = l.max_date
            AND db.pe_ttm > 0 AND db.pe_ttm < 1000
            AND db.pb > 0 AND db.pb < 100
    )
    SELECT
        '创业板' as market,
        ROUND(AVG(pe_ttm), 2) as avg_pe_ttm,
        ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY pe_ttm), 2) as median_pe_ttm,
        ROUND(AVG(pb), 2) as avg_pb,
        ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY pb), 2) as median_pb,
        ROUND(AVG(turnover_rate), 2) as avg_turnover
    FROM gem_valuation
    UNION ALL
    SELECT
        '主板' as market,
        ROUND(AVG(pe_ttm), 2) as avg_pe_ttm,
        ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY pe_ttm), 2) as median_pe_ttm,
        ROUND(AVG(pb), 2) as avg_pb,
        ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY pb), 2) as median_pb,
        ROUND(AVG(turnover_rate), 2) as avg_turnover
    FROM main_valuation
    """
    df = conn.execute(sql).df()
    print("\n【估值对比(最新)】")
    print(df.to_string(index=False))

    return df


def investment_strategy(conn):
    """投资策略分析"""
    print("\n" + "="*60)
    print("三、投资策略分析")
    print("="*60)

    # 市值因子
    sql = """
    WITH latest_date AS (
        SELECT MAX(trade_date) as max_date
        FROM daily_basic
        WHERE trade_date >= '20250101'
    ),
    mv_groups AS (
        SELECT
            db.ts_code,
            CASE
                WHEN db.total_mv / 10000 < 50 THEN '1.小盘(<50亿)'
                WHEN db.total_mv / 10000 < 100 THEN '2.中小盘(50-100亿)'
                WHEN db.total_mv / 10000 < 300 THEN '3.中盘(100-300亿)'
                ELSE '4.大盘(>300亿)'
            END as mv_group
        FROM daily_basic db
        JOIN stock_basic s ON db.ts_code = s.ts_code
        CROSS JOIN latest_date l
        WHERE (s.ts_code LIKE '300%' OR s.ts_code LIKE '301%')
            AND s.list_status = 'L'
            AND db.trade_date = l.max_date
    ),
    returns AS (
        SELECT
            d.ts_code,
            SUM(d.pct_chg) as total_return
        FROM daily d
        WHERE d.trade_date >= '20240101'
        GROUP BY d.ts_code
    )
    SELECT
        m.mv_group,
        COUNT(*) as stock_count,
        ROUND(AVG(r.total_return), 2) as avg_return,
        ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY r.total_return), 2) as median_return
    FROM mv_groups m
    JOIN returns r ON m.ts_code = r.ts_code
    GROUP BY m.mv_group
    ORDER BY m.mv_group
    """
    df = conn.execute(sql).df()
    print("\n【市值因子分析】")
    print(df.to_string(index=False))

    # PEG选股
    sql = """
    WITH growth_data AS (
        SELECT
            f.ts_code,
            s.name,
            s.industry,
            f.netprofit_yoy as profit_growth,
            f.tr_yoy as revenue_growth,
            f.roe
        FROM fina_indicator_vip f
        JOIN stock_basic s ON f.ts_code = s.ts_code
        WHERE (s.ts_code LIKE '300%' OR s.ts_code LIKE '301%')
            AND s.list_status = 'L'
            AND f.end_date = '20231231'
    ),
    valuation_data AS (
        SELECT
            db.ts_code,
            db.pe_ttm,
            db.total_mv / 10000 as total_mv_billion
        FROM daily_basic db
        CROSS JOIN (SELECT MAX(trade_date) as max_date FROM daily_basic WHERE trade_date >= '20250101') l
        WHERE db.trade_date = l.max_date
            AND db.pe_ttm > 0 AND db.pe_ttm < 200
    )
    SELECT
        g.ts_code,
        g.name,
        g.industry,
        ROUND(v.pe_ttm, 2) as pe_ttm,
        ROUND(g.profit_growth, 2) as profit_growth,
        ROUND(v.pe_ttm / NULLIF(g.profit_growth, 0), 2) as peg,
        ROUND(g.revenue_growth, 2) as revenue_growth,
        ROUND(g.roe, 2) as roe,
        ROUND(v.total_mv_billion, 2) as mv_billion
    FROM growth_data g
    JOIN valuation_data v ON g.ts_code = v.ts_code
    WHERE g.profit_growth > 20
        AND g.revenue_growth > 10
        AND g.roe > 10
        AND v.pe_ttm / NULLIF(g.profit_growth, 0) < 1.5
        AND v.pe_ttm / NULLIF(g.profit_growth, 0) > 0
    ORDER BY peg
    LIMIT 20
    """
    df = conn.execute(sql).df()
    print("\n【PEG选股结果(低估高成长)】")
    print(df.to_string(index=False))

    # 保存选股结果
    output_file = OUTPUT_DIR / f"peg_stocks_{datetime.now().strftime('%Y%m%d')}.csv"
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n选股结果已保存到: {output_file}")

    return df


def main():
    """主函数"""
    print("="*60)
    print("创业板股票特征研究")
    print(f"报告日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    try:
        conn = get_connection()

        # 1. 基本统计
        basic_statistics(conn)

        # 2. 交易特征
        trading_features(conn)

        # 3. 投资策略
        investment_strategy(conn)

        print("\n" + "="*60)
        print("分析完成！")
        print("="*60)

    except Exception as e:
        print(f"错误: {e}")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
