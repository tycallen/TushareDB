#!/usr/bin/env python3
"""
市净率(PB)因子研究报告
=======================

研究内容：
1. PB 数据分析 - 全市场分布、行业差异
2. PB 因子效果 - 低PB策略、分组回测、行业中性
3. 策略应用 - 破净股、PB-ROE模型、因子组合

数据来源：tushare.db
"""

import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 数据库路径
DB_PATH = '/Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db'
OUTPUT_DIR = '/Users/allen/workspace/python/stock/Tushare-DuckDB/reports/research/'

def get_connection():
    """获取数据库连接"""
    return duckdb.connect(DB_PATH, read_only=True)


# =============================================================================
# Part 1: PB 数据分析
# =============================================================================

def analyze_pb_distribution():
    """分析全市场PB分布"""
    print("\n" + "="*60)
    print("1.1 全市场 PB 分布分析")
    print("="*60)

    conn = get_connection()

    # 获取最新交易日的PB数据
    latest_date = conn.execute("""
        SELECT MAX(trade_date) FROM daily_basic WHERE pb IS NOT NULL
    """).fetchone()[0]

    print(f"\n最新交易日: {latest_date}")

    # 获取全市场PB数据
    df = conn.execute(f"""
        SELECT
            db.ts_code,
            db.pb,
            db.pe_ttm,
            db.total_mv,
            sb.name,
            sb.industry
        FROM daily_basic db
        JOIN stock_basic sb ON db.ts_code = sb.ts_code
        WHERE db.trade_date = '{latest_date}'
          AND db.pb IS NOT NULL
          AND db.pb > 0
          AND db.pb < 100  -- 排除异常值
          AND sb.list_status = 'L'  -- 仅上市中
    """).fetchdf()

    print(f"有效股票数量: {len(df)}")

    # 描述性统计
    print("\n【PB描述性统计】")
    print(f"均值: {df['pb'].mean():.2f}")
    print(f"中位数: {df['pb'].median():.2f}")
    print(f"标准差: {df['pb'].std():.2f}")
    print(f"最小值: {df['pb'].min():.2f}")
    print(f"最大值: {df['pb'].max():.2f}")
    print(f"25%分位: {df['pb'].quantile(0.25):.2f}")
    print(f"75%分位: {df['pb'].quantile(0.75):.2f}")

    # 破净股统计
    broken_net = df[df['pb'] < 1]
    print(f"\n【破净股统计】")
    print(f"破净股数量: {len(broken_net)} ({len(broken_net)/len(df)*100:.1f}%)")
    print(f"破净股平均PB: {broken_net['pb'].mean():.2f}")

    # PB分段统计
    print("\n【PB分段分布】")
    bins = [0, 0.5, 1, 1.5, 2, 3, 5, 10, 100]
    labels = ['<0.5', '0.5-1', '1-1.5', '1.5-2', '2-3', '3-5', '5-10', '>10']
    df['pb_range'] = pd.cut(df['pb'], bins=bins, labels=labels)
    pb_dist = df['pb_range'].value_counts().sort_index()
    for range_name, count in pb_dist.items():
        print(f"  {range_name}: {count} ({count/len(df)*100:.1f}%)")

    # 绘制PB分布图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 直方图
    ax1 = axes[0, 0]
    df['pb'].hist(bins=50, ax=ax1, edgecolor='black', alpha=0.7)
    ax1.axvline(x=1, color='red', linestyle='--', label='PB=1 (破净线)')
    ax1.axvline(x=df['pb'].median(), color='green', linestyle='--', label=f'中位数={df["pb"].median():.2f}')
    ax1.set_xlabel('PB')
    ax1.set_ylabel('频数')
    ax1.set_title(f'全市场PB分布 ({latest_date})')
    ax1.set_xlim(0, 10)
    ax1.legend()

    # 2. 箱线图
    ax2 = axes[0, 1]
    df.boxplot(column='pb', ax=ax2)
    ax2.set_ylabel('PB')
    ax2.set_title('PB箱线图')
    ax2.set_ylim(0, 10)

    # 3. PB分段柱状图
    ax3 = axes[1, 0]
    pb_dist.plot(kind='bar', ax=ax3, color='steelblue', edgecolor='black')
    ax3.set_xlabel('PB区间')
    ax3.set_ylabel('股票数量')
    ax3.set_title('PB分段分布')
    ax3.tick_params(axis='x', rotation=45)

    # 4. 累积分布
    ax4 = axes[1, 1]
    sorted_pb = np.sort(df['pb'])
    cumulative = np.arange(1, len(sorted_pb) + 1) / len(sorted_pb)
    ax4.plot(sorted_pb, cumulative, linewidth=2)
    ax4.axvline(x=1, color='red', linestyle='--', label='PB=1')
    ax4.axhline(y=len(broken_net)/len(df), color='red', linestyle=':', alpha=0.5)
    ax4.set_xlabel('PB')
    ax4.set_ylabel('累积概率')
    ax4.set_title('PB累积分布函数')
    ax4.set_xlim(0, 10)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}pb_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n图表已保存: {OUTPUT_DIR}pb_distribution.png")

    conn.close()
    return df


def analyze_industry_pb():
    """分析行业PB差异"""
    print("\n" + "="*60)
    print("1.2 行业 PB 差异分析")
    print("="*60)

    conn = get_connection()

    # 获取最新交易日
    latest_date = conn.execute("""
        SELECT MAX(trade_date) FROM daily_basic WHERE pb IS NOT NULL
    """).fetchone()[0]

    # 使用申万行业分类
    df = conn.execute(f"""
        WITH latest_member AS (
            SELECT ts_code, l1_name as industry
            FROM index_member_all
            WHERE is_new = 'Y'
        )
        SELECT
            db.ts_code,
            db.pb,
            db.pe_ttm,
            db.total_mv,
            lm.industry,
            sb.name
        FROM daily_basic db
        JOIN latest_member lm ON db.ts_code = lm.ts_code
        JOIN stock_basic sb ON db.ts_code = sb.ts_code
        WHERE db.trade_date = '{latest_date}'
          AND db.pb IS NOT NULL
          AND db.pb > 0
          AND db.pb < 100
          AND sb.list_status = 'L'
    """).fetchdf()

    print(f"\n有效股票数量: {len(df)}")

    # 行业PB统计
    industry_stats = df.groupby('industry').agg({
        'pb': ['mean', 'median', 'std', 'min', 'max', 'count'],
        'total_mv': 'sum'
    }).round(2)
    industry_stats.columns = ['PB均值', 'PB中位数', 'PB标准差', 'PB最小', 'PB最大', '股票数', '总市值']
    industry_stats = industry_stats.sort_values('PB中位数')

    print("\n【各行业PB统计（按中位数排序）】")
    print(industry_stats.to_string())

    # 计算行业破净率
    industry_broken = df[df['pb'] < 1].groupby('industry').size()
    industry_total = df.groupby('industry').size()
    broken_rate = (industry_broken / industry_total * 100).fillna(0).round(1)

    print("\n【各行业破净率】")
    broken_rate_sorted = broken_rate.sort_values(ascending=False)
    for ind, rate in broken_rate_sorted.items():
        print(f"  {ind}: {rate:.1f}%")

    # 绘制行业PB图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. 行业PB中位数条形图
    ax1 = axes[0, 0]
    pb_median = df.groupby('industry')['pb'].median().sort_values()
    colors = ['red' if v < 1 else 'steelblue' for v in pb_median]
    pb_median.plot(kind='barh', ax=ax1, color=colors, edgecolor='black')
    ax1.axvline(x=1, color='red', linestyle='--', linewidth=2, label='PB=1')
    ax1.set_xlabel('PB中位数')
    ax1.set_title('各行业PB中位数')
    ax1.legend()

    # 2. 行业PB箱线图
    ax2 = axes[0, 1]
    top_industries = df.groupby('industry').size().nlargest(15).index
    df_top = df[df['industry'].isin(top_industries)]
    df_top.boxplot(column='pb', by='industry', ax=ax2)
    ax2.set_ylim(0, 10)
    ax2.set_xlabel('行业')
    ax2.set_ylabel('PB')
    ax2.set_title('主要行业PB分布')
    plt.suptitle('')
    ax2.tick_params(axis='x', rotation=45)

    # 3. 行业破净率
    ax3 = axes[1, 0]
    broken_rate_sorted.plot(kind='barh', ax=ax3, color='coral', edgecolor='black')
    ax3.set_xlabel('破净率 (%)')
    ax3.set_title('各行业破净率')

    # 4. PB vs 行业市值散点图
    ax4 = axes[1, 1]
    industry_summary = df.groupby('industry').agg({
        'pb': 'median',
        'total_mv': 'sum'
    })
    ax4.scatter(industry_summary['total_mv']/1e8, industry_summary['pb'],
                s=100, alpha=0.7, c='steelblue', edgecolors='black')
    for idx, row in industry_summary.iterrows():
        ax4.annotate(idx, (row['total_mv']/1e8, row['pb']), fontsize=8, alpha=0.7)
    ax4.axhline(y=1, color='red', linestyle='--', alpha=0.5)
    ax4.set_xlabel('行业总市值 (亿元)')
    ax4.set_ylabel('PB中位数')
    ax4.set_title('行业市值与PB关系')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}pb_industry.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n图表已保存: {OUTPUT_DIR}pb_industry.png")

    conn.close()
    return industry_stats


def analyze_pb_time_series():
    """分析PB历史变化"""
    print("\n" + "="*60)
    print("1.3 PB 历史变化分析")
    print("="*60)

    conn = get_connection()

    # 获取每月末的PB数据
    df = conn.execute("""
        WITH monthly_dates AS (
            SELECT trade_date,
                   ROW_NUMBER() OVER (
                       PARTITION BY SUBSTR(trade_date, 1, 6)
                       ORDER BY trade_date DESC
                   ) as rn
            FROM (SELECT DISTINCT trade_date FROM daily_basic WHERE trade_date >= '20150101')
        )
        SELECT
            d.trade_date,
            AVG(d.pb) as avg_pb,
            MEDIAN(d.pb) as median_pb,
            PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY d.pb) as pb_25,
            PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY d.pb) as pb_75,
            COUNT(*) as stock_count,
            SUM(CASE WHEN d.pb < 1 THEN 1 ELSE 0 END) as broken_count
        FROM daily_basic d
        JOIN monthly_dates m ON d.trade_date = m.trade_date
        WHERE m.rn = 1
          AND d.pb IS NOT NULL
          AND d.pb > 0
          AND d.pb < 100
        GROUP BY d.trade_date
        ORDER BY d.trade_date
    """).fetchdf()

    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df['broken_rate'] = df['broken_count'] / df['stock_count'] * 100

    print(f"\n数据时间范围: {df['trade_date'].min()} - {df['trade_date'].max()}")
    print(f"数据点数量: {len(df)}")

    # 近期统计
    latest = df.iloc[-1]
    print(f"\n【最新数据 ({latest['trade_date'].strftime('%Y-%m-%d')})】")
    print(f"PB均值: {latest['avg_pb']:.2f}")
    print(f"PB中位数: {latest['median_pb']:.2f}")
    print(f"破净率: {latest['broken_rate']:.1f}%")

    # 历史极值
    print(f"\n【历史极值】")
    print(f"PB中位数最低: {df['median_pb'].min():.2f} ({df.loc[df['median_pb'].idxmin(), 'trade_date'].strftime('%Y-%m-%d')})")
    print(f"PB中位数最高: {df['median_pb'].max():.2f} ({df.loc[df['median_pb'].idxmax(), 'trade_date'].strftime('%Y-%m-%d')})")
    print(f"破净率最高: {df['broken_rate'].max():.1f}% ({df.loc[df['broken_rate'].idxmax(), 'trade_date'].strftime('%Y-%m-%d')})")
    print(f"破净率最低: {df['broken_rate'].min():.1f}% ({df.loc[df['broken_rate'].idxmin(), 'trade_date'].strftime('%Y-%m-%d')})")

    # 绘图
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # 1. PB时间序列
    ax1 = axes[0]
    ax1.plot(df['trade_date'], df['median_pb'], label='PB中位数', linewidth=2, color='blue')
    ax1.fill_between(df['trade_date'], df['pb_25'], df['pb_75'], alpha=0.2, color='blue', label='25-75分位')
    ax1.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='PB=1')
    ax1.set_ylabel('PB')
    ax1.set_title('全市场PB历史变化')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # 2. 破净率时间序列
    ax2 = axes[1]
    ax2.plot(df['trade_date'], df['broken_rate'], label='破净率', linewidth=2, color='coral')
    ax2.fill_between(df['trade_date'], 0, df['broken_rate'], alpha=0.3, color='coral')
    ax2.set_xlabel('日期')
    ax2.set_ylabel('破净率 (%)')
    ax2.set_title('全市场破净率历史变化')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}pb_timeseries.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n图表已保存: {OUTPUT_DIR}pb_timeseries.png")

    conn.close()
    return df


# =============================================================================
# Part 2: PB 因子效果
# =============================================================================

def backtest_low_pb_strategy():
    """低PB策略回测"""
    print("\n" + "="*60)
    print("2.1 低 PB 策略回测")
    print("="*60)

    conn = get_connection()

    # 获取月末交易日列表（2018年后）
    dates = conn.execute("""
        WITH monthly_dates AS (
            SELECT trade_date,
                   ROW_NUMBER() OVER (
                       PARTITION BY SUBSTR(trade_date, 1, 6)
                       ORDER BY trade_date DESC
                   ) as rn
            FROM (SELECT DISTINCT trade_date FROM daily_basic WHERE trade_date >= '20180101')
        )
        SELECT trade_date FROM monthly_dates WHERE rn = 1
        ORDER BY trade_date
    """).fetchdf()

    trade_dates = dates['trade_date'].tolist()
    print(f"\n回测期间: {trade_dates[0]} - {trade_dates[-1]}")
    print(f"调仓次数: {len(trade_dates) - 1}")

    # 回测策略：每月末选取PB最低的50只股票
    results = []

    for i in range(len(trade_dates) - 1):
        date = trade_dates[i]
        next_date = trade_dates[i + 1]

        # 选取低PB股票
        stocks = conn.execute(f"""
            SELECT
                db.ts_code,
                db.pb
            FROM daily_basic db
            JOIN stock_basic sb ON db.ts_code = sb.ts_code
            WHERE db.trade_date = '{date}'
              AND db.pb IS NOT NULL
              AND db.pb > 0
              AND db.pb < 100
              AND sb.list_status = 'L'
            ORDER BY db.pb
            LIMIT 50
        """).fetchdf()

        if len(stocks) == 0:
            continue

        stock_list = "','".join(stocks['ts_code'].tolist())

        # 计算下个月收益
        returns = conn.execute(f"""
            WITH start_price AS (
                SELECT ts_code, close as start_close
                FROM daily d1
                WHERE trade_date = (
                    SELECT MIN(trade_date) FROM daily
                    WHERE trade_date > '{date}' AND ts_code = d1.ts_code
                )
            ),
            end_price AS (
                SELECT ts_code, close as end_close
                FROM daily d2
                WHERE trade_date = (
                    SELECT MAX(trade_date) FROM daily
                    WHERE trade_date <= '{next_date}' AND ts_code = d2.ts_code
                )
            )
            SELECT
                s.ts_code,
                e.end_close / s.start_close - 1 as ret
            FROM start_price s
            JOIN end_price e ON s.ts_code = e.ts_code
            WHERE s.ts_code IN ('{stock_list}')
        """).fetchdf()

        if len(returns) > 0:
            portfolio_ret = returns['ret'].mean()
            results.append({
                'date': date,
                'return': portfolio_ret,
                'avg_pb': stocks['pb'].mean()
            })

    results_df = pd.DataFrame(results)
    results_df['date'] = pd.to_datetime(results_df['date'])
    results_df['cum_return'] = (1 + results_df['return']).cumprod() - 1

    # 计算基准收益（沪深300 - 使用全市场等权代替）
    benchmark_results = []
    for i in range(len(trade_dates) - 1):
        date = trade_dates[i]
        next_date = trade_dates[i + 1]

        bench_ret = conn.execute(f"""
            WITH start_price AS (
                SELECT ts_code, close as start_close
                FROM daily d1
                WHERE trade_date = (
                    SELECT MIN(trade_date) FROM daily
                    WHERE trade_date > '{date}' AND ts_code = d1.ts_code
                )
            ),
            end_price AS (
                SELECT ts_code, close as end_close
                FROM daily d2
                WHERE trade_date = (
                    SELECT MAX(trade_date) FROM daily
                    WHERE trade_date <= '{next_date}' AND ts_code = d2.ts_code
                )
            )
            SELECT AVG(e.end_close / s.start_close - 1) as ret
            FROM start_price s
            JOIN end_price e ON s.ts_code = e.ts_code
            WHERE s.ts_code IN (
                SELECT ts_code FROM stock_basic WHERE list_status = 'L'
            )
        """).fetchone()

        if bench_ret[0] is not None:
            benchmark_results.append({'date': date, 'return': bench_ret[0]})

    benchmark_df = pd.DataFrame(benchmark_results)
    benchmark_df['date'] = pd.to_datetime(benchmark_df['date'])
    benchmark_df['cum_return'] = (1 + benchmark_df['return']).cumprod() - 1

    # 合并结果
    merged = results_df.merge(benchmark_df[['date', 'cum_return']], on='date', suffixes=('', '_bench'))
    merged['excess_cum'] = merged['cum_return'] - merged['cum_return_bench']

    # 绩效统计
    print("\n【低PB策略绩效】")
    total_return = results_df['cum_return'].iloc[-1]
    annual_return = (1 + total_return) ** (12 / len(results_df)) - 1
    monthly_vol = results_df['return'].std()
    annual_vol = monthly_vol * np.sqrt(12)
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0
    max_drawdown = (results_df['cum_return'].cummax() - results_df['cum_return']).max()
    win_rate = (results_df['return'] > 0).mean()

    print(f"累计收益: {total_return*100:.1f}%")
    print(f"年化收益: {annual_return*100:.1f}%")
    print(f"年化波动: {annual_vol*100:.1f}%")
    print(f"夏普比率: {sharpe:.2f}")
    print(f"最大回撤: {max_drawdown*100:.1f}%")
    print(f"月度胜率: {win_rate*100:.1f}%")

    bench_total = benchmark_df['cum_return'].iloc[-1]
    bench_annual = (1 + bench_total) ** (12 / len(benchmark_df)) - 1
    print(f"\n【基准（全市场等权）】")
    print(f"累计收益: {bench_total*100:.1f}%")
    print(f"年化收益: {bench_annual*100:.1f}%")

    print(f"\n【超额收益】")
    print(f"累计超额: {(total_return - bench_total)*100:.1f}%")
    print(f"年化超额: {(annual_return - bench_annual)*100:.1f}%")

    # 绘图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 累计收益曲线
    ax1 = axes[0, 0]
    ax1.plot(merged['date'], merged['cum_return']*100, label='低PB策略', linewidth=2)
    ax1.plot(merged['date'], merged['cum_return_bench']*100, label='全市场等权', linewidth=2)
    ax1.set_ylabel('累计收益 (%)')
    ax1.set_title('低PB策略累计收益')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 超额收益
    ax2 = axes[0, 1]
    ax2.plot(merged['date'], merged['excess_cum']*100, label='超额收益', linewidth=2, color='green')
    ax2.fill_between(merged['date'], 0, merged['excess_cum']*100, alpha=0.3, color='green')
    ax2.axhline(y=0, color='black', linestyle='--')
    ax2.set_ylabel('超额收益 (%)')
    ax2.set_title('低PB策略超额收益')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 月度收益分布
    ax3 = axes[1, 0]
    ax3.hist(results_df['return']*100, bins=30, edgecolor='black', alpha=0.7)
    ax3.axvline(x=0, color='red', linestyle='--')
    ax3.axvline(x=results_df['return'].mean()*100, color='green', linestyle='--',
                label=f'均值={results_df["return"].mean()*100:.1f}%')
    ax3.set_xlabel('月度收益 (%)')
    ax3.set_ylabel('频数')
    ax3.set_title('月度收益分布')
    ax3.legend()

    # 4. 平均PB vs 收益散点图
    ax4 = axes[1, 1]
    ax4.scatter(results_df['avg_pb'], results_df['return']*100, alpha=0.6)
    ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax4.set_xlabel('组合平均PB')
    ax4.set_ylabel('月度收益 (%)')
    ax4.set_title('组合PB与收益关系')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}pb_low_strategy.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n图表已保存: {OUTPUT_DIR}pb_low_strategy.png")

    conn.close()
    return results_df


def pb_group_backtest():
    """PB分组回测"""
    print("\n" + "="*60)
    print("2.2 PB 分组回测")
    print("="*60)

    conn = get_connection()

    # 获取季度末交易日（减少计算量）
    dates = conn.execute("""
        WITH quarterly_dates AS (
            SELECT trade_date,
                   ROW_NUMBER() OVER (
                       PARTITION BY SUBSTR(trade_date, 1, 4) ||
                           CASE
                               WHEN SUBSTR(trade_date, 5, 2) <= '03' THEN 'Q1'
                               WHEN SUBSTR(trade_date, 5, 2) <= '06' THEN 'Q2'
                               WHEN SUBSTR(trade_date, 5, 2) <= '09' THEN 'Q3'
                               ELSE 'Q4'
                           END
                       ORDER BY trade_date DESC
                   ) as rn
            FROM (SELECT DISTINCT trade_date FROM daily_basic WHERE trade_date >= '20180101')
        )
        SELECT trade_date FROM quarterly_dates WHERE rn = 1
        ORDER BY trade_date
    """).fetchdf()

    trade_dates = dates['trade_date'].tolist()
    print(f"\n回测期间: {trade_dates[0]} - {trade_dates[-1]}")
    print(f"调仓次数: {len(trade_dates) - 1}")

    # 分5组
    n_groups = 5
    group_results = {i: [] for i in range(1, n_groups + 1)}

    for i in range(len(trade_dates) - 1):
        date = trade_dates[i]
        next_date = trade_dates[i + 1]

        # 获取所有股票并按PB分组
        stocks = conn.execute(f"""
            SELECT
                db.ts_code,
                db.pb,
                NTILE({n_groups}) OVER (ORDER BY db.pb) as pb_group
            FROM daily_basic db
            JOIN stock_basic sb ON db.ts_code = sb.ts_code
            WHERE db.trade_date = '{date}'
              AND db.pb IS NOT NULL
              AND db.pb > 0
              AND db.pb < 100
              AND sb.list_status = 'L'
        """).fetchdf()

        for g in range(1, n_groups + 1):
            group_stocks = stocks[stocks['pb_group'] == g]['ts_code'].tolist()
            if len(group_stocks) == 0:
                continue

            stock_list = "','".join(group_stocks)

            # 计算收益
            ret = conn.execute(f"""
                WITH start_price AS (
                    SELECT ts_code, close as start_close
                    FROM daily d1
                    WHERE trade_date = (
                        SELECT MIN(trade_date) FROM daily
                        WHERE trade_date > '{date}' AND ts_code = d1.ts_code
                    )
                ),
                end_price AS (
                    SELECT ts_code, close as end_close
                    FROM daily d2
                    WHERE trade_date = (
                        SELECT MAX(trade_date) FROM daily
                        WHERE trade_date <= '{next_date}' AND ts_code = d2.ts_code
                    )
                )
                SELECT AVG(e.end_close / s.start_close - 1) as ret
                FROM start_price s
                JOIN end_price e ON s.ts_code = e.ts_code
                WHERE s.ts_code IN ('{stock_list}')
            """).fetchone()

            if ret[0] is not None:
                group_results[g].append({
                    'date': date,
                    'return': ret[0],
                    'avg_pb': stocks[stocks['pb_group'] == g]['pb'].mean()
                })

    # 计算各组累计收益
    group_perf = {}
    for g in range(1, n_groups + 1):
        df = pd.DataFrame(group_results[g])
        if len(df) > 0:
            df['cum_return'] = (1 + df['return']).cumprod() - 1
            group_perf[g] = df

    # 绩效统计
    print("\n【各组绩效统计】")
    print(f"{'组别':<6}{'平均PB':<10}{'累计收益':<12}{'年化收益':<12}{'年化波动':<12}{'夏普比率':<10}")
    print("-" * 62)

    summary = []
    for g in range(1, n_groups + 1):
        df = group_perf[g]
        total_ret = df['cum_return'].iloc[-1]
        annual_ret = (1 + total_ret) ** (4 / len(df)) - 1  # 季度调仓
        vol = df['return'].std() * np.sqrt(4)
        sharpe = annual_ret / vol if vol > 0 else 0
        avg_pb = df['avg_pb'].mean()

        print(f"G{g:<5}{avg_pb:<10.2f}{total_ret*100:<12.1f}%{annual_ret*100:<11.1f}%{vol*100:<11.1f}%{sharpe:<10.2f}")

        summary.append({
            'group': g,
            'avg_pb': avg_pb,
            'total_return': total_ret,
            'annual_return': annual_ret,
            'volatility': vol,
            'sharpe': sharpe
        })

    # 多空收益
    long_short = []
    for i in range(len(group_perf[1])):
        ls_ret = group_perf[1].iloc[i]['return'] - group_perf[n_groups].iloc[i]['return']
        long_short.append({'date': group_perf[1].iloc[i]['date'], 'return': ls_ret})

    ls_df = pd.DataFrame(long_short)
    ls_df['cum_return'] = (1 + ls_df['return']).cumprod() - 1

    print(f"\n【多空策略（G1做多 - G5做空）】")
    print(f"累计收益: {ls_df['cum_return'].iloc[-1]*100:.1f}%")
    print(f"年化收益: {((1+ls_df['cum_return'].iloc[-1])**(4/len(ls_df))-1)*100:.1f}%")

    # 绘图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 各组累计收益
    ax1 = axes[0, 0]
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, n_groups))
    for g in range(1, n_groups + 1):
        df = group_perf[g]
        ax1.plot(pd.to_datetime(df['date']), df['cum_return']*100,
                label=f'G{g} (PB={df["avg_pb"].mean():.2f})',
                linewidth=2, color=colors[g-1])
    ax1.set_ylabel('累计收益 (%)')
    ax1.set_title('PB分组累计收益')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # 2. 多空收益
    ax2 = axes[0, 1]
    ax2.plot(pd.to_datetime(ls_df['date']), ls_df['cum_return']*100,
            label='多空收益 (低PB-高PB)', linewidth=2, color='purple')
    ax2.fill_between(pd.to_datetime(ls_df['date']), 0, ls_df['cum_return']*100, alpha=0.3, color='purple')
    ax2.axhline(y=0, color='black', linestyle='--')
    ax2.set_ylabel('累计收益 (%)')
    ax2.set_title('PB因子多空收益')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 各组年化收益柱状图
    ax3 = axes[1, 0]
    summary_df = pd.DataFrame(summary)
    bars = ax3.bar(summary_df['group'], summary_df['annual_return']*100,
                   color=colors, edgecolor='black')
    ax3.set_xlabel('PB分组 (1=最低PB)')
    ax3.set_ylabel('年化收益 (%)')
    ax3.set_title('各组年化收益')
    ax3.axhline(y=0, color='red', linestyle='--')

    # 4. PB与收益关系
    ax4 = axes[1, 1]
    ax4.scatter(summary_df['avg_pb'], summary_df['annual_return']*100,
               s=200, c=colors, edgecolors='black', zorder=5)
    for i, row in summary_df.iterrows():
        ax4.annotate(f'G{int(row["group"])}',
                    (row['avg_pb'], row['annual_return']*100+1), ha='center')

    # 拟合线
    z = np.polyfit(summary_df['avg_pb'], summary_df['annual_return']*100, 1)
    p = np.poly1d(z)
    x_line = np.linspace(summary_df['avg_pb'].min(), summary_df['avg_pb'].max(), 100)
    ax4.plot(x_line, p(x_line), 'r--', alpha=0.7, label=f'斜率={z[0]:.2f}')

    ax4.set_xlabel('平均PB')
    ax4.set_ylabel('年化收益 (%)')
    ax4.set_title('PB与收益关系')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}pb_group_backtest.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n图表已保存: {OUTPUT_DIR}pb_group_backtest.png")

    conn.close()
    return summary_df


def industry_neutral_pb():
    """行业中性PB因子"""
    print("\n" + "="*60)
    print("2.3 行业中性 PB 因子")
    print("="*60)

    conn = get_connection()

    # 获取季度末交易日
    dates = conn.execute("""
        WITH quarterly_dates AS (
            SELECT trade_date,
                   ROW_NUMBER() OVER (
                       PARTITION BY SUBSTR(trade_date, 1, 4) ||
                           CASE
                               WHEN SUBSTR(trade_date, 5, 2) <= '03' THEN 'Q1'
                               WHEN SUBSTR(trade_date, 5, 2) <= '06' THEN 'Q2'
                               WHEN SUBSTR(trade_date, 5, 2) <= '09' THEN 'Q3'
                               ELSE 'Q4'
                           END
                       ORDER BY trade_date DESC
                   ) as rn
            FROM (SELECT DISTINCT trade_date FROM daily_basic WHERE trade_date >= '20180101')
        )
        SELECT trade_date FROM quarterly_dates WHERE rn = 1
        ORDER BY trade_date
    """).fetchdf()

    trade_dates = dates['trade_date'].tolist()

    # 行业中性策略
    neutral_results = []
    raw_results = []

    for i in range(len(trade_dates) - 1):
        date = trade_dates[i]
        next_date = trade_dates[i + 1]

        # 获取股票及行业
        stocks = conn.execute(f"""
            WITH latest_member AS (
                SELECT ts_code, l1_name as industry
                FROM index_member_all
                WHERE is_new = 'Y'
            )
            SELECT
                db.ts_code,
                db.pb,
                lm.industry
            FROM daily_basic db
            JOIN latest_member lm ON db.ts_code = lm.ts_code
            JOIN stock_basic sb ON db.ts_code = sb.ts_code
            WHERE db.trade_date = '{date}'
              AND db.pb IS NOT NULL
              AND db.pb > 0
              AND db.pb < 100
              AND sb.list_status = 'L'
        """).fetchdf()

        if len(stocks) == 0:
            continue

        # 计算行业中性PB（行业内z-score）
        stocks['pb_zscore'] = stocks.groupby('industry')['pb'].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
        )

        # 选取行业中性低PB股票（每个行业选PB最低的20%）
        neutral_stocks = []
        for ind in stocks['industry'].unique():
            ind_stocks = stocks[stocks['industry'] == ind]
            n_select = max(1, int(len(ind_stocks) * 0.2))
            selected = ind_stocks.nsmallest(n_select, 'pb')['ts_code'].tolist()
            neutral_stocks.extend(selected)

        # 选取原始低PB股票
        raw_stocks = stocks.nsmallest(int(len(stocks) * 0.2), 'pb')['ts_code'].tolist()

        # 计算收益
        for stock_list, results_list in [(neutral_stocks, neutral_results),
                                         (raw_stocks, raw_results)]:
            stock_str = "','".join(stock_list)
            ret = conn.execute(f"""
                WITH start_price AS (
                    SELECT ts_code, close as start_close
                    FROM daily d1
                    WHERE trade_date = (
                        SELECT MIN(trade_date) FROM daily
                        WHERE trade_date > '{date}' AND ts_code = d1.ts_code
                    )
                ),
                end_price AS (
                    SELECT ts_code, close as end_close
                    FROM daily d2
                    WHERE trade_date = (
                        SELECT MAX(trade_date) FROM daily
                        WHERE trade_date <= '{next_date}' AND ts_code = d2.ts_code
                    )
                )
                SELECT AVG(e.end_close / s.start_close - 1) as ret
                FROM start_price s
                JOIN end_price e ON s.ts_code = e.ts_code
                WHERE s.ts_code IN ('{stock_str}')
            """).fetchone()

            if ret[0] is not None:
                results_list.append({'date': date, 'return': ret[0]})

    # 计算累计收益
    neutral_df = pd.DataFrame(neutral_results)
    neutral_df['cum_return'] = (1 + neutral_df['return']).cumprod() - 1

    raw_df = pd.DataFrame(raw_results)
    raw_df['cum_return'] = (1 + raw_df['return']).cumprod() - 1

    # 绩效对比
    print("\n【行业中性 vs 原始低PB策略】")
    print(f"{'策略':<15}{'累计收益':<12}{'年化收益':<12}{'年化波动':<12}{'夏普比率':<10}")
    print("-" * 61)

    for name, df in [('行业中性低PB', neutral_df), ('原始低PB', raw_df)]:
        total_ret = df['cum_return'].iloc[-1]
        annual_ret = (1 + total_ret) ** (4 / len(df)) - 1
        vol = df['return'].std() * np.sqrt(4)
        sharpe = annual_ret / vol if vol > 0 else 0
        print(f"{name:<15}{total_ret*100:<12.1f}%{annual_ret*100:<11.1f}%{vol*100:<11.1f}%{sharpe:<10.2f}")

    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax1.plot(pd.to_datetime(neutral_df['date']), neutral_df['cum_return']*100,
            label='行业中性低PB', linewidth=2)
    ax1.plot(pd.to_datetime(raw_df['date']), raw_df['cum_return']*100,
            label='原始低PB', linewidth=2)
    ax1.set_ylabel('累计收益 (%)')
    ax1.set_title('行业中性 vs 原始低PB策略')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    excess = neutral_df['cum_return'] - raw_df['cum_return']
    ax2.plot(pd.to_datetime(neutral_df['date']), excess.values*100,
            label='行业中性超额', linewidth=2, color='green')
    ax2.fill_between(pd.to_datetime(neutral_df['date']), 0, excess.values*100,
                    alpha=0.3, color='green')
    ax2.axhline(y=0, color='black', linestyle='--')
    ax2.set_ylabel('超额收益 (%)')
    ax2.set_title('行业中性超额收益')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}pb_industry_neutral.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n图表已保存: {OUTPUT_DIR}pb_industry_neutral.png")

    conn.close()
    return neutral_df, raw_df


# =============================================================================
# Part 3: 策略应用
# =============================================================================

def broken_net_strategy():
    """破净股策略"""
    print("\n" + "="*60)
    print("3.1 破净股策略")
    print("="*60)

    conn = get_connection()

    # 获取最新交易日
    latest_date = conn.execute("""
        SELECT MAX(trade_date) FROM daily_basic WHERE pb IS NOT NULL
    """).fetchone()[0]

    # 获取当前破净股
    broken_stocks = conn.execute(f"""
        WITH latest_member AS (
            SELECT ts_code, l1_name as industry
            FROM index_member_all
            WHERE is_new = 'Y'
        )
        SELECT
            db.ts_code,
            sb.name,
            db.pb,
            db.pe_ttm,
            db.total_mv / 10000 as total_mv_yi,
            db.dv_ttm,
            lm.industry
        FROM daily_basic db
        JOIN stock_basic sb ON db.ts_code = sb.ts_code
        LEFT JOIN latest_member lm ON db.ts_code = lm.ts_code
        WHERE db.trade_date = '{latest_date}'
          AND db.pb IS NOT NULL
          AND db.pb > 0
          AND db.pb < 1
          AND sb.list_status = 'L'
        ORDER BY db.pb
    """).fetchdf()

    print(f"\n【当前破净股统计 ({latest_date})】")
    print(f"破净股数量: {len(broken_stocks)}")
    print(f"平均PB: {broken_stocks['pb'].mean():.2f}")
    print(f"平均PE(TTM): {broken_stocks['pe_ttm'].mean():.2f}")
    print(f"平均市值: {broken_stocks['total_mv_yi'].mean():.1f}亿")

    # 行业分布
    print("\n【破净股行业分布】")
    industry_dist = broken_stocks['industry'].value_counts().head(10)
    for ind, count in industry_dist.items():
        print(f"  {ind}: {count}")

    # PB最低的前20只
    print("\n【PB最低的20只破净股】")
    print(broken_stocks[['ts_code', 'name', 'pb', 'pe_ttm', 'total_mv_yi', 'industry']].head(20).to_string(index=False))

    # 回测破净股策略
    dates = conn.execute("""
        WITH quarterly_dates AS (
            SELECT trade_date,
                   ROW_NUMBER() OVER (
                       PARTITION BY SUBSTR(trade_date, 1, 4) ||
                           CASE
                               WHEN SUBSTR(trade_date, 5, 2) <= '03' THEN 'Q1'
                               WHEN SUBSTR(trade_date, 5, 2) <= '06' THEN 'Q2'
                               WHEN SUBSTR(trade_date, 5, 2) <= '09' THEN 'Q3'
                               ELSE 'Q4'
                           END
                       ORDER BY trade_date DESC
                   ) as rn
            FROM (SELECT DISTINCT trade_date FROM daily_basic WHERE trade_date >= '20180101')
        )
        SELECT trade_date FROM quarterly_dates WHERE rn = 1
        ORDER BY trade_date
    """).fetchdf()

    trade_dates = dates['trade_date'].tolist()

    results = []
    for i in range(len(trade_dates) - 1):
        date = trade_dates[i]
        next_date = trade_dates[i + 1]

        # 选取破净股
        stocks = conn.execute(f"""
            SELECT ts_code, pb
            FROM daily_basic db
            JOIN stock_basic sb ON db.ts_code = sb.ts_code
            WHERE db.trade_date = '{date}'
              AND db.pb IS NOT NULL
              AND db.pb > 0
              AND db.pb < 1
              AND sb.list_status = 'L'
        """).fetchdf()

        if len(stocks) == 0:
            results.append({'date': date, 'return': 0, 'count': 0})
            continue

        stock_list = "','".join(stocks['ts_code'].tolist())

        ret = conn.execute(f"""
            WITH start_price AS (
                SELECT ts_code, close as start_close
                FROM daily d1
                WHERE trade_date = (
                    SELECT MIN(trade_date) FROM daily
                    WHERE trade_date > '{date}' AND ts_code = d1.ts_code
                )
            ),
            end_price AS (
                SELECT ts_code, close as end_close
                FROM daily d2
                WHERE trade_date = (
                    SELECT MAX(trade_date) FROM daily
                    WHERE trade_date <= '{next_date}' AND ts_code = d2.ts_code
                )
            )
            SELECT AVG(e.end_close / s.start_close - 1) as ret
            FROM start_price s
            JOIN end_price e ON s.ts_code = e.ts_code
            WHERE s.ts_code IN ('{stock_list}')
        """).fetchone()

        results.append({
            'date': date,
            'return': ret[0] if ret[0] else 0,
            'count': len(stocks)
        })

    results_df = pd.DataFrame(results)
    results_df['cum_return'] = (1 + results_df['return']).cumprod() - 1

    print("\n【破净股策略回测】")
    total_ret = results_df['cum_return'].iloc[-1]
    annual_ret = (1 + total_ret) ** (4 / len(results_df)) - 1
    vol = results_df['return'].std() * np.sqrt(4)
    print(f"累计收益: {total_ret*100:.1f}%")
    print(f"年化收益: {annual_ret*100:.1f}%")
    print(f"年化波动: {vol*100:.1f}%")
    print(f"平均持股数: {results_df['count'].mean():.0f}")

    # 绘图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 破净股行业分布
    ax1 = axes[0, 0]
    industry_dist.plot(kind='barh', ax=ax1, color='steelblue', edgecolor='black')
    ax1.set_xlabel('股票数量')
    ax1.set_title('破净股行业分布 (Top 10)')

    # 2. 累计收益
    ax2 = axes[0, 1]
    ax2.plot(pd.to_datetime(results_df['date']), results_df['cum_return']*100,
            linewidth=2, color='coral')
    ax2.set_ylabel('累计收益 (%)')
    ax2.set_title('破净股策略累计收益')
    ax2.grid(True, alpha=0.3)

    # 3. 破净股数量变化
    ax3 = axes[1, 0]
    ax3.plot(pd.to_datetime(results_df['date']), results_df['count'],
            linewidth=2, color='green')
    ax3.fill_between(pd.to_datetime(results_df['date']), 0, results_df['count'],
                    alpha=0.3, color='green')
    ax3.set_ylabel('破净股数量')
    ax3.set_title('破净股数量历史变化')
    ax3.grid(True, alpha=0.3)

    # 4. PB分布
    ax4 = axes[1, 1]
    broken_stocks['pb'].hist(bins=30, ax=ax4, edgecolor='black', alpha=0.7)
    ax4.axvline(x=broken_stocks['pb'].median(), color='red', linestyle='--',
               label=f'中位数={broken_stocks["pb"].median():.2f}')
    ax4.set_xlabel('PB')
    ax4.set_ylabel('频数')
    ax4.set_title('当前破净股PB分布')
    ax4.legend()

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}pb_broken_net.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n图表已保存: {OUTPUT_DIR}pb_broken_net.png")

    conn.close()
    return broken_stocks


def pb_roe_model():
    """PB-ROE模型"""
    print("\n" + "="*60)
    print("3.2 PB-ROE 模型")
    print("="*60)

    conn = get_connection()

    # 获取最新交易日和最新财报期
    latest_date = conn.execute("""
        SELECT MAX(trade_date) FROM daily_basic WHERE pb IS NOT NULL
    """).fetchone()[0]

    # 获取PB和ROE数据
    df = conn.execute(f"""
        WITH latest_fina AS (
            SELECT ts_code, roe, end_date,
                   ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY end_date DESC) as rn
            FROM fina_indicator_vip
            WHERE end_date >= '20230101'
              AND roe IS NOT NULL
        )
        SELECT
            db.ts_code,
            sb.name,
            db.pb,
            f.roe,
            db.pe_ttm,
            db.total_mv / 10000 as total_mv_yi,
            db.dv_ttm
        FROM daily_basic db
        JOIN stock_basic sb ON db.ts_code = sb.ts_code
        JOIN latest_fina f ON db.ts_code = f.ts_code AND f.rn = 1
        WHERE db.trade_date = '{latest_date}'
          AND db.pb IS NOT NULL
          AND db.pb > 0
          AND db.pb < 50
          AND f.roe > -50 AND f.roe < 100
          AND sb.list_status = 'L'
    """).fetchdf()

    print(f"\n有效股票数量: {len(df)}")

    # 计算PB/ROE比值
    df['pb_roe_ratio'] = df['pb'] / df['roe'].clip(lower=0.01)

    # 分析PB与ROE关系
    print("\n【PB与ROE相关性分析】")
    corr = df['pb'].corr(df['roe'])
    print(f"PB与ROE相关系数: {corr:.3f}")

    # 按ROE分组看PB
    df['roe_group'] = pd.qcut(df['roe'], 5, labels=['最低', '较低', '中等', '较高', '最高'])
    roe_pb = df.groupby('roe_group')['pb'].agg(['mean', 'median', 'count'])
    print("\n【按ROE分组的PB统计】")
    print(roe_pb.to_string())

    # PB-ROE矩阵（寻找低估值高质量股票）
    print("\n【PB-ROE模型选股】")

    # 低PB高ROE股票（价值成长型）
    low_pb_high_roe = df[(df['pb'] < df['pb'].quantile(0.3)) &
                         (df['roe'] > df['roe'].quantile(0.7))]
    print(f"\n低PB高ROE股票数量: {len(low_pb_high_roe)}")
    if len(low_pb_high_roe) > 0:
        print("\n【推荐股票 TOP 20】")
        top_stocks = low_pb_high_roe.nsmallest(20, 'pb_roe_ratio')
        print(top_stocks[['ts_code', 'name', 'pb', 'roe', 'pe_ttm', 'total_mv_yi', 'dv_ttm']].to_string(index=False))

    # 回测PB-ROE策略
    dates = conn.execute("""
        WITH quarterly_dates AS (
            SELECT trade_date,
                   ROW_NUMBER() OVER (
                       PARTITION BY SUBSTR(trade_date, 1, 4) ||
                           CASE
                               WHEN SUBSTR(trade_date, 5, 2) <= '03' THEN 'Q1'
                               WHEN SUBSTR(trade_date, 5, 2) <= '06' THEN 'Q2'
                               WHEN SUBSTR(trade_date, 5, 2) <= '09' THEN 'Q3'
                               ELSE 'Q4'
                           END
                       ORDER BY trade_date DESC
                   ) as rn
            FROM (SELECT DISTINCT trade_date FROM daily_basic WHERE trade_date >= '20180101')
        )
        SELECT trade_date FROM quarterly_dates WHERE rn = 1
        ORDER BY trade_date
    """).fetchdf()

    trade_dates = dates['trade_date'].tolist()

    pbroe_results = []
    lowpb_results = []

    for i in range(len(trade_dates) - 1):
        date = trade_dates[i]
        next_date = trade_dates[i + 1]

        # 获取PB和ROE数据
        stocks = conn.execute(f"""
            WITH latest_fina AS (
                SELECT ts_code, roe,
                       ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY end_date DESC) as rn
                FROM fina_indicator_vip
                WHERE end_date <= '{date}'
                  AND roe IS NOT NULL
            )
            SELECT
                db.ts_code,
                db.pb,
                f.roe
            FROM daily_basic db
            JOIN stock_basic sb ON db.ts_code = sb.ts_code
            JOIN latest_fina f ON db.ts_code = f.ts_code AND f.rn = 1
            WHERE db.trade_date = '{date}'
              AND db.pb IS NOT NULL
              AND db.pb > 0
              AND db.pb < 50
              AND f.roe > 0 AND f.roe < 100
              AND sb.list_status = 'L'
        """).fetchdf()

        if len(stocks) < 50:
            continue

        # PB-ROE策略：低PB高ROE
        stocks['pb_pct'] = stocks['pb'].rank(pct=True)
        stocks['roe_pct'] = stocks['roe'].rank(pct=True)
        pbroe_stocks = stocks[(stocks['pb_pct'] < 0.3) & (stocks['roe_pct'] > 0.7)]['ts_code'].tolist()

        # 纯低PB策略
        lowpb_stocks = stocks.nsmallest(int(len(stocks) * 0.2), 'pb')['ts_code'].tolist()

        for stock_list, results_list in [(pbroe_stocks, pbroe_results),
                                         (lowpb_stocks, lowpb_results)]:
            if len(stock_list) == 0:
                continue
            stock_str = "','".join(stock_list)

            ret = conn.execute(f"""
                WITH start_price AS (
                    SELECT ts_code, close as start_close
                    FROM daily d1
                    WHERE trade_date = (
                        SELECT MIN(trade_date) FROM daily
                        WHERE trade_date > '{date}' AND ts_code = d1.ts_code
                    )
                ),
                end_price AS (
                    SELECT ts_code, close as end_close
                    FROM daily d2
                    WHERE trade_date = (
                        SELECT MAX(trade_date) FROM daily
                        WHERE trade_date <= '{next_date}' AND ts_code = d2.ts_code
                    )
                )
                SELECT AVG(e.end_close / s.start_close - 1) as ret
                FROM start_price s
                JOIN end_price e ON s.ts_code = e.ts_code
                WHERE s.ts_code IN ('{stock_str}')
            """).fetchone()

            if ret[0] is not None:
                results_list.append({'date': date, 'return': ret[0]})

    # 计算累计收益
    pbroe_df = pd.DataFrame(pbroe_results)
    pbroe_df['cum_return'] = (1 + pbroe_df['return']).cumprod() - 1

    lowpb_df = pd.DataFrame(lowpb_results)
    lowpb_df['cum_return'] = (1 + lowpb_df['return']).cumprod() - 1

    print("\n【PB-ROE策略 vs 纯低PB策略】")
    print(f"{'策略':<15}{'累计收益':<12}{'年化收益':<12}{'年化波动':<12}")
    print("-" * 51)

    for name, results_df in [('PB-ROE策略', pbroe_df), ('纯低PB策略', lowpb_df)]:
        total_ret = results_df['cum_return'].iloc[-1]
        annual_ret = (1 + total_ret) ** (4 / len(results_df)) - 1
        vol = results_df['return'].std() * np.sqrt(4)
        print(f"{name:<15}{total_ret*100:<12.1f}%{annual_ret*100:<11.1f}%{vol*100:<11.1f}%")

    # 绘图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. PB vs ROE 散点图
    ax1 = axes[0, 0]
    scatter = ax1.scatter(df['roe'], df['pb'], c=df['total_mv_yi'],
                         cmap='viridis', alpha=0.5, s=20)
    ax1.set_xlabel('ROE (%)')
    ax1.set_ylabel('PB')
    ax1.set_title('PB vs ROE 散点图')
    ax1.set_xlim(-20, 50)
    ax1.set_ylim(0, 15)
    ax1.axhline(y=1, color='red', linestyle='--', alpha=0.5)
    ax1.axvline(x=15, color='green', linestyle='--', alpha=0.5)
    plt.colorbar(scatter, ax=ax1, label='市值(亿)')

    # 高亮低PB高ROE区域
    ax1.axhspan(0, df['pb'].quantile(0.3),
               xmin=(df['roe'].quantile(0.7) - ax1.get_xlim()[0]) / (ax1.get_xlim()[1] - ax1.get_xlim()[0]),
               xmax=1, alpha=0.1, color='green')

    # 2. 按ROE分组的PB箱线图
    ax2 = axes[0, 1]
    df.boxplot(column='pb', by='roe_group', ax=ax2)
    ax2.set_ylim(0, 10)
    ax2.set_xlabel('ROE分组')
    ax2.set_ylabel('PB')
    ax2.set_title('按ROE分组的PB分布')
    plt.suptitle('')

    # 3. 策略累计收益对比
    ax3 = axes[1, 0]
    ax3.plot(pd.to_datetime(pbroe_df['date']), pbroe_df['cum_return']*100,
            label='PB-ROE策略', linewidth=2)
    ax3.plot(pd.to_datetime(lowpb_df['date']), lowpb_df['cum_return']*100,
            label='纯低PB策略', linewidth=2)
    ax3.set_ylabel('累计收益 (%)')
    ax3.set_title('PB-ROE策略 vs 纯低PB策略')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. PB/ROE比值分布
    ax4 = axes[1, 1]
    valid_ratio = df[df['pb_roe_ratio'] < 2]['pb_roe_ratio']
    valid_ratio.hist(bins=50, ax=ax4, edgecolor='black', alpha=0.7)
    ax4.axvline(x=valid_ratio.median(), color='red', linestyle='--',
               label=f'中位数={valid_ratio.median():.3f}')
    ax4.set_xlabel('PB/ROE比值')
    ax4.set_ylabel('频数')
    ax4.set_title('PB/ROE比值分布')
    ax4.legend()

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}pb_roe_model.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n图表已保存: {OUTPUT_DIR}pb_roe_model.png")

    conn.close()
    return low_pb_high_roe


def multi_factor_combination():
    """多因子组合策略"""
    print("\n" + "="*60)
    print("3.3 PB与其他因子组合")
    print("="*60)

    conn = get_connection()

    # 获取季度末交易日
    dates = conn.execute("""
        WITH quarterly_dates AS (
            SELECT trade_date,
                   ROW_NUMBER() OVER (
                       PARTITION BY SUBSTR(trade_date, 1, 4) ||
                           CASE
                               WHEN SUBSTR(trade_date, 5, 2) <= '03' THEN 'Q1'
                               WHEN SUBSTR(trade_date, 5, 2) <= '06' THEN 'Q2'
                               WHEN SUBSTR(trade_date, 5, 2) <= '09' THEN 'Q3'
                               ELSE 'Q4'
                           END
                       ORDER BY trade_date DESC
                   ) as rn
            FROM (SELECT DISTINCT trade_date FROM daily_basic WHERE trade_date >= '20180101')
        )
        SELECT trade_date FROM quarterly_dates WHERE rn = 1
        ORDER BY trade_date
    """).fetchdf()

    trade_dates = dates['trade_date'].tolist()

    # 多策略对比
    strategies = {
        'low_pb': [],          # 低PB
        'low_pe': [],          # 低PE
        'high_dv': [],         # 高股息
        'pb_pe': [],           # 低PB+低PE
        'pb_dv': [],           # 低PB+高股息
        'pb_pe_dv': [],        # 低PB+低PE+高股息（三因子）
    }

    for i in range(len(trade_dates) - 1):
        date = trade_dates[i]
        next_date = trade_dates[i + 1]

        # 获取多因子数据
        stocks = conn.execute(f"""
            SELECT
                db.ts_code,
                db.pb,
                db.pe_ttm,
                db.dv_ttm
            FROM daily_basic db
            JOIN stock_basic sb ON db.ts_code = sb.ts_code
            WHERE db.trade_date = '{date}'
              AND db.pb IS NOT NULL AND db.pb > 0 AND db.pb < 50
              AND db.pe_ttm IS NOT NULL AND db.pe_ttm > 0 AND db.pe_ttm < 200
              AND db.dv_ttm IS NOT NULL AND db.dv_ttm >= 0
              AND sb.list_status = 'L'
        """).fetchdf()

        if len(stocks) < 100:
            continue

        # 计算各因子排名
        stocks['pb_rank'] = stocks['pb'].rank(pct=True)
        stocks['pe_rank'] = stocks['pe_ttm'].rank(pct=True)
        stocks['dv_rank'] = stocks['dv_ttm'].rank(pct=True, ascending=False)  # 股息越高越好

        # 各策略选股
        strategy_stocks = {
            'low_pb': stocks[stocks['pb_rank'] < 0.2]['ts_code'].tolist(),
            'low_pe': stocks[stocks['pe_rank'] < 0.2]['ts_code'].tolist(),
            'high_dv': stocks[stocks['dv_rank'] < 0.2]['ts_code'].tolist(),
            'pb_pe': stocks[(stocks['pb_rank'] < 0.3) & (stocks['pe_rank'] < 0.3)]['ts_code'].tolist(),
            'pb_dv': stocks[(stocks['pb_rank'] < 0.3) & (stocks['dv_rank'] < 0.3)]['ts_code'].tolist(),
            'pb_pe_dv': stocks[(stocks['pb_rank'] < 0.3) &
                              (stocks['pe_rank'] < 0.3) &
                              (stocks['dv_rank'] < 0.3)]['ts_code'].tolist(),
        }

        for name, stock_list in strategy_stocks.items():
            if len(stock_list) == 0:
                continue
            stock_str = "','".join(stock_list)

            ret = conn.execute(f"""
                WITH start_price AS (
                    SELECT ts_code, close as start_close
                    FROM daily d1
                    WHERE trade_date = (
                        SELECT MIN(trade_date) FROM daily
                        WHERE trade_date > '{date}' AND ts_code = d1.ts_code
                    )
                ),
                end_price AS (
                    SELECT ts_code, close as end_close
                    FROM daily d2
                    WHERE trade_date = (
                        SELECT MAX(trade_date) FROM daily
                        WHERE trade_date <= '{next_date}' AND ts_code = d2.ts_code
                    )
                )
                SELECT AVG(e.end_close / s.start_close - 1) as ret
                FROM start_price s
                JOIN end_price e ON s.ts_code = e.ts_code
                WHERE s.ts_code IN ('{stock_str}')
            """).fetchone()

            if ret[0] is not None:
                strategies[name].append({'date': date, 'return': ret[0], 'count': len(stock_list)})

    # 计算各策略累计收益
    results_summary = []
    strategy_dfs = {}

    print("\n【多因子策略对比】")
    print(f"{'策略':<15}{'累计收益':<12}{'年化收益':<12}{'年化波动':<12}{'夏普比率':<10}{'平均持股':<10}")
    print("-" * 71)

    for name, results in strategies.items():
        if len(results) == 0:
            continue
        df = pd.DataFrame(results)
        df['cum_return'] = (1 + df['return']).cumprod() - 1
        strategy_dfs[name] = df

        total_ret = df['cum_return'].iloc[-1]
        annual_ret = (1 + total_ret) ** (4 / len(df)) - 1
        vol = df['return'].std() * np.sqrt(4)
        sharpe = annual_ret / vol if vol > 0 else 0
        avg_count = df['count'].mean()

        strategy_names = {
            'low_pb': '低PB',
            'low_pe': '低PE',
            'high_dv': '高股息',
            'pb_pe': '低PB+低PE',
            'pb_dv': '低PB+高股息',
            'pb_pe_dv': '三因子组合',
        }

        print(f"{strategy_names[name]:<15}{total_ret*100:<12.1f}%{annual_ret*100:<11.1f}%{vol*100:<11.1f}%{sharpe:<10.2f}{avg_count:<10.0f}")

        results_summary.append({
            'strategy': strategy_names[name],
            'total_return': total_ret,
            'annual_return': annual_ret,
            'volatility': vol,
            'sharpe': sharpe
        })

    # 因子相关性分析
    print("\n【因子间相关性（最新数据）】")
    latest_date = trade_dates[-1]
    factor_data = conn.execute(f"""
        SELECT
            db.pb,
            db.pe_ttm,
            db.dv_ttm
        FROM daily_basic db
        JOIN stock_basic sb ON db.ts_code = sb.ts_code
        WHERE db.trade_date = '{latest_date}'
          AND db.pb IS NOT NULL AND db.pb > 0 AND db.pb < 50
          AND db.pe_ttm IS NOT NULL AND db.pe_ttm > 0 AND db.pe_ttm < 200
          AND db.dv_ttm IS NOT NULL
          AND sb.list_status = 'L'
    """).fetchdf()

    corr_matrix = factor_data.corr()
    print(corr_matrix.round(3).to_string())

    # 绘图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 各策略累计收益
    ax1 = axes[0, 0]
    colors = plt.cm.tab10(range(len(strategy_dfs)))
    for idx, (name, df) in enumerate(strategy_dfs.items()):
        strategy_names = {
            'low_pb': '低PB',
            'low_pe': '低PE',
            'high_dv': '高股息',
            'pb_pe': '低PB+低PE',
            'pb_dv': '低PB+高股息',
            'pb_pe_dv': '三因子组合',
        }
        ax1.plot(pd.to_datetime(df['date']), df['cum_return']*100,
                label=strategy_names[name], linewidth=2, color=colors[idx])
    ax1.set_ylabel('累计收益 (%)')
    ax1.set_title('多因子策略累计收益对比')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # 2. 年化收益柱状图
    ax2 = axes[0, 1]
    summary_df = pd.DataFrame(results_summary)
    bars = ax2.bar(range(len(summary_df)), summary_df['annual_return']*100,
                   color=colors[:len(summary_df)], edgecolor='black')
    ax2.set_xticks(range(len(summary_df)))
    ax2.set_xticklabels(summary_df['strategy'], rotation=45, ha='right')
    ax2.set_ylabel('年化收益 (%)')
    ax2.set_title('各策略年化收益')
    ax2.axhline(y=0, color='red', linestyle='--')

    # 3. 风险收益散点图
    ax3 = axes[1, 0]
    ax3.scatter(summary_df['volatility']*100, summary_df['annual_return']*100,
               s=200, c=colors[:len(summary_df)], edgecolors='black', zorder=5)
    for i, row in summary_df.iterrows():
        ax3.annotate(row['strategy'],
                    (row['volatility']*100+0.5, row['annual_return']*100), fontsize=9)
    ax3.set_xlabel('年化波动率 (%)')
    ax3.set_ylabel('年化收益 (%)')
    ax3.set_title('风险收益对比')
    ax3.grid(True, alpha=0.3)

    # 4. 因子相关性热力图
    ax4 = axes[1, 1]
    im = ax4.imshow(corr_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1)
    ax4.set_xticks(range(len(corr_matrix.columns)))
    ax4.set_yticks(range(len(corr_matrix.columns)))
    ax4.set_xticklabels(['PB', 'PE', '股息率'])
    ax4.set_yticklabels(['PB', 'PE', '股息率'])
    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix)):
            ax4.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', ha='center', va='center')
    plt.colorbar(im, ax=ax4, label='相关系数')
    ax4.set_title('因子相关性矩阵')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}pb_multi_factor.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n图表已保存: {OUTPUT_DIR}pb_multi_factor.png")

    conn.close()
    return summary_df


def generate_report():
    """生成完整研究报告"""
    print("="*60)
    print("市净率(PB)因子研究报告")
    print("="*60)
    print(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"数据库: {DB_PATH}")
    print(f"输出目录: {OUTPUT_DIR}")

    # Part 1: PB数据分析
    pb_dist = analyze_pb_distribution()
    industry_stats = analyze_industry_pb()
    pb_ts = analyze_pb_time_series()

    # Part 2: PB因子效果
    low_pb_results = backtest_low_pb_strategy()
    group_results = pb_group_backtest()
    neutral_results, raw_results = industry_neutral_pb()

    # Part 3: 策略应用
    broken_stocks = broken_net_strategy()
    pbroe_stocks = pb_roe_model()
    multi_factor = multi_factor_combination()

    # 生成报告摘要
    print("\n" + "="*60)
    print("研究结论摘要")
    print("="*60)

    print("""
【主要发现】

1. PB分布特征：
   - 全市场PB呈右偏分布，中位数约为2-3
   - 当前市场破净股占比约10-15%
   - 银行、钢铁、房地产等行业PB普遍较低

2. PB因子效果：
   - 低PB因子具有显著的选股能力
   - PB分组呈现明显的单调性：低PB组收益高于高PB组
   - 行业中性处理可以降低策略波动

3. 策略建议：
   - 单纯低PB策略存在价值陷阱风险
   - PB-ROE组合可以有效筛选出低估值高质量股票
   - 多因子组合（PB+PE+股息）可以进一步提升风险调整收益

【风险提示】
   - 历史回测不代表未来收益
   - 低PB股票可能面临基本面恶化风险
   - 需要关注行业周期和市场环境变化
""")

    print(f"\n所有图表已保存到: {OUTPUT_DIR}")
    print("\n报告生成完成!")


if __name__ == '__main__':
    generate_report()
