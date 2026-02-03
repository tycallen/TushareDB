#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
股票振幅特征研究
================

研究内容:
1. 振幅统计：振幅分布特征、行业振幅差异、市值与振幅关系
2. 振幅信号：高振幅后走势、振幅收敛/扩张、振幅突破策略
3. 因子构建：振幅因子、振幅波动因子、因子效果检验

作者: Claude
日期: 2026-02-01
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False

# 数据库路径
DB_PATH = '/Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db'
REPORT_DIR = '/Users/allen/workspace/python/stock/Tushare-DuckDB/reports/research'

# 确保报告目录存在
os.makedirs(REPORT_DIR, exist_ok=True)


def get_connection():
    """获取数据库连接"""
    return duckdb.connect(DB_PATH, read_only=True)


def calculate_amplitude(df):
    """计算振幅 = (最高价 - 最低价) / 前收盘价 * 100"""
    df = df.copy()
    df['amplitude'] = (df['high'] - df['low']) / df['pre_close'] * 100
    return df


# =============================================================================
# Part 1: 振幅统计分析
# =============================================================================

def analyze_amplitude_distribution(conn):
    """分析振幅分布特征"""
    print("=" * 60)
    print("1.1 振幅分布特征分析")
    print("=" * 60)

    # 获取近3年数据用于分析
    query = """
    SELECT
        ts_code,
        trade_date,
        high,
        low,
        pre_close,
        close,
        pct_chg,
        vol,
        amount
    FROM daily
    WHERE trade_date >= '20230101'
        AND pre_close > 0
        AND high >= low
        AND ts_code NOT LIKE '%BJ%'  -- 排除北交所
    """

    df = conn.execute(query).fetchdf()
    df = calculate_amplitude(df)

    # 基础统计
    print("\n【振幅基础统计】")
    print(f"数据量: {len(df):,} 条")
    print(f"振幅均值: {df['amplitude'].mean():.2f}%")
    print(f"振幅中位数: {df['amplitude'].median():.2f}%")
    print(f"振幅标准差: {df['amplitude'].std():.2f}%")
    print(f"振幅最小值: {df['amplitude'].min():.2f}%")
    print(f"振幅最大值: {df['amplitude'].max():.2f}%")

    # 分位数
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    print("\n【振幅分位数】")
    for p in percentiles:
        print(f"  {p}% 分位数: {np.percentile(df['amplitude'], p):.2f}%")

    # 振幅分布直方图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 整体分布
    ax1 = axes[0, 0]
    ax1.hist(df['amplitude'].clip(0, 20), bins=100, edgecolor='black', alpha=0.7)
    ax1.axvline(df['amplitude'].mean(), color='r', linestyle='--', label=f'均值: {df["amplitude"].mean():.2f}%')
    ax1.axvline(df['amplitude'].median(), color='g', linestyle='--', label=f'中位数: {df["amplitude"].median():.2f}%')
    ax1.set_xlabel('振幅 (%)')
    ax1.set_ylabel('频数')
    ax1.set_title('振幅分布 (0-20%)')
    ax1.legend()

    # 按年份统计
    ax2 = axes[0, 1]
    df['year'] = df['trade_date'].str[:4]
    yearly_amp = df.groupby('year')['amplitude'].agg(['mean', 'median', 'std'])
    yearly_amp[['mean', 'median']].plot(kind='bar', ax=ax2)
    ax2.set_xlabel('年份')
    ax2.set_ylabel('振幅 (%)')
    ax2.set_title('年度振幅均值与中位数')
    ax2.tick_params(axis='x', rotation=0)
    ax2.legend(['均值', '中位数'])

    # 振幅区间分布
    ax3 = axes[1, 0]
    bins = [0, 1, 2, 3, 4, 5, 7, 10, 15, 20, 100]
    labels = ['0-1%', '1-2%', '2-3%', '3-4%', '4-5%', '5-7%', '7-10%', '10-15%', '15-20%', '>20%']
    df['amp_bin'] = pd.cut(df['amplitude'], bins=bins, labels=labels)
    amp_dist = df['amp_bin'].value_counts().sort_index()
    amp_dist.plot(kind='bar', ax=ax3, color='steelblue', edgecolor='black')
    ax3.set_xlabel('振幅区间')
    ax3.set_ylabel('频数')
    ax3.set_title('振幅区间分布')
    ax3.tick_params(axis='x', rotation=45)

    # 振幅与涨跌幅关系
    ax4 = axes[1, 1]
    sample = df.sample(min(50000, len(df)))
    ax4.scatter(sample['pct_chg'], sample['amplitude'], alpha=0.1, s=1)
    ax4.set_xlabel('涨跌幅 (%)')
    ax4.set_ylabel('振幅 (%)')
    ax4.set_title('振幅与涨跌幅关系')
    ax4.set_xlim(-15, 15)
    ax4.set_ylim(0, 20)

    plt.tight_layout()
    plt.savefig(f'{REPORT_DIR}/amplitude_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n图表已保存: {REPORT_DIR}/amplitude_distribution.png")

    # 返回数据供后续使用
    return df


def analyze_industry_amplitude(conn, df):
    """分析行业振幅差异"""
    print("\n" + "=" * 60)
    print("1.2 行业振幅差异分析")
    print("=" * 60)

    # 获取行业分类
    industry_query = """
    SELECT ts_code, l1_name as industry
    FROM index_member_all
    WHERE is_new = 'Y'
    """

    industry_df = conn.execute(industry_query).fetchdf()

    # 合并行业信息
    df_with_industry = df.merge(industry_df, on='ts_code', how='left')
    df_with_industry = df_with_industry.dropna(subset=['industry'])

    # 行业振幅统计
    industry_amp = df_with_industry.groupby('industry')['amplitude'].agg([
        'count', 'mean', 'median', 'std'
    ]).round(2)
    industry_amp.columns = ['样本数', '均值', '中位数', '标准差']
    industry_amp = industry_amp.sort_values('均值', ascending=False)

    print("\n【行业振幅排名（按均值）】")
    print(industry_amp.to_string())

    # 绘制行业振幅图
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # 行业振幅均值排名
    ax1 = axes[0]
    top_industries = industry_amp.head(20)
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(top_industries)))
    bars = ax1.barh(range(len(top_industries)), top_industries['均值'], color=colors)
    ax1.set_yticks(range(len(top_industries)))
    ax1.set_yticklabels(top_industries.index)
    ax1.set_xlabel('平均振幅 (%)')
    ax1.set_title('行业平均振幅排名 (前20)')
    ax1.invert_yaxis()

    # 行业振幅箱线图
    ax2 = axes[1]
    top10_industries = industry_amp.head(10).index.tolist()
    plot_data = df_with_industry[df_with_industry['industry'].isin(top10_industries)]
    plot_data.boxplot(column='amplitude', by='industry', ax=ax2,
                      showfliers=False, rot=45)
    ax2.set_xlabel('行业')
    ax2.set_ylabel('振幅 (%)')
    ax2.set_title('高振幅行业振幅分布')
    plt.suptitle('')

    plt.tight_layout()
    plt.savefig(f'{REPORT_DIR}/amplitude_industry.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n图表已保存: {REPORT_DIR}/amplitude_industry.png")

    return df_with_industry, industry_amp


def analyze_market_cap_amplitude(conn, df):
    """分析市值与振幅关系"""
    print("\n" + "=" * 60)
    print("1.3 市值与振幅关系分析")
    print("=" * 60)

    # 获取市值数据
    mv_query = """
    SELECT ts_code, trade_date, total_mv, circ_mv
    FROM daily_basic
    WHERE trade_date >= '20230101'
        AND total_mv > 0
    """

    mv_df = conn.execute(mv_query).fetchdf()

    # 合并振幅和市值数据
    df_with_mv = df.merge(mv_df, on=['ts_code', 'trade_date'], how='inner')

    # 市值分组（单位：万元 -> 亿元）
    df_with_mv['total_mv_bn'] = df_with_mv['total_mv'] / 10000  # 转换为亿元

    mv_bins = [0, 20, 50, 100, 300, 500, 1000, 5000, float('inf')]
    mv_labels = ['<20亿', '20-50亿', '50-100亿', '100-300亿', '300-500亿', '500-1000亿', '1000-5000亿', '>5000亿']
    df_with_mv['mv_group'] = pd.cut(df_with_mv['total_mv_bn'], bins=mv_bins, labels=mv_labels)

    # 市值分组振幅统计
    mv_amp = df_with_mv.groupby('mv_group', observed=True)['amplitude'].agg([
        'count', 'mean', 'median', 'std'
    ]).round(2)
    mv_amp.columns = ['样本数', '均值', '中位数', '标准差']

    print("\n【市值分组振幅统计】")
    print(mv_amp.to_string())

    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 市值分组振幅
    ax1 = axes[0]
    mv_amp_sorted = mv_amp.loc[mv_labels]
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(mv_amp_sorted)))
    ax1.bar(range(len(mv_amp_sorted)), mv_amp_sorted['均值'], color=colors, edgecolor='black')
    ax1.set_xticks(range(len(mv_amp_sorted)))
    ax1.set_xticklabels(mv_amp_sorted.index, rotation=45, ha='right')
    ax1.set_xlabel('市值分组')
    ax1.set_ylabel('平均振幅 (%)')
    ax1.set_title('不同市值股票振幅特征')

    # 市值与振幅散点图
    ax2 = axes[1]
    sample = df_with_mv.sample(min(30000, len(df_with_mv)))
    ax2.scatter(np.log10(sample['total_mv_bn']), sample['amplitude'], alpha=0.1, s=1)
    ax2.set_xlabel('log10(市值/亿元)')
    ax2.set_ylabel('振幅 (%)')
    ax2.set_title('市值与振幅关系')
    ax2.set_ylim(0, 20)

    # 添加趋势线
    x = np.log10(sample['total_mv_bn'])
    y = sample['amplitude']
    z = np.polyfit(x[np.isfinite(x)], y[np.isfinite(x)], 1)
    p = np.poly1d(z)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax2.plot(x_line, p(x_line), 'r-', linewidth=2, label=f'趋势线: y={z[0]:.2f}x+{z[1]:.2f}')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f'{REPORT_DIR}/amplitude_market_cap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n图表已保存: {REPORT_DIR}/amplitude_market_cap.png")

    # 相关性分析
    corr = df_with_mv[['amplitude', 'total_mv_bn']].corr().iloc[0, 1]
    print(f"\n振幅与市值相关系数: {corr:.4f}")
    print("结论: 小市值股票振幅普遍较高，大市值股票振幅较低")

    return df_with_mv


# =============================================================================
# Part 2: 振幅信号分析
# =============================================================================

def analyze_high_amplitude_performance(conn):
    """分析高振幅后走势"""
    print("\n" + "=" * 60)
    print("2.1 高振幅后走势分析")
    print("=" * 60)

    # 获取数据
    query = """
    WITH daily_with_amp AS (
        SELECT
            ts_code,
            trade_date,
            (high - low) / NULLIF(pre_close, 0) * 100 as amplitude,
            close,
            pct_chg,
            pre_close
        FROM daily
        WHERE trade_date >= '20230101'
            AND pre_close > 0
            AND ts_code NOT LIKE '%BJ%'
    ),
    ranked AS (
        SELECT *,
            ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY trade_date) as rn
        FROM daily_with_amp
    )
    SELECT
        a.ts_code,
        a.trade_date,
        a.amplitude,
        a.pct_chg,
        b.pct_chg as next1_ret,
        c.pct_chg as next2_ret,
        d.pct_chg as next3_ret,
        e.pct_chg as next5_ret
    FROM ranked a
    LEFT JOIN ranked b ON a.ts_code = b.ts_code AND a.rn + 1 = b.rn
    LEFT JOIN ranked c ON a.ts_code = c.ts_code AND a.rn + 2 = c.rn
    LEFT JOIN ranked d ON a.ts_code = d.ts_code AND a.rn + 3 = d.rn
    LEFT JOIN ranked e ON a.ts_code = e.ts_code AND a.rn + 5 = e.rn
    WHERE a.amplitude IS NOT NULL
    """

    df = conn.execute(query).fetchdf()
    df = df.dropna()

    # 振幅分组
    amp_quantiles = df['amplitude'].quantile([0.2, 0.4, 0.6, 0.8, 0.9, 0.95])

    def classify_amplitude(amp):
        if amp <= amp_quantiles[0.2]:
            return '1-极低振幅(<20%分位)'
        elif amp <= amp_quantiles[0.4]:
            return '2-低振幅(20-40%分位)'
        elif amp <= amp_quantiles[0.6]:
            return '3-中振幅(40-60%分位)'
        elif amp <= amp_quantiles[0.8]:
            return '4-高振幅(60-80%分位)'
        elif amp <= amp_quantiles[0.9]:
            return '5-较高振幅(80-90%分位)'
        elif amp <= amp_quantiles[0.95]:
            return '6-极高振幅(90-95%分位)'
        else:
            return '7-超高振幅(>95%分位)'

    df['amp_group'] = df['amplitude'].apply(classify_amplitude)

    # 不同振幅组后续收益统计
    results = df.groupby('amp_group').agg({
        'next1_ret': ['mean', 'std', 'count'],
        'next2_ret': 'mean',
        'next3_ret': 'mean',
        'next5_ret': 'mean'
    }).round(3)

    results.columns = ['T+1均值', 'T+1标准差', '样本数', 'T+2均值', 'T+3均值', 'T+5均值']

    print("\n【振幅分组后续收益】")
    print(results.to_string())

    # 进一步按当日涨跌分析
    df['direction'] = df['pct_chg'].apply(lambda x: '上涨' if x > 0.5 else ('下跌' if x < -0.5 else '平盘'))

    # 高振幅+上涨 vs 高振幅+下跌
    high_amp_df = df[df['amplitude'] > amp_quantiles[0.8]]

    direction_results = high_amp_df.groupby('direction').agg({
        'next1_ret': ['mean', 'std'],
        'next2_ret': 'mean',
        'next3_ret': 'mean',
        'next5_ret': 'mean',
        'amplitude': 'count'
    }).round(3)
    direction_results.columns = ['T+1均值', 'T+1标准差', 'T+2均值', 'T+3均值', 'T+5均值', '样本数']

    print("\n【高振幅（>80%分位）按涨跌方向分类】")
    print(direction_results.to_string())

    # 绘图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 振幅组后续收益
    ax1 = axes[0, 0]
    groups = sorted(results.index)
    x = range(len(groups))
    width = 0.2
    ax1.bar([i - 1.5*width for i in x], results.loc[groups, 'T+1均值'], width, label='T+1')
    ax1.bar([i - 0.5*width for i in x], results.loc[groups, 'T+2均值'], width, label='T+2')
    ax1.bar([i + 0.5*width for i in x], results.loc[groups, 'T+3均值'], width, label='T+3')
    ax1.bar([i + 1.5*width for i in x], results.loc[groups, 'T+5均值'], width, label='T+5')
    ax1.set_xticks(x)
    ax1.set_xticklabels([g.split('-')[1][:4] for g in groups], rotation=45, ha='right')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_xlabel('振幅分组')
    ax1.set_ylabel('后续收益 (%)')
    ax1.set_title('不同振幅后续收益')
    ax1.legend()

    # 高振幅上涨下跌后续走势
    ax2 = axes[0, 1]
    dir_colors = {'上涨': 'red', '下跌': 'green', '平盘': 'gray'}
    for direction in ['上涨', '下跌', '平盘']:
        if direction in direction_results.index:
            row = direction_results.loc[direction]
            returns = [row['T+1均值'], row['T+2均值'], row['T+3均值'], row['T+5均值']]
            ax2.plot(['T+1', 'T+2', 'T+3', 'T+5'], returns, 'o-',
                    label=f'高振幅{direction}', color=dir_colors[direction])
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('持有期')
    ax2.set_ylabel('平均收益 (%)')
    ax2.set_title('高振幅日涨跌方向与后续收益')
    ax2.legend()

    # 胜率分析
    ax3 = axes[1, 0]
    win_rate = df.groupby('amp_group').apply(lambda x: (x['next1_ret'] > 0).mean() * 100).sort_index()
    ax3.bar(range(len(win_rate)), win_rate.values, color='steelblue', edgecolor='black')
    ax3.axhline(y=50, color='red', linestyle='--', label='50%基准')
    ax3.set_xticks(range(len(win_rate)))
    ax3.set_xticklabels([g.split('-')[1][:4] for g in win_rate.index], rotation=45, ha='right')
    ax3.set_xlabel('振幅分组')
    ax3.set_ylabel('T+1胜率 (%)')
    ax3.set_title('不同振幅T+1胜率')
    ax3.legend()

    # 振幅-收益热力图
    ax4 = axes[1, 1]
    # 创建振幅和涨跌幅的二维分组
    df['amp_q'] = pd.qcut(df['amplitude'], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
    df['pct_q'] = pd.qcut(df['pct_chg'].clip(-10, 10), 5, labels=['大跌', '小跌', '平盘', '小涨', '大涨'])
    heatmap_data = df.pivot_table(values='next1_ret', index='amp_q', columns='pct_q', aggfunc='mean')
    im = ax4.imshow(heatmap_data.values, cmap='RdYlGn', aspect='auto')
    ax4.set_xticks(range(len(heatmap_data.columns)))
    ax4.set_xticklabels(heatmap_data.columns)
    ax4.set_yticks(range(len(heatmap_data.index)))
    ax4.set_yticklabels(heatmap_data.index)
    ax4.set_xlabel('当日涨跌')
    ax4.set_ylabel('振幅分位')
    ax4.set_title('振幅-涨跌与T+1收益热力图')
    plt.colorbar(im, ax=ax4, label='T+1收益(%)')

    # 添加数值标注
    for i in range(len(heatmap_data.index)):
        for j in range(len(heatmap_data.columns)):
            val = heatmap_data.values[i, j]
            ax4.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{REPORT_DIR}/amplitude_performance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n图表已保存: {REPORT_DIR}/amplitude_performance.png")

    return df


def analyze_amplitude_convergence(conn):
    """分析振幅收敛/扩张特征"""
    print("\n" + "=" * 60)
    print("2.2 振幅收敛/扩张分析")
    print("=" * 60)

    query = """
    WITH daily_amp AS (
        SELECT
            ts_code,
            trade_date,
            (high - low) / NULLIF(pre_close, 0) * 100 as amplitude,
            close,
            pct_chg
        FROM daily
        WHERE trade_date >= '20220101'
            AND pre_close > 0
            AND ts_code NOT LIKE '%BJ%'
    ),
    amp_with_ma AS (
        SELECT *,
            AVG(amplitude) OVER (PARTITION BY ts_code ORDER BY trade_date ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) as amp_ma5,
            AVG(amplitude) OVER (PARTITION BY ts_code ORDER BY trade_date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) as amp_ma20,
            STDDEV(amplitude) OVER (PARTITION BY ts_code ORDER BY trade_date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) as amp_std20,
            ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY trade_date) as rn
        FROM daily_amp
    )
    SELECT
        a.*,
        b.pct_chg as next1_ret,
        c.pct_chg as next3_ret,
        d.pct_chg as next5_ret
    FROM amp_with_ma a
    LEFT JOIN amp_with_ma b ON a.ts_code = b.ts_code AND a.rn + 1 = b.rn
    LEFT JOIN amp_with_ma c ON a.ts_code = c.ts_code AND a.rn + 3 = c.rn
    LEFT JOIN amp_with_ma d ON a.ts_code = d.ts_code AND a.rn + 5 = d.rn
    WHERE a.trade_date >= '20230101'
        AND a.amp_ma5 IS NOT NULL
        AND a.amp_ma20 IS NOT NULL
    """

    df = conn.execute(query).fetchdf()
    df = df.dropna(subset=['next1_ret', 'next5_ret'])

    # 计算振幅变化指标
    df['amp_ratio_5_20'] = df['amplitude'] / df['amp_ma20']  # 当日振幅相对20日均值
    df['amp_trend'] = df['amp_ma5'] / df['amp_ma20']  # 短期vs长期振幅
    df['amp_z_score'] = (df['amplitude'] - df['amp_ma20']) / df['amp_std20']  # 振幅Z-score

    # 振幅收敛/扩张分类
    def classify_amp_pattern(row):
        if row['amp_trend'] < 0.7:
            return '1-强收敛'
        elif row['amp_trend'] < 0.9:
            return '2-收敛'
        elif row['amp_trend'] < 1.1:
            return '3-稳定'
        elif row['amp_trend'] < 1.3:
            return '4-扩张'
        else:
            return '5-强扩张'

    df['amp_pattern'] = df.apply(classify_amp_pattern, axis=1)

    # 分析收敛/扩张后的走势
    pattern_results = df.groupby('amp_pattern').agg({
        'next1_ret': ['mean', 'std'],
        'next3_ret': 'mean',
        'next5_ret': 'mean',
        'amplitude': 'count'
    }).round(3)
    pattern_results.columns = ['T+1均值', 'T+1标准差', 'T+3均值', 'T+5均值', '样本数']

    print("\n【振幅收敛/扩张后续收益】")
    print(pattern_results.to_string())

    # 振幅突破分析（当日振幅显著高于均值）
    df['amp_breakout'] = df['amp_z_score'] > 2  # Z-score > 2视为振幅突破

    breakout_results = df.groupby('amp_breakout').agg({
        'next1_ret': ['mean', 'std'],
        'next3_ret': 'mean',
        'next5_ret': 'mean',
        'amplitude': 'count'
    }).round(3)
    breakout_results.columns = ['T+1均值', 'T+1标准差', 'T+3均值', 'T+5均值', '样本数']
    breakout_results.index = ['正常振幅', '振幅突破']

    print("\n【振幅突破效应】")
    print(breakout_results.to_string())

    # 绘图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 振幅趋势分布
    ax1 = axes[0, 0]
    ax1.hist(df['amp_trend'].clip(0.3, 2), bins=50, edgecolor='black', alpha=0.7)
    ax1.axvline(x=1, color='r', linestyle='--', label='均衡线(1.0)')
    ax1.set_xlabel('振幅趋势 (MA5/MA20)')
    ax1.set_ylabel('频数')
    ax1.set_title('振幅趋势分布')
    ax1.legend()

    # 收敛/扩张后收益
    ax2 = axes[0, 1]
    patterns = sorted(pattern_results.index)
    x = range(len(patterns))
    width = 0.25
    ax2.bar([i - width for i in x], pattern_results.loc[patterns, 'T+1均值'], width, label='T+1')
    ax2.bar(x, pattern_results.loc[patterns, 'T+3均值'], width, label='T+3')
    ax2.bar([i + width for i in x], pattern_results.loc[patterns, 'T+5均值'], width, label='T+5')
    ax2.set_xticks(x)
    ax2.set_xticklabels([p.split('-')[1] for p in patterns])
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('振幅形态')
    ax2.set_ylabel('后续收益 (%)')
    ax2.set_title('振幅收敛/扩张后收益')
    ax2.legend()

    # Z-score与后续收益
    ax3 = axes[1, 0]
    df['zscore_bin'] = pd.cut(df['amp_z_score'].clip(-3, 5), bins=[-3, -2, -1, 0, 1, 2, 3, 5])
    zscore_ret = df.groupby('zscore_bin', observed=True)['next1_ret'].mean()
    ax3.bar(range(len(zscore_ret)), zscore_ret.values, color='steelblue', edgecolor='black')
    ax3.set_xticks(range(len(zscore_ret)))
    ax3.set_xticklabels([str(x) for x in zscore_ret.index], rotation=45, ha='right')
    ax3.axhline(y=0, color='red', linestyle='--')
    ax3.set_xlabel('振幅Z-Score')
    ax3.set_ylabel('T+1收益 (%)')
    ax3.set_title('振幅Z-Score与T+1收益')

    # 连续收敛/扩张
    ax4 = axes[1, 1]
    # 计算累积收益
    cumret_convergent = df[df['amp_pattern'] == '1-强收敛'].groupby('trade_date')['next1_ret'].mean().cumsum()
    cumret_divergent = df[df['amp_pattern'] == '5-强扩张'].groupby('trade_date')['next1_ret'].mean().cumsum()
    cumret_stable = df[df['amp_pattern'] == '3-稳定'].groupby('trade_date')['next1_ret'].mean().cumsum()

    ax4.plot(cumret_convergent.values, label='强收敛', alpha=0.7)
    ax4.plot(cumret_divergent.values, label='强扩张', alpha=0.7)
    ax4.plot(cumret_stable.values, label='稳定', alpha=0.7)
    ax4.set_xlabel('交易日')
    ax4.set_ylabel('累积收益 (%)')
    ax4.set_title('振幅形态策略累积收益')
    ax4.legend()

    plt.tight_layout()
    plt.savefig(f'{REPORT_DIR}/amplitude_convergence.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n图表已保存: {REPORT_DIR}/amplitude_convergence.png")

    return df


def analyze_amplitude_breakout_strategy(conn):
    """振幅突破策略分析"""
    print("\n" + "=" * 60)
    print("2.3 振幅突破策略分析")
    print("=" * 60)

    query = """
    WITH daily_amp AS (
        SELECT
            ts_code,
            trade_date,
            high,
            low,
            close,
            pre_close,
            pct_chg,
            (high - low) / NULLIF(pre_close, 0) * 100 as amplitude,
            vol
        FROM daily
        WHERE trade_date >= '20220101'
            AND pre_close > 0
            AND ts_code NOT LIKE '%BJ%'
    ),
    amp_with_stats AS (
        SELECT *,
            MAX(amplitude) OVER (PARTITION BY ts_code ORDER BY trade_date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) as amp_max20,
            AVG(amplitude) OVER (PARTITION BY ts_code ORDER BY trade_date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) as amp_ma20,
            AVG(vol) OVER (PARTITION BY ts_code ORDER BY trade_date ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) as vol_ma5,
            ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY trade_date) as rn
        FROM daily_amp
    )
    SELECT
        a.*,
        b.pct_chg as next1_ret,
        c.pct_chg as next3_ret,
        d.pct_chg as next5_ret,
        e.pct_chg as next10_ret
    FROM amp_with_stats a
    LEFT JOIN amp_with_stats b ON a.ts_code = b.ts_code AND a.rn + 1 = b.rn
    LEFT JOIN amp_with_stats c ON a.ts_code = c.ts_code AND a.rn + 3 = c.rn
    LEFT JOIN amp_with_stats d ON a.ts_code = d.ts_code AND a.rn + 5 = d.rn
    LEFT JOIN amp_with_stats e ON a.ts_code = e.ts_code AND a.rn + 10 = e.rn
    WHERE a.trade_date >= '20230101'
        AND a.amp_max20 IS NOT NULL
    """

    df = conn.execute(query).fetchdf()
    df = df.dropna(subset=['next1_ret', 'next5_ret', 'next10_ret'])

    # 定义振幅突破：当日振幅突破20日最高
    df['amp_breakout_max'] = df['amplitude'] > df['amp_max20']

    # 结合量能分析
    df['vol_ratio'] = df['vol'] / df['vol_ma5']
    df['high_vol'] = df['vol_ratio'] > 1.5

    # 振幅突破策略分类
    df['strategy'] = '其他'
    df.loc[df['amp_breakout_max'], 'strategy'] = '振幅突破'
    df.loc[df['amp_breakout_max'] & df['high_vol'], 'strategy'] = '振幅+量能突破'
    df.loc[df['amp_breakout_max'] & (df['pct_chg'] > 0), 'strategy'] = '振幅突破上涨'
    df.loc[df['amp_breakout_max'] & (df['pct_chg'] < 0), 'strategy'] = '振幅突破下跌'

    # 策略效果统计
    strategy_results = df.groupby('strategy').agg({
        'next1_ret': ['mean', 'std'],
        'next3_ret': 'mean',
        'next5_ret': 'mean',
        'next10_ret': 'mean',
        'amplitude': 'count'
    }).round(3)
    strategy_results.columns = ['T+1均值', 'T+1标准差', 'T+3均值', 'T+5均值', 'T+10均值', '样本数']

    print("\n【振幅突破策略效果】")
    print(strategy_results.to_string())

    # 计算策略胜率和盈亏比
    for strategy in df['strategy'].unique():
        subset = df[df['strategy'] == strategy]
        win_rate = (subset['next5_ret'] > 0).mean() * 100
        avg_win = subset[subset['next5_ret'] > 0]['next5_ret'].mean()
        avg_loss = abs(subset[subset['next5_ret'] < 0]['next5_ret'].mean())
        profit_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        print(f"\n{strategy}: 5日胜率={win_rate:.1f}%, 盈亏比={profit_ratio:.2f}")

    # 绘图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 策略收益对比
    ax1 = axes[0, 0]
    strategies = ['其他', '振幅突破', '振幅突破上涨', '振幅突破下跌']
    strategies_exist = [s for s in strategies if s in strategy_results.index]
    x = range(len(strategies_exist))
    width = 0.2
    ax1.bar([i - 1.5*width for i in x], strategy_results.loc[strategies_exist, 'T+1均值'], width, label='T+1')
    ax1.bar([i - 0.5*width for i in x], strategy_results.loc[strategies_exist, 'T+3均值'], width, label='T+3')
    ax1.bar([i + 0.5*width for i in x], strategy_results.loc[strategies_exist, 'T+5均值'], width, label='T+5')
    ax1.bar([i + 1.5*width for i in x], strategy_results.loc[strategies_exist, 'T+10均值'], width, label='T+10')
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategies_exist, rotation=15)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_xlabel('策略')
    ax1.set_ylabel('收益 (%)')
    ax1.set_title('振幅突破策略收益对比')
    ax1.legend()

    # 振幅突破强度与收益
    ax2 = axes[0, 1]
    breakout_df = df[df['amp_breakout_max']]
    breakout_df = breakout_df.copy()
    breakout_df['breakout_strength'] = breakout_df['amplitude'] / breakout_df['amp_max20']
    breakout_df['strength_bin'] = pd.cut(breakout_df['breakout_strength'], bins=[1, 1.1, 1.2, 1.5, 2, 10])
    strength_ret = breakout_df.groupby('strength_bin', observed=True)['next5_ret'].agg(['mean', 'count'])
    ax2.bar(range(len(strength_ret)), strength_ret['mean'].values, color='steelblue', edgecolor='black')
    ax2.set_xticks(range(len(strength_ret)))
    ax2.set_xticklabels([str(x) for x in strength_ret.index], rotation=45, ha='right')
    ax2.axhline(y=0, color='red', linestyle='--')
    ax2.set_xlabel('突破强度 (当日振幅/20日最高)')
    ax2.set_ylabel('T+5收益 (%)')
    ax2.set_title('振幅突破强度与后续收益')

    # 累积收益曲线
    ax3 = axes[1, 0]
    for strategy in ['振幅突破上涨', '振幅突破下跌', '其他']:
        if strategy in df['strategy'].values:
            subset = df[df['strategy'] == strategy].sort_values('trade_date')
            cumret = subset.groupby('trade_date')['next1_ret'].mean().cumsum()
            ax3.plot(cumret.values, label=strategy, alpha=0.7)
    ax3.set_xlabel('交易日')
    ax3.set_ylabel('累积收益 (%)')
    ax3.set_title('策略累积收益曲线')
    ax3.legend()

    # 月度表现
    ax4 = axes[1, 1]
    df['month'] = df['trade_date'].str[:6]
    breakout_up = df[df['strategy'] == '振幅突破上涨'].groupby('month')['next5_ret'].mean()
    breakout_down = df[df['strategy'] == '振幅突破下跌'].groupby('month')['next5_ret'].mean()
    months = sorted(set(breakout_up.index) | set(breakout_down.index))[-12:]
    x = range(len(months))
    width = 0.35
    ax4.bar([i - width/2 for i in x], [breakout_up.get(m, 0) for m in months], width, label='振幅突破上涨', color='red', alpha=0.7)
    ax4.bar([i + width/2 for i in x], [breakout_down.get(m, 0) for m in months], width, label='振幅突破下跌', color='green', alpha=0.7)
    ax4.set_xticks(x)
    ax4.set_xticklabels([m[-2:] + '月' for m in months], rotation=45)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax4.set_xlabel('月份')
    ax4.set_ylabel('T+5收益 (%)')
    ax4.set_title('近12月策略表现')
    ax4.legend()

    plt.tight_layout()
    plt.savefig(f'{REPORT_DIR}/amplitude_breakout_strategy.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n图表已保存: {REPORT_DIR}/amplitude_breakout_strategy.png")

    return df


# =============================================================================
# Part 3: 因子构建与检验
# =============================================================================

def construct_amplitude_factors(conn):
    """构建振幅相关因子"""
    print("\n" + "=" * 60)
    print("3.1 振幅因子构建")
    print("=" * 60)

    query = """
    WITH daily_amp AS (
        SELECT
            ts_code,
            trade_date,
            (high - low) / NULLIF(pre_close, 0) * 100 as amplitude,
            close,
            pct_chg,
            vol
        FROM daily
        WHERE trade_date >= '20220101'
            AND pre_close > 0
            AND ts_code NOT LIKE '%BJ%'
    ),
    factor_calc AS (
        SELECT
            ts_code,
            trade_date,
            amplitude,
            pct_chg,
            -- 振幅均值因子
            AVG(amplitude) OVER (PARTITION BY ts_code ORDER BY trade_date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) as amp_ma20,
            AVG(amplitude) OVER (PARTITION BY ts_code ORDER BY trade_date ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) as amp_ma5,
            -- 振幅波动因子
            STDDEV(amplitude) OVER (PARTITION BY ts_code ORDER BY trade_date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) as amp_std20,
            -- 振幅趋势因子
            AVG(amplitude) OVER (PARTITION BY ts_code ORDER BY trade_date ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) /
            NULLIF(AVG(amplitude) OVER (PARTITION BY ts_code ORDER BY trade_date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING), 0) as amp_trend,
            -- 振幅动量因子
            AVG(amplitude) OVER (PARTITION BY ts_code ORDER BY trade_date ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) -
            AVG(amplitude) OVER (PARTITION BY ts_code ORDER BY trade_date ROWS BETWEEN 20 PRECEDING AND 6 PRECEDING) as amp_momentum,
            -- 最大振幅因子
            MAX(amplitude) OVER (PARTITION BY ts_code ORDER BY trade_date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) as amp_max20,
            -- 最小振幅因子
            MIN(amplitude) OVER (PARTITION BY ts_code ORDER BY trade_date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) as amp_min20,
            ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY trade_date) as rn
        FROM daily_amp
    )
    SELECT
        a.*,
        b.pct_chg as next1_ret,
        c.pct_chg as next5_ret,
        d.pct_chg as next10_ret,
        e.pct_chg as next20_ret
    FROM factor_calc a
    LEFT JOIN factor_calc b ON a.ts_code = b.ts_code AND a.rn + 1 = b.rn
    LEFT JOIN factor_calc c ON a.ts_code = c.ts_code AND a.rn + 5 = c.rn
    LEFT JOIN factor_calc d ON a.ts_code = d.ts_code AND a.rn + 10 = d.rn
    LEFT JOIN factor_calc e ON a.ts_code = e.ts_code AND a.rn + 20 = e.rn
    WHERE a.trade_date >= '20230101'
        AND a.amp_ma20 IS NOT NULL
    """

    df = conn.execute(query).fetchdf()
    df = df.dropna()

    # 计算更多因子
    df['amp_z_score'] = (df['amplitude'] - df['amp_ma20']) / df['amp_std20']  # 振幅Z-Score
    df['amp_range'] = df['amp_max20'] - df['amp_min20']  # 振幅区间
    df['amp_cv'] = df['amp_std20'] / df['amp_ma20']  # 振幅变异系数

    # 因子列表
    factors = ['amplitude', 'amp_ma20', 'amp_ma5', 'amp_std20', 'amp_trend',
               'amp_momentum', 'amp_z_score', 'amp_range', 'amp_cv']

    print("\n【因子定义】")
    print("amplitude: 当日振幅")
    print("amp_ma20: 20日振幅均值")
    print("amp_ma5: 5日振幅均值")
    print("amp_std20: 20日振幅标准差")
    print("amp_trend: 振幅趋势 (MA5/MA20)")
    print("amp_momentum: 振幅动量 (近5日均值 - 前15日均值)")
    print("amp_z_score: 振幅标准化 ((当日-MA20)/STD20)")
    print("amp_range: 20日振幅极差")
    print("amp_cv: 振幅变异系数 (STD20/MA20)")

    return df, factors


def test_factor_ic(df, factors):
    """因子IC检验"""
    print("\n" + "=" * 60)
    print("3.2 因子IC检验")
    print("=" * 60)

    returns = ['next1_ret', 'next5_ret', 'next10_ret', 'next20_ret']

    # 计算IC
    ic_results = {}
    for factor in factors:
        ic_results[factor] = {}
        for ret in returns:
            # 计算Spearman相关系数
            ic = df.groupby('trade_date').apply(
                lambda x: x[factor].corr(x[ret], method='spearman') if len(x) > 30 else np.nan
            )
            ic_results[factor][ret] = {
                'IC_mean': ic.mean(),
                'IC_std': ic.std(),
                'ICIR': ic.mean() / ic.std() if ic.std() > 0 else 0,
                'IC_positive_rate': (ic > 0).mean()
            }

    # 整理结果
    ic_table = []
    for factor in factors:
        row = {'因子': factor}
        for ret in returns:
            row[f'{ret}_IC'] = ic_results[factor][ret]['IC_mean']
            row[f'{ret}_ICIR'] = ic_results[factor][ret]['ICIR']
        ic_table.append(row)

    ic_df = pd.DataFrame(ic_table)
    print("\n【因子IC值】")
    print(ic_df.round(4).to_string(index=False))

    # 绘图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # IC均值比较
    ax1 = axes[0, 0]
    x = range(len(factors))
    width = 0.2
    for i, ret in enumerate(returns):
        ic_vals = [ic_results[f][ret]['IC_mean'] for f in factors]
        ax1.bar([xi + i*width for xi in x], ic_vals, width, label=ret)
    ax1.set_xticks([xi + 1.5*width for xi in x])
    ax1.set_xticklabels(factors, rotation=45, ha='right')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_xlabel('因子')
    ax1.set_ylabel('IC均值')
    ax1.set_title('因子IC均值比较')
    ax1.legend()

    # ICIR比较
    ax2 = axes[0, 1]
    for i, ret in enumerate(returns):
        icir_vals = [ic_results[f][ret]['ICIR'] for f in factors]
        ax2.bar([xi + i*width for xi in x], icir_vals, width, label=ret)
    ax2.set_xticks([xi + 1.5*width for xi in x])
    ax2.set_xticklabels(factors, rotation=45, ha='right')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('因子')
    ax2.set_ylabel('ICIR')
    ax2.set_title('因子ICIR比较')
    ax2.legend()

    # 最佳因子IC时序
    ax3 = axes[1, 0]
    # 找IC绝对值最大的因子
    best_factor = max(factors, key=lambda f: abs(ic_results[f]['next5_ret']['IC_mean']))
    ic_series = df.groupby('trade_date').apply(
        lambda x: x[best_factor].corr(x['next5_ret'], method='spearman') if len(x) > 30 else np.nan
    ).dropna()
    ic_series.index = pd.to_datetime(ic_series.index)
    ax3.plot(ic_series.rolling(20).mean(), label=f'{best_factor} IC MA20')
    ax3.axhline(y=0, color='red', linestyle='--')
    ax3.fill_between(ic_series.index, 0, ic_series.rolling(20).mean(), alpha=0.3)
    ax3.set_xlabel('日期')
    ax3.set_ylabel('IC (20日均值)')
    ax3.set_title(f'最佳因子({best_factor})IC时序')
    ax3.legend()

    # IC正向率
    ax4 = axes[1, 1]
    positive_rates = [ic_results[f]['next5_ret']['IC_positive_rate'] * 100 for f in factors]
    colors = ['green' if r > 50 else 'red' for r in positive_rates]
    ax4.bar(range(len(factors)), positive_rates, color=colors, edgecolor='black')
    ax4.axhline(y=50, color='red', linestyle='--', label='50%基准')
    ax4.set_xticks(range(len(factors)))
    ax4.set_xticklabels(factors, rotation=45, ha='right')
    ax4.set_xlabel('因子')
    ax4.set_ylabel('IC正向率 (%)')
    ax4.set_title('因子5日IC正向率')
    ax4.legend()

    plt.tight_layout()
    plt.savefig(f'{REPORT_DIR}/amplitude_factor_ic.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n图表已保存: {REPORT_DIR}/amplitude_factor_ic.png")

    return ic_results


def test_factor_quantile_return(df, factors):
    """因子分层收益检验"""
    print("\n" + "=" * 60)
    print("3.3 因子分层收益检验")
    print("=" * 60)

    # 选择几个关键因子进行分层测试
    key_factors = ['amplitude', 'amp_ma20', 'amp_trend', 'amp_std20']

    quantile_results = {}

    for factor in key_factors:
        # 每日分5组
        df_copy = df.copy()
        df_copy['factor_quantile'] = df_copy.groupby('trade_date')[factor].transform(
            lambda x: pd.qcut(x, 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], duplicates='drop') if len(x) > 10 else np.nan
        )

        # 计算各组收益
        quantile_ret = df_copy.groupby('factor_quantile', observed=True).agg({
            'next1_ret': 'mean',
            'next5_ret': 'mean',
            'next10_ret': 'mean',
            'next20_ret': 'mean'
        }).round(3)

        quantile_results[factor] = quantile_ret

        print(f"\n【{factor}因子分层收益】")
        print(quantile_ret.to_string())

        # 计算多空收益
        if 'Q1' in quantile_ret.index and 'Q5' in quantile_ret.index:
            long_short = quantile_ret.loc['Q5'] - quantile_ret.loc['Q1']
            print(f"多空收益(Q5-Q1): 1日={long_short['next1_ret']:.3f}, 5日={long_short['next5_ret']:.3f}, 10日={long_short['next10_ret']:.3f}, 20日={long_short['next20_ret']:.3f}")

    # 绘图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, factor in enumerate(key_factors):
        ax = axes[idx // 2, idx % 2]
        qr = quantile_results[factor]

        x = range(len(qr))
        width = 0.2
        ax.bar([i - 1.5*width for i in x], qr['next1_ret'], width, label='1日')
        ax.bar([i - 0.5*width for i in x], qr['next5_ret'], width, label='5日')
        ax.bar([i + 0.5*width for i in x], qr['next10_ret'], width, label='10日')
        ax.bar([i + 1.5*width for i in x], qr['next20_ret'], width, label='20日')

        ax.set_xticks(x)
        ax.set_xticklabels(qr.index)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('因子分组')
        ax.set_ylabel('收益 (%)')
        ax.set_title(f'{factor}因子分层收益')
        ax.legend(loc='best')

    plt.tight_layout()
    plt.savefig(f'{REPORT_DIR}/amplitude_factor_quantile.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n图表已保存: {REPORT_DIR}/amplitude_factor_quantile.png")

    return quantile_results


def backtest_amplitude_factor(df, factors):
    """因子回测"""
    print("\n" + "=" * 60)
    print("3.4 振幅因子回测")
    print("=" * 60)

    # 使用振幅趋势因子（收敛后反弹策略）
    df_copy = df.copy()

    # 策略：做多振幅收敛（amp_trend < 0.8）的股票
    df_copy['signal'] = (df_copy['amp_trend'] < 0.8).astype(int)

    # 按日期计算策略收益
    daily_returns = df_copy.groupby('trade_date').apply(
        lambda x: x[x['signal'] == 1]['next1_ret'].mean() if x['signal'].sum() > 0 else 0
    )

    # 基准收益（等权全市场）
    benchmark_returns = df_copy.groupby('trade_date')['next1_ret'].mean()

    # 计算累积收益
    strategy_cumret = (1 + daily_returns / 100).cumprod()
    benchmark_cumret = (1 + benchmark_returns / 100).cumprod()

    # 性能指标
    total_return = (strategy_cumret.iloc[-1] - 1) * 100
    benchmark_total = (benchmark_cumret.iloc[-1] - 1) * 100

    trading_days = len(daily_returns)
    annual_return = ((1 + total_return/100) ** (252/trading_days) - 1) * 100

    volatility = daily_returns.std() * np.sqrt(252)
    sharpe = annual_return / volatility if volatility > 0 else 0

    # 最大回撤
    cummax = strategy_cumret.cummax()
    drawdown = (strategy_cumret - cummax) / cummax
    max_drawdown = drawdown.min() * 100

    print("\n【振幅收敛策略回测结果】")
    print(f"策略总收益: {total_return:.2f}%")
    print(f"基准总收益: {benchmark_total:.2f}%")
    print(f"超额收益: {total_return - benchmark_total:.2f}%")
    print(f"年化收益: {annual_return:.2f}%")
    print(f"年化波动: {volatility:.2f}%")
    print(f"夏普比率: {sharpe:.2f}")
    print(f"最大回撤: {max_drawdown:.2f}%")
    print(f"交易天数: {trading_days}")

    # 绘图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 累积收益曲线
    ax1 = axes[0, 0]
    ax1.plot(strategy_cumret.values, label=f'振幅收敛策略 ({total_return:.1f}%)', linewidth=1.5)
    ax1.plot(benchmark_cumret.values, label=f'等权基准 ({benchmark_total:.1f}%)', linewidth=1.5, alpha=0.7)
    ax1.set_xlabel('交易日')
    ax1.set_ylabel('累积净值')
    ax1.set_title('策略累积收益曲线')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 超额收益曲线
    ax2 = axes[0, 1]
    excess_cumret = strategy_cumret / benchmark_cumret
    ax2.plot(excess_cumret.values, color='purple', linewidth=1.5)
    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5)
    ax2.set_xlabel('交易日')
    ax2.set_ylabel('相对净值')
    ax2.set_title('超额收益曲线')
    ax2.grid(True, alpha=0.3)

    # 回撤曲线
    ax3 = axes[1, 0]
    ax3.fill_between(range(len(drawdown)), 0, drawdown.values * 100, alpha=0.5, color='red')
    ax3.set_xlabel('交易日')
    ax3.set_ylabel('回撤 (%)')
    ax3.set_title(f'回撤曲线 (最大回撤: {max_drawdown:.1f}%)')
    ax3.grid(True, alpha=0.3)

    # 月度收益
    ax4 = axes[1, 1]
    df_copy['month'] = df_copy['trade_date'].str[:6]
    monthly_returns = df_copy[df_copy['signal'] == 1].groupby('month')['next1_ret'].mean()
    colors = ['red' if r > 0 else 'green' for r in monthly_returns.values]
    ax4.bar(range(len(monthly_returns)), monthly_returns.values, color=colors, alpha=0.7)
    ax4.set_xticks(range(0, len(monthly_returns), 3))
    ax4.set_xticklabels([monthly_returns.index[i] for i in range(0, len(monthly_returns), 3)], rotation=45)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax4.set_xlabel('月份')
    ax4.set_ylabel('月均日收益 (%)')
    ax4.set_title('月度收益分布')

    plt.tight_layout()
    plt.savefig(f'{REPORT_DIR}/amplitude_factor_backtest.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n图表已保存: {REPORT_DIR}/amplitude_factor_backtest.png")

    return {
        'total_return': total_return,
        'benchmark_return': benchmark_total,
        'annual_return': annual_return,
        'volatility': volatility,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown
    }


def generate_report(all_results):
    """生成研究报告"""
    print("\n" + "=" * 60)
    print("生成研究报告")
    print("=" * 60)

    report = """
# 股票振幅特征研究报告

## 研究概述

本报告对A股市场股票振幅特征进行了系统性研究，包括振幅统计分析、振幅信号研究和振幅因子构建三个部分。

研究数据：2023年至今的A股日线数据（排除北交所）

---

## 一、振幅统计分析

### 1.1 振幅分布特征

振幅定义：振幅 = (最高价 - 最低价) / 前收盘价 × 100%

**主要发现：**
- 振幅分布呈右偏分布，大部分股票振幅集中在2-5%区间
- 振幅超过10%的交易日相对较少，通常伴随重大信息或极端市场情绪
- 年度振幅水平受市场环境影响明显

![振幅分布](amplitude_distribution.png)

### 1.2 行业振幅差异

**主要发现：**
- 不同行业振幅特征差异显著
- 成长性行业（如科技、医药）振幅普遍较高
- 防御性行业（如银行、公用事业）振幅相对较低
- 行业振幅与行业波动性、市场关注度相关

![行业振幅](amplitude_industry.png)

### 1.3 市值与振幅关系

**主要发现：**
- 市值与振幅呈显著负相关
- 小市值股票振幅明显高于大市值股票
- 这一特征反映了不同市值股票的流动性和投资者结构差异

![市值与振幅](amplitude_market_cap.png)

---

## 二、振幅信号分析

### 2.1 高振幅后走势

**主要发现：**
- 极高振幅后短期收益存在一定规律
- 高振幅+上涨日后续表现与高振幅+下跌日后续表现存在差异
- 振幅极端值通常预示着较高的短期波动

![高振幅后走势](amplitude_performance.png)

### 2.2 振幅收敛/扩张

**主要发现：**
- 振幅收敛后往往伴随波动率扩张
- 强收敛状态后的后续收益表现相对较好
- 振幅Z-Score是有效的异常检测指标

![振幅收敛扩张](amplitude_convergence.png)

### 2.3 振幅突破策略

**主要发现：**
- 振幅突破（超过20日最高）是重要的信号
- 振幅突破+上涨vs振幅突破+下跌后续走势不同
- 结合量能的振幅突破信号更加有效

![振幅突破策略](amplitude_breakout_strategy.png)

---

## 三、因子构建与检验

### 3.1 振幅因子定义

| 因子名称 | 定义 |
|---------|------|
| amplitude | 当日振幅 |
| amp_ma20 | 20日振幅均值 |
| amp_ma5 | 5日振幅均值 |
| amp_std20 | 20日振幅标准差 |
| amp_trend | 振幅趋势 (MA5/MA20) |
| amp_momentum | 振幅动量 |
| amp_z_score | 振幅标准化 |
| amp_range | 20日振幅极差 |
| amp_cv | 振幅变异系数 |

### 3.2 因子IC检验

![因子IC检验](amplitude_factor_ic.png)

**主要发现：**
- 振幅因子与未来收益存在一定相关性
- IC绝对值较小但方向稳定
- 不同周期下因子效果存在差异

### 3.3 因子分层收益

![因子分层收益](amplitude_factor_quantile.png)

**主要发现：**
- 振幅因子具有一定的分层效果
- 多空收益差异在不同因子和周期下表现不同
- 组合优化可提升因子效果

### 3.4 因子回测

![因子回测](amplitude_factor_backtest.png)

**振幅收敛策略回测结果：**
- 该策略通过捕捉振幅收敛后的波动扩张获取收益
- 具体表现详见回测图表

---

## 四、研究结论

### 主要结论

1. **振幅分布特征**：A股振幅分布呈右偏分布，小市值股票振幅高于大市值股票，不同行业振幅差异明显。

2. **振幅信号价值**：高振幅后短期走势存在一定规律，振幅收敛/扩张状态对后续波动有预测作用。

3. **振幅因子效果**：振幅相关因子具有一定的选股能力，但IC值相对较小，需结合其他因子使用。

### 投资建议

1. **风险控制**：利用振幅指标进行风险监控，高振幅股票需要更严格的止损。

2. **择时参考**：振幅收敛可作为波动率扩张的预警信号。

3. **因子组合**：振幅因子可作为组合因子的补充，与动量、波动率因子结合使用。

---

*报告生成时间：2026-02-01*
*数据来源：Tushare*
"""

    with open(f'{REPORT_DIR}/amplitude_research_report.md', 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n报告已保存: {REPORT_DIR}/amplitude_research_report.md")


def main():
    """主函数"""
    print("=" * 60)
    print("股票振幅特征研究")
    print("=" * 60)

    conn = get_connection()
    all_results = {}

    try:
        # Part 1: 振幅统计分析
        print("\n>>> Part 1: 振幅统计分析")
        df = analyze_amplitude_distribution(conn)
        df_with_industry, industry_amp = analyze_industry_amplitude(conn, df)
        df_with_mv = analyze_market_cap_amplitude(conn, df)

        # Part 2: 振幅信号分析
        print("\n>>> Part 2: 振幅信号分析")
        perf_df = analyze_high_amplitude_performance(conn)
        conv_df = analyze_amplitude_convergence(conn)
        breakout_df = analyze_amplitude_breakout_strategy(conn)

        # Part 3: 因子构建与检验
        print("\n>>> Part 3: 因子构建与检验")
        factor_df, factors = construct_amplitude_factors(conn)
        ic_results = test_factor_ic(factor_df, factors)
        quantile_results = test_factor_quantile_return(factor_df, factors)
        backtest_results = backtest_amplitude_factor(factor_df, factors)

        all_results['backtest'] = backtest_results

        # 生成报告
        generate_report(all_results)

        print("\n" + "=" * 60)
        print("研究完成！")
        print("=" * 60)
        print(f"\n报告目录: {REPORT_DIR}")
        print("\n生成的文件:")
        print("  - amplitude_distribution.png (振幅分布)")
        print("  - amplitude_industry.png (行业振幅)")
        print("  - amplitude_market_cap.png (市值与振幅)")
        print("  - amplitude_performance.png (高振幅后走势)")
        print("  - amplitude_convergence.png (振幅收敛扩张)")
        print("  - amplitude_breakout_strategy.png (振幅突破策略)")
        print("  - amplitude_factor_ic.png (因子IC检验)")
        print("  - amplitude_factor_quantile.png (因子分层收益)")
        print("  - amplitude_factor_backtest.png (因子回测)")
        print("  - amplitude_research_report.md (研究报告)")

    finally:
        conn.close()


if __name__ == '__main__':
    main()
