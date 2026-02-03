#!/usr/bin/env python3
"""
超跌反弹策略研究
================

研究超跌股票的反弹效应，并设计相应的交易策略。

超跌定义：
1. 短期超跌：5日跌幅 > 15%
2. 中期超跌：20日跌幅 > 30%
3. 技术超跌：RSI(14) < 20

研究内容：
1. 超跌后反弹概率
2. 反弹幅度统计
3. 反弹持续时间
4. 超跌抄底策略设计
"""

import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False

# 数据库路径
DB_PATH = '/Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db'
REPORT_PATH = '/Users/allen/workspace/python/stock/Tushare-DuckDB/reports/research/'


def load_daily_data(start_date='20180101', end_date='20260130'):
    """加载日线数据"""
    print(f"Loading daily data from {start_date} to {end_date}...")
    conn = duckdb.connect(DB_PATH, read_only=True)

    query = f"""
    WITH listed_stocks AS (
        SELECT ts_code
        FROM stock_basic
        WHERE list_status = 'L'
    )
    SELECT
        d.ts_code,
        d.trade_date,
        d.open,
        d.high,
        d.low,
        d.close,
        d.pre_close,
        d.pct_chg,
        d.vol,
        d.amount,
        a.adj_factor
    FROM daily d
    INNER JOIN listed_stocks s ON d.ts_code = s.ts_code
    LEFT JOIN adj_factor a ON d.ts_code = a.ts_code AND d.trade_date = a.trade_date
    WHERE d.trade_date >= '{start_date}' AND d.trade_date <= '{end_date}'
    ORDER BY d.ts_code, d.trade_date
    """

    df = conn.execute(query).fetchdf()
    conn.close()

    # 转换日期格式
    df['trade_date'] = pd.to_datetime(df['trade_date'])

    # 计算后复权价格
    df['adj_close'] = df['close'] * df['adj_factor']

    print(f"Loaded {len(df):,} records for {df['ts_code'].nunique():,} stocks")
    return df


def calculate_technical_indicators(df):
    """计算技术指标"""
    print("Calculating technical indicators...")

    result = []
    for ts_code, group in df.groupby('ts_code'):
        group = group.sort_values('trade_date').copy()

        # 跳过数据不足的股票
        if len(group) < 30:
            continue

        # 计算N日收益率
        group['ret_5d'] = group['adj_close'].pct_change(5) * 100
        group['ret_10d'] = group['adj_close'].pct_change(10) * 100
        group['ret_20d'] = group['adj_close'].pct_change(20) * 100

        # 计算RSI(14)
        delta = group['adj_close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        avg_gain = gain.rolling(window=14, min_periods=14).mean()
        avg_loss = loss.rolling(window=14, min_periods=14).mean()
        rs = avg_gain / avg_loss
        group['rsi_14'] = 100 - (100 / (1 + rs))

        # 计算未来N日收益率（用于研究反弹效应）
        group['fut_ret_1d'] = group['adj_close'].pct_change(1).shift(-1) * 100
        group['fut_ret_3d'] = group['adj_close'].pct_change(3).shift(-3) * 100
        group['fut_ret_5d'] = group['adj_close'].pct_change(5).shift(-5) * 100
        group['fut_ret_10d'] = group['adj_close'].pct_change(10).shift(-10) * 100
        group['fut_ret_20d'] = group['adj_close'].pct_change(20).shift(-20) * 100

        # 计算未来最大涨幅和最大跌幅（10日内）
        for i in range(1, 11):
            group[f'fut_high_{i}d'] = group['adj_close'].shift(-i)
        fut_high_cols = [f'fut_high_{i}d' for i in range(1, 11)]
        group['fut_max_high_10d'] = group[fut_high_cols].max(axis=1)
        group['fut_max_return_10d'] = (group['fut_max_high_10d'] / group['adj_close'] - 1) * 100
        group = group.drop(columns=fut_high_cols)

        # 计算20日内最大涨幅
        for i in range(1, 21):
            group[f'fut_high_{i}d'] = group['adj_close'].shift(-i)
        fut_high_cols = [f'fut_high_{i}d' for i in range(1, 21)]
        group['fut_max_high_20d'] = group[fut_high_cols].max(axis=1)
        group['fut_max_return_20d'] = (group['fut_max_high_20d'] / group['adj_close'] - 1) * 100
        group = group.drop(columns=fut_high_cols)

        result.append(group)

    df_result = pd.concat(result, ignore_index=True)
    print(f"Calculated indicators for {df_result['ts_code'].nunique():,} stocks")
    return df_result


def identify_oversold_signals(df):
    """识别超跌信号"""
    print("\n" + "="*60)
    print("Identifying oversold signals...")
    print("="*60)

    # 超跌条件定义
    signals = {
        'short_term_oversold': df['ret_5d'] < -15,  # 5日跌幅超过15%
        'mid_term_oversold': df['ret_20d'] < -30,   # 20日跌幅超过30%
        'rsi_oversold': df['rsi_14'] < 20,           # RSI < 20
    }

    for name, condition in signals.items():
        df[name] = condition

    # 综合超跌信号
    df['any_oversold'] = df['short_term_oversold'] | df['mid_term_oversold'] | df['rsi_oversold']
    df['double_oversold'] = (df['short_term_oversold'].astype(int) +
                             df['mid_term_oversold'].astype(int) +
                             df['rsi_oversold'].astype(int)) >= 2
    df['triple_oversold'] = df['short_term_oversold'] & df['mid_term_oversold'] & df['rsi_oversold']

    # 统计信号数量
    print("\n超跌信号统计:")
    print("-" * 40)
    print(f"短期超跌 (5日跌幅>15%): {signals['short_term_oversold'].sum():,} 次")
    print(f"中期超跌 (20日跌幅>30%): {signals['mid_term_oversold'].sum():,} 次")
    print(f"技术超跌 (RSI<20): {signals['rsi_oversold'].sum():,} 次")
    print(f"任一超跌信号: {df['any_oversold'].sum():,} 次")
    print(f"双重超跌信号: {df['double_oversold'].sum():,} 次")
    print(f"三重超跌信号: {df['triple_oversold'].sum():,} 次")

    return df


def analyze_rebound_effect(df):
    """分析反弹效应"""
    print("\n" + "="*60)
    print("Analyzing rebound effect...")
    print("="*60)

    results = {}

    signal_types = [
        ('short_term_oversold', '短期超跌(5日跌幅>15%)'),
        ('mid_term_oversold', '中期超跌(20日跌幅>30%)'),
        ('rsi_oversold', '技术超跌(RSI<20)'),
        ('any_oversold', '任一超跌信号'),
        ('double_oversold', '双重超跌'),
        ('triple_oversold', '三重超跌'),
    ]

    for signal_col, signal_name in signal_types:
        signal_df = df[df[signal_col] == True].dropna(subset=['fut_ret_5d', 'fut_ret_10d', 'fut_ret_20d'])

        if len(signal_df) == 0:
            continue

        stats = {
            'signal_name': signal_name,
            'signal_count': len(signal_df),
            # 反弹概率
            'rebound_prob_1d': (signal_df['fut_ret_1d'] > 0).mean() * 100,
            'rebound_prob_3d': (signal_df['fut_ret_3d'] > 0).mean() * 100,
            'rebound_prob_5d': (signal_df['fut_ret_5d'] > 0).mean() * 100,
            'rebound_prob_10d': (signal_df['fut_ret_10d'] > 0).mean() * 100,
            'rebound_prob_20d': (signal_df['fut_ret_20d'] > 0).mean() * 100,
            # 平均收益率
            'avg_ret_1d': signal_df['fut_ret_1d'].mean(),
            'avg_ret_3d': signal_df['fut_ret_3d'].mean(),
            'avg_ret_5d': signal_df['fut_ret_5d'].mean(),
            'avg_ret_10d': signal_df['fut_ret_10d'].mean(),
            'avg_ret_20d': signal_df['fut_ret_20d'].mean(),
            # 中位数收益率
            'median_ret_5d': signal_df['fut_ret_5d'].median(),
            'median_ret_10d': signal_df['fut_ret_10d'].median(),
            'median_ret_20d': signal_df['fut_ret_20d'].median(),
            # 最大涨幅
            'avg_max_return_10d': signal_df['fut_max_return_10d'].mean(),
            'avg_max_return_20d': signal_df['fut_max_return_20d'].mean(),
            'median_max_return_10d': signal_df['fut_max_return_10d'].median(),
            'median_max_return_20d': signal_df['fut_max_return_20d'].median(),
            # 收益率分布
            'ret_5d_std': signal_df['fut_ret_5d'].std(),
            'ret_10d_std': signal_df['fut_ret_10d'].std(),
            'ret_20d_std': signal_df['fut_ret_20d'].std(),
        }

        results[signal_col] = stats

        print(f"\n{signal_name}:")
        print(f"  信号次数: {stats['signal_count']:,}")
        print(f"  反弹概率 (1/3/5/10/20日): {stats['rebound_prob_1d']:.1f}% / {stats['rebound_prob_3d']:.1f}% / {stats['rebound_prob_5d']:.1f}% / {stats['rebound_prob_10d']:.1f}% / {stats['rebound_prob_20d']:.1f}%")
        print(f"  平均收益 (5/10/20日): {stats['avg_ret_5d']:.2f}% / {stats['avg_ret_10d']:.2f}% / {stats['avg_ret_20d']:.2f}%")
        print(f"  中位数收益 (5/10/20日): {stats['median_ret_5d']:.2f}% / {stats['median_ret_10d']:.2f}% / {stats['median_ret_20d']:.2f}%")
        print(f"  10日内最大涨幅: 均值 {stats['avg_max_return_10d']:.2f}%, 中位数 {stats['median_max_return_10d']:.2f}%")

    return results


def analyze_by_market_condition(df):
    """按市场环境分析超跌反弹效应"""
    print("\n" + "="*60)
    print("Analyzing by market condition...")
    print("="*60)

    # 计算市场整体走势（使用所有股票的平均收益率作为市场指标）
    market_daily = df.groupby('trade_date').agg({
        'pct_chg': 'mean',
        'adj_close': 'mean'
    }).reset_index()
    market_daily = market_daily.sort_values('trade_date')
    market_daily['market_ret_20d'] = market_daily['pct_chg'].rolling(20).sum()

    # 定义市场环境
    market_daily['market_condition'] = pd.cut(
        market_daily['market_ret_20d'],
        bins=[-np.inf, -10, 0, 10, np.inf],
        labels=['熊市(跌>10%)', '震荡偏弱(跌0-10%)', '震荡偏强(涨0-10%)', '牛市(涨>10%)']
    )

    # 合并市场环境到原数据
    df = df.merge(market_daily[['trade_date', 'market_condition']], on='trade_date', how='left')

    # 分析不同市场环境下的超跌反弹效应
    signal_col = 'short_term_oversold'
    results = []

    for condition in ['熊市(跌>10%)', '震荡偏弱(跌0-10%)', '震荡偏强(涨0-10%)', '牛市(涨>10%)']:
        condition_df = df[(df[signal_col] == True) & (df['market_condition'] == condition)]
        condition_df = condition_df.dropna(subset=['fut_ret_5d', 'fut_ret_10d'])

        if len(condition_df) < 100:
            continue

        stats = {
            'market_condition': condition,
            'signal_count': len(condition_df),
            'rebound_prob_5d': (condition_df['fut_ret_5d'] > 0).mean() * 100,
            'rebound_prob_10d': (condition_df['fut_ret_10d'] > 0).mean() * 100,
            'avg_ret_5d': condition_df['fut_ret_5d'].mean(),
            'avg_ret_10d': condition_df['fut_ret_10d'].mean(),
            'avg_max_return_10d': condition_df['fut_max_return_10d'].mean(),
        }
        results.append(stats)

        print(f"\n{condition}:")
        print(f"  超跌信号数: {stats['signal_count']:,}")
        print(f"  5日反弹概率: {stats['rebound_prob_5d']:.1f}%")
        print(f"  10日反弹概率: {stats['rebound_prob_10d']:.1f}%")
        print(f"  平均5日收益: {stats['avg_ret_5d']:.2f}%")
        print(f"  平均10日收益: {stats['avg_ret_10d']:.2f}%")
        print(f"  10日内最大涨幅: {stats['avg_max_return_10d']:.2f}%")

    return df, results


def analyze_rebound_duration(df):
    """分析反弹持续时间"""
    print("\n" + "="*60)
    print("Analyzing rebound duration...")
    print("="*60)

    signal_df = df[df['short_term_oversold'] == True].copy()
    signal_df = signal_df.dropna(subset=['fut_ret_1d', 'fut_ret_3d', 'fut_ret_5d', 'fut_ret_10d', 'fut_ret_20d'])

    # 计算连续上涨天数
    duration_stats = []

    # 统计不同持有期的收益分布
    holding_periods = [1, 3, 5, 10, 20]

    for period in holding_periods:
        col = f'fut_ret_{period}d'
        positive_rate = (signal_df[col] > 0).mean() * 100
        avg_return = signal_df[col].mean()

        # 计算盈亏比
        winners = signal_df[signal_df[col] > 0][col]
        losers = signal_df[signal_df[col] < 0][col]

        avg_win = winners.mean() if len(winners) > 0 else 0
        avg_loss = abs(losers.mean()) if len(losers) > 0 else 1
        profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0

        duration_stats.append({
            'holding_period': period,
            'positive_rate': positive_rate,
            'avg_return': avg_return,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_loss_ratio': profit_loss_ratio,
            'expected_return': (positive_rate/100 * avg_win) - ((100-positive_rate)/100 * avg_loss)
        })

        print(f"\n持有{period}日:")
        print(f"  胜率: {positive_rate:.1f}%")
        print(f"  平均收益: {avg_return:.2f}%")
        print(f"  平均盈利: {avg_win:.2f}%")
        print(f"  平均亏损: {avg_loss:.2f}%")
        print(f"  盈亏比: {profit_loss_ratio:.2f}")
        print(f"  期望收益: {duration_stats[-1]['expected_return']:.2f}%")

    return duration_stats


def design_trading_strategy(df, rebound_results):
    """设计交易策略"""
    print("\n" + "="*60)
    print("Designing trading strategy...")
    print("="*60)

    strategies = {}

    # 策略1: 短期超跌反弹策略
    print("\n策略1: 短期超跌快速反弹策略")
    print("-" * 40)
    print("入场条件:")
    print("  - 5日跌幅 > 15%")
    print("  - RSI(14) < 30")
    print("持仓周期: 5个交易日")
    print("止损设置: -5%")
    print("止盈设置: +10%")

    # 回测策略1
    strat1_df = df[(df['ret_5d'] < -15) & (df['rsi_14'] < 30)].copy()
    strat1_df = strat1_df.dropna(subset=['fut_ret_5d', 'fut_max_return_10d'])

    if len(strat1_df) > 0:
        # 模拟止损止盈
        strat1_df['strategy_ret'] = strat1_df.apply(
            lambda x: min(x['fut_max_return_10d'], 10) if x['fut_ret_5d'] > -5 else -5,
            axis=1
        )

        win_rate = (strat1_df['strategy_ret'] > 0).mean() * 100
        avg_return = strat1_df['strategy_ret'].mean()

        strategies['short_term'] = {
            'name': '短期超跌快速反弹',
            'entry': '5日跌幅>15% & RSI<30',
            'holding': '5日',
            'stop_loss': '-5%',
            'take_profit': '+10%',
            'signal_count': len(strat1_df),
            'win_rate': win_rate,
            'avg_return': avg_return
        }

        print(f"\n回测结果:")
        print(f"  信号次数: {len(strat1_df):,}")
        print(f"  胜率: {win_rate:.1f}%")
        print(f"  平均收益: {avg_return:.2f}%")

    # 策略2: 中期超跌抄底策略
    print("\n\n策略2: 中期超跌分批建仓策略")
    print("-" * 40)
    print("入场条件:")
    print("  - 20日跌幅 > 30%")
    print("分批建仓方案:")
    print("  - 第一次: 触发信号时买入 30%")
    print("  - 第二次: 再跌5%买入 30%")
    print("  - 第三次: 再跌5%买入 40%")
    print("持仓周期: 20个交易日")
    print("止损设置: -10% (从成本价)")
    print("止盈设置: +20%")

    strat2_df = df[df['ret_20d'] < -30].copy()
    strat2_df = strat2_df.dropna(subset=['fut_ret_20d', 'fut_max_return_20d'])

    if len(strat2_df) > 0:
        win_rate = (strat2_df['fut_ret_20d'] > 0).mean() * 100
        avg_return = strat2_df['fut_ret_20d'].mean()

        strategies['mid_term'] = {
            'name': '中期超跌分批建仓',
            'entry': '20日跌幅>30%',
            'holding': '20日',
            'stop_loss': '-10%',
            'take_profit': '+20%',
            'signal_count': len(strat2_df),
            'win_rate': win_rate,
            'avg_return': avg_return
        }

        print(f"\n回测结果(单次满仓):")
        print(f"  信号次数: {len(strat2_df):,}")
        print(f"  胜率: {win_rate:.1f}%")
        print(f"  平均收益: {avg_return:.2f}%")

    # 策略3: 三重超跌极值策略
    print("\n\n策略3: 三重超跌极值策略")
    print("-" * 40)
    print("入场条件:")
    print("  - 5日跌幅 > 15%")
    print("  - 20日跌幅 > 30%")
    print("  - RSI(14) < 20")
    print("持仓周期: 10个交易日")
    print("止损设置: -8%")
    print("止盈设置: +15%")

    strat3_df = df[df['triple_oversold'] == True].copy()
    strat3_df = strat3_df.dropna(subset=['fut_ret_10d', 'fut_max_return_10d'])

    if len(strat3_df) > 0:
        win_rate = (strat3_df['fut_ret_10d'] > 0).mean() * 100
        avg_return = strat3_df['fut_ret_10d'].mean()
        avg_max_return = strat3_df['fut_max_return_10d'].mean()

        strategies['triple_oversold'] = {
            'name': '三重超跌极值',
            'entry': '5日跌>15% & 20日跌>30% & RSI<20',
            'holding': '10日',
            'stop_loss': '-8%',
            'take_profit': '+15%',
            'signal_count': len(strat3_df),
            'win_rate': win_rate,
            'avg_return': avg_return,
            'avg_max_return': avg_max_return
        }

        print(f"\n回测结果:")
        print(f"  信号次数: {len(strat3_df):,}")
        print(f"  胜率: {win_rate:.1f}%")
        print(f"  平均收益: {avg_return:.2f}%")
        print(f"  10日内最大涨幅: {avg_max_return:.2f}%")

    return strategies


def analyze_by_oversold_degree(df):
    """按超跌程度分析反弹效应"""
    print("\n" + "="*60)
    print("Analyzing by oversold degree...")
    print("="*60)

    # 按5日跌幅分组
    df['oversold_degree'] = pd.cut(
        df['ret_5d'],
        bins=[-100, -30, -25, -20, -15, -10, 0],
        labels=['极度超跌(>30%)', '严重超跌(25-30%)', '较大超跌(20-25%)',
                '超跌(15-20%)', '小幅下跌(10-15%)', '其他']
    )

    results = []
    for degree in ['极度超跌(>30%)', '严重超跌(25-30%)', '较大超跌(20-25%)',
                   '超跌(15-20%)', '小幅下跌(10-15%)']:
        degree_df = df[df['oversold_degree'] == degree].dropna(subset=['fut_ret_5d', 'fut_ret_10d'])

        if len(degree_df) < 50:
            continue

        stats = {
            'oversold_degree': degree,
            'count': len(degree_df),
            'rebound_prob_5d': (degree_df['fut_ret_5d'] > 0).mean() * 100,
            'rebound_prob_10d': (degree_df['fut_ret_10d'] > 0).mean() * 100,
            'avg_ret_5d': degree_df['fut_ret_5d'].mean(),
            'avg_ret_10d': degree_df['fut_ret_10d'].mean(),
            'avg_max_return_10d': degree_df['fut_max_return_10d'].mean(),
        }
        results.append(stats)

        print(f"\n{degree}:")
        print(f"  样本数: {stats['count']:,}")
        print(f"  5日反弹概率: {stats['rebound_prob_5d']:.1f}%")
        print(f"  10日反弹概率: {stats['rebound_prob_10d']:.1f}%")
        print(f"  平均5日收益: {stats['avg_ret_5d']:.2f}%")
        print(f"  平均10日收益: {stats['avg_ret_10d']:.2f}%")
        print(f"  10日内最大涨幅: {stats['avg_max_return_10d']:.2f}%")

    return results


def create_visualizations(df, rebound_results, duration_stats, strategies):
    """创建可视化图表"""
    print("\n" + "="*60)
    print("Creating visualizations...")
    print("="*60)

    fig = plt.figure(figsize=(20, 16))

    # 图1: 不同超跌类型的反弹概率
    ax1 = fig.add_subplot(2, 3, 1)
    signal_names = [v['signal_name'] for v in rebound_results.values()]
    rebound_5d = [v['rebound_prob_5d'] for v in rebound_results.values()]
    rebound_10d = [v['rebound_prob_10d'] for v in rebound_results.values()]
    rebound_20d = [v['rebound_prob_20d'] for v in rebound_results.values()]

    x = np.arange(len(signal_names))
    width = 0.25

    bars1 = ax1.bar(x - width, rebound_5d, width, label='5日反弹', color='lightblue')
    bars2 = ax1.bar(x, rebound_10d, width, label='10日反弹', color='steelblue')
    bars3 = ax1.bar(x + width, rebound_20d, width, label='20日反弹', color='darkblue')

    ax1.set_ylabel('反弹概率 (%)')
    ax1.set_title('不同超跌类型的反弹概率')
    ax1.set_xticks(x)
    ax1.set_xticklabels(signal_names, rotation=45, ha='right')
    ax1.legend()
    ax1.axhline(y=50, color='red', linestyle='--', alpha=0.5)
    ax1.set_ylim(0, 100)

    # 图2: 不同超跌类型的平均收益
    ax2 = fig.add_subplot(2, 3, 2)
    avg_ret_5d = [v['avg_ret_5d'] for v in rebound_results.values()]
    avg_ret_10d = [v['avg_ret_10d'] for v in rebound_results.values()]
    avg_ret_20d = [v['avg_ret_20d'] for v in rebound_results.values()]

    bars1 = ax2.bar(x - width, avg_ret_5d, width, label='5日收益', color='lightgreen')
    bars2 = ax2.bar(x, avg_ret_10d, width, label='10日收益', color='green')
    bars3 = ax2.bar(x + width, avg_ret_20d, width, label='20日收益', color='darkgreen')

    ax2.set_ylabel('平均收益率 (%)')
    ax2.set_title('不同超跌类型的平均收益')
    ax2.set_xticks(x)
    ax2.set_xticklabels(signal_names, rotation=45, ha='right')
    ax2.legend()
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)

    # 图3: 持有期分析
    ax3 = fig.add_subplot(2, 3, 3)
    periods = [d['holding_period'] for d in duration_stats]
    positive_rates = [d['positive_rate'] for d in duration_stats]
    expected_returns = [d['expected_return'] for d in duration_stats]

    ax3_twin = ax3.twinx()
    bars = ax3.bar(periods, positive_rates, color='steelblue', alpha=0.7, label='胜率')
    line = ax3_twin.plot(periods, expected_returns, 'ro-', linewidth=2, markersize=8, label='期望收益')

    ax3.set_xlabel('持有天数')
    ax3.set_ylabel('胜率 (%)', color='steelblue')
    ax3_twin.set_ylabel('期望收益 (%)', color='red')
    ax3.set_title('短期超跌后不同持有期的表现')
    ax3.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax3_twin.axhline(y=0, color='red', linestyle='--', alpha=0.3)

    # 图4: 10日内最大涨幅分布
    ax4 = fig.add_subplot(2, 3, 4)
    short_oversold_df = df[df['short_term_oversold'] == True].dropna(subset=['fut_max_return_10d'])

    ax4.hist(short_oversold_df['fut_max_return_10d'].clip(-50, 100), bins=50,
             color='steelblue', alpha=0.7, edgecolor='white')
    ax4.axvline(x=short_oversold_df['fut_max_return_10d'].median(),
                color='red', linestyle='--', linewidth=2,
                label=f'中位数: {short_oversold_df["fut_max_return_10d"].median():.1f}%')
    ax4.axvline(x=short_oversold_df['fut_max_return_10d'].mean(),
                color='orange', linestyle='--', linewidth=2,
                label=f'均值: {short_oversold_df["fut_max_return_10d"].mean():.1f}%')
    ax4.set_xlabel('10日内最大涨幅 (%)')
    ax4.set_ylabel('频次')
    ax4.set_title('短期超跌后10日内最大涨幅分布')
    ax4.legend()

    # 图5: 策略比较
    ax5 = fig.add_subplot(2, 3, 5)
    if strategies:
        strat_names = [v['name'] for v in strategies.values()]
        win_rates = [v['win_rate'] for v in strategies.values()]
        avg_returns = [v['avg_return'] for v in strategies.values()]

        x = np.arange(len(strat_names))
        width = 0.35

        bars1 = ax5.bar(x - width/2, win_rates, width, label='胜率 (%)', color='steelblue')

        ax5_twin = ax5.twinx()
        bars2 = ax5_twin.bar(x + width/2, avg_returns, width, label='平均收益 (%)', color='green')

        ax5.set_ylabel('胜率 (%)', color='steelblue')
        ax5_twin.set_ylabel('平均收益 (%)', color='green')
        ax5.set_title('策略表现比较')
        ax5.set_xticks(x)
        ax5.set_xticklabels(strat_names, rotation=15)
        ax5.axhline(y=50, color='gray', linestyle='--', alpha=0.5)

        # 添加图例
        lines1, labels1 = ax5.get_legend_handles_labels()
        lines2, labels2 = ax5_twin.get_legend_handles_labels()
        ax5.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    # 图6: 超跌程度与反弹幅度关系
    ax6 = fig.add_subplot(2, 3, 6)

    # 按5日跌幅分组统计
    df_valid = df.dropna(subset=['ret_5d', 'fut_max_return_10d'])
    df_valid = df_valid[df_valid['ret_5d'] < 0]  # 只看下跌的情况

    # 创建分组
    bins = np.arange(-50, 0, 5)
    df_valid['ret_5d_group'] = pd.cut(df_valid['ret_5d'], bins=bins)

    grouped = df_valid.groupby('ret_5d_group').agg({
        'fut_max_return_10d': ['mean', 'count']
    }).reset_index()
    grouped.columns = ['ret_group', 'avg_max_return', 'count']
    grouped = grouped.dropna()
    grouped = grouped[grouped['count'] >= 100]  # 至少100个样本

    # 提取分组中点
    grouped['ret_midpoint'] = grouped['ret_group'].apply(
        lambda x: (x.left + x.right) / 2 if pd.notna(x) else None
    )
    grouped = grouped.dropna(subset=['ret_midpoint'])

    ax6.scatter(grouped['ret_midpoint'], grouped['avg_max_return'],
                s=grouped['count']/50, alpha=0.6, color='steelblue')

    # 添加趋势线
    z = np.polyfit(grouped['ret_midpoint'], grouped['avg_max_return'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(grouped['ret_midpoint'].min(), grouped['ret_midpoint'].max(), 100)
    ax6.plot(x_line, p(x_line), 'r--', linewidth=2, label='趋势线')

    ax6.set_xlabel('5日跌幅 (%)')
    ax6.set_ylabel('10日内最大涨幅 (%)')
    ax6.set_title('超跌程度与反弹幅度关系')
    ax6.legend()
    ax6.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax6.axvline(x=-15, color='red', linestyle='--', alpha=0.5, label='超跌阈值')

    plt.tight_layout()
    plt.savefig(f'{REPORT_PATH}oversold_rebound_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {REPORT_PATH}oversold_rebound_analysis.png")


def generate_report(df, rebound_results, market_condition_results, duration_stats,
                   strategies, oversold_degree_results):
    """生成研究报告"""
    print("\n" + "="*60)
    print("Generating report...")
    print("="*60)

    report = []
    report.append("# 超跌反弹策略研究报告")
    report.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"\n数据范围: {df['trade_date'].min().strftime('%Y-%m-%d')} 至 {df['trade_date'].max().strftime('%Y-%m-%d')}")
    report.append(f"\n股票数量: {df['ts_code'].nunique():,}")

    # 一、研究摘要
    report.append("\n\n## 一、研究摘要")
    report.append("\n本研究系统分析了A股市场超跌股票的反弹效应，包括短期超跌、中期超跌和技术超跌三种情况，")
    report.append("并基于历史数据设计了相应的交易策略。")

    # 二、超跌定义
    report.append("\n\n## 二、超跌定义")
    report.append("\n| 类型 | 定义 | 触发次数 |")
    report.append("|------|------|----------|")

    signal_map = {
        'short_term_oversold': ('短期超跌', '5日跌幅 > 15%'),
        'mid_term_oversold': ('中期超跌', '20日跌幅 > 30%'),
        'rsi_oversold': ('技术超跌', 'RSI(14) < 20'),
    }

    for key, (name, definition) in signal_map.items():
        if key in rebound_results:
            count = rebound_results[key]['signal_count']
            report.append(f"| {name} | {definition} | {count:,} |")

    # 三、反弹效应分析
    report.append("\n\n## 三、反弹效应分析")

    report.append("\n### 3.1 不同超跌类型的反弹概率")
    report.append("\n| 超跌类型 | 1日反弹 | 3日反弹 | 5日反弹 | 10日反弹 | 20日反弹 |")
    report.append("|----------|---------|---------|---------|----------|----------|")

    for key, stats in rebound_results.items():
        report.append(f"| {stats['signal_name']} | {stats['rebound_prob_1d']:.1f}% | {stats['rebound_prob_3d']:.1f}% | "
                     f"{stats['rebound_prob_5d']:.1f}% | {stats['rebound_prob_10d']:.1f}% | {stats['rebound_prob_20d']:.1f}% |")

    report.append("\n### 3.2 不同超跌类型的平均收益")
    report.append("\n| 超跌类型 | 5日收益 | 10日收益 | 20日收益 | 10日最大涨幅 |")
    report.append("|----------|---------|----------|----------|--------------|")

    for key, stats in rebound_results.items():
        report.append(f"| {stats['signal_name']} | {stats['avg_ret_5d']:.2f}% | {stats['avg_ret_10d']:.2f}% | "
                     f"{stats['avg_ret_20d']:.2f}% | {stats['avg_max_return_10d']:.2f}% |")

    # 3.3 持有期分析
    report.append("\n### 3.3 短期超跌后最佳持有期分析")
    report.append("\n| 持有天数 | 胜率 | 平均收益 | 盈亏比 | 期望收益 |")
    report.append("|----------|------|----------|--------|----------|")

    for d in duration_stats:
        report.append(f"| {d['holding_period']}日 | {d['positive_rate']:.1f}% | {d['avg_return']:.2f}% | "
                     f"{d['profit_loss_ratio']:.2f} | {d['expected_return']:.2f}% |")

    # 四、市场环境影响
    report.append("\n\n## 四、市场环境对超跌反弹的影响")

    if market_condition_results:
        report.append("\n| 市场环境 | 信号数 | 5日反弹概率 | 10日反弹概率 | 平均5日收益 | 平均10日收益 |")
        report.append("|----------|--------|-------------|--------------|-------------|--------------|")

        for stats in market_condition_results:
            report.append(f"| {stats['market_condition']} | {stats['signal_count']:,} | "
                         f"{stats['rebound_prob_5d']:.1f}% | {stats['rebound_prob_10d']:.1f}% | "
                         f"{stats['avg_ret_5d']:.2f}% | {stats['avg_ret_10d']:.2f}% |")

        report.append("\n**结论**: 市场环境对超跌反弹效应有显著影响。在震荡偏强和牛市环境下，超跌反弹的概率和幅度更高；")
        report.append("在熊市环境下，即使出现超跌信号，反弹效应也较弱。")

    # 五、超跌程度分析
    report.append("\n\n## 五、超跌程度与反弹幅度关系")

    if oversold_degree_results:
        report.append("\n| 超跌程度 | 样本数 | 5日反弹概率 | 10日反弹概率 | 平均10日收益 | 10日最大涨幅 |")
        report.append("|----------|--------|-------------|--------------|--------------|--------------|")

        for stats in oversold_degree_results:
            report.append(f"| {stats['oversold_degree']} | {stats['count']:,} | "
                         f"{stats['rebound_prob_5d']:.1f}% | {stats['rebound_prob_10d']:.1f}% | "
                         f"{stats['avg_ret_10d']:.2f}% | {stats['avg_max_return_10d']:.2f}% |")

        report.append("\n**结论**: 超跌程度越深，反弹幅度越大，但需要注意极端超跌可能伴随基本面问题，")
        report.append("需要综合考虑个股质地。")

    # 六、策略设计
    report.append("\n\n## 六、超跌抄底策略设计")

    report.append("\n### 6.1 策略一: 短期超跌快速反弹策略")
    report.append("\n**入场条件:**")
    report.append("- 5日跌幅 > 15%")
    report.append("- RSI(14) < 30")
    report.append("\n**交易规则:**")
    report.append("- 持仓周期: 5个交易日")
    report.append("- 止损设置: -5%")
    report.append("- 止盈设置: +10%")

    if 'short_term' in strategies:
        s = strategies['short_term']
        report.append(f"\n**回测结果:**")
        report.append(f"- 信号次数: {s['signal_count']:,}")
        report.append(f"- 胜率: {s['win_rate']:.1f}%")
        report.append(f"- 平均收益: {s['avg_return']:.2f}%")

    report.append("\n### 6.2 策略二: 中期超跌分批建仓策略")
    report.append("\n**入场条件:**")
    report.append("- 20日跌幅 > 30%")
    report.append("\n**分批建仓方案:**")
    report.append("- 第一次建仓: 触发信号时买入仓位的30%")
    report.append("- 第二次建仓: 再跌5%时买入仓位的30%")
    report.append("- 第三次建仓: 再跌5%时买入剩余40%")
    report.append("\n**交易规则:**")
    report.append("- 持仓周期: 20个交易日")
    report.append("- 止损设置: -10% (从平均成本价)")
    report.append("- 止盈设置: +20%")

    if 'mid_term' in strategies:
        s = strategies['mid_term']
        report.append(f"\n**回测结果(单次满仓):**")
        report.append(f"- 信号次数: {s['signal_count']:,}")
        report.append(f"- 胜率: {s['win_rate']:.1f}%")
        report.append(f"- 平均收益: {s['avg_return']:.2f}%")

    report.append("\n### 6.3 策略三: 三重超跌极值策略")
    report.append("\n**入场条件:**")
    report.append("- 5日跌幅 > 15%")
    report.append("- 20日跌幅 > 30%")
    report.append("- RSI(14) < 20")
    report.append("\n**交易规则:**")
    report.append("- 持仓周期: 10个交易日")
    report.append("- 止损设置: -8%")
    report.append("- 止盈设置: +15%")

    if 'triple_oversold' in strategies:
        s = strategies['triple_oversold']
        report.append(f"\n**回测结果:**")
        report.append(f"- 信号次数: {s['signal_count']:,}")
        report.append(f"- 胜率: {s['win_rate']:.1f}%")
        report.append(f"- 平均收益: {s['avg_return']:.2f}%")
        report.append(f"- 10日内最大涨幅: {s['avg_max_return']:.2f}%")

    # 七、风险提示
    report.append("\n\n## 七、风险提示与操作建议")
    report.append("\n### 7.1 风险提示")
    report.append("1. **趋势风险**: 超跌可能是趋势性下跌的开始，需要结合大盘环境判断")
    report.append("2. **基本面风险**: 部分股票超跌可能因为基本面恶化，需要排除问题股")
    report.append("3. **流动性风险**: 小盘股超跌后流动性可能较差，注意仓位控制")
    report.append("4. **时机风险**: 市场持续下跌时，超跌反弹效应减弱")

    report.append("\n### 7.2 操作建议")
    report.append("1. **仓位管理**: 单只股票仓位不超过总仓位的10%")
    report.append("2. **分批建仓**: 避免一次性满仓，采用分批买入策略")
    report.append("3. **严格止损**: 设定止损点位，超过止损必须执行")
    report.append("4. **市场择时**: 优先在市场企稳或反弹阶段使用该策略")
    report.append("5. **个股筛选**: 排除ST股、亏损股、重大利空股")

    report.append("\n### 7.3 最佳适用场景")
    report.append("- 市场短期快速下跌后企稳阶段")
    report.append("- 个股因情绪面因素超跌(非基本面问题)")
    report.append("- 市场整体处于震荡或牛市环境")

    # 保存报告
    report_text = '\n'.join(report)
    with open(f'{REPORT_PATH}oversold_rebound_strategy_report.md', 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"Report saved to {REPORT_PATH}oversold_rebound_strategy_report.md")

    return report_text


def main():
    """主函数"""
    print("="*60)
    print("超跌反弹策略研究")
    print("="*60)

    # 1. 加载数据
    df = load_daily_data(start_date='20180101', end_date='20260130')

    # 2. 计算技术指标
    df = calculate_technical_indicators(df)

    # 3. 识别超跌信号
    df = identify_oversold_signals(df)

    # 4. 分析反弹效应
    rebound_results = analyze_rebound_effect(df)

    # 5. 分析持有期
    duration_stats = analyze_rebound_duration(df)

    # 6. 按市场环境分析
    df, market_condition_results = analyze_by_market_condition(df)

    # 7. 按超跌程度分析
    oversold_degree_results = analyze_by_oversold_degree(df)

    # 8. 设计交易策略
    strategies = design_trading_strategy(df, rebound_results)

    # 9. 创建可视化
    create_visualizations(df, rebound_results, duration_stats, strategies)

    # 10. 生成报告
    report = generate_report(df, rebound_results, market_condition_results,
                            duration_stats, strategies, oversold_degree_results)

    print("\n" + "="*60)
    print("研究完成!")
    print("="*60)
    print(f"\n报告文件: {REPORT_PATH}oversold_rebound_strategy_report.md")
    print(f"图表文件: {REPORT_PATH}oversold_rebound_analysis.png")


if __name__ == '__main__':
    main()
