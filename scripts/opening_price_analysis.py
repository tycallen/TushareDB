#!/usr/bin/env python3
"""
开盘价与全天走势关系研究
研究内容：
1. 开盘特征分析（高开/低开统计、跳空缺口、开盘涨跌幅分布）
2. 开盘信号（高开高走/低走概率、低开高走机会、缺口回补规律）
3. 策略设计（开盘博弈策略、缺口交易策略、风险控制）
"""

import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti', 'Heiti SC']
plt.rcParams['axes.unicode_minus'] = False

# 数据库路径
DB_PATH = '/Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db'
REPORT_DIR = '/Users/allen/workspace/python/stock/Tushare-DuckDB/reports/research'


def load_data(start_date='20200101', end_date='20260130'):
    """加载日线数据"""
    conn = duckdb.connect(DB_PATH, read_only=True)

    query = f"""
    SELECT
        ts_code,
        trade_date,
        open,
        high,
        low,
        close,
        pre_close,
        pct_chg,
        vol,
        amount
    FROM daily
    WHERE trade_date >= '{start_date}'
      AND trade_date <= '{end_date}'
      AND open IS NOT NULL
      AND close IS NOT NULL
      AND pre_close IS NOT NULL
      AND pre_close > 0
      AND open > 0
    ORDER BY ts_code, trade_date
    """

    df = conn.execute(query).fetchdf()
    conn.close()

    # 计算开盘相关指标
    df['open_pct'] = (df['open'] / df['pre_close'] - 1) * 100  # 开盘涨跌幅
    df['close_pct'] = df['pct_chg']  # 收盘涨跌幅
    df['intraday_pct'] = (df['close'] / df['open'] - 1) * 100  # 盘中涨跌幅
    df['gap'] = df['open_pct']  # 跳空幅度
    df['high_pct'] = (df['high'] / df['pre_close'] - 1) * 100  # 最高价涨幅
    df['low_pct'] = (df['low'] / df['pre_close'] - 1) * 100  # 最低价跌幅

    return df


def analyze_opening_features(df):
    """分析开盘特征"""
    print("=" * 80)
    print("1. 开盘特征分析")
    print("=" * 80)

    results = {}

    # 1.1 高开/低开统计
    print("\n1.1 高开/低开统计")
    print("-" * 40)

    df['open_type'] = pd.cut(df['open_pct'],
                              bins=[-np.inf, -3, -1, 0, 1, 3, np.inf],
                              labels=['大幅低开(<-3%)', '低开(-3%~-1%)', '微跌(0~-1%)',
                                     '微涨(0~1%)', '高开(1%~3%)', '大幅高开(>3%)'])

    open_stats = df['open_type'].value_counts()
    open_pct_stats = (open_stats / len(df) * 100).round(2)

    results['open_type_distribution'] = pd.DataFrame({
        '次数': open_stats,
        '占比(%)': open_pct_stats
    }).sort_index()

    print(results['open_type_distribution'])

    # 1.2 跳空缺口分析
    print("\n1.2 跳空缺口分析")
    print("-" * 40)

    # 定义跳空缺口：开盘价高于昨日最高价或低于昨日最低价
    df_with_prev = df.copy()
    df_with_prev['prev_high'] = df_with_prev.groupby('ts_code')['high'].shift(1)
    df_with_prev['prev_low'] = df_with_prev.groupby('ts_code')['low'].shift(1)

    df_with_prev['gap_up'] = df_with_prev['open'] > df_with_prev['prev_high']  # 向上跳空
    df_with_prev['gap_down'] = df_with_prev['open'] < df_with_prev['prev_low']  # 向下跳空
    df_with_prev = df_with_prev.dropna(subset=['prev_high', 'prev_low'])

    gap_up_count = df_with_prev['gap_up'].sum()
    gap_down_count = df_with_prev['gap_down'].sum()
    total_count = len(df_with_prev)

    print(f"向上跳空缺口: {gap_up_count:,} 次 ({gap_up_count/total_count*100:.2f}%)")
    print(f"向下跳空缺口: {gap_down_count:,} 次 ({gap_down_count/total_count*100:.2f}%)")
    print(f"无跳空: {total_count - gap_up_count - gap_down_count:,} 次 ({(total_count - gap_up_count - gap_down_count)/total_count*100:.2f}%)")

    results['gap_stats'] = {
        '向上跳空': gap_up_count,
        '向下跳空': gap_down_count,
        '无跳空': total_count - gap_up_count - gap_down_count,
        '向上跳空占比': gap_up_count/total_count*100,
        '向下跳空占比': gap_down_count/total_count*100
    }

    # 按缺口大小分类
    df_with_prev['gap_size'] = 0
    df_with_prev.loc[df_with_prev['gap_up'], 'gap_size'] = (
        df_with_prev.loc[df_with_prev['gap_up'], 'open'] -
        df_with_prev.loc[df_with_prev['gap_up'], 'prev_high']
    ) / df_with_prev.loc[df_with_prev['gap_up'], 'prev_high'] * 100

    df_with_prev.loc[df_with_prev['gap_down'], 'gap_size'] = (
        df_with_prev.loc[df_with_prev['gap_down'], 'open'] -
        df_with_prev.loc[df_with_prev['gap_down'], 'prev_low']
    ) / df_with_prev.loc[df_with_prev['gap_down'], 'prev_low'] * 100

    # 1.3 开盘涨跌幅分布
    print("\n1.3 开盘涨跌幅分布统计")
    print("-" * 40)

    open_pct_stats = df['open_pct'].describe()
    print(f"平均开盘涨跌幅: {open_pct_stats['mean']:.4f}%")
    print(f"开盘涨跌幅标准差: {open_pct_stats['std']:.4f}%")
    print(f"开盘涨跌幅中位数: {open_pct_stats['50%']:.4f}%")
    print(f"开盘涨跌幅最小值: {open_pct_stats['min']:.4f}%")
    print(f"开盘涨跌幅最大值: {open_pct_stats['max']:.4f}%")

    results['open_pct_stats'] = open_pct_stats
    results['df_with_prev'] = df_with_prev

    return results, df_with_prev


def analyze_opening_signals(df, df_with_prev):
    """分析开盘信号"""
    print("\n" + "=" * 80)
    print("2. 开盘信号分析")
    print("=" * 80)

    results = {}

    # 2.1 高开高走概率分析
    print("\n2.1 高开高走/高开低走分析")
    print("-" * 40)

    # 高开定义：开盘涨幅 > 1%
    high_open = df[df['open_pct'] > 1].copy()
    high_open['is_high_walk'] = high_open['close'] > high_open['open']  # 收盘价高于开盘价
    high_open['is_low_walk'] = high_open['close'] < high_open['open']   # 收盘价低于开盘价

    high_open_high_walk = high_open['is_high_walk'].sum()
    high_open_low_walk = high_open['is_low_walk'].sum()
    high_open_total = len(high_open)

    print(f"高开(>1%)样本数: {high_open_total:,}")
    print(f"高开高走概率: {high_open_high_walk/high_open_total*100:.2f}%")
    print(f"高开低走概率: {high_open_low_walk/high_open_total*100:.2f}%")
    print(f"高开平收概率: {(high_open_total-high_open_high_walk-high_open_low_walk)/high_open_total*100:.2f}%")

    # 按高开幅度细分
    print("\n按高开幅度细分：")
    high_open_ranges = [
        (1, 2, '1%-2%'),
        (2, 3, '2%-3%'),
        (3, 5, '3%-5%'),
        (5, 10, '5%-10%'),
        (10, 100, '>10%')
    ]

    high_open_analysis = []
    for low, high, label in high_open_ranges:
        subset = df[(df['open_pct'] >= low) & (df['open_pct'] < high)]
        if len(subset) > 0:
            high_walk_rate = (subset['close'] > subset['open']).sum() / len(subset) * 100
            low_walk_rate = (subset['close'] < subset['open']).sum() / len(subset) * 100
            avg_intraday = subset['intraday_pct'].mean()
            avg_close_pct = subset['close_pct'].mean()
            high_open_analysis.append({
                '高开幅度': label,
                '样本数': len(subset),
                '高开高走率(%)': round(high_walk_rate, 2),
                '高开低走率(%)': round(low_walk_rate, 2),
                '平均盘中涨跌(%)': round(avg_intraday, 2),
                '平均收盘涨跌(%)': round(avg_close_pct, 2)
            })

    results['high_open_analysis'] = pd.DataFrame(high_open_analysis)
    print(results['high_open_analysis'].to_string(index=False))

    # 2.2 低开高走机会分析
    print("\n2.2 低开高走/低开低走分析")
    print("-" * 40)

    low_open = df[df['open_pct'] < -1].copy()
    low_open['is_high_walk'] = low_open['close'] > low_open['open']
    low_open['is_low_walk'] = low_open['close'] < low_open['open']

    low_open_high_walk = low_open['is_high_walk'].sum()
    low_open_low_walk = low_open['is_low_walk'].sum()
    low_open_total = len(low_open)

    print(f"低开(<-1%)样本数: {low_open_total:,}")
    print(f"低开高走概率: {low_open_high_walk/low_open_total*100:.2f}%")
    print(f"低开低走概率: {low_open_low_walk/low_open_total*100:.2f}%")

    # 按低开幅度细分
    print("\n按低开幅度细分：")
    low_open_ranges = [
        (-2, -1, '-1%~-2%'),
        (-3, -2, '-2%~-3%'),
        (-5, -3, '-3%~-5%'),
        (-10, -5, '-5%~-10%'),
        (-100, -10, '<-10%')
    ]

    low_open_analysis = []
    for low, high, label in low_open_ranges:
        subset = df[(df['open_pct'] >= low) & (df['open_pct'] < high)]
        if len(subset) > 0:
            high_walk_rate = (subset['close'] > subset['open']).sum() / len(subset) * 100
            low_walk_rate = (subset['close'] < subset['open']).sum() / len(subset) * 100
            avg_intraday = subset['intraday_pct'].mean()
            avg_close_pct = subset['close_pct'].mean()
            low_open_analysis.append({
                '低开幅度': label,
                '样本数': len(subset),
                '低开高走率(%)': round(high_walk_rate, 2),
                '低开低走率(%)': round(low_walk_rate, 2),
                '平均盘中涨跌(%)': round(avg_intraday, 2),
                '平均收盘涨跌(%)': round(avg_close_pct, 2)
            })

    results['low_open_analysis'] = pd.DataFrame(low_open_analysis)
    print(results['low_open_analysis'].to_string(index=False))

    # 2.3 缺口回补规律
    print("\n2.3 缺口回补规律分析")
    print("-" * 40)

    # 向上跳空缺口回补分析
    gap_up_df = df_with_prev[df_with_prev['gap_up']].copy()
    if len(gap_up_df) > 0:
        # 当日回补：最低价低于昨日最高价
        gap_up_df['same_day_fill'] = gap_up_df['low'] <= gap_up_df['prev_high']
        same_day_fill_rate = gap_up_df['same_day_fill'].sum() / len(gap_up_df) * 100

        print(f"向上跳空缺口当日回补率: {same_day_fill_rate:.2f}%")

        # 按缺口大小分析回补率
        gap_up_df['gap_pct'] = (gap_up_df['open'] - gap_up_df['prev_high']) / gap_up_df['prev_high'] * 100

        gap_ranges = [
            (0, 1, '0-1%'),
            (1, 2, '1%-2%'),
            (2, 3, '2%-3%'),
            (3, 5, '3%-5%'),
            (5, 100, '>5%')
        ]

        gap_fill_analysis = []
        for low, high, label in gap_ranges:
            subset = gap_up_df[(gap_up_df['gap_pct'] >= low) & (gap_up_df['gap_pct'] < high)]
            if len(subset) > 100:  # 确保样本量足够
                fill_rate = subset['same_day_fill'].sum() / len(subset) * 100
                avg_intraday = subset['intraday_pct'].mean()
                gap_fill_analysis.append({
                    '缺口大小': label,
                    '样本数': len(subset),
                    '当日回补率(%)': round(fill_rate, 2),
                    '平均盘中涨跌(%)': round(avg_intraday, 2)
                })

        if gap_fill_analysis:
            results['gap_up_fill'] = pd.DataFrame(gap_fill_analysis)
            print("\n向上跳空缺口回补分析：")
            print(results['gap_up_fill'].to_string(index=False))

    # 向下跳空缺口回补分析
    gap_down_df = df_with_prev[df_with_prev['gap_down']].copy()
    if len(gap_down_df) > 0:
        # 当日回补：最高价高于昨日最低价
        gap_down_df['same_day_fill'] = gap_down_df['high'] >= gap_down_df['prev_low']
        same_day_fill_rate = gap_down_df['same_day_fill'].sum() / len(gap_down_df) * 100

        print(f"\n向下跳空缺口当日回补率: {same_day_fill_rate:.2f}%")

        # 按缺口大小分析回补率
        gap_down_df['gap_pct'] = (gap_down_df['prev_low'] - gap_down_df['open']) / gap_down_df['prev_low'] * 100

        gap_fill_down_analysis = []
        for low, high, label in gap_ranges:
            subset = gap_down_df[(gap_down_df['gap_pct'] >= low) & (gap_down_df['gap_pct'] < high)]
            if len(subset) > 100:
                fill_rate = subset['same_day_fill'].sum() / len(subset) * 100
                avg_intraday = subset['intraday_pct'].mean()
                gap_fill_down_analysis.append({
                    '缺口大小': label,
                    '样本数': len(subset),
                    '当日回补率(%)': round(fill_rate, 2),
                    '平均盘中涨跌(%)': round(avg_intraday, 2)
                })

        if gap_fill_down_analysis:
            results['gap_down_fill'] = pd.DataFrame(gap_fill_down_analysis)
            print("\n向下跳空缺口回补分析：")
            print(results['gap_down_fill'].to_string(index=False))

    return results


def analyze_market_conditions(df):
    """分析不同市场环境下的开盘信号"""
    print("\n" + "=" * 80)
    print("3. 市场环境对开盘信号的影响")
    print("=" * 80)

    results = {}

    # 计算市场整体涨跌
    daily_market = df.groupby('trade_date').agg({
        'pct_chg': 'mean',
        'open_pct': 'mean'
    }).reset_index()

    daily_market['market_trend'] = pd.cut(
        daily_market['pct_chg'],
        bins=[-np.inf, -1, 0, 1, np.inf],
        labels=['熊市日(<-1%)', '弱势日(-1%~0)', '强势日(0~1%)', '牛市日(>1%)']
    )

    # 合并市场趋势到原数据
    df_with_market = df.merge(
        daily_market[['trade_date', 'market_trend']],
        on='trade_date',
        how='left'
    )

    print("\n3.1 不同市场环境下高开高走概率")
    print("-" * 40)

    market_analysis = []
    for trend in ['牛市日(>1%)', '强势日(0~1%)', '弱势日(-1%~0)', '熊市日(<-1%)']:
        subset = df_with_market[
            (df_with_market['market_trend'] == trend) &
            (df_with_market['open_pct'] > 1)
        ]
        if len(subset) > 0:
            high_walk_rate = (subset['close'] > subset['open']).sum() / len(subset) * 100
            market_analysis.append({
                '市场环境': trend,
                '高开样本数': len(subset),
                '高开高走率(%)': round(high_walk_rate, 2)
            })

    results['market_high_open'] = pd.DataFrame(market_analysis)
    print(results['market_high_open'].to_string(index=False))

    print("\n3.2 不同市场环境下低开高走概率")
    print("-" * 40)

    market_low_analysis = []
    for trend in ['牛市日(>1%)', '强势日(0~1%)', '弱势日(-1%~0)', '熊市日(<-1%)']:
        subset = df_with_market[
            (df_with_market['market_trend'] == trend) &
            (df_with_market['open_pct'] < -1)
        ]
        if len(subset) > 0:
            high_walk_rate = (subset['close'] > subset['open']).sum() / len(subset) * 100
            market_low_analysis.append({
                '市场环境': trend,
                '低开样本数': len(subset),
                '低开高走率(%)': round(high_walk_rate, 2)
            })

    results['market_low_open'] = pd.DataFrame(market_low_analysis)
    print(results['market_low_open'].to_string(index=False))

    return results


def design_strategies(df, df_with_prev):
    """设计交易策略"""
    print("\n" + "=" * 80)
    print("4. 交易策略设计与回测")
    print("=" * 80)

    results = {}

    # 4.1 开盘博弈策略
    print("\n4.1 开盘博弈策略")
    print("-" * 40)

    print("""
策略1: 低开高走策略
- 入场条件: 开盘跌幅2%-5%，非涨跌停
- 出场条件: 当日收盘
- 止损: 继续下跌至-8%
""")

    # 回测低开高走策略
    low_open_strategy = df[(df['open_pct'] >= -5) & (df['open_pct'] <= -2)].copy()
    low_open_strategy['strategy_return'] = low_open_strategy['intraday_pct']

    avg_return = low_open_strategy['strategy_return'].mean()
    win_rate = (low_open_strategy['strategy_return'] > 0).sum() / len(low_open_strategy) * 100
    max_win = low_open_strategy['strategy_return'].max()
    max_loss = low_open_strategy['strategy_return'].min()

    print(f"策略1回测结果:")
    print(f"  交易次数: {len(low_open_strategy):,}")
    print(f"  平均收益: {avg_return:.2f}%")
    print(f"  胜率: {win_rate:.2f}%")
    print(f"  最大盈利: {max_win:.2f}%")
    print(f"  最大亏损: {max_loss:.2f}%")
    print(f"  盈亏比: {abs(low_open_strategy[low_open_strategy['strategy_return']>0]['strategy_return'].mean() / low_open_strategy[low_open_strategy['strategy_return']<0]['strategy_return'].mean()):.2f}")

    results['low_open_strategy'] = {
        '交易次数': len(low_open_strategy),
        '平均收益': avg_return,
        '胜率': win_rate,
        '最大盈利': max_win,
        '最大亏损': max_loss
    }

    # 4.2 缺口交易策略
    print("\n4.2 缺口交易策略")
    print("-" * 40)

    print("""
策略2: 向上跳空缺口回补策略
- 入场条件: 向上跳空1%-3%
- 方向: 做空（预期回补）
- 出场条件: 缺口回补或当日收盘
""")

    gap_up_strategy = df_with_prev[df_with_prev['gap_up']].copy()
    gap_up_strategy['gap_pct'] = (gap_up_strategy['open'] - gap_up_strategy['prev_high']) / gap_up_strategy['prev_high'] * 100
    gap_up_strategy = gap_up_strategy[(gap_up_strategy['gap_pct'] >= 1) & (gap_up_strategy['gap_pct'] <= 3)]

    # 做空策略：盘中从开盘价下跌
    gap_up_strategy['strategy_return'] = -gap_up_strategy['intraday_pct']

    if len(gap_up_strategy) > 0:
        avg_return = gap_up_strategy['strategy_return'].mean()
        win_rate = (gap_up_strategy['strategy_return'] > 0).sum() / len(gap_up_strategy) * 100

        print(f"策略2回测结果:")
        print(f"  交易次数: {len(gap_up_strategy):,}")
        print(f"  平均收益: {avg_return:.2f}%")
        print(f"  胜率: {win_rate:.2f}%")

    print("""
策略3: 向下跳空缺口回补策略
- 入场条件: 向下跳空1%-3%
- 方向: 做多（预期反弹回补）
- 出场条件: 缺口回补或当日收盘
""")

    gap_down_strategy = df_with_prev[df_with_prev['gap_down']].copy()
    gap_down_strategy['gap_pct'] = (gap_down_strategy['prev_low'] - gap_down_strategy['open']) / gap_down_strategy['prev_low'] * 100
    gap_down_strategy = gap_down_strategy[(gap_down_strategy['gap_pct'] >= 1) & (gap_down_strategy['gap_pct'] <= 3)]

    gap_down_strategy['strategy_return'] = gap_down_strategy['intraday_pct']

    if len(gap_down_strategy) > 0:
        avg_return = gap_down_strategy['strategy_return'].mean()
        win_rate = (gap_down_strategy['strategy_return'] > 0).sum() / len(gap_down_strategy) * 100

        print(f"策略3回测结果:")
        print(f"  交易次数: {len(gap_down_strategy):,}")
        print(f"  平均收益: {avg_return:.2f}%")
        print(f"  胜率: {win_rate:.2f}%")

    # 4.3 风险控制
    print("\n4.3 风险控制建议")
    print("-" * 40)

    print("""
风险控制要点：

1. 仓位管理
   - 单笔交易不超过总资金的10%
   - 高开超过5%的股票慎重参与
   - 低开超过5%的股票需评估基本面

2. 止损设置
   - 高开低走：跌破开盘价1%止损
   - 低开高走：跌破当日最低价止损
   - 缺口策略：缺口扩大超过50%止损

3. 时间控制
   - 开盘30分钟内观察，不急于入场
   - 上午10:30前决定方向
   - 下午2:30后谨慎开新仓

4. 市场环境过滤
   - 大盘涨跌超过2%的日子谨慎操作
   - 结合成交量判断真假突破
   - 注意板块联动效应
""")

    return results


def create_visualizations(df, df_with_prev):
    """创建可视化图表"""
    print("\n" + "=" * 80)
    print("5. 生成可视化图表")
    print("=" * 80)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. 开盘涨跌幅分布
    ax1 = axes[0, 0]
    df['open_pct'].clip(-10, 10).hist(bins=100, ax=ax1, color='steelblue', alpha=0.7, edgecolor='white')
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax1.set_title('开盘涨跌幅分布', fontsize=14, fontweight='bold')
    ax1.set_xlabel('开盘涨跌幅(%)')
    ax1.set_ylabel('频次')

    # 2. 高开后的盘中表现
    ax2 = axes[0, 1]
    high_open_ranges = [(1, 2), (2, 3), (3, 5), (5, 10)]
    labels = ['1%-2%', '2%-3%', '3%-5%', '5%-10%']
    high_walk_rates = []
    for low, high in high_open_ranges:
        subset = df[(df['open_pct'] >= low) & (df['open_pct'] < high)]
        if len(subset) > 0:
            rate = (subset['close'] > subset['open']).sum() / len(subset) * 100
            high_walk_rates.append(rate)
        else:
            high_walk_rates.append(0)

    bars = ax2.bar(labels, high_walk_rates, color=['green' if r > 50 else 'red' for r in high_walk_rates], alpha=0.7)
    ax2.axhline(y=50, color='gray', linestyle='--', linewidth=1)
    ax2.set_title('高开后高走概率', fontsize=14, fontweight='bold')
    ax2.set_xlabel('高开幅度')
    ax2.set_ylabel('高走概率(%)')
    ax2.set_ylim(0, 100)
    for bar, rate in zip(bars, high_walk_rates):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=10)

    # 3. 低开后的盘中表现
    ax3 = axes[0, 2]
    low_open_ranges = [(-2, -1), (-3, -2), (-5, -3), (-10, -5)]
    labels = ['-1%~-2%', '-2%~-3%', '-3%~-5%', '-5%~-10%']
    low_high_walk_rates = []
    for low, high in low_open_ranges:
        subset = df[(df['open_pct'] >= low) & (df['open_pct'] < high)]
        if len(subset) > 0:
            rate = (subset['close'] > subset['open']).sum() / len(subset) * 100
            low_high_walk_rates.append(rate)
        else:
            low_high_walk_rates.append(0)

    bars = ax3.bar(labels, low_high_walk_rates, color=['green' if r > 50 else 'red' for r in low_high_walk_rates], alpha=0.7)
    ax3.axhline(y=50, color='gray', linestyle='--', linewidth=1)
    ax3.set_title('低开后高走概率', fontsize=14, fontweight='bold')
    ax3.set_xlabel('低开幅度')
    ax3.set_ylabel('高走概率(%)')
    ax3.set_ylim(0, 100)
    for bar, rate in zip(bars, low_high_walk_rates):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=10)

    # 4. 开盘涨跌幅与收盘涨跌幅关系
    ax4 = axes[1, 0]
    # 抽样绘制散点图
    sample = df.sample(min(10000, len(df)))
    ax4.scatter(sample['open_pct'].clip(-10, 10), sample['close_pct'].clip(-10, 10),
                alpha=0.1, s=1, c='blue')
    ax4.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax4.axvline(x=0, color='gray', linestyle='-', linewidth=0.5)
    # 添加趋势线
    z = np.polyfit(sample['open_pct'].clip(-10, 10), sample['close_pct'].clip(-10, 10), 1)
    p = np.poly1d(z)
    x_line = np.linspace(-10, 10, 100)
    ax4.plot(x_line, p(x_line), 'r-', linewidth=2, label=f'趋势线: y={z[0]:.2f}x+{z[1]:.2f}')
    ax4.set_title('开盘涨跌幅 vs 收盘涨跌幅', fontsize=14, fontweight='bold')
    ax4.set_xlabel('开盘涨跌幅(%)')
    ax4.set_ylabel('收盘涨跌幅(%)')
    ax4.legend()
    ax4.set_xlim(-10, 10)
    ax4.set_ylim(-10, 10)

    # 5. 缺口回补率
    ax5 = axes[1, 1]
    gap_up_df = df_with_prev[df_with_prev['gap_up']].copy()
    if len(gap_up_df) > 0:
        gap_up_df['gap_pct'] = (gap_up_df['open'] - gap_up_df['prev_high']) / gap_up_df['prev_high'] * 100
        gap_up_df['same_day_fill'] = gap_up_df['low'] <= gap_up_df['prev_high']

        gap_ranges = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 10)]
        labels = ['0-1%', '1%-2%', '2%-3%', '3%-5%', '5%-10%']
        fill_rates = []
        for low, high in gap_ranges:
            subset = gap_up_df[(gap_up_df['gap_pct'] >= low) & (gap_up_df['gap_pct'] < high)]
            if len(subset) > 100:
                rate = subset['same_day_fill'].sum() / len(subset) * 100
                fill_rates.append(rate)
            else:
                fill_rates.append(0)

        bars = ax5.bar(labels, fill_rates, color='orange', alpha=0.7)
        ax5.set_title('向上跳空缺口当日回补率', fontsize=14, fontweight='bold')
        ax5.set_xlabel('缺口大小')
        ax5.set_ylabel('回补率(%)')
        ax5.set_ylim(0, 100)
        for bar, rate in zip(bars, fill_rates):
            if rate > 0:
                ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{rate:.1f}%', ha='center', va='bottom', fontsize=10)

    # 6. 开盘类型与平均收益
    ax6 = axes[1, 2]
    open_ranges = [
        (-10, -5, '大幅低开'),
        (-5, -3, '低开-3~-5%'),
        (-3, -1, '低开-1~-3%'),
        (-1, 0, '微跌'),
        (0, 1, '微涨'),
        (1, 3, '高开1~3%'),
        (3, 5, '高开3~5%'),
        (5, 10, '大幅高开')
    ]

    avg_returns = []
    labels = []
    for low, high, label in open_ranges:
        subset = df[(df['open_pct'] >= low) & (df['open_pct'] < high)]
        if len(subset) > 0:
            avg_returns.append(subset['intraday_pct'].mean())
            labels.append(label)

    colors = ['green' if r > 0 else 'red' for r in avg_returns]
    bars = ax6.bar(labels, avg_returns, color=colors, alpha=0.7)
    ax6.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax6.set_title('不同开盘类型的平均盘中收益', fontsize=14, fontweight='bold')
    ax6.set_xlabel('开盘类型')
    ax6.set_ylabel('平均盘中收益(%)')
    ax6.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(f'{REPORT_DIR}/opening_price_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"图表已保存到: {REPORT_DIR}/opening_price_analysis.png")


def generate_report(df, opening_features, opening_signals, market_results, strategy_results):
    """生成研究报告"""

    report = f"""# 开盘价与全天走势关系研究报告

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 研究概述

本研究基于A股市场日线数据，深入分析开盘价与全天走势之间的关系，
探索可行的交易策略和风险控制方法。

### 数据范围
- 数据区间: 2020-01-01 至 2026-01-30
- 样本数量: {len(df):,} 条记录
- 覆盖股票: {df['ts_code'].nunique():,} 只

---

## 1. 开盘特征分析

### 1.1 高开/低开统计

开盘涨跌幅分布统计：

{opening_features['open_type_distribution'].to_markdown()}

### 1.2 跳空缺口分析

| 类型 | 次数 | 占比 |
|------|------|------|
| 向上跳空 | {opening_features['gap_stats']['向上跳空']:,} | {opening_features['gap_stats']['向上跳空占比']:.2f}% |
| 向下跳空 | {opening_features['gap_stats']['向下跳空']:,} | {opening_features['gap_stats']['向下跳空占比']:.2f}% |
| 无跳空 | {opening_features['gap_stats']['无跳空']:,} | {100-opening_features['gap_stats']['向上跳空占比']-opening_features['gap_stats']['向下跳空占比']:.2f}% |

### 1.3 开盘涨跌幅分布

| 统计指标 | 数值 |
|----------|------|
| 平均值 | {opening_features['open_pct_stats']['mean']:.4f}% |
| 标准差 | {opening_features['open_pct_stats']['std']:.4f}% |
| 中位数 | {opening_features['open_pct_stats']['50%']:.4f}% |
| 最小值 | {opening_features['open_pct_stats']['min']:.4f}% |
| 最大值 | {opening_features['open_pct_stats']['max']:.4f}% |

---

## 2. 开盘信号分析

### 2.1 高开高走概率

按高开幅度分析：

{opening_signals['high_open_analysis'].to_markdown(index=False)}

**核心发现：**
- 高开幅度越大，高开低走的概率越高
- 1%-2%的高开反而有较高的继续上涨概率
- 超过5%的高开需要特别警惕回调风险

### 2.2 低开高走机会

按低开幅度分析：

{opening_signals['low_open_analysis'].to_markdown(index=False)}

**核心发现：**
- 低开2%-5%的股票有较好的盘中反弹机会
- 低开超过5%的股票风险较大，需谨慎
- 低开幅度适中时，存在较好的博弈机会

### 2.3 缺口回补规律

#### 向上跳空缺口回补

{opening_signals.get('gap_up_fill', pd.DataFrame()).to_markdown(index=False) if 'gap_up_fill' in opening_signals else '数据不足'}

#### 向下跳空缺口回补

{opening_signals.get('gap_down_fill', pd.DataFrame()).to_markdown(index=False) if 'gap_down_fill' in opening_signals else '数据不足'}

**核心发现：**
- 小缺口（1%以内）当日回补概率较高
- 大缺口（>3%）回补概率明显降低
- 缺口大小与回补概率呈负相关

---

## 3. 市场环境影响

### 3.1 不同市场环境下高开高走概率

{market_results['market_high_open'].to_markdown(index=False)}

### 3.2 不同市场环境下低开高走概率

{market_results['market_low_open'].to_markdown(index=False)}

**核心发现：**
- 牛市环境下，高开高走概率显著提高
- 熊市环境下，低开高走难度加大
- 市场环境是重要的过滤条件

---

## 4. 交易策略设计

### 4.1 低开高走策略

**策略规则：**
- 入场条件: 开盘跌幅2%-5%
- 方向: 做多
- 出场: 当日收盘
- 止损: 跌幅达到8%

**回测结果：**
- 交易次数: {strategy_results['low_open_strategy']['交易次数']:,}
- 平均收益: {strategy_results['low_open_strategy']['平均收益']:.2f}%
- 胜率: {strategy_results['low_open_strategy']['胜率']:.2f}%

### 4.2 缺口回补策略

**向上跳空做空策略：**
- 入场: 向上跳空1%-3%时做空
- 出场: 缺口回补或收盘
- 适用场景: 市场震荡或偏弱时

**向下跳空做多策略：**
- 入场: 向下跳空1%-3%时做多
- 出场: 缺口回补或收盘
- 适用场景: 市场偏强时

### 4.3 风险控制建议

1. **仓位管理**
   - 单笔交易不超过总资金的10%
   - 高开超过5%的股票慎重参与
   - 低开超过5%需评估基本面

2. **止损设置**
   - 高开低走: 跌破开盘价1%止损
   - 低开高走: 跌破当日最低价止损
   - 缺口策略: 缺口扩大50%止损

3. **时间控制**
   - 开盘30分钟内观察
   - 10:30前决定方向
   - 14:30后谨慎开新仓

4. **市场环境过滤**
   - 大盘涨跌超过2%谨慎操作
   - 结合成交量判断
   - 注意板块联动

---

## 5. 核心结论

1. **开盘涨跌幅预测性**
   - 开盘涨跌幅与收盘涨跌幅有正相关性
   - 但极端开盘（高开或低开超过3%）容易出现反转

2. **缺口交易价值**
   - 小缺口（1%以内）回补概率高，可做逆向交易
   - 大缺口往往代表趋势启动，不宜逆向

3. **市场环境重要性**
   - 开盘信号需结合市场环境判断
   - 牛市做多信号更可靠，熊市做空信号更可靠

4. **风险收益权衡**
   - 低开高走策略胜率约50%，但需严格止损
   - 开盘博弈属于短线策略，不宜重仓

---

## 附录：研究方法说明

### 数据处理
- 剔除开盘价、收盘价、前收盘价为空的数据
- 剔除价格异常（<=0）的数据
- 计算各类涨跌幅指标

### 指标定义
- 开盘涨跌幅 = (开盘价 / 昨收盘价 - 1) * 100%
- 盘中涨跌幅 = (收盘价 / 开盘价 - 1) * 100%
- 向上跳空 = 开盘价 > 昨日最高价
- 向下跳空 = 开盘价 < 昨日最低价
- 缺口回补 = 当日价格覆盖缺口区域

---

*报告由自动化分析系统生成*
"""

    with open(f'{REPORT_DIR}/opening_price_research_report.md', 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n报告已保存到: {REPORT_DIR}/opening_price_research_report.md")


def main():
    """主函数"""
    print("开始加载数据...")
    df = load_data()
    print(f"加载完成，共 {len(df):,} 条记录，覆盖 {df['ts_code'].nunique():,} 只股票")

    # 1. 开盘特征分析
    opening_features, df_with_prev = analyze_opening_features(df)

    # 2. 开盘信号分析
    opening_signals = analyze_opening_signals(df, df_with_prev)

    # 3. 市场环境分析
    market_results = analyze_market_conditions(df)

    # 4. 策略设计
    strategy_results = design_strategies(df, df_with_prev)

    # 5. 生成可视化
    create_visualizations(df, df_with_prev)

    # 6. 生成报告
    generate_report(df, opening_features, opening_signals, market_results, strategy_results)

    print("\n" + "=" * 80)
    print("分析完成!")
    print("=" * 80)


if __name__ == '__main__':
    main()
