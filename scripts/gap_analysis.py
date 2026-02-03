#!/usr/bin/env python3
"""
跳空缺口形态研究分析

研究内容：
1. 缺口类型识别（普通缺口、突破缺口、持续缺口、衰竭缺口）
2. 缺口统计分析（频率、大小分布、回补概率）
3. 交易策略应用

Author: Claude AI
Date: 2026-02-01
"""

import duckdb
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 数据库路径
DB_PATH = '/Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db'
REPORT_PATH = '/Users/allen/workspace/python/stock/Tushare-DuckDB/reports/research'


def get_connection():
    """获取数据库连接"""
    return duckdb.connect(DB_PATH, read_only=True)


def load_daily_data(conn, start_date='20200101', end_date='20260130'):
    """加载日线数据"""
    query = f"""
    SELECT
        d.ts_code, d.trade_date, d.open, d.high, d.low, d.close,
        d.pre_close, d.pct_chg, d.vol, d.amount,
        s.name, s.industry
    FROM daily d
    LEFT JOIN stock_basic s ON d.ts_code = s.ts_code
    WHERE d.trade_date >= '{start_date}'
      AND d.trade_date <= '{end_date}'
      AND d.vol > 0
    ORDER BY d.ts_code, d.trade_date
    """
    return conn.execute(query).fetchdf()


def identify_gaps(df):
    """
    识别跳空缺口

    跳空缺口定义：
    - 向上跳空：当日最低价 > 前一日最高价
    - 向下跳空：当日最高价 < 前一日最低价
    """
    # 按股票和日期排序
    df = df.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)

    # 计算前一日的high和low
    df['prev_high'] = df.groupby('ts_code')['high'].shift(1)
    df['prev_low'] = df.groupby('ts_code')['low'].shift(1)
    df['prev_close'] = df.groupby('ts_code')['close'].shift(1)

    # 识别缺口
    df['gap_up'] = df['low'] > df['prev_high']  # 向上跳空
    df['gap_down'] = df['high'] < df['prev_low']  # 向下跳空
    df['has_gap'] = df['gap_up'] | df['gap_down']

    # 计算缺口大小
    df['gap_size'] = 0.0
    df.loc[df['gap_up'], 'gap_size'] = (df.loc[df['gap_up'], 'low'] - df.loc[df['gap_up'], 'prev_high']) / df.loc[df['gap_up'], 'prev_close'] * 100
    df.loc[df['gap_down'], 'gap_size'] = (df.loc[df['gap_down'], 'high'] - df.loc[df['gap_down'], 'prev_low']) / df.loc[df['gap_down'], 'prev_close'] * 100

    # 缺口方向
    df['gap_direction'] = 'none'
    df.loc[df['gap_up'], 'gap_direction'] = 'up'
    df.loc[df['gap_down'], 'gap_direction'] = 'down'

    return df


def classify_gap_type(df):
    """
    分类缺口类型

    1. 普通缺口（Common Gap）：成交量无明显变化，缺口较小，通常在几日内回补
    2. 突破缺口（Breakaway Gap）：伴随成交量放大，价格突破重要支撑/阻力位
    3. 持续缺口（Continuation Gap）：出现在趋势中途，也称为量度缺口
    4. 衰竭缺口（Exhaustion Gap）：出现在趋势末期，随后趋势反转
    """
    df = df.copy()

    # 计算技术指标
    # 20日均量
    df['vol_ma20'] = df.groupby('ts_code')['vol'].transform(lambda x: x.rolling(20, min_periods=10).mean())
    # 成交量比率
    df['vol_ratio'] = df['vol'] / df['vol_ma20']

    # 计算价格趋势
    df['ma20'] = df.groupby('ts_code')['close'].transform(lambda x: x.rolling(20, min_periods=10).mean())
    df['ma60'] = df.groupby('ts_code')['close'].transform(lambda x: x.rolling(60, min_periods=30).mean())

    # 计算过去20日涨跌幅
    df['return_20d'] = df.groupby('ts_code')['close'].transform(lambda x: x.pct_change(20) * 100)
    # 计算未来5日涨跌幅（用于判断缺口后走势）
    df['future_return_5d'] = df.groupby('ts_code')['close'].transform(lambda x: x.shift(-5) / x - 1) * 100

    # 计算20日最高价和最低价（用于判断突破）
    df['high_20d'] = df.groupby('ts_code')['high'].transform(lambda x: x.rolling(20, min_periods=10).max())
    df['low_20d'] = df.groupby('ts_code')['low'].transform(lambda x: x.rolling(20, min_periods=10).min())

    # 初始化缺口类型
    df['gap_type'] = 'none'

    # 只对有缺口的记录进行分类
    gap_mask = df['has_gap']

    # 分类逻辑
    for idx in df[gap_mask].index:
        row = df.loc[idx]
        gap_dir = row['gap_direction']
        gap_size = abs(row['gap_size'])
        vol_ratio = row['vol_ratio'] if pd.notna(row['vol_ratio']) else 1.0
        return_20d = row['return_20d'] if pd.notna(row['return_20d']) else 0

        # 判断是否突破关键位
        is_breakout = False
        if gap_dir == 'up' and pd.notna(row['high_20d']):
            is_breakout = row['close'] > row['high_20d']
        elif gap_dir == 'down' and pd.notna(row['low_20d']):
            is_breakout = row['close'] < row['low_20d']

        # 分类规则
        if gap_size < 1.0 and vol_ratio < 1.5:
            # 普通缺口：缺口小，成交量无明显放大
            df.loc[idx, 'gap_type'] = 'common'
        elif is_breakout and vol_ratio >= 1.5:
            # 突破缺口：突破关键位，成交量放大
            df.loc[idx, 'gap_type'] = 'breakaway'
        elif abs(return_20d) > 15 and vol_ratio >= 1.2:
            # 衰竭缺口：已有较大涨跌幅，可能是趋势末期
            if (gap_dir == 'up' and return_20d > 15) or (gap_dir == 'down' and return_20d < -15):
                df.loc[idx, 'gap_type'] = 'exhaustion'
            else:
                df.loc[idx, 'gap_type'] = 'continuation'
        elif 5 < abs(return_20d) <= 15 and vol_ratio >= 1.2:
            # 持续缺口：趋势中途，成交量放大
            df.loc[idx, 'gap_type'] = 'continuation'
        else:
            # 其他情况归为普通缺口
            df.loc[idx, 'gap_type'] = 'common'

    return df


def check_gap_fill(df, max_days=20):
    """
    检查缺口回补情况

    回补定义：
    - 向上跳空回补：后续某日最低价 <= 缺口下沿（前一日最高价）
    - 向下跳空回补：后续某日最高价 >= 缺口上沿（前一日最低价）
    """
    df = df.copy()
    df['gap_filled'] = False
    df['fill_days'] = np.nan
    df['fill_date'] = None

    # 只检查有缺口的记录
    gap_records = df[df['has_gap']].copy()

    for idx in gap_records.index:
        row = df.loc[idx]
        ts_code = row['ts_code']
        trade_date = row['trade_date']
        gap_dir = row['gap_direction']

        # 获取后续max_days天的数据
        future_data = df[(df['ts_code'] == ts_code) & (df['trade_date'] > trade_date)].head(max_days)

        if len(future_data) == 0:
            continue

        if gap_dir == 'up':
            # 向上跳空，检查是否回补到前一日最高价
            gap_bottom = row['prev_high']
            filled_mask = future_data['low'] <= gap_bottom
        else:
            # 向下跳空，检查是否回补到前一日最低价
            gap_top = row['prev_low']
            filled_mask = future_data['high'] >= gap_top

        if filled_mask.any():
            fill_idx = filled_mask.idxmax()
            fill_row = df.loc[fill_idx]
            df.loc[idx, 'gap_filled'] = True
            # 计算回补天数
            fill_date = fill_row['trade_date']
            df.loc[idx, 'fill_date'] = fill_date
            # 简单计算索引差作为天数估计
            df.loc[idx, 'fill_days'] = list(future_data.index).index(fill_idx) + 1

    return df


def analyze_gap_statistics(df):
    """分析缺口统计数据"""
    gaps = df[df['has_gap']].copy()

    stats = {}

    # 1. 缺口发生频率
    total_records = len(df)
    total_gaps = len(gaps)
    gap_up_count = len(gaps[gaps['gap_direction'] == 'up'])
    gap_down_count = len(gaps[gaps['gap_direction'] == 'down'])

    stats['frequency'] = {
        'total_records': total_records,
        'total_gaps': total_gaps,
        'gap_ratio': total_gaps / total_records * 100,
        'gap_up_count': gap_up_count,
        'gap_down_count': gap_down_count,
        'gap_up_ratio': gap_up_count / total_gaps * 100 if total_gaps > 0 else 0,
        'gap_down_ratio': gap_down_count / total_gaps * 100 if total_gaps > 0 else 0
    }

    # 2. 缺口大小分布
    gap_sizes = gaps['gap_size'].abs()
    stats['size_distribution'] = {
        'mean': gap_sizes.mean(),
        'median': gap_sizes.median(),
        'std': gap_sizes.std(),
        'min': gap_sizes.min(),
        'max': gap_sizes.max(),
        'percentile_25': gap_sizes.quantile(0.25),
        'percentile_75': gap_sizes.quantile(0.75),
        'percentile_90': gap_sizes.quantile(0.90),
        'percentile_95': gap_sizes.quantile(0.95)
    }

    # 按缺口大小分组
    stats['size_groups'] = {
        'small_1pct': len(gap_sizes[gap_sizes < 1]),
        'medium_1_3pct': len(gap_sizes[(gap_sizes >= 1) & (gap_sizes < 3)]),
        'large_3_5pct': len(gap_sizes[(gap_sizes >= 3) & (gap_sizes < 5)]),
        'very_large_5pct': len(gap_sizes[gap_sizes >= 5])
    }

    # 3. 缺口类型分布
    stats['type_distribution'] = gaps['gap_type'].value_counts().to_dict()

    # 4. 缺口回补统计
    filled_gaps = gaps[gaps['gap_filled'] == True]
    stats['fill_stats'] = {
        'total_gaps': len(gaps),
        'filled_count': len(filled_gaps),
        'fill_ratio': len(filled_gaps) / len(gaps) * 100 if len(gaps) > 0 else 0,
        'avg_fill_days': filled_gaps['fill_days'].mean() if len(filled_gaps) > 0 else np.nan,
        'median_fill_days': filled_gaps['fill_days'].median() if len(filled_gaps) > 0 else np.nan
    }

    # 按缺口方向统计回补率
    for direction in ['up', 'down']:
        dir_gaps = gaps[gaps['gap_direction'] == direction]
        dir_filled = dir_gaps[dir_gaps['gap_filled'] == True]
        stats['fill_stats'][f'{direction}_fill_ratio'] = len(dir_filled) / len(dir_gaps) * 100 if len(dir_gaps) > 0 else 0
        stats['fill_stats'][f'{direction}_avg_fill_days'] = dir_filled['fill_days'].mean() if len(dir_filled) > 0 else np.nan

    # 按缺口类型统计回补率
    stats['fill_by_type'] = {}
    for gap_type in ['common', 'breakaway', 'continuation', 'exhaustion']:
        type_gaps = gaps[gaps['gap_type'] == gap_type]
        type_filled = type_gaps[type_gaps['gap_filled'] == True]
        stats['fill_by_type'][gap_type] = {
            'count': len(type_gaps),
            'filled_count': len(type_filled),
            'fill_ratio': len(type_filled) / len(type_gaps) * 100 if len(type_gaps) > 0 else 0,
            'avg_fill_days': type_filled['fill_days'].mean() if len(type_filled) > 0 else np.nan
        }

    return stats


def analyze_gap_returns(df):
    """分析缺口后的收益表现"""
    gaps = df[df['has_gap']].copy()

    returns = {}

    # 按缺口方向和类型分析
    for direction in ['up', 'down']:
        returns[direction] = {}
        dir_gaps = gaps[gaps['gap_direction'] == direction]

        if len(dir_gaps) == 0:
            continue

        # 整体收益
        returns[direction]['overall'] = {
            'count': len(dir_gaps),
            'avg_5d_return': dir_gaps['future_return_5d'].mean(),
            'win_rate': (dir_gaps['future_return_5d'] > 0).mean() * 100,
            'avg_gap_size': dir_gaps['gap_size'].abs().mean()
        }

        # 按类型分析
        for gap_type in ['common', 'breakaway', 'continuation', 'exhaustion']:
            type_gaps = dir_gaps[dir_gaps['gap_type'] == gap_type]
            if len(type_gaps) > 0:
                returns[direction][gap_type] = {
                    'count': len(type_gaps),
                    'avg_5d_return': type_gaps['future_return_5d'].mean(),
                    'win_rate': (type_gaps['future_return_5d'] > 0).mean() * 100,
                    'avg_gap_size': type_gaps['gap_size'].abs().mean()
                }

    return returns


def plot_gap_analysis(df, stats, returns, output_path):
    """生成缺口分析图表"""
    gaps = df[df['has_gap']].copy()

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    # 1. 缺口大小分布直方图
    ax1 = axes[0, 0]
    gap_sizes = gaps['gap_size'].abs()
    ax1.hist(gap_sizes[gap_sizes < 10], bins=50, edgecolor='black', alpha=0.7)
    ax1.axvline(gap_sizes.median(), color='r', linestyle='--', label=f'Median: {gap_sizes.median():.2f}%')
    ax1.axvline(gap_sizes.mean(), color='g', linestyle='--', label=f'Mean: {gap_sizes.mean():.2f}%')
    ax1.set_xlabel('Gap Size (%)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Gap Size Distribution')
    ax1.legend()

    # 2. 缺口类型分布饼图
    ax2 = axes[0, 1]
    type_counts = gaps['gap_type'].value_counts()
    type_labels = {'common': 'Common', 'breakaway': 'Breakaway', 'continuation': 'Continuation', 'exhaustion': 'Exhaustion'}
    labels = [type_labels.get(t, t) for t in type_counts.index]
    ax2.pie(type_counts.values, labels=labels, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Gap Type Distribution')

    # 3. 缺口方向对比
    ax3 = axes[1, 0]
    direction_counts = [stats['frequency']['gap_up_count'], stats['frequency']['gap_down_count']]
    ax3.bar(['Up Gap', 'Down Gap'], direction_counts, color=['green', 'red'], alpha=0.7)
    ax3.set_ylabel('Count')
    ax3.set_title('Gap Direction Distribution')
    for i, v in enumerate(direction_counts):
        ax3.text(i, v + 100, str(v), ha='center')

    # 4. 各类型缺口回补率
    ax4 = axes[1, 1]
    types = ['common', 'breakaway', 'continuation', 'exhaustion']
    fill_ratios = [stats['fill_by_type'][t]['fill_ratio'] for t in types]
    bars = ax4.bar([type_labels[t] for t in types], fill_ratios, color=['blue', 'orange', 'green', 'red'], alpha=0.7)
    ax4.set_ylabel('Fill Rate (%)')
    ax4.set_title('Gap Fill Rate by Type (within 20 days)')
    for bar, ratio in zip(bars, fill_ratios):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{ratio:.1f}%', ha='center')

    # 5. 向上缺口5日收益分布
    ax5 = axes[2, 0]
    up_gaps = gaps[gaps['gap_direction'] == 'up']
    up_returns = up_gaps['future_return_5d'].dropna()
    ax5.hist(up_returns[(up_returns > -20) & (up_returns < 20)], bins=50, edgecolor='black', alpha=0.7, color='green')
    ax5.axvline(0, color='black', linestyle='-', linewidth=2)
    ax5.axvline(up_returns.mean(), color='r', linestyle='--', label=f'Mean: {up_returns.mean():.2f}%')
    ax5.set_xlabel('5-Day Return (%)')
    ax5.set_ylabel('Frequency')
    ax5.set_title('5-Day Return After Up Gap')
    ax5.legend()

    # 6. 向下缺口5日收益分布
    ax6 = axes[2, 1]
    down_gaps = gaps[gaps['gap_direction'] == 'down']
    down_returns = down_gaps['future_return_5d'].dropna()
    ax6.hist(down_returns[(down_returns > -20) & (down_returns < 20)], bins=50, edgecolor='black', alpha=0.7, color='red')
    ax6.axvline(0, color='black', linestyle='-', linewidth=2)
    ax6.axvline(down_returns.mean(), color='r', linestyle='--', label=f'Mean: {down_returns.mean():.2f}%')
    ax6.set_xlabel('5-Day Return (%)')
    ax6.set_ylabel('Frequency')
    ax6.set_title('5-Day Return After Down Gap')
    ax6.legend()

    plt.tight_layout()
    plt.savefig(f'{output_path}/gap_analysis_charts.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Charts saved to {output_path}/gap_analysis_charts.png")


def generate_report(stats, returns, output_path):
    """生成分析报告"""
    report = []
    report.append("=" * 80)
    report.append("                     跳空缺口形态研究报告")
    report.append("=" * 80)
    report.append(f"\n报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"数据周期: 2020-01-01 至 2026-01-30")
    report.append("\n")

    # 1. 缺口概述
    report.append("-" * 80)
    report.append("一、缺口概述")
    report.append("-" * 80)
    report.append(f"\n总交易记录数: {stats['frequency']['total_records']:,}")
    report.append(f"跳空缺口总数: {stats['frequency']['total_gaps']:,}")
    report.append(f"缺口发生比例: {stats['frequency']['gap_ratio']:.2f}%")
    report.append(f"\n向上跳空缺口: {stats['frequency']['gap_up_count']:,} ({stats['frequency']['gap_up_ratio']:.1f}%)")
    report.append(f"向下跳空缺口: {stats['frequency']['gap_down_count']:,} ({stats['frequency']['gap_down_ratio']:.1f}%)")

    # 2. 缺口类型分布
    report.append("\n" + "-" * 80)
    report.append("二、缺口类型分布")
    report.append("-" * 80)
    report.append("\n缺口类型定义:")
    report.append("  - 普通缺口(Common): 缺口较小(<1%)，成交量无明显变化，通常快速回补")
    report.append("  - 突破缺口(Breakaway): 伴随成交量放大，价格突破重要支撑/阻力位")
    report.append("  - 持续缺口(Continuation): 出现在趋势中途，成交量放大，确认趋势延续")
    report.append("  - 衰竭缺口(Exhaustion): 出现在趋势末期，随后可能趋势反转")
    report.append("\n各类型数量统计:")
    type_names = {'common': '普通缺口', 'breakaway': '突破缺口', 'continuation': '持续缺口', 'exhaustion': '衰竭缺口'}
    for gap_type, count in stats['type_distribution'].items():
        ratio = count / stats['frequency']['total_gaps'] * 100
        report.append(f"  {type_names.get(gap_type, gap_type)}: {count:,} ({ratio:.1f}%)")

    # 3. 缺口大小分布
    report.append("\n" + "-" * 80)
    report.append("三、缺口大小分布")
    report.append("-" * 80)
    report.append(f"\n缺口大小统计 (绝对值):")
    report.append(f"  平均值: {stats['size_distribution']['mean']:.2f}%")
    report.append(f"  中位数: {stats['size_distribution']['median']:.2f}%")
    report.append(f"  标准差: {stats['size_distribution']['std']:.2f}%")
    report.append(f"  最小值: {stats['size_distribution']['min']:.2f}%")
    report.append(f"  最大值: {stats['size_distribution']['max']:.2f}%")
    report.append(f"  25%分位: {stats['size_distribution']['percentile_25']:.2f}%")
    report.append(f"  75%分位: {stats['size_distribution']['percentile_75']:.2f}%")
    report.append(f"  90%分位: {stats['size_distribution']['percentile_90']:.2f}%")
    report.append(f"  95%分位: {stats['size_distribution']['percentile_95']:.2f}%")

    report.append("\n缺口大小分组:")
    size_groups = stats['size_groups']
    total = sum(size_groups.values())
    report.append(f"  小缺口 (<1%): {size_groups['small_1pct']:,} ({size_groups['small_1pct']/total*100:.1f}%)")
    report.append(f"  中等缺口 (1-3%): {size_groups['medium_1_3pct']:,} ({size_groups['medium_1_3pct']/total*100:.1f}%)")
    report.append(f"  大缺口 (3-5%): {size_groups['large_3_5pct']:,} ({size_groups['large_3_5pct']/total*100:.1f}%)")
    report.append(f"  超大缺口 (>=5%): {size_groups['very_large_5pct']:,} ({size_groups['very_large_5pct']/total*100:.1f}%)")

    # 4. 缺口回补分析
    report.append("\n" + "-" * 80)
    report.append("四、缺口回补分析 (20个交易日内)")
    report.append("-" * 80)
    report.append(f"\n整体回补情况:")
    report.append(f"  总缺口数: {stats['fill_stats']['total_gaps']:,}")
    report.append(f"  已回补数: {stats['fill_stats']['filled_count']:,}")
    report.append(f"  回补比例: {stats['fill_stats']['fill_ratio']:.1f}%")
    report.append(f"  平均回补天数: {stats['fill_stats']['avg_fill_days']:.1f} 天")
    report.append(f"  中位回补天数: {stats['fill_stats']['median_fill_days']:.1f} 天")

    report.append(f"\n按方向统计:")
    report.append(f"  向上缺口回补率: {stats['fill_stats']['up_fill_ratio']:.1f}%，平均 {stats['fill_stats']['up_avg_fill_days']:.1f} 天")
    report.append(f"  向下缺口回补率: {stats['fill_stats']['down_fill_ratio']:.1f}%，平均 {stats['fill_stats']['down_avg_fill_days']:.1f} 天")

    report.append(f"\n按类型统计:")
    for gap_type in ['common', 'breakaway', 'continuation', 'exhaustion']:
        type_stat = stats['fill_by_type'][gap_type]
        name = type_names.get(gap_type, gap_type)
        if type_stat['count'] > 0:
            report.append(f"  {name}: 回补率 {type_stat['fill_ratio']:.1f}%，平均 {type_stat['avg_fill_days']:.1f} 天")

    # 5. 缺口后收益分析
    report.append("\n" + "-" * 80)
    report.append("五、缺口后收益分析 (5个交易日)")
    report.append("-" * 80)

    if 'up' in returns:
        report.append("\n【向上跳空缺口】")
        if 'overall' in returns['up']:
            r = returns['up']['overall']
            report.append(f"  整体表现: 样本数 {r['count']:,}, 平均收益 {r['avg_5d_return']:.2f}%, 胜率 {r['win_rate']:.1f}%")
        for gap_type in ['common', 'breakaway', 'continuation', 'exhaustion']:
            if gap_type in returns['up']:
                r = returns['up'][gap_type]
                name = type_names.get(gap_type, gap_type)
                report.append(f"  {name}: 样本数 {r['count']:,}, 平均收益 {r['avg_5d_return']:.2f}%, 胜率 {r['win_rate']:.1f}%")

    if 'down' in returns:
        report.append("\n【向下跳空缺口】")
        if 'overall' in returns['down']:
            r = returns['down']['overall']
            report.append(f"  整体表现: 样本数 {r['count']:,}, 平均收益 {r['avg_5d_return']:.2f}%, 胜率 {r['win_rate']:.1f}%")
        for gap_type in ['common', 'breakaway', 'continuation', 'exhaustion']:
            if gap_type in returns['down']:
                r = returns['down'][gap_type]
                name = type_names.get(gap_type, gap_type)
                report.append(f"  {name}: 样本数 {r['count']:,}, 平均收益 {r['avg_5d_return']:.2f}%, 胜率 {r['win_rate']:.1f}%")

    # 6. 交易策略建议
    report.append("\n" + "-" * 80)
    report.append("六、交易策略建议")
    report.append("-" * 80)

    report.append("\n【缺口交易策略】")
    report.append("\n1. 突破缺口策略:")
    report.append("   - 入场条件: 向上突破缺口，成交量放大(>1.5倍均量)，突破前期高点")
    report.append("   - 操作方式: 缺口当日或次日回踩不破缺口时买入")
    report.append("   - 止损位: 缺口下沿(前一日最高价)下方1-2%")
    report.append("   - 止盈位: 根据缺口大小设定，通常为缺口大小的1.5-2倍")

    report.append("\n2. 持续缺口策略:")
    report.append("   - 入场条件: 趋势中途出现持续缺口，顺应主趋势方向")
    report.append("   - 操作方式: 缺口确认后顺势加仓")
    report.append("   - 止损位: 缺口完全回补时止损")
    report.append("   - 止盈位: 可用缺口量度目标(缺口起点到终点的距离再往上投射)")

    report.append("\n3. 衰竭缺口反转策略:")
    report.append("   - 入场条件: 大涨/大跌后出现反向缺口，成交量异常放大")
    report.append("   - 操作方式: 等待缺口回补确认后反向操作")
    report.append("   - 止损位: 衰竭缺口最高/最低点")
    report.append("   - 风险提示: 需要结合其他技术指标确认反转信号")

    report.append("\n【缺口回补策略】")
    report.append("\n1. 普通缺口回补策略:")
    report.append(f"   - 普通缺口回补率高达 {stats['fill_by_type']['common']['fill_ratio']:.1f}%")
    report.append("   - 操作方式: 向上普通缺口后做空，向下普通缺口后做多")
    report.append("   - 目标位: 缺口完全回补")
    report.append("   - 止损位: 缺口扩大超过1%时止损")

    report.append("\n2. 缺口不回补策略:")
    report.append("   - 突破缺口和持续缺口回补率相对较低")
    report.append("   - 操作方式: 顺缺口方向持有，将缺口作为支撑/阻力位")
    report.append("   - 止损位: 缺口被完全回补时考虑止损")

    report.append("\n【风险控制】")
    report.append("\n1. 仓位管理:")
    report.append("   - 单笔交易风险不超过总资金的2%")
    report.append("   - 缺口交易属于高波动策略，建议适度降低仓位")
    report.append("   - 大缺口(>3%)的不确定性高，更需谨慎")

    report.append("\n2. 止损原则:")
    report.append("   - 突破缺口: 缺口下沿止损")
    report.append("   - 持续缺口: 缺口中点止损")
    report.append("   - 普通缺口: 缺口扩大止损")

    report.append("\n3. 风险提示:")
    report.append("   - 涨跌停造成的缺口不适用常规策略")
    report.append("   - 重大消息驱动的缺口需结合基本面分析")
    report.append("   - 开盘集合竞价期间的异常波动需谨慎对待")
    report.append("   - 历史统计不代表未来表现，需动态调整策略")

    report.append("\n" + "=" * 80)
    report.append("                           报告结束")
    report.append("=" * 80)

    # 保存报告
    report_text = "\n".join(report)
    with open(f'{output_path}/gap_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(report_text)
    print(f"\nReport saved to {output_path}/gap_analysis_report.txt")

    return report_text


def main():
    """主函数"""
    print("=" * 60)
    print("开始跳空缺口形态研究分析...")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1/6] 加载日线数据...")
    conn = get_connection()
    df = load_daily_data(conn)
    print(f"    加载完成，共 {len(df):,} 条记录")

    # 2. 识别缺口
    print("\n[2/6] 识别跳空缺口...")
    df = identify_gaps(df)
    gap_count = df['has_gap'].sum()
    print(f"    识别完成，共 {gap_count:,} 个缺口")

    # 3. 分类缺口类型
    print("\n[3/6] 分类缺口类型...")
    df = classify_gap_type(df)
    print("    分类完成")

    # 4. 检查缺口回补
    print("\n[4/6] 检查缺口回补情况...")
    df = check_gap_fill(df)
    filled_count = df[df['has_gap']]['gap_filled'].sum()
    print(f"    检查完成，{filled_count:,} 个缺口已回补")

    # 5. 统计分析
    print("\n[5/6] 进行统计分析...")
    stats = analyze_gap_statistics(df)
    returns = analyze_gap_returns(df)
    print("    统计完成")

    # 6. 生成报告和图表
    print("\n[6/6] 生成报告和图表...")
    plot_gap_analysis(df, stats, returns, REPORT_PATH)
    generate_report(stats, returns, REPORT_PATH)

    # 保存详细数据
    gaps_df = df[df['has_gap']][['ts_code', 'trade_date', 'name', 'industry',
                                  'gap_direction', 'gap_size', 'gap_type',
                                  'gap_filled', 'fill_days', 'vol_ratio',
                                  'future_return_5d']].copy()
    gaps_df.to_csv(f'{REPORT_PATH}/gap_details.csv', index=False, encoding='utf-8-sig')
    print(f"    详细数据已保存到 {REPORT_PATH}/gap_details.csv")

    print("\n" + "=" * 60)
    print("分析完成！")
    print("=" * 60)

    conn.close()


if __name__ == '__main__':
    main()
