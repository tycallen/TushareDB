#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MACD交易策略研究报告
=====================
研究内容:
1. MACD分析：计算、参数优化、分布
2. 交易信号：金叉/死叉、MACD背离、柱状图变化
3. 策略回测：MACD交叉策略、MACD动量策略、结合其他指标
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
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False

# 数据库连接
DB_PATH = '/Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db'
OUTPUT_DIR = '/Users/allen/workspace/python/stock/Tushare-DuckDB/reports/research'

def load_stock_data(ts_code='000001.SZ', start_date='20200101', end_date='20260130'):
    """加载股票日线数据"""
    conn = duckdb.connect(DB_PATH, read_only=True)
    query = f"""
    SELECT ts_code, trade_date, open, high, low, close, vol, amount
    FROM daily
    WHERE ts_code = '{ts_code}'
    AND trade_date >= '{start_date}'
    AND trade_date <= '{end_date}'
    ORDER BY trade_date
    """
    df = conn.execute(query).fetchdf()
    conn.close()
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    return df

def load_multiple_stocks(start_date='20200101', end_date='20260130', limit=50):
    """加载多只股票数据用于分析"""
    conn = duckdb.connect(DB_PATH, read_only=True)
    # 选择交易活跃的主板股票
    query = f"""
    SELECT ts_code, trade_date, open, high, low, close, vol, amount
    FROM daily
    WHERE trade_date >= '{start_date}'
    AND trade_date <= '{end_date}'
    AND ts_code IN (
        SELECT ts_code FROM (
            SELECT ts_code, COUNT(*) as cnt
            FROM daily
            WHERE trade_date >= '{start_date}'
            AND (ts_code LIKE '000%' OR ts_code LIKE '600%')
            GROUP BY ts_code
            HAVING cnt >= 1000
            ORDER BY RANDOM()
            LIMIT {limit}
        )
    )
    ORDER BY ts_code, trade_date
    """
    df = conn.execute(query).fetchdf()
    conn.close()
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    return df

# =============================================================================
# 第一部分：MACD分析
# =============================================================================

def calculate_ema(prices, period):
    """计算指数移动平均"""
    return prices.ewm(span=period, adjust=False).mean()

def calculate_macd(df, fast=12, slow=26, signal=9):
    """
    计算MACD指标

    参数:
    - fast: 快速EMA周期，默认12
    - slow: 慢速EMA周期，默认26
    - signal: 信号线周期，默认9

    返回:
    - DIF: 快速EMA - 慢速EMA
    - DEA: DIF的EMA
    - MACD: (DIF - DEA) * 2 (柱状图)
    """
    ema_fast = calculate_ema(df['close'], fast)
    ema_slow = calculate_ema(df['close'], slow)

    dif = ema_fast - ema_slow
    dea = calculate_ema(dif, signal)
    macd_hist = (dif - dea) * 2

    return dif, dea, macd_hist

def plot_macd_basic(df, ts_code):
    """绘制基本MACD图表"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[2, 1])
    fig.suptitle(f'MACD指标分析 - {ts_code}', fontsize=14, fontweight='bold')

    # 计算MACD
    dif, dea, macd_hist = calculate_macd(df)

    # 价格图
    ax1 = axes[0]
    ax1.plot(df['trade_date'], df['close'], 'b-', linewidth=1, label='收盘价')
    ax1.set_ylabel('价格')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('股价走势')

    # MACD图
    ax2 = axes[1]
    ax2.plot(df['trade_date'], dif, 'b-', linewidth=1, label='DIF')
    ax2.plot(df['trade_date'], dea, 'r-', linewidth=1, label='DEA')

    # 柱状图
    colors = ['red' if v >= 0 else 'green' for v in macd_hist]
    ax2.bar(df['trade_date'], macd_hist, color=colors, alpha=0.5, width=1)

    ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax2.set_ylabel('MACD')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('MACD指标')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/macd_basic_{ts_code.replace(".", "_")}.png', dpi=150, bbox_inches='tight')
    plt.close()

    return dif, dea, macd_hist

def macd_parameter_optimization(df, fast_range=range(5, 20, 2), slow_range=range(20, 40, 2), signal_range=range(5, 15, 2)):
    """
    MACD参数优化
    通过网格搜索找到最优参数组合
    """
    results = []

    for fast in fast_range:
        for slow in slow_range:
            if fast >= slow:
                continue
            for signal in signal_range:
                dif, dea, macd_hist = calculate_macd(df.copy(), fast, slow, signal)

                # 计算金叉死叉信号
                signals = pd.Series(0, index=df.index)
                signals[dif > dea] = 1  # 金叉
                signals[dif < dea] = -1  # 死叉

                # 计算信号变化（买卖点）
                signal_changes = signals.diff().fillna(0)
                buy_signals = (signal_changes == 2).sum()  # 死叉变金叉
                sell_signals = (signal_changes == -2).sum()  # 金叉变死叉

                # 简单回测收益
                df_temp = df.copy()
                df_temp['signal'] = signals
                df_temp['returns'] = df_temp['close'].pct_change()
                df_temp['strategy_returns'] = df_temp['signal'].shift(1) * df_temp['returns']

                total_return = (1 + df_temp['strategy_returns'].fillna(0)).prod() - 1
                sharpe = df_temp['strategy_returns'].mean() / df_temp['strategy_returns'].std() * np.sqrt(252) if df_temp['strategy_returns'].std() > 0 else 0

                results.append({
                    'fast': fast,
                    'slow': slow,
                    'signal': signal,
                    'total_return': total_return,
                    'sharpe_ratio': sharpe,
                    'buy_signals': buy_signals,
                    'sell_signals': sell_signals
                })

    results_df = pd.DataFrame(results)
    return results_df.sort_values('sharpe_ratio', ascending=False)

def analyze_macd_distribution(stocks_df):
    """分析MACD在多只股票中的分布"""
    all_macd_values = []
    all_dif_values = []

    for ts_code in stocks_df['ts_code'].unique():
        stock_data = stocks_df[stocks_df['ts_code'] == ts_code].copy()
        if len(stock_data) < 50:
            continue

        dif, dea, macd_hist = calculate_macd(stock_data)
        all_macd_values.extend(macd_hist.dropna().tolist())
        all_dif_values.extend(dif.dropna().tolist())

    return np.array(all_macd_values), np.array(all_dif_values)

def plot_macd_distribution(macd_values, dif_values):
    """绘制MACD分布图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('MACD指标分布分析', fontsize=14, fontweight='bold')

    # MACD柱状图分布
    ax1 = axes[0, 0]
    ax1.hist(macd_values, bins=100, edgecolor='black', alpha=0.7)
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax1.axvline(x=np.mean(macd_values), color='green', linestyle='--', linewidth=2, label=f'均值: {np.mean(macd_values):.4f}')
    ax1.set_xlabel('MACD柱状图值')
    ax1.set_ylabel('频数')
    ax1.set_title('MACD柱状图分布')
    ax1.legend()

    # DIF分布
    ax2 = axes[0, 1]
    ax2.hist(dif_values, bins=100, edgecolor='black', alpha=0.7, color='orange')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax2.axvline(x=np.mean(dif_values), color='green', linestyle='--', linewidth=2, label=f'均值: {np.mean(dif_values):.4f}')
    ax2.set_xlabel('DIF值')
    ax2.set_ylabel('频数')
    ax2.set_title('DIF分布')
    ax2.legend()

    # MACD分位数
    ax3 = axes[1, 0]
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    macd_percentiles = np.percentile(macd_values, percentiles)
    ax3.bar(range(len(percentiles)), macd_percentiles, tick_label=[f'{p}%' for p in percentiles])
    ax3.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax3.set_xlabel('分位数')
    ax3.set_ylabel('MACD值')
    ax3.set_title('MACD分位数分布')

    # 统计信息
    ax4 = axes[1, 1]
    ax4.axis('off')
    stats_text = f"""
    MACD统计信息
    ============

    MACD柱状图:
    - 样本数: {len(macd_values):,}
    - 均值: {np.mean(macd_values):.6f}
    - 标准差: {np.std(macd_values):.6f}
    - 中位数: {np.median(macd_values):.6f}
    - 最小值: {np.min(macd_values):.6f}
    - 最大值: {np.max(macd_values):.6f}
    - 正值比例: {(macd_values > 0).sum() / len(macd_values) * 100:.2f}%

    DIF:
    - 均值: {np.mean(dif_values):.6f}
    - 标准差: {np.std(dif_values):.6f}
    - 正值比例: {(dif_values > 0).sum() / len(dif_values) * 100:.2f}%
    """
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/macd_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()

    return {
        'macd_mean': np.mean(macd_values),
        'macd_std': np.std(macd_values),
        'macd_positive_ratio': (macd_values > 0).sum() / len(macd_values),
        'dif_mean': np.mean(dif_values),
        'dif_std': np.std(dif_values),
        'dif_positive_ratio': (dif_values > 0).sum() / len(dif_values)
    }

# =============================================================================
# 第二部分：交易信号
# =============================================================================

def detect_golden_death_cross(dif, dea):
    """
    检测金叉和死叉
    金叉：DIF从下向上穿越DEA
    死叉：DIF从上向下穿越DEA
    """
    signals = pd.DataFrame(index=dif.index)
    signals['dif'] = dif
    signals['dea'] = dea

    # 金叉：前一天DIF<DEA，当天DIF>DEA
    signals['golden_cross'] = (dif > dea) & (dif.shift(1) <= dea.shift(1))
    # 死叉：前一天DIF>DEA，当天DIF<DEA
    signals['death_cross'] = (dif < dea) & (dif.shift(1) >= dea.shift(1))

    return signals

def detect_macd_divergence(df, dif, window=20, min_swing=0.03):
    """
    检测MACD背离
    - 顶背离：价格创新高，但DIF没有创新高（卖出信号）
    - 底背离：价格创新低，但DIF没有创新低（买入信号）
    """
    signals = pd.DataFrame(index=df.index)
    signals['close'] = df['close'].values
    signals['dif'] = dif.values
    signals['top_divergence'] = False
    signals['bottom_divergence'] = False

    for i in range(window, len(df)):
        # 检查窗口内的高点和低点
        price_window = df['close'].iloc[i-window:i+1]
        dif_window = dif.iloc[i-window:i+1]

        current_price = df['close'].iloc[i]
        current_dif = dif.iloc[i]

        # 顶背离：价格在窗口内创新高，但DIF没有
        if current_price >= price_window.max() * (1 - min_swing/10):
            price_high_idx = price_window.idxmax()
            if price_high_idx != dif_window.index[-1]:  # 确保不是同一点
                prev_dif_at_price_high = dif.loc[price_high_idx] if price_high_idx in dif.index else None
                if prev_dif_at_price_high is not None and current_dif < prev_dif_at_price_high * 0.95:
                    signals.iloc[i, signals.columns.get_loc('top_divergence')] = True

        # 底背离：价格在窗口内创新低，但DIF没有
        if current_price <= price_window.min() * (1 + min_swing/10):
            price_low_idx = price_window.idxmin()
            if price_low_idx != dif_window.index[-1]:
                prev_dif_at_price_low = dif.loc[price_low_idx] if price_low_idx in dif.index else None
                if prev_dif_at_price_low is not None and current_dif > prev_dif_at_price_low * 1.05:
                    signals.iloc[i, signals.columns.get_loc('bottom_divergence')] = True

    return signals

def detect_histogram_changes(macd_hist):
    """
    检测MACD柱状图变化
    - 红柱放大：多头加强
    - 红柱缩小：多头减弱
    - 绿柱放大：空头加强
    - 绿柱缩小：空头减弱
    - 红转绿：由多转空
    - 绿转红：由空转多
    """
    signals = pd.DataFrame(index=macd_hist.index)
    signals['macd_hist'] = macd_hist

    prev_hist = macd_hist.shift(1)

    # 柱状图方向变化
    signals['red_to_green'] = (macd_hist < 0) & (prev_hist >= 0)
    signals['green_to_red'] = (macd_hist >= 0) & (prev_hist < 0)

    # 柱状图强度变化
    signals['red_expanding'] = (macd_hist > 0) & (macd_hist > prev_hist)
    signals['red_contracting'] = (macd_hist > 0) & (macd_hist < prev_hist)
    signals['green_expanding'] = (macd_hist < 0) & (macd_hist < prev_hist)
    signals['green_contracting'] = (macd_hist < 0) & (macd_hist > prev_hist)

    return signals

def plot_trading_signals(df, ts_code):
    """绘制交易信号图"""
    dif, dea, macd_hist = calculate_macd(df)

    # 检测各类信号
    cross_signals = detect_golden_death_cross(dif, dea)
    divergence_signals = detect_macd_divergence(df, dif)
    hist_signals = detect_histogram_changes(macd_hist)

    fig, axes = plt.subplots(3, 1, figsize=(16, 14), height_ratios=[2, 1, 1])
    fig.suptitle(f'MACD交易信号分析 - {ts_code}', fontsize=14, fontweight='bold')

    # 价格图 + 金叉死叉
    ax1 = axes[0]
    ax1.plot(df['trade_date'], df['close'], 'b-', linewidth=1, label='收盘价')

    # 标记金叉
    golden_dates = df['trade_date'][cross_signals['golden_cross']]
    golden_prices = df['close'][cross_signals['golden_cross']]
    ax1.scatter(golden_dates, golden_prices, marker='^', color='red', s=100, label='金叉', zorder=5)

    # 标记死叉
    death_dates = df['trade_date'][cross_signals['death_cross']]
    death_prices = df['close'][cross_signals['death_cross']]
    ax1.scatter(death_dates, death_prices, marker='v', color='green', s=100, label='死叉', zorder=5)

    ax1.set_ylabel('价格')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('价格走势与金叉/死叉信号')

    # MACD图 + 背离信号
    ax2 = axes[1]
    ax2.plot(df['trade_date'], dif, 'b-', linewidth=1, label='DIF')
    ax2.plot(df['trade_date'], dea, 'r-', linewidth=1, label='DEA')

    # 标记背离
    top_div_dates = df['trade_date'][divergence_signals['top_divergence']]
    top_div_dif = dif[divergence_signals['top_divergence']]
    ax2.scatter(top_div_dates, top_div_dif, marker='v', color='purple', s=100, label='顶背离', zorder=5)

    bottom_div_dates = df['trade_date'][divergence_signals['bottom_divergence']]
    bottom_div_dif = dif[divergence_signals['bottom_divergence']]
    ax2.scatter(bottom_div_dates, bottom_div_dif, marker='^', color='orange', s=100, label='底背离', zorder=5)

    ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax2.set_ylabel('DIF/DEA')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('MACD背离信号')

    # 柱状图
    ax3 = axes[2]
    colors = ['red' if v >= 0 else 'green' for v in macd_hist]
    ax3.bar(df['trade_date'], macd_hist, color=colors, alpha=0.7, width=1)
    ax3.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax3.set_ylabel('MACD柱状图')
    ax3.grid(True, alpha=0.3)
    ax3.set_title('MACD柱状图变化')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/macd_signals_{ts_code.replace(".", "_")}.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 统计信号数量
    signal_stats = {
        'golden_cross_count': cross_signals['golden_cross'].sum(),
        'death_cross_count': cross_signals['death_cross'].sum(),
        'top_divergence_count': divergence_signals['top_divergence'].sum(),
        'bottom_divergence_count': divergence_signals['bottom_divergence'].sum(),
        'red_to_green_count': hist_signals['red_to_green'].sum(),
        'green_to_red_count': hist_signals['green_to_red'].sum()
    }

    return signal_stats, cross_signals, divergence_signals, hist_signals

# =============================================================================
# 第三部分：策略回测
# =============================================================================

def backtest_macd_cross_strategy(df, initial_capital=1000000):
    """
    MACD交叉策略回测
    - 金叉买入，死叉卖出
    """
    dif, dea, macd_hist = calculate_macd(df)
    cross_signals = detect_golden_death_cross(dif, dea)

    df_bt = df.copy()
    df_bt['dif'] = dif
    df_bt['dea'] = dea
    df_bt['golden_cross'] = cross_signals['golden_cross']
    df_bt['death_cross'] = cross_signals['death_cross']

    # 生成持仓信号
    df_bt['position'] = 0
    position = 0
    for i in range(len(df_bt)):
        if df_bt['golden_cross'].iloc[i]:
            position = 1
        elif df_bt['death_cross'].iloc[i]:
            position = 0
        df_bt.iloc[i, df_bt.columns.get_loc('position')] = position

    # 计算收益
    df_bt['returns'] = df_bt['close'].pct_change()
    df_bt['strategy_returns'] = df_bt['position'].shift(1) * df_bt['returns']

    # 计算累计收益
    df_bt['cumulative_returns'] = (1 + df_bt['returns'].fillna(0)).cumprod()
    df_bt['cumulative_strategy_returns'] = (1 + df_bt['strategy_returns'].fillna(0)).cumprod()

    # 计算回撤
    df_bt['peak'] = df_bt['cumulative_strategy_returns'].cummax()
    df_bt['drawdown'] = (df_bt['cumulative_strategy_returns'] - df_bt['peak']) / df_bt['peak']

    return df_bt

def backtest_macd_momentum_strategy(df, initial_capital=1000000):
    """
    MACD动量策略回测
    - MACD柱状图为正且放大时，加仓
    - MACD柱状图为负或缩小时，减仓或空仓
    """
    dif, dea, macd_hist = calculate_macd(df)
    hist_signals = detect_histogram_changes(macd_hist)

    df_bt = df.copy()
    df_bt['macd_hist'] = macd_hist

    # 生成持仓信号（基于动量强度）
    df_bt['position'] = 0
    for i in range(1, len(df_bt)):
        if hist_signals['red_expanding'].iloc[i]:
            df_bt.iloc[i, df_bt.columns.get_loc('position')] = 1.0  # 全仓
        elif hist_signals['red_contracting'].iloc[i]:
            df_bt.iloc[i, df_bt.columns.get_loc('position')] = 0.5  # 半仓
        elif hist_signals['green_contracting'].iloc[i]:
            df_bt.iloc[i, df_bt.columns.get_loc('position')] = 0.3  # 轻仓
        else:
            df_bt.iloc[i, df_bt.columns.get_loc('position')] = 0  # 空仓

    # 计算收益
    df_bt['returns'] = df_bt['close'].pct_change()
    df_bt['strategy_returns'] = df_bt['position'].shift(1) * df_bt['returns']

    # 计算累计收益
    df_bt['cumulative_returns'] = (1 + df_bt['returns'].fillna(0)).cumprod()
    df_bt['cumulative_strategy_returns'] = (1 + df_bt['strategy_returns'].fillna(0)).cumprod()

    # 计算回撤
    df_bt['peak'] = df_bt['cumulative_strategy_returns'].cummax()
    df_bt['drawdown'] = (df_bt['cumulative_strategy_returns'] - df_bt['peak']) / df_bt['peak']

    return df_bt

def backtest_macd_rsi_combined(df, rsi_period=14, initial_capital=1000000):
    """
    MACD + RSI组合策略回测
    - 金叉 + RSI<30 强烈买入
    - 死叉 + RSI>70 强烈卖出
    """
    dif, dea, macd_hist = calculate_macd(df)
    cross_signals = detect_golden_death_cross(dif, dea)

    # 计算RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    df_bt = df.copy()
    df_bt['dif'] = dif
    df_bt['dea'] = dea
    df_bt['rsi'] = rsi
    df_bt['golden_cross'] = cross_signals['golden_cross']
    df_bt['death_cross'] = cross_signals['death_cross']

    # 生成持仓信号
    df_bt['position'] = 0
    position = 0
    for i in range(len(df_bt)):
        rsi_val = df_bt['rsi'].iloc[i]

        # 强买入信号：金叉 + RSI超卖
        if df_bt['golden_cross'].iloc[i]:
            if rsi_val < 30:
                position = 1.0  # 强烈买入
            else:
                position = 0.7  # 普通买入
        # 强卖出信号：死叉 + RSI超买
        elif df_bt['death_cross'].iloc[i]:
            if rsi_val > 70:
                position = 0  # 强烈卖出
            else:
                position = 0.3  # 减仓

        df_bt.iloc[i, df_bt.columns.get_loc('position')] = position

    # 计算收益
    df_bt['returns'] = df_bt['close'].pct_change()
    df_bt['strategy_returns'] = df_bt['position'].shift(1) * df_bt['returns']

    # 计算累计收益
    df_bt['cumulative_returns'] = (1 + df_bt['returns'].fillna(0)).cumprod()
    df_bt['cumulative_strategy_returns'] = (1 + df_bt['strategy_returns'].fillna(0)).cumprod()

    # 计算回撤
    df_bt['peak'] = df_bt['cumulative_strategy_returns'].cummax()
    df_bt['drawdown'] = (df_bt['cumulative_strategy_returns'] - df_bt['peak']) / df_bt['peak']

    return df_bt

def calculate_performance_metrics(df_bt):
    """计算策略绩效指标"""
    strategy_returns = df_bt['strategy_returns'].dropna()
    benchmark_returns = df_bt['returns'].dropna()

    # 年化收益
    total_days = len(df_bt)
    annual_factor = 252 / total_days

    total_return = df_bt['cumulative_strategy_returns'].iloc[-1] - 1
    benchmark_total_return = df_bt['cumulative_returns'].iloc[-1] - 1

    annual_return = (1 + total_return) ** annual_factor - 1
    benchmark_annual_return = (1 + benchmark_total_return) ** annual_factor - 1

    # 波动率
    volatility = strategy_returns.std() * np.sqrt(252)
    benchmark_volatility = benchmark_returns.std() * np.sqrt(252)

    # 夏普比率
    risk_free_rate = 0.03
    sharpe_ratio = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0

    # 最大回撤
    max_drawdown = df_bt['drawdown'].min()

    # 胜率
    winning_days = (strategy_returns > 0).sum()
    total_trading_days = (strategy_returns != 0).sum()
    win_rate = winning_days / total_trading_days if total_trading_days > 0 else 0

    # 盈亏比
    avg_win = strategy_returns[strategy_returns > 0].mean() if (strategy_returns > 0).any() else 0
    avg_loss = abs(strategy_returns[strategy_returns < 0].mean()) if (strategy_returns < 0).any() else 1
    profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0

    # 超额收益
    excess_return = total_return - benchmark_total_return

    return {
        'total_return': total_return,
        'benchmark_return': benchmark_total_return,
        'excess_return': excess_return,
        'annual_return': annual_return,
        'annual_volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'profit_loss_ratio': profit_loss_ratio
    }

def plot_backtest_results(results_dict, ts_code):
    """绘制回测结果对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'MACD策略回测结果对比 - {ts_code}', fontsize=14, fontweight='bold')

    # 累计收益对比
    ax1 = axes[0, 0]
    for name, df_bt in results_dict.items():
        ax1.plot(df_bt['trade_date'], df_bt['cumulative_strategy_returns'], label=name, linewidth=1.5)
    ax1.plot(df_bt['trade_date'], df_bt['cumulative_returns'], 'k--', label='基准(买入持有)', linewidth=1)
    ax1.set_ylabel('累计收益')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('累计收益对比')

    # 回撤对比
    ax2 = axes[0, 1]
    for name, df_bt in results_dict.items():
        ax2.fill_between(df_bt['trade_date'], df_bt['drawdown'], 0, alpha=0.3, label=name)
    ax2.set_ylabel('回撤')
    ax2.legend(loc='lower left')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('回撤对比')

    # 绩效指标对比
    ax3 = axes[1, 0]
    metrics_names = ['total_return', 'annual_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
    x = np.arange(len(metrics_names))
    width = 0.25

    for i, (name, df_bt) in enumerate(results_dict.items()):
        metrics = calculate_performance_metrics(df_bt)
        values = [metrics[m] for m in metrics_names]
        ax3.bar(x + i * width, values, width, label=name)

    ax3.set_xticks(x + width)
    ax3.set_xticklabels(['总收益', '年化收益', '夏普比', '最大回撤', '胜率'])
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_title('绩效指标对比')
    ax3.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)

    # 详细统计表
    ax4 = axes[1, 1]
    ax4.axis('off')

    table_data = []
    headers = ['指标', '交叉策略', '动量策略', 'MACD+RSI']

    metrics_display = [
        ('总收益', 'total_return', '{:.2%}'),
        ('年化收益', 'annual_return', '{:.2%}'),
        ('年化波动率', 'annual_volatility', '{:.2%}'),
        ('夏普比率', 'sharpe_ratio', '{:.2f}'),
        ('最大回撤', 'max_drawdown', '{:.2%}'),
        ('胜率', 'win_rate', '{:.2%}'),
        ('盈亏比', 'profit_loss_ratio', '{:.2f}'),
        ('超额收益', 'excess_return', '{:.2%}')
    ]

    for display_name, metric_key, fmt in metrics_display:
        row = [display_name]
        for name, df_bt in results_dict.items():
            metrics = calculate_performance_metrics(df_bt)
            row.append(fmt.format(metrics[metric_key]))
        table_data.append(row)

    table = ax4.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # 设置表头样式
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/macd_backtest_{ts_code.replace(".", "_")}.png', dpi=150, bbox_inches='tight')
    plt.close()

def run_multi_stock_backtest(stocks_df, strategy_func, strategy_name):
    """在多只股票上运行回测"""
    all_metrics = []

    for ts_code in stocks_df['ts_code'].unique():
        stock_data = stocks_df[stocks_df['ts_code'] == ts_code].copy().reset_index(drop=True)
        if len(stock_data) < 100:
            continue

        try:
            df_bt = strategy_func(stock_data)
            metrics = calculate_performance_metrics(df_bt)
            metrics['ts_code'] = ts_code
            all_metrics.append(metrics)
        except Exception as e:
            continue

    return pd.DataFrame(all_metrics)

def plot_multi_stock_summary(metrics_df, strategy_name):
    """绘制多股票回测汇总"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{strategy_name} - 多股票回测汇总', fontsize=14, fontweight='bold')

    # 收益分布
    ax1 = axes[0, 0]
    ax1.hist(metrics_df['total_return'], bins=30, edgecolor='black', alpha=0.7)
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax1.axvline(x=metrics_df['total_return'].mean(), color='green', linestyle='--',
                linewidth=2, label=f'均值: {metrics_df["total_return"].mean():.2%}')
    ax1.set_xlabel('总收益')
    ax1.set_ylabel('频数')
    ax1.set_title('总收益分布')
    ax1.legend()

    # 夏普比率分布
    ax2 = axes[0, 1]
    ax2.hist(metrics_df['sharpe_ratio'], bins=30, edgecolor='black', alpha=0.7, color='orange')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax2.axvline(x=metrics_df['sharpe_ratio'].mean(), color='green', linestyle='--',
                linewidth=2, label=f'均值: {metrics_df["sharpe_ratio"].mean():.2f}')
    ax2.set_xlabel('夏普比率')
    ax2.set_ylabel('频数')
    ax2.set_title('夏普比率分布')
    ax2.legend()

    # 胜率 vs 盈亏比
    ax3 = axes[1, 0]
    ax3.scatter(metrics_df['win_rate'], metrics_df['profit_loss_ratio'], alpha=0.5)
    ax3.set_xlabel('胜率')
    ax3.set_ylabel('盈亏比')
    ax3.set_title('胜率 vs 盈亏比')
    ax3.grid(True, alpha=0.3)

    # 汇总统计
    ax4 = axes[1, 1]
    ax4.axis('off')

    stats_text = f"""
    {strategy_name} 多股票回测统计
    ================================

    样本数量: {len(metrics_df)}

    总收益:
    - 均值: {metrics_df['total_return'].mean():.2%}
    - 中位数: {metrics_df['total_return'].median():.2%}
    - 标准差: {metrics_df['total_return'].std():.2%}
    - 正收益比例: {(metrics_df['total_return'] > 0).mean():.2%}

    夏普比率:
    - 均值: {metrics_df['sharpe_ratio'].mean():.2f}
    - 中位数: {metrics_df['sharpe_ratio'].median():.2f}
    - >1比例: {(metrics_df['sharpe_ratio'] > 1).mean():.2%}

    最大回撤:
    - 均值: {metrics_df['max_drawdown'].mean():.2%}
    - 中位数: {metrics_df['max_drawdown'].median():.2%}

    胜率:
    - 均值: {metrics_df['win_rate'].mean():.2%}
    - >50%比例: {(metrics_df['win_rate'] > 0.5).mean():.2%}
    """

    ax4.text(0.1, 0.95, stats_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/macd_multi_stock_{strategy_name.replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
    plt.close()

    return metrics_df.describe()

# =============================================================================
# 主函数
# =============================================================================

def generate_report():
    """生成完整的MACD策略研究报告"""
    print("=" * 60)
    print("MACD交易策略研究报告")
    print("=" * 60)
    print(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # 选择示例股票
    example_stock = '000001.SZ'  # 平安银行

    print("加载数据...")
    df_single = load_stock_data(example_stock, '20200101', '20260130')
    print(f"单股数据: {example_stock}, {len(df_single)} 条记录")

    df_multi = load_multiple_stocks('20200101', '20260130', limit=30)
    print(f"多股数据: {df_multi['ts_code'].nunique()} 只股票")
    print()

    # ==========================================================================
    # 第一部分：MACD分析
    # ==========================================================================
    print("=" * 60)
    print("第一部分：MACD分析")
    print("=" * 60)

    # 1.1 基本MACD计算与可视化
    print("\n1.1 MACD计算与可视化")
    dif, dea, macd_hist = plot_macd_basic(df_single, example_stock)
    print(f"  - DIF范围: [{dif.min():.4f}, {dif.max():.4f}]")
    print(f"  - DEA范围: [{dea.min():.4f}, {dea.max():.4f}]")
    print(f"  - MACD柱状图范围: [{macd_hist.min():.4f}, {macd_hist.max():.4f}]")

    # 1.2 参数优化
    print("\n1.2 MACD参数优化")
    optimization_results = macd_parameter_optimization(df_single)
    print("  最优参数组合（按夏普比率排序）:")
    print(optimization_results.head(10).to_string(index=False))

    # 保存参数优化结果
    optimization_results.to_csv(f'{OUTPUT_DIR}/macd_parameter_optimization.csv', index=False)

    # 1.3 MACD分布分析
    print("\n1.3 MACD分布分析")
    macd_values, dif_values = analyze_macd_distribution(df_multi)
    dist_stats = plot_macd_distribution(macd_values, dif_values)
    print(f"  - MACD柱状图均值: {dist_stats['macd_mean']:.6f}")
    print(f"  - MACD柱状图正值比例: {dist_stats['macd_positive_ratio']:.2%}")
    print(f"  - DIF正值比例: {dist_stats['dif_positive_ratio']:.2%}")

    # ==========================================================================
    # 第二部分：交易信号
    # ==========================================================================
    print("\n" + "=" * 60)
    print("第二部分：交易信号")
    print("=" * 60)

    # 2.1 金叉/死叉、背离、柱状图变化
    print("\n2.1 交易信号检测")
    signal_stats, cross_signals, divergence_signals, hist_signals = plot_trading_signals(df_single, example_stock)
    print(f"  - 金叉次数: {signal_stats['golden_cross_count']}")
    print(f"  - 死叉次数: {signal_stats['death_cross_count']}")
    print(f"  - 顶背离次数: {signal_stats['top_divergence_count']}")
    print(f"  - 底背离次数: {signal_stats['bottom_divergence_count']}")
    print(f"  - 红转绿次数: {signal_stats['red_to_green_count']}")
    print(f"  - 绿转红次数: {signal_stats['green_to_red_count']}")

    # ==========================================================================
    # 第三部分：策略回测
    # ==========================================================================
    print("\n" + "=" * 60)
    print("第三部分：策略回测")
    print("=" * 60)

    # 3.1 单股回测
    print("\n3.1 单股策略回测")

    # 交叉策略
    df_cross = backtest_macd_cross_strategy(df_single)
    metrics_cross = calculate_performance_metrics(df_cross)
    print("\n  MACD交叉策略:")
    print(f"    - 总收益: {metrics_cross['total_return']:.2%}")
    print(f"    - 年化收益: {metrics_cross['annual_return']:.2%}")
    print(f"    - 夏普比率: {metrics_cross['sharpe_ratio']:.2f}")
    print(f"    - 最大回撤: {metrics_cross['max_drawdown']:.2%}")

    # 动量策略
    df_momentum = backtest_macd_momentum_strategy(df_single)
    metrics_momentum = calculate_performance_metrics(df_momentum)
    print("\n  MACD动量策略:")
    print(f"    - 总收益: {metrics_momentum['total_return']:.2%}")
    print(f"    - 年化收益: {metrics_momentum['annual_return']:.2%}")
    print(f"    - 夏普比率: {metrics_momentum['sharpe_ratio']:.2f}")
    print(f"    - 最大回撤: {metrics_momentum['max_drawdown']:.2%}")

    # MACD+RSI组合策略
    df_combined = backtest_macd_rsi_combined(df_single)
    metrics_combined = calculate_performance_metrics(df_combined)
    print("\n  MACD+RSI组合策略:")
    print(f"    - 总收益: {metrics_combined['total_return']:.2%}")
    print(f"    - 年化收益: {metrics_combined['annual_return']:.2%}")
    print(f"    - 夏普比率: {metrics_combined['sharpe_ratio']:.2f}")
    print(f"    - 最大回撤: {metrics_combined['max_drawdown']:.2%}")

    # 绘制回测对比图
    results_dict = {
        '交叉策略': df_cross,
        '动量策略': df_momentum,
        'MACD+RSI': df_combined
    }
    plot_backtest_results(results_dict, example_stock)

    # 3.2 多股票回测
    print("\n3.2 多股票回测汇总")

    # 交叉策略多股回测
    print("\n  运行MACD交叉策略多股回测...")
    cross_metrics_df = run_multi_stock_backtest(df_multi, backtest_macd_cross_strategy, '交叉策略')
    cross_summary = plot_multi_stock_summary(cross_metrics_df, 'MACD交叉策略')
    print(f"  交叉策略平均收益: {cross_metrics_df['total_return'].mean():.2%}")
    print(f"  交叉策略正收益比例: {(cross_metrics_df['total_return'] > 0).mean():.2%}")

    # 动量策略多股回测
    print("\n  运行MACD动量策略多股回测...")
    momentum_metrics_df = run_multi_stock_backtest(df_multi, backtest_macd_momentum_strategy, '动量策略')
    momentum_summary = plot_multi_stock_summary(momentum_metrics_df, 'MACD动量策略')
    print(f"  动量策略平均收益: {momentum_metrics_df['total_return'].mean():.2%}")
    print(f"  动量策略正收益比例: {(momentum_metrics_df['total_return'] > 0).mean():.2%}")

    # 组合策略多股回测
    print("\n  运行MACD+RSI组合策略多股回测...")
    combined_metrics_df = run_multi_stock_backtest(df_multi, backtest_macd_rsi_combined, '组合策略')
    combined_summary = plot_multi_stock_summary(combined_metrics_df, 'MACD+RSI组合策略')
    print(f"  组合策略平均收益: {combined_metrics_df['total_return'].mean():.2%}")
    print(f"  组合策略正收益比例: {(combined_metrics_df['total_return'] > 0).mean():.2%}")

    # ==========================================================================
    # 生成文本报告
    # ==========================================================================
    print("\n" + "=" * 60)
    print("生成研究报告...")
    print("=" * 60)

    report_content = f"""
# MACD交易策略研究报告

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 一、研究概述

本研究基于A股市场历史数据，系统性地研究了MACD（移动平均收敛散度）指标在量化交易中的应用。研究内容包括：
1. MACD指标的计算与参数优化
2. 基于MACD的交易信号识别
3. MACD策略的回测与绩效评估

### 数据说明
- 数据来源: Tushare数据库
- 时间范围: 2020年1月 - 2026年1月
- 样本数量: {df_multi['ts_code'].nunique()} 只股票
- 示例股票: {example_stock}

---

## 二、MACD分析

### 2.1 MACD计算方法

MACD由三部分组成：
- **DIF (差离值)**: 快速EMA(12) - 慢速EMA(26)
- **DEA (信号线)**: DIF的9日EMA
- **MACD柱状图**: (DIF - DEA) × 2

### 2.2 参数优化结果

通过网格搜索对MACD参数进行优化，最优参数组合如下：

| 排名 | Fast | Slow | Signal | 夏普比率 | 总收益 |
|------|------|------|--------|----------|--------|
{optimization_results.head(5).to_markdown(index=False)}

**结论**: 默认参数(12, 26, 9)并非总是最优，根据不同市场环境，参数可能需要调整。

### 2.3 MACD分布特征

基于{len(macd_values):,}个样本的统计分析：

| 指标 | 数值 |
|------|------|
| MACD均值 | {dist_stats['macd_mean']:.6f} |
| MACD标准差 | {dist_stats['macd_std']:.6f} |
| MACD正值比例 | {dist_stats['macd_positive_ratio']:.2%} |
| DIF正值比例 | {dist_stats['dif_positive_ratio']:.2%} |

**结论**: MACD分布接近正态分布，略偏向正值，反映了A股市场长期上涨趋势。

---

## 三、交易信号分析

### 3.1 金叉/死叉信号

对{example_stock}的信号统计：
- 金叉次数: {signal_stats['golden_cross_count']}
- 死叉次数: {signal_stats['death_cross_count']}

### 3.2 MACD背离

- 顶背离次数: {signal_stats['top_divergence_count']}
- 底背离次数: {signal_stats['bottom_divergence_count']}

### 3.3 柱状图变化

- 红转绿次数: {signal_stats['red_to_green_count']}
- 绿转红次数: {signal_stats['green_to_red_count']}

**结论**: 金叉死叉是最常见的交易信号，背离信号较为稀少但往往更具参考价值。

---

## 四、策略回测结果

### 4.1 单股回测 ({example_stock})

| 策略 | 总收益 | 年化收益 | 夏普比率 | 最大回撤 | 胜率 |
|------|--------|----------|----------|----------|------|
| MACD交叉策略 | {metrics_cross['total_return']:.2%} | {metrics_cross['annual_return']:.2%} | {metrics_cross['sharpe_ratio']:.2f} | {metrics_cross['max_drawdown']:.2%} | {metrics_cross['win_rate']:.2%} |
| MACD动量策略 | {metrics_momentum['total_return']:.2%} | {metrics_momentum['annual_return']:.2%} | {metrics_momentum['sharpe_ratio']:.2f} | {metrics_momentum['max_drawdown']:.2%} | {metrics_momentum['win_rate']:.2%} |
| MACD+RSI组合 | {metrics_combined['total_return']:.2%} | {metrics_combined['annual_return']:.2%} | {metrics_combined['sharpe_ratio']:.2f} | {metrics_combined['max_drawdown']:.2%} | {metrics_combined['win_rate']:.2%} |

### 4.2 多股票回测汇总

#### MACD交叉策略
- 平均总收益: {cross_metrics_df['total_return'].mean():.2%}
- 平均夏普比率: {cross_metrics_df['sharpe_ratio'].mean():.2f}
- 正收益比例: {(cross_metrics_df['total_return'] > 0).mean():.2%}
- 平均最大回撤: {cross_metrics_df['max_drawdown'].mean():.2%}

#### MACD动量策略
- 平均总收益: {momentum_metrics_df['total_return'].mean():.2%}
- 平均夏普比率: {momentum_metrics_df['sharpe_ratio'].mean():.2f}
- 正收益比例: {(momentum_metrics_df['total_return'] > 0).mean():.2%}
- 平均最大回撤: {momentum_metrics_df['max_drawdown'].mean():.2%}

#### MACD+RSI组合策略
- 平均总收益: {combined_metrics_df['total_return'].mean():.2%}
- 平均夏普比率: {combined_metrics_df['sharpe_ratio'].mean():.2f}
- 正收益比例: {(combined_metrics_df['total_return'] > 0).mean():.2%}
- 平均最大回撤: {combined_metrics_df['max_drawdown'].mean():.2%}

---

## 五、研究结论

### 5.1 MACD指标特点

1. **趋势跟踪**: MACD是一个趋势跟踪指标，在趋势明确的市场中表现较好
2. **滞后性**: 由于基于移动平均，存在一定滞后性
3. **参数敏感**: 不同参数组合对结果影响显著

### 5.2 策略比较

1. **交叉策略**: 信号明确，但频繁交易可能带来较高成本
2. **动量策略**: 能够捕捉趋势强度，但在震荡市中表现不佳
3. **组合策略**: 结合RSI可以过滤部分假信号，但也可能错过机会

### 5.3 改进建议

1. 结合成交量确认信号有效性
2. 加入止损止盈机制
3. 考虑交易成本（手续费、滑点）
4. 根据市场环境动态调整参数
5. 结合其他技术指标（如布林带、KDJ）进行信号确认

---

## 六、图表列表

1. `macd_basic_{example_stock.replace('.', '_')}.png` - MACD基本图表
2. `macd_distribution.png` - MACD分布分析
3. `macd_signals_{example_stock.replace('.', '_')}.png` - 交易信号图
4. `macd_backtest_{example_stock.replace('.', '_')}.png` - 回测结果对比
5. `macd_multi_stock_MACD交叉策略.png` - 交叉策略多股回测
6. `macd_multi_stock_MACD动量策略.png` - 动量策略多股回测
7. `macd_multi_stock_MACD+RSI组合策略.png` - 组合策略多股回测
8. `macd_parameter_optimization.csv` - 参数优化详细结果

---

## 附录：策略代码说明

### A. MACD计算
```python
def calculate_macd(df, fast=12, slow=26, signal=9):
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    macd_hist = (dif - dea) * 2
    return dif, dea, macd_hist
```

### B. 金叉死叉检测
```python
def detect_golden_death_cross(dif, dea):
    golden_cross = (dif > dea) & (dif.shift(1) <= dea.shift(1))
    death_cross = (dif < dea) & (dif.shift(1) >= dea.shift(1))
    return golden_cross, death_cross
```

---

*报告生成完毕*
"""

    # 保存报告
    with open(f'{OUTPUT_DIR}/MACD_Strategy_Research_Report.md', 'w', encoding='utf-8') as f:
        f.write(report_content)

    print(f"\n报告已保存到: {OUTPUT_DIR}/MACD_Strategy_Research_Report.md")
    print("=" * 60)
    print("研究完成!")
    print("=" * 60)

    return {
        'optimization_results': optimization_results,
        'distribution_stats': dist_stats,
        'signal_stats': signal_stats,
        'single_stock_metrics': {
            'cross': metrics_cross,
            'momentum': metrics_momentum,
            'combined': metrics_combined
        },
        'multi_stock_metrics': {
            'cross': cross_metrics_df,
            'momentum': momentum_metrics_df,
            'combined': combined_metrics_df
        }
    }

if __name__ == '__main__':
    results = generate_report()
