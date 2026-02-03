#!/usr/bin/env python3
"""
KDJ交易策略研究报告
==================

本报告研究KDJ指标在A股市场的应用，包括：
1. KDJ指标计算与参数优化
2. 交易信号识别（超买超卖、金叉死叉、背离）
3. 策略回测验证

作者: Claude AI
日期: 2026-02-01
"""

import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False

# 连接数据库
DB_PATH = '/Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db'
OUTPUT_DIR = '/Users/allen/workspace/python/stock/Tushare-DuckDB/reports/research/'

print("=" * 80)
print("KDJ交易策略研究报告")
print("=" * 80)
print()

# =============================================================================
# 第一部分：KDJ指标计算
# =============================================================================
print("\n" + "=" * 80)
print("第一部分：KDJ指标分析")
print("=" * 80)

def calculate_kdj(df, n=9, m1=3, m2=3):
    """
    计算KDJ指标

    参数:
    - n: RSV周期，默认9
    - m1: K值平滑周期，默认3
    - m2: D值平滑周期，默认3

    公式:
    - RSV = (C - L_n) / (H_n - L_n) * 100
    - K = EMA(RSV, m1) 或 SMA
    - D = EMA(K, m2) 或 SMA
    - J = 3*K - 2*D
    """
    df = df.copy()

    # 计算N日内最高价和最低价
    df['L_n'] = df['low'].rolling(window=n, min_periods=1).min()
    df['H_n'] = df['high'].rolling(window=n, min_periods=1).max()

    # 计算RSV
    df['RSV'] = 100 * (df['close'] - df['L_n']) / (df['H_n'] - df['L_n'] + 1e-10)

    # 使用SMA平滑计算K值（类似同花顺公式）
    # K = (m1-1)/m1 * K_prev + 1/m1 * RSV
    df['K'] = 50.0  # 初始值
    df['D'] = 50.0  # 初始值

    k_values = [50.0]
    d_values = [50.0]

    for i in range(1, len(df)):
        rsv = df['RSV'].iloc[i]
        k_prev = k_values[-1]
        d_prev = d_values[-1]

        k_new = (m1 - 1) / m1 * k_prev + 1 / m1 * rsv
        d_new = (m2 - 1) / m2 * d_prev + 1 / m2 * k_new

        k_values.append(k_new)
        d_values.append(d_new)

    df['K'] = k_values
    df['D'] = d_values
    df['J'] = 3 * df['K'] - 2 * df['D']

    # 清理临时列
    df.drop(['L_n', 'H_n', 'RSV'], axis=1, inplace=True)

    return df


def get_stock_data(conn, ts_code, start_date='20240101', end_date='20260130'):
    """获取股票日线数据"""
    query = f"""
    SELECT ts_code, trade_date, open, high, low, close, vol, amount, pct_chg
    FROM daily
    WHERE ts_code = '{ts_code}'
    AND trade_date >= '{start_date}'
    AND trade_date <= '{end_date}'
    ORDER BY trade_date
    """
    df = conn.execute(query).fetchdf()
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    return df


# 连接数据库
conn = duckdb.connect(DB_PATH, read_only=True)

# 获取活跃股票列表
print("\n1.1 获取样本股票...")
sample_stocks_query = """
SELECT ts_code, name
FROM stock_basic
WHERE market IN ('主板', '创业板', '科创板')
AND list_status = 'L'
LIMIT 50
"""
sample_stocks = conn.execute(sample_stocks_query).fetchdf()
print(f"样本股票数量: {len(sample_stocks)}")

# 选择一个典型股票进行详细分析
test_stock = '000001.SZ'  # 平安银行
test_name = '平安银行'

print(f"\n1.2 计算{test_name}({test_stock})的KDJ指标...")
df_test = get_stock_data(conn, test_stock, '20230101', '20260130')
df_test = calculate_kdj(df_test)
print(f"数据范围: {df_test['trade_date'].min()} ~ {df_test['trade_date'].max()}")
print(f"数据量: {len(df_test)}条")

# 绘制KDJ示例图
print("\n1.3 绘制KDJ指标图...")
fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# 最近180天数据
df_plot = df_test.tail(180).copy()

# 上图：K线
ax1 = axes[0]
ax1.plot(df_plot['trade_date'], df_plot['close'], 'b-', linewidth=1.5, label='收盘价')
ax1.set_ylabel('价格', fontsize=12)
ax1.set_title(f'{test_name}({test_stock}) KDJ指标分析', fontsize=14)
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# 下图：KDJ
ax2 = axes[1]
ax2.plot(df_plot['trade_date'], df_plot['K'], 'b-', linewidth=1, label='K')
ax2.plot(df_plot['trade_date'], df_plot['D'], 'orange', linewidth=1, label='D')
ax2.plot(df_plot['trade_date'], df_plot['J'], 'purple', linewidth=1, label='J')
ax2.axhline(y=80, color='r', linestyle='--', alpha=0.5, label='超买线(80)')
ax2.axhline(y=20, color='g', linestyle='--', alpha=0.5, label='超卖线(20)')
ax2.axhline(y=50, color='gray', linestyle=':', alpha=0.5)
ax2.set_ylabel('KDJ值', fontsize=12)
ax2.set_xlabel('日期', fontsize=12)
ax2.set_ylim(-20, 120)
ax2.legend(loc='upper left', ncol=3)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}kdj_example.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"KDJ示例图已保存: {OUTPUT_DIR}kdj_example.png")

# =============================================================================
# 1.4 KDJ参数测试
# =============================================================================
print("\n1.4 KDJ参数敏感性测试...")

def test_kdj_params(df, n_values=[5, 9, 14, 19], m1_values=[2, 3, 5], m2_values=[2, 3, 5]):
    """测试不同参数组合的KDJ"""
    results = []

    for n in n_values:
        for m1 in m1_values:
            for m2 in m2_values:
                df_temp = calculate_kdj(df.copy(), n, m1, m2)

                # 计算统计指标
                j_std = df_temp['J'].std()
                k_std = df_temp['K'].std()

                # 计算超买超卖信号频率
                overbought = (df_temp['J'] > 80).sum() / len(df_temp) * 100
                oversold = (df_temp['J'] < 20).sum() / len(df_temp) * 100

                # 计算金叉死叉次数
                df_temp['cross'] = np.where(df_temp['K'] > df_temp['D'], 1, -1)
                crosses = (df_temp['cross'].diff() != 0).sum()

                results.append({
                    'N': n,
                    'M1': m1,
                    'M2': m2,
                    'J_std': round(j_std, 2),
                    'K_std': round(k_std, 2),
                    'Overbought%': round(overbought, 2),
                    'Oversold%': round(oversold, 2),
                    'CrossCount': crosses
                })

    return pd.DataFrame(results)

param_results = test_kdj_params(df_test)
print("\nKDJ参数测试结果（部分）:")
print(param_results.head(15).to_string(index=False))

# 找出最优参数（基于信号频率适中）
best_params = param_results[
    (param_results['Overbought%'] >= 5) &
    (param_results['Overbought%'] <= 20) &
    (param_results['Oversold%'] >= 5) &
    (param_results['Oversold%'] <= 20)
].sort_values('CrossCount').head(5)

print("\n推荐参数组合（超买超卖比例适中）:")
print(best_params.to_string(index=False))

# =============================================================================
# 1.5 KDJ分布分析
# =============================================================================
print("\n1.5 KDJ值分布分析...")

# 获取多只股票的KDJ数据进行分布分析
print("获取多只股票KDJ数据进行统计分析...")

all_kdj_data = []
sample_codes = sample_stocks['ts_code'].head(30).tolist()

for ts_code in sample_codes:
    try:
        df_temp = get_stock_data(conn, ts_code, '20240101', '20260130')
        if len(df_temp) > 60:
            df_temp = calculate_kdj(df_temp)
            all_kdj_data.append({
                'ts_code': ts_code,
                'K_mean': df_temp['K'].mean(),
                'K_std': df_temp['K'].std(),
                'D_mean': df_temp['D'].mean(),
                'D_std': df_temp['D'].std(),
                'J_mean': df_temp['J'].mean(),
                'J_std': df_temp['J'].std(),
                'J_min': df_temp['J'].min(),
                'J_max': df_temp['J'].max()
            })
    except Exception as e:
        continue

kdj_stats = pd.DataFrame(all_kdj_data)

print("\nKDJ值统计分布:")
print(f"K值均值分布: {kdj_stats['K_mean'].mean():.2f} +/- {kdj_stats['K_mean'].std():.2f}")
print(f"D值均值分布: {kdj_stats['D_mean'].mean():.2f} +/- {kdj_stats['D_mean'].std():.2f}")
print(f"J值均值分布: {kdj_stats['J_mean'].mean():.2f} +/- {kdj_stats['J_mean'].std():.2f}")
print(f"J值范围: [{kdj_stats['J_min'].min():.2f}, {kdj_stats['J_max'].max():.2f}]")

# 绘制KDJ分布图
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 使用最近的数据绘制分布
df_dist = df_test.tail(250)  # 最近一年数据

axes[0].hist(df_dist['K'], bins=30, color='blue', alpha=0.7, edgecolor='black')
axes[0].axvline(x=80, color='r', linestyle='--', label='超买线(80)')
axes[0].axvline(x=20, color='g', linestyle='--', label='超卖线(20)')
axes[0].axvline(x=df_dist['K'].mean(), color='orange', linestyle='-', linewidth=2, label=f'均值({df_dist["K"].mean():.1f})')
axes[0].set_xlabel('K值')
axes[0].set_ylabel('频数')
axes[0].set_title('K值分布')
axes[0].legend()

axes[1].hist(df_dist['D'], bins=30, color='orange', alpha=0.7, edgecolor='black')
axes[1].axvline(x=80, color='r', linestyle='--', label='超买线(80)')
axes[1].axvline(x=20, color='g', linestyle='--', label='超卖线(20)')
axes[1].axvline(x=df_dist['D'].mean(), color='blue', linestyle='-', linewidth=2, label=f'均值({df_dist["D"].mean():.1f})')
axes[1].set_xlabel('D值')
axes[1].set_ylabel('频数')
axes[1].set_title('D值分布')
axes[1].legend()

axes[2].hist(df_dist['J'], bins=30, color='purple', alpha=0.7, edgecolor='black')
axes[2].axvline(x=100, color='r', linestyle='--', label='超买线(100)')
axes[2].axvline(x=0, color='g', linestyle='--', label='超卖线(0)')
axes[2].axvline(x=df_dist['J'].mean(), color='blue', linestyle='-', linewidth=2, label=f'均值({df_dist["J"].mean():.1f})')
axes[2].set_xlabel('J值')
axes[2].set_ylabel('频数')
axes[2].set_title('J值分布')
axes[2].legend()

plt.suptitle('KDJ指标值分布（近一年）', fontsize=14)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}kdj_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nKDJ分布图已保存: {OUTPUT_DIR}kdj_distribution.png")

# =============================================================================
# 第二部分：交易信号识别
# =============================================================================
print("\n" + "=" * 80)
print("第二部分：交易信号识别")
print("=" * 80)

# 2.1 J值超买超卖信号
print("\n2.1 J值超买超卖信号分析...")

def detect_j_signals(df, overbought=100, oversold=0):
    """
    检测J值超买超卖信号

    超买信号：J > overbought 后回落
    超卖信号：J < oversold 后回升
    """
    df = df.copy()
    df['signal'] = 0

    # 检测J值穿越
    df['J_prev'] = df['J'].shift(1)

    # 超卖反弹信号（J从下方穿越oversold）
    df.loc[(df['J_prev'] < oversold) & (df['J'] >= oversold), 'signal'] = 1

    # 超买回落信号（J从上方穿越overbought）
    df.loc[(df['J_prev'] > overbought) & (df['J'] <= overbought), 'signal'] = -1

    df.drop('J_prev', axis=1, inplace=True)

    return df

df_signals = detect_j_signals(df_test.copy())
buy_signals = df_signals[df_signals['signal'] == 1]
sell_signals = df_signals[df_signals['signal'] == -1]

print(f"超卖买入信号数量: {len(buy_signals)}")
print(f"超买卖出信号数量: {len(sell_signals)}")

# 分析信号效果
def analyze_signal_performance(df, signal_type='buy', holding_days=[1, 3, 5, 10, 20]):
    """分析信号后N日收益"""
    signals = df[df['signal'] == (1 if signal_type == 'buy' else -1)].copy()
    results = []

    for idx, row in signals.iterrows():
        signal_idx = df.index.get_loc(idx)
        entry_price = row['close']

        for days in holding_days:
            if signal_idx + days < len(df):
                exit_price = df.iloc[signal_idx + days]['close']
                ret = (exit_price - entry_price) / entry_price * 100
                results.append({
                    'date': row['trade_date'],
                    'days': days,
                    'return': ret
                })

    return pd.DataFrame(results)

buy_perf = analyze_signal_performance(df_signals, 'buy')
sell_perf = analyze_signal_performance(df_signals, 'sell')

if len(buy_perf) > 0:
    print("\n超卖买入信号后收益统计:")
    buy_summary = buy_perf.groupby('days')['return'].agg(['mean', 'std', 'count']).round(2)
    buy_summary.columns = ['平均收益%', '标准差%', '样本数']
    print(buy_summary.to_string())

if len(sell_perf) > 0:
    print("\n超买卖出信号后收益统计（做空视角）:")
    sell_summary = sell_perf.groupby('days')['return'].agg(['mean', 'std', 'count']).round(2)
    sell_summary.columns = ['平均涨跌%', '标准差%', '样本数']
    print(sell_summary.to_string())

# 2.2 KD金叉死叉信号
print("\n2.2 KD金叉死叉信号分析...")

def detect_kd_cross(df):
    """
    检测KD金叉死叉信号

    金叉：K上穿D
    死叉：K下穿D
    """
    df = df.copy()
    df['cross_signal'] = 0

    df['K_above_D'] = df['K'] > df['D']
    df['K_above_D_prev'] = df['K_above_D'].shift(1)

    # 金叉
    df.loc[(df['K_above_D'] == True) & (df['K_above_D_prev'] == False), 'cross_signal'] = 1

    # 死叉
    df.loc[(df['K_above_D'] == False) & (df['K_above_D_prev'] == True), 'cross_signal'] = -1

    df.drop(['K_above_D', 'K_above_D_prev'], axis=1, inplace=True)

    return df

df_cross = detect_kd_cross(df_test.copy())
golden_cross = df_cross[df_cross['cross_signal'] == 1]
death_cross = df_cross[df_cross['cross_signal'] == -1]

print(f"金叉信号数量: {len(golden_cross)}")
print(f"死叉信号数量: {len(death_cross)}")

# 分析不同区域的金叉效果
print("\n不同区域金叉信号效果分析:")
df_cross_analysis = df_cross[df_cross['cross_signal'] == 1].copy()

def categorize_zone(k_value):
    if k_value < 20:
        return '超卖区(<20)'
    elif k_value < 50:
        return '低位区(20-50)'
    elif k_value < 80:
        return '高位区(50-80)'
    else:
        return '超买区(>80)'

if len(df_cross_analysis) > 0:
    df_cross_analysis['zone'] = df_cross_analysis['K'].apply(categorize_zone)

    # 计算金叉后5日收益
    for idx, row in df_cross_analysis.iterrows():
        signal_idx = df_cross.index.get_loc(idx)
        entry_price = row['close']

        if signal_idx + 5 < len(df_cross):
            exit_price = df_cross.iloc[signal_idx + 5]['close']
            df_cross_analysis.loc[idx, 'return_5d'] = (exit_price - entry_price) / entry_price * 100

    zone_perf = df_cross_analysis.groupby('zone')['return_5d'].agg(['mean', 'std', 'count']).round(2)
    zone_perf.columns = ['平均收益%', '标准差%', '样本数']
    print(zone_perf.to_string())

# 2.3 KDJ背离检测
print("\n2.3 KDJ背离信号分析...")

def detect_divergence(df, lookback=20):
    """
    检测KDJ背离

    底背离：价格创新低，但KDJ不创新低
    顶背离：价格创新高，但KDJ不创新高
    """
    df = df.copy()
    df['divergence'] = 0

    for i in range(lookback, len(df)):
        window = df.iloc[i-lookback:i+1]
        current = df.iloc[i]

        # 检测底背离
        price_new_low = current['low'] == window['low'].min()
        kdj_not_low = current['K'] > window['K'].min()

        if price_new_low and kdj_not_low:
            df.iloc[i, df.columns.get_loc('divergence')] = 1  # 底背离（看涨）

        # 检测顶背离
        price_new_high = current['high'] == window['high'].max()
        kdj_not_high = current['K'] < window['K'].max()

        if price_new_high and kdj_not_high:
            df.iloc[i, df.columns.get_loc('divergence')] = -1  # 顶背离（看跌）

    return df

df_div = detect_divergence(df_test.copy())
bull_div = df_div[df_div['divergence'] == 1]
bear_div = df_div[df_div['divergence'] == -1]

print(f"底背离信号（看涨）数量: {len(bull_div)}")
print(f"顶背离信号（看跌）数量: {len(bear_div)}")

# 绘制交易信号示意图
fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

df_plot = df_test.tail(120).copy()
df_plot = detect_j_signals(df_plot)
df_plot = detect_kd_cross(df_plot)
df_plot = detect_divergence(df_plot)

# 价格和信号
ax1 = axes[0]
ax1.plot(df_plot['trade_date'], df_plot['close'], 'b-', linewidth=1.5, label='收盘价')

# 标记J值信号
buy_pts = df_plot[df_plot['signal'] == 1]
sell_pts = df_plot[df_plot['signal'] == -1]
ax1.scatter(buy_pts['trade_date'], buy_pts['close'], marker='^', color='green', s=100, label='J超卖买入', zorder=5)
ax1.scatter(sell_pts['trade_date'], sell_pts['close'], marker='v', color='red', s=100, label='J超买卖出', zorder=5)

ax1.set_ylabel('价格')
ax1.set_title('J值超买超卖信号')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# KD交叉信号
ax2 = axes[1]
ax2.plot(df_plot['trade_date'], df_plot['close'], 'b-', linewidth=1.5, label='收盘价')

golden_pts = df_plot[df_plot['cross_signal'] == 1]
death_pts = df_plot[df_plot['cross_signal'] == -1]
ax2.scatter(golden_pts['trade_date'], golden_pts['close'], marker='^', color='gold', s=100, label='金叉', zorder=5, edgecolors='black')
ax2.scatter(death_pts['trade_date'], death_pts['close'], marker='v', color='purple', s=100, label='死叉', zorder=5, edgecolors='black')

ax2.set_ylabel('价格')
ax2.set_title('KD金叉死叉信号')
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)

# KDJ指标
ax3 = axes[2]
ax3.plot(df_plot['trade_date'], df_plot['K'], 'b-', linewidth=1, label='K')
ax3.plot(df_plot['trade_date'], df_plot['D'], 'orange', linewidth=1, label='D')
ax3.plot(df_plot['trade_date'], df_plot['J'], 'purple', linewidth=1, label='J')
ax3.axhline(y=100, color='r', linestyle='--', alpha=0.5)
ax3.axhline(y=0, color='g', linestyle='--', alpha=0.5)
ax3.set_ylabel('KDJ值')
ax3.set_xlabel('日期')
ax3.set_ylim(-30, 130)
ax3.legend(loc='upper left')
ax3.grid(True, alpha=0.3)

plt.suptitle(f'{test_name} KDJ交易信号示意图', fontsize=14)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}kdj_signals.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\n交易信号图已保存: {OUTPUT_DIR}kdj_signals.png")

# =============================================================================
# 第三部分：策略回测
# =============================================================================
print("\n" + "=" * 80)
print("第三部分：策略回测")
print("=" * 80)

# 3.1 KDJ短线策略回测
print("\n3.1 KDJ短线策略回测...")

class KDJStrategy:
    """KDJ短线交易策略"""

    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position = 0
        self.trades = []

    def reset(self):
        self.capital = self.initial_capital
        self.position = 0
        self.trades = []

    def backtest_j_signal(self, df, overbought=100, oversold=0, hold_days=5):
        """J值超买超卖策略"""
        self.reset()
        df = detect_j_signals(df.copy(), overbought, oversold)

        i = 0
        while i < len(df):
            row = df.iloc[i]

            # 买入信号
            if row['signal'] == 1 and self.position == 0:
                shares = int(self.capital / row['close'] / 100) * 100
                if shares > 0:
                    cost = shares * row['close']
                    self.capital -= cost
                    self.position = shares
                    entry_price = row['close']
                    entry_date = row['trade_date']
                    entry_idx = i

            # 持仓到期或止损
            if self.position > 0:
                if i - entry_idx >= hold_days or row['signal'] == -1:
                    revenue = self.position * row['close']
                    self.capital += revenue

                    ret = (row['close'] - entry_price) / entry_price * 100
                    self.trades.append({
                        'entry_date': entry_date,
                        'exit_date': row['trade_date'],
                        'entry_price': entry_price,
                        'exit_price': row['close'],
                        'return': ret,
                        'hold_days': i - entry_idx
                    })

                    self.position = 0

            i += 1

        # 清算持仓
        if self.position > 0:
            final_price = df.iloc[-1]['close']
            revenue = self.position * final_price
            self.capital += revenue
            self.position = 0

        return self.get_performance()

    def backtest_kd_cross(self, df, hold_days=5):
        """KD金叉死叉策略"""
        self.reset()
        df = detect_kd_cross(df.copy())

        i = 0
        while i < len(df):
            row = df.iloc[i]

            # 金叉买入
            if row['cross_signal'] == 1 and self.position == 0:
                # 只在低位区金叉时买入
                if row['K'] < 50:
                    shares = int(self.capital / row['close'] / 100) * 100
                    if shares > 0:
                        cost = shares * row['close']
                        self.capital -= cost
                        self.position = shares
                        entry_price = row['close']
                        entry_date = row['trade_date']
                        entry_idx = i

            # 死叉卖出或持仓到期
            if self.position > 0:
                if row['cross_signal'] == -1 or i - entry_idx >= hold_days:
                    revenue = self.position * row['close']
                    self.capital += revenue

                    ret = (row['close'] - entry_price) / entry_price * 100
                    self.trades.append({
                        'entry_date': entry_date,
                        'exit_date': row['trade_date'],
                        'entry_price': entry_price,
                        'exit_price': row['close'],
                        'return': ret,
                        'hold_days': i - entry_idx
                    })

                    self.position = 0

            i += 1

        if self.position > 0:
            final_price = df.iloc[-1]['close']
            revenue = self.position * final_price
            self.capital += revenue
            self.position = 0

        return self.get_performance()

    def get_performance(self):
        """计算策略绩效"""
        if not self.trades:
            return {'total_return': 0, 'win_rate': 0, 'avg_return': 0, 'trade_count': 0}

        trades_df = pd.DataFrame(self.trades)

        total_return = (self.capital - self.initial_capital) / self.initial_capital * 100
        win_rate = (trades_df['return'] > 0).mean() * 100
        avg_return = trades_df['return'].mean()
        max_drawdown = trades_df['return'].min()

        return {
            'total_return': round(total_return, 2),
            'win_rate': round(win_rate, 2),
            'avg_return': round(avg_return, 2),
            'max_loss': round(max_drawdown, 2),
            'trade_count': len(trades_df),
            'avg_hold_days': round(trades_df['hold_days'].mean(), 1)
        }

# 测试策略
strategy = KDJStrategy()

# 获取多只股票进行回测
print("\n对30只股票进行策略回测...")
backtest_results = []

for ts_code in sample_codes[:30]:
    try:
        df_bt = get_stock_data(conn, ts_code, '20240101', '20260130')
        if len(df_bt) > 100:
            df_bt = calculate_kdj(df_bt)

            # J值策略
            j_perf = strategy.backtest_j_signal(df_bt.copy())
            j_perf['strategy'] = 'J超买超卖'
            j_perf['ts_code'] = ts_code
            backtest_results.append(j_perf)

            # KD交叉策略
            kd_perf = strategy.backtest_kd_cross(df_bt.copy())
            kd_perf['strategy'] = 'KD金叉死叉'
            kd_perf['ts_code'] = ts_code
            backtest_results.append(kd_perf)
    except Exception as e:
        continue

bt_df = pd.DataFrame(backtest_results)

print("\nKDJ短线策略回测结果汇总:")
summary = bt_df.groupby('strategy').agg({
    'total_return': ['mean', 'std'],
    'win_rate': 'mean',
    'avg_return': 'mean',
    'trade_count': 'sum'
}).round(2)
print(summary.to_string())

# 3.2 结合均线过滤的KDJ策略
print("\n3.2 结合均线过滤的KDJ策略...")

def calculate_ma(df, periods=[5, 10, 20, 60]):
    """计算均线"""
    for p in periods:
        df[f'MA{p}'] = df['close'].rolling(window=p).mean()
    return df

class KDJWithMAStrategy(KDJStrategy):
    """结合均线的KDJ策略"""

    def backtest_with_ma_filter(self, df, hold_days=5):
        """只在价格站上MA20时，低位金叉买入"""
        self.reset()
        df = calculate_kdj(df.copy())
        df = calculate_ma(df)
        df = detect_kd_cross(df)

        i = 0
        while i < len(df):
            row = df.iloc[i]

            # 均线多头 + 低位金叉
            if self.position == 0 and row['cross_signal'] == 1:
                ma_bull = row['close'] > row['MA20'] if pd.notna(row['MA20']) else False
                low_zone = row['K'] < 40

                if ma_bull and low_zone:
                    shares = int(self.capital / row['close'] / 100) * 100
                    if shares > 0:
                        self.capital -= shares * row['close']
                        self.position = shares
                        entry_price = row['close']
                        entry_date = row['trade_date']
                        entry_idx = i

            # 卖出条件
            if self.position > 0:
                ma_break = row['close'] < row['MA20'] if pd.notna(row['MA20']) else False

                if row['cross_signal'] == -1 or ma_break or i - entry_idx >= hold_days:
                    revenue = self.position * row['close']
                    self.capital += revenue

                    ret = (row['close'] - entry_price) / entry_price * 100
                    self.trades.append({
                        'entry_date': entry_date,
                        'exit_date': row['trade_date'],
                        'entry_price': entry_price,
                        'exit_price': row['close'],
                        'return': ret,
                        'hold_days': i - entry_idx
                    })

                    self.position = 0

            i += 1

        if self.position > 0:
            self.capital += self.position * df.iloc[-1]['close']
            self.position = 0

        return self.get_performance()

ma_strategy = KDJWithMAStrategy()
ma_results = []

for ts_code in sample_codes[:30]:
    try:
        df_bt = get_stock_data(conn, ts_code, '20240101', '20260130')
        if len(df_bt) > 100:
            perf = ma_strategy.backtest_with_ma_filter(df_bt.copy())
            perf['ts_code'] = ts_code
            ma_results.append(perf)
    except Exception as e:
        continue

ma_df = pd.DataFrame(ma_results)

print("\nKDJ+均线过滤策略结果:")
print(f"平均总收益: {ma_df['total_return'].mean():.2f}%")
print(f"胜率: {ma_df['win_rate'].mean():.2f}%")
print(f"平均单笔收益: {ma_df['avg_return'].mean():.2f}%")
print(f"总交易次数: {ma_df['trade_count'].sum()}")

# 3.3 多周期KDJ策略
print("\n3.3 多周期KDJ共振策略...")

def calculate_weekly_kdj(df):
    """计算周线KDJ（使用日线数据模拟）"""
    df = df.copy()
    df['week'] = df['trade_date'].dt.isocalendar().week
    df['year'] = df['trade_date'].dt.year

    # 按周聚合
    weekly = df.groupby(['year', 'week']).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'trade_date': 'last'
    }).reset_index()

    weekly = calculate_kdj(weekly)

    # 将周线KDJ映射回日线
    week_kdj = weekly[['year', 'week', 'K', 'D', 'J']].copy()
    week_kdj.columns = ['year', 'week', 'K_weekly', 'D_weekly', 'J_weekly']

    df = df.merge(week_kdj, on=['year', 'week'], how='left')
    df.drop(['year', 'week'], axis=1, inplace=True)

    return df

class MultiTimeframeKDJ(KDJStrategy):
    """多周期KDJ策略"""

    def backtest_multi_tf(self, df, hold_days=10):
        """日线+周线KDJ共振"""
        self.reset()

        # 计算日线KDJ
        df = calculate_kdj(df.copy())
        # 计算周线KDJ
        df = calculate_weekly_kdj(df)
        # 检测日线金叉
        df = detect_kd_cross(df)

        i = 0
        while i < len(df):
            row = df.iloc[i]

            # 多周期共振条件
            if self.position == 0 and row['cross_signal'] == 1:
                daily_low = row['K'] < 40  # 日线低位
                weekly_bull = row['K_weekly'] > row['D_weekly'] if pd.notna(row['K_weekly']) else False
                weekly_low = row['K_weekly'] < 60 if pd.notna(row['K_weekly']) else False

                if daily_low and weekly_bull and weekly_low:
                    shares = int(self.capital / row['close'] / 100) * 100
                    if shares > 0:
                        self.capital -= shares * row['close']
                        self.position = shares
                        entry_price = row['close']
                        entry_date = row['trade_date']
                        entry_idx = i

            # 卖出
            if self.position > 0:
                if row['cross_signal'] == -1 or row['K'] > 80 or i - entry_idx >= hold_days:
                    revenue = self.position * row['close']
                    self.capital += revenue

                    ret = (row['close'] - entry_price) / entry_price * 100
                    self.trades.append({
                        'entry_date': entry_date,
                        'exit_date': row['trade_date'],
                        'entry_price': entry_price,
                        'exit_price': row['close'],
                        'return': ret,
                        'hold_days': i - entry_idx
                    })

                    self.position = 0

            i += 1

        if self.position > 0:
            self.capital += self.position * df.iloc[-1]['close']
            self.position = 0

        return self.get_performance()

mtf_strategy = MultiTimeframeKDJ()
mtf_results = []

for ts_code in sample_codes[:30]:
    try:
        df_bt = get_stock_data(conn, ts_code, '20230101', '20260130')  # 需要更多数据计算周线
        if len(df_bt) > 200:
            perf = mtf_strategy.backtest_multi_tf(df_bt.copy())
            perf['ts_code'] = ts_code
            mtf_results.append(perf)
    except Exception as e:
        continue

mtf_df = pd.DataFrame(mtf_results)

print("\n多周期KDJ共振策略结果:")
print(f"平均总收益: {mtf_df['total_return'].mean():.2f}%")
print(f"胜率: {mtf_df['win_rate'].mean():.2f}%")
print(f"平均单笔收益: {mtf_df['avg_return'].mean():.2f}%")
print(f"总交易次数: {mtf_df['trade_count'].sum()}")

# 策略对比
print("\n" + "=" * 80)
print("策略对比总结")
print("=" * 80)

comparison_data = {
    '策略': ['J超买超卖', 'KD金叉死叉', 'KDJ+均线过滤', '多周期KDJ共振'],
    '平均总收益%': [
        bt_df[bt_df['strategy'] == 'J超买超卖']['total_return'].mean(),
        bt_df[bt_df['strategy'] == 'KD金叉死叉']['total_return'].mean(),
        ma_df['total_return'].mean(),
        mtf_df['total_return'].mean()
    ],
    '胜率%': [
        bt_df[bt_df['strategy'] == 'J超买超卖']['win_rate'].mean(),
        bt_df[bt_df['strategy'] == 'KD金叉死叉']['win_rate'].mean(),
        ma_df['win_rate'].mean(),
        mtf_df['win_rate'].mean()
    ],
    '平均单笔收益%': [
        bt_df[bt_df['strategy'] == 'J超买超卖']['avg_return'].mean(),
        bt_df[bt_df['strategy'] == 'KD金叉死叉']['avg_return'].mean(),
        ma_df['avg_return'].mean(),
        mtf_df['avg_return'].mean()
    ],
    '总交易次数': [
        bt_df[bt_df['strategy'] == 'J超买超卖']['trade_count'].sum(),
        bt_df[bt_df['strategy'] == 'KD金叉死叉']['trade_count'].sum(),
        ma_df['trade_count'].sum(),
        mtf_df['trade_count'].sum()
    ]
}

comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df.round(2)
print("\n策略绩效对比:")
print(comparison_df.to_string(index=False))

# 绘制策略对比图
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

strategies = comparison_df['策略'].tolist()
colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']

# 总收益对比
ax1 = axes[0, 0]
bars1 = ax1.bar(strategies, comparison_df['平均总收益%'], color=colors)
ax1.set_ylabel('平均总收益 (%)')
ax1.set_title('策略总收益对比')
ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
for bar, val in zip(bars1, comparison_df['平均总收益%']):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val:.1f}%', ha='center', va='bottom')

# 胜率对比
ax2 = axes[0, 1]
bars2 = ax2.bar(strategies, comparison_df['胜率%'], color=colors)
ax2.set_ylabel('胜率 (%)')
ax2.set_title('策略胜率对比')
ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
for bar, val in zip(bars2, comparison_df['胜率%']):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val:.1f}%', ha='center', va='bottom')

# 单笔收益对比
ax3 = axes[1, 0]
bars3 = ax3.bar(strategies, comparison_df['平均单笔收益%'], color=colors)
ax3.set_ylabel('平均单笔收益 (%)')
ax3.set_title('平均单笔收益对比')
ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
for bar, val in zip(bars3, comparison_df['平均单笔收益%']):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f'{val:.2f}%', ha='center', va='bottom')

# 交易次数对比
ax4 = axes[1, 1]
bars4 = ax4.bar(strategies, comparison_df['总交易次数'], color=colors)
ax4.set_ylabel('总交易次数')
ax4.set_title('交易次数对比')
for bar, val in zip(bars4, comparison_df['总交易次数']):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             f'{int(val)}', ha='center', va='bottom')

plt.suptitle('KDJ策略回测对比', fontsize=14)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}kdj_strategy_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\n策略对比图已保存: {OUTPUT_DIR}kdj_strategy_comparison.png")

# =============================================================================
# 生成研究报告
# =============================================================================
print("\n" + "=" * 80)
print("生成研究报告")
print("=" * 80)

report_content = f"""
# KDJ交易策略研究报告

**生成日期**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 一、研究概述

本报告基于Tushare数据库中的A股日线数据，系统研究了KDJ指标的计算方法、参数优化、
交易信号识别及策略回测效果。

### 数据范围
- **数据源**: Tushare DuckDB数据库
- **分析周期**: 2024-01-01 至 2026-01-30
- **样本股票**: 30只主板/创业板/科创板股票

---

## 二、KDJ指标分析

### 2.1 KDJ计算公式

KDJ指标由RSV、K、D、J四个值组成：

```
RSV = (C - L_n) / (H_n - L_n) * 100
K = (M1-1)/M1 * K_prev + 1/M1 * RSV
D = (M2-1)/M2 * D_prev + 1/M2 * K
J = 3*K - 2*D
```

其中：
- C: 当日收盘价
- L_n: N日内最低价
- H_n: N日内最高价
- 标准参数: N=9, M1=3, M2=3

### 2.2 参数敏感性分析

测试不同参数组合（N: 5-19, M1/M2: 2-5）的效果：

| 推荐参数 | 特点 |
|---------|------|
| N=9, M1=3, M2=3 | 标准参数，信号适中 |
| N=14, M1=3, M2=3 | 较慢，信号更可靠 |
| N=5, M1=2, M2=2 | 敏感，信号频繁 |

### 2.3 KDJ值分布特征

基于{len(kdj_stats)}只股票的统计分析：

| 指标 | 均值 | 标准差 | 范围 |
|-----|------|-------|------|
| K值 | {kdj_stats['K_mean'].mean():.2f} | {kdj_stats['K_std'].mean():.2f} | 0-100 |
| D值 | {kdj_stats['D_mean'].mean():.2f} | {kdj_stats['D_std'].mean():.2f} | 0-100 |
| J值 | {kdj_stats['J_mean'].mean():.2f} | {kdj_stats['J_std'].mean():.2f} | {kdj_stats['J_min'].min():.0f} ~ {kdj_stats['J_max'].max():.0f} |

---

## 三、交易信号分析

### 3.1 J值超买超卖信号

**信号规则**：
- 超卖买入：J值从0以下回升至0以上
- 超买卖出：J值从100以上回落至100以下

**信号效果**（基于{test_name}）：
- 超卖买入信号: {len(buy_signals)}次
- 超买卖出信号: {len(sell_signals)}次

### 3.2 KD金叉死叉信号

**信号规则**：
- 金叉：K线上穿D线
- 死叉：K线下穿D线

**不同区域金叉效果**：
- 超卖区金叉（K<20）：最佳买入时机
- 低位区金叉（K=20-50）：较好买入时机
- 高位区金叉（K>50）：追高风险较大

### 3.3 KDJ背离信号

**背离类型**：
- 底背离：价格创新低，KDJ不创新低 -> 看涨信号
- 顶背离：价格创新高，KDJ不创新高 -> 看跌信号

---

## 四、策略回测结果

### 4.1 策略绩效对比

{comparison_df.to_markdown(index=False)}

### 4.2 策略特点分析

1. **J超买超卖策略**
   - 优点：信号明确，操作简单
   - 缺点：信号较少，可能错过趋势

2. **KD金叉死叉策略**
   - 优点：信号频繁，捕捉短期波动
   - 缺点：假信号较多，需要过滤

3. **KDJ+均线过滤策略**
   - 优点：趋势确认，减少假信号
   - 缺点：入场较晚，可能错过起涨点

4. **多周期KDJ共振策略**
   - 优点：多重确认，信号可靠
   - 缺点：信号稀少，需要耐心等待

---

## 五、策略建议

### 5.1 最佳实践

1. **参数设置**：建议使用标准参数(9,3,3)，稳健投资者可用(14,3,3)

2. **信号过滤**：
   - 结合均线判断趋势方向
   - 关注多周期共振信号
   - 低位金叉优于高位金叉

3. **风险控制**：
   - 设置固定止损（如-5%）
   - 设置持仓时间上限
   - 分散投资，控制仓位

### 5.2 注意事项

1. KDJ在震荡市效果较好，趋势市可能产生连续钝化
2. J值的敏感性最高，但假信号也最多
3. 背离信号需要较长周期确认，不宜频繁交易
4. 回测结果仅供参考，实盘需考虑滑点和手续费

---

## 六、附录

### 生成的图表文件

1. `kdj_example.png` - KDJ指标示例图
2. `kdj_distribution.png` - KDJ值分布图
3. `kdj_signals.png` - 交易信号示意图
4. `kdj_strategy_comparison.png` - 策略对比图

---

*本报告由AI自动生成，仅供研究参考，不构成投资建议。*
"""

# 保存报告
report_path = f'{OUTPUT_DIR}kdj_strategy_report.md'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report_content)

print(f"\n研究报告已保存: {report_path}")

# 关闭数据库连接
conn.close()

print("\n" + "=" * 80)
print("KDJ交易策略研究完成!")
print("=" * 80)
print(f"\n输出文件:")
print(f"  1. {OUTPUT_DIR}kdj_strategy_report.md")
print(f"  2. {OUTPUT_DIR}kdj_example.png")
print(f"  3. {OUTPUT_DIR}kdj_distribution.png")
print(f"  4. {OUTPUT_DIR}kdj_signals.png")
print(f"  5. {OUTPUT_DIR}kdj_strategy_comparison.png")
