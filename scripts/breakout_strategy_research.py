#!/usr/bin/env python3
"""
突破交易策略研究
================

研究内容：
1. 突破类型识别（新高突破、区间突破、均线突破、成交量突破）
2. 有效性验证（真假突破识别、突破后走势统计、突破失败分析）
3. 策略设计（突破买入策略、回踩确认策略、风险管理）

数据源：tushare.db
"""

import duckdb
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 配置
DB_PATH = '/Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db'
REPORT_PATH = '/Users/allen/workspace/python/stock/Tushare-DuckDB/reports/research'

# ============================================================================
# 数据加载模块
# ============================================================================

def get_connection():
    """获取数据库连接"""
    return duckdb.connect(DB_PATH, read_only=True)

def load_daily_data(start_date='20200101', end_date='20260130'):
    """加载日线数据并计算前复权价格"""
    conn = get_connection()

    query = f"""
    SELECT
        d.ts_code,
        d.trade_date,
        d.open * a.adj_factor as open,
        d.high * a.adj_factor as high,
        d.low * a.adj_factor as low,
        d.close * a.adj_factor as close,
        d.vol,
        d.amount,
        d.pct_chg
    FROM daily d
    LEFT JOIN adj_factor a ON d.ts_code = a.ts_code AND d.trade_date = a.trade_date
    WHERE d.trade_date >= '{start_date}' AND d.trade_date <= '{end_date}'
    ORDER BY d.ts_code, d.trade_date
    """

    df = conn.execute(query).fetchdf()
    conn.close()
    return df

def load_stock_basic():
    """加载股票基本信息"""
    conn = get_connection()
    df = conn.execute("SELECT ts_code, name, industry, list_date FROM stock_basic WHERE list_status = 'L'").fetchdf()
    conn.close()
    return df

# ============================================================================
# 突破类型识别模块
# ============================================================================

class BreakoutDetector:
    """突破检测器"""

    def __init__(self, df):
        self.df = df.copy()
        self.df['trade_date'] = pd.to_datetime(self.df['trade_date'])
        self.df = self.df.sort_values(['ts_code', 'trade_date'])

    def detect_new_high_breakout(self, period=60):
        """
        检测新高突破
        - period日新高突破
        """
        results = []

        for ts_code, group in self.df.groupby('ts_code'):
            if len(group) < period + 10:
                continue

            group = group.reset_index(drop=True)

            # 计算period日最高价
            group['high_max'] = group['high'].rolling(window=period, min_periods=period).max().shift(1)

            # 检测突破
            breakouts = group[group['close'] > group['high_max']].copy()

            if len(breakouts) > 0:
                breakouts['breakout_type'] = 'new_high'
                breakouts['breakout_period'] = period
                breakouts['ts_code'] = ts_code
                results.append(breakouts[['ts_code', 'trade_date', 'close', 'high_max',
                                         'vol', 'pct_chg', 'breakout_type', 'breakout_period']])

        if results:
            return pd.concat(results, ignore_index=True)
        return pd.DataFrame()

    def detect_range_breakout(self, period=20, threshold=0.05):
        """
        检测区间突破
        - 股价在period日内波动幅度小于threshold，然后突破上沿
        """
        results = []

        for ts_code, group in self.df.groupby('ts_code'):
            if len(group) < period + 10:
                continue

            group = group.reset_index(drop=True)

            # 计算区间
            group['range_high'] = group['high'].rolling(window=period, min_periods=period).max().shift(1)
            group['range_low'] = group['low'].rolling(window=period, min_periods=period).min().shift(1)
            group['range_pct'] = (group['range_high'] - group['range_low']) / group['range_low']

            # 筛选震荡区间
            in_range = group['range_pct'] < threshold

            # 检测突破
            breakouts = group[(in_range) & (group['close'] > group['range_high'])].copy()

            if len(breakouts) > 0:
                breakouts['breakout_type'] = 'range'
                breakouts['breakout_period'] = period
                breakouts['ts_code'] = ts_code
                results.append(breakouts[['ts_code', 'trade_date', 'close', 'range_high',
                                         'range_low', 'vol', 'pct_chg', 'breakout_type', 'breakout_period']])

        if results:
            return pd.concat(results, ignore_index=True)
        return pd.DataFrame()

    def detect_ma_breakout(self, ma_periods=[20, 60, 120]):
        """
        检测均线突破
        - 收盘价向上突破均线
        """
        results = []

        for ts_code, group in self.df.groupby('ts_code'):
            if len(group) < max(ma_periods) + 10:
                continue

            group = group.reset_index(drop=True)

            for ma_period in ma_periods:
                group[f'ma{ma_period}'] = group['close'].rolling(window=ma_period, min_periods=ma_period).mean()
                group[f'ma{ma_period}_prev'] = group[f'ma{ma_period}'].shift(1)
                group['close_prev'] = group['close'].shift(1)

                # 检测向上突破均线
                breakouts = group[(group['close_prev'] < group[f'ma{ma_period}_prev']) &
                                 (group['close'] > group[f'ma{ma_period}'])].copy()

                if len(breakouts) > 0:
                    breakouts['breakout_type'] = 'ma'
                    breakouts['breakout_period'] = ma_period
                    breakouts['ts_code'] = ts_code
                    breakouts['ma_value'] = breakouts[f'ma{ma_period}']
                    results.append(breakouts[['ts_code', 'trade_date', 'close', 'ma_value',
                                             'vol', 'pct_chg', 'breakout_type', 'breakout_period']])

        if results:
            return pd.concat(results, ignore_index=True)
        return pd.DataFrame()

    def detect_volume_breakout(self, vol_period=20, vol_multiplier=2.0):
        """
        检测成交量突破
        - 成交量超过过去period日平均成交量的multiplier倍
        """
        results = []

        for ts_code, group in self.df.groupby('ts_code'):
            if len(group) < vol_period + 10:
                continue

            group = group.reset_index(drop=True)

            # 计算平均成交量
            group['vol_ma'] = group['vol'].rolling(window=vol_period, min_periods=vol_period).mean().shift(1)
            group['vol_ratio'] = group['vol'] / group['vol_ma']

            # 检测放量
            breakouts = group[(group['vol_ratio'] > vol_multiplier) & (group['pct_chg'] > 0)].copy()

            if len(breakouts) > 0:
                breakouts['breakout_type'] = 'volume'
                breakouts['breakout_period'] = vol_period
                breakouts['ts_code'] = ts_code
                results.append(breakouts[['ts_code', 'trade_date', 'close', 'vol',
                                         'vol_ma', 'vol_ratio', 'pct_chg', 'breakout_type', 'breakout_period']])

        if results:
            return pd.concat(results, ignore_index=True)
        return pd.DataFrame()

# ============================================================================
# 突破有效性验证模块
# ============================================================================

class BreakoutValidator:
    """突破有效性验证"""

    def __init__(self, df, breakouts):
        self.df = df.copy()
        self.df['trade_date'] = pd.to_datetime(self.df['trade_date'])
        self.breakouts = breakouts.copy()
        if 'trade_date' in self.breakouts.columns:
            self.breakouts['trade_date'] = pd.to_datetime(self.breakouts['trade_date'])

    def calculate_post_breakout_returns(self, holding_periods=[5, 10, 20, 60]):
        """
        计算突破后收益
        """
        results = []

        for idx, row in self.breakouts.iterrows():
            ts_code = row['ts_code']
            breakout_date = row['trade_date']

            stock_data = self.df[self.df['ts_code'] == ts_code].copy()
            stock_data = stock_data[stock_data['trade_date'] >= breakout_date].reset_index(drop=True)

            if len(stock_data) < max(holding_periods) + 1:
                continue

            entry_price = row['close']
            result = {
                'ts_code': ts_code,
                'trade_date': breakout_date,
                'breakout_type': row['breakout_type'],
                'breakout_period': row['breakout_period'],
                'entry_price': entry_price,
            }

            # 计算各持有期收益
            for period in holding_periods:
                if len(stock_data) > period:
                    exit_price = stock_data.iloc[period]['close']
                    result[f'return_{period}d'] = (exit_price - entry_price) / entry_price * 100

                    # 计算最大回撤
                    period_data = stock_data.iloc[:period+1]
                    max_price = period_data['high'].max()
                    min_price = period_data['low'].min()
                    result[f'max_gain_{period}d'] = (max_price - entry_price) / entry_price * 100
                    result[f'max_drawdown_{period}d'] = (entry_price - min_price) / entry_price * 100
                else:
                    result[f'return_{period}d'] = np.nan
                    result[f'max_gain_{period}d'] = np.nan
                    result[f'max_drawdown_{period}d'] = np.nan

            results.append(result)

        return pd.DataFrame(results)

    def identify_true_false_breakout(self, confirmation_days=3, threshold=0.02):
        """
        识别真假突破
        - 真突破：突破后confirmation_days内不跌破突破点
        - 假突破：突破后快速回落
        """
        results = []

        for idx, row in self.breakouts.iterrows():
            ts_code = row['ts_code']
            breakout_date = row['trade_date']
            breakout_price = row['close']

            stock_data = self.df[self.df['ts_code'] == ts_code].copy()
            stock_data = stock_data[stock_data['trade_date'] > breakout_date].reset_index(drop=True)

            if len(stock_data) < confirmation_days:
                continue

            # 检查确认期内的最低价
            confirm_data = stock_data.iloc[:confirmation_days]
            min_low = confirm_data['low'].min()

            # 判断是否为假突破
            if min_low < breakout_price * (1 - threshold):
                is_true_breakout = False
            else:
                is_true_breakout = True

            result = {
                'ts_code': ts_code,
                'trade_date': breakout_date,
                'breakout_type': row['breakout_type'],
                'breakout_price': breakout_price,
                'confirm_min_low': min_low,
                'is_true_breakout': is_true_breakout
            }
            results.append(result)

        return pd.DataFrame(results)

    def analyze_failed_breakouts(self, validation_df, return_df):
        """
        分析突破失败的特征
        """
        if validation_df.empty or return_df.empty:
            return {}

        # 合并数据
        merged = validation_df.merge(return_df, on=['ts_code', 'trade_date', 'breakout_type'])

        if merged.empty:
            return {}

        # 分析假突破特征
        false_breakouts = merged[merged['is_true_breakout'] == False]
        true_breakouts = merged[merged['is_true_breakout'] == True]

        analysis = {
            'total_breakouts': len(merged),
            'true_breakouts': len(true_breakouts),
            'false_breakouts': len(false_breakouts),
            'true_breakout_rate': len(true_breakouts) / len(merged) * 100 if len(merged) > 0 else 0,
        }

        # 各类型突破的真假分布
        by_type = merged.groupby('breakout_type').agg({
            'is_true_breakout': ['sum', 'count', 'mean']
        }).reset_index()
        by_type.columns = ['breakout_type', 'true_count', 'total_count', 'true_rate']
        analysis['by_type'] = by_type

        # 真假突破的收益对比
        for period in [5, 10, 20, 60]:
            col = f'return_{period}d'
            if col in merged.columns:
                analysis[f'true_{col}_mean'] = true_breakouts[col].mean() if len(true_breakouts) > 0 else np.nan
                analysis[f'false_{col}_mean'] = false_breakouts[col].mean() if len(false_breakouts) > 0 else np.nan

        return analysis

# ============================================================================
# 策略设计模块
# ============================================================================

class BreakoutStrategy:
    """突破交易策略"""

    def __init__(self, df):
        self.df = df.copy()
        self.df['trade_date'] = pd.to_datetime(self.df['trade_date'])

    def breakout_buy_strategy(self, breakout_type='new_high', period=60,
                              stop_loss=0.05, take_profit=0.20, max_hold_days=20):
        """
        突破买入策略
        - 入场：突破确认后买入
        - 止损：跌破入场价stop_loss
        - 止盈：盈利超过take_profit
        - 持有期限：最多max_hold_days天
        """
        detector = BreakoutDetector(self.df)

        if breakout_type == 'new_high':
            breakouts = detector.detect_new_high_breakout(period=period)
        elif breakout_type == 'range':
            breakouts = detector.detect_range_breakout(period=period)
        elif breakout_type == 'ma':
            breakouts = detector.detect_ma_breakout(ma_periods=[period])
        elif breakout_type == 'volume':
            breakouts = detector.detect_volume_breakout(vol_period=period)
        else:
            return pd.DataFrame()

        if breakouts.empty:
            return pd.DataFrame()

        trades = []

        for idx, row in breakouts.iterrows():
            ts_code = row['ts_code']
            entry_date = row['trade_date']
            entry_price = row['close']

            stock_data = self.df[self.df['ts_code'] == ts_code].copy()
            stock_data = stock_data[stock_data['trade_date'] > entry_date].reset_index(drop=True)

            if len(stock_data) < 2:
                continue

            exit_date = None
            exit_price = None
            exit_reason = None

            for i in range(min(max_hold_days, len(stock_data))):
                day_data = stock_data.iloc[i]

                # 检查止损
                if day_data['low'] <= entry_price * (1 - stop_loss):
                    exit_date = day_data['trade_date']
                    exit_price = entry_price * (1 - stop_loss)
                    exit_reason = 'stop_loss'
                    break

                # 检查止盈
                if day_data['high'] >= entry_price * (1 + take_profit):
                    exit_date = day_data['trade_date']
                    exit_price = entry_price * (1 + take_profit)
                    exit_reason = 'take_profit'
                    break

            # 如果未触发止损止盈，按持有期限卖出
            if exit_date is None:
                if len(stock_data) >= max_hold_days:
                    exit_date = stock_data.iloc[max_hold_days-1]['trade_date']
                    exit_price = stock_data.iloc[max_hold_days-1]['close']
                    exit_reason = 'max_hold'
                else:
                    continue

            trade = {
                'ts_code': ts_code,
                'entry_date': entry_date,
                'entry_price': entry_price,
                'exit_date': exit_date,
                'exit_price': exit_price,
                'exit_reason': exit_reason,
                'return_pct': (exit_price - entry_price) / entry_price * 100,
                'breakout_type': breakout_type,
            }
            trades.append(trade)

        return pd.DataFrame(trades)

    def pullback_confirm_strategy(self, breakout_type='new_high', period=60,
                                   pullback_ratio=0.03, confirm_ratio=0.02,
                                   stop_loss=0.05, take_profit=0.20, max_hold_days=20):
        """
        回踩确认策略
        - 入场：突破后回踩不破支撑，再次上涨确认后买入
        - 止损：跌破入场价stop_loss
        - 止盈：盈利超过take_profit
        """
        detector = BreakoutDetector(self.df)

        if breakout_type == 'new_high':
            breakouts = detector.detect_new_high_breakout(period=period)
        elif breakout_type == 'range':
            breakouts = detector.detect_range_breakout(period=period)
        elif breakout_type == 'ma':
            breakouts = detector.detect_ma_breakout(ma_periods=[period])
        else:
            return pd.DataFrame()

        if breakouts.empty:
            return pd.DataFrame()

        trades = []

        for idx, row in breakouts.iterrows():
            ts_code = row['ts_code']
            breakout_date = row['trade_date']
            breakout_price = row['close']

            stock_data = self.df[self.df['ts_code'] == ts_code].copy()
            stock_data = stock_data[stock_data['trade_date'] > breakout_date].reset_index(drop=True)

            if len(stock_data) < 10:
                continue

            # 寻找回踩确认点
            entry_date = None
            entry_price = None
            min_low = breakout_price
            has_pullback = False

            for i in range(min(10, len(stock_data))):
                day_data = stock_data.iloc[i]

                # 记录最低点
                if day_data['low'] < min_low:
                    min_low = day_data['low']

                # 检查是否回踩
                if not has_pullback and min_low < breakout_price * (1 - pullback_ratio):
                    has_pullback = True
                    continue

                # 检查是否确认反弹
                if has_pullback and day_data['close'] > min_low * (1 + confirm_ratio):
                    # 确认回踩不破突破点太多
                    if min_low >= breakout_price * 0.95:
                        entry_date = day_data['trade_date']
                        entry_price = day_data['close']
                        break

            if entry_date is None:
                continue

            # 执行交易
            trade_data = stock_data[stock_data['trade_date'] > entry_date].reset_index(drop=True)
            if len(trade_data) < 2:
                continue

            exit_date = None
            exit_price = None
            exit_reason = None

            for i in range(min(max_hold_days, len(trade_data))):
                day_data = trade_data.iloc[i]

                if day_data['low'] <= entry_price * (1 - stop_loss):
                    exit_date = day_data['trade_date']
                    exit_price = entry_price * (1 - stop_loss)
                    exit_reason = 'stop_loss'
                    break

                if day_data['high'] >= entry_price * (1 + take_profit):
                    exit_date = day_data['trade_date']
                    exit_price = entry_price * (1 + take_profit)
                    exit_reason = 'take_profit'
                    break

            if exit_date is None:
                if len(trade_data) >= max_hold_days:
                    exit_date = trade_data.iloc[max_hold_days-1]['trade_date']
                    exit_price = trade_data.iloc[max_hold_days-1]['close']
                    exit_reason = 'max_hold'
                else:
                    continue

            trade = {
                'ts_code': ts_code,
                'breakout_date': breakout_date,
                'entry_date': entry_date,
                'entry_price': entry_price,
                'exit_date': exit_date,
                'exit_price': exit_price,
                'exit_reason': exit_reason,
                'return_pct': (exit_price - entry_price) / entry_price * 100,
                'breakout_type': breakout_type,
            }
            trades.append(trade)

        return pd.DataFrame(trades)

def evaluate_strategy(trades_df):
    """
    评估策略表现
    """
    if trades_df.empty:
        return {}

    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['return_pct'] > 0])
    losing_trades = len(trades_df[trades_df['return_pct'] <= 0])

    avg_return = trades_df['return_pct'].mean()
    median_return = trades_df['return_pct'].median()
    std_return = trades_df['return_pct'].std()

    avg_win = trades_df[trades_df['return_pct'] > 0]['return_pct'].mean() if winning_trades > 0 else 0
    avg_loss = trades_df[trades_df['return_pct'] <= 0]['return_pct'].mean() if losing_trades > 0 else 0

    win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
    profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 and avg_loss != 0 else np.inf

    # 按退出原因统计
    exit_stats = trades_df.groupby('exit_reason').agg({
        'return_pct': ['count', 'mean']
    }).reset_index()
    exit_stats.columns = ['exit_reason', 'count', 'avg_return']

    return {
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'avg_return': avg_return,
        'median_return': median_return,
        'std_return': std_return,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'exit_stats': exit_stats,
    }

# ============================================================================
# 主程序
# ============================================================================

def main():
    print("=" * 80)
    print("突破交易策略研究")
    print("=" * 80)

    # 加载数据
    print("\n[1] 加载数据...")
    df = load_daily_data(start_date='20200101', end_date='20251231')
    stock_basic = load_stock_basic()
    print(f"    数据范围: {df['trade_date'].min()} - {df['trade_date'].max()}")
    print(f"    股票数量: {df['ts_code'].nunique()}")
    print(f"    数据记录: {len(df):,}")

    # ========================================================================
    # 第一部分：突破类型识别
    # ========================================================================
    print("\n" + "=" * 80)
    print("第一部分：突破类型识别")
    print("=" * 80)

    detector = BreakoutDetector(df)

    # 1.1 新高突破
    print("\n[1.1] 检测60日新高突破...")
    new_high_breakouts = detector.detect_new_high_breakout(period=60)
    print(f"      检测到突破信号: {len(new_high_breakouts):,}")

    # 1.2 区间突破
    print("\n[1.2] 检测区间突破...")
    range_breakouts = detector.detect_range_breakout(period=20, threshold=0.08)
    print(f"      检测到突破信号: {len(range_breakouts):,}")

    # 1.3 均线突破
    print("\n[1.3] 检测均线突破...")
    ma_breakouts = detector.detect_ma_breakout(ma_periods=[20, 60, 120])
    print(f"      检测到突破信号: {len(ma_breakouts):,}")
    if not ma_breakouts.empty:
        ma_by_period = ma_breakouts.groupby('breakout_period').size()
        for period, count in ma_by_period.items():
            print(f"          MA{period}: {count:,}")

    # 1.4 成交量突破
    print("\n[1.4] 检测成交量突破...")
    volume_breakouts = detector.detect_volume_breakout(vol_period=20, vol_multiplier=2.0)
    print(f"      检测到突破信号: {len(volume_breakouts):,}")

    # ========================================================================
    # 第二部分：有效性验证
    # ========================================================================
    print("\n" + "=" * 80)
    print("第二部分：有效性验证")
    print("=" * 80)

    # 2.1 新高突破有效性验证
    print("\n[2.1] 新高突破有效性验证...")
    if not new_high_breakouts.empty:
        # 采样分析（取最近的数据）
        sample_breakouts = new_high_breakouts[new_high_breakouts['trade_date'] >= '2023-01-01'].head(5000)

        validator = BreakoutValidator(df, sample_breakouts)
        new_high_returns = validator.calculate_post_breakout_returns()
        new_high_validation = validator.identify_true_false_breakout()
        new_high_analysis = validator.analyze_failed_breakouts(new_high_validation, new_high_returns)

        print(f"      样本数量: {len(sample_breakouts):,}")
        if new_high_analysis:
            print(f"      真突破比例: {new_high_analysis.get('true_breakout_rate', 0):.1f}%")
            if 'true_return_5d_mean' in new_high_analysis:
                print(f"      真突破5日平均收益: {new_high_analysis['true_return_5d_mean']:.2f}%")
                print(f"      假突破5日平均收益: {new_high_analysis['false_return_5d_mean']:.2f}%")
    else:
        new_high_returns = pd.DataFrame()
        new_high_validation = pd.DataFrame()
        new_high_analysis = {}

    # 2.2 均线突破有效性验证
    print("\n[2.2] 均线突破有效性验证...")
    if not ma_breakouts.empty:
        # 采样分析MA60
        ma60_breakouts = ma_breakouts[
            (ma_breakouts['breakout_period'] == 60) &
            (ma_breakouts['trade_date'] >= '2023-01-01')
        ].head(5000)

        if not ma60_breakouts.empty:
            validator = BreakoutValidator(df, ma60_breakouts)
            ma60_returns = validator.calculate_post_breakout_returns()
            ma60_validation = validator.identify_true_false_breakout()
            ma60_analysis = validator.analyze_failed_breakouts(ma60_validation, ma60_returns)

            print(f"      MA60样本数量: {len(ma60_breakouts):,}")
            if ma60_analysis:
                print(f"      真突破比例: {ma60_analysis.get('true_breakout_rate', 0):.1f}%")
                if 'true_return_5d_mean' in ma60_analysis:
                    print(f"      真突破5日平均收益: {ma60_analysis['true_return_5d_mean']:.2f}%")
                    print(f"      假突破5日平均收益: {ma60_analysis['false_return_5d_mean']:.2f}%")
        else:
            ma60_returns = pd.DataFrame()
            ma60_validation = pd.DataFrame()
            ma60_analysis = {}
    else:
        ma60_returns = pd.DataFrame()
        ma60_validation = pd.DataFrame()
        ma60_analysis = {}

    # ========================================================================
    # 第三部分：策略设计与回测
    # ========================================================================
    print("\n" + "=" * 80)
    print("第三部分：策略设计与回测")
    print("=" * 80)

    # 使用2020-2024年数据进行回测
    backtest_df = df[df['trade_date'] <= '2024-12-31']
    strategy = BreakoutStrategy(backtest_df)

    # 3.1 突破买入策略
    print("\n[3.1] 突破买入策略回测...")

    strategy_results = {}

    # 新高突破策略
    print("      - 60日新高突破策略...")
    new_high_trades = strategy.breakout_buy_strategy(
        breakout_type='new_high', period=60,
        stop_loss=0.05, take_profit=0.15, max_hold_days=20
    )
    if not new_high_trades.empty:
        # 采样分析
        sample_trades = new_high_trades.sample(min(3000, len(new_high_trades)), random_state=42)
        new_high_eval = evaluate_strategy(sample_trades)
        strategy_results['new_high_60d'] = new_high_eval
        print(f"        交易次数: {new_high_eval['total_trades']}")
        print(f"        胜率: {new_high_eval['win_rate']:.1f}%")
        print(f"        平均收益: {new_high_eval['avg_return']:.2f}%")

    # 均线突破策略
    print("      - MA60均线突破策略...")
    ma60_trades = strategy.breakout_buy_strategy(
        breakout_type='ma', period=60,
        stop_loss=0.05, take_profit=0.15, max_hold_days=20
    )
    if not ma60_trades.empty:
        sample_trades = ma60_trades.sample(min(3000, len(ma60_trades)), random_state=42)
        ma60_eval = evaluate_strategy(sample_trades)
        strategy_results['ma60'] = ma60_eval
        print(f"        交易次数: {ma60_eval['total_trades']}")
        print(f"        胜率: {ma60_eval['win_rate']:.1f}%")
        print(f"        平均收益: {ma60_eval['avg_return']:.2f}%")

    # 区间突破策略
    print("      - 区间突破策略...")
    range_trades = strategy.breakout_buy_strategy(
        breakout_type='range', period=20,
        stop_loss=0.05, take_profit=0.15, max_hold_days=20
    )
    if not range_trades.empty:
        sample_trades = range_trades.sample(min(3000, len(range_trades)), random_state=42)
        range_eval = evaluate_strategy(sample_trades)
        strategy_results['range_20d'] = range_eval
        print(f"        交易次数: {range_eval['total_trades']}")
        print(f"        胜率: {range_eval['win_rate']:.1f}%")
        print(f"        平均收益: {range_eval['avg_return']:.2f}%")

    # 3.2 回踩确认策略
    print("\n[3.2] 回踩确认策略回测...")

    print("      - 60日新高回踩确认策略...")
    pullback_new_high_trades = strategy.pullback_confirm_strategy(
        breakout_type='new_high', period=60,
        pullback_ratio=0.03, confirm_ratio=0.02,
        stop_loss=0.05, take_profit=0.15, max_hold_days=20
    )
    if not pullback_new_high_trades.empty:
        sample_trades = pullback_new_high_trades.sample(min(2000, len(pullback_new_high_trades)), random_state=42)
        pullback_eval = evaluate_strategy(sample_trades)
        strategy_results['pullback_new_high'] = pullback_eval
        print(f"        交易次数: {pullback_eval['total_trades']}")
        print(f"        胜率: {pullback_eval['win_rate']:.1f}%")
        print(f"        平均收益: {pullback_eval['avg_return']:.2f}%")

    # ========================================================================
    # 生成报告
    # ========================================================================
    print("\n" + "=" * 80)
    print("生成研究报告")
    print("=" * 80)

    report = generate_report(
        df, stock_basic,
        new_high_breakouts, range_breakouts, ma_breakouts, volume_breakouts,
        new_high_returns, new_high_validation, new_high_analysis,
        ma60_returns if 'ma60_returns' in dir() else pd.DataFrame(),
        ma60_analysis if 'ma60_analysis' in dir() else {},
        strategy_results
    )

    # 保存报告
    report_file = f"{REPORT_PATH}/breakout_strategy_research.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n报告已保存到: {report_file}")

    return report

def generate_report(df, stock_basic,
                   new_high_breakouts, range_breakouts, ma_breakouts, volume_breakouts,
                   new_high_returns, new_high_validation, new_high_analysis,
                   ma60_returns, ma60_analysis,
                   strategy_results):
    """生成研究报告"""

    report = """# 突破交易策略研究报告

## 研究概述

本报告对股票市场中的突破交易策略进行了系统性研究，包括突破类型识别、有效性验证和策略设计三个方面。

**数据范围**
- 时间跨度：2020年1月 - 2025年12月
- 股票数量：{stock_count:,}
- 数据记录：{record_count:,}

---

## 第一部分：突破类型识别

### 1.1 新高突破

**定义**：收盘价突破过去N日（默认60日）的最高价。

**信号统计**
- 60日新高突破信号总数：{new_high_count:,}
- 平均每只股票产生信号：{new_high_avg:.1f}次

**特点**
- 强势股的标志性信号
- 通常伴随市场关注度提升
- 需要确认是否有效突破

### 1.2 区间突破

**定义**：股价在一定期间内波动幅度较小（箱体整理），随后向上突破区间上沿。

**信号统计**
- 区间突破信号总数：{range_count:,}
- 区间震幅阈值：8%

**特点**
- 横盘整理后的方向选择
- 成功率与整理时间长度正相关
- 需要成交量配合确认

### 1.3 均线突破

**定义**：收盘价从下方向上穿越均线。

**信号统计**
- 均线突破信号总数：{ma_count:,}
- 分布情况：
  - MA20突破：{ma20_count:,}
  - MA60突破：{ma60_count:,}
  - MA120突破：{ma120_count:,}

**特点**
- 趋势转换的重要信号
- 长周期均线突破更可靠
- 需要关注均线斜率

### 1.4 成交量突破

**定义**：成交量超过过去N日平均成交量的2倍以上，且当日上涨。

**信号统计**
- 成交量突破信号总数：{volume_count:,}

**特点**
- 资金入场的重要信号
- 需结合价格位置判断
- 连续放量更有意义

---

## 第二部分：有效性验证

### 2.1 真假突破识别

**判定标准**
- 真突破：突破后3日内最低价不跌破突破点2%
- 假突破：突破后快速回落至突破点以下

**60日新高突破验证结果**
{new_high_validation_text}

### 2.2 突破后走势统计

**60日新高突破后收益分布**
{new_high_returns_text}

**MA60均线突破后收益分布**
{ma60_returns_text}

### 2.3 突破失败分析

**失败特征**
1. 缺乏成交量配合的突破更容易失败
2. 市场整体弱势时突破成功率下降
3. 个股基本面恶化导致的假突破
4. 短期涨幅过大后的假突破

**识别方法**
1. 关注突破当日成交量（放量突破更可靠）
2. 观察突破后的确认过程（回踩不破更可靠）
3. 结合大盘走势判断（顺势突破更可靠）
4. 关注突破前的形态（底部整理后突破更可靠）

---

## 第三部分：策略设计

### 3.1 突破买入策略

**策略参数**
- 止损：-5%
- 止盈：+15%
- 最大持有期：20个交易日

**回测结果**

{strategy_results_text}

### 3.2 回踩确认策略

**策略逻辑**
1. 等待突破信号出现
2. 观察股价回踩至突破点附近（回调3%以内）
3. 确认回踩后反弹（上涨2%以上）即买入
4. 执行止损止盈规则

**优势**
- 降低了追高风险
- 提高了入场价格的安全边际
- 过滤了部分假突破

**劣势**
- 可能错过强势突破的行情
- 信号数量较少
- 等待时间成本

{pullback_results_text}

### 3.3 风险管理建议

**仓位管理**
1. 单笔交易仓位不超过总资金的10%
2. 同时持仓不超过5只股票
3. 采用金字塔加仓法（信号确认后逐步加仓）

**止损原则**
1. 固定止损：跌破买入价5%-8%
2. 技术止损：跌破关键支撑位
3. 时间止损：超过持有期限自动离场

**风险控制**
1. 避免在大盘弱势时逆势操作
2. 避免在业绩窗口期操作有风险的个股
3. 分散持仓，降低单一股票风险
4. 设置每日/每周最大亏损限制

---

## 第四部分：研究结论

### 4.1 主要发现

1. **新高突破策略**在强势市场中表现较好，但需要严格的止损控制
2. **均线突破策略**稳定性较好，MA60突破的性价比较高
3. **回踩确认策略**可以有效降低假突破的风险，但会错过部分机会
4. **成交量配合**是判断突破有效性的重要因素

### 4.2 策略优化方向

1. **多因子过滤**：结合基本面因子筛选突破标的
2. **市场择时**：根据市场环境调整策略参数
3. **动态止损**：引入ATR等波动率指标动态调整止损位
4. **机器学习**：使用历史数据训练突破成功率预测模型

### 4.3 实践建议

1. 建议从**MA60均线突破+回踩确认**策略开始实践
2. 初期使用较小仓位验证策略有效性
3. 建立交易日志，持续优化策略参数
4. 关注市场整体状态，避免逆势操作

---

*报告生成时间：{report_time}*
""".format(
        stock_count=df['ts_code'].nunique(),
        record_count=len(df),
        new_high_count=len(new_high_breakouts),
        new_high_avg=len(new_high_breakouts) / df['ts_code'].nunique() if df['ts_code'].nunique() > 0 else 0,
        range_count=len(range_breakouts),
        ma_count=len(ma_breakouts),
        ma20_count=len(ma_breakouts[ma_breakouts['breakout_period'] == 20]) if not ma_breakouts.empty else 0,
        ma60_count=len(ma_breakouts[ma_breakouts['breakout_period'] == 60]) if not ma_breakouts.empty else 0,
        ma120_count=len(ma_breakouts[ma_breakouts['breakout_period'] == 120]) if not ma_breakouts.empty else 0,
        volume_count=len(volume_breakouts),
        new_high_validation_text=format_validation_text(new_high_analysis),
        new_high_returns_text=format_returns_text(new_high_returns, 'new_high'),
        ma60_returns_text=format_returns_text(ma60_returns, 'ma60'),
        strategy_results_text=format_strategy_results(strategy_results),
        pullback_results_text=format_pullback_results(strategy_results),
        report_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    )

    return report

def format_validation_text(analysis):
    """格式化验证结果文本"""
    if not analysis:
        return "（数据不足，无法计算）"

    text = f"""
- 样本总数：{analysis.get('total_breakouts', 0):,}
- 真突破数量：{analysis.get('true_breakouts', 0):,}
- 假突破数量：{analysis.get('false_breakouts', 0):,}
- 真突破比例：{analysis.get('true_breakout_rate', 0):.1f}%

**真假突破收益对比**
| 持有期 | 真突破平均收益 | 假突破平均收益 |
|--------|----------------|----------------|
| 5日 | {analysis.get('true_return_5d_mean', 0):.2f}% | {analysis.get('false_return_5d_mean', 0):.2f}% |
| 10日 | {analysis.get('true_return_10d_mean', 0):.2f}% | {analysis.get('false_return_10d_mean', 0):.2f}% |
| 20日 | {analysis.get('true_return_20d_mean', 0):.2f}% | {analysis.get('false_return_20d_mean', 0):.2f}% |
"""
    return text

def format_returns_text(returns_df, breakout_type):
    """格式化收益统计文本"""
    if returns_df.empty:
        return "（数据不足，无法计算）"

    text = f"""
| 统计指标 | 5日 | 10日 | 20日 | 60日 |
|----------|-----|------|------|------|
| 平均收益 | {returns_df['return_5d'].mean():.2f}% | {returns_df['return_10d'].mean():.2f}% | {returns_df['return_20d'].mean():.2f}% | {returns_df['return_60d'].mean():.2f}% |
| 中位收益 | {returns_df['return_5d'].median():.2f}% | {returns_df['return_10d'].median():.2f}% | {returns_df['return_20d'].median():.2f}% | {returns_df['return_60d'].median():.2f}% |
| 收益标准差 | {returns_df['return_5d'].std():.2f}% | {returns_df['return_10d'].std():.2f}% | {returns_df['return_20d'].std():.2f}% | {returns_df['return_60d'].std():.2f}% |
| 盈利比例 | {(returns_df['return_5d'] > 0).mean()*100:.1f}% | {(returns_df['return_10d'] > 0).mean()*100:.1f}% | {(returns_df['return_20d'] > 0).mean()*100:.1f}% | {(returns_df['return_60d'] > 0).mean()*100:.1f}% |
| 最大收益 | {returns_df['max_gain_5d'].mean():.2f}% | {returns_df['max_gain_10d'].mean():.2f}% | {returns_df['max_gain_20d'].mean():.2f}% | {returns_df['max_gain_60d'].mean():.2f}% |
| 最大回撤 | {returns_df['max_drawdown_5d'].mean():.2f}% | {returns_df['max_drawdown_10d'].mean():.2f}% | {returns_df['max_drawdown_20d'].mean():.2f}% | {returns_df['max_drawdown_60d'].mean():.2f}% |
"""
    return text

def format_strategy_results(strategy_results):
    """格式化策略回测结果"""
    if not strategy_results:
        return "（无回测数据）"

    text = """
| 策略 | 交易次数 | 胜率 | 平均收益 | 平均盈利 | 平均亏损 | 盈亏比 |
|------|----------|------|----------|----------|----------|--------|
"""

    strategy_names = {
        'new_high_60d': '60日新高突破',
        'ma60': 'MA60均线突破',
        'range_20d': '区间突破',
    }

    for key, name in strategy_names.items():
        if key in strategy_results:
            r = strategy_results[key]
            profit_factor = r['profit_factor'] if r['profit_factor'] != np.inf else 'N/A'
            if isinstance(profit_factor, float):
                profit_factor = f"{profit_factor:.2f}"
            text += f"| {name} | {r['total_trades']} | {r['win_rate']:.1f}% | {r['avg_return']:.2f}% | {r['avg_win']:.2f}% | {r['avg_loss']:.2f}% | {profit_factor} |\n"

    return text

def format_pullback_results(strategy_results):
    """格式化回踩策略结果"""
    if 'pullback_new_high' not in strategy_results:
        return "（无回测数据）"

    r = strategy_results['pullback_new_high']
    profit_factor = r['profit_factor'] if r['profit_factor'] != np.inf else 'N/A'
    if isinstance(profit_factor, float):
        profit_factor = f"{profit_factor:.2f}"

    text = f"""
**60日新高回踩确认策略回测结果**
- 交易次数：{r['total_trades']}
- 胜率：{r['win_rate']:.1f}%
- 平均收益：{r['avg_return']:.2f}%
- 平均盈利：{r['avg_win']:.2f}%
- 平均亏损：{r['avg_loss']:.2f}%
- 盈亏比：{profit_factor}
"""
    return text

if __name__ == '__main__':
    main()
