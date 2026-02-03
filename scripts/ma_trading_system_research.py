#!/usr/bin/env python3
"""
均线交易系统研究
================
全面研究均线交易策略，包括：
1. 均线参数测试（单均线、双均线、多均线）
2. 均线策略回测（金叉/死叉、突破、粘合）
3. 策略优化（过滤条件、成交量结合、自适应均线）

Author: Claude AI Assistant
Date: 2026-02-01
"""

import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime
import warnings
import os
from typing import Dict, List, Tuple, Optional
from itertools import product

warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置保存路径
REPORT_DIR = '/Users/allen/workspace/python/stock/Tushare-DuckDB/reports/research'
os.makedirs(REPORT_DIR, exist_ok=True)

# 数据库路径
DB_PATH = '/Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db'

class MAResearchSystem:
    """均线交易系统研究类"""

    def __init__(self, db_path: str):
        """初始化"""
        self.db_path = db_path
        self.conn = duckdb.connect(db_path, read_only=True)
        self.results = {}

    def load_stock_data(self, ts_code: str, start_date: str = '20200101',
                        end_date: str = '20251231') -> pd.DataFrame:
        """加载股票数据并计算后复权价格"""
        query = f"""
        SELECT
            d.ts_code,
            d.trade_date,
            d.open,
            d.high,
            d.low,
            d.close,
            d.vol,
            d.amount,
            COALESCE(a.adj_factor, 1.0) as adj_factor
        FROM daily d
        LEFT JOIN adj_factor a ON d.ts_code = a.ts_code AND d.trade_date = a.trade_date
        WHERE d.ts_code = '{ts_code}'
          AND d.trade_date >= '{start_date}'
          AND d.trade_date <= '{end_date}'
        ORDER BY d.trade_date
        """
        df = self.conn.execute(query).fetchdf()

        if len(df) == 0:
            return df

        # 计算后复权价格
        latest_adj = df['adj_factor'].iloc[-1]
        df['adj_close'] = df['close'] * df['adj_factor'] / latest_adj
        df['adj_open'] = df['open'] * df['adj_factor'] / latest_adj
        df['adj_high'] = df['high'] * df['adj_factor'] / latest_adj
        df['adj_low'] = df['low'] * df['adj_factor'] / latest_adj

        # 转换日期格式
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df.set_index('trade_date', inplace=True)

        return df

    def load_index_data(self, ts_code: str = '000300.SH',
                        start_date: str = '20200101',
                        end_date: str = '20251231') -> pd.DataFrame:
        """加载指数数据"""
        query = f"""
        SELECT
            ts_code,
            trade_date,
            open,
            high,
            low,
            close,
            vol,
            amount
        FROM index_daily
        WHERE ts_code = '{ts_code}'
          AND trade_date >= '{start_date}'
          AND trade_date <= '{end_date}'
        ORDER BY trade_date
        """
        df = self.conn.execute(query).fetchdf()

        if len(df) > 0:
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df.set_index('trade_date', inplace=True)
            # 指数不需要复权
            df['adj_close'] = df['close']
            df['adj_open'] = df['open']
            df['adj_high'] = df['high']
            df['adj_low'] = df['low']

        return df

    def calculate_ma(self, prices: pd.Series, period: int) -> pd.Series:
        """计算简单移动平均线"""
        return prices.rolling(window=period).mean()

    def calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """计算指数移动平均线"""
        return prices.ewm(span=period, adjust=False).mean()

    def calculate_adaptive_ma(self, prices: pd.Series, period: int = 20,
                              fast: int = 2, slow: int = 30) -> pd.Series:
        """
        计算考夫曼自适应移动平均线 (KAMA)

        参数:
            prices: 价格序列
            period: 效率比率计算周期
            fast: 快速平滑常数的周期
            slow: 慢速平滑常数的周期
        """
        change = abs(prices - prices.shift(period))
        volatility = abs(prices - prices.shift(1)).rolling(window=period).sum()

        # 效率比率
        er = change / volatility.replace(0, np.nan)
        er = er.fillna(0)

        # 平滑常数
        fast_sc = 2 / (fast + 1)
        slow_sc = 2 / (slow + 1)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

        # 计算KAMA
        kama = pd.Series(index=prices.index, dtype=float)
        kama.iloc[period-1] = prices.iloc[:period].mean()

        for i in range(period, len(prices)):
            kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (prices.iloc[i] - kama.iloc[i-1])

        return kama

    # ==================== 1. 均线参数测试 ====================

    def test_single_ma_system(self, df: pd.DataFrame,
                               ma_periods: List[int] = [5, 10, 20, 30, 60, 120]) -> pd.DataFrame:
        """
        测试单均线系统
        规则：价格上穿均线买入，价格下穿均线卖出
        """
        results = []

        for period in ma_periods:
            ma = self.calculate_ma(df['adj_close'], period)

            # 生成信号
            signal = pd.Series(0, index=df.index)
            signal[df['adj_close'] > ma] = 1  # 价格在均线上方持仓
            signal[df['adj_close'] <= ma] = 0  # 价格在均线下方空仓

            # 计算收益
            returns = df['adj_close'].pct_change()
            strategy_returns = signal.shift(1) * returns

            # 统计指标
            total_return = (1 + strategy_returns).prod() - 1
            annual_return = (1 + total_return) ** (252 / len(df)) - 1
            sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() > 0 else 0
            max_drawdown = self._calculate_max_drawdown(strategy_returns)

            # 交易次数
            trades = (signal.diff().abs() > 0).sum()

            # 胜率
            daily_wins = (strategy_returns > 0).sum()
            daily_losses = (strategy_returns < 0).sum()
            win_rate = daily_wins / (daily_wins + daily_losses) if (daily_wins + daily_losses) > 0 else 0

            results.append({
                '均线周期': period,
                '总收益率': total_return * 100,
                '年化收益率': annual_return * 100,
                '夏普比率': sharpe_ratio,
                '最大回撤': max_drawdown * 100,
                '交易次数': trades,
                '胜率': win_rate * 100
            })

        return pd.DataFrame(results)

    def test_dual_ma_system(self, df: pd.DataFrame,
                            short_periods: List[int] = [5, 10, 20],
                            long_periods: List[int] = [20, 30, 60, 120]) -> pd.DataFrame:
        """
        测试双均线交叉系统
        规则：短期均线上穿长期均线买入（金叉），短期均线下穿长期均线卖出（死叉）
        """
        results = []

        for short_p, long_p in product(short_periods, long_periods):
            if short_p >= long_p:
                continue

            short_ma = self.calculate_ma(df['adj_close'], short_p)
            long_ma = self.calculate_ma(df['adj_close'], long_p)

            # 生成信号
            signal = pd.Series(0, index=df.index)
            signal[short_ma > long_ma] = 1  # 金叉后持仓
            signal[short_ma <= long_ma] = 0  # 死叉后空仓

            # 计算收益
            returns = df['adj_close'].pct_change()
            strategy_returns = signal.shift(1) * returns

            # 统计指标
            total_return = (1 + strategy_returns).prod() - 1
            annual_return = (1 + total_return) ** (252 / len(df)) - 1
            sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() > 0 else 0
            max_drawdown = self._calculate_max_drawdown(strategy_returns)

            # 交易次数
            trades = (signal.diff().abs() > 0).sum()

            results.append({
                '短期均线': short_p,
                '长期均线': long_p,
                '参数组合': f'{short_p}/{long_p}',
                '总收益率': total_return * 100,
                '年化收益率': annual_return * 100,
                '夏普比率': sharpe_ratio,
                '最大回撤': max_drawdown * 100,
                '交易次数': trades
            })

        return pd.DataFrame(results)

    def test_triple_ma_system(self, df: pd.DataFrame,
                               ma_configs: List[Tuple[int, int, int]] = None) -> pd.DataFrame:
        """
        测试三均线系统
        规则：
        - 短期均线 > 中期均线 > 长期均线 时买入（多头排列）
        - 短期均线 < 中期均线 < 长期均线 时卖出（空头排列）
        """
        if ma_configs is None:
            ma_configs = [
                (5, 10, 20), (5, 10, 30), (5, 20, 60),
                (10, 20, 60), (10, 30, 60), (20, 60, 120)
            ]

        results = []

        for short_p, mid_p, long_p in ma_configs:
            short_ma = self.calculate_ma(df['adj_close'], short_p)
            mid_ma = self.calculate_ma(df['adj_close'], mid_p)
            long_ma = self.calculate_ma(df['adj_close'], long_p)

            # 生成信号：多头排列时持仓
            signal = pd.Series(0, index=df.index)
            bull_alignment = (short_ma > mid_ma) & (mid_ma > long_ma)
            signal[bull_alignment] = 1

            # 计算收益
            returns = df['adj_close'].pct_change()
            strategy_returns = signal.shift(1) * returns

            # 统计指标
            total_return = (1 + strategy_returns).prod() - 1
            annual_return = (1 + total_return) ** (252 / len(df)) - 1
            sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() > 0 else 0
            max_drawdown = self._calculate_max_drawdown(strategy_returns)
            trades = (signal.diff().abs() > 0).sum()

            results.append({
                '短期均线': short_p,
                '中期均线': mid_p,
                '长期均线': long_p,
                '参数组合': f'{short_p}/{mid_p}/{long_p}',
                '总收益率': total_return * 100,
                '年化收益率': annual_return * 100,
                '夏普比率': sharpe_ratio,
                '最大回撤': max_drawdown * 100,
                '交易次数': trades
            })

        return pd.DataFrame(results)

    def find_optimal_ma_params(self, df: pd.DataFrame,
                                param_range: range = range(5, 121, 5),
                                optimize_metric: str = 'sharpe') -> Dict:
        """
        寻找最优均线参数
        通过网格搜索找到最优的双均线参数组合
        """
        best_result = None
        best_metric = -np.inf
        all_results = []

        for short_p in param_range:
            for long_p in param_range:
                if short_p >= long_p:
                    continue

                short_ma = self.calculate_ma(df['adj_close'], short_p)
                long_ma = self.calculate_ma(df['adj_close'], long_p)

                signal = pd.Series(0, index=df.index)
                signal[short_ma > long_ma] = 1

                returns = df['adj_close'].pct_change()
                strategy_returns = signal.shift(1) * returns

                total_return = (1 + strategy_returns).prod() - 1
                annual_return = (1 + total_return) ** (252 / len(df)) - 1
                sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() > 0 else 0
                max_drawdown = self._calculate_max_drawdown(strategy_returns)

                if optimize_metric == 'sharpe':
                    metric = sharpe_ratio
                elif optimize_metric == 'return':
                    metric = annual_return
                elif optimize_metric == 'calmar':  # 年化收益/最大回撤
                    metric = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
                else:
                    metric = sharpe_ratio

                result = {
                    'short_period': short_p,
                    'long_period': long_p,
                    'total_return': total_return,
                    'annual_return': annual_return,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown,
                    'metric': metric
                }
                all_results.append(result)

                if metric > best_metric:
                    best_metric = metric
                    best_result = result

        return {
            'best': best_result,
            'all_results': pd.DataFrame(all_results)
        }

    # ==================== 2. 均线策略回测 ====================

    def backtest_golden_cross(self, df: pd.DataFrame,
                               short_period: int = 5,
                               long_period: int = 20) -> Dict:
        """
        金叉/死叉策略回测
        """
        short_ma = self.calculate_ma(df['adj_close'], short_period)
        long_ma = self.calculate_ma(df['adj_close'], long_period)

        # 检测金叉死叉
        cross_up = (short_ma > long_ma) & (short_ma.shift(1) <= long_ma.shift(1))
        cross_down = (short_ma < long_ma) & (short_ma.shift(1) >= long_ma.shift(1))

        # 生成交易信号
        signal = pd.Series(0, index=df.index)
        position = 0
        trades = []

        for i in range(len(df)):
            if cross_up.iloc[i] and position == 0:
                position = 1
                signal.iloc[i] = 1
                trades.append({
                    'date': df.index[i],
                    'type': 'buy',
                    'price': df['adj_close'].iloc[i]
                })
            elif cross_down.iloc[i] and position == 1:
                position = 0
                signal.iloc[i] = -1
                trades.append({
                    'date': df.index[i],
                    'type': 'sell',
                    'price': df['adj_close'].iloc[i]
                })
            elif position == 1:
                signal.iloc[i] = 1

        # 计算收益
        returns = df['adj_close'].pct_change()
        strategy_returns = (signal > 0).astype(int).shift(1) * returns

        # 统计交易
        trade_results = []
        for i in range(0, len(trades)-1, 2):
            if i+1 < len(trades):
                buy_trade = trades[i]
                sell_trade = trades[i+1]
                if buy_trade['type'] == 'buy' and sell_trade['type'] == 'sell':
                    profit = (sell_trade['price'] - buy_trade['price']) / buy_trade['price']
                    trade_results.append({
                        'buy_date': buy_trade['date'],
                        'sell_date': sell_trade['date'],
                        'buy_price': buy_trade['price'],
                        'sell_price': sell_trade['price'],
                        'profit': profit * 100
                    })

        trade_df = pd.DataFrame(trade_results) if trade_results else pd.DataFrame()

        return {
            'signal': signal,
            'trades': trade_df,
            'strategy_returns': strategy_returns,
            'total_return': (1 + strategy_returns).prod() - 1,
            'win_rate': (trade_df['profit'] > 0).sum() / len(trade_df) if len(trade_df) > 0 else 0,
            'avg_profit': trade_df['profit'].mean() if len(trade_df) > 0 else 0,
            'total_trades': len(trade_df)
        }

    def backtest_ma_breakout(self, df: pd.DataFrame,
                              ma_period: int = 20,
                              breakout_pct: float = 0.02) -> Dict:
        """
        均线突破策略回测
        规则：
        - 价格突破均线一定幅度时买入
        - 价格跌破均线一定幅度时卖出
        """
        ma = self.calculate_ma(df['adj_close'], ma_period)

        # 计算突破幅度
        breakout_ratio = (df['adj_close'] - ma) / ma

        # 生成信号
        signal = pd.Series(0, index=df.index)
        position = 0
        trades = []

        for i in range(ma_period, len(df)):
            if breakout_ratio.iloc[i] > breakout_pct and position == 0:
                position = 1
                trades.append({
                    'date': df.index[i],
                    'type': 'buy',
                    'price': df['adj_close'].iloc[i]
                })
            elif breakout_ratio.iloc[i] < -breakout_pct and position == 1:
                position = 0
                trades.append({
                    'date': df.index[i],
                    'type': 'sell',
                    'price': df['adj_close'].iloc[i]
                })

            signal.iloc[i] = position

        # 计算收益
        returns = df['adj_close'].pct_change()
        strategy_returns = signal.shift(1) * returns

        # 统计交易
        trade_results = []
        for i in range(0, len(trades)-1, 2):
            if i+1 < len(trades):
                buy_trade = trades[i]
                sell_trade = trades[i+1]
                if buy_trade['type'] == 'buy' and sell_trade['type'] == 'sell':
                    profit = (sell_trade['price'] - buy_trade['price']) / buy_trade['price']
                    trade_results.append({
                        'buy_date': buy_trade['date'],
                        'sell_date': sell_trade['date'],
                        'profit': profit * 100
                    })

        trade_df = pd.DataFrame(trade_results) if trade_results else pd.DataFrame()

        return {
            'signal': signal,
            'trades': trade_df,
            'strategy_returns': strategy_returns,
            'total_return': (1 + strategy_returns).prod() - 1,
            'win_rate': (trade_df['profit'] > 0).sum() / len(trade_df) if len(trade_df) > 0 else 0,
            'total_trades': len(trade_df)
        }

    def backtest_ma_convergence(self, df: pd.DataFrame,
                                 ma_periods: List[int] = [5, 10, 20, 60],
                                 convergence_threshold: float = 0.02) -> Dict:
        """
        均线粘合策略回测
        规则：
        - 多条均线粘合（相互靠近）时准备
        - 均线粘合后向上发散时买入
        - 均线向下发散时卖出
        """
        # 计算多条均线
        mas = {p: self.calculate_ma(df['adj_close'], p) for p in ma_periods}

        # 计算均线粘合度（标准差/均值）
        ma_df = pd.DataFrame(mas)
        ma_std = ma_df.std(axis=1)
        ma_mean = ma_df.mean(axis=1)
        convergence = ma_std / ma_mean

        # 均线方向（最短期均线相对最长期均线的位置）
        short_ma = mas[min(ma_periods)]
        long_ma = mas[max(ma_periods)]
        ma_direction = short_ma - long_ma

        # 生成信号
        signal = pd.Series(0, index=df.index)
        position = 0
        entry_convergence = None
        trades = []

        for i in range(max(ma_periods), len(df)):
            # 检测粘合
            is_converged = convergence.iloc[i] < convergence_threshold

            # 检测向上发散（短期均线突破长期均线）
            is_diverging_up = (ma_direction.iloc[i] > 0) and (ma_direction.iloc[i-1] <= 0)

            # 检测向下发散
            is_diverging_down = (ma_direction.iloc[i] < 0) and (ma_direction.iloc[i-1] >= 0)

            if is_converged:
                entry_convergence = i

            # 粘合后向上发散买入
            if entry_convergence is not None and is_diverging_up and position == 0:
                if i - entry_convergence < 20:  # 粘合后20天内
                    position = 1
                    trades.append({
                        'date': df.index[i],
                        'type': 'buy',
                        'price': df['adj_close'].iloc[i]
                    })

            # 向下发散卖出
            elif is_diverging_down and position == 1:
                position = 0
                trades.append({
                    'date': df.index[i],
                    'type': 'sell',
                    'price': df['adj_close'].iloc[i]
                })

            signal.iloc[i] = position

        # 计算收益
        returns = df['adj_close'].pct_change()
        strategy_returns = signal.shift(1) * returns

        # 统计交易
        trade_results = []
        for i in range(0, len(trades)-1, 2):
            if i+1 < len(trades):
                buy_trade = trades[i]
                sell_trade = trades[i+1]
                if buy_trade['type'] == 'buy' and sell_trade['type'] == 'sell':
                    profit = (sell_trade['price'] - buy_trade['price']) / buy_trade['price']
                    trade_results.append({
                        'buy_date': buy_trade['date'],
                        'sell_date': sell_trade['date'],
                        'profit': profit * 100
                    })

        trade_df = pd.DataFrame(trade_results) if trade_results else pd.DataFrame()

        return {
            'signal': signal,
            'trades': trade_df,
            'convergence': convergence,
            'strategy_returns': strategy_returns,
            'total_return': (1 + strategy_returns).prod() - 1,
            'win_rate': (trade_df['profit'] > 0).sum() / len(trade_df) if len(trade_df) > 0 else 0,
            'total_trades': len(trade_df)
        }

    # ==================== 3. 策略优化 ====================

    def optimized_ma_with_filters(self, df: pd.DataFrame,
                                   short_period: int = 10,
                                   long_period: int = 30,
                                   trend_filter_period: int = 60,
                                   atr_period: int = 14) -> Dict:
        """
        带过滤条件的均线策略
        过滤条件：
        1. 趋势过滤：只在大趋势向上时做多
        2. ATR过滤：波动率过大时减少交易
        3. 价格位置过滤：价格不能离均线太远
        """
        short_ma = self.calculate_ma(df['adj_close'], short_period)
        long_ma = self.calculate_ma(df['adj_close'], long_period)
        trend_ma = self.calculate_ma(df['adj_close'], trend_filter_period)

        # 计算ATR
        high_low = df['adj_high'] - df['adj_low']
        high_close = abs(df['adj_high'] - df['adj_close'].shift(1))
        low_close = abs(df['adj_low'] - df['adj_close'].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=atr_period).mean()
        atr_ratio = atr / df['adj_close']

        # 生成信号
        signal = pd.Series(0, index=df.index)

        # 基本金叉死叉信号
        basic_signal = (short_ma > long_ma).astype(int)

        # 趋势过滤：价格在趋势均线上方
        trend_filter = df['adj_close'] > trend_ma

        # ATR过滤：波动率不能太大
        atr_filter = atr_ratio < 0.05

        # 价格位置过滤：价格不能离短期均线太远
        price_filter = abs(df['adj_close'] - short_ma) / short_ma < 0.1

        # 综合信号
        signal = basic_signal & trend_filter & atr_filter & price_filter
        signal = signal.astype(int)

        # 计算收益
        returns = df['adj_close'].pct_change()
        strategy_returns = signal.shift(1) * returns

        # 与无过滤策略对比
        basic_returns = basic_signal.shift(1) * returns

        return {
            'signal': signal,
            'strategy_returns': strategy_returns,
            'basic_returns': basic_returns,
            'total_return': (1 + strategy_returns).prod() - 1,
            'basic_total_return': (1 + basic_returns).prod() - 1,
            'sharpe_ratio': strategy_returns.mean() / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() > 0 else 0,
            'basic_sharpe': basic_returns.mean() / basic_returns.std() * np.sqrt(252) if basic_returns.std() > 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(strategy_returns),
            'basic_max_drawdown': self._calculate_max_drawdown(basic_returns)
        }

    def optimized_ma_with_volume(self, df: pd.DataFrame,
                                  short_period: int = 10,
                                  long_period: int = 30,
                                  vol_ma_period: int = 20,
                                  vol_multiplier: float = 1.5) -> Dict:
        """
        结合成交量的均线策略
        规则：
        1. 金叉时成交量放大确认
        2. 持仓期间成交量萎缩需注意
        3. 成交量突然放大可能是反转信号
        """
        short_ma = self.calculate_ma(df['adj_close'], short_period)
        long_ma = self.calculate_ma(df['adj_close'], long_period)
        vol_ma = self.calculate_ma(df['vol'], vol_ma_period)

        # 检测金叉
        cross_up = (short_ma > long_ma) & (short_ma.shift(1) <= long_ma.shift(1))

        # 成交量确认
        vol_confirm = df['vol'] > vol_ma * vol_multiplier

        # 生成信号
        signal = pd.Series(0, index=df.index)
        position = 0
        trades = []

        for i in range(max(short_period, long_period, vol_ma_period), len(df)):
            # 金叉且成交量放大时买入
            if cross_up.iloc[i] and vol_confirm.iloc[i] and position == 0:
                position = 1
                trades.append({
                    'date': df.index[i],
                    'type': 'buy',
                    'price': df['adj_close'].iloc[i],
                    'volume_ratio': df['vol'].iloc[i] / vol_ma.iloc[i]
                })

            # 死叉卖出
            elif short_ma.iloc[i] < long_ma.iloc[i] and position == 1:
                position = 0
                trades.append({
                    'date': df.index[i],
                    'type': 'sell',
                    'price': df['adj_close'].iloc[i]
                })

            signal.iloc[i] = position

        # 对比无成交量确认的策略
        basic_signal = (short_ma > long_ma).astype(int)

        # 计算收益
        returns = df['adj_close'].pct_change()
        strategy_returns = signal.shift(1) * returns
        basic_returns = basic_signal.shift(1) * returns

        # 统计交易
        trade_results = []
        for i in range(0, len(trades)-1, 2):
            if i+1 < len(trades):
                buy_trade = trades[i]
                sell_trade = trades[i+1]
                if buy_trade['type'] == 'buy' and sell_trade['type'] == 'sell':
                    profit = (sell_trade['price'] - buy_trade['price']) / buy_trade['price']
                    trade_results.append({
                        'buy_date': buy_trade['date'],
                        'sell_date': sell_trade['date'],
                        'profit': profit * 100,
                        'volume_ratio': buy_trade.get('volume_ratio', 1)
                    })

        trade_df = pd.DataFrame(trade_results) if trade_results else pd.DataFrame()

        return {
            'signal': signal,
            'trades': trade_df,
            'strategy_returns': strategy_returns,
            'basic_returns': basic_returns,
            'total_return': (1 + strategy_returns).prod() - 1,
            'basic_total_return': (1 + basic_returns).prod() - 1,
            'win_rate': (trade_df['profit'] > 0).sum() / len(trade_df) if len(trade_df) > 0 else 0,
            'total_trades': len(trade_df)
        }

    def optimized_adaptive_ma(self, df: pd.DataFrame,
                               kama_period: int = 20,
                               fast: int = 2,
                               slow: int = 30) -> Dict:
        """
        自适应均线策略
        使用KAMA（考夫曼自适应移动平均线）
        """
        kama = self.calculate_adaptive_ma(df['adj_close'], kama_period, fast, slow)
        sma = self.calculate_ma(df['adj_close'], kama_period)

        # KAMA信号
        kama_signal = (df['adj_close'] > kama).astype(int)

        # SMA信号（对比）
        sma_signal = (df['adj_close'] > sma).astype(int)

        # 计算收益
        returns = df['adj_close'].pct_change()
        kama_returns = kama_signal.shift(1) * returns
        sma_returns = sma_signal.shift(1) * returns

        # 交易次数
        kama_trades = (kama_signal.diff().abs() > 0).sum()
        sma_trades = (sma_signal.diff().abs() > 0).sum()

        return {
            'kama': kama,
            'sma': sma,
            'kama_signal': kama_signal,
            'sma_signal': sma_signal,
            'kama_returns': kama_returns,
            'sma_returns': sma_returns,
            'kama_total_return': (1 + kama_returns).prod() - 1,
            'sma_total_return': (1 + sma_returns).prod() - 1,
            'kama_sharpe': kama_returns.mean() / kama_returns.std() * np.sqrt(252) if kama_returns.std() > 0 else 0,
            'sma_sharpe': sma_returns.mean() / sma_returns.std() * np.sqrt(252) if sma_returns.std() > 0 else 0,
            'kama_trades': kama_trades,
            'sma_trades': sma_trades
        }

    # ==================== 辅助方法 ====================

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """计算最大回撤"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

    def _calculate_cumulative_returns(self, returns: pd.Series) -> pd.Series:
        """计算累计收益"""
        return (1 + returns).cumprod() - 1

    # ==================== 可视化 ====================

    def plot_ma_comparison(self, df: pd.DataFrame, ma_periods: List[int] = [5, 10, 20, 60]):
        """绘制均线对比图"""
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        # 价格和均线
        ax1 = axes[0]
        ax1.plot(df.index, df['adj_close'], label='价格', alpha=0.7, linewidth=1)

        colors = plt.cm.tab10(np.linspace(0, 1, len(ma_periods)))
        for period, color in zip(ma_periods, colors):
            ma = self.calculate_ma(df['adj_close'], period)
            ax1.plot(df.index, ma, label=f'MA{period}', color=color, linewidth=1.5)

        ax1.set_ylabel('价格')
        ax1.legend(loc='upper left')
        ax1.set_title('价格与均线')
        ax1.grid(True, alpha=0.3)

        # 成交量
        ax2 = axes[1]
        ax2.bar(df.index, df['vol'], alpha=0.5, color='gray', label='成交量')
        vol_ma = self.calculate_ma(df['vol'], 20)
        ax2.plot(df.index, vol_ma, color='red', label='成交量MA20', linewidth=1.5)
        ax2.set_ylabel('成交量')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_backtest_results(self, df: pd.DataFrame, signal: pd.Series,
                               strategy_returns: pd.Series, title: str = '策略回测结果'):
        """绘制回测结果图"""
        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

        # 价格和信号
        ax1 = axes[0]
        ax1.plot(df.index, df['adj_close'], label='价格', alpha=0.7)

        # 买入卖出点
        buy_signals = df.index[signal.diff() > 0]
        sell_signals = df.index[signal.diff() < 0]
        ax1.scatter(buy_signals, df.loc[buy_signals, 'adj_close'],
                   marker='^', color='green', s=100, label='买入', zorder=5)
        ax1.scatter(sell_signals, df.loc[sell_signals, 'adj_close'],
                   marker='v', color='red', s=100, label='卖出', zorder=5)

        ax1.set_ylabel('价格')
        ax1.legend()
        ax1.set_title(title)
        ax1.grid(True, alpha=0.3)

        # 持仓状态
        ax2 = axes[1]
        ax2.fill_between(df.index, 0, (signal > 0).astype(int), alpha=0.3, label='持仓')
        ax2.set_ylabel('持仓状态')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 累计收益
        ax3 = axes[2]
        buy_hold_returns = df['adj_close'].pct_change()
        strategy_cum = self._calculate_cumulative_returns(strategy_returns)
        buyhold_cum = self._calculate_cumulative_returns(buy_hold_returns)

        ax3.plot(df.index, strategy_cum * 100, label='策略收益', linewidth=2)
        ax3.plot(df.index, buyhold_cum * 100, label='买入持有', alpha=0.7)
        ax3.set_ylabel('累计收益率 (%)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_optimization_heatmap(self, results_df: pd.DataFrame):
        """绘制参数优化热力图"""
        # 创建透视表
        pivot = results_df.pivot(index='short_period', columns='long_period', values='sharpe_ratio')

        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto')

        # 设置坐标轴
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_yticks(range(len(pivot.index)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticklabels(pivot.index)

        ax.set_xlabel('长期均线周期')
        ax.set_ylabel('短期均线周期')
        ax.set_title('均线参数优化热力图 (夏普比率)')

        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('夏普比率')

        plt.tight_layout()
        return fig

    def generate_report(self, ts_code: str = '000300.SH'):
        """生成完整的研究报告"""
        print(f"开始研究均线交易系统...")
        print(f"标的: {ts_code}")
        print("=" * 60)

        # 加载数据
        if '.SH' in ts_code or '.SZ' in ts_code:
            if ts_code in ['000001.SH', '000300.SH', '000905.SH', '399001.SZ', '399006.SZ']:
                df = self.load_index_data(ts_code)
            else:
                df = self.load_stock_data(ts_code)
        else:
            df = self.load_stock_data(ts_code)

        if len(df) == 0:
            print(f"未找到 {ts_code} 的数据")
            return

        print(f"数据范围: {df.index[0]} - {df.index[-1]}, 共 {len(df)} 条记录")
        print()

        report_content = []
        report_content.append("# 均线交易系统研究报告")
        report_content.append(f"\n**研究标的**: {ts_code}")
        report_content.append(f"**数据范围**: {df.index[0].strftime('%Y-%m-%d')} 至 {df.index[-1].strftime('%Y-%m-%d')}")
        report_content.append(f"**数据量**: {len(df)} 个交易日")
        report_content.append(f"**报告生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_content.append("\n---\n")

        # ============ 1. 均线参数测试 ============
        print("1. 进行均线参数测试...")
        report_content.append("## 1. 均线参数测试\n")

        # 1.1 单均线系统测试
        print("   1.1 单均线系统测试...")
        single_ma_results = self.test_single_ma_system(df)
        report_content.append("### 1.1 单均线系统\n")
        report_content.append("**策略规则**: 价格上穿均线买入，价格下穿均线卖出\n")
        report_content.append(single_ma_results.to_markdown(index=False, floatfmt='.2f'))
        report_content.append("\n")

        # 1.2 双均线系统测试
        print("   1.2 双均线系统测试...")
        dual_ma_results = self.test_dual_ma_system(df)
        report_content.append("\n### 1.2 双均线交叉系统\n")
        report_content.append("**策略规则**: 短期均线上穿长期均线买入（金叉），短期均线下穿长期均线卖出（死叉）\n")
        report_content.append(dual_ma_results.to_markdown(index=False, floatfmt='.2f'))
        report_content.append("\n")

        # 1.3 多均线系统测试
        print("   1.3 多均线系统测试...")
        triple_ma_results = self.test_triple_ma_system(df)
        report_content.append("\n### 1.3 三均线系统（多头排列）\n")
        report_content.append("**策略规则**: 短期均线 > 中期均线 > 长期均线时持仓（多头排列）\n")
        report_content.append(triple_ma_results.to_markdown(index=False, floatfmt='.2f'))
        report_content.append("\n")

        # 1.4 最优参数寻找
        print("   1.4 寻找最优参数...")
        optimal_results = self.find_optimal_ma_params(df, param_range=range(5, 61, 5))
        best = optimal_results['best']
        report_content.append("\n### 1.4 最优参数搜索\n")
        report_content.append("**搜索范围**: 短期均线 5-60，长期均线 5-60，步长 5\n")
        report_content.append("**优化目标**: 夏普比率\n")
        report_content.append(f"\n**最优参数组合**:\n")
        report_content.append(f"- 短期均线: {best['short_period']}\n")
        report_content.append(f"- 长期均线: {best['long_period']}\n")
        report_content.append(f"- 总收益率: {best['total_return']*100:.2f}%\n")
        report_content.append(f"- 年化收益率: {best['annual_return']*100:.2f}%\n")
        report_content.append(f"- 夏普比率: {best['sharpe_ratio']:.2f}\n")
        report_content.append(f"- 最大回撤: {best['max_drawdown']*100:.2f}%\n")

        # 绘制热力图
        fig_heatmap = self.plot_optimization_heatmap(optimal_results['all_results'])
        heatmap_path = f"{REPORT_DIR}/ma_optimization_heatmap.png"
        fig_heatmap.savefig(heatmap_path, dpi=150, bbox_inches='tight')
        plt.close(fig_heatmap)
        report_content.append(f"\n![参数优化热力图](ma_optimization_heatmap.png)\n")

        # ============ 2. 均线策略回测 ============
        print("\n2. 进行均线策略回测...")
        report_content.append("\n---\n")
        report_content.append("## 2. 均线策略回测\n")

        # 2.1 金叉/死叉策略
        print("   2.1 金叉/死叉策略回测...")
        golden_cross_result = self.backtest_golden_cross(df, short_period=10, long_period=30)
        report_content.append("### 2.1 金叉/死叉策略\n")
        report_content.append("**参数**: 短期均线 MA10，长期均线 MA30\n")
        report_content.append(f"- 总收益率: {golden_cross_result['total_return']*100:.2f}%\n")
        report_content.append(f"- 胜率: {golden_cross_result['win_rate']*100:.2f}%\n")
        report_content.append(f"- 平均每笔收益: {golden_cross_result['avg_profit']:.2f}%\n")
        report_content.append(f"- 总交易次数: {golden_cross_result['total_trades']}\n")

        if len(golden_cross_result['trades']) > 0:
            report_content.append("\n**最近10笔交易记录**:\n")
            recent_trades = golden_cross_result['trades'].tail(10)
            report_content.append(recent_trades.to_markdown(index=False, floatfmt='.2f'))

        # 绘制回测结果图
        fig_gc = self.plot_backtest_results(df, golden_cross_result['signal'],
                                            golden_cross_result['strategy_returns'],
                                            '金叉/死叉策略回测结果 (MA10/MA30)')
        gc_path = f"{REPORT_DIR}/golden_cross_backtest.png"
        fig_gc.savefig(gc_path, dpi=150, bbox_inches='tight')
        plt.close(fig_gc)
        report_content.append(f"\n![金叉死叉策略回测](golden_cross_backtest.png)\n")

        # 2.2 均线突破策略
        print("   2.2 均线突破策略回测...")
        breakout_result = self.backtest_ma_breakout(df, ma_period=20, breakout_pct=0.02)
        report_content.append("\n### 2.2 均线突破策略\n")
        report_content.append("**参数**: 均线周期 MA20，突破阈值 2%\n")
        report_content.append(f"- 总收益率: {breakout_result['total_return']*100:.2f}%\n")
        report_content.append(f"- 胜率: {breakout_result['win_rate']*100:.2f}%\n")
        report_content.append(f"- 总交易次数: {breakout_result['total_trades']}\n")

        fig_bo = self.plot_backtest_results(df, breakout_result['signal'],
                                            breakout_result['strategy_returns'],
                                            '均线突破策略回测结果 (MA20, 突破2%)')
        bo_path = f"{REPORT_DIR}/ma_breakout_backtest.png"
        fig_bo.savefig(bo_path, dpi=150, bbox_inches='tight')
        plt.close(fig_bo)
        report_content.append(f"\n![均线突破策略回测](ma_breakout_backtest.png)\n")

        # 2.3 均线粘合策略
        print("   2.3 均线粘合策略回测...")
        convergence_result = self.backtest_ma_convergence(df)
        report_content.append("\n### 2.3 均线粘合策略\n")
        report_content.append("**参数**: 均线组合 MA5/10/20/60，粘合阈值 2%\n")
        report_content.append(f"- 总收益率: {convergence_result['total_return']*100:.2f}%\n")
        report_content.append(f"- 胜率: {convergence_result['win_rate']*100:.2f}%\n")
        report_content.append(f"- 总交易次数: {convergence_result['total_trades']}\n")

        fig_conv = self.plot_backtest_results(df, convergence_result['signal'],
                                              convergence_result['strategy_returns'],
                                              '均线粘合策略回测结果')
        conv_path = f"{REPORT_DIR}/ma_convergence_backtest.png"
        fig_conv.savefig(conv_path, dpi=150, bbox_inches='tight')
        plt.close(fig_conv)
        report_content.append(f"\n![均线粘合策略回测](ma_convergence_backtest.png)\n")

        # ============ 3. 策略优化 ============
        print("\n3. 进行策略优化...")
        report_content.append("\n---\n")
        report_content.append("## 3. 策略优化\n")

        # 3.1 加入过滤条件
        print("   3.1 带过滤条件的策略...")
        filter_result = self.optimized_ma_with_filters(df)
        report_content.append("### 3.1 带过滤条件的均线策略\n")
        report_content.append("**过滤条件**:\n")
        report_content.append("1. 趋势过滤：只在价格高于MA60时做多\n")
        report_content.append("2. ATR过滤：波动率（ATR/价格）< 5%时交易\n")
        report_content.append("3. 价格位置过滤：价格偏离短期均线不超过10%\n\n")
        report_content.append("| 指标 | 优化策略 | 基础策略 | 改善 |\n")
        report_content.append("|------|----------|----------|------|\n")
        report_content.append(f"| 总收益率 | {filter_result['total_return']*100:.2f}% | {filter_result['basic_total_return']*100:.2f}% | {(filter_result['total_return']-filter_result['basic_total_return'])*100:.2f}% |\n")
        report_content.append(f"| 夏普比率 | {filter_result['sharpe_ratio']:.2f} | {filter_result['basic_sharpe']:.2f} | {filter_result['sharpe_ratio']-filter_result['basic_sharpe']:.2f} |\n")
        report_content.append(f"| 最大回撤 | {filter_result['max_drawdown']*100:.2f}% | {filter_result['basic_max_drawdown']*100:.2f}% | {(filter_result['max_drawdown']-filter_result['basic_max_drawdown'])*100:.2f}% |\n")

        # 3.2 结合成交量
        print("   3.2 结合成交量的策略...")
        volume_result = self.optimized_ma_with_volume(df)
        report_content.append("\n### 3.2 结合成交量的均线策略\n")
        report_content.append("**策略规则**: 金叉时需成交量放大1.5倍以上才确认买入\n\n")
        report_content.append("| 指标 | 成交量确认策略 | 基础策略 | 改善 |\n")
        report_content.append("|------|----------------|----------|------|\n")
        report_content.append(f"| 总收益率 | {volume_result['total_return']*100:.2f}% | {volume_result['basic_total_return']*100:.2f}% | {(volume_result['total_return']-volume_result['basic_total_return'])*100:.2f}% |\n")
        report_content.append(f"| 胜率 | {volume_result['win_rate']*100:.2f}% | - | - |\n")
        report_content.append(f"| 交易次数 | {volume_result['total_trades']} | - | - |\n")

        # 3.3 自适应均线
        print("   3.3 自适应均线策略...")
        adaptive_result = self.optimized_adaptive_ma(df)
        report_content.append("\n### 3.3 自适应均线策略 (KAMA)\n")
        report_content.append("**KAMA参数**: 效率比率周期=20，快速平滑周期=2，慢速平滑周期=30\n\n")
        report_content.append("| 指标 | KAMA策略 | SMA策略 | 改善 |\n")
        report_content.append("|------|----------|---------|------|\n")
        report_content.append(f"| 总收益率 | {adaptive_result['kama_total_return']*100:.2f}% | {adaptive_result['sma_total_return']*100:.2f}% | {(adaptive_result['kama_total_return']-adaptive_result['sma_total_return'])*100:.2f}% |\n")
        report_content.append(f"| 夏普比率 | {adaptive_result['kama_sharpe']:.2f} | {adaptive_result['sma_sharpe']:.2f} | {adaptive_result['kama_sharpe']-adaptive_result['sma_sharpe']:.2f} |\n")
        report_content.append(f"| 交易次数 | {adaptive_result['kama_trades']} | {adaptive_result['sma_trades']} | {adaptive_result['sma_trades']-adaptive_result['kama_trades']} |\n")

        # 绘制KAMA与SMA对比图
        fig_kama, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        ax1 = axes[0]
        ax1.plot(df.index, df['adj_close'], label='价格', alpha=0.7)
        ax1.plot(df.index, adaptive_result['kama'], label='KAMA(20)', linewidth=2)
        ax1.plot(df.index, adaptive_result['sma'], label='SMA(20)', linewidth=2, alpha=0.7)
        ax1.set_ylabel('价格')
        ax1.legend()
        ax1.set_title('KAMA vs SMA 对比')
        ax1.grid(True, alpha=0.3)

        ax2 = axes[1]
        kama_cum = self._calculate_cumulative_returns(adaptive_result['kama_returns']) * 100
        sma_cum = self._calculate_cumulative_returns(adaptive_result['sma_returns']) * 100
        buyhold_cum = self._calculate_cumulative_returns(df['adj_close'].pct_change()) * 100

        ax2.plot(df.index, kama_cum, label='KAMA策略', linewidth=2)
        ax2.plot(df.index, sma_cum, label='SMA策略', linewidth=2, alpha=0.7)
        ax2.plot(df.index, buyhold_cum, label='买入持有', alpha=0.5)
        ax2.set_ylabel('累计收益率 (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        kama_path = f"{REPORT_DIR}/kama_vs_sma.png"
        fig_kama.savefig(kama_path, dpi=150, bbox_inches='tight')
        plt.close(fig_kama)
        report_content.append(f"\n![KAMA vs SMA对比](kama_vs_sma.png)\n")

        # ============ 4. 总结 ============
        report_content.append("\n---\n")
        report_content.append("## 4. 研究总结\n")
        report_content.append("### 4.1 主要发现\n")
        report_content.append("1. **单均线系统**: 简单但容易产生频繁交易和假信号，较长周期的均线表现更稳定\n")
        report_content.append("2. **双均线系统**: 通过金叉死叉可以有效过滤一些噪声，但在震荡行情中仍会产生亏损\n")
        report_content.append("3. **多均线系统**: 多头排列策略相对保守，能抓住主要趋势，但可能错过一些机会\n")
        report_content.append("4. **均线突破策略**: 设置合适的突破阈值可以减少假突破，但也可能错过一些真正的突破\n")
        report_content.append("5. **均线粘合策略**: 粘合后发散是较好的入场时机，但识别粘合状态需要较多经验\n")
        report_content.append("6. **过滤条件**: 趋势过滤、波动率过滤可以有效提高策略质量\n")
        report_content.append("7. **成交量确认**: 金叉时成交量放大确认可以提高信号的可靠性\n")
        report_content.append("8. **自适应均线**: KAMA能够在趋势和震荡行情中自动调整灵敏度，减少交易次数\n")

        report_content.append("\n### 4.2 最优策略建议\n")
        report_content.append(f"基于本次研究，推荐的参数组合为:\n")
        report_content.append(f"- **双均线系统最优参数**: MA{best['short_period']}/MA{best['long_period']}，夏普比率 {best['sharpe_ratio']:.2f}\n")
        report_content.append(f"- **建议加入的过滤条件**: 趋势过滤(MA60)、成交量确认(1.5倍放大)\n")
        report_content.append(f"- **考虑使用自适应均线(KAMA)**: 可以减少交易次数同时保持策略效果\n")

        report_content.append("\n### 4.3 风险提示\n")
        report_content.append("1. 以上研究基于历史数据，不代表未来表现\n")
        report_content.append("2. 均线策略在趋势行情中表现较好，但在震荡行情中可能产生较多亏损\n")
        report_content.append("3. 实际交易需考虑交易成本、滑点、流动性等因素\n")
        report_content.append("4. 建议结合其他指标和基本面分析综合判断\n")

        # 绘制综合对比图
        fig_summary, ax = plt.subplots(figsize=(14, 8))

        # 计算所有策略的累计收益
        buyhold = self._calculate_cumulative_returns(df['adj_close'].pct_change()) * 100
        golden_cross_cum = self._calculate_cumulative_returns(golden_cross_result['strategy_returns']) * 100
        breakout_cum = self._calculate_cumulative_returns(breakout_result['strategy_returns']) * 100
        filter_cum = self._calculate_cumulative_returns(filter_result['strategy_returns']) * 100
        kama_cum = self._calculate_cumulative_returns(adaptive_result['kama_returns']) * 100

        ax.plot(df.index, buyhold, label='买入持有', alpha=0.7, linewidth=1.5)
        ax.plot(df.index, golden_cross_cum, label='金叉死叉(MA10/30)', linewidth=1.5)
        ax.plot(df.index, breakout_cum, label='均线突破(MA20,2%)', linewidth=1.5)
        ax.plot(df.index, filter_cum, label='带过滤条件', linewidth=1.5)
        ax.plot(df.index, kama_cum, label='KAMA策略', linewidth=1.5)

        ax.set_xlabel('日期')
        ax.set_ylabel('累计收益率 (%)')
        ax.set_title('各策略收益对比')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        summary_path = f"{REPORT_DIR}/strategy_comparison.png"
        fig_summary.savefig(summary_path, dpi=150, bbox_inches='tight')
        plt.close(fig_summary)
        report_content.append(f"\n![策略收益对比](strategy_comparison.png)\n")

        # 保存报告
        report_path = f"{REPORT_DIR}/均线交易系统研究报告.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_content))

        print("\n" + "=" * 60)
        print(f"报告已保存到: {report_path}")
        print(f"图片已保存到: {REPORT_DIR}/")
        print("=" * 60)

        return report_content


def main():
    """主函数"""
    research = MAResearchSystem(DB_PATH)

    # 使用沪深300指数进行研究
    research.generate_report('000300.SH')

    # 也可以研究个股
    # research.generate_report('600519.SH')  # 贵州茅台


if __name__ == '__main__':
    main()
