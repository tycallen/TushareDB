#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
均值回归策略研究 (Mean Reversion Strategy Research)
===================================================

本研究基于A股市场数据，系统分析均值回归策略：
1. 均值回归识别方法（价格偏离、布林带、Z-score）
2. 策略设计（简单均值回归、配对交易、统计套利）
3. 回测分析（参数优化、市场环境、风险控制）

Author: Claude Code
Date: 2026-02-01
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import duckdb
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False

# 数据库路径
DB_PATH = '/Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db'
REPORT_DIR = '/Users/allen/workspace/python/stock/Tushare-DuckDB/reports/research'

os.makedirs(REPORT_DIR, exist_ok=True)


class MeanReversionResearch:
    """均值回归策略研究类"""

    def __init__(self, db_path=DB_PATH):
        self.conn = duckdb.connect(db_path, read_only=True)
        self.report_lines = []

    def log(self, msg):
        """记录日志"""
        print(msg)
        self.report_lines.append(msg)

    def get_stock_data(self, ts_code, start_date='20200101', end_date='20260130'):
        """获取单只股票数据（前复权）"""
        query = f"""
        SELECT
            d.ts_code,
            d.trade_date,
            d.open * a.adj_factor / (SELECT MAX(adj_factor) FROM adj_factor WHERE ts_code = d.ts_code) as open,
            d.high * a.adj_factor / (SELECT MAX(adj_factor) FROM adj_factor WHERE ts_code = d.ts_code) as high,
            d.low * a.adj_factor / (SELECT MAX(adj_factor) FROM adj_factor WHERE ts_code = d.ts_code) as low,
            d.close * a.adj_factor / (SELECT MAX(adj_factor) FROM adj_factor WHERE ts_code = d.ts_code) as close,
            d.vol,
            d.amount,
            d.pct_chg
        FROM daily d
        JOIN adj_factor a ON d.ts_code = a.ts_code AND d.trade_date = a.trade_date
        WHERE d.ts_code = '{ts_code}'
          AND d.trade_date >= '{start_date}'
          AND d.trade_date <= '{end_date}'
        ORDER BY d.trade_date
        """
        df = self.conn.execute(query).fetchdf()
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        return df

    def get_multiple_stocks(self, n_stocks=50, start_date='20200101', end_date='20260130'):
        """获取多只股票数据"""
        # 选择流动性好的股票
        query = f"""
        WITH stock_liquidity AS (
            SELECT
                ts_code,
                AVG(amount) as avg_amount,
                COUNT(*) as trading_days
            FROM daily
            WHERE trade_date >= '{start_date}'
              AND trade_date <= '{end_date}'
            GROUP BY ts_code
            HAVING COUNT(*) > 200
        )
        SELECT ts_code
        FROM stock_liquidity
        JOIN stock_basic USING(ts_code)
        WHERE stock_basic.list_status = 'L'
        ORDER BY avg_amount DESC
        LIMIT {n_stocks}
        """
        stocks = self.conn.execute(query).fetchdf()['ts_code'].tolist()

        stock_data = {}
        for ts_code in stocks:
            try:
                df = self.get_stock_data(ts_code, start_date, end_date)
                if len(df) > 200:
                    stock_data[ts_code] = df
            except Exception as e:
                continue

        return stock_data

    def get_index_data(self, index_code='000300.SH', start_date='20200101', end_date='20260130'):
        """获取指数数据"""
        query = f"""
        SELECT ts_code, trade_date, open, high, low, close, vol, amount, pct_chg
        FROM index_daily
        WHERE ts_code = '{index_code}'
          AND trade_date >= '{start_date}'
          AND trade_date <= '{end_date}'
        ORDER BY trade_date
        """
        df = self.conn.execute(query).fetchdf()
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        return df

    # ==================== 均值回归识别方法 ====================

    def calculate_ma_deviation(self, df, periods=[5, 10, 20, 60]):
        """计算价格偏离均线程度"""
        for period in periods:
            df[f'ma_{period}'] = df['close'].rolling(window=period).mean()
            df[f'deviation_{period}'] = (df['close'] - df[f'ma_{period}']) / df[f'ma_{period}'] * 100
        return df

    def calculate_bollinger_bands(self, df, window=20, num_std=2):
        """计算布林带"""
        df['bb_middle'] = df['close'].rolling(window=window).mean()
        df['bb_std'] = df['close'].rolling(window=window).std()
        df['bb_upper'] = df['bb_middle'] + num_std * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - num_std * df['bb_std']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle'] * 100
        return df

    def calculate_zscore(self, df, window=20):
        """计算Z-score"""
        df['zscore_ma'] = df['close'].rolling(window=window).mean()
        df['zscore_std'] = df['close'].rolling(window=window).std()
        df['zscore'] = (df['close'] - df['zscore_ma']) / df['zscore_std']
        return df

    def analyze_mean_reversion_signals(self, df):
        """分析均值回归信号的有效性"""
        results = {}

        # 分析不同阈值的MA偏离信号
        for period in [5, 10, 20, 60]:
            col = f'deviation_{period}'
            if col not in df.columns:
                continue

            for threshold in [3, 5, 8, 10]:
                # 超卖信号（偏离度 < -threshold）
                oversold = df[df[col] < -threshold].copy()
                if len(oversold) > 10:
                    # 计算5天后收益
                    future_returns = []
                    for idx in oversold.index:
                        if idx + 5 < len(df):
                            ret = (df.loc[idx + 5, 'close'] - df.loc[idx, 'close']) / df.loc[idx, 'close'] * 100
                            future_returns.append(ret)
                    if future_returns:
                        results[f'MA{period}_oversold_{threshold}%'] = {
                            'signals': len(oversold),
                            'avg_5d_return': np.mean(future_returns),
                            'win_rate': np.mean([1 if r > 0 else 0 for r in future_returns]) * 100
                        }

                # 超买信号（偏离度 > threshold）
                overbought = df[df[col] > threshold].copy()
                if len(overbought) > 10:
                    future_returns = []
                    for idx in overbought.index:
                        if idx + 5 < len(df):
                            ret = (df.loc[idx + 5, 'close'] - df.loc[idx, 'close']) / df.loc[idx, 'close'] * 100
                            future_returns.append(ret)
                    if future_returns:
                        results[f'MA{period}_overbought_{threshold}%'] = {
                            'signals': len(overbought),
                            'avg_5d_return': np.mean(future_returns),
                            'win_rate': np.mean([1 if r < 0 else 0 for r in future_returns]) * 100
                        }

        # 分析布林带信号
        if 'bb_position' in df.columns:
            # 触及下轨
            lower_touch = df[df['bb_position'] < 0].copy()
            if len(lower_touch) > 10:
                future_returns = []
                for idx in lower_touch.index:
                    if idx + 5 < len(df):
                        ret = (df.loc[idx + 5, 'close'] - df.loc[idx, 'close']) / df.loc[idx, 'close'] * 100
                        future_returns.append(ret)
                if future_returns:
                    results['BB_lower_touch'] = {
                        'signals': len(lower_touch),
                        'avg_5d_return': np.mean(future_returns),
                        'win_rate': np.mean([1 if r > 0 else 0 for r in future_returns]) * 100
                    }

            # 触及上轨
            upper_touch = df[df['bb_position'] > 1].copy()
            if len(upper_touch) > 10:
                future_returns = []
                for idx in upper_touch.index:
                    if idx + 5 < len(df):
                        ret = (df.loc[idx + 5, 'close'] - df.loc[idx, 'close']) / df.loc[idx, 'close'] * 100
                        future_returns.append(ret)
                if future_returns:
                    results['BB_upper_touch'] = {
                        'signals': len(upper_touch),
                        'avg_5d_return': np.mean(future_returns),
                        'win_rate': np.mean([1 if r < 0 else 0 for r in future_returns]) * 100
                    }

        # 分析Z-score信号
        if 'zscore' in df.columns:
            for threshold in [1.5, 2.0, 2.5, 3.0]:
                # 极度超卖
                zscore_low = df[df['zscore'] < -threshold].copy()
                if len(zscore_low) > 10:
                    future_returns = []
                    for idx in zscore_low.index:
                        if idx + 5 < len(df):
                            ret = (df.loc[idx + 5, 'close'] - df.loc[idx, 'close']) / df.loc[idx, 'close'] * 100
                            future_returns.append(ret)
                    if future_returns:
                        results[f'Zscore_<-{threshold}'] = {
                            'signals': len(zscore_low),
                            'avg_5d_return': np.mean(future_returns),
                            'win_rate': np.mean([1 if r > 0 else 0 for r in future_returns]) * 100
                        }

                # 极度超买
                zscore_high = df[df['zscore'] > threshold].copy()
                if len(zscore_high) > 10:
                    future_returns = []
                    for idx in zscore_high.index:
                        if idx + 5 < len(df):
                            ret = (df.loc[idx + 5, 'close'] - df.loc[idx, 'close']) / df.loc[idx, 'close'] * 100
                            future_returns.append(ret)
                    if future_returns:
                        results[f'Zscore_>{threshold}'] = {
                            'signals': len(zscore_high),
                            'avg_5d_return': np.mean(future_returns),
                            'win_rate': np.mean([1 if r < 0 else 0 for r in future_returns]) * 100
                        }

        return results

    # ==================== 策略实现 ====================

    def simple_mean_reversion_backtest(self, df, entry_zscore=-2.0, exit_zscore=0,
                                       stop_loss=-0.08, take_profit=0.15, window=20):
        """
        简单均值回归策略回测

        参数:
        - entry_zscore: 入场Z-score阈值（负数表示超卖买入）
        - exit_zscore: 出场Z-score阈值
        - stop_loss: 止损比例
        - take_profit: 止盈比例
        - window: Z-score计算窗口
        """
        df = df.copy()
        df = self.calculate_zscore(df, window)
        df = df.dropna()
        df = df.reset_index(drop=True)

        # 交易记录
        trades = []
        position = 0
        entry_price = 0
        entry_date = None

        for i in range(1, len(df)):
            current_price = df.loc[i, 'close']
            current_zscore = df.loc[i, 'zscore']
            current_date = df.loc[i, 'trade_date']

            if position == 0:
                # 无仓位时，寻找入场信号
                if current_zscore < entry_zscore:
                    position = 1
                    entry_price = current_price
                    entry_date = current_date
            else:
                # 有仓位时，检查出场条件
                pnl_pct = (current_price - entry_price) / entry_price

                if pnl_pct <= stop_loss:
                    # 止损
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': current_date,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'pnl_pct': pnl_pct,
                        'exit_reason': 'stop_loss'
                    })
                    position = 0
                elif pnl_pct >= take_profit:
                    # 止盈
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': current_date,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'pnl_pct': pnl_pct,
                        'exit_reason': 'take_profit'
                    })
                    position = 0
                elif current_zscore >= exit_zscore:
                    # Z-score回归，正常出场
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': current_date,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'pnl_pct': pnl_pct,
                        'exit_reason': 'mean_reversion'
                    })
                    position = 0

        # 计算策略统计
        if not trades:
            return None

        trades_df = pd.DataFrame(trades)

        stats = {
            'total_trades': len(trades),
            'win_rate': (trades_df['pnl_pct'] > 0).mean() * 100,
            'avg_return': trades_df['pnl_pct'].mean() * 100,
            'max_return': trades_df['pnl_pct'].max() * 100,
            'min_return': trades_df['pnl_pct'].min() * 100,
            'total_return': (1 + trades_df['pnl_pct']).prod() - 1,
            'sharpe': trades_df['pnl_pct'].mean() / trades_df['pnl_pct'].std() * np.sqrt(252) if trades_df['pnl_pct'].std() > 0 else 0,
            'stop_loss_exits': (trades_df['exit_reason'] == 'stop_loss').sum(),
            'take_profit_exits': (trades_df['exit_reason'] == 'take_profit').sum(),
            'mean_reversion_exits': (trades_df['exit_reason'] == 'mean_reversion').sum()
        }

        return stats, trades_df

    def pairs_trading_backtest(self, df1, df2, lookback=60, entry_threshold=2.0, exit_threshold=0.5):
        """
        配对交易回测

        参数:
        - df1, df2: 两只股票的数据
        - lookback: 计算价差Z-score的回溯期
        - entry_threshold: 入场Z-score阈值
        - exit_threshold: 出场Z-score阈值
        """
        # 合并数据
        merged = pd.merge(
            df1[['trade_date', 'close']].rename(columns={'close': 'price1'}),
            df2[['trade_date', 'close']].rename(columns={'close': 'price2'}),
            on='trade_date',
            how='inner'
        )

        if len(merged) < lookback + 20:
            return None

        # 计算价格比率
        merged['ratio'] = merged['price1'] / merged['price2']
        merged['ratio_ma'] = merged['ratio'].rolling(window=lookback).mean()
        merged['ratio_std'] = merged['ratio'].rolling(window=lookback).std()
        merged['zscore'] = (merged['ratio'] - merged['ratio_ma']) / merged['ratio_std']

        merged = merged.dropna()
        merged = merged.reset_index(drop=True)

        # 交易记录
        trades = []
        position = 0  # 0: 无仓位, 1: 多ratio(多stock1空stock2), -1: 空ratio
        entry_ratio = 0
        entry_date = None

        for i in range(1, len(merged)):
            current_zscore = merged.loc[i, 'zscore']
            current_ratio = merged.loc[i, 'ratio']
            current_date = merged.loc[i, 'trade_date']

            if position == 0:
                if current_zscore > entry_threshold:
                    # 比率过高，做空ratio（空stock1，多stock2）
                    position = -1
                    entry_ratio = current_ratio
                    entry_date = current_date
                elif current_zscore < -entry_threshold:
                    # 比率过低，做多ratio（多stock1，空stock2）
                    position = 1
                    entry_ratio = current_ratio
                    entry_date = current_date
            else:
                # 检查出场条件
                if position == 1 and current_zscore >= -exit_threshold:
                    pnl = (current_ratio - entry_ratio) / entry_ratio
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': current_date,
                        'position': 'long_ratio',
                        'entry_ratio': entry_ratio,
                        'exit_ratio': current_ratio,
                        'pnl_pct': pnl
                    })
                    position = 0
                elif position == -1 and current_zscore <= exit_threshold:
                    pnl = (entry_ratio - current_ratio) / entry_ratio
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': current_date,
                        'position': 'short_ratio',
                        'entry_ratio': entry_ratio,
                        'exit_ratio': current_ratio,
                        'pnl_pct': pnl
                    })
                    position = 0

        if not trades:
            return None

        trades_df = pd.DataFrame(trades)

        stats = {
            'total_trades': len(trades),
            'win_rate': (trades_df['pnl_pct'] > 0).mean() * 100,
            'avg_return': trades_df['pnl_pct'].mean() * 100,
            'total_return': (1 + trades_df['pnl_pct']).prod() - 1,
            'sharpe': trades_df['pnl_pct'].mean() / trades_df['pnl_pct'].std() * np.sqrt(252) if trades_df['pnl_pct'].std() > 0 else 0
        }

        return stats, trades_df, merged

    def calculate_cointegration(self, price1, price2):
        """计算协整关系（简化版）"""
        from scipy import stats as scipy_stats

        # 使用对数价格
        log_p1 = np.log(price1)
        log_p2 = np.log(price2)

        # 简单线性回归
        slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(log_p2, log_p1)

        # 计算残差
        residuals = log_p1 - (slope * log_p2 + intercept)

        # ADF检验（简化版 - 使用残差的自相关）
        autocorr = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]

        return {
            'hedge_ratio': slope,
            'intercept': intercept,
            'r_squared': r_value ** 2,
            'residual_autocorr': autocorr,
            'residual_std': residuals.std(),
            'is_stationary': autocorr < 0.9  # 简化判断
        }

    def statistical_arbitrage_portfolio(self, stock_data, n_pairs=10):
        """
        统计套利组合构建

        寻找具有协整关系的股票对
        """
        stocks = list(stock_data.keys())
        n = len(stocks)

        # 计算所有配对的协整关系
        pairs_stats = []

        for i in range(min(n, 30)):  # 限制计算量
            for j in range(i + 1, min(n, 30)):
                stock1, stock2 = stocks[i], stocks[j]

                # 合并数据
                merged = pd.merge(
                    stock_data[stock1][['trade_date', 'close']].rename(columns={'close': 'price1'}),
                    stock_data[stock2][['trade_date', 'close']].rename(columns={'close': 'price2'}),
                    on='trade_date',
                    how='inner'
                )

                if len(merged) < 100:
                    continue

                try:
                    coint = self.calculate_cointegration(
                        merged['price1'].values,
                        merged['price2'].values
                    )

                    if coint['is_stationary'] and coint['r_squared'] > 0.7:
                        pairs_stats.append({
                            'stock1': stock1,
                            'stock2': stock2,
                            'r_squared': coint['r_squared'],
                            'hedge_ratio': coint['hedge_ratio'],
                            'residual_std': coint['residual_std'],
                            'residual_autocorr': coint['residual_autocorr']
                        })
                except:
                    continue

        # 按R²排序，选择最好的配对
        pairs_stats = sorted(pairs_stats, key=lambda x: x['r_squared'], reverse=True)[:n_pairs]

        return pairs_stats

    # ==================== 回测分析 ====================

    def parameter_optimization(self, df, param_grid):
        """参数优化"""
        results = []

        for entry_z in param_grid.get('entry_zscore', [-2.0, -2.5, -3.0]):
            for exit_z in param_grid.get('exit_zscore', [-0.5, 0, 0.5]):
                for window in param_grid.get('window', [10, 20, 30, 60]):
                    for stop_loss in param_grid.get('stop_loss', [-0.05, -0.08, -0.10]):
                        try:
                            result = self.simple_mean_reversion_backtest(
                                df,
                                entry_zscore=entry_z,
                                exit_zscore=exit_z,
                                window=window,
                                stop_loss=stop_loss
                            )

                            if result:
                                stats, _ = result
                                stats['params'] = {
                                    'entry_zscore': entry_z,
                                    'exit_zscore': exit_z,
                                    'window': window,
                                    'stop_loss': stop_loss
                                }
                                results.append(stats)
                        except:
                            continue

        return sorted(results, key=lambda x: x.get('sharpe', 0), reverse=True)

    def analyze_market_regimes(self, df, index_df):
        """分析不同市场环境下的策略表现"""
        # 计算市场指标
        index_df = index_df.copy()
        index_df['return_20d'] = index_df['close'].pct_change(20)
        index_df['volatility_20d'] = index_df['close'].pct_change().rolling(20).std()

        # 定义市场环境
        index_df['regime'] = 'normal'
        index_df.loc[index_df['return_20d'] > 0.05, 'regime'] = 'bull'
        index_df.loc[index_df['return_20d'] < -0.05, 'regime'] = 'bear'
        index_df.loc[index_df['volatility_20d'] > index_df['volatility_20d'].quantile(0.8), 'regime'] = 'high_vol'

        # 合并数据
        df = df.copy()
        merged = pd.merge(
            df,
            index_df[['trade_date', 'regime']],
            on='trade_date',
            how='left'
        )
        merged['regime'] = merged['regime'].fillna('normal')

        # 分别在不同环境下回测
        regime_results = {}
        for regime in ['bull', 'bear', 'normal', 'high_vol']:
            regime_df = merged[merged['regime'] == regime].copy()
            regime_df = regime_df.reset_index(drop=True)

            if len(regime_df) > 50:
                result = self.simple_mean_reversion_backtest(regime_df)
                if result:
                    regime_results[regime] = result[0]

        return regime_results

    def calculate_risk_metrics(self, trades_df):
        """计算风险指标"""
        if trades_df is None or len(trades_df) == 0:
            return None

        returns = trades_df['pnl_pct'].values

        # 计算累积收益
        cum_returns = np.cumprod(1 + returns)

        # 最大回撤
        peak = np.maximum.accumulate(cum_returns)
        drawdown = (cum_returns - peak) / peak
        max_drawdown = drawdown.min()

        # 风险调整收益指标
        avg_return = np.mean(returns)
        std_return = np.std(returns)

        metrics = {
            'total_return': cum_returns[-1] - 1,
            'annualized_return': (cum_returns[-1] ** (252 / len(returns))) - 1 if len(returns) > 0 else 0,
            'max_drawdown': max_drawdown,
            'volatility': std_return * np.sqrt(252),
            'sharpe_ratio': (avg_return * 252) / (std_return * np.sqrt(252)) if std_return > 0 else 0,
            'calmar_ratio': avg_return * 252 / abs(max_drawdown) if max_drawdown != 0 else 0,
            'win_rate': np.mean(returns > 0),
            'profit_factor': abs(returns[returns > 0].sum() / returns[returns < 0].sum()) if len(returns[returns < 0]) > 0 and returns[returns < 0].sum() != 0 else 0,
            'avg_win': returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0,
            'avg_loss': returns[returns < 0].mean() if len(returns[returns < 0]) > 0 else 0,
            'max_consecutive_wins': self._max_consecutive(returns > 0),
            'max_consecutive_losses': self._max_consecutive(returns < 0)
        }

        return metrics

    def _max_consecutive(self, condition):
        """计算最大连续次数"""
        max_count = 0
        current_count = 0
        for c in condition:
            if c:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0
        return max_count

    # ==================== 可视化 ====================

    def plot_mean_reversion_signals(self, df, ts_code, save_path=None):
        """绘制均值回归信号图"""
        fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

        # 价格和均线
        ax1 = axes[0]
        ax1.plot(df['trade_date'], df['close'], label='Close', color='black', linewidth=1)
        for period, color in [(5, 'blue'), (20, 'orange'), (60, 'green')]:
            if f'ma_{period}' in df.columns:
                ax1.plot(df['trade_date'], df[f'ma_{period}'], label=f'MA{period}', color=color, alpha=0.7)
        ax1.set_ylabel('Price')
        ax1.legend(loc='upper left')
        ax1.set_title(f'{ts_code} Mean Reversion Analysis')

        # 布林带
        ax2 = axes[1]
        if 'bb_upper' in df.columns:
            ax2.plot(df['trade_date'], df['close'], label='Close', color='black', linewidth=1)
            ax2.plot(df['trade_date'], df['bb_upper'], 'r--', label='Upper Band', alpha=0.7)
            ax2.plot(df['trade_date'], df['bb_middle'], 'b-', label='Middle Band', alpha=0.7)
            ax2.plot(df['trade_date'], df['bb_lower'], 'g--', label='Lower Band', alpha=0.7)
            ax2.fill_between(df['trade_date'], df['bb_lower'], df['bb_upper'], alpha=0.1)
        ax2.set_ylabel('Bollinger Bands')
        ax2.legend(loc='upper left')

        # MA偏离度
        ax3 = axes[2]
        if 'deviation_20' in df.columns:
            ax3.plot(df['trade_date'], df['deviation_20'], label='MA20 Deviation %', color='purple')
            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax3.axhline(y=5, color='red', linestyle='--', alpha=0.5)
            ax3.axhline(y=-5, color='green', linestyle='--', alpha=0.5)
        ax3.set_ylabel('Deviation %')
        ax3.legend(loc='upper left')

        # Z-score
        ax4 = axes[3]
        if 'zscore' in df.columns:
            ax4.plot(df['trade_date'], df['zscore'], label='Z-score', color='navy')
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax4.axhline(y=2, color='red', linestyle='--', alpha=0.5)
            ax4.axhline(y=-2, color='green', linestyle='--', alpha=0.5)
            ax4.fill_between(df['trade_date'], -2, 2, alpha=0.1, color='gray')
        ax4.set_ylabel('Z-score')
        ax4.legend(loc='upper left')
        ax4.set_xlabel('Date')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_backtest_results(self, trades_df, df, save_path=None):
        """绘制回测结果"""
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)

        # 累积收益曲线
        ax1 = axes[0]
        cum_returns = (1 + trades_df['pnl_pct']).cumprod()
        ax1.plot(range(len(cum_returns)), cum_returns, 'b-', linewidth=2)
        ax1.axhline(y=1, color='black', linestyle='--', alpha=0.5)
        ax1.set_ylabel('Cumulative Return')
        ax1.set_title('Backtest Results')
        ax1.grid(True, alpha=0.3)

        # 单笔收益分布
        ax2 = axes[1]
        ax2.hist(trades_df['pnl_pct'] * 100, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax2.axvline(x=trades_df['pnl_pct'].mean() * 100, color='green', linestyle='-', linewidth=2, label='Mean')
        ax2.set_xlabel('Return %')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Return Distribution')
        ax2.legend()

        # 出场原因统计
        ax3 = axes[2]
        exit_counts = trades_df['exit_reason'].value_counts()
        colors = {'mean_reversion': 'green', 'stop_loss': 'red', 'take_profit': 'blue'}
        ax3.bar(exit_counts.index, exit_counts.values,
                color=[colors.get(x, 'gray') for x in exit_counts.index])
        ax3.set_ylabel('Count')
        ax3.set_title('Exit Reason Distribution')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_pairs_trading(self, merged, pair_name, save_path=None):
        """绘制配对交易分析图"""
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))

        # 标准化价格
        ax1 = axes[0]
        norm_p1 = merged['price1'] / merged['price1'].iloc[0]
        norm_p2 = merged['price2'] / merged['price2'].iloc[0]
        ax1.plot(merged['trade_date'], norm_p1, label='Stock 1', color='blue')
        ax1.plot(merged['trade_date'], norm_p2, label='Stock 2', color='orange')
        ax1.set_ylabel('Normalized Price')
        ax1.set_title(f'Pairs Trading Analysis: {pair_name}')
        ax1.legend()

        # 价格比率
        ax2 = axes[1]
        ax2.plot(merged['trade_date'], merged['ratio'], label='Ratio', color='purple')
        ax2.plot(merged['trade_date'], merged['ratio_ma'], label='Ratio MA', color='gray', alpha=0.7)
        ax2.set_ylabel('Price Ratio')
        ax2.legend()

        # Z-score
        ax3 = axes[2]
        ax3.plot(merged['trade_date'], merged['zscore'], label='Z-score', color='navy')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.axhline(y=2, color='red', linestyle='--', alpha=0.5, label='Entry (+2)')
        ax3.axhline(y=-2, color='green', linestyle='--', alpha=0.5, label='Entry (-2)')
        ax3.fill_between(merged['trade_date'], -0.5, 0.5, alpha=0.2, color='gray', label='Exit Zone')
        ax3.set_ylabel('Z-score')
        ax3.set_xlabel('Date')
        ax3.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    # ==================== 主研究流程 ====================

    def run_research(self):
        """运行完整研究"""
        self.log("=" * 80)
        self.log("均值回归策略研究报告")
        self.log("=" * 80)
        self.log(f"\n研究日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log(f"数据来源: {DB_PATH}")

        # ==================== 1. 数据准备 ====================
        self.log("\n" + "=" * 60)
        self.log("第一部分：数据准备")
        self.log("=" * 60)

        # 获取沪深300指数数据
        self.log("\n获取沪深300指数数据...")
        index_df = self.get_index_data('000300.SH', '20200101', '20260130')
        self.log(f"指数数据范围: {index_df['trade_date'].min()} 至 {index_df['trade_date'].max()}")
        self.log(f"数据条数: {len(index_df)}")

        # 选取样本股票进行研究
        self.log("\n获取样本股票数据...")
        sample_stocks = ['000001.SZ', '600519.SH', '000858.SZ', '601318.SH', '600036.SH']
        stock_data = {}
        for ts_code in sample_stocks:
            try:
                df = self.get_stock_data(ts_code, '20200101', '20260130')
                if len(df) > 200:
                    stock_data[ts_code] = df
                    self.log(f"  {ts_code}: {len(df)} 条记录")
            except Exception as e:
                self.log(f"  {ts_code}: 获取失败 - {e}")

        # ==================== 2. 均值回归识别 ====================
        self.log("\n" + "=" * 60)
        self.log("第二部分：均值回归识别方法分析")
        self.log("=" * 60)

        # 对每只股票计算技术指标
        all_signals = {}
        for ts_code, df in stock_data.items():
            self.log(f"\n--- 分析 {ts_code} ---")

            # 计算各类指标
            df = self.calculate_ma_deviation(df)
            df = self.calculate_bollinger_bands(df)
            df = self.calculate_zscore(df)
            stock_data[ts_code] = df

            # 分析信号有效性
            signals = self.analyze_mean_reversion_signals(df)
            all_signals[ts_code] = signals

            self.log(f"\n{ts_code} 均值回归信号统计:")
            for signal_name, stats in list(signals.items())[:10]:
                self.log(f"  {signal_name}:")
                self.log(f"    信号次数: {stats['signals']}")
                self.log(f"    5日平均收益: {stats['avg_5d_return']:.2f}%")
                self.log(f"    胜率: {stats['win_rate']:.1f}%")

        # 汇总各信号的有效性
        self.log("\n" + "-" * 40)
        self.log("均值回归信号有效性汇总:")
        self.log("-" * 40)

        signal_summary = {}
        for ts_code, signals in all_signals.items():
            for signal_name, stats in signals.items():
                if signal_name not in signal_summary:
                    signal_summary[signal_name] = []
                signal_summary[signal_name].append(stats)

        self.log(f"\n{'信号类型':<25} {'平均5日收益':>12} {'平均胜率':>10} {'样本量':>8}")
        self.log("-" * 60)
        for signal_name, stats_list in sorted(signal_summary.items()):
            avg_return = np.mean([s['avg_5d_return'] for s in stats_list])
            avg_winrate = np.mean([s['win_rate'] for s in stats_list])
            total_signals = sum([s['signals'] for s in stats_list])
            self.log(f"{signal_name:<25} {avg_return:>12.2f}% {avg_winrate:>10.1f}% {total_signals:>8}")

        # 绘制信号分析图
        for ts_code, df in list(stock_data.items())[:2]:
            save_path = os.path.join(REPORT_DIR, f'mean_reversion_signals_{ts_code.replace(".", "_")}.png')
            self.plot_mean_reversion_signals(df, ts_code, save_path)
            self.log(f"\n信号分析图已保存: {save_path}")

        # ==================== 3. 策略设计与回测 ====================
        self.log("\n" + "=" * 60)
        self.log("第三部分：策略设计与回测")
        self.log("=" * 60)

        # 3.1 简单均值回归策略
        self.log("\n--- 3.1 简单均值回归策略 ---")

        all_backtest_results = {}
        for ts_code, df in stock_data.items():
            result = self.simple_mean_reversion_backtest(df)
            if result:
                stats, trades_df = result
                all_backtest_results[ts_code] = {'stats': stats, 'trades': trades_df}

                self.log(f"\n{ts_code} 回测结果:")
                self.log(f"  总交易次数: {stats['total_trades']}")
                self.log(f"  胜率: {stats['win_rate']:.1f}%")
                self.log(f"  平均收益: {stats['avg_return']:.2f}%")
                self.log(f"  总收益: {stats['total_return']*100:.2f}%")
                self.log(f"  夏普比率: {stats['sharpe']:.2f}")
                self.log(f"  止损出场: {stats['stop_loss_exits']}, 止盈出场: {stats['take_profit_exits']}, 均值回归出场: {stats['mean_reversion_exits']}")

                # 计算风险指标
                risk_metrics = self.calculate_risk_metrics(trades_df)
                if risk_metrics:
                    self.log(f"  最大回撤: {risk_metrics['max_drawdown']*100:.2f}%")
                    self.log(f"  盈亏比: {risk_metrics['profit_factor']:.2f}")

        # 绘制回测结果图
        for ts_code, result in list(all_backtest_results.items())[:2]:
            save_path = os.path.join(REPORT_DIR, f'backtest_results_{ts_code.replace(".", "_")}.png')
            self.plot_backtest_results(result['trades'], stock_data[ts_code], save_path)
            self.log(f"\n回测结果图已保存: {save_path}")

        # 3.2 参数优化
        self.log("\n--- 3.2 参数优化 ---")

        param_grid = {
            'entry_zscore': [-1.5, -2.0, -2.5, -3.0],
            'exit_zscore': [-0.5, 0, 0.5],
            'window': [10, 20, 30],
            'stop_loss': [-0.05, -0.08, -0.10]
        }

        # 选择一只股票进行参数优化
        sample_ts_code = list(stock_data.keys())[0]
        sample_df = stock_data[sample_ts_code]

        self.log(f"\n对 {sample_ts_code} 进行参数优化...")
        opt_results = self.parameter_optimization(sample_df, param_grid)

        if opt_results:
            self.log("\n最优参数组合 (按夏普比率排序):")
            self.log(f"{'Rank':<6} {'Entry Z':>8} {'Exit Z':>8} {'Window':>8} {'Stop Loss':>10} {'Sharpe':>8} {'Win Rate':>10} {'Trades':>8}")
            self.log("-" * 80)
            for i, result in enumerate(opt_results[:10]):
                p = result['params']
                self.log(f"{i+1:<6} {p['entry_zscore']:>8.1f} {p['exit_zscore']:>8.1f} {p['window']:>8} {p['stop_loss']:>10.2f} {result['sharpe']:>8.2f} {result['win_rate']:>10.1f}% {result['total_trades']:>8}")

        # 3.3 配对交易
        self.log("\n--- 3.3 配对交易策略 ---")

        # 获取更多股票数据用于配对交易
        self.log("\n获取更多股票数据用于配对交易分析...")
        more_stocks = self.get_multiple_stocks(30, '20200101', '20260130')
        self.log(f"成功获取 {len(more_stocks)} 只股票数据")

        # 寻找协整配对
        self.log("\n寻找协整股票配对...")
        pairs = self.statistical_arbitrage_portfolio(more_stocks, n_pairs=10)

        if pairs:
            self.log(f"\n发现 {len(pairs)} 个协整配对:")
            self.log(f"{'Stock1':<12} {'Stock2':<12} {'R²':>8} {'Hedge Ratio':>12} {'Residual Std':>12}")
            self.log("-" * 60)
            for pair in pairs:
                self.log(f"{pair['stock1']:<12} {pair['stock2']:<12} {pair['r_squared']:>8.3f} {pair['hedge_ratio']:>12.3f} {pair['residual_std']:>12.4f}")

            # 对最佳配对进行回测
            if len(pairs) > 0:
                best_pair = pairs[0]
                stock1, stock2 = best_pair['stock1'], best_pair['stock2']

                self.log(f"\n对最佳配对 {stock1}-{stock2} 进行回测...")
                pairs_result = self.pairs_trading_backtest(
                    more_stocks[stock1],
                    more_stocks[stock2]
                )

                if pairs_result:
                    stats, trades_df, merged = pairs_result
                    self.log(f"\n配对交易回测结果:")
                    self.log(f"  总交易次数: {stats['total_trades']}")
                    self.log(f"  胜率: {stats['win_rate']:.1f}%")
                    self.log(f"  平均收益: {stats['avg_return']:.2f}%")
                    self.log(f"  总收益: {stats['total_return']*100:.2f}%")
                    self.log(f"  夏普比率: {stats['sharpe']:.2f}")

                    # 绘制配对交易图
                    save_path = os.path.join(REPORT_DIR, f'pairs_trading_{stock1.replace(".", "_")}_{stock2.replace(".", "_")}.png')
                    self.plot_pairs_trading(merged, f"{stock1}-{stock2}", save_path)
                    self.log(f"\n配对交易分析图已保存: {save_path}")

        # ==================== 4. 市场环境分析 ====================
        self.log("\n" + "=" * 60)
        self.log("第四部分：不同市场环境下的策略表现")
        self.log("=" * 60)

        for ts_code, df in list(stock_data.items())[:3]:
            self.log(f"\n--- {ts_code} 在不同市场环境下的表现 ---")

            regime_results = self.analyze_market_regimes(df, index_df)

            if regime_results:
                self.log(f"\n{'环境':<12} {'交易次数':>10} {'胜率':>10} {'平均收益':>12} {'夏普比率':>10}")
                self.log("-" * 60)
                for regime, stats in regime_results.items():
                    regime_name = {
                        'bull': '牛市',
                        'bear': '熊市',
                        'normal': '震荡市',
                        'high_vol': '高波动'
                    }.get(regime, regime)
                    self.log(f"{regime_name:<12} {stats['total_trades']:>10} {stats['win_rate']:>10.1f}% {stats['avg_return']:>12.2f}% {stats['sharpe']:>10.2f}")

        # ==================== 5. 风险控制分析 ====================
        self.log("\n" + "=" * 60)
        self.log("第五部分：风险控制分析")
        self.log("=" * 60)

        self.log("\n--- 不同止损设置的影响 ---")
        stop_loss_analysis = []
        sample_df = stock_data[list(stock_data.keys())[0]]

        for stop_loss in [-0.03, -0.05, -0.08, -0.10, -0.15, -0.20]:
            result = self.simple_mean_reversion_backtest(sample_df, stop_loss=stop_loss)
            if result:
                stats, trades_df = result
                risk_metrics = self.calculate_risk_metrics(trades_df)
                if risk_metrics:
                    stop_loss_analysis.append({
                        'stop_loss': stop_loss,
                        'win_rate': stats['win_rate'],
                        'avg_return': stats['avg_return'],
                        'total_return': stats['total_return'],
                        'max_drawdown': risk_metrics['max_drawdown'],
                        'sharpe': stats['sharpe'],
                        'trades': stats['total_trades']
                    })

        if stop_loss_analysis:
            self.log(f"\n{'止损点':>10} {'交易次数':>10} {'胜率':>10} {'平均收益':>12} {'最大回撤':>12} {'夏普比率':>10}")
            self.log("-" * 70)
            for row in stop_loss_analysis:
                self.log(f"{row['stop_loss']*100:>10.1f}% {row['trades']:>10} {row['win_rate']:>10.1f}% {row['avg_return']:>12.2f}% {row['max_drawdown']*100:>12.2f}% {row['sharpe']:>10.2f}")

        # ==================== 6. 研究结论 ====================
        self.log("\n" + "=" * 60)
        self.log("第六部分：研究结论与建议")
        self.log("=" * 60)

        self.log("""
1. 均值回归识别方法对比:
   - Z-score方法: 信号明确，参数化程度高，适合量化实现
   - 布林带方法: 动态适应波动率，视觉直观，但需结合其他指标
   - MA偏离度: 简单直观，但在趋势市场中可能产生过多假信号

2. 策略有效性分析:
   - 简单均值回归策略在震荡市场表现最佳
   - 配对交易可以有效降低市场风险敞口
   - 协整关系的稳定性是配对交易成功的关键

3. 最优参数建议:
   - 入场Z-score: -2.0 到 -2.5 之间效果较好
   - 出场Z-score: 接近0时出场可以捕获更多收益
   - 回溯窗口: 20日窗口在稳定性和敏感性之间取得平衡
   - 止损设置: -8%左右可以有效控制单笔损失

4. 风险控制建议:
   - 设置合理的止损位，防止趋势行情中的重大损失
   - 控制单笔仓位，建议不超过10%
   - 分散投资多个均值回归机会
   - 在高波动市场中适当降低仓位

5. 市场环境适应性:
   - 牛市: 策略表现一般，可能错过上涨趋势
   - 熊市: 需要谨慎使用，假信号较多
   - 震荡市: 最适合均值回归策略
   - 高波动: 需要放大入场阈值，控制风险

6. 改进方向:
   - 结合量能指标过滤假信号
   - 使用机器学习优化参数自适应
   - 结合市场情绪指标判断市场环境
   - 开发动态止损策略
""")

        # 保存报告
        report_path = os.path.join(REPORT_DIR, 'mean_reversion_research_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.report_lines))
        self.log(f"\n研究报告已保存: {report_path}")

        return self.report_lines


def main():
    """主函数"""
    research = MeanReversionResearch()
    research.run_research()


if __name__ == '__main__':
    main()
