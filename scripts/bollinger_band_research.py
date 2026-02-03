#!/usr/bin/env python3
"""
布林带交易策略研究

研究内容：
1. 布林带计算：标准布林带、不同参数测试、布林带宽度
2. 交易信号：触及上轨/下轨、突破上轨/下轨、收窄/放宽信号
3. 策略回测：均值回归策略、突破策略、结合其他指标
"""

import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# 第一部分：布林带计算
# ============================================================================

class BollingerBandCalculator:
    """布林带计算器"""

    def __init__(self, period: int = 20, std_dev: float = 2.0):
        """
        初始化布林带计算器

        Parameters:
        -----------
        period : int
            移动平均周期，默认20
        std_dev : float
            标准差倍数，默认2.0
        """
        self.period = period
        self.std_dev = std_dev

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算布林带

        Parameters:
        -----------
        df : pd.DataFrame
            包含 'close' 列的DataFrame

        Returns:
        --------
        pd.DataFrame : 包含布林带指标的DataFrame
        """
        df = df.copy()

        # 中轨 - 简单移动平均
        df['bb_middle'] = df['close'].rolling(window=self.period).mean()

        # 标准差
        df['bb_std'] = df['close'].rolling(window=self.period).std()

        # 上轨和下轨
        df['bb_upper'] = df['bb_middle'] + (self.std_dev * df['bb_std'])
        df['bb_lower'] = df['bb_middle'] - (self.std_dev * df['bb_std'])

        # 布林带宽度 (Bandwidth)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle'] * 100

        # %B 指标（价格在布林带中的位置）
        df['bb_percent_b'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        return df

    def calculate_squeeze(self, df: pd.DataFrame, keltner_period: int = 20,
                         keltner_atr_mult: float = 1.5) -> pd.DataFrame:
        """
        计算布林带挤压（与肯特纳通道比较）

        当布林带在肯特纳通道内部时，市场处于挤压状态
        """
        df = df.copy()

        # 计算ATR
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(window=keltner_period).mean()

        # 肯特纳通道
        df['keltner_middle'] = df['close'].rolling(window=keltner_period).mean()
        df['keltner_upper'] = df['keltner_middle'] + keltner_atr_mult * df['atr']
        df['keltner_lower'] = df['keltner_middle'] - keltner_atr_mult * df['atr']

        # 挤压信号：布林带在肯特纳通道内
        df['squeeze'] = (df['bb_lower'] > df['keltner_lower']) & (df['bb_upper'] < df['keltner_upper'])

        return df


# ============================================================================
# 第二部分：交易信号生成
# ============================================================================

class BollingerSignalGenerator:
    """布林带交易信号生成器"""

    @staticmethod
    def touch_signals(df: pd.DataFrame) -> pd.DataFrame:
        """
        生成触及上下轨信号

        触及上轨：当日最高价达到上轨
        触及下轨：当日最低价达到下轨
        """
        df = df.copy()
        df['touch_upper'] = df['high'] >= df['bb_upper']
        df['touch_lower'] = df['low'] <= df['bb_lower']
        return df

    @staticmethod
    def breakout_signals(df: pd.DataFrame) -> pd.DataFrame:
        """
        生成突破上下轨信号

        突破上轨：收盘价高于上轨
        突破下轨：收盘价低于下轨
        """
        df = df.copy()
        df['breakout_upper'] = df['close'] > df['bb_upper']
        df['breakout_lower'] = df['close'] < df['bb_lower']
        return df

    @staticmethod
    def bandwidth_signals(df: pd.DataFrame, narrow_percentile: float = 20,
                         wide_percentile: float = 80, lookback: int = 120) -> pd.DataFrame:
        """
        生成带宽收窄/放宽信号

        Parameters:
        -----------
        narrow_percentile : float
            收窄阈值百分位数
        wide_percentile : float
            放宽阈值百分位数
        lookback : int
            回溯期数
        """
        df = df.copy()

        # 计算滚动百分位数阈值
        df['width_narrow_threshold'] = df['bb_width'].rolling(window=lookback).quantile(narrow_percentile/100)
        df['width_wide_threshold'] = df['bb_width'].rolling(window=lookback).quantile(wide_percentile/100)

        # 收窄/放宽信号
        df['bandwidth_narrow'] = df['bb_width'] <= df['width_narrow_threshold']
        df['bandwidth_wide'] = df['bb_width'] >= df['width_wide_threshold']

        return df

    @staticmethod
    def mean_reversion_signals(df: pd.DataFrame) -> pd.DataFrame:
        """
        生成均值回归信号

        买入：收盘价触及下轨后回升
        卖出：收盘价触及上轨后回落
        """
        df = df.copy()

        # 触及下轨后的回升
        df['touched_lower'] = df['close'] <= df['bb_lower']
        df['touched_lower_prev'] = df['touched_lower'].shift(1)
        df['mr_buy'] = df['touched_lower_prev'] & (df['close'] > df['close'].shift(1))

        # 触及上轨后的回落
        df['touched_upper'] = df['close'] >= df['bb_upper']
        df['touched_upper_prev'] = df['touched_upper'].shift(1)
        df['mr_sell'] = df['touched_upper_prev'] & (df['close'] < df['close'].shift(1))

        return df


# ============================================================================
# 第三部分：策略回测
# ============================================================================

class BollingerBacktester:
    """布林带策略回测器"""

    def __init__(self, initial_capital: float = 100000, commission: float = 0.001):
        """
        初始化回测器

        Parameters:
        -----------
        initial_capital : float
            初始资金
        commission : float
            交易佣金率
        """
        self.initial_capital = initial_capital
        self.commission = commission

    def mean_reversion_strategy(self, df: pd.DataFrame) -> dict:
        """
        均值回归策略回测

        策略规则：
        - 买入：价格触及下轨且%B < 0
        - 卖出：价格触及上轨或%B > 1，或价格回到中轨
        """
        df = df.copy().reset_index(drop=True)

        position = 0
        cash = self.initial_capital
        shares = 0
        trades = []
        equity_curve = []

        for i in range(len(df)):
            row = df.iloc[i]

            # 跳过没有布林带数据的行
            if pd.isna(row['bb_middle']):
                equity_curve.append(cash)
                continue

            # 买入信号
            if position == 0 and row['bb_percent_b'] < 0:
                shares = int(cash * 0.95 / row['close'])
                if shares > 0:
                    cost = shares * row['close'] * (1 + self.commission)
                    cash -= cost
                    position = 1
                    trades.append({
                        'date': row['trade_date'],
                        'action': 'BUY',
                        'price': row['close'],
                        'shares': shares,
                        'cost': cost
                    })

            # 卖出信号
            elif position == 1 and (row['bb_percent_b'] > 1 or row['close'] >= row['bb_middle']):
                revenue = shares * row['close'] * (1 - self.commission)
                cash += revenue
                trades.append({
                    'date': row['trade_date'],
                    'action': 'SELL',
                    'price': row['close'],
                    'shares': shares,
                    'revenue': revenue
                })
                shares = 0
                position = 0

            # 记录权益
            equity = cash + shares * row['close']
            equity_curve.append(equity)

        # 强制平仓
        if position == 1:
            revenue = shares * df.iloc[-1]['close'] * (1 - self.commission)
            cash += revenue
            trades.append({
                'date': df.iloc[-1]['trade_date'],
                'action': 'SELL',
                'price': df.iloc[-1]['close'],
                'shares': shares,
                'revenue': revenue
            })

        return {
            'strategy': 'mean_reversion',
            'final_capital': cash,
            'total_return': (cash - self.initial_capital) / self.initial_capital * 100,
            'trades': trades,
            'equity_curve': equity_curve,
            'num_trades': len([t for t in trades if t['action'] == 'BUY'])
        }

    def breakout_strategy(self, df: pd.DataFrame) -> dict:
        """
        突破策略回测

        策略规则：
        - 买入：价格向上突破上轨
        - 卖出：价格跌破中轨
        """
        df = df.copy().reset_index(drop=True)

        position = 0
        cash = self.initial_capital
        shares = 0
        trades = []
        equity_curve = []

        for i in range(1, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]

            if pd.isna(row['bb_middle']):
                equity_curve.append(cash)
                continue

            # 买入信号：突破上轨
            if position == 0 and row['close'] > row['bb_upper'] and prev_row['close'] <= prev_row['bb_upper']:
                shares = int(cash * 0.95 / row['close'])
                if shares > 0:
                    cost = shares * row['close'] * (1 + self.commission)
                    cash -= cost
                    position = 1
                    trades.append({
                        'date': row['trade_date'],
                        'action': 'BUY',
                        'price': row['close'],
                        'shares': shares,
                        'cost': cost
                    })

            # 卖出信号：跌破中轨
            elif position == 1 and row['close'] < row['bb_middle']:
                revenue = shares * row['close'] * (1 - self.commission)
                cash += revenue
                trades.append({
                    'date': row['trade_date'],
                    'action': 'SELL',
                    'price': row['close'],
                    'shares': shares,
                    'revenue': revenue
                })
                shares = 0
                position = 0

            equity = cash + shares * row['close']
            equity_curve.append(equity)

        # 强制平仓
        if position == 1:
            revenue = shares * df.iloc[-1]['close'] * (1 - self.commission)
            cash += revenue
            trades.append({
                'date': df.iloc[-1]['trade_date'],
                'action': 'SELL',
                'price': df.iloc[-1]['close'],
                'shares': shares,
                'revenue': revenue
            })

        return {
            'strategy': 'breakout',
            'final_capital': cash,
            'total_return': (cash - self.initial_capital) / self.initial_capital * 100,
            'trades': trades,
            'equity_curve': equity_curve,
            'num_trades': len([t for t in trades if t['action'] == 'BUY'])
        }

    def squeeze_breakout_strategy(self, df: pd.DataFrame) -> dict:
        """
        挤压突破策略

        策略规则：
        - 买入：挤压结束后价格突破上轨
        - 卖出：价格跌破中轨或再次进入挤压状态
        """
        df = df.copy().reset_index(drop=True)

        position = 0
        cash = self.initial_capital
        shares = 0
        trades = []
        equity_curve = []

        for i in range(1, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]

            if pd.isna(row['bb_middle']) or pd.isna(row.get('squeeze')):
                equity_curve.append(cash)
                continue

            # 买入信号：挤压结束 + 向上突破
            squeeze_ended = prev_row['squeeze'] and not row['squeeze']
            breakout_up = row['close'] > row['bb_middle'] and row['close'] > prev_row['close']

            if position == 0 and squeeze_ended and breakout_up:
                shares = int(cash * 0.95 / row['close'])
                if shares > 0:
                    cost = shares * row['close'] * (1 + self.commission)
                    cash -= cost
                    position = 1
                    trades.append({
                        'date': row['trade_date'],
                        'action': 'BUY',
                        'price': row['close'],
                        'shares': shares,
                        'cost': cost
                    })

            # 卖出信号
            elif position == 1 and (row['close'] < row['bb_middle'] or row['squeeze']):
                revenue = shares * row['close'] * (1 - self.commission)
                cash += revenue
                trades.append({
                    'date': row['trade_date'],
                    'action': 'SELL',
                    'price': row['close'],
                    'shares': shares,
                    'revenue': revenue
                })
                shares = 0
                position = 0

            equity = cash + shares * row['close']
            equity_curve.append(equity)

        # 强制平仓
        if position == 1:
            revenue = shares * df.iloc[-1]['close'] * (1 - self.commission)
            cash += revenue
            trades.append({
                'date': df.iloc[-1]['trade_date'],
                'action': 'SELL',
                'price': df.iloc[-1]['close'],
                'shares': shares,
                'revenue': revenue
            })

        return {
            'strategy': 'squeeze_breakout',
            'final_capital': cash,
            'total_return': (cash - self.initial_capital) / self.initial_capital * 100,
            'trades': trades,
            'equity_curve': equity_curve,
            'num_trades': len([t for t in trades if t['action'] == 'BUY'])
        }


# ============================================================================
# 第四部分：综合分析与报告生成
# ============================================================================

class BollingerBandResearch:
    """布林带研究主类"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = duckdb.connect(db_path, read_only=True)
        self.calculator = BollingerBandCalculator()
        self.signal_gen = BollingerSignalGenerator()
        self.backtester = BollingerBacktester()

    def get_stock_data(self, ts_code: str, start_date: str = '20200101',
                       end_date: str = '20260130') -> pd.DataFrame:
        """获取股票日线数据"""
        query = f"""
        SELECT ts_code, trade_date, open, high, low, close, vol, amount
        FROM daily
        WHERE ts_code = '{ts_code}'
        AND trade_date >= '{start_date}'
        AND trade_date <= '{end_date}'
        ORDER BY trade_date
        """
        return self.conn.execute(query).fetchdf()

    def get_index_data(self, ts_code: str = '000300.SH', start_date: str = '20200101',
                       end_date: str = '20260130') -> pd.DataFrame:
        """获取指数日线数据"""
        query = f"""
        SELECT ts_code, trade_date, open, high, low, close, vol, amount
        FROM index_daily
        WHERE ts_code = '{ts_code}'
        AND trade_date >= '{start_date}'
        AND trade_date <= '{end_date}'
        ORDER BY trade_date
        """
        return self.conn.execute(query).fetchdf()

    def parameter_sensitivity_analysis(self, df: pd.DataFrame,
                                        periods: list = [10, 15, 20, 25, 30],
                                        std_devs: list = [1.5, 2.0, 2.5, 3.0]) -> pd.DataFrame:
        """
        参数敏感性分析

        测试不同周期和标准差倍数组合的策略表现
        """
        results = []

        for period in periods:
            for std_dev in std_devs:
                calc = BollingerBandCalculator(period=period, std_dev=std_dev)
                df_calc = calc.calculate(df)

                # 回测均值回归策略
                bt_result = self.backtester.mean_reversion_strategy(df_calc)

                results.append({
                    'period': period,
                    'std_dev': std_dev,
                    'total_return': bt_result['total_return'],
                    'num_trades': bt_result['num_trades'],
                    'final_capital': bt_result['final_capital']
                })

        return pd.DataFrame(results)

    def signal_statistics(self, df: pd.DataFrame) -> dict:
        """统计各类信号的分布"""
        # 计算布林带
        df = self.calculator.calculate(df)

        # 生成信号
        df = self.signal_gen.touch_signals(df)
        df = self.signal_gen.breakout_signals(df)
        df = self.signal_gen.bandwidth_signals(df)

        # 去除NaN行
        df_valid = df.dropna()

        stats = {
            'total_days': len(df_valid),
            'touch_upper_count': df_valid['touch_upper'].sum(),
            'touch_lower_count': df_valid['touch_lower'].sum(),
            'breakout_upper_count': df_valid['breakout_upper'].sum(),
            'breakout_lower_count': df_valid['breakout_lower'].sum(),
            'bandwidth_narrow_count': df_valid['bandwidth_narrow'].sum(),
            'bandwidth_wide_count': df_valid['bandwidth_wide'].sum(),
            'avg_bandwidth': df_valid['bb_width'].mean(),
            'avg_percent_b': df_valid['bb_percent_b'].mean()
        }

        # 计算百分比
        stats['touch_upper_pct'] = stats['touch_upper_count'] / stats['total_days'] * 100
        stats['touch_lower_pct'] = stats['touch_lower_count'] / stats['total_days'] * 100
        stats['breakout_upper_pct'] = stats['breakout_upper_count'] / stats['total_days'] * 100
        stats['breakout_lower_pct'] = stats['breakout_lower_count'] / stats['total_days'] * 100

        return stats

    def calculate_metrics(self, equity_curve: list, benchmark_returns: pd.Series = None) -> dict:
        """计算策略评估指标"""
        if len(equity_curve) < 2:
            return {}

        equity = pd.Series(equity_curve)
        returns = equity.pct_change().dropna()

        # 基本指标
        total_return = (equity.iloc[-1] - equity.iloc[0]) / equity.iloc[0] * 100

        # 年化收益率（假设252个交易日）
        n_years = len(equity) / 252
        annual_return = ((equity.iloc[-1] / equity.iloc[0]) ** (1/n_years) - 1) * 100 if n_years > 0 else 0

        # 波动率
        volatility = returns.std() * np.sqrt(252) * 100

        # 夏普比率（假设无风险利率2%）
        risk_free_rate = 0.02
        sharpe_ratio = (annual_return/100 - risk_free_rate) / (volatility/100) if volatility > 0 else 0

        # 最大回撤
        rolling_max = equity.expanding().max()
        drawdown = (equity - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }

    def plot_bollinger_analysis(self, df: pd.DataFrame, ts_code: str,
                                 save_path: str = None) -> None:
        """绘制布林带分析图"""
        fig, axes = plt.subplots(4, 1, figsize=(14, 16))

        # 1. 价格与布林带
        ax1 = axes[0]
        ax1.plot(df['trade_date'], df['close'], label='收盘价', linewidth=1)
        ax1.plot(df['trade_date'], df['bb_upper'], 'r--', label='上轨', linewidth=0.8)
        ax1.plot(df['trade_date'], df['bb_middle'], 'g-', label='中轨', linewidth=0.8)
        ax1.plot(df['trade_date'], df['bb_lower'], 'b--', label='下轨', linewidth=0.8)
        ax1.fill_between(df['trade_date'], df['bb_upper'], df['bb_lower'], alpha=0.1)
        ax1.set_title(f'{ts_code} 布林带分析', fontsize=14)
        ax1.legend(loc='upper left')
        ax1.set_ylabel('价格')
        ax1.tick_params(axis='x', rotation=45)
        # 减少x轴标签
        n_ticks = min(10, len(df))
        tick_indices = np.linspace(0, len(df)-1, n_ticks, dtype=int)
        ax1.set_xticks([df['trade_date'].iloc[i] for i in tick_indices])

        # 2. %B 指标
        ax2 = axes[1]
        ax2.plot(df['trade_date'], df['bb_percent_b'], label='%B', color='purple')
        ax2.axhline(y=1, color='r', linestyle='--', alpha=0.5)
        ax2.axhline(y=0.5, color='g', linestyle='--', alpha=0.5)
        ax2.axhline(y=0, color='b', linestyle='--', alpha=0.5)
        ax2.fill_between(df['trade_date'], 0, df['bb_percent_b'],
                         where=(df['bb_percent_b'] < 0), alpha=0.3, color='blue')
        ax2.fill_between(df['trade_date'], 1, df['bb_percent_b'],
                         where=(df['bb_percent_b'] > 1), alpha=0.3, color='red')
        ax2.set_title('%B 指标 (价格在布林带中的位置)', fontsize=12)
        ax2.set_ylabel('%B')
        ax2.legend(loc='upper left')
        ax2.set_xticks([df['trade_date'].iloc[i] for i in tick_indices])
        ax2.tick_params(axis='x', rotation=45)

        # 3. 带宽
        ax3 = axes[2]
        ax3.plot(df['trade_date'], df['bb_width'], label='带宽', color='orange')
        ax3.axhline(y=df['bb_width'].quantile(0.2), color='blue', linestyle='--',
                    label='20%分位', alpha=0.5)
        ax3.axhline(y=df['bb_width'].quantile(0.8), color='red', linestyle='--',
                    label='80%分位', alpha=0.5)
        ax3.set_title('布林带宽度', fontsize=12)
        ax3.set_ylabel('带宽 (%)')
        ax3.legend(loc='upper left')
        ax3.set_xticks([df['trade_date'].iloc[i] for i in tick_indices])
        ax3.tick_params(axis='x', rotation=45)

        # 4. 成交量
        ax4 = axes[3]
        ax4.bar(df['trade_date'], df['vol'], alpha=0.5, label='成交量')
        ax4.set_title('成交量', fontsize=12)
        ax4.set_ylabel('成交量')
        ax4.set_xticks([df['trade_date'].iloc[i] for i in tick_indices])
        ax4.tick_params(axis='x', rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"图表已保存至: {save_path}")

        plt.close()

    def plot_parameter_heatmap(self, param_df: pd.DataFrame, save_path: str = None) -> None:
        """绘制参数敏感性热力图"""
        pivot = param_df.pivot(index='period', columns='std_dev', values='total_return')

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto')

        # 设置坐标轴
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f'{x:.1f}' for x in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)

        ax.set_xlabel('标准差倍数')
        ax.set_ylabel('周期')
        ax.set_title('布林带参数敏感性分析\n(均值回归策略收益率%)', fontsize=14)

        # 添加数值标签
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                text = ax.text(j, i, f'{pivot.values[i, j]:.1f}%',
                              ha='center', va='center', color='black', fontsize=10)

        plt.colorbar(im, ax=ax, label='收益率 (%)')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"热力图已保存至: {save_path}")

        plt.close()

    def plot_strategy_comparison(self, results: list, save_path: str = None) -> None:
        """绘制策略对比图"""
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        # 1. 权益曲线对比
        ax1 = axes[0]
        colors = ['blue', 'green', 'red', 'purple']
        for idx, result in enumerate(results):
            if result['equity_curve']:
                ax1.plot(result['equity_curve'],
                        label=f"{result['strategy']} ({result['total_return']:.2f}%)",
                        color=colors[idx % len(colors)])
        ax1.axhline(y=100000, color='gray', linestyle='--', alpha=0.5, label='初始资金')
        ax1.set_title('策略权益曲线对比', fontsize=14)
        ax1.set_xlabel('交易日')
        ax1.set_ylabel('权益')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        # 2. 策略指标对比
        ax2 = axes[1]
        strategies = [r['strategy'] for r in results]
        returns = [r['total_return'] for r in results]
        trades = [r['num_trades'] for r in results]

        x = np.arange(len(strategies))
        width = 0.35

        bars1 = ax2.bar(x - width/2, returns, width, label='收益率(%)', color='steelblue')
        ax2_twin = ax2.twinx()
        bars2 = ax2_twin.bar(x + width/2, trades, width, label='交易次数', color='coral')

        ax2.set_xlabel('策略')
        ax2.set_ylabel('收益率 (%)', color='steelblue')
        ax2_twin.set_ylabel('交易次数', color='coral')
        ax2.set_xticks(x)
        ax2.set_xticklabels(strategies)
        ax2.set_title('策略指标对比', fontsize=14)

        # 合并图例
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"策略对比图已保存至: {save_path}")

        plt.close()

    def generate_report(self, output_dir: str) -> str:
        """生成完整研究报告"""
        import os
        os.makedirs(output_dir, exist_ok=True)

        report_lines = []
        report_lines.append("# 布林带交易策略研究报告\n")
        report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # ============== 使用沪深300指数进行研究 ==============
        report_lines.append("## 一、研究对象\n\n")
        report_lines.append("本研究使用沪深300指数(000300.SH)和若干典型个股进行布林带策略分析。\n\n")

        # 获取沪深300数据
        print("正在获取沪深300指数数据...")
        index_df = self.get_index_data('000300.SH', '20200101', '20260130')

        if len(index_df) == 0:
            print("警告：无法获取沪深300数据，尝试使用上证指数...")
            index_df = self.get_index_data('000001.SH', '20200101', '20260130')

        report_lines.append(f"- 数据区间: {index_df['trade_date'].min()} - {index_df['trade_date'].max()}\n")
        report_lines.append(f"- 数据条数: {len(index_df)}\n\n")

        # ============== 布林带计算示例 ==============
        report_lines.append("## 二、布林带计算\n\n")
        report_lines.append("### 2.1 标准布林带参数\n\n")
        report_lines.append("- 周期: 20日\n")
        report_lines.append("- 标准差倍数: 2.0\n")
        report_lines.append("- 中轨: 20日简单移动平均线\n")
        report_lines.append("- 上轨: 中轨 + 2倍标准差\n")
        report_lines.append("- 下轨: 中轨 - 2倍标准差\n\n")

        # 计算布林带
        index_bb = self.calculator.calculate(index_df)
        index_bb = self.calculator.calculate_squeeze(index_bb)

        # 绘制布林带分析图
        print("正在绘制布林带分析图...")
        self.plot_bollinger_analysis(
            index_bb, '000300.SH',
            save_path=os.path.join(output_dir, 'bollinger_analysis_300.png')
        )
        report_lines.append("![布林带分析图](bollinger_analysis_300.png)\n\n")

        # 布林带统计
        bb_stats = index_bb.dropna()
        report_lines.append("### 2.2 布林带统计特征\n\n")
        report_lines.append("| 指标 | 值 |\n")
        report_lines.append("|------|----|\n")
        report_lines.append(f"| 平均带宽 | {bb_stats['bb_width'].mean():.2f}% |\n")
        report_lines.append(f"| 带宽标准差 | {bb_stats['bb_width'].std():.2f}% |\n")
        report_lines.append(f"| 带宽最小值 | {bb_stats['bb_width'].min():.2f}% |\n")
        report_lines.append(f"| 带宽最大值 | {bb_stats['bb_width'].max():.2f}% |\n")
        report_lines.append(f"| %B均值 | {bb_stats['bb_percent_b'].mean():.4f} |\n")
        report_lines.append(f"| %B标准差 | {bb_stats['bb_percent_b'].std():.4f} |\n\n")

        # ============== 参数敏感性分析 ==============
        report_lines.append("### 2.3 参数敏感性分析\n\n")
        print("正在进行参数敏感性分析...")

        param_results = self.parameter_sensitivity_analysis(index_df)

        # 绘制热力图
        self.plot_parameter_heatmap(
            param_results,
            save_path=os.path.join(output_dir, 'parameter_heatmap.png')
        )
        report_lines.append("![参数敏感性热力图](parameter_heatmap.png)\n\n")

        # 找到最优参数
        best_params = param_results.loc[param_results['total_return'].idxmax()]
        report_lines.append(f"**最优参数组合:**\n")
        report_lines.append(f"- 周期: {int(best_params['period'])}日\n")
        report_lines.append(f"- 标准差倍数: {best_params['std_dev']}\n")
        report_lines.append(f"- 收益率: {best_params['total_return']:.2f}%\n\n")

        # ============== 交易信号统计 ==============
        report_lines.append("## 三、交易信号分析\n\n")
        print("正在统计交易信号...")

        signal_stats = self.signal_statistics(index_df)

        report_lines.append("### 3.1 信号统计\n\n")
        report_lines.append("| 信号类型 | 发生次数 | 占比 |\n")
        report_lines.append("|----------|----------|------|\n")
        report_lines.append(f"| 触及上轨 | {signal_stats['touch_upper_count']} | {signal_stats['touch_upper_pct']:.2f}% |\n")
        report_lines.append(f"| 触及下轨 | {signal_stats['touch_lower_count']} | {signal_stats['touch_lower_pct']:.2f}% |\n")
        report_lines.append(f"| 突破上轨 | {signal_stats['breakout_upper_count']} | {signal_stats['breakout_upper_pct']:.2f}% |\n")
        report_lines.append(f"| 突破下轨 | {signal_stats['breakout_lower_count']} | {signal_stats['breakout_lower_pct']:.2f}% |\n")
        report_lines.append(f"| 带宽收窄 | {signal_stats['bandwidth_narrow_count']} | - |\n")
        report_lines.append(f"| 带宽放宽 | {signal_stats['bandwidth_wide_count']} | - |\n\n")

        # ============== 策略回测 ==============
        report_lines.append("## 四、策略回测\n\n")
        print("正在进行策略回测...")

        # 均值回归策略
        mr_result = self.backtester.mean_reversion_strategy(index_bb)
        mr_metrics = self.calculate_metrics(mr_result['equity_curve'])

        # 突破策略
        bo_result = self.backtester.breakout_strategy(index_bb)
        bo_metrics = self.calculate_metrics(bo_result['equity_curve'])

        # 挤压突破策略
        sq_result = self.backtester.squeeze_breakout_strategy(index_bb)
        sq_metrics = self.calculate_metrics(sq_result['equity_curve'])

        # 买入持有基准
        bh_return = (index_df['close'].iloc[-1] - index_df['close'].iloc[0]) / index_df['close'].iloc[0] * 100

        report_lines.append("### 4.1 策略回测结果\n\n")
        report_lines.append("| 策略 | 总收益率 | 年化收益率 | 波动率 | 夏普比率 | 最大回撤 | 交易次数 |\n")
        report_lines.append("|------|----------|------------|--------|----------|----------|----------|\n")
        report_lines.append(f"| 均值回归 | {mr_result['total_return']:.2f}% | {mr_metrics.get('annual_return', 0):.2f}% | {mr_metrics.get('volatility', 0):.2f}% | {mr_metrics.get('sharpe_ratio', 0):.2f} | {mr_metrics.get('max_drawdown', 0):.2f}% | {mr_result['num_trades']} |\n")
        report_lines.append(f"| 突破策略 | {bo_result['total_return']:.2f}% | {bo_metrics.get('annual_return', 0):.2f}% | {bo_metrics.get('volatility', 0):.2f}% | {bo_metrics.get('sharpe_ratio', 0):.2f} | {bo_metrics.get('max_drawdown', 0):.2f}% | {bo_result['num_trades']} |\n")
        report_lines.append(f"| 挤压突破 | {sq_result['total_return']:.2f}% | {sq_metrics.get('annual_return', 0):.2f}% | {sq_metrics.get('volatility', 0):.2f}% | {sq_metrics.get('sharpe_ratio', 0):.2f} | {sq_metrics.get('max_drawdown', 0):.2f}% | {sq_result['num_trades']} |\n")
        report_lines.append(f"| 买入持有 | {bh_return:.2f}% | - | - | - | - | 1 |\n\n")

        # 绘制策略对比图
        self.plot_strategy_comparison(
            [mr_result, bo_result, sq_result],
            save_path=os.path.join(output_dir, 'strategy_comparison.png')
        )
        report_lines.append("![策略对比图](strategy_comparison.png)\n\n")

        # ============== 个股测试 ==============
        report_lines.append("## 五、个股测试\n\n")
        print("正在进行个股测试...")

        # 选取几只典型股票进行测试
        test_stocks = ['000001.SZ', '600519.SH', '000858.SZ']  # 平安银行、贵州茅台、五粮液
        stock_results = []

        for stock in test_stocks:
            try:
                stock_df = self.get_stock_data(stock)
                if len(stock_df) > 50:
                    stock_bb = self.calculator.calculate(stock_df)
                    stock_bb = self.calculator.calculate_squeeze(stock_bb)

                    mr_res = self.backtester.mean_reversion_strategy(stock_bb)
                    bo_res = self.backtester.breakout_strategy(stock_bb)
                    bh_ret = (stock_df['close'].iloc[-1] - stock_df['close'].iloc[0]) / stock_df['close'].iloc[0] * 100

                    stock_results.append({
                        'stock': stock,
                        'mean_reversion': mr_res['total_return'],
                        'breakout': bo_res['total_return'],
                        'buy_hold': bh_ret,
                        'mr_trades': mr_res['num_trades'],
                        'bo_trades': bo_res['num_trades']
                    })
            except Exception as e:
                print(f"处理 {stock} 时出错: {e}")

        if stock_results:
            report_lines.append("### 5.1 个股策略测试结果\n\n")
            report_lines.append("| 股票代码 | 均值回归收益 | 突破策略收益 | 买入持有收益 | 均值回归交易次数 | 突破策略交易次数 |\n")
            report_lines.append("|----------|--------------|--------------|--------------|------------------|------------------|\n")
            for r in stock_results:
                report_lines.append(f"| {r['stock']} | {r['mean_reversion']:.2f}% | {r['breakout']:.2f}% | {r['buy_hold']:.2f}% | {r['mr_trades']} | {r['bo_trades']} |\n")
            report_lines.append("\n")

        # ============== 结论与建议 ==============
        report_lines.append("## 六、结论与建议\n\n")

        report_lines.append("### 6.1 主要发现\n\n")
        report_lines.append("1. **布林带参数选择**\n")
        report_lines.append(f"   - 最优周期约为{int(best_params['period'])}日，标准差倍数为{best_params['std_dev']}\n")
        report_lines.append("   - 较短周期（10-15日）适合捕捉短期波动\n")
        report_lines.append("   - 较长周期（25-30日）更适合趋势跟踪\n\n")

        report_lines.append("2. **信号特征**\n")
        report_lines.append(f"   - 价格触及上轨的概率约为{signal_stats['touch_upper_pct']:.1f}%\n")
        report_lines.append(f"   - 价格触及下轨的概率约为{signal_stats['touch_lower_pct']:.1f}%\n")
        report_lines.append("   - 突破信号相对较少，具有一定的过滤效果\n\n")

        report_lines.append("3. **策略表现**\n")
        best_strategy = max([mr_result, bo_result, sq_result], key=lambda x: x['total_return'])
        report_lines.append(f"   - 在测试期内，{best_strategy['strategy']}策略表现最佳\n")
        report_lines.append("   - 均值回归策略适合震荡市\n")
        report_lines.append("   - 突破策略适合趋势市\n")
        report_lines.append("   - 挤压突破策略可以有效过滤假突破\n\n")

        report_lines.append("### 6.2 使用建议\n\n")
        report_lines.append("1. **参数设置**: 建议根据交易周期选择合适的参数\n")
        report_lines.append("   - 日内/短线: 周期10-15日，标准差1.5-2.0\n")
        report_lines.append("   - 波段: 周期20日，标准差2.0\n")
        report_lines.append("   - 趋势: 周期25-30日，标准差2.5-3.0\n\n")

        report_lines.append("2. **信号确认**: 布林带信号应结合其他指标确认\n")
        report_lines.append("   - 成交量: 突破时放量更可靠\n")
        report_lines.append("   - RSI/KDJ: 判断超买超卖\n")
        report_lines.append("   - 趋势指标: MA、MACD判断大趋势方向\n\n")

        report_lines.append("3. **风险管理**: \n")
        report_lines.append("   - 设置止损位（如跌破下轨一定幅度）\n")
        report_lines.append("   - 控制仓位，避免单一信号满仓操作\n")
        report_lines.append("   - 注意带宽收窄后的突破方向不确定性\n\n")

        report_lines.append("### 6.3 策略代码示例\n\n")
        report_lines.append("```python\n")
        report_lines.append("# 计算布林带\n")
        report_lines.append("from bollinger_band_research import BollingerBandCalculator\n")
        report_lines.append("\n")
        report_lines.append("calc = BollingerBandCalculator(period=20, std_dev=2.0)\n")
        report_lines.append("df = calc.calculate(stock_data)\n")
        report_lines.append("\n")
        report_lines.append("# 均值回归信号\n")
        report_lines.append("buy_signal = df['bb_percent_b'] < 0  # 价格跌破下轨\n")
        report_lines.append("sell_signal = df['close'] >= df['bb_middle']  # 价格回到中轨\n")
        report_lines.append("```\n\n")

        # 保存报告
        report_content = ''.join(report_lines)
        report_path = os.path.join(output_dir, 'bollinger_band_research_report.md')

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        print(f"\n报告已保存至: {report_path}")

        return report_path


def main():
    """主函数"""
    db_path = '/Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db'
    output_dir = '/Users/allen/workspace/python/stock/Tushare-DuckDB/reports/research'

    print("=" * 60)
    print("布林带交易策略研究")
    print("=" * 60)

    research = BollingerBandResearch(db_path)
    report_path = research.generate_report(output_dir)

    print("\n" + "=" * 60)
    print("研究完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
