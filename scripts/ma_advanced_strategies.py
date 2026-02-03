#!/usr/bin/env python3
"""
均线高级策略研究
================
研究更多高级均线策略，包括：
1. 均线斜率策略
2. 均线离散度策略
3. 均线通道策略
4. 均线与市场状态结合

Author: Claude AI Assistant
Date: 2026-02-01
"""

import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
from datetime import datetime
from typing import Dict, List

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

REPORT_DIR = '/Users/allen/workspace/python/stock/Tushare-DuckDB/reports/research'
DB_PATH = '/Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db'


class AdvancedMAStrategies:
    """高级均线策略研究"""

    def __init__(self, db_path: str):
        self.conn = duckdb.connect(db_path, read_only=True)

    def load_index_data(self, ts_code: str = '000300.SH',
                        start_date: str = '20100101',
                        end_date: str = '20251231') -> pd.DataFrame:
        """加载指数数据 - 使用更长的历史数据"""
        query = f"""
        SELECT
            ts_code,
            trade_date,
            open, high, low, close, vol, amount
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
            df['adj_close'] = df['close']
            df['adj_high'] = df['high']
            df['adj_low'] = df['low']

        return df

    def calculate_ma(self, prices: pd.Series, period: int) -> pd.Series:
        """计算简单移动平均线"""
        return prices.rolling(window=period).mean()

    def calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """计算指数移动平均线"""
        return prices.ewm(span=period, adjust=False).mean()

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """计算最大回撤"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

    # ========== 1. 均线斜率策略 ==========
    def ma_slope_strategy(self, df: pd.DataFrame, ma_period: int = 20,
                           slope_lookback: int = 5, slope_threshold: float = 0.001) -> Dict:
        """
        均线斜率策略
        规则：
        - 当均线斜率为正且超过阈值时买入
        - 当均线斜率为负时卖出
        """
        ma = self.calculate_ma(df['adj_close'], ma_period)

        # 计算均线斜率（5日变化率）
        ma_slope = (ma - ma.shift(slope_lookback)) / ma.shift(slope_lookback)

        # 生成信号
        signal = pd.Series(0, index=df.index)
        signal[ma_slope > slope_threshold] = 1
        signal[ma_slope <= 0] = 0

        # 计算收益
        returns = df['adj_close'].pct_change()
        strategy_returns = signal.shift(1) * returns
        buyhold_returns = returns

        total_return = (1 + strategy_returns).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(df)) - 1
        sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() > 0 else 0
        max_drawdown = self._calculate_max_drawdown(strategy_returns)

        bh_total_return = (1 + buyhold_returns).prod() - 1

        return {
            'name': '均线斜率策略',
            'params': f'MA{ma_period}, 斜率周期{slope_lookback}, 阈值{slope_threshold}',
            'signal': signal,
            'ma_slope': ma_slope,
            'strategy_returns': strategy_returns,
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'bh_total_return': bh_total_return,
            'trades': (signal.diff().abs() > 0).sum()
        }

    # ========== 2. 均线离散度策略 ==========
    def ma_dispersion_strategy(self, df: pd.DataFrame,
                                ma_periods: List[int] = [5, 10, 20, 60],
                                dispersion_threshold: float = 0.03) -> Dict:
        """
        均线离散度策略
        规则：
        - 当多条均线收敛（离散度低）时等待
        - 当均线向上发散且离散度增加时买入
        - 当均线向下发散时卖出
        """
        mas = {p: self.calculate_ma(df['adj_close'], p) for p in ma_periods}
        ma_df = pd.DataFrame(mas)

        # 计算离散度（标准差/均值）
        dispersion = ma_df.std(axis=1) / ma_df.mean(axis=1)

        # 均线方向（最短期vs最长期）
        short_ma = mas[min(ma_periods)]
        long_ma = mas[max(ma_periods)]
        ma_direction = (short_ma - long_ma) / long_ma

        # 生成信号
        signal = pd.Series(0, index=df.index)

        # 之前是低离散度，现在离散度增加且方向向上
        prev_low_dispersion = dispersion.shift(1) < dispersion_threshold
        dispersion_increasing = dispersion > dispersion.shift(1)
        upward_direction = ma_direction > 0

        signal[(prev_low_dispersion | (dispersion < dispersion_threshold * 2)) &
               dispersion_increasing & upward_direction] = 1
        signal[ma_direction < 0] = 0

        # 平滑信号
        signal = signal.replace(to_replace=0, method='ffill')

        # 计算收益
        returns = df['adj_close'].pct_change()
        strategy_returns = signal.shift(1) * returns
        buyhold_returns = returns

        total_return = (1 + strategy_returns).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(df)) - 1
        sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() > 0 else 0
        max_drawdown = self._calculate_max_drawdown(strategy_returns)

        bh_total_return = (1 + buyhold_returns).prod() - 1

        return {
            'name': '均线离散度策略',
            'params': f'均线{ma_periods}, 阈值{dispersion_threshold}',
            'signal': signal,
            'dispersion': dispersion,
            'strategy_returns': strategy_returns,
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'bh_total_return': bh_total_return,
            'trades': (signal.diff().abs() > 0).sum()
        }

    # ========== 3. 均线通道策略 ==========
    def ma_channel_strategy(self, df: pd.DataFrame,
                             ma_period: int = 20,
                             channel_multiplier: float = 0.03) -> Dict:
        """
        均线通道策略（类似布林带但更简单）
        规则：
        - 价格突破上轨买入
        - 价格跌破下轨卖出
        """
        ma = self.calculate_ma(df['adj_close'], ma_period)
        upper_band = ma * (1 + channel_multiplier)
        lower_band = ma * (1 - channel_multiplier)

        # 生成信号
        signal = pd.Series(0, index=df.index)
        position = 0

        for i in range(ma_period, len(df)):
            if df['adj_close'].iloc[i] > upper_band.iloc[i] and position == 0:
                position = 1
            elif df['adj_close'].iloc[i] < lower_band.iloc[i] and position == 1:
                position = 0
            signal.iloc[i] = position

        # 计算收益
        returns = df['adj_close'].pct_change()
        strategy_returns = signal.shift(1) * returns
        buyhold_returns = returns

        total_return = (1 + strategy_returns).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(df)) - 1
        sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() > 0 else 0
        max_drawdown = self._calculate_max_drawdown(strategy_returns)

        bh_total_return = (1 + buyhold_returns).prod() - 1

        return {
            'name': '均线通道策略',
            'params': f'MA{ma_period}, 通道宽度{channel_multiplier*100}%',
            'signal': signal,
            'ma': ma,
            'upper_band': upper_band,
            'lower_band': lower_band,
            'strategy_returns': strategy_returns,
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'bh_total_return': bh_total_return,
            'trades': (signal.diff().abs() > 0).sum()
        }

    # ========== 4. 市场状态自适应策略 ==========
    def market_regime_strategy(self, df: pd.DataFrame,
                                 short_period: int = 10,
                                 long_period: int = 50,
                                 vol_period: int = 20) -> Dict:
        """
        市场状态自适应均线策略
        根据市场状态（趋势/震荡）调整策略参数
        """
        # 计算均线
        short_ma = self.calculate_ma(df['adj_close'], short_period)
        long_ma = self.calculate_ma(df['adj_close'], long_period)

        # 计算波动率
        returns = df['adj_close'].pct_change()
        volatility = returns.rolling(window=vol_period).std()
        vol_ma = volatility.rolling(window=vol_period * 2).mean()

        # 判断市场状态
        # 高波动 = 当前波动率 > 平均波动率
        high_vol = volatility > vol_ma

        # 趋势状态 = 短期均线与长期均线的差距
        trend_strength = abs(short_ma - long_ma) / long_ma
        strong_trend = trend_strength > 0.02

        # 生成信号
        signal = pd.Series(0, index=df.index)

        # 策略逻辑：
        # 1. 强趋势 + 低波动 = 跟随趋势
        # 2. 强趋势 + 高波动 = 谨慎跟随（减仓）
        # 3. 弱趋势 + 高波动 = 不操作
        # 4. 弱趋势 + 低波动 = 等待突破

        for i in range(long_period, len(df)):
            if strong_trend.iloc[i]:
                if short_ma.iloc[i] > long_ma.iloc[i]:  # 上升趋势
                    if not high_vol.iloc[i]:
                        signal.iloc[i] = 1  # 满仓
                    else:
                        signal.iloc[i] = 0.5  # 半仓（这里简化为1或0）
                else:  # 下降趋势
                    signal.iloc[i] = 0
            else:
                # 弱趋势，保持之前的仓位或减仓
                signal.iloc[i] = 0

        signal = (signal > 0).astype(int)

        # 计算收益
        strategy_returns = signal.shift(1) * returns
        buyhold_returns = returns

        total_return = (1 + strategy_returns).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(df)) - 1
        sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() > 0 else 0
        max_drawdown = self._calculate_max_drawdown(strategy_returns)

        bh_total_return = (1 + buyhold_returns).prod() - 1

        return {
            'name': '市场状态自适应策略',
            'params': f'短期MA{short_period}, 长期MA{long_period}, 波动率周期{vol_period}',
            'signal': signal,
            'volatility': volatility,
            'trend_strength': trend_strength,
            'strategy_returns': strategy_returns,
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'bh_total_return': bh_total_return,
            'trades': (signal.diff().abs() > 0).sum()
        }

    # ========== 5. EMA vs SMA 对比 ==========
    def ema_vs_sma_comparison(self, df: pd.DataFrame,
                               short_period: int = 10,
                               long_period: int = 30) -> Dict:
        """
        EMA vs SMA 双均线策略对比
        """
        # SMA版本
        sma_short = self.calculate_ma(df['adj_close'], short_period)
        sma_long = self.calculate_ma(df['adj_close'], long_period)
        sma_signal = (sma_short > sma_long).astype(int)

        # EMA版本
        ema_short = self.calculate_ema(df['adj_close'], short_period)
        ema_long = self.calculate_ema(df['adj_close'], long_period)
        ema_signal = (ema_short > ema_long).astype(int)

        # 计算收益
        returns = df['adj_close'].pct_change()
        sma_returns = sma_signal.shift(1) * returns
        ema_returns = ema_signal.shift(1) * returns
        buyhold_returns = returns

        results = {
            'SMA': {
                'total_return': (1 + sma_returns).prod() - 1,
                'annual_return': ((1 + sma_returns).prod()) ** (252 / len(df)) - 1,
                'sharpe_ratio': sma_returns.mean() / sma_returns.std() * np.sqrt(252) if sma_returns.std() > 0 else 0,
                'max_drawdown': self._calculate_max_drawdown(sma_returns),
                'trades': (sma_signal.diff().abs() > 0).sum()
            },
            'EMA': {
                'total_return': (1 + ema_returns).prod() - 1,
                'annual_return': ((1 + ema_returns).prod()) ** (252 / len(df)) - 1,
                'sharpe_ratio': ema_returns.mean() / ema_returns.std() * np.sqrt(252) if ema_returns.std() > 0 else 0,
                'max_drawdown': self._calculate_max_drawdown(ema_returns),
                'trades': (ema_signal.diff().abs() > 0).sum()
            },
            'BuyHold': {
                'total_return': (1 + buyhold_returns).prod() - 1
            },
            'sma_signal': sma_signal,
            'ema_signal': ema_signal,
            'sma_returns': sma_returns,
            'ema_returns': ema_returns
        }

        return results

    def generate_report(self):
        """生成高级策略研究报告"""
        print("正在加载数据...")
        df = self.load_index_data('000300.SH', start_date='20100101')

        if len(df) == 0:
            print("未找到数据")
            return

        print(f"数据范围: {df.index[0]} - {df.index[-1]}, 共 {len(df)} 条记录")

        report = []
        report.append("# 均线高级策略研究报告\n")
        report.append(f"**研究标的**: 沪深300指数 (000300.SH)")
        report.append(f"**数据范围**: {df.index[0].strftime('%Y-%m-%d')} 至 {df.index[-1].strftime('%Y-%m-%d')}")
        report.append(f"**数据量**: {len(df)} 个交易日")
        report.append(f"**报告生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("\n---\n")

        all_results = []

        # 1. 均线斜率策略
        print("测试均线斜率策略...")
        slope_result = self.ma_slope_strategy(df)
        all_results.append(slope_result)
        report.append("## 1. 均线斜率策略\n")
        report.append(f"**策略参数**: {slope_result['params']}\n")
        report.append("**策略逻辑**: 当均线斜率为正且超过阈值时买入，斜率为负时卖出\n")
        report.append(f"- 总收益率: {slope_result['total_return']*100:.2f}%")
        report.append(f"- 年化收益率: {slope_result['annual_return']*100:.2f}%")
        report.append(f"- 夏普比率: {slope_result['sharpe_ratio']:.2f}")
        report.append(f"- 最大回撤: {slope_result['max_drawdown']*100:.2f}%")
        report.append(f"- 交易次数: {slope_result['trades']}")
        report.append(f"- 买入持有收益: {slope_result['bh_total_return']*100:.2f}%\n")

        # 2. 均线离散度策略
        print("测试均线离散度策略...")
        dispersion_result = self.ma_dispersion_strategy(df)
        all_results.append(dispersion_result)
        report.append("\n## 2. 均线离散度策略\n")
        report.append(f"**策略参数**: {dispersion_result['params']}\n")
        report.append("**策略逻辑**: 多条均线收敛后向上发散时买入，向下发散时卖出\n")
        report.append(f"- 总收益率: {dispersion_result['total_return']*100:.2f}%")
        report.append(f"- 年化收益率: {dispersion_result['annual_return']*100:.2f}%")
        report.append(f"- 夏普比率: {dispersion_result['sharpe_ratio']:.2f}")
        report.append(f"- 最大回撤: {dispersion_result['max_drawdown']*100:.2f}%")
        report.append(f"- 交易次数: {dispersion_result['trades']}")
        report.append(f"- 买入持有收益: {dispersion_result['bh_total_return']*100:.2f}%\n")

        # 3. 均线通道策略
        print("测试均线通道策略...")
        channel_result = self.ma_channel_strategy(df)
        all_results.append(channel_result)
        report.append("\n## 3. 均线通道策略\n")
        report.append(f"**策略参数**: {channel_result['params']}\n")
        report.append("**策略逻辑**: 价格突破均线上轨买入，跌破下轨卖出\n")
        report.append(f"- 总收益率: {channel_result['total_return']*100:.2f}%")
        report.append(f"- 年化收益率: {channel_result['annual_return']*100:.2f}%")
        report.append(f"- 夏普比率: {channel_result['sharpe_ratio']:.2f}")
        report.append(f"- 最大回撤: {channel_result['max_drawdown']*100:.2f}%")
        report.append(f"- 交易次数: {channel_result['trades']}")
        report.append(f"- 买入持有收益: {channel_result['bh_total_return']*100:.2f}%\n")

        # 4. 市场状态自适应策略
        print("测试市场状态自适应策略...")
        regime_result = self.market_regime_strategy(df)
        all_results.append(regime_result)
        report.append("\n## 4. 市场状态自适应策略\n")
        report.append(f"**策略参数**: {regime_result['params']}\n")
        report.append("**策略逻辑**: 根据趋势强度和波动率状态动态调整仓位\n")
        report.append(f"- 总收益率: {regime_result['total_return']*100:.2f}%")
        report.append(f"- 年化收益率: {regime_result['annual_return']*100:.2f}%")
        report.append(f"- 夏普比率: {regime_result['sharpe_ratio']:.2f}")
        report.append(f"- 最大回撤: {regime_result['max_drawdown']*100:.2f}%")
        report.append(f"- 交易次数: {regime_result['trades']}")
        report.append(f"- 买入持有收益: {regime_result['bh_total_return']*100:.2f}%\n")

        # 5. EMA vs SMA 对比
        print("EMA vs SMA 对比...")
        ema_sma_result = self.ema_vs_sma_comparison(df)
        report.append("\n## 5. EMA vs SMA 对比\n")
        report.append("**参数**: 短期10日，长期30日\n")
        report.append("\n| 指标 | SMA策略 | EMA策略 | 买入持有 |")
        report.append("|------|---------|---------|----------|")
        report.append(f"| 总收益率 | {ema_sma_result['SMA']['total_return']*100:.2f}% | {ema_sma_result['EMA']['total_return']*100:.2f}% | {ema_sma_result['BuyHold']['total_return']*100:.2f}% |")
        report.append(f"| 年化收益率 | {ema_sma_result['SMA']['annual_return']*100:.2f}% | {ema_sma_result['EMA']['annual_return']*100:.2f}% | - |")
        report.append(f"| 夏普比率 | {ema_sma_result['SMA']['sharpe_ratio']:.2f} | {ema_sma_result['EMA']['sharpe_ratio']:.2f} | - |")
        report.append(f"| 最大回撤 | {ema_sma_result['SMA']['max_drawdown']*100:.2f}% | {ema_sma_result['EMA']['max_drawdown']*100:.2f}% | - |")
        report.append(f"| 交易次数 | {ema_sma_result['SMA']['trades']} | {ema_sma_result['EMA']['trades']} | - |")

        # 策略对比汇总
        report.append("\n---\n")
        report.append("## 6. 策略对比汇总\n")

        summary_data = []
        for r in all_results:
            summary_data.append({
                '策略名称': r['name'],
                '总收益率(%)': r['total_return'] * 100,
                '年化收益(%)': r['annual_return'] * 100,
                '夏普比率': r['sharpe_ratio'],
                '最大回撤(%)': r['max_drawdown'] * 100,
                '交易次数': r['trades'],
                '超额收益(%)': (r['total_return'] - r['bh_total_return']) * 100
            })

        summary_df = pd.DataFrame(summary_data)
        report.append(summary_df.to_markdown(index=False, floatfmt='.2f'))

        # 绘制对比图
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 累计收益对比
        ax1 = axes[0, 0]
        buyhold_cum = ((1 + df['adj_close'].pct_change()).cumprod() - 1) * 100

        for r in all_results:
            cum_ret = ((1 + r['strategy_returns']).cumprod() - 1) * 100
            ax1.plot(df.index, cum_ret, label=r['name'], linewidth=1.5)

        ax1.plot(df.index, buyhold_cum, label='买入持有', alpha=0.5, linewidth=1)
        ax1.set_ylabel('累计收益率 (%)')
        ax1.set_title('各策略累计收益对比')
        ax1.legend(loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)

        # 策略对比柱状图
        ax2 = axes[0, 1]
        x = range(len(summary_df))
        colors = ['steelblue', 'green', 'orange', 'red']
        ax2.bar(x, summary_df['年化收益(%)'], color=colors, alpha=0.7)
        ax2.set_xticks(x)
        ax2.set_xticklabels([r['name'][:6] for r in all_results], rotation=45, ha='right')
        ax2.set_ylabel('年化收益率 (%)')
        ax2.set_title('各策略年化收益对比')
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.grid(True, alpha=0.3)

        # 夏普比率对比
        ax3 = axes[1, 0]
        ax3.bar(x, summary_df['夏普比率'], color=colors, alpha=0.7)
        ax3.set_xticks(x)
        ax3.set_xticklabels([r['name'][:6] for r in all_results], rotation=45, ha='right')
        ax3.set_ylabel('夏普比率')
        ax3.set_title('各策略夏普比率对比')
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax3.grid(True, alpha=0.3)

        # 最大回撤对比
        ax4 = axes[1, 1]
        ax4.bar(x, abs(summary_df['最大回撤(%)']), color=colors, alpha=0.7)
        ax4.set_xticks(x)
        ax4.set_xticklabels([r['name'][:6] for r in all_results], rotation=45, ha='right')
        ax4.set_ylabel('最大回撤 (%, 绝对值)')
        ax4.set_title('各策略最大回撤对比')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(f'{REPORT_DIR}/ma_advanced_strategies.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

        report.append(f"\n![高级策略对比](ma_advanced_strategies.png)\n")

        # 结论
        report.append("\n---\n")
        report.append("## 7. 研究结论\n")

        best_sharpe_idx = summary_df['夏普比率'].idxmax()
        best_return_idx = summary_df['年化收益(%)'].idxmax()
        lowest_dd_idx = summary_df['最大回撤(%)'].idxmax()  # 回撤是负数，max是最小的回撤

        report.append(f"1. **最高夏普比率策略**: {summary_df.loc[best_sharpe_idx, '策略名称']}，夏普比率 {summary_df.loc[best_sharpe_idx, '夏普比率']:.2f}")
        report.append(f"2. **最高年化收益策略**: {summary_df.loc[best_return_idx, '策略名称']}，年化收益 {summary_df.loc[best_return_idx, '年化收益(%)']:.2f}%")
        report.append(f"3. **最低回撤策略**: {summary_df.loc[lowest_dd_idx, '策略名称']}，最大回撤 {summary_df.loc[lowest_dd_idx, '最大回撤(%)']:.2f}%")

        report.append("\n### 策略建议\n")
        report.append("- **均线斜率策略**: 适合趋势明显的市场，能够较早捕捉趋势变化")
        report.append("- **均线离散度策略**: 适合从震荡转为趋势的阶段，粘合后发散信号较为可靠")
        report.append("- **均线通道策略**: 简单有效的趋势跟随策略，但需要合理设置通道宽度")
        report.append("- **市场状态自适应策略**: 综合考虑趋势和波动，适合复杂市场环境")
        report.append("- **EMA vs SMA**: EMA对价格变化更敏感，适合短期交易；SMA更稳定，适合中长期")

        # 保存报告
        report_path = f'{REPORT_DIR}/均线高级策略研究报告.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))

        print(f"\n报告已保存到: {report_path}")
        print(f"图片已保存到: {REPORT_DIR}/ma_advanced_strategies.png")

        return summary_df


def main():
    research = AdvancedMAStrategies(DB_PATH)
    research.generate_report()


if __name__ == '__main__':
    main()
