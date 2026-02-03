#!/usr/bin/env python3
"""
成交量异常检测研究
Volume Anomaly Detection Research

研究内容：
1. 异常定义：突发放量、持续缩量、量价背离、天量见天价
2. 检测方法：统计方法(Z-score)、移动窗口检测、机器学习方法
3. 信号验证：异常成交量后收益、不同市场状态的差异、行业特征影响
4. 策略应用：放量突破策略、缩量回调买入、量价配合策略
"""

import os
import sys
import warnings
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import json

import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import zscore
import matplotlib

# 设置中文显示
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti', 'Songti SC']
matplotlib.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore')

# 配置
DB_PATH = '/Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db'
REPORT_DIR = '/Users/allen/workspace/python/stock/Tushare-DuckDB/reports/research'
os.makedirs(REPORT_DIR, exist_ok=True)


class VolumeAnomalyDetector:
    """成交量异常检测器"""

    def __init__(self, db_path: str):
        self.conn = duckdb.connect(db_path, read_only=True)
        self.results = {}

    def close(self):
        self.conn.close()

    def load_stock_data(self, ts_code: str = None, start_date: str = '20200101',
                        end_date: str = None) -> pd.DataFrame:
        """加载股票数据"""
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')

        sql = f"""
        SELECT d.ts_code, d.trade_date, d.open, d.high, d.low, d.close,
               d.pre_close, d.pct_chg, d.vol, d.amount
        FROM daily d
        WHERE d.trade_date BETWEEN '{start_date}' AND '{end_date}'
        """
        if ts_code:
            sql += f" AND d.ts_code = '{ts_code}'"
        sql += " ORDER BY d.ts_code, d.trade_date"

        return self.conn.execute(sql).df()

    def load_market_data(self, start_date: str = '20200101',
                         end_date: str = None) -> pd.DataFrame:
        """加载市场整体数据"""
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')

        sql = f"""
        SELECT trade_date,
               SUM(vol) as total_vol,
               SUM(amount) as total_amount,
               AVG(pct_chg) as avg_pct_chg,
               COUNT(*) as stock_count
        FROM daily
        WHERE trade_date BETWEEN '{start_date}' AND '{end_date}'
        GROUP BY trade_date
        ORDER BY trade_date
        """
        return self.conn.execute(sql).df()

    def load_industry_data(self) -> pd.DataFrame:
        """加载行业分类数据"""
        sql = """
        SELECT ts_code, l1_name as industry, l2_name as sub_industry
        FROM index_member_all
        WHERE is_new = 'Y' OR is_new IS NULL
        """
        return self.conn.execute(sql).df()


class AnomalyDefinitions:
    """异常类型定义和检测"""

    @staticmethod
    def detect_volume_spike(df: pd.DataFrame, window: int = 20,
                            threshold: float = 3.0) -> pd.DataFrame:
        """
        检测突发放量（成交量是过去N日均量的threshold倍以上）
        """
        df = df.copy()
        df['vol_ma'] = df.groupby('ts_code')['vol'].transform(
            lambda x: x.rolling(window=window, min_periods=5).mean().shift(1)
        )
        df['vol_ratio'] = df['vol'] / df['vol_ma']
        df['is_volume_spike'] = df['vol_ratio'] >= threshold
        return df

    @staticmethod
    def detect_volume_shrink(df: pd.DataFrame, window: int = 5,
                             threshold: float = 0.5) -> pd.DataFrame:
        """
        检测持续缩量（连续N日成交量低于均值的threshold倍）
        """
        df = df.copy()
        df['vol_ma20'] = df.groupby('ts_code')['vol'].transform(
            lambda x: x.rolling(window=20, min_periods=5).mean()
        )
        df['is_low_vol'] = df['vol'] < df['vol_ma20'] * threshold

        # 检测连续缩量
        df['shrink_days'] = df.groupby('ts_code')['is_low_vol'].transform(
            lambda x: x.groupby((~x).cumsum()).cumsum()
        )
        df['is_sustained_shrink'] = df['shrink_days'] >= window
        return df

    @staticmethod
    def detect_volume_price_divergence(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        """
        检测量价背离
        - 价升量减：价格上涨但成交量减少
        - 价跌量增：价格下跌但成交量增加
        """
        df = df.copy()

        # 计算价格变化
        df['price_change'] = df.groupby('ts_code')['close'].transform(
            lambda x: x.pct_change(window)
        )

        # 计算成交量变化
        df['vol_change'] = df.groupby('ts_code')['vol'].transform(
            lambda x: x.pct_change(window)
        )

        # 价升量减
        df['price_up_vol_down'] = (df['price_change'] > 0.03) & (df['vol_change'] < -0.2)

        # 价跌量增
        df['price_down_vol_up'] = (df['price_change'] < -0.03) & (df['vol_change'] > 0.2)

        df['is_divergence'] = df['price_up_vol_down'] | df['price_down_vol_up']
        return df

    @staticmethod
    def detect_climax_volume(df: pd.DataFrame, vol_window: int = 60,
                             price_window: int = 20) -> pd.DataFrame:
        """
        检测天量见天价（成交量创N日新高且价格创M日新高）
        """
        df = df.copy()

        # 成交量是否创新高
        df['vol_max'] = df.groupby('ts_code')['vol'].transform(
            lambda x: x.rolling(window=vol_window, min_periods=20).max()
        )
        df['is_vol_high'] = df['vol'] >= df['vol_max'] * 0.95

        # 价格是否创新高
        df['price_max'] = df.groupby('ts_code')['high'].transform(
            lambda x: x.rolling(window=price_window, min_periods=10).max()
        )
        df['is_price_high'] = df['high'] >= df['price_max'] * 0.98

        # 天量见天价
        df['is_climax'] = df['is_vol_high'] & df['is_price_high']
        return df


class StatisticalDetection:
    """统计方法检测"""

    @staticmethod
    def zscore_detection(df: pd.DataFrame, window: int = 60,
                         threshold: float = 2.5) -> pd.DataFrame:
        """
        Z-score 异常检测
        """
        df = df.copy()

        # 计算成交量的对数以处理偏态分布
        df['log_vol'] = np.log1p(df['vol'])

        # 滚动Z-score
        df['vol_mean'] = df.groupby('ts_code')['log_vol'].transform(
            lambda x: x.rolling(window=window, min_periods=20).mean().shift(1)
        )
        df['vol_std'] = df.groupby('ts_code')['log_vol'].transform(
            lambda x: x.rolling(window=window, min_periods=20).std().shift(1)
        )

        df['vol_zscore'] = (df['log_vol'] - df['vol_mean']) / df['vol_std']
        df['is_zscore_anomaly'] = df['vol_zscore'].abs() >= threshold
        df['zscore_high'] = df['vol_zscore'] >= threshold
        df['zscore_low'] = df['vol_zscore'] <= -threshold

        return df

    @staticmethod
    def moving_window_detection(df: pd.DataFrame, short_window: int = 5,
                                long_window: int = 20, threshold: float = 2.0) -> pd.DataFrame:
        """
        移动窗口异常检测 - 比较短期均量与长期均量
        """
        df = df.copy()

        df['vol_ma_short'] = df.groupby('ts_code')['vol'].transform(
            lambda x: x.rolling(window=short_window, min_periods=2).mean()
        )
        df['vol_ma_long'] = df.groupby('ts_code')['vol'].transform(
            lambda x: x.rolling(window=long_window, min_periods=10).mean().shift(short_window)
        )

        df['vol_ma_ratio'] = df['vol_ma_short'] / df['vol_ma_long']
        df['is_window_anomaly_high'] = df['vol_ma_ratio'] >= threshold
        df['is_window_anomaly_low'] = df['vol_ma_ratio'] <= (1 / threshold)

        return df

    @staticmethod
    def percentile_detection(df: pd.DataFrame, window: int = 252,
                             high_pct: float = 95, low_pct: float = 5) -> pd.DataFrame:
        """
        百分位数异常检测
        """
        df = df.copy()

        df['vol_high_pct'] = df.groupby('ts_code')['vol'].transform(
            lambda x: x.rolling(window=window, min_periods=60).quantile(high_pct/100).shift(1)
        )
        df['vol_low_pct'] = df.groupby('ts_code')['vol'].transform(
            lambda x: x.rolling(window=window, min_periods=60).quantile(low_pct/100).shift(1)
        )

        df['is_pct_high'] = df['vol'] >= df['vol_high_pct']
        df['is_pct_low'] = df['vol'] <= df['vol_low_pct']

        return df


class MLDetection:
    """机器学习方法检测"""

    @staticmethod
    def isolation_forest_detection(df: pd.DataFrame, contamination: float = 0.05) -> pd.DataFrame:
        """
        Isolation Forest 异常检测
        """
        try:
            from sklearn.ensemble import IsolationForest
        except ImportError:
            print("sklearn not available, skipping isolation forest detection")
            df['if_anomaly'] = False
            return df

        df = df.copy()

        # 准备特征
        features = ['vol', 'vol_ma', 'vol_ratio'] if 'vol_ma' in df.columns else ['vol']

        results = []
        for ts_code, group in df.groupby('ts_code'):
            group = group.copy()
            valid_mask = group[features].notna().all(axis=1)

            if valid_mask.sum() < 30:
                group['if_anomaly'] = False
            else:
                X = group.loc[valid_mask, features].values
                clf = IsolationForest(contamination=contamination, random_state=42)
                predictions = np.ones(len(group))
                predictions[valid_mask] = clf.fit_predict(X)
                group['if_anomaly'] = predictions == -1

            results.append(group)

        return pd.concat(results, ignore_index=True)

    @staticmethod
    def local_outlier_factor(df: pd.DataFrame, n_neighbors: int = 20) -> pd.DataFrame:
        """
        Local Outlier Factor (LOF) 检测
        """
        try:
            from sklearn.neighbors import LocalOutlierFactor
        except ImportError:
            print("sklearn not available, skipping LOF detection")
            df['lof_anomaly'] = False
            return df

        df = df.copy()

        features = ['vol', 'pct_chg']

        results = []
        for ts_code, group in df.groupby('ts_code'):
            group = group.copy()
            valid_mask = group[features].notna().all(axis=1)

            if valid_mask.sum() < 50:
                group['lof_anomaly'] = False
            else:
                X = group.loc[valid_mask, features].values
                # 标准化
                X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
                clf = LocalOutlierFactor(n_neighbors=min(n_neighbors, len(X)//2))
                predictions = np.ones(len(group))
                predictions[valid_mask] = clf.fit_predict(X)
                group['lof_anomaly'] = predictions == -1

            results.append(group)

        return pd.concat(results, ignore_index=True)


class SignalValidation:
    """信号验证"""

    @staticmethod
    def calculate_forward_returns(df: pd.DataFrame, periods: List[int] = [1, 3, 5, 10, 20]) -> pd.DataFrame:
        """
        计算异常信号后的收益
        """
        df = df.copy()

        for period in periods:
            df[f'ret_{period}d'] = df.groupby('ts_code')['close'].transform(
                lambda x: x.shift(-period) / x - 1
            )

        return df

    @staticmethod
    def analyze_signal_returns(df: pd.DataFrame, signal_col: str,
                               periods: List[int] = [1, 3, 5, 10, 20]) -> Dict:
        """
        分析信号收益
        """
        results = {}

        signal_df = df[df[signal_col] == True]
        non_signal_df = df[df[signal_col] == False]

        for period in periods:
            ret_col = f'ret_{period}d'
            if ret_col not in df.columns:
                continue

            signal_returns = signal_df[ret_col].dropna()
            baseline_returns = non_signal_df[ret_col].dropna()

            if len(signal_returns) > 10 and len(baseline_returns) > 10:
                results[f'{period}d'] = {
                    'signal_count': len(signal_returns),
                    'signal_mean': signal_returns.mean() * 100,
                    'signal_std': signal_returns.std() * 100,
                    'signal_win_rate': (signal_returns > 0).mean() * 100,
                    'baseline_mean': baseline_returns.mean() * 100,
                    'excess_return': (signal_returns.mean() - baseline_returns.mean()) * 100,
                    't_stat': stats.ttest_ind(signal_returns, baseline_returns)[0],
                    'p_value': stats.ttest_ind(signal_returns, baseline_returns)[1]
                }

        return results

    @staticmethod
    def market_state_analysis(df: pd.DataFrame, market_df: pd.DataFrame,
                              signal_col: str) -> Dict:
        """
        不同市场状态下的信号表现
        """
        # 定义市场状态
        market_df = market_df.copy()
        market_df['mkt_ret_20d'] = market_df['avg_pct_chg'].rolling(20).mean()

        market_df['market_state'] = 'neutral'
        market_df.loc[market_df['mkt_ret_20d'] > 0.5, 'market_state'] = 'bull'
        market_df.loc[market_df['mkt_ret_20d'] < -0.5, 'market_state'] = 'bear'

        # 合并
        df = df.merge(market_df[['trade_date', 'market_state']], on='trade_date', how='left')

        results = {}
        for state in ['bull', 'neutral', 'bear']:
            state_df = df[df['market_state'] == state]
            if len(state_df) == 0:
                continue

            signal_df = state_df[state_df[signal_col] == True]
            if len(signal_df) < 10:
                continue

            if 'ret_5d' in df.columns:
                results[state] = {
                    'signal_count': len(signal_df),
                    'mean_return_5d': signal_df['ret_5d'].mean() * 100,
                    'win_rate_5d': (signal_df['ret_5d'] > 0).mean() * 100,
                }

        return results

    @staticmethod
    def industry_analysis(df: pd.DataFrame, industry_df: pd.DataFrame,
                          signal_col: str) -> pd.DataFrame:
        """
        行业特征影响分析
        """
        # 合并行业数据
        df = df.merge(industry_df[['ts_code', 'industry']], on='ts_code', how='left')

        results = []
        for industry, group in df.groupby('industry'):
            if pd.isna(industry) or len(group) < 100:
                continue

            signal_df = group[group[signal_col] == True]
            if len(signal_df) < 10:
                continue

            if 'ret_5d' in df.columns:
                results.append({
                    'industry': industry,
                    'signal_count': len(signal_df),
                    'signal_ratio': len(signal_df) / len(group) * 100,
                    'mean_return_5d': signal_df['ret_5d'].mean() * 100,
                    'win_rate_5d': (signal_df['ret_5d'] > 0).mean() * 100,
                })

        return pd.DataFrame(results).sort_values('mean_return_5d', ascending=False)


class TradingStrategies:
    """交易策略"""

    @staticmethod
    def breakout_volume_strategy(df: pd.DataFrame, vol_threshold: float = 2.5,
                                 price_breakout_pct: float = 0.02) -> pd.DataFrame:
        """
        放量突破策略
        条件：
        1. 成交量是过去20日均量的vol_threshold倍以上
        2. 收盘价突破过去20日高点price_breakout_pct以上
        3. 收阳线（收盘价 > 开盘价）
        """
        df = df.copy()

        # 成交量条件
        df['vol_ma20'] = df.groupby('ts_code')['vol'].transform(
            lambda x: x.rolling(window=20, min_periods=10).mean().shift(1)
        )
        df['vol_cond'] = df['vol'] >= df['vol_ma20'] * vol_threshold

        # 价格突破条件
        df['high_20d'] = df.groupby('ts_code')['high'].transform(
            lambda x: x.rolling(window=20, min_periods=10).max().shift(1)
        )
        df['price_breakout'] = df['close'] >= df['high_20d'] * (1 + price_breakout_pct)

        # 阳线条件
        df['is_bullish'] = df['close'] > df['open']

        # 策略信号
        df['breakout_signal'] = df['vol_cond'] & df['price_breakout'] & df['is_bullish']

        return df

    @staticmethod
    def pullback_buy_strategy(df: pd.DataFrame, shrink_threshold: float = 0.6,
                              pullback_pct: float = 0.05) -> pd.DataFrame:
        """
        缩量回调买入策略
        条件：
        1. 前期有明显上涨（过去20日涨幅超过15%）
        2. 当前处于回调（从高点回落pullback_pct以上）
        3. 成交量萎缩（低于均量的shrink_threshold）
        """
        df = df.copy()

        # 前期上涨
        df['ret_20d'] = df.groupby('ts_code')['close'].transform(
            lambda x: x / x.shift(20) - 1
        )
        df['prior_uptrend'] = df['ret_20d'] > 0.15

        # 回调幅度
        df['high_10d'] = df.groupby('ts_code')['high'].transform(
            lambda x: x.rolling(window=10, min_periods=5).max()
        )
        df['pullback'] = 1 - df['close'] / df['high_10d']
        df['in_pullback'] = df['pullback'] >= pullback_pct

        # 成交量萎缩
        df['vol_ma20'] = df.groupby('ts_code')['vol'].transform(
            lambda x: x.rolling(window=20, min_periods=10).mean()
        )
        df['vol_shrink'] = df['vol'] < df['vol_ma20'] * shrink_threshold

        # 策略信号
        df['pullback_signal'] = df['prior_uptrend'] & df['in_pullback'] & df['vol_shrink']

        return df

    @staticmethod
    def volume_price_harmony_strategy(df: pd.DataFrame) -> pd.DataFrame:
        """
        量价配合策略
        条件：
        1. 价格上涨（当日涨幅 > 1%）
        2. 成交量温和放大（1.5-3倍均量）
        3. 价格站上5日均线和20日均线
        """
        df = df.copy()

        # 价格上涨
        df['price_up'] = df['pct_chg'] > 1

        # 成交量温和放大
        df['vol_ma20'] = df.groupby('ts_code')['vol'].transform(
            lambda x: x.rolling(window=20, min_periods=10).mean().shift(1)
        )
        df['vol_ratio'] = df['vol'] / df['vol_ma20']
        df['vol_moderate'] = (df['vol_ratio'] >= 1.5) & (df['vol_ratio'] <= 3.0)

        # 均线条件
        df['ma5'] = df.groupby('ts_code')['close'].transform(
            lambda x: x.rolling(window=5, min_periods=3).mean()
        )
        df['ma20'] = df.groupby('ts_code')['close'].transform(
            lambda x: x.rolling(window=20, min_periods=10).mean()
        )
        df['above_ma'] = (df['close'] > df['ma5']) & (df['close'] > df['ma20'])

        # 策略信号
        df['harmony_signal'] = df['price_up'] & df['vol_moderate'] & df['above_ma']

        return df


def backtest_strategy(df: pd.DataFrame, signal_col: str,
                      holding_period: int = 5) -> Dict:
    """
    策略回测
    """
    signals = df[df[signal_col] == True].copy()

    if len(signals) == 0:
        return {'error': 'No signals found'}

    ret_col = f'ret_{holding_period}d'
    if ret_col not in signals.columns:
        return {'error': f'Return column {ret_col} not found'}

    returns = signals[ret_col].dropna()

    if len(returns) == 0:
        return {'error': 'No valid returns'}

    # 计算各种指标
    total_trades = len(returns)
    win_trades = (returns > 0).sum()
    loss_trades = (returns <= 0).sum()

    avg_return = returns.mean()
    avg_win = returns[returns > 0].mean() if win_trades > 0 else 0
    avg_loss = returns[returns <= 0].mean() if loss_trades > 0 else 0

    win_rate = win_trades / total_trades
    profit_factor = abs(avg_win * win_trades / (avg_loss * loss_trades)) if avg_loss != 0 and loss_trades > 0 else np.inf

    sharpe = returns.mean() / returns.std() * np.sqrt(252 / holding_period) if returns.std() > 0 else 0

    # 最大回撤（简化计算）
    cumulative = (1 + returns).cumprod()
    max_drawdown = (cumulative / cumulative.cummax() - 1).min()

    return {
        'total_trades': total_trades,
        'win_trades': win_trades,
        'loss_trades': loss_trades,
        'win_rate': win_rate * 100,
        'avg_return': avg_return * 100,
        'avg_win': avg_win * 100,
        'avg_loss': avg_loss * 100,
        'profit_factor': profit_factor,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown * 100,
        'total_return': returns.sum() * 100
    }


def generate_visualizations(df: pd.DataFrame, market_df: pd.DataFrame,
                           industry_results: pd.DataFrame, save_dir: str):
    """
    生成可视化图表
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. 异常类型分布
    ax = axes[0, 0]
    anomaly_cols = ['is_volume_spike', 'is_sustained_shrink', 'is_divergence',
                    'is_climax', 'is_zscore_anomaly']
    existing_cols = [c for c in anomaly_cols if c in df.columns]
    if existing_cols:
        counts = [df[col].sum() for col in existing_cols]
        labels = ['突发放量', '持续缩量', '量价背离', '天量天价', 'Z-score异常'][:len(existing_cols)]
        ax.bar(labels, counts, color=['red', 'blue', 'green', 'orange', 'purple'][:len(existing_cols)])
        ax.set_title('异常类型分布', fontsize=12)
        ax.set_ylabel('信号数量')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 2. 市场成交量时序
    ax = axes[0, 1]
    if len(market_df) > 0:
        market_df_plot = market_df.copy()
        market_df_plot['date'] = pd.to_datetime(market_df_plot['trade_date'])
        market_df_plot = market_df_plot.set_index('date')
        market_df_plot['total_vol_ma20'] = market_df_plot['total_vol'].rolling(20).mean()
        ax.plot(market_df_plot.index, market_df_plot['total_vol'] / 1e8, alpha=0.5, label='日成交量')
        ax.plot(market_df_plot.index, market_df_plot['total_vol_ma20'] / 1e8, label='20日均量')
        ax.set_title('市场成交量走势(亿手)', fontsize=12)
        ax.legend()
        ax.set_xlabel('日期')

    # 3. 放量信号后收益分布
    ax = axes[0, 2]
    if 'is_volume_spike' in df.columns and 'ret_5d' in df.columns:
        spike_returns = df[df['is_volume_spike'] == True]['ret_5d'].dropna() * 100
        normal_returns = df[df['is_volume_spike'] == False]['ret_5d'].dropna() * 100
        if len(spike_returns) > 10:
            ax.hist(spike_returns, bins=50, alpha=0.5, label='放量信号', density=True)
            ax.hist(normal_returns.sample(min(len(normal_returns), 10000)),
                   bins=50, alpha=0.5, label='正常交易', density=True)
            ax.axvline(spike_returns.mean(), color='red', linestyle='--',
                      label=f'放量均值: {spike_returns.mean():.2f}%')
            ax.set_title('放量信号后5日收益分布', fontsize=12)
            ax.set_xlabel('收益率(%)')
            ax.legend()

    # 4. 策略信号收益对比
    ax = axes[1, 0]
    strategy_cols = ['breakout_signal', 'pullback_signal', 'harmony_signal']
    existing_strategy = [c for c in strategy_cols if c in df.columns]
    if existing_strategy and 'ret_5d' in df.columns:
        means = []
        labels = []
        for col in existing_strategy:
            ret = df[df[col] == True]['ret_5d'].mean() * 100
            means.append(ret)
            if 'breakout' in col:
                labels.append('放量突破')
            elif 'pullback' in col:
                labels.append('缩量回调')
            else:
                labels.append('量价配合')

        baseline = df['ret_5d'].mean() * 100
        colors = ['green' if m > baseline else 'red' for m in means]
        ax.bar(labels, means, color=colors)
        ax.axhline(baseline, color='gray', linestyle='--', label=f'基准: {baseline:.2f}%')
        ax.set_title('策略信号5日平均收益', fontsize=12)
        ax.set_ylabel('收益率(%)')
        ax.legend()

    # 5. 行业异常分布
    ax = axes[1, 1]
    if len(industry_results) > 0:
        top_industries = industry_results.head(10)
        ax.barh(top_industries['industry'], top_industries['mean_return_5d'])
        ax.set_title('行业放量信号收益TOP10', fontsize=12)
        ax.set_xlabel('平均5日收益(%)')

    # 6. 量价关系散点图
    ax = axes[1, 2]
    if 'vol_ratio' in df.columns:
        sample = df[['vol_ratio', 'pct_chg']].dropna().sample(min(5000, len(df)))
        ax.scatter(sample['vol_ratio'], sample['pct_chg'], alpha=0.3, s=5)
        ax.set_xlim(0, 5)
        ax.set_ylim(-10, 10)
        ax.set_xlabel('成交量比率')
        ax.set_ylabel('涨跌幅(%)')
        ax.set_title('量价关系散点图', fontsize=12)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(1, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/volume_anomaly_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {save_dir}/volume_anomaly_analysis.png")


def generate_report(results: Dict, save_path: str):
    """
    生成研究报告
    """
    report = []
    report.append("# 成交量异常检测研究报告")
    report.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 1. 研究概述
    report.append("## 一、研究概述\n")
    report.append("本研究基于A股市场历史数据，对成交量异常进行系统性研究，包括：")
    report.append("1. 异常定义：突发放量、持续缩量、量价背离、天量见天价")
    report.append("2. 检测方法：统计方法(Z-score)、移动窗口检测、机器学习方法")
    report.append("3. 信号验证：异常成交量后收益、不同市场状态的差异、行业特征影响")
    report.append("4. 策略应用：放量突破策略、缩量回调买入、量价配合策略\n")

    # 2. 数据说明
    if 'data_info' in results:
        report.append("## 二、数据说明\n")
        info = results['data_info']
        report.append(f"- 数据范围: {info.get('start_date', 'N/A')} 至 {info.get('end_date', 'N/A')}")
        report.append(f"- 股票数量: {info.get('stock_count', 'N/A')}")
        report.append(f"- 总记录数: {info.get('record_count', 'N/A'):,}")
        report.append("")

    # 3. 异常类型统计
    if 'anomaly_stats' in results:
        report.append("## 三、异常类型统计\n")
        report.append("| 异常类型 | 信号数量 | 占比 |")
        report.append("|---------|---------|------|")
        for anomaly_type, stats in results['anomaly_stats'].items():
            report.append(f"| {anomaly_type} | {stats['count']:,} | {stats['ratio']:.2%} |")
        report.append("")

    # 4. 信号收益分析
    if 'signal_returns' in results:
        report.append("## 四、信号收益分析\n")
        for signal_name, returns in results['signal_returns'].items():
            report.append(f"### {signal_name}\n")
            report.append("| 持有期 | 信号数 | 平均收益 | 胜率 | 超额收益 | t统计量 | p值 |")
            report.append("|-------|--------|---------|------|---------|--------|-----|")
            for period, stats in returns.items():
                report.append(f"| {period} | {stats['signal_count']:,} | {stats['signal_mean']:.2f}% | "
                            f"{stats['signal_win_rate']:.1f}% | {stats['excess_return']:.2f}% | "
                            f"{stats['t_stat']:.2f} | {stats['p_value']:.4f} |")
            report.append("")

    # 5. 市场状态分析
    if 'market_state_analysis' in results:
        report.append("## 五、不同市场状态下的表现\n")
        for signal_name, states in results['market_state_analysis'].items():
            report.append(f"### {signal_name}\n")
            report.append("| 市场状态 | 信号数量 | 5日平均收益 | 胜率 |")
            report.append("|---------|---------|------------|------|")
            for state, stats in states.items():
                state_name = {'bull': '牛市', 'neutral': '震荡', 'bear': '熊市'}.get(state, state)
                report.append(f"| {state_name} | {stats['signal_count']:,} | "
                            f"{stats['mean_return_5d']:.2f}% | {stats['win_rate_5d']:.1f}% |")
            report.append("")

    # 6. 行业分析
    if 'industry_analysis' in results and len(results['industry_analysis']) > 0:
        report.append("## 六、行业特征分析\n")
        report.append("### 放量信号收益TOP10行业\n")
        report.append("| 排名 | 行业 | 信号数量 | 信号占比 | 5日平均收益 | 胜率 |")
        report.append("|-----|------|---------|---------|------------|------|")
        for i, (_, row) in enumerate(results['industry_analysis'].head(10).iterrows()):
            report.append(f"| {i+1} | {row['industry']} | {row['signal_count']:,.0f} | "
                        f"{row['signal_ratio']:.2f}% | {row['mean_return_5d']:.2f}% | "
                        f"{row['win_rate_5d']:.1f}% |")
        report.append("")

    # 7. 策略回测结果
    if 'backtest_results' in results:
        report.append("## 七、策略回测结果\n")
        report.append("| 策略 | 交易次数 | 胜率 | 平均收益 | 盈亏比 | 夏普比率 | 最大回撤 |")
        report.append("|-----|---------|------|---------|-------|---------|---------|")
        for strategy, stats in results['backtest_results'].items():
            if 'error' in stats:
                continue
            report.append(f"| {strategy} | {stats['total_trades']:,} | {stats['win_rate']:.1f}% | "
                        f"{stats['avg_return']:.2f}% | {stats['profit_factor']:.2f} | "
                        f"{stats['sharpe_ratio']:.2f} | {stats['max_drawdown']:.1f}% |")
        report.append("")

    # 8. 研究结论
    report.append("## 八、研究结论\n")
    report.append("### 8.1 异常检测有效性\n")

    # 基于结果生成结论
    if 'signal_returns' in results:
        spike_returns = results['signal_returns'].get('突发放量', {})
        if '5d' in spike_returns:
            excess = spike_returns['5d'].get('excess_return', 0)
            p_val = spike_returns['5d'].get('p_value', 1)
            if excess > 0 and p_val < 0.05:
                report.append(f"- **突发放量信号有效**: 5日超额收益{excess:.2f}%，统计显著(p={p_val:.4f})")
            elif excess < 0:
                report.append(f"- **突发放量信号为负向指标**: 5日超额收益{excess:.2f}%，可能是短期见顶信号")
            else:
                report.append(f"- **突发放量信号效果不显著**: 5日超额收益{excess:.2f}%")

    report.append("\n### 8.2 策略建议\n")

    if 'backtest_results' in results:
        best_strategy = None
        best_sharpe = -np.inf
        for strategy, stats in results['backtest_results'].items():
            if 'error' not in stats and stats['sharpe_ratio'] > best_sharpe:
                best_sharpe = stats['sharpe_ratio']
                best_strategy = strategy

        if best_strategy:
            stats = results['backtest_results'][best_strategy]
            report.append(f"- 表现最佳策略: **{best_strategy}**")
            report.append(f"  - 夏普比率: {stats['sharpe_ratio']:.2f}")
            report.append(f"  - 胜率: {stats['win_rate']:.1f}%")
            report.append(f"  - 盈亏比: {stats['profit_factor']:.2f}")

    report.append("\n### 8.3 风险提示\n")
    report.append("1. 历史回测不代表未来表现")
    report.append("2. 交易成本（佣金、滑点）未纳入计算")
    report.append("3. 极端行情下信号可能失效")
    report.append("4. 建议结合其他技术指标和基本面分析使用")

    # 保存报告
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    print(f"Report saved to {save_path}")
    return '\n'.join(report)


def main():
    """主函数"""
    print("=" * 60)
    print("成交量异常检测研究")
    print("=" * 60)

    # 初始化
    detector = VolumeAnomalyDetector(DB_PATH)
    results = {}

    try:
        # 1. 加载数据
        print("\n[1/7] 加载数据...")
        start_date = '20200101'
        end_date = '20260130'

        df = detector.load_stock_data(start_date=start_date, end_date=end_date)
        market_df = detector.load_market_data(start_date=start_date, end_date=end_date)
        industry_df = detector.load_industry_data()

        results['data_info'] = {
            'start_date': start_date,
            'end_date': end_date,
            'stock_count': df['ts_code'].nunique(),
            'record_count': len(df)
        }
        print(f"  加载 {len(df):,} 条记录, {df['ts_code'].nunique():,} 只股票")

        # 2. 异常检测
        print("\n[2/7] 异常类型检测...")

        # 突发放量
        print("  - 检测突发放量...")
        df = AnomalyDefinitions.detect_volume_spike(df, window=20, threshold=3.0)

        # 持续缩量
        print("  - 检测持续缩量...")
        df = AnomalyDefinitions.detect_volume_shrink(df, window=5, threshold=0.5)

        # 量价背离
        print("  - 检测量价背离...")
        df = AnomalyDefinitions.detect_volume_price_divergence(df, window=5)

        # 天量见天价
        print("  - 检测天量见天价...")
        df = AnomalyDefinitions.detect_climax_volume(df, vol_window=60, price_window=20)

        # 统计异常
        results['anomaly_stats'] = {
            '突发放量(3倍)': {'count': df['is_volume_spike'].sum(),
                            'ratio': df['is_volume_spike'].mean()},
            '持续缩量(5日)': {'count': df['is_sustained_shrink'].sum(),
                             'ratio': df['is_sustained_shrink'].mean()},
            '量价背离': {'count': df['is_divergence'].sum(),
                        'ratio': df['is_divergence'].mean()},
            '天量见天价': {'count': df['is_climax'].sum(),
                          'ratio': df['is_climax'].mean()}
        }

        for k, v in results['anomaly_stats'].items():
            print(f"    {k}: {v['count']:,} ({v['ratio']:.2%})")

        # 3. 统计方法检测
        print("\n[3/7] 统计方法检测...")

        # Z-score检测
        print("  - Z-score检测...")
        df = StatisticalDetection.zscore_detection(df, window=60, threshold=2.5)

        # 移动窗口检测
        print("  - 移动窗口检测...")
        df = StatisticalDetection.moving_window_detection(df, short_window=5,
                                                          long_window=20, threshold=2.0)

        # 百分位数检测
        print("  - 百分位数检测...")
        df = StatisticalDetection.percentile_detection(df, window=252,
                                                       high_pct=95, low_pct=5)

        results['anomaly_stats']['Z-score异常(2.5σ)'] = {
            'count': df['is_zscore_anomaly'].sum(),
            'ratio': df['is_zscore_anomaly'].mean()
        }
        print(f"    Z-score异常: {df['is_zscore_anomaly'].sum():,}")

        # 4. 机器学习检测
        print("\n[4/7] 机器学习方法检测...")

        # 使用采样数据进行ML检测（加速处理）
        sample_df = df.sample(min(500000, len(df)), random_state=42)

        print("  - Isolation Forest检测...")
        sample_df = MLDetection.isolation_forest_detection(sample_df, contamination=0.05)

        print("  - Local Outlier Factor检测...")
        sample_df = MLDetection.local_outlier_factor(sample_df, n_neighbors=20)

        print(f"    IF异常: {sample_df['if_anomaly'].sum():,}")
        print(f"    LOF异常: {sample_df['lof_anomaly'].sum():,}")

        # 5. 信号验证
        print("\n[5/7] 信号验证...")

        # 计算前向收益
        df = SignalValidation.calculate_forward_returns(df, periods=[1, 3, 5, 10, 20])

        # 分析各类信号收益
        results['signal_returns'] = {}

        signal_mapping = {
            '突发放量': 'is_volume_spike',
            '持续缩量': 'is_sustained_shrink',
            '量价背离': 'is_divergence',
            '天量见天价': 'is_climax',
            'Z-score高异常': 'zscore_high',
            'Z-score低异常': 'zscore_low'
        }

        for name, col in signal_mapping.items():
            if col in df.columns:
                print(f"  - 分析{name}信号...")
                returns = SignalValidation.analyze_signal_returns(df, col, [1, 3, 5, 10, 20])
                if returns:
                    results['signal_returns'][name] = returns

        # 市场状态分析
        print("  - 市场状态分析...")
        results['market_state_analysis'] = {}
        for name, col in [('突发放量', 'is_volume_spike'), ('天量见天价', 'is_climax')]:
            if col in df.columns:
                state_results = SignalValidation.market_state_analysis(df, market_df, col)
                if state_results:
                    results['market_state_analysis'][name] = state_results

        # 行业分析
        print("  - 行业特征分析...")
        industry_results = SignalValidation.industry_analysis(df, industry_df, 'is_volume_spike')
        results['industry_analysis'] = industry_results

        # 6. 策略回测
        print("\n[6/7] 策略回测...")

        # 放量突破策略
        print("  - 放量突破策略...")
        df = TradingStrategies.breakout_volume_strategy(df, vol_threshold=2.5, price_breakout_pct=0.02)

        # 缩量回调策略
        print("  - 缩量回调策略...")
        df = TradingStrategies.pullback_buy_strategy(df, shrink_threshold=0.6, pullback_pct=0.05)

        # 量价配合策略
        print("  - 量价配合策略...")
        df = TradingStrategies.volume_price_harmony_strategy(df)

        # 回测
        results['backtest_results'] = {}
        for name, col in [('放量突破', 'breakout_signal'),
                          ('缩量回调', 'pullback_signal'),
                          ('量价配合', 'harmony_signal')]:
            if col in df.columns:
                backtest = backtest_strategy(df, col, holding_period=5)
                results['backtest_results'][name] = backtest
                if 'error' not in backtest:
                    print(f"    {name}: 交易{backtest['total_trades']:,}次, "
                          f"胜率{backtest['win_rate']:.1f}%, 夏普{backtest['sharpe_ratio']:.2f}")

        # 7. 生成报告和可视化
        print("\n[7/7] 生成报告...")

        # 生成可视化
        generate_visualizations(df, market_df, industry_results, REPORT_DIR)

        # 生成报告
        report_path = f"{REPORT_DIR}/volume_anomaly_research_report.md"
        report = generate_report(results, report_path)

        # 保存详细数据
        summary_data = {
            'data_info': results['data_info'],
            'anomaly_stats': results['anomaly_stats'],
            'backtest_results': results['backtest_results']
        }

        with open(f"{REPORT_DIR}/volume_anomaly_summary.json", 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2, default=str)

        print("\n" + "=" * 60)
        print("研究完成!")
        print("=" * 60)
        print(f"\n报告已保存至:")
        print(f"  - {report_path}")
        print(f"  - {REPORT_DIR}/volume_anomaly_analysis.png")
        print(f"  - {REPORT_DIR}/volume_anomaly_summary.json")

    finally:
        detector.close()

    return results


if __name__ == '__main__':
    results = main()
