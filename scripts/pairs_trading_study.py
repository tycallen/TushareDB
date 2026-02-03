#!/usr/bin/env python3
"""
A股市场配对交易策略研究
========================

研究内容：
1. 配对选择方法（相关性、协整、距离、基本面）
2. 协整检验（Engle-Granger、Johansen）
3. 价差建模（OU过程、半衰期）
4. 交易信号设计（Z-score、Bollinger Bands）
5. 策略回测
6. 风险管理

作者: Claude (Anthropic)
日期: 2026-01-31
"""

import duckdb
import pandas as pd
import numpy as np
from datetime import datetime
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# 统计检验库
from scipy import stats
from scipy.optimize import minimize
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

# 绘图
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False

# 配置
DB_PATH = '/Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db'
REPORT_PATH = '/Users/allen/workspace/python/stock/Tushare-DuckDB/reports/quality_check/pairs_trading_study.md'
FIG_DIR = '/Users/allen/workspace/python/stock/Tushare-DuckDB/reports/quality_check/figures'

import os
os.makedirs(FIG_DIR, exist_ok=True)


class PairsTradingResearch:
    """配对交易研究类"""

    def __init__(self, db_path: str):
        self.conn = duckdb.connect(db_path, read_only=True)
        self.report_sections = []

    def add_section(self, title: str, content: str):
        """添加报告章节"""
        self.report_sections.append(f"\n## {title}\n\n{content}")

    def get_industry_stocks(self, industry: str, min_days: int = 500) -> pd.DataFrame:
        """获取行业内股票列表"""
        query = f"""
        SELECT DISTINCT i.ts_code, i.name, i.l2_name,
               COUNT(d.trade_date) as trading_days,
               MIN(d.trade_date) as first_date,
               MAX(d.trade_date) as last_date
        FROM index_member_all i
        JOIN daily d ON i.ts_code = d.ts_code
        WHERE i.l1_name = '{industry}'
          AND i.is_new = 'Y'
          AND d.trade_date >= '20200101'
        GROUP BY i.ts_code, i.name, i.l2_name
        HAVING COUNT(d.trade_date) >= {min_days}
        ORDER BY trading_days DESC
        """
        return self.conn.execute(query).fetchdf()

    def get_price_data(self, ts_codes: list, start_date: str, end_date: str) -> pd.DataFrame:
        """获取价格数据"""
        codes_str = "','".join(ts_codes)
        query = f"""
        SELECT ts_code, trade_date, close, vol, amount
        FROM daily
        WHERE ts_code IN ('{codes_str}')
          AND trade_date >= '{start_date}'
          AND trade_date <= '{end_date}'
        ORDER BY trade_date, ts_code
        """
        df = self.conn.execute(query).fetchdf()
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        return df

    def get_daily_basic(self, ts_codes: list, start_date: str, end_date: str) -> pd.DataFrame:
        """获取估值数据"""
        codes_str = "','".join(ts_codes)
        query = f"""
        SELECT ts_code, trade_date, pe_ttm, pb, total_mv, circ_mv
        FROM daily_basic
        WHERE ts_code IN ('{codes_str}')
          AND trade_date >= '{start_date}'
          AND trade_date <= '{end_date}'
        ORDER BY trade_date, ts_code
        """
        df = self.conn.execute(query).fetchdf()
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        return df


class PairSelection:
    """配对选择方法"""

    @staticmethod
    def correlation_based(price_df: pd.DataFrame, min_corr: float = 0.8) -> list:
        """基于相关性的配对选择"""
        # 将价格数据转为宽表
        pivot = price_df.pivot(index='trade_date', columns='ts_code', values='close')
        pivot = pivot.dropna(axis=1, thresh=int(len(pivot) * 0.9))  # 至少90%数据
        pivot = pivot.dropna()

        # 计算收益率相关性
        returns = pivot.pct_change().dropna()
        corr_matrix = returns.corr()

        pairs = []
        codes = list(corr_matrix.columns)
        for i, code1 in enumerate(codes):
            for code2 in codes[i+1:]:
                corr = corr_matrix.loc[code1, code2]
                if corr >= min_corr:
                    pairs.append({
                        'stock1': code1,
                        'stock2': code2,
                        'correlation': corr
                    })

        return sorted(pairs, key=lambda x: -x['correlation'])

    @staticmethod
    def cointegration_based(price_df: pd.DataFrame, p_value_threshold: float = 0.05) -> list:
        """基于协整的配对选择"""
        pivot = price_df.pivot(index='trade_date', columns='ts_code', values='close')
        pivot = pivot.dropna(axis=1, thresh=int(len(pivot) * 0.9))
        pivot = pivot.dropna()

        pairs = []
        codes = list(pivot.columns)

        for i, code1 in enumerate(codes):
            for code2 in codes[i+1:]:
                try:
                    # Engle-Granger协整检验
                    score, pvalue, _ = coint(pivot[code1], pivot[code2])
                    if pvalue < p_value_threshold:
                        pairs.append({
                            'stock1': code1,
                            'stock2': code2,
                            'coint_pvalue': pvalue,
                            'coint_stat': score
                        })
                except Exception:
                    continue

        return sorted(pairs, key=lambda x: x['coint_pvalue'])

    @staticmethod
    def distance_based(price_df: pd.DataFrame, top_n: int = 20) -> list:
        """基于距离的配对选择（SSD方法）"""
        pivot = price_df.pivot(index='trade_date', columns='ts_code', values='close')
        pivot = pivot.dropna(axis=1, thresh=int(len(pivot) * 0.9))
        pivot = pivot.dropna()

        # 标准化价格
        normalized = (pivot - pivot.mean()) / pivot.std()

        pairs = []
        codes = list(normalized.columns)

        for i, code1 in enumerate(codes):
            for code2 in codes[i+1:]:
                # 计算标准化价格的距离（SSD）
                ssd = np.sum((normalized[code1] - normalized[code2])**2)
                pairs.append({
                    'stock1': code1,
                    'stock2': code2,
                    'ssd': ssd
                })

        return sorted(pairs, key=lambda x: x['ssd'])[:top_n]

    @staticmethod
    def fundamental_based(price_df: pd.DataFrame, basic_df: pd.DataFrame,
                          pe_threshold: float = 0.3, pb_threshold: float = 0.3) -> list:
        """基于基本面的配对选择"""
        # 获取最新估值数据
        latest_date = basic_df['trade_date'].max()
        latest_basic = basic_df[basic_df['trade_date'] == latest_date].copy()

        if len(latest_basic) < 2:
            return []

        pairs = []
        codes = latest_basic['ts_code'].unique()

        for i, code1 in enumerate(codes):
            for code2 in codes[i+1:]:
                try:
                    row1 = latest_basic[latest_basic['ts_code'] == code1].iloc[0]
                    row2 = latest_basic[latest_basic['ts_code'] == code2].iloc[0]

                    # 计算PE和PB的相对差异
                    pe_diff = abs(row1['pe_ttm'] - row2['pe_ttm']) / max(row1['pe_ttm'], row2['pe_ttm'], 1)
                    pb_diff = abs(row1['pb'] - row2['pb']) / max(row1['pb'], row2['pb'], 0.1)

                    # 市值相似度
                    mv_ratio = min(row1['total_mv'], row2['total_mv']) / max(row1['total_mv'], row2['total_mv'], 1)

                    if pe_diff < pe_threshold and pb_diff < pb_threshold and mv_ratio > 0.3:
                        pairs.append({
                            'stock1': code1,
                            'stock2': code2,
                            'pe_diff': pe_diff,
                            'pb_diff': pb_diff,
                            'mv_ratio': mv_ratio,
                            'fundamental_score': 1 - (pe_diff + pb_diff) / 2
                        })
                except Exception:
                    continue

        return sorted(pairs, key=lambda x: -x['fundamental_score'])


class CointegrationTest:
    """协整检验"""

    @staticmethod
    def engle_granger_test(y: np.ndarray, x: np.ndarray) -> dict:
        """Engle-Granger两步法协整检验"""
        # 第一步：OLS回归
        x_with_const = sm.add_constant(x)
        model = sm.OLS(y, x_with_const).fit()
        residuals = model.resid

        # 第二步：对残差进行ADF检验
        adf_result = adfuller(residuals, maxlag=None, autolag='AIC')

        # 计算协整向量
        hedge_ratio = model.params[1]

        return {
            'hedge_ratio': hedge_ratio,
            'intercept': model.params[0],
            'adf_stat': adf_result[0],
            'adf_pvalue': adf_result[1],
            'adf_critical_values': adf_result[4],
            'residuals': residuals,
            'is_cointegrated': adf_result[1] < 0.05,
            'r_squared': model.rsquared
        }

    @staticmethod
    def johansen_test(price_matrix: pd.DataFrame, det_order: int = 0, k_ar_diff: int = 1) -> dict:
        """Johansen协整检验"""
        try:
            result = coint_johansen(price_matrix.values, det_order, k_ar_diff)

            # 迹统计量和最大特征值统计量
            trace_stat = result.lr1
            max_eigen_stat = result.lr2

            # 临界值（90%, 95%, 99%）
            trace_cv = result.cvt
            max_eigen_cv = result.cvm

            # 判断协整关系数量
            n_coint = 0
            for i in range(len(trace_stat)):
                if trace_stat[i] > trace_cv[i, 1]:  # 95%临界值
                    n_coint = len(trace_stat) - i
                    break

            return {
                'trace_stat': trace_stat,
                'trace_cv_95': trace_cv[:, 1],
                'max_eigen_stat': max_eigen_stat,
                'max_eigen_cv_95': max_eigen_cv[:, 1],
                'eigenvectors': result.evec,
                'n_cointegration': n_coint
            }
        except Exception as e:
            return {'error': str(e)}

    @staticmethod
    def test_stability(y: np.ndarray, x: np.ndarray, window_size: int = 120) -> dict:
        """协整关系稳定性检验（滚动窗口）"""
        n = len(y)
        hedge_ratios = []
        pvalues = []
        dates = []

        for i in range(window_size, n):
            y_window = y[i-window_size:i]
            x_window = x[i-window_size:i]

            try:
                result = CointegrationTest.engle_granger_test(y_window, x_window)
                hedge_ratios.append(result['hedge_ratio'])
                pvalues.append(result['adf_pvalue'])
                dates.append(i)
            except:
                hedge_ratios.append(np.nan)
                pvalues.append(np.nan)
                dates.append(i)

        hedge_ratios = np.array(hedge_ratios)
        pvalues = np.array(pvalues)

        return {
            'hedge_ratios': hedge_ratios,
            'pvalues': pvalues,
            'dates': dates,
            'hedge_ratio_std': np.nanstd(hedge_ratios),
            'hedge_ratio_mean': np.nanmean(hedge_ratios),
            'pct_cointegrated': np.nanmean(np.array(pvalues) < 0.05) * 100,
            'is_stable': np.nanstd(hedge_ratios) / np.nanmean(np.abs(hedge_ratios)) < 0.2
        }

    @staticmethod
    def detect_structural_break(residuals: np.ndarray, window: int = 60) -> dict:
        """结构变化检测（Chow检验思想）"""
        n = len(residuals)
        break_points = []

        # 计算滚动均值和标准差
        rolling_mean = pd.Series(residuals).rolling(window).mean().values
        rolling_std = pd.Series(residuals).rolling(window).std().values

        # 检测均值突变
        for i in range(window*2, n):
            mean_before = np.mean(residuals[i-window:i])
            mean_after = np.mean(residuals[max(0,i-window*2):i-window])
            std_combined = np.std(residuals[i-window*2:i])

            if std_combined > 0:
                t_stat = abs(mean_before - mean_after) / (std_combined / np.sqrt(window))
                if t_stat > 2.5:  # 显著性阈值
                    break_points.append({
                        'index': i,
                        't_stat': t_stat
                    })

        return {
            'break_points': break_points,
            'n_breaks': len(break_points),
            'has_structural_break': len(break_points) > 0
        }


class SpreadModeling:
    """价差建模"""

    @staticmethod
    def calculate_spread(y: np.ndarray, x: np.ndarray, hedge_ratio: float, intercept: float = 0) -> np.ndarray:
        """计算价差"""
        return y - hedge_ratio * x - intercept

    @staticmethod
    def spread_statistics(spread: np.ndarray) -> dict:
        """价差统计特性"""
        return {
            'mean': np.mean(spread),
            'std': np.std(spread),
            'skewness': stats.skew(spread),
            'kurtosis': stats.kurtosis(spread),
            'min': np.min(spread),
            'max': np.max(spread),
            'median': np.median(spread),
            'normality_pvalue': stats.normaltest(spread)[1] if len(spread) > 20 else np.nan
        }

    @staticmethod
    def mean_reversion_speed(spread: np.ndarray) -> dict:
        """均值回归速度（AR(1)模型）"""
        spread_lag = spread[:-1]
        spread_diff = spread[1:] - spread[:-1]

        # 回归: delta_spread = alpha + beta * spread_lag + epsilon
        X = sm.add_constant(spread_lag)
        model = sm.OLS(spread_diff, X).fit()

        # 均值回归速度 theta = -beta
        theta = -model.params[1]

        return {
            'theta': theta,
            'theta_se': model.bse[1],
            'theta_pvalue': model.pvalues[1],
            'is_mean_reverting': theta > 0 and model.pvalues[1] < 0.05
        }

    @staticmethod
    def half_life(spread: np.ndarray) -> dict:
        """半衰期计算"""
        spread_lag = spread[:-1]
        spread_diff = spread[1:] - spread[:-1]

        X = sm.add_constant(spread_lag)
        model = sm.OLS(spread_diff, X).fit()

        # 半衰期 = -ln(2) / ln(1 + beta) ≈ -ln(2) / beta (当beta较小时)
        beta = model.params[1]

        if beta < 0:  # 均值回归条件
            half_life = -np.log(2) / beta
        else:
            half_life = np.inf

        return {
            'half_life': half_life,
            'beta': beta,
            'half_life_days': half_life if half_life != np.inf else None,
            'is_valid': beta < 0
        }

    @staticmethod
    def ou_process_estimation(spread: np.ndarray, dt: float = 1/252) -> dict:
        """OU过程参数估计
        dS = theta * (mu - S) * dt + sigma * dW
        """
        n = len(spread)
        S = spread[:-1]
        dS = spread[1:] - spread[:-1]

        # 最大似然估计
        def neg_log_likelihood(params):
            theta, mu, sigma = params
            if theta <= 0 or sigma <= 0:
                return 1e10

            # OU过程的条件分布
            mean = S + theta * (mu - S) * dt
            var = sigma**2 * (1 - np.exp(-2*theta*dt)) / (2*theta)

            if var <= 0:
                return 1e10

            ll = -0.5 * np.sum(np.log(2*np.pi*var) + (spread[1:] - mean)**2 / var)
            return -ll

        # 初始值估计
        theta_init = 0.5
        mu_init = np.mean(spread)
        sigma_init = np.std(dS) / np.sqrt(dt)

        try:
            result = minimize(
                neg_log_likelihood,
                [theta_init, mu_init, sigma_init],
                method='L-BFGS-B',
                bounds=[(0.01, 10), (None, None), (0.01, None)]
            )

            theta, mu, sigma = result.x
            half_life = np.log(2) / theta

            return {
                'theta': theta,
                'mu': mu,
                'sigma': sigma,
                'half_life': half_life,
                'half_life_days': half_life * 252,  # 转换为交易日
                'convergence': result.success
            }
        except Exception as e:
            return {'error': str(e)}


class TradingSignals:
    """交易信号设计"""

    @staticmethod
    def zscore_signal(spread: np.ndarray, lookback: int = 20,
                      entry_threshold: float = 2.0, exit_threshold: float = 0.0) -> dict:
        """Z-score信号"""
        spread_series = pd.Series(spread)
        rolling_mean = spread_series.rolling(lookback).mean()
        rolling_std = spread_series.rolling(lookback).std()

        zscore = (spread_series - rolling_mean) / rolling_std

        # 信号生成
        signals = np.zeros(len(spread))
        position = 0

        for i in range(lookback, len(spread)):
            z = zscore.iloc[i]

            if position == 0:
                if z > entry_threshold:
                    signals[i] = -1  # 做空价差
                    position = -1
                elif z < -entry_threshold:
                    signals[i] = 1  # 做多价差
                    position = 1
            elif position == 1:
                if z > exit_threshold:
                    signals[i] = 0  # 平仓
                    position = 0
                else:
                    signals[i] = 1  # 持仓
            elif position == -1:
                if z < -exit_threshold:
                    signals[i] = 0  # 平仓
                    position = 0
                else:
                    signals[i] = -1  # 持仓

        return {
            'zscore': zscore.values,
            'signals': signals,
            'rolling_mean': rolling_mean.values,
            'rolling_std': rolling_std.values
        }

    @staticmethod
    def bollinger_bands_signal(spread: np.ndarray, lookback: int = 20,
                               num_std: float = 2.0) -> dict:
        """布林带信号"""
        spread_series = pd.Series(spread)
        rolling_mean = spread_series.rolling(lookback).mean()
        rolling_std = spread_series.rolling(lookback).std()

        upper_band = rolling_mean + num_std * rolling_std
        lower_band = rolling_mean - num_std * rolling_std

        signals = np.zeros(len(spread))
        position = 0

        for i in range(lookback, len(spread)):
            if position == 0:
                if spread[i] > upper_band.iloc[i]:
                    signals[i] = -1  # 做空价差
                    position = -1
                elif spread[i] < lower_band.iloc[i]:
                    signals[i] = 1  # 做多价差
                    position = 1
            elif position == 1:
                if spread[i] > rolling_mean.iloc[i]:
                    signals[i] = 0
                    position = 0
                else:
                    signals[i] = 1
            elif position == -1:
                if spread[i] < rolling_mean.iloc[i]:
                    signals[i] = 0
                    position = 0
                else:
                    signals[i] = -1

        return {
            'upper_band': upper_band.values,
            'lower_band': lower_band.values,
            'middle_band': rolling_mean.values,
            'signals': signals
        }

    @staticmethod
    def dynamic_threshold_signal(spread: np.ndarray, lookback: int = 60,
                                 vol_lookback: int = 20, base_threshold: float = 2.0) -> dict:
        """动态阈值信号（基于波动率调整）"""
        spread_series = pd.Series(spread)
        rolling_mean = spread_series.rolling(lookback).mean()
        rolling_std = spread_series.rolling(lookback).std()

        # 动态调整阈值：波动率高时提高阈值
        recent_vol = spread_series.rolling(vol_lookback).std()
        long_term_vol = spread_series.rolling(lookback).std()
        vol_ratio = recent_vol / long_term_vol

        # 动态阈值
        dynamic_threshold = base_threshold * vol_ratio
        dynamic_threshold = dynamic_threshold.clip(lower=1.5, upper=3.0)

        zscore = (spread_series - rolling_mean) / rolling_std

        signals = np.zeros(len(spread))
        position = 0

        for i in range(lookback, len(spread)):
            z = zscore.iloc[i]
            thresh = dynamic_threshold.iloc[i]

            if position == 0:
                if z > thresh:
                    signals[i] = -1
                    position = -1
                elif z < -thresh:
                    signals[i] = 1
                    position = 1
            elif position == 1:
                if z > 0:
                    signals[i] = 0
                    position = 0
                else:
                    signals[i] = 1
            elif position == -1:
                if z < 0:
                    signals[i] = 0
                    position = 0
                else:
                    signals[i] = -1

        return {
            'zscore': zscore.values,
            'dynamic_threshold': dynamic_threshold.values,
            'signals': signals
        }

    @staticmethod
    def signal_with_stop_loss(spread: np.ndarray, signals: np.ndarray,
                              stop_loss_std: float = 3.0, lookback: int = 20) -> dict:
        """添加止损的信号"""
        spread_series = pd.Series(spread)
        rolling_std = spread_series.rolling(lookback).std()

        modified_signals = signals.copy()
        position = 0
        entry_price = 0
        stop_loss_triggered = 0

        for i in range(lookback, len(spread)):
            if signals[i] != 0 and position == 0:
                position = signals[i]
                entry_price = spread[i]
            elif position != 0:
                # 检查止损
                loss = (spread[i] - entry_price) * position
                if loss < -stop_loss_std * rolling_std.iloc[i]:
                    modified_signals[i] = 0
                    position = 0
                    stop_loss_triggered += 1
                elif signals[i] == 0:
                    position = 0

        return {
            'signals': modified_signals,
            'stop_loss_count': stop_loss_triggered
        }


class PairsBacktest:
    """配对交易回测"""

    def __init__(self, price1: np.ndarray, price2: np.ndarray,
                 hedge_ratio: float, dates: np.ndarray):
        self.price1 = price1
        self.price2 = price2
        self.hedge_ratio = hedge_ratio
        self.dates = dates

    def run_backtest(self, signals: np.ndarray,
                     transaction_cost: float = 0.001,
                     capital: float = 1000000) -> dict:
        """运行回测"""
        n = len(signals)

        # 收益计算
        returns1 = np.diff(self.price1) / self.price1[:-1]
        returns2 = np.diff(self.price2) / self.price2[:-1]

        # 策略收益 = 做多股票1 + 做空股票2 * hedge_ratio（或相反）
        # position > 0: 做多价差 (long stock1, short stock2)
        # position < 0: 做空价差 (short stock1, long stock2)

        pnl = np.zeros(n)
        position = 0
        positions = np.zeros(n)
        trades = 0

        for i in range(1, n):
            if i < len(returns1):
                if position != 0:
                    # 持仓收益
                    daily_pnl = position * (returns1[i-1] - self.hedge_ratio * returns2[i-1])
                    pnl[i] = daily_pnl

                # 仓位变化
                new_position = signals[i]
                if new_position != position:
                    # 交易成本
                    pnl[i] -= abs(new_position - position) * transaction_cost * (1 + self.hedge_ratio)
                    trades += 1

                position = new_position
                positions[i] = position

        # 累计收益
        cumulative_pnl = np.cumsum(pnl)
        equity_curve = capital * (1 + cumulative_pnl)

        # 绩效指标
        total_return = cumulative_pnl[-1]
        annual_return = total_return * 252 / n

        # 计算夏普比率
        daily_returns = pnl[pnl != 0] if np.sum(pnl != 0) > 0 else pnl
        if len(daily_returns) > 1 and np.std(daily_returns) > 0:
            sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
        else:
            sharpe = 0

        # 最大回撤
        cummax = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - cummax) / cummax
        max_drawdown = np.min(drawdown)

        # 胜率
        trade_returns = []
        entry_pnl = 0
        in_trade = False
        for i in range(1, n):
            if positions[i] != 0 and not in_trade:
                in_trade = True
                entry_pnl = cumulative_pnl[i-1] if i > 0 else 0
            elif positions[i] == 0 and in_trade:
                trade_returns.append(cumulative_pnl[i] - entry_pnl)
                in_trade = False

        win_rate = np.mean(np.array(trade_returns) > 0) if len(trade_returns) > 0 else 0

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'num_trades': trades,
            'win_rate': win_rate,
            'equity_curve': equity_curve,
            'pnl': pnl,
            'positions': positions,
            'cumulative_pnl': cumulative_pnl
        }


class RiskManagement:
    """风险管理"""

    @staticmethod
    def pair_breakdown_risk(stability_result: dict, recent_pvalue: float) -> dict:
        """配对关系破裂风险评估"""
        risk_score = 0
        risk_factors = []

        # 协整稳定性
        if not stability_result.get('is_stable', False):
            risk_score += 30
            risk_factors.append("对冲比率不稳定")

        # 协整显著性
        if stability_result.get('pct_cointegrated', 100) < 80:
            risk_score += 20
            risk_factors.append(f"协整关系仅在{stability_result['pct_cointegrated']:.1f}%的时间有效")

        # 最近协整p值
        if recent_pvalue > 0.1:
            risk_score += 30
            risk_factors.append(f"当前协整p值较高: {recent_pvalue:.3f}")
        elif recent_pvalue > 0.05:
            risk_score += 15
            risk_factors.append(f"当前协整p值边缘: {recent_pvalue:.3f}")

        return {
            'risk_score': risk_score,
            'risk_level': 'HIGH' if risk_score >= 50 else 'MEDIUM' if risk_score >= 30 else 'LOW',
            'risk_factors': risk_factors
        }

    @staticmethod
    def liquidity_risk(vol1: np.ndarray, vol2: np.ndarray,
                       amount1: np.ndarray, amount2: np.ndarray,
                       position_size: float) -> dict:
        """流动性风险评估"""
        # 平均成交量和成交额
        avg_vol1, avg_vol2 = np.mean(vol1), np.mean(vol2)
        avg_amount1, avg_amount2 = np.mean(amount1), np.mean(amount2)

        # 假设单次交易不超过日均成交额的5%
        max_position1 = avg_amount1 * 0.05 * 10000  # 成交额单位是千元
        max_position2 = avg_amount2 * 0.05 * 10000

        # 流动性风险
        liquidity_ratio1 = position_size / max_position1 if max_position1 > 0 else float('inf')
        liquidity_ratio2 = position_size / max_position2 if max_position2 > 0 else float('inf')

        max_ratio = max(liquidity_ratio1, liquidity_ratio2)

        return {
            'avg_daily_amount1': avg_amount1 * 10000,
            'avg_daily_amount2': avg_amount2 * 10000,
            'max_position_size': min(max_position1, max_position2),
            'current_liquidity_ratio': max_ratio,
            'liquidity_risk': 'HIGH' if max_ratio > 0.5 else 'MEDIUM' if max_ratio > 0.2 else 'LOW'
        }

    @staticmethod
    def leverage_control(equity: float, position_value: float,
                         max_leverage: float = 2.0) -> dict:
        """杠杆控制"""
        current_leverage = position_value / equity if equity > 0 else float('inf')

        recommended_position = equity * max_leverage

        return {
            'current_leverage': current_leverage,
            'max_leverage': max_leverage,
            'recommended_max_position': recommended_position,
            'leverage_status': 'EXCEEDED' if current_leverage > max_leverage else 'OK',
            'utilization_rate': current_leverage / max_leverage
        }


def run_pairs_trading_study():
    """运行配对交易研究"""

    print("=" * 60)
    print("A股市场配对交易策略研究")
    print("=" * 60)

    research = PairsTradingResearch(DB_PATH)

    # ========== 1. 数据准备 ==========
    print("\n[1] 数据准备...")

    # 选择银行行业作为研究对象
    bank_stocks = research.get_industry_stocks('银行', min_days=800)
    print(f"  银行股数量: {len(bank_stocks)}")

    # 获取价格数据
    ts_codes = bank_stocks['ts_code'].tolist()
    price_df = research.get_price_data(ts_codes, '20200101', '20251231')
    basic_df = research.get_daily_basic(ts_codes, '20200101', '20251231')

    print(f"  价格数据行数: {len(price_df):,}")
    print(f"  日期范围: {price_df['trade_date'].min()} - {price_df['trade_date'].max()}")

    # ========== 2. 配对选择 ==========
    print("\n[2] 配对选择方法比较...")

    # 2.1 相关性配对
    corr_pairs = PairSelection.correlation_based(price_df, min_corr=0.85)
    print(f"  相关性配对数量 (ρ≥0.85): {len(corr_pairs)}")

    # 2.2 协整配对
    coint_pairs = PairSelection.cointegration_based(price_df, p_value_threshold=0.05)
    print(f"  协整配对数量 (p<0.05): {len(coint_pairs)}")

    # 2.3 距离配对
    dist_pairs = PairSelection.distance_based(price_df, top_n=20)
    print(f"  距离配对数量: {len(dist_pairs)}")

    # 2.4 基本面配对
    fund_pairs = PairSelection.fundamental_based(price_df, basic_df)
    print(f"  基本面配对数量: {len(fund_pairs)}")

    # ========== 3. 选取最佳配对进行深入分析 ==========
    print("\n[3] 深入分析最佳配对...")

    # 选择协整检验最显著的配对
    if len(coint_pairs) > 0:
        best_pair = coint_pairs[0]
        stock1, stock2 = best_pair['stock1'], best_pair['stock2']
    else:
        # 备选：选择相关性最高的配对
        best_pair = corr_pairs[0] if len(corr_pairs) > 0 else None
        if best_pair:
            stock1, stock2 = best_pair['stock1'], best_pair['stock2']
        else:
            print("  错误：没有找到合适的配对")
            return

    # 获取配对名称
    stock_names = research.conn.execute(f"""
        SELECT ts_code, name FROM index_member_all
        WHERE ts_code IN ('{stock1}', '{stock2}')
    """).fetchdf()
    name1 = stock_names[stock_names['ts_code'] == stock1]['name'].values[0]
    name2 = stock_names[stock_names['ts_code'] == stock2]['name'].values[0]

    print(f"  选择配对: {name1}({stock1}) vs {name2}({stock2})")

    # 准备配对价格数据
    pivot = price_df.pivot(index='trade_date', columns='ts_code', values='close')
    pivot = pivot[[stock1, stock2]].dropna()
    dates = pivot.index.values
    price1 = pivot[stock1].values
    price2 = pivot[stock2].values

    print(f"  配对数据量: {len(pivot)} 个交易日")

    # ========== 4. 协整检验 ==========
    print("\n[4] 协整检验...")

    # 4.1 Engle-Granger检验
    eg_result = CointegrationTest.engle_granger_test(price1, price2)
    print(f"  Engle-Granger检验:")
    print(f"    对冲比率: {eg_result['hedge_ratio']:.4f}")
    print(f"    ADF统计量: {eg_result['adf_stat']:.4f}")
    print(f"    p值: {eg_result['adf_pvalue']:.4f}")
    print(f"    是否协整: {'是' if eg_result['is_cointegrated'] else '否'}")

    # 4.2 Johansen检验
    johansen_result = CointegrationTest.johansen_test(pivot)
    if 'error' not in johansen_result:
        print(f"  Johansen检验:")
        print(f"    协整关系数量: {johansen_result['n_cointegration']}")
        print(f"    迹统计量: {johansen_result['trace_stat']}")
        print(f"    95%临界值: {johansen_result['trace_cv_95']}")

    # 4.3 协整稳定性检验
    stability_result = CointegrationTest.test_stability(price1, price2, window_size=120)
    print(f"  协整稳定性:")
    print(f"    对冲比率均值: {stability_result['hedge_ratio_mean']:.4f}")
    print(f"    对冲比率标准差: {stability_result['hedge_ratio_std']:.4f}")
    print(f"    协整有效时间比例: {stability_result['pct_cointegrated']:.1f}%")
    print(f"    稳定性判断: {'稳定' if stability_result['is_stable'] else '不稳定'}")

    # 4.4 结构变化检测
    residuals = eg_result['residuals']
    break_result = CointegrationTest.detect_structural_break(residuals)
    print(f"  结构变化检测:")
    print(f"    断点数量: {break_result['n_breaks']}")
    print(f"    是否存在结构变化: {'是' if break_result['has_structural_break'] else '否'}")

    # ========== 5. 价差建模 ==========
    print("\n[5] 价差建模...")

    # 计算价差
    spread = SpreadModeling.calculate_spread(price1, price2,
                                             eg_result['hedge_ratio'],
                                             eg_result['intercept'])

    # 5.1 价差统计特性
    spread_stats = SpreadModeling.spread_statistics(spread)
    print(f"  价差统计特性:")
    print(f"    均值: {spread_stats['mean']:.4f}")
    print(f"    标准差: {spread_stats['std']:.4f}")
    print(f"    偏度: {spread_stats['skewness']:.4f}")
    print(f"    峰度: {spread_stats['kurtosis']:.4f}")

    # 5.2 均值回归速度
    mr_result = SpreadModeling.mean_reversion_speed(spread)
    print(f"  均值回归:")
    print(f"    theta: {mr_result['theta']:.4f}")
    print(f"    是否均值回归: {'是' if mr_result['is_mean_reverting'] else '否'}")

    # 5.3 半衰期
    hl_result = SpreadModeling.half_life(spread)
    print(f"  半衰期:")
    if hl_result['is_valid']:
        print(f"    半衰期: {hl_result['half_life_days']:.1f} 天")
    else:
        print(f"    半衰期: 无效（非均值回归）")

    # 5.4 OU过程估计
    ou_result = SpreadModeling.ou_process_estimation(spread)
    if 'error' not in ou_result:
        print(f"  OU过程参数:")
        print(f"    theta (回归速度): {ou_result['theta']:.4f}")
        print(f"    mu (长期均值): {ou_result['mu']:.4f}")
        print(f"    sigma (波动率): {ou_result['sigma']:.4f}")
        print(f"    半衰期: {ou_result['half_life_days']:.1f} 天")

    # ========== 6. 交易信号设计 ==========
    print("\n[6] 交易信号设计...")

    # 6.1 Z-score信号
    zscore_signal = TradingSignals.zscore_signal(spread, lookback=20,
                                                  entry_threshold=2.0,
                                                  exit_threshold=0.0)
    print(f"  Z-score信号:")
    print(f"    开仓次数: {np.sum(np.diff(zscore_signal['signals']) != 0)}")

    # 6.2 布林带信号
    bb_signal = TradingSignals.bollinger_bands_signal(spread, lookback=20, num_std=2.0)
    print(f"  布林带信号:")
    print(f"    开仓次数: {np.sum(np.diff(bb_signal['signals']) != 0)}")

    # 6.3 动态阈值信号
    dyn_signal = TradingSignals.dynamic_threshold_signal(spread, lookback=60,
                                                          vol_lookback=20,
                                                          base_threshold=2.0)
    print(f"  动态阈值信号:")
    print(f"    开仓次数: {np.sum(np.diff(dyn_signal['signals']) != 0)}")

    # 6.4 添加止损
    sl_signal = TradingSignals.signal_with_stop_loss(spread, zscore_signal['signals'],
                                                      stop_loss_std=3.0)
    print(f"  止损信号:")
    print(f"    触发止损次数: {sl_signal['stop_loss_count']}")

    # ========== 7. 策略回测 ==========
    print("\n[7] 策略回测...")

    backtest = PairsBacktest(price1, price2, eg_result['hedge_ratio'], dates)

    # 不同信号的回测结果
    signals_dict = {
        'Z-score': zscore_signal['signals'],
        'Bollinger': bb_signal['signals'],
        'Dynamic': dyn_signal['signals'],
        'Z-score+止损': sl_signal['signals']
    }

    backtest_results = {}
    for name, signals in signals_dict.items():
        result = backtest.run_backtest(signals, transaction_cost=0.001)
        backtest_results[name] = result
        print(f"  {name}策略:")
        print(f"    总收益: {result['total_return']*100:.2f}%")
        print(f"    年化收益: {result['annual_return']*100:.2f}%")
        print(f"    夏普比率: {result['sharpe_ratio']:.2f}")
        print(f"    最大回撤: {result['max_drawdown']*100:.2f}%")
        print(f"    交易次数: {result['num_trades']}")
        print(f"    胜率: {result['win_rate']*100:.1f}%")

    # 交易成本敏感性分析
    print("\n  交易成本敏感性分析 (Z-score策略):")
    for tc in [0.0005, 0.001, 0.002, 0.003]:
        result = backtest.run_backtest(zscore_signal['signals'], transaction_cost=tc)
        print(f"    成本{tc*100:.2f}%: 年化{result['annual_return']*100:.2f}%, 夏普{result['sharpe_ratio']:.2f}")

    # ========== 8. 风险管理 ==========
    print("\n[8] 风险管理...")

    # 8.1 配对关系破裂风险
    recent_pvalue = stability_result['pvalues'][-1] if len(stability_result['pvalues']) > 0 else 1.0
    breakdown_risk = RiskManagement.pair_breakdown_risk(stability_result, recent_pvalue)
    print(f"  配对关系破裂风险:")
    print(f"    风险评分: {breakdown_risk['risk_score']}")
    print(f"    风险等级: {breakdown_risk['risk_level']}")
    if breakdown_risk['risk_factors']:
        for factor in breakdown_risk['risk_factors']:
            print(f"    - {factor}")

    # 8.2 流动性风险
    vol_df = price_df.pivot(index='trade_date', columns='ts_code', values='vol')
    amount_df = price_df.pivot(index='trade_date', columns='ts_code', values='amount')

    vol1 = vol_df[stock1].dropna().values
    vol2 = vol_df[stock2].dropna().values
    amount1 = amount_df[stock1].dropna().values
    amount2 = amount_df[stock2].dropna().values

    liquidity_risk = RiskManagement.liquidity_risk(vol1, vol2, amount1, amount2,
                                                   position_size=5000000)  # 500万仓位
    print(f"  流动性风险:")
    print(f"    {name1}日均成交额: {liquidity_risk['avg_daily_amount1']/10000:.0f}万")
    print(f"    {name2}日均成交额: {liquidity_risk['avg_daily_amount2']/10000:.0f}万")
    print(f"    建议最大仓位: {liquidity_risk['max_position_size']/10000:.0f}万")
    print(f"    流动性风险等级: {liquidity_risk['liquidity_risk']}")

    # 8.3 杠杆控制
    leverage = RiskManagement.leverage_control(equity=10000000, position_value=5000000)
    print(f"  杠杆控制:")
    print(f"    当前杠杆: {leverage['current_leverage']:.2f}x")
    print(f"    最大杠杆: {leverage['max_leverage']:.2f}x")
    print(f"    杠杆状态: {leverage['leverage_status']}")

    # ========== 9. 策略容量分析 ==========
    print("\n[9] 策略容量分析...")

    # 基于流动性的容量估算
    daily_turnover = min(liquidity_risk['avg_daily_amount1'],
                        liquidity_risk['avg_daily_amount2'])
    # 假设单日交易不超过市场成交额的5%，持仓周期约10天
    max_capacity = daily_turnover * 0.05 * 10
    print(f"  单配对策略容量估算: {max_capacity/10000:.0f}万")

    # 如果使用多个配对
    n_pairs = len(coint_pairs)
    total_capacity = max_capacity * min(n_pairs, 10)  # 最多使用10个配对
    print(f"  多配对策略容量估算 ({min(n_pairs, 10)}对): {total_capacity/100000000:.2f}亿")

    # ========== 10. 生成可视化 ==========
    print("\n[10] 生成可视化...")

    # 图1: 配对价格走势
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    # 价格走势
    ax1 = axes[0, 0]
    ax1.plot(dates, price1, label=name1, alpha=0.8)
    ax1.plot(dates, price2 * eg_result['hedge_ratio'], label=f"{name2}×{eg_result['hedge_ratio']:.2f}", alpha=0.8)
    ax1.set_title('配对价格走势（对冲后）')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 价差
    ax2 = axes[0, 1]
    ax2.plot(dates, spread, label='价差', color='purple', alpha=0.8)
    ax2.axhline(y=np.mean(spread), color='r', linestyle='--', label='均值')
    ax2.axhline(y=np.mean(spread) + 2*np.std(spread), color='g', linestyle='--', alpha=0.5)
    ax2.axhline(y=np.mean(spread) - 2*np.std(spread), color='g', linestyle='--', alpha=0.5)
    ax2.set_title('价差序列')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Z-score
    ax3 = axes[1, 0]
    zscore = zscore_signal['zscore']
    ax3.plot(dates, zscore, label='Z-score', color='blue', alpha=0.8)
    ax3.axhline(y=2, color='r', linestyle='--', alpha=0.5)
    ax3.axhline(y=-2, color='r', linestyle='--', alpha=0.5)
    ax3.axhline(y=0, color='g', linestyle='--', alpha=0.5)
    ax3.fill_between(dates, 2, 3, alpha=0.2, color='red')
    ax3.fill_between(dates, -2, -3, alpha=0.2, color='green')
    ax3.set_title('Z-score信号')
    ax3.set_ylim(-4, 4)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 策略权益曲线
    ax4 = axes[1, 1]
    for name, result in backtest_results.items():
        ax4.plot(dates, result['equity_curve'], label=name, alpha=0.8)
    ax4.set_title('策略权益曲线')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 对冲比率稳定性
    ax5 = axes[2, 0]
    hr_dates = dates[120:]
    ax5.plot(hr_dates, stability_result['hedge_ratios'], label='对冲比率', color='orange')
    ax5.axhline(y=eg_result['hedge_ratio'], color='r', linestyle='--', label='全样本估计')
    ax5.set_title('对冲比率稳定性（滚动120日）')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 回撤分析
    ax6 = axes[2, 1]
    best_result = backtest_results['Z-score']
    cummax = np.maximum.accumulate(best_result['equity_curve'])
    drawdown = (best_result['equity_curve'] - cummax) / cummax * 100
    ax6.fill_between(dates, drawdown, 0, alpha=0.5, color='red')
    ax6.set_title('回撤分析 (Z-score策略)')
    ax6.set_ylabel('回撤 (%)')
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = f"{FIG_DIR}/pairs_trading_analysis.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存图表: {fig_path}")

    # 图2: 协整检验可视化
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))

    # 残差序列
    ax1 = axes2[0, 0]
    ax1.plot(dates, residuals, color='purple', alpha=0.8)
    ax1.axhline(y=0, color='r', linestyle='--')
    ax1.set_title('协整残差序列')
    ax1.grid(True, alpha=0.3)

    # 残差分布
    ax2 = axes2[0, 1]
    ax2.hist(residuals, bins=50, density=True, alpha=0.7, color='purple')
    x = np.linspace(residuals.min(), residuals.max(), 100)
    ax2.plot(x, stats.norm.pdf(x, np.mean(residuals), np.std(residuals)),
             'r-', lw=2, label='正态分布')
    ax2.set_title('残差分布')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 协整p值滚动
    ax3 = axes2[1, 0]
    pv_dates = dates[120:]
    ax3.plot(pv_dates, stability_result['pvalues'], color='blue', alpha=0.8)
    ax3.axhline(y=0.05, color='r', linestyle='--', label='5%显著性水平')
    ax3.axhline(y=0.10, color='orange', linestyle='--', label='10%显著性水平')
    ax3.set_title('协整检验p值（滚动120日）')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 0.5)

    # QQ图
    ax4 = axes2[1, 1]
    stats.probplot(residuals, dist="norm", plot=ax4)
    ax4.set_title('残差Q-Q图')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    fig2_path = f"{FIG_DIR}/cointegration_analysis.png"
    plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存图表: {fig2_path}")

    # ========== 11. 生成报告 ==========
    print("\n[11] 生成研究报告...")

    report = f"""# A股市场配对交易策略研究报告

> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
> 研究范围: 银行板块
> 数据区间: 2020-01-01 至 2025-12-31

---

## 1. 研究摘要

本报告对A股银行板块进行配对交易策略研究，包括配对选择、协整检验、价差建模、
交易信号设计、策略回测和风险管理等内容。

### 主要发现

- **配对选择**: 在{len(bank_stocks)}只银行股中，发现{len(coint_pairs)}个协整配对，
  {len(corr_pairs)}个高相关配对
- **最佳配对**: {name1}({stock1}) vs {name2}({stock2})
- **协整检验**: ADF统计量 {eg_result['adf_stat']:.4f}, p值 {eg_result['adf_pvalue']:.4f}
- **策略表现**: Z-score策略年化收益 {backtest_results['Z-score']['annual_return']*100:.2f}%,
  夏普比率 {backtest_results['Z-score']['sharpe_ratio']:.2f}

---

## 2. 配对选择方法

### 2.1 基于相关性的配对

计算股票收益率相关系数，选择相关性大于0.85的配对。

| 排名 | 股票1 | 股票2 | 相关系数 |
|------|-------|-------|----------|
"""

    # 添加相关性配对表格
    for i, pair in enumerate(corr_pairs[:10]):
        report += f"| {i+1} | {pair['stock1']} | {pair['stock2']} | {pair['correlation']:.4f} |\n"

    report += f"""
**方法评价**:
- 优点: 计算简单，直观易懂
- 缺点: 相关性不代表协整关系，可能出现伪回归

### 2.2 基于协整的配对

使用Engle-Granger检验，选择p值小于0.05的配对。

| 排名 | 股票1 | 股票2 | 协整p值 | 协整统计量 |
|------|-------|-------|---------|-----------|
"""

    for i, pair in enumerate(coint_pairs[:10]):
        report += f"| {i+1} | {pair['stock1']} | {pair['stock2']} | {pair['coint_pvalue']:.4f} | {pair['coint_stat']:.4f} |\n"

    report += f"""
**方法评价**:
- 优点: 理论基础扎实，价差具有均值回归特性
- 缺点: 协整关系可能不稳定，需要定期重新检验

### 2.3 基于距离的配对

计算标准化价格的SSD（Sum of Squared Differences），选择距离最小的配对。

| 排名 | 股票1 | 股票2 | SSD |
|------|-------|-------|-----|
"""

    for i, pair in enumerate(dist_pairs[:10]):
        report += f"| {i+1} | {pair['stock1']} | {pair['stock2']} | {pair['ssd']:.2f} |\n"

    report += f"""
**方法评价**:
- 优点: 不依赖特定统计假设
- 缺点: 纯统计方法，缺乏经济学解释

### 2.4 基于基本面的配对

选择PE、PB相似且市值接近的股票对。

| 排名 | 股票1 | 股票2 | PE差异 | PB差异 | 市值比 |
|------|-------|-------|--------|--------|--------|
"""

    for i, pair in enumerate(fund_pairs[:10]):
        report += f"| {i+1} | {pair['stock1']} | {pair['stock2']} | {pair['pe_diff']:.2%} | {pair['pb_diff']:.2%} | {pair['mv_ratio']:.2%} |\n"

    report += f"""
**方法评价**:
- 优点: 具有基本面支撑，配对关系更稳健
- 缺点: 需要额外的估值数据，更新频率较低

---

## 3. 协整检验

### 3.1 Engle-Granger两步法

对选定配对 **{name1}** vs **{name2}** 进行协整检验：

**第一步: OLS回归**
```
{name1}_price = α + β × {name2}_price + ε
```

回归结果:
- 对冲比率 β = {eg_result['hedge_ratio']:.4f}
- 截距项 α = {eg_result['intercept']:.4f}
- R² = {eg_result['r_squared']:.4f}

**第二步: 残差ADF检验**
- ADF统计量: {eg_result['adf_stat']:.4f}
- p值: {eg_result['adf_pvalue']:.4f}
- 临界值(1%): {eg_result['adf_critical_values']['1%']:.4f}
- 临界值(5%): {eg_result['adf_critical_values']['5%']:.4f}
- 临界值(10%): {eg_result['adf_critical_values']['10%']:.4f}

**结论**: {'拒绝单位根原假设，价差序列平稳，两股票协整' if eg_result['is_cointegrated'] else '无法拒绝单位根原假设，可能不协整'}

### 3.2 Johansen检验

"""

    if 'error' not in johansen_result:
        report += f"""| 原假设 | 迹统计量 | 95%临界值 | 结论 |
|--------|----------|-----------|------|
| r=0 | {johansen_result['trace_stat'][0]:.4f} | {johansen_result['trace_cv_95'][0]:.4f} | {'拒绝' if johansen_result['trace_stat'][0] > johansen_result['trace_cv_95'][0] else '不拒绝'} |
| r≤1 | {johansen_result['trace_stat'][1]:.4f} | {johansen_result['trace_cv_95'][1]:.4f} | {'拒绝' if johansen_result['trace_stat'][1] > johansen_result['trace_cv_95'][1] else '不拒绝'} |

**结论**: 存在 {johansen_result['n_cointegration']} 个协整关系
"""
    else:
        report += f"检验失败: {johansen_result['error']}\n"

    report += f"""
### 3.3 协整关系稳定性

使用120日滚动窗口检验协整关系稳定性：

- 对冲比率均值: {stability_result['hedge_ratio_mean']:.4f}
- 对冲比率标准差: {stability_result['hedge_ratio_std']:.4f}
- 变异系数: {stability_result['hedge_ratio_std']/abs(stability_result['hedge_ratio_mean'])*100:.2f}%
- 协整有效时间比例: {stability_result['pct_cointegrated']:.1f}%
- 稳定性判断: **{'稳定' if stability_result['is_stable'] else '不稳定'}**

### 3.4 结构变化检测

- 检测到断点数量: {break_result['n_breaks']}
- 是否存在结构变化: **{'是' if break_result['has_structural_break'] else '否'}**

"""

    if break_result['n_breaks'] > 0:
        report += "断点位置:\n"
        for bp in break_result['break_points'][:5]:
            report += f"  - 第{bp['index']}个交易日, t统计量={bp['t_stat']:.2f}\n"

    report += f"""
---

## 4. 价差建模

### 4.1 价差统计特性

价差定义: **spread = {name1}_price - {eg_result['hedge_ratio']:.4f} × {name2}_price - {eg_result['intercept']:.4f}**

| 统计量 | 值 |
|--------|-----|
| 均值 | {spread_stats['mean']:.4f} |
| 标准差 | {spread_stats['std']:.4f} |
| 偏度 | {spread_stats['skewness']:.4f} |
| 峰度 | {spread_stats['kurtosis']:.4f} |
| 最小值 | {spread_stats['min']:.4f} |
| 最大值 | {spread_stats['max']:.4f} |
| 正态性p值 | {spread_stats['normality_pvalue']:.4f} |

**正态性检验**: {'接近正态分布' if spread_stats['normality_pvalue'] > 0.05 else '显著偏离正态分布'}

### 4.2 均值回归速度

使用AR(1)模型估计均值回归速度:
```
Δspread_t = α + θ × spread_{t-1} + ε_t
```

- θ (回归速度): {mr_result['theta']:.4f}
- 标准误: {mr_result['theta_se']:.4f}
- p值: {mr_result['theta_pvalue']:.4f}
- 是否均值回归: **{'是' if mr_result['is_mean_reverting'] else '否'}**

### 4.3 半衰期

半衰期 = -ln(2) / β

"""

    if hl_result['is_valid']:
        report += f"""- 半衰期: **{hl_result['half_life_days']:.1f} 个交易日** (约{hl_result['half_life_days']/20:.1f}个月)
- β系数: {hl_result['beta']:.4f}

**解读**: 当价差偏离均值后，平均需要 {hl_result['half_life_days']:.1f} 个交易日回归一半。
"""
    else:
        report += "半衰期计算无效（价差不具有均值回归特性）\n"

    report += f"""
### 4.4 OU过程建模

Ornstein-Uhlenbeck过程:
```
dS = θ(μ - S)dt + σdW
```

"""

    if 'error' not in ou_result:
        report += f"""| 参数 | 估计值 | 含义 |
|------|--------|------|
| θ | {ou_result['theta']:.4f} | 回归速度 |
| μ | {ou_result['mu']:.4f} | 长期均值 |
| σ | {ou_result['sigma']:.4f} | 波动率 |
| 半衰期 | {ou_result['half_life_days']:.1f}天 | 回归一半所需时间 |

**模型拟合**: {'收敛' if ou_result['convergence'] else '未收敛'}
"""
    else:
        report += f"OU过程估计失败: {ou_result['error']}\n"

    report += f"""
---

## 5. 交易信号设计

### 5.1 Z-score信号

**参数设置**:
- 回看期: 20日
- 开仓阈值: ±2.0
- 平仓阈值: 0

**交易规则**:
- Z-score > 2: 做空价差（卖出股票1，买入股票2×对冲比率）
- Z-score < -2: 做多价差（买入股票1，卖出股票2×对冲比率）
- Z-score回归到0附近: 平仓

### 5.2 布林带信号

**参数设置**:
- 回看期: 20日
- 布林带宽度: 2倍标准差

**交易规则**:
- 价差突破上轨: 做空价差
- 价差突破下轨: 做多价差
- 价差回归中轨: 平仓

### 5.3 动态阈值信号

**参数设置**:
- 长期回看期: 60日
- 波动率回看期: 20日
- 基准阈值: 2.0
- 阈值范围: [1.5, 3.0]

**交易规则**:
- 根据近期波动率与长期波动率的比值动态调整开仓阈值
- 波动率高时提高阈值，减少交易频率
- 波动率低时降低阈值，增加交易机会

### 5.4 止损设计

**止损规则**:
- 止损阈值: 3倍滚动标准差
- 触发条件: 持仓亏损超过阈值时强制平仓
- 触发次数: {sl_signal['stop_loss_count']}次

---

## 6. 策略回测

### 6.1 回测设置

- 回测区间: 2020-01-01 至 2025-12-31
- 初始资金: 100万
- 单边交易成本: 0.1%（含佣金、印花税、滑点）
- 仓位管理: 满仓进出

### 6.2 策略绩效对比

| 策略 | 总收益 | 年化收益 | 夏普比率 | 最大回撤 | 交易次数 | 胜率 |
|------|--------|----------|----------|----------|----------|------|
"""

    for name, result in backtest_results.items():
        report += f"| {name} | {result['total_return']*100:.2f}% | {result['annual_return']*100:.2f}% | {result['sharpe_ratio']:.2f} | {result['max_drawdown']*100:.2f}% | {result['num_trades']} | {result['win_rate']*100:.1f}% |\n"

    report += f"""
### 6.3 交易成本敏感性

| 单边成本 | 年化收益 | 夏普比率 | 收益变化 |
|----------|----------|----------|----------|
"""

    base_return = backtest_results['Z-score']['annual_return']
    for tc in [0.0005, 0.001, 0.002, 0.003]:
        result = backtest.run_backtest(zscore_signal['signals'], transaction_cost=tc)
        change = (result['annual_return'] - base_return) / abs(base_return) * 100 if tc != 0.001 else 0
        report += f"| {tc*100:.2f}% | {result['annual_return']*100:.2f}% | {result['sharpe_ratio']:.2f} | {change:+.1f}% |\n"

    report += f"""
**结论**: 交易成本对策略收益影响显著，需要尽量降低交易频率和成本。

### 6.4 策略容量估算

基于流动性的容量估算：
- 单配对日均成交额: {min(liquidity_risk['avg_daily_amount1'], liquidity_risk['avg_daily_amount2'])/10000:.0f}万
- 单日最大交易量(5%市场份额): {daily_turnover*0.05/10000:.0f}万
- 平均持仓周期: 约10个交易日
- **单配对策略容量**: {max_capacity/10000:.0f}万
- **多配对策略容量({min(n_pairs, 10)}对)**: {total_capacity/100000000:.2f}亿

---

## 7. 风险管理

### 7.1 配对关系破裂风险

| 风险因素 | 评估 |
|----------|------|
| 协整稳定性 | {'稳定' if stability_result['is_stable'] else '不稳定'} |
| 协整有效比例 | {stability_result['pct_cointegrated']:.1f}% |
| 当前协整p值 | {recent_pvalue:.4f} |
| **综合风险评分** | **{breakdown_risk['risk_score']}/100** |
| **风险等级** | **{breakdown_risk['risk_level']}** |

**风险因素**:
"""

    for factor in breakdown_risk['risk_factors']:
        report += f"- {factor}\n"

    if not breakdown_risk['risk_factors']:
        report += "- 暂无显著风险因素\n"

    report += f"""
**应对措施**:
1. 定期（每月/每季）重新检验协整关系
2. 设置协整p值预警阈值（如0.1）
3. 对冲比率变化超过20%时重新估计
4. 使用滚动窗口动态调整对冲比率

### 7.2 流动性风险

| 指标 | {name1} | {name2} |
|------|---------|---------|
| 日均成交额 | {liquidity_risk['avg_daily_amount1']/10000:.0f}万 | {liquidity_risk['avg_daily_amount2']/10000:.0f}万 |

- 建议最大单次交易规模: {liquidity_risk['max_position_size']/10000:.0f}万
- 当前流动性风险等级: **{liquidity_risk['liquidity_risk']}**

**应对措施**:
1. 分批建仓和平仓，避免市场冲击
2. 设置单日交易量上限（不超过日均成交额的5%）
3. 避免在开盘和收盘前后15分钟交易
4. 使用限价单而非市价单

### 7.3 杠杆控制

- 建议最大杠杆: {leverage['max_leverage']:.1f}倍
- 当前杠杆: {leverage['current_leverage']:.2f}倍
- 杠杆状态: **{leverage['leverage_status']}**

**杠杆使用建议**:
1. 普通市场环境: 1.5-2倍杠杆
2. 高波动市场: 1-1.5倍杠杆
3. 极端市场: 不使用杠杆

### 7.4 其他风险

**1. 模型风险**
- 协整模型可能存在过拟合
- 历史数据不代表未来表现
- 建议使用滚动窗口、样本外测试验证

**2. 执行风险**
- 两腿交易可能无法同时成交
- 建议使用算法交易确保同步执行
- 预留足够的滑点buffer

**3. 系统性风险**
- 行业政策变化可能导致所有配对同时失效
- 建议跨行业配置，分散单一行业风险

---

## 8. 研究结论与建议

### 8.1 主要结论

1. **配对选择**: 银行板块中存在多个协整配对，{name1}与{name2}协整关系最为显著
2. **协整特性**: 该配对协整关系{'稳定' if stability_result['is_stable'] else '存在不稳定性'}，
   半衰期约{hl_result['half_life_days']:.1f if hl_result['is_valid'] else float('nan')}个交易日
3. **策略表现**: Z-score策略年化收益{backtest_results['Z-score']['annual_return']*100:.2f}%，
   夏普比率{backtest_results['Z-score']['sharpe_ratio']:.2f}
4. **风险评估**: 配对关系破裂风险**{breakdown_risk['risk_level']}**，流动性风险**{liquidity_risk['liquidity_risk']}**

### 8.2 策略优化建议

1. **配对组合**: 使用多个协整配对构建组合，分散单一配对风险
2. **动态对冲**: 采用滚动窗口估计对冲比率，适应市场变化
3. **信号优化**: 结合基本面信息调整交易信号
4. **风控增强**: 增加止损、协整失效预警等风控机制

### 8.3 实盘注意事项

1. 交易成本是盈利的关键，尽量降低佣金率
2. 注意涨跌停限制，A股可能无法及时平仓
3. 融券券源可能不足，影响做空执行
4. 建议先小规模试验，验证策略有效性

---

## 附录

### A. 可视化图表

![配对交易分析](figures/pairs_trading_analysis.png)

![协整分析](figures/cointegration_analysis.png)

### B. 数据说明

- 数据来源: Tushare
- 价格数据: 日线收盘价
- 估值数据: PE_TTM, PB
- 行业分类: 申万一级行业

### C. 参考文献

1. Engle, R. F., & Granger, C. W. (1987). Co-integration and error correction: representation, estimation, and testing. Econometrica, 55(2), 251-276.
2. Johansen, S. (1991). Estimation and hypothesis testing of cointegration vectors in Gaussian vector autoregressive models. Econometrica, 59(6), 1551-1580.
3. Gatev, E., Goetzmann, W. N., & Rouwenhorst, K. G. (2006). Pairs trading: Performance of a relative-value arbitrage rule. The Review of Financial Studies, 19(3), 797-827.
4. Vidyamurthy, G. (2004). Pairs Trading: quantitative methods and analysis. John Wiley & Sons.

---

*本报告由Claude AI自动生成，仅供研究参考，不构成投资建议。*
"""

    # 保存报告
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n报告已保存至: {REPORT_PATH}")
    print("=" * 60)

    research.conn.close()

    return {
        'coint_pairs': coint_pairs,
        'corr_pairs': corr_pairs,
        'backtest_results': backtest_results,
        'eg_result': eg_result,
        'ou_result': ou_result,
        'hl_result': hl_result
    }


if __name__ == "__main__":
    results = run_pairs_trading_study()
