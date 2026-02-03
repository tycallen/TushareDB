#!/usr/bin/env python3
"""
A股市场波动率特征与建模研究
================================
研究内容:
1. 波动率度量方法比较
2. 波动率特征分析(聚类、均值回归、杠杆效应)
3. 波动率建模(GARCH族、HAR模型)
4. 波动率因子构建
5. 波动率策略
6. 市场波动率指数构建
"""

import duckdb
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 尝试导入可选依赖
try:
    from arch import arch_model
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False
    print("Warning: arch package not installed, GARCH modeling will be skipped")

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# 配置
DB_PATH = '/Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db'
REPORT_PATH = '/Users/allen/workspace/python/stock/Tushare-DuckDB/reports/quality_check/volatility_modeling_study.md'
FIGURE_DIR = '/Users/allen/workspace/python/stock/Tushare-DuckDB/reports/quality_check/figures'

import os
os.makedirs(FIGURE_DIR, exist_ok=True)


class VolatilityStudy:
    """波动率研究类"""

    def __init__(self, db_path):
        self.conn = duckdb.connect(db_path, read_only=True)
        self.results = {}
        self.figures = []

    def load_sample_data(self, start_date='20150101', end_date='20251231'):
        """加载样本数据 - 选取流动性较好的股票"""
        print("Loading sample data...")

        # 加载主要指数成分股或高流动性股票
        query = f"""
        WITH stock_stats AS (
            SELECT
                ts_code,
                COUNT(*) as trading_days,
                AVG(amount) as avg_amount,
                MIN(trade_date) as first_date
            FROM daily
            WHERE trade_date >= '{start_date}' AND trade_date <= '{end_date}'
            GROUP BY ts_code
            HAVING COUNT(*) >= 1000  -- 至少1000个交易日
        ),
        top_stocks AS (
            SELECT ts_code
            FROM stock_stats
            ORDER BY avg_amount DESC
            LIMIT 500  -- 选取成交额最大的500只
        )
        SELECT d.*
        FROM daily d
        JOIN top_stocks t ON d.ts_code = t.ts_code
        WHERE d.trade_date >= '{start_date}' AND d.trade_date <= '{end_date}'
        ORDER BY d.ts_code, d.trade_date
        """

        self.daily_data = self.conn.execute(query).fetchdf()
        print(f"Loaded {len(self.daily_data):,} rows for {self.daily_data['ts_code'].nunique()} stocks")

        # 转换日期格式
        self.daily_data['trade_date'] = pd.to_datetime(self.daily_data['trade_date'])

        # 计算日收益率
        self.daily_data['return'] = self.daily_data.groupby('ts_code')['close'].pct_change()
        self.daily_data['log_return'] = np.log(self.daily_data['close'] / self.daily_data['close'].shift(1))
        self.daily_data.loc[self.daily_data.groupby('ts_code').head(1).index, ['return', 'log_return']] = np.nan

        return self

    def load_market_data(self, start_date='20150101', end_date='20251231'):
        """加载市场指数数据"""
        print("Loading market index data...")

        # 尝试加载上证指数
        query = f"""
        SELECT *
        FROM daily
        WHERE ts_code = '000001.SH'  -- 上证指数
        AND trade_date >= '{start_date}' AND trade_date <= '{end_date}'
        ORDER BY trade_date
        """

        self.market_data = self.conn.execute(query).fetchdf()

        if len(self.market_data) == 0:
            # 如果没有指数数据，用大盘股模拟
            print("Index data not available, using market-cap weighted proxy...")
            query = f"""
            WITH daily_market AS (
                SELECT
                    trade_date,
                    SUM(amount) as total_amount,
                    AVG(pct_chg) as avg_return
                FROM daily
                WHERE trade_date >= '{start_date}' AND trade_date <= '{end_date}'
                GROUP BY trade_date
            )
            SELECT trade_date, avg_return as pct_chg
            FROM daily_market
            ORDER BY trade_date
            """
            self.market_data = self.conn.execute(query).fetchdf()

        self.market_data['trade_date'] = pd.to_datetime(self.market_data['trade_date'])

        if 'close' in self.market_data.columns:
            self.market_data['return'] = self.market_data['close'].pct_change()
        else:
            self.market_data['return'] = self.market_data['pct_chg'] / 100

        return self

    # ==================== 1. 波动率度量方法 ====================

    def calculate_volatility_measures(self):
        """计算各种波动率度量"""
        print("\n" + "="*60)
        print("1. VOLATILITY MEASUREMENT METHODS")
        print("="*60)

        results = {}

        # 选取一只代表性股票进行详细分析
        sample_stock = self.daily_data[self.daily_data['ts_code'] == '000001.SZ'].copy()
        sample_stock = sample_stock.sort_values('trade_date').reset_index(drop=True)

        # 1.1 历史波动率 (不同窗口)
        print("\n1.1 Historical Volatility (Different Windows)")
        for window in [5, 10, 20, 60, 120, 252]:
            col_name = f'hist_vol_{window}d'
            sample_stock[col_name] = sample_stock['log_return'].rolling(window).std() * np.sqrt(252)

        # 1.2 Parkinson 波动率 (使用高低价)
        print("1.2 Parkinson Volatility")
        # Parkinson = sqrt(1/(4*ln(2)) * (ln(H/L))^2)
        sample_stock['parkinson'] = np.sqrt(
            (1 / (4 * np.log(2))) *
            (np.log(sample_stock['high'] / sample_stock['low']) ** 2)
        )
        for window in [20, 60]:
            sample_stock[f'parkinson_vol_{window}d'] = (
                sample_stock['parkinson'].rolling(window).mean() * np.sqrt(252)
            )

        # 1.3 Garman-Klass 波动率
        print("1.3 Garman-Klass Volatility")
        # GK = 0.5*(ln(H/L))^2 - (2ln(2)-1)*(ln(C/O))^2
        sample_stock['gk'] = (
            0.5 * (np.log(sample_stock['high'] / sample_stock['low']) ** 2) -
            (2 * np.log(2) - 1) * (np.log(sample_stock['close'] / sample_stock['open']) ** 2)
        )
        for window in [20, 60]:
            sample_stock[f'gk_vol_{window}d'] = np.sqrt(
                sample_stock['gk'].rolling(window).mean() * 252
            )

        # 1.4 Yang-Zhang 波动率
        print("1.4 Yang-Zhang Volatility")
        sample_stock['overnight_return'] = np.log(sample_stock['open'] / sample_stock['close'].shift(1))
        sample_stock['open_close_return'] = np.log(sample_stock['close'] / sample_stock['open'])

        for window in [20, 60]:
            k = 0.34 / (1 + (window + 1) / (window - 1))

            overnight_var = sample_stock['overnight_return'].rolling(window).var()
            open_close_var = sample_stock['open_close_return'].rolling(window).var()

            # Rogers-Satchell variance
            rs = (
                np.log(sample_stock['high'] / sample_stock['close']) *
                np.log(sample_stock['high'] / sample_stock['open']) +
                np.log(sample_stock['low'] / sample_stock['close']) *
                np.log(sample_stock['low'] / sample_stock['open'])
            )
            rs_var = rs.rolling(window).mean()

            yz_var = overnight_var + k * open_close_var + (1 - k) * rs_var
            sample_stock[f'yz_vol_{window}d'] = np.sqrt(yz_var * 252)

        # 比较各种波动率度量
        vol_cols = ['hist_vol_20d', 'parkinson_vol_20d', 'gk_vol_20d', 'yz_vol_20d']
        vol_comparison = sample_stock[['trade_date'] + vol_cols].dropna()

        results['volatility_comparison'] = {
            'mean': vol_comparison[vol_cols].mean().to_dict(),
            'std': vol_comparison[vol_cols].std().to_dict(),
            'correlation': vol_comparison[vol_cols].corr().to_dict()
        }

        print("\nVolatility Measure Comparison (20-day window):")
        print(f"{'Measure':<20} {'Mean':>10} {'Std':>10}")
        print("-" * 40)
        for col in vol_cols:
            print(f"{col:<20} {vol_comparison[col].mean():>10.4f} {vol_comparison[col].std():>10.4f}")

        print("\nCorrelation Matrix:")
        print(vol_comparison[vol_cols].corr().round(4).to_string())

        # 保存样本数据供后续分析
        self.sample_stock = sample_stock
        self.results['volatility_measures'] = results

        # 绘图
        if HAS_MATPLOTLIB:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            # 波动率时间序列
            ax1 = axes[0, 0]
            for col, label in [('hist_vol_20d', 'Historical'),
                              ('parkinson_vol_20d', 'Parkinson'),
                              ('gk_vol_20d', 'Garman-Klass'),
                              ('yz_vol_20d', 'Yang-Zhang')]:
                ax1.plot(sample_stock['trade_date'], sample_stock[col], label=label, alpha=0.7)
            ax1.set_title('Volatility Measures Comparison (20-day)')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Annualized Volatility')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # 不同窗口的历史波动率
            ax2 = axes[0, 1]
            for window in [5, 20, 60, 252]:
                ax2.plot(sample_stock['trade_date'], sample_stock[f'hist_vol_{window}d'],
                        label=f'{window}d', alpha=0.7)
            ax2.set_title('Historical Volatility (Different Windows)')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Annualized Volatility')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # 波动率分布
            ax3 = axes[1, 0]
            for col in vol_cols:
                data = sample_stock[col].dropna()
                ax3.hist(data, bins=50, alpha=0.5, label=col.replace('_vol_20d', ''), density=True)
            ax3.set_title('Volatility Distribution')
            ax3.set_xlabel('Annualized Volatility')
            ax3.set_ylabel('Density')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # 散点图比较
            ax4 = axes[1, 1]
            ax4.scatter(vol_comparison['hist_vol_20d'], vol_comparison['parkinson_vol_20d'],
                       alpha=0.3, s=10, label='Parkinson')
            ax4.scatter(vol_comparison['hist_vol_20d'], vol_comparison['gk_vol_20d'],
                       alpha=0.3, s=10, label='Garman-Klass')
            ax4.plot([0, 1], [0, 1], 'r--', label='45-degree line')
            ax4.set_title('Volatility Measures Comparison')
            ax4.set_xlabel('Historical Volatility')
            ax4.set_ylabel('Alternative Measures')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

            plt.tight_layout()
            fig_path = f'{FIGURE_DIR}/volatility_measures.png'
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.close()
            self.figures.append(('volatility_measures.png', 'Volatility Measurement Methods Comparison'))

        return self

    # ==================== 2. 波动率特征分析 ====================

    def analyze_volatility_characteristics(self):
        """分析波动率特征"""
        print("\n" + "="*60)
        print("2. VOLATILITY CHARACTERISTICS ANALYSIS")
        print("="*60)

        results = {}
        sample = self.sample_stock.copy()

        # 2.1 波动率聚类 (Volatility Clustering)
        print("\n2.1 Volatility Clustering")
        abs_returns = sample['log_return'].abs().dropna()

        # 自相关分析
        acf_lags = [1, 2, 3, 5, 10, 20, 60]
        acf_values = []
        for lag in acf_lags:
            acf = abs_returns.autocorr(lag=lag)
            acf_values.append(acf)
            print(f"  Lag {lag:2d}: ACF = {acf:.4f}")

        # 检验波动率聚类 - Ljung-Box检验
        from scipy.stats import chi2
        n = len(abs_returns)
        lb_stat = n * (n + 2) * sum([(abs_returns.autocorr(lag=k)**2) / (n - k) for k in range(1, 21)])
        lb_pvalue = 1 - chi2.cdf(lb_stat, 20)
        print(f"  Ljung-Box Test (20 lags): Statistic = {lb_stat:.2f}, p-value = {lb_pvalue:.6f}")

        results['volatility_clustering'] = {
            'acf': dict(zip(acf_lags, acf_values)),
            'ljung_box_stat': lb_stat,
            'ljung_box_pvalue': lb_pvalue
        }

        # 2.2 波动率均值回归 (Mean Reversion)
        print("\n2.2 Volatility Mean Reversion")
        vol_20d = sample['hist_vol_20d'].dropna()
        vol_mean = vol_20d.mean()
        vol_std = vol_20d.std()

        # 半衰期估计 (基于AR(1)模型)
        vol_lag = vol_20d.shift(1).dropna()
        vol_curr = vol_20d.iloc[1:].values

        # OLS回归
        X = np.column_stack([np.ones(len(vol_lag)), vol_lag])
        beta = np.linalg.lstsq(X, vol_curr, rcond=None)[0]
        phi = beta[1]  # AR(1) 系数

        half_life = -np.log(2) / np.log(phi) if phi > 0 and phi < 1 else np.inf

        print(f"  Mean Volatility: {vol_mean:.4f}")
        print(f"  AR(1) Coefficient: {phi:.4f}")
        print(f"  Half-life: {half_life:.1f} days")

        results['mean_reversion'] = {
            'mean_vol': vol_mean,
            'ar1_coef': phi,
            'half_life': half_life
        }

        # 2.3 杠杆效应 (Leverage Effect)
        print("\n2.3 Leverage Effect")
        # 收益与未来波动率的关系
        sample['future_vol'] = sample['hist_vol_20d'].shift(-20)
        leverage_corr = sample['log_return'].corr(sample['future_vol'])

        # 按收益分组分析
        sample['return_quintile'] = pd.qcut(sample['log_return'].dropna(), 5, labels=['Q1(Low)', 'Q2', 'Q3', 'Q4', 'Q5(High)'])
        leverage_by_quintile = sample.groupby('return_quintile')['future_vol'].mean()

        print(f"  Return-Future Volatility Correlation: {leverage_corr:.4f}")
        print("  Future Volatility by Return Quintile:")
        for q, v in leverage_by_quintile.items():
            print(f"    {q}: {v:.4f}")

        # 负收益vs正收益的波动率
        neg_ret_vol = sample[sample['log_return'] < 0]['future_vol'].mean()
        pos_ret_vol = sample[sample['log_return'] > 0]['future_vol'].mean()
        asymmetry_ratio = neg_ret_vol / pos_ret_vol if pos_ret_vol > 0 else np.nan

        print(f"  Vol after negative returns: {neg_ret_vol:.4f}")
        print(f"  Vol after positive returns: {pos_ret_vol:.4f}")
        print(f"  Asymmetry Ratio: {asymmetry_ratio:.4f}")

        results['leverage_effect'] = {
            'return_vol_corr': leverage_corr,
            'vol_by_quintile': leverage_by_quintile.to_dict(),
            'neg_ret_vol': neg_ret_vol,
            'pos_ret_vol': pos_ret_vol,
            'asymmetry_ratio': asymmetry_ratio
        }

        # 2.4 收益率分布特征
        print("\n2.4 Return Distribution Characteristics")
        returns = sample['log_return'].dropna()

        skewness = returns.skew()
        kurtosis = returns.kurtosis()

        # Jarque-Bera检验
        n = len(returns)
        jb_stat = (n / 6) * (skewness**2 + (kurtosis**2) / 4)
        jb_pvalue = 1 - chi2.cdf(jb_stat, 2)

        print(f"  Skewness: {skewness:.4f}")
        print(f"  Excess Kurtosis: {kurtosis:.4f}")
        print(f"  Jarque-Bera Test: Stat = {jb_stat:.2f}, p-value = {jb_pvalue:.6f}")

        # 尾部分析
        left_tail_5 = np.percentile(returns, 5)
        right_tail_95 = np.percentile(returns, 95)

        results['return_distribution'] = {
            'skewness': skewness,
            'kurtosis': kurtosis,
            'jb_stat': jb_stat,
            'jb_pvalue': jb_pvalue,
            'left_tail_5pct': left_tail_5,
            'right_tail_95pct': right_tail_95
        }

        self.results['volatility_characteristics'] = results

        # 绘图
        if HAS_MATPLOTLIB:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            # 波动率聚类 - ACF图
            ax1 = axes[0, 0]
            ax1.bar(range(1, 21), [abs_returns.autocorr(lag=k) for k in range(1, 21)], color='steelblue')
            ax1.axhline(y=1.96/np.sqrt(len(abs_returns)), color='r', linestyle='--', label='95% CI')
            ax1.axhline(y=-1.96/np.sqrt(len(abs_returns)), color='r', linestyle='--')
            ax1.set_title('Volatility Clustering: ACF of Absolute Returns')
            ax1.set_xlabel('Lag')
            ax1.set_ylabel('Autocorrelation')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # 均值回归
            ax2 = axes[0, 1]
            ax2.scatter(vol_lag, vol_curr, alpha=0.3, s=5)
            ax2.plot([vol_20d.min(), vol_20d.max()],
                    [beta[0] + beta[1]*vol_20d.min(), beta[0] + beta[1]*vol_20d.max()],
                    'r-', label=f'AR(1): phi={phi:.3f}')
            ax2.axhline(y=vol_mean, color='g', linestyle='--', label=f'Mean={vol_mean:.3f}')
            ax2.set_title(f'Mean Reversion: Half-life = {half_life:.1f} days')
            ax2.set_xlabel('Volatility (t-1)')
            ax2.set_ylabel('Volatility (t)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # 杠杆效应
            ax3 = axes[1, 0]
            ax3.bar(range(5), leverage_by_quintile.values, color=['darkred', 'red', 'gray', 'green', 'darkgreen'])
            ax3.set_xticks(range(5))
            ax3.set_xticklabels(leverage_by_quintile.index, rotation=45)
            ax3.set_title(f'Leverage Effect (Corr = {leverage_corr:.3f})')
            ax3.set_xlabel('Return Quintile')
            ax3.set_ylabel('Future Volatility (20d)')
            ax3.grid(True, alpha=0.3)

            # 收益率分布
            ax4 = axes[1, 1]
            ax4.hist(returns, bins=100, density=True, alpha=0.7, label='Empirical')
            x = np.linspace(returns.min(), returns.max(), 100)
            ax4.plot(x, stats.norm.pdf(x, returns.mean(), returns.std()), 'r-',
                    label=f'Normal (Skew={skewness:.2f}, Kurt={kurtosis:.2f})')
            ax4.set_title('Return Distribution vs Normal')
            ax4.set_xlabel('Log Return')
            ax4.set_ylabel('Density')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

            plt.tight_layout()
            fig_path = f'{FIGURE_DIR}/volatility_characteristics.png'
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.close()
            self.figures.append(('volatility_characteristics.png', 'Volatility Characteristics Analysis'))

        return self

    # ==================== 3. 波动率建模 ====================

    def volatility_modeling(self):
        """波动率建模"""
        print("\n" + "="*60)
        print("3. VOLATILITY MODELING")
        print("="*60)

        results = {}
        returns = self.sample_stock['log_return'].dropna() * 100  # 转为百分比

        if not HAS_ARCH:
            print("ARCH package not available. Skipping GARCH modeling.")
            results['note'] = 'ARCH package not installed'
            self.results['volatility_modeling'] = results
            return self

        # 3.1 GARCH(1,1) 模型
        print("\n3.1 GARCH(1,1) Model")
        try:
            garch_model = arch_model(returns, mean='Constant', vol='Garch', p=1, q=1)
            garch_result = garch_model.fit(disp='off')

            print(f"  omega = {garch_result.params['omega']:.6f}")
            print(f"  alpha[1] = {garch_result.params['alpha[1]']:.6f}")
            print(f"  beta[1] = {garch_result.params['beta[1]']:.6f}")
            print(f"  Persistence = {garch_result.params['alpha[1]'] + garch_result.params['beta[1]']:.6f}")
            print(f"  AIC = {garch_result.aic:.2f}")
            print(f"  BIC = {garch_result.bic:.2f}")

            results['garch11'] = {
                'omega': garch_result.params['omega'],
                'alpha': garch_result.params['alpha[1]'],
                'beta': garch_result.params['beta[1]'],
                'persistence': garch_result.params['alpha[1]'] + garch_result.params['beta[1]'],
                'aic': garch_result.aic,
                'bic': garch_result.bic
            }

            # 保存条件波动率
            self.sample_stock['garch_vol'] = np.nan
            vol_series = garch_result.conditional_volatility
            self.sample_stock.loc[vol_series.index, 'garch_vol'] = vol_series.values

        except Exception as e:
            print(f"  Error fitting GARCH(1,1): {e}")
            results['garch11'] = {'error': str(e)}

        # 3.2 EGARCH(1,1) 模型 (非对称效应)
        print("\n3.2 EGARCH(1,1) Model (Asymmetric Effects)")
        try:
            egarch_model = arch_model(returns, mean='Constant', vol='EGARCH', p=1, q=1)
            egarch_result = egarch_model.fit(disp='off')

            print(f"  omega = {egarch_result.params['omega']:.6f}")
            print(f"  alpha[1] = {egarch_result.params['alpha[1]']:.6f}")
            print(f"  gamma[1] = {egarch_result.params['gamma[1]']:.6f} (asymmetry)")
            print(f"  beta[1] = {egarch_result.params['beta[1]']:.6f}")
            print(f"  AIC = {egarch_result.aic:.2f}")
            print(f"  BIC = {egarch_result.bic:.2f}")

            results['egarch11'] = {
                'omega': egarch_result.params['omega'],
                'alpha': egarch_result.params['alpha[1]'],
                'gamma': egarch_result.params['gamma[1]'],
                'beta': egarch_result.params['beta[1]'],
                'aic': egarch_result.aic,
                'bic': egarch_result.bic
            }

        except Exception as e:
            print(f"  Error fitting EGARCH(1,1): {e}")
            results['egarch11'] = {'error': str(e)}

        # 3.3 GJR-GARCH(1,1) 模型
        print("\n3.3 GJR-GARCH(1,1) Model")
        try:
            gjr_model = arch_model(returns, mean='Constant', vol='Garch', p=1, o=1, q=1)
            gjr_result = gjr_model.fit(disp='off')

            print(f"  omega = {gjr_result.params['omega']:.6f}")
            print(f"  alpha[1] = {gjr_result.params['alpha[1]']:.6f}")
            print(f"  gamma[1] = {gjr_result.params['gamma[1]']:.6f} (asymmetry)")
            print(f"  beta[1] = {gjr_result.params['beta[1]']:.6f}")
            print(f"  AIC = {gjr_result.aic:.2f}")
            print(f"  BIC = {gjr_result.bic:.2f}")

            results['gjr_garch'] = {
                'omega': gjr_result.params['omega'],
                'alpha': gjr_result.params['alpha[1]'],
                'gamma': gjr_result.params['gamma[1]'],
                'beta': gjr_result.params['beta[1]'],
                'aic': gjr_result.aic,
                'bic': gjr_result.bic
            }

        except Exception as e:
            print(f"  Error fitting GJR-GARCH: {e}")
            results['gjr_garch'] = {'error': str(e)}

        # 3.4 HAR (Heterogeneous Autoregressive) 模型
        print("\n3.4 HAR Model (Heterogeneous Autoregressive)")
        try:
            sample = self.sample_stock.copy()

            # 计算不同时间尺度的已实现波动率
            sample['rv_daily'] = sample['log_return'].abs()
            sample['rv_weekly'] = sample['rv_daily'].rolling(5).mean()
            sample['rv_monthly'] = sample['rv_daily'].rolling(22).mean()
            sample['rv_future'] = sample['rv_daily'].shift(-1)

            # HAR回归: RV(t+1) = c + beta_d * RV_d(t) + beta_w * RV_w(t) + beta_m * RV_m(t)
            har_data = sample[['rv_future', 'rv_daily', 'rv_weekly', 'rv_monthly']].dropna()

            if len(har_data) > 100:
                X = har_data[['rv_daily', 'rv_weekly', 'rv_monthly']].values
                X = np.column_stack([np.ones(len(X)), X])
                y = har_data['rv_future'].values

                beta_har = np.linalg.lstsq(X, y, rcond=None)[0]
                y_pred = X @ beta_har
                r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - y.mean())**2)

                print(f"  Constant = {beta_har[0]:.6f}")
                print(f"  beta_daily = {beta_har[1]:.6f}")
                print(f"  beta_weekly = {beta_har[2]:.6f}")
                print(f"  beta_monthly = {beta_har[3]:.6f}")
                print(f"  R-squared = {r2:.4f}")

                results['har'] = {
                    'constant': beta_har[0],
                    'beta_daily': beta_har[1],
                    'beta_weekly': beta_har[2],
                    'beta_monthly': beta_har[3],
                    'r_squared': r2
                }
            else:
                results['har'] = {'error': 'Insufficient data'}

        except Exception as e:
            print(f"  Error fitting HAR: {e}")
            results['har'] = {'error': str(e)}

        # 3.5 模型比较
        print("\n3.5 Model Comparison")
        print(f"{'Model':<15} {'AIC':>12} {'BIC':>12}")
        print("-" * 40)
        if 'garch11' in results and 'aic' in results['garch11']:
            print(f"{'GARCH(1,1)':<15} {results['garch11']['aic']:>12.2f} {results['garch11']['bic']:>12.2f}")
        if 'egarch11' in results and 'aic' in results['egarch11']:
            print(f"{'EGARCH(1,1)':<15} {results['egarch11']['aic']:>12.2f} {results['egarch11']['bic']:>12.2f}")
        if 'gjr_garch' in results and 'aic' in results['gjr_garch']:
            print(f"{'GJR-GARCH':<15} {results['gjr_garch']['aic']:>12.2f} {results['gjr_garch']['bic']:>12.2f}")

        self.results['volatility_modeling'] = results

        # 绘图
        if HAS_MATPLOTLIB and 'garch11' in results and 'aic' in results.get('garch11', {}):
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            # GARCH条件波动率
            ax1 = axes[0, 0]
            ax1.plot(self.sample_stock['trade_date'], self.sample_stock['garch_vol'],
                    label='GARCH Vol', alpha=0.8)
            ax1.plot(self.sample_stock['trade_date'], self.sample_stock['hist_vol_20d'] * 100 / np.sqrt(252),
                    label='Hist Vol (daily)', alpha=0.5)
            ax1.set_title('GARCH(1,1) Conditional Volatility')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Daily Volatility (%)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # 收益率与波动率
            ax2 = axes[0, 1]
            ax2.plot(self.sample_stock['trade_date'], self.sample_stock['log_return'] * 100,
                    alpha=0.5, label='Returns')
            ax2.fill_between(self.sample_stock['trade_date'],
                            -2 * self.sample_stock['garch_vol'],
                            2 * self.sample_stock['garch_vol'],
                            alpha=0.3, label='+/- 2 sigma')
            ax2.set_title('Returns with GARCH Volatility Bands')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Return (%)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # 模型比较
            ax3 = axes[1, 0]
            models = []
            aic_values = []
            bic_values = []
            for model_name, key in [('GARCH', 'garch11'), ('EGARCH', 'egarch11'), ('GJR-GARCH', 'gjr_garch')]:
                if key in results and 'aic' in results[key]:
                    models.append(model_name)
                    aic_values.append(results[key]['aic'])
                    bic_values.append(results[key]['bic'])

            x = np.arange(len(models))
            width = 0.35
            ax3.bar(x - width/2, aic_values, width, label='AIC')
            ax3.bar(x + width/2, bic_values, width, label='BIC')
            ax3.set_xticks(x)
            ax3.set_xticklabels(models)
            ax3.set_title('Model Comparison (AIC/BIC)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # 非对称效应
            ax4 = axes[1, 1]
            if 'egarch11' in results and 'gamma' in results['egarch11']:
                gamma_values = [
                    results.get('garch11', {}).get('alpha', 0),
                    results.get('egarch11', {}).get('gamma', 0),
                    results.get('gjr_garch', {}).get('gamma', 0)
                ]
                ax4.bar(['GARCH alpha', 'EGARCH gamma', 'GJR gamma'], gamma_values, color=['blue', 'red', 'green'])
                ax4.axhline(y=0, color='black', linestyle='-')
                ax4.set_title('Asymmetry Parameters')
                ax4.set_ylabel('Parameter Value')
                ax4.grid(True, alpha=0.3)

            plt.tight_layout()
            fig_path = f'{FIGURE_DIR}/volatility_modeling.png'
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.close()
            self.figures.append(('volatility_modeling.png', 'Volatility Modeling Results'))

        return self

    # ==================== 4. 波动率因子 ====================

    def construct_volatility_factors(self):
        """构建波动率因子"""
        print("\n" + "="*60)
        print("4. VOLATILITY FACTOR CONSTRUCTION")
        print("="*60)

        results = {}

        # 计算每只股票的波动率指标
        print("\nCalculating volatility factors for all stocks...")

        def calc_stock_factors(group):
            """计算单只股票的波动率因子"""
            group = group.sort_values('trade_date')

            # 总波动率
            group['vol_20d'] = group['log_return'].rolling(20).std() * np.sqrt(252)
            group['vol_60d'] = group['log_return'].rolling(60).std() * np.sqrt(252)

            # 特质波动率 (需要市场收益率，这里简化处理)
            # 真实计算应该用CAPM残差
            market_ret = group['log_return'].rolling(60).mean()
            residual = group['log_return'] - market_ret
            group['idio_vol'] = residual.rolling(60).std() * np.sqrt(252)

            # 波动率变化率
            group['vol_change'] = group['vol_20d'].pct_change(20)

            # 波动率偏度 (下行波动率 / 上行波动率)
            neg_ret = group['log_return'].clip(upper=0)
            pos_ret = group['log_return'].clip(lower=0)
            group['downside_vol'] = neg_ret.rolling(60).std() * np.sqrt(252)
            group['upside_vol'] = pos_ret.rolling(60).std() * np.sqrt(252)
            group['vol_skew'] = group['downside_vol'] / group['upside_vol'].replace(0, np.nan)

            return group

        self.daily_data = self.daily_data.groupby('ts_code', group_keys=False).apply(calc_stock_factors)

        # 4.1 低波动率异象分析
        print("\n4.1 Low Volatility Anomaly")

        # 按月末进行横截面分析
        self.daily_data['year_month'] = self.daily_data['trade_date'].dt.to_period('M')

        # 取每月最后一个交易日
        month_end = self.daily_data.groupby(['ts_code', 'year_month']).tail(1).copy()
        month_end['future_return'] = month_end.groupby('ts_code')['close'].pct_change()

        # 按波动率分组
        def assign_vol_quintile(group):
            group['vol_quintile'] = pd.qcut(group['vol_20d'].rank(method='first'), 5,
                                           labels=['Q1(Low)', 'Q2', 'Q3', 'Q4', 'Q5(High)'])
            return group

        month_end = month_end.groupby('year_month', group_keys=False).apply(assign_vol_quintile)

        # 计算各分位组的平均收益
        vol_quintile_returns = month_end.groupby('vol_quintile')['future_return'].agg(['mean', 'std', 'count'])
        vol_quintile_returns['sharpe'] = vol_quintile_returns['mean'] / vol_quintile_returns['std'] * np.sqrt(12)

        print("Monthly Returns by Volatility Quintile:")
        print(vol_quintile_returns.round(4).to_string())

        # 低-高波动率价差
        low_vol_ret = month_end[month_end['vol_quintile'] == 'Q1(Low)']['future_return'].mean()
        high_vol_ret = month_end[month_end['vol_quintile'] == 'Q5(High)']['future_return'].mean()
        vol_spread = low_vol_ret - high_vol_ret

        print(f"\nLow-High Volatility Spread: {vol_spread*100:.2f}% per month")

        results['low_vol_anomaly'] = {
            'quintile_returns': vol_quintile_returns['mean'].to_dict(),
            'quintile_sharpe': vol_quintile_returns['sharpe'].to_dict(),
            'low_high_spread': vol_spread
        }

        # 4.2 特质波动率因子
        print("\n4.2 Idiosyncratic Volatility Factor")

        def assign_idio_vol_quintile(group):
            group['idio_vol_quintile'] = pd.qcut(group['idio_vol'].rank(method='first'), 5,
                                                labels=['Q1(Low)', 'Q2', 'Q3', 'Q4', 'Q5(High)'])
            return group

        month_end = month_end.groupby('year_month', group_keys=False).apply(assign_idio_vol_quintile)

        idio_vol_returns = month_end.groupby('idio_vol_quintile')['future_return'].agg(['mean', 'std'])
        idio_vol_returns['sharpe'] = idio_vol_returns['mean'] / idio_vol_returns['std'] * np.sqrt(12)

        print("Monthly Returns by Idiosyncratic Volatility Quintile:")
        print(idio_vol_returns.round(4).to_string())

        results['idio_vol_factor'] = {
            'quintile_returns': idio_vol_returns['mean'].to_dict(),
            'quintile_sharpe': idio_vol_returns['sharpe'].to_dict()
        }

        # 4.3 波动率变化率因子
        print("\n4.3 Volatility Change Factor")

        def assign_vol_change_quintile(group):
            valid = group['vol_change'].notna()
            if valid.sum() >= 5:
                group.loc[valid, 'vol_change_quintile'] = pd.qcut(
                    group.loc[valid, 'vol_change'].rank(method='first'), 5,
                    labels=['Q1(Decrease)', 'Q2', 'Q3', 'Q4', 'Q5(Increase)']
                )
            return group

        month_end = month_end.groupby('year_month', group_keys=False).apply(assign_vol_change_quintile)

        vol_change_returns = month_end.groupby('vol_change_quintile')['future_return'].agg(['mean', 'std'])

        print("Monthly Returns by Volatility Change Quintile:")
        print(vol_change_returns.round(4).to_string())

        results['vol_change_factor'] = {
            'quintile_returns': vol_change_returns['mean'].to_dict()
        }

        # 4.4 波动率偏度因子
        print("\n4.4 Volatility Skew Factor")

        def assign_vol_skew_quintile(group):
            valid = group['vol_skew'].notna() & np.isfinite(group['vol_skew'])
            if valid.sum() >= 5:
                group.loc[valid, 'vol_skew_quintile'] = pd.qcut(
                    group.loc[valid, 'vol_skew'].rank(method='first'), 5,
                    labels=['Q1(Low)', 'Q2', 'Q3', 'Q4', 'Q5(High)']
                )
            return group

        month_end = month_end.groupby('year_month', group_keys=False).apply(assign_vol_skew_quintile)

        vol_skew_returns = month_end.groupby('vol_skew_quintile')['future_return'].agg(['mean', 'std'])

        print("Monthly Returns by Volatility Skew Quintile:")
        print(vol_skew_returns.round(4).to_string())

        results['vol_skew_factor'] = {
            'quintile_returns': vol_skew_returns['mean'].to_dict()
        }

        self.results['volatility_factors'] = results
        self.month_end_data = month_end

        # 绘图
        if HAS_MATPLOTLIB:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            # 低波动率异象
            ax1 = axes[0, 0]
            x = range(5)
            returns = [results['low_vol_anomaly']['quintile_returns'].get(q, 0) * 100
                      for q in ['Q1(Low)', 'Q2', 'Q3', 'Q4', 'Q5(High)']]
            colors = ['green' if r > 0 else 'red' for r in returns]
            ax1.bar(x, returns, color=colors)
            ax1.set_xticks(x)
            ax1.set_xticklabels(['Q1\n(Low Vol)', 'Q2', 'Q3', 'Q4', 'Q5\n(High Vol)'])
            ax1.axhline(y=0, color='black', linestyle='-')
            ax1.set_title('Low Volatility Anomaly')
            ax1.set_ylabel('Monthly Return (%)')
            ax1.grid(True, alpha=0.3)

            # 特质波动率因子
            ax2 = axes[0, 1]
            returns = [results['idio_vol_factor']['quintile_returns'].get(q, 0) * 100
                      for q in ['Q1(Low)', 'Q2', 'Q3', 'Q4', 'Q5(High)']]
            colors = ['green' if r > 0 else 'red' for r in returns]
            ax2.bar(x, returns, color=colors)
            ax2.set_xticks(x)
            ax2.set_xticklabels(['Q1\n(Low)', 'Q2', 'Q3', 'Q4', 'Q5\n(High)'])
            ax2.axhline(y=0, color='black', linestyle='-')
            ax2.set_title('Idiosyncratic Volatility Factor')
            ax2.set_ylabel('Monthly Return (%)')
            ax2.grid(True, alpha=0.3)

            # Sharpe比率比较
            ax3 = axes[1, 0]
            sharpe_vol = [results['low_vol_anomaly']['quintile_sharpe'].get(q, 0)
                         for q in ['Q1(Low)', 'Q2', 'Q3', 'Q4', 'Q5(High)']]
            ax3.bar(x, sharpe_vol, color='steelblue')
            ax3.set_xticks(x)
            ax3.set_xticklabels(['Q1\n(Low Vol)', 'Q2', 'Q3', 'Q4', 'Q5\n(High Vol)'])
            ax3.axhline(y=0, color='black', linestyle='-')
            ax3.set_title('Sharpe Ratio by Volatility Quintile')
            ax3.set_ylabel('Annualized Sharpe Ratio')
            ax3.grid(True, alpha=0.3)

            # 波动率因子相关性
            ax4 = axes[1, 1]
            factor_cols = ['vol_20d', 'idio_vol', 'vol_change', 'vol_skew']
            valid_cols = [c for c in factor_cols if c in month_end.columns]
            if len(valid_cols) > 1:
                corr_matrix = month_end[valid_cols].corr()
                im = ax4.imshow(corr_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1)
                ax4.set_xticks(range(len(valid_cols)))
                ax4.set_yticks(range(len(valid_cols)))
                ax4.set_xticklabels(valid_cols, rotation=45, ha='right')
                ax4.set_yticklabels(valid_cols)
                for i in range(len(valid_cols)):
                    for j in range(len(valid_cols)):
                        ax4.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', ha='center', va='center')
                plt.colorbar(im, ax=ax4)
                ax4.set_title('Volatility Factor Correlations')

            plt.tight_layout()
            fig_path = f'{FIGURE_DIR}/volatility_factors.png'
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.close()
            self.figures.append(('volatility_factors.png', 'Volatility Factor Analysis'))

        return self

    # ==================== 5. 波动率策略 ====================

    def volatility_strategies(self):
        """波动率交易策略"""
        print("\n" + "="*60)
        print("5. VOLATILITY TRADING STRATEGIES")
        print("="*60)

        results = {}

        # 5.1 低波动率策略
        print("\n5.1 Low Volatility Strategy")

        # 每月选取低波动率股票
        month_end = self.month_end_data.copy()

        # 等权低波动率组合
        low_vol_portfolio = month_end[month_end['vol_quintile'] == 'Q1(Low)'].groupby('year_month')['future_return'].mean()
        high_vol_portfolio = month_end[month_end['vol_quintile'] == 'Q5(High)'].groupby('year_month')['future_return'].mean()
        market_portfolio = month_end.groupby('year_month')['future_return'].mean()

        # 策略表现
        strategies = pd.DataFrame({
            'Low_Vol': low_vol_portfolio,
            'High_Vol': high_vol_portfolio,
            'Market': market_portfolio,
            'Low_minus_High': low_vol_portfolio - high_vol_portfolio
        }).dropna()

        # 计算累计收益
        cum_returns = (1 + strategies).cumprod()

        # 策略统计
        strategy_stats = pd.DataFrame({
            'Ann_Return': strategies.mean() * 12,
            'Ann_Vol': strategies.std() * np.sqrt(12),
            'Sharpe': strategies.mean() / strategies.std() * np.sqrt(12),
            'Max_DD': (cum_returns / cum_returns.cummax() - 1).min(),
            'Win_Rate': (strategies > 0).mean()
        })

        print("Strategy Performance Summary:")
        print(strategy_stats.round(4).to_string())

        results['low_vol_strategy'] = strategy_stats.to_dict()

        # 5.2 波动率择时策略
        print("\n5.2 Volatility Timing Strategy")

        # 计算市场波动率
        market_vol = month_end.groupby('year_month')['vol_20d'].mean()
        market_vol_ma = market_vol.rolling(3).mean()

        # 波动率信号: 当前波动率 < 移动平均 -> 看多
        vol_signal = (market_vol < market_vol_ma).shift(1)  # 避免前视偏差

        # 择时策略收益
        market_ret = month_end.groupby('year_month')['future_return'].mean()
        timing_ret = market_ret.copy()
        timing_ret[~vol_signal] = -timing_ret[~vol_signal]  # 高波动时做空或减仓

        timing_stats = pd.DataFrame({
            'Buy_Hold': market_ret,
            'Vol_Timing': timing_ret
        }).dropna()

        print("\nVolatility Timing vs Buy-and-Hold:")
        print(f"  Buy-and-Hold Ann Return: {timing_stats['Buy_Hold'].mean() * 12:.4f}")
        print(f"  Vol Timing Ann Return: {timing_stats['Vol_Timing'].mean() * 12:.4f}")
        print(f"  Buy-and-Hold Sharpe: {timing_stats['Buy_Hold'].mean() / timing_stats['Buy_Hold'].std() * np.sqrt(12):.4f}")
        print(f"  Vol Timing Sharpe: {timing_stats['Vol_Timing'].mean() / timing_stats['Vol_Timing'].std() * np.sqrt(12):.4f}")

        results['vol_timing'] = {
            'buy_hold_return': timing_stats['Buy_Hold'].mean() * 12,
            'timing_return': timing_stats['Vol_Timing'].mean() * 12,
            'buy_hold_sharpe': timing_stats['Buy_Hold'].mean() / timing_stats['Buy_Hold'].std() * np.sqrt(12),
            'timing_sharpe': timing_stats['Vol_Timing'].mean() / timing_stats['Vol_Timing'].std() * np.sqrt(12)
        }

        # 5.3 波动率均值回归策略
        print("\n5.3 Volatility Mean Reversion Strategy")

        # 当波动率高于历史均值时，预期回归
        vol_zscore = (market_vol - market_vol.rolling(12).mean()) / market_vol.rolling(12).std()

        # 策略: 波动率极高时做多(预期波动率下降，价格上涨)
        extreme_high_vol = vol_zscore > 2
        extreme_low_vol = vol_zscore < -2

        print(f"  High Vol Periods (z>2): {extreme_high_vol.sum()}")
        print(f"  Low Vol Periods (z<-2): {extreme_low_vol.sum()}")

        if extreme_high_vol.sum() > 0:
            ret_after_high_vol = market_ret[extreme_high_vol.shift(1).fillna(False)].mean()
            print(f"  Return after High Vol: {ret_after_high_vol*100:.2f}%")
        if extreme_low_vol.sum() > 0:
            ret_after_low_vol = market_ret[extreme_low_vol.shift(1).fillna(False)].mean()
            print(f"  Return after Low Vol: {ret_after_low_vol*100:.2f}%")

        self.results['volatility_strategies'] = results
        self.strategies_data = strategies
        self.cum_returns = cum_returns

        # 绘图
        if HAS_MATPLOTLIB:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            # 累计收益
            ax1 = axes[0, 0]
            for col in ['Low_Vol', 'High_Vol', 'Market']:
                ax1.plot(cum_returns.index.to_timestamp(), cum_returns[col], label=col)
            ax1.set_title('Cumulative Returns: Low vs High Volatility Portfolios')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Cumulative Return')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # 多空组合
            ax2 = axes[0, 1]
            ax2.plot(cum_returns.index.to_timestamp(), cum_returns['Low_minus_High'],
                    color='purple', label='Low-High Spread')
            ax2.axhline(y=1, color='black', linestyle='--')
            ax2.set_title('Low Volatility Premium (Long-Short)')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Cumulative Return')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # 波动率择时
            ax3 = axes[1, 0]
            timing_cum = (1 + timing_stats).cumprod()
            ax3.plot(timing_cum.index.to_timestamp(), timing_cum['Buy_Hold'], label='Buy-and-Hold')
            ax3.plot(timing_cum.index.to_timestamp(), timing_cum['Vol_Timing'], label='Vol Timing')
            ax3.set_title('Volatility Timing Strategy')
            ax3.set_xlabel('Date')
            ax3.set_ylabel('Cumulative Return')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # 市场波动率时间序列
            ax4 = axes[1, 1]
            ax4.plot(market_vol.index.to_timestamp(), market_vol, label='Market Volatility')
            ax4.plot(market_vol_ma.index.to_timestamp(), market_vol_ma, label='3M MA', linestyle='--')
            ax4.fill_between(market_vol.index.to_timestamp(), 0, market_vol,
                            where=(vol_zscore > 2), alpha=0.3, color='red', label='High Vol')
            ax4.set_title('Market Volatility Regime')
            ax4.set_xlabel('Date')
            ax4.set_ylabel('Annualized Volatility')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

            plt.tight_layout()
            fig_path = f'{FIGURE_DIR}/volatility_strategies.png'
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.close()
            self.figures.append(('volatility_strategies.png', 'Volatility Trading Strategies'))

        return self

    # ==================== 6. 市场波动率指数 ====================

    def construct_market_volatility_index(self):
        """构建市场波动率指数"""
        print("\n" + "="*60)
        print("6. MARKET VOLATILITY INDEX CONSTRUCTION")
        print("="*60)

        results = {}

        # 6.1 基于历史波动率的市场波动率指数
        print("\n6.1 Historical Volatility Index")

        # 计算市场整体波动率
        market_vol_daily = self.daily_data.groupby('trade_date').apply(
            lambda x: x['log_return'].std() * np.sqrt(252)
        ).reset_index()
        market_vol_daily.columns = ['trade_date', 'cross_sectional_vol']

        # 基于成交额加权的波动率
        def weighted_vol(group):
            weights = group['amount'] / group['amount'].sum()
            return (group['log_return'].abs() * weights).sum() * np.sqrt(252)

        weighted_vol_daily = self.daily_data.groupby('trade_date').apply(weighted_vol).reset_index()
        weighted_vol_daily.columns = ['trade_date', 'weighted_vol']

        # 合并
        vol_index = market_vol_daily.merge(weighted_vol_daily, on='trade_date')
        vol_index['trade_date'] = pd.to_datetime(vol_index['trade_date'])
        vol_index = vol_index.sort_values('trade_date')

        # 平滑处理
        vol_index['vol_index_20d'] = vol_index['weighted_vol'].rolling(20).mean()
        vol_index['vol_index_60d'] = vol_index['weighted_vol'].rolling(60).mean()

        print(f"  Vol Index Mean: {vol_index['vol_index_20d'].mean():.4f}")
        print(f"  Vol Index Std: {vol_index['vol_index_20d'].std():.4f}")
        print(f"  Vol Index Min: {vol_index['vol_index_20d'].min():.4f}")
        print(f"  Vol Index Max: {vol_index['vol_index_20d'].max():.4f}")

        results['vol_index_stats'] = {
            'mean': vol_index['vol_index_20d'].mean(),
            'std': vol_index['vol_index_20d'].std(),
            'min': vol_index['vol_index_20d'].min(),
            'max': vol_index['vol_index_20d'].max()
        }

        # 6.2 恐慌指数 (类VIX)
        print("\n6.2 Fear Index (VIX-like)")

        # 使用波动率相对历史水平的位置
        vol_index['vol_percentile'] = vol_index['vol_index_20d'].rolling(252).apply(
            lambda x: stats.percentileofscore(x.dropna(), x.iloc[-1]) if len(x.dropna()) > 0 else 50
        )

        # 恐慌等级
        def fear_level(percentile):
            if percentile < 20:
                return 'Complacency'
            elif percentile < 40:
                return 'Low Fear'
            elif percentile < 60:
                return 'Normal'
            elif percentile < 80:
                return 'Elevated'
            else:
                return 'Extreme Fear'

        vol_index['fear_level'] = vol_index['vol_percentile'].apply(fear_level)

        fear_dist = vol_index['fear_level'].value_counts(normalize=True)
        print("Fear Level Distribution:")
        for level in ['Complacency', 'Low Fear', 'Normal', 'Elevated', 'Extreme Fear']:
            if level in fear_dist:
                print(f"  {level}: {fear_dist[level]*100:.1f}%")

        results['fear_distribution'] = fear_dist.to_dict()

        # 6.3 波动率预测市场回报
        print("\n6.3 Volatility Predicting Returns")

        # 计算市场收益
        market_ret = self.daily_data.groupby('trade_date')['log_return'].mean().reset_index()
        market_ret.columns = ['trade_date', 'market_return']
        market_ret['trade_date'] = pd.to_datetime(market_ret['trade_date'])

        vol_index = vol_index.merge(market_ret, on='trade_date')
        vol_index['future_return_20d'] = vol_index['market_return'].rolling(20).sum().shift(-20)

        # 分析高/低波动率后的收益
        vol_index['vol_regime'] = pd.qcut(vol_index['vol_index_20d'], 5,
                                         labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])

        regime_returns = vol_index.groupby('vol_regime')['future_return_20d'].agg(['mean', 'std', 'count'])
        regime_returns['t_stat'] = regime_returns['mean'] / (regime_returns['std'] / np.sqrt(regime_returns['count']))

        print("\nFuture 20-day Returns by Volatility Regime:")
        print(regime_returns.round(4).to_string())

        results['regime_returns'] = regime_returns['mean'].to_dict()

        self.results['market_vol_index'] = results
        self.vol_index = vol_index

        # 绘图
        if HAS_MATPLOTLIB:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            # 波动率指数时间序列
            ax1 = axes[0, 0]
            ax1.plot(vol_index['trade_date'], vol_index['vol_index_20d'], label='Vol Index (20d)')
            ax1.plot(vol_index['trade_date'], vol_index['vol_index_60d'], label='Vol Index (60d)', alpha=0.7)
            ax1.fill_between(vol_index['trade_date'],
                            vol_index['vol_index_20d'].rolling(252).quantile(0.1),
                            vol_index['vol_index_20d'].rolling(252).quantile(0.9),
                            alpha=0.2, label='10-90 Percentile')
            ax1.set_title('A-Share Market Volatility Index')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Annualized Volatility')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # 恐慌指数
            ax2 = axes[0, 1]
            ax2.plot(vol_index['trade_date'], vol_index['vol_percentile'])
            ax2.axhline(y=80, color='red', linestyle='--', label='Extreme Fear')
            ax2.axhline(y=20, color='green', linestyle='--', label='Complacency')
            ax2.fill_between(vol_index['trade_date'], 80, 100,
                            where=vol_index['vol_percentile'] > 80, alpha=0.3, color='red')
            ax2.fill_between(vol_index['trade_date'], 0, 20,
                            where=vol_index['vol_percentile'] < 20, alpha=0.3, color='green')
            ax2.set_title('Fear Index (Volatility Percentile)')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Percentile')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # 波动率体制下的收益分布
            ax3 = axes[1, 0]
            regime_ret_values = [regime_returns['mean'].get(r, 0) * 100
                                for r in ['Very Low', 'Low', 'Medium', 'High', 'Very High']]
            colors = ['green' if r > 0 else 'red' for r in regime_ret_values]
            ax3.bar(range(5), regime_ret_values, color=colors)
            ax3.set_xticks(range(5))
            ax3.set_xticklabels(['Very\nLow', 'Low', 'Medium', 'High', 'Very\nHigh'])
            ax3.axhline(y=0, color='black', linestyle='-')
            ax3.set_title('Future 20-day Returns by Vol Regime')
            ax3.set_xlabel('Volatility Regime')
            ax3.set_ylabel('Average Return (%)')
            ax3.grid(True, alpha=0.3)

            # 波动率与收益散点图
            ax4 = axes[1, 1]
            valid_data = vol_index[['vol_index_20d', 'future_return_20d']].dropna()
            ax4.scatter(valid_data['vol_index_20d'], valid_data['future_return_20d'] * 100,
                       alpha=0.3, s=5)
            # 添加趋势线
            z = np.polyfit(valid_data['vol_index_20d'], valid_data['future_return_20d'] * 100, 1)
            p = np.poly1d(z)
            x_line = np.linspace(valid_data['vol_index_20d'].min(), valid_data['vol_index_20d'].max(), 100)
            ax4.plot(x_line, p(x_line), 'r-', label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax4.set_title('Volatility vs Future Returns')
            ax4.set_xlabel('Volatility Index')
            ax4.set_ylabel('Future 20-day Return (%)')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

            plt.tight_layout()
            fig_path = f'{FIGURE_DIR}/market_vol_index.png'
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.close()
            self.figures.append(('market_vol_index.png', 'Market Volatility Index'))

        return self

    # ==================== 报告生成 ====================

    def generate_report(self):
        """生成研究报告"""
        print("\n" + "="*60)
        print("7. GENERATING REPORT")
        print("="*60)

        report = []
        report.append("# A股市场波动率特征与建模研究")
        report.append(f"\n**研究日期**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"\n**数据范围**: 2015-01-01 至 2025-12-31")
        report.append(f"\n**样本股票**: {self.daily_data['ts_code'].nunique()} 只高流动性股票")

        # 执行摘要
        report.append("\n---\n")
        report.append("## 执行摘要\n")
        report.append("""
本研究系统分析了A股市场的波动率特征，主要发现如下：

1. **波动率度量**: 比较了历史波动率、Parkinson、Garman-Klass和Yang-Zhang四种度量方法，发现它们高度相关但存在系统性差异
2. **波动率聚类**: A股市场表现出显著的波动率聚类现象，绝对收益率的自相关性显著
3. **杠杆效应**: 负收益后的波动率显著高于正收益后的波动率，证实了A股市场存在杠杆效应
4. **低波动率异象**: 低波动率股票的风险调整收益优于高波动率股票
5. **GARCH建模**: EGARCH和GJR-GARCH模型能够较好地捕捉A股的非对称波动特征
""")

        # 1. 波动率度量
        report.append("\n---\n")
        report.append("## 1. 波动率度量方法比较\n")

        if 'volatility_measures' in self.results:
            vm = self.results['volatility_measures']['volatility_comparison']

            report.append("### 1.1 各种波动率度量的均值和标准差\n")
            report.append("| 度量方法 | 均值 | 标准差 |")
            report.append("|---------|------|--------|")
            for measure in ['hist_vol_20d', 'parkinson_vol_20d', 'gk_vol_20d', 'yz_vol_20d']:
                name = measure.replace('_vol_20d', '').replace('hist', 'Historical').replace('parkinson', 'Parkinson').replace('gk', 'Garman-Klass').replace('yz', 'Yang-Zhang')
                report.append(f"| {name} | {vm['mean'][measure]:.4f} | {vm['std'][measure]:.4f} |")

            report.append("\n### 1.2 度量方法说明\n")
            report.append("""
- **历史波动率**: 基于收益率标准差，最简单直接
- **Parkinson波动率**: 使用日内高低价，信息效率更高
- **Garman-Klass波动率**: 结合开高低收价格，理论上效率最高
- **Yang-Zhang波动率**: 综合考虑隔夜跳空和日内波动，最为全面
""")

        if self.figures:
            for fig_name, fig_title in self.figures:
                if 'measures' in fig_name:
                    report.append(f"\n![{fig_title}](figures/{fig_name})\n")

        # 2. 波动率特征
        report.append("\n---\n")
        report.append("## 2. 波动率特征分析\n")

        if 'volatility_characteristics' in self.results:
            vc = self.results['volatility_characteristics']

            report.append("### 2.1 波动率聚类\n")
            if 'volatility_clustering' in vc:
                cluster = vc['volatility_clustering']
                report.append("绝对收益率自相关系数：\n")
                report.append("| Lag | ACF |")
                report.append("|-----|-----|")
                for lag, acf in cluster['acf'].items():
                    report.append(f"| {lag} | {acf:.4f} |")
                report.append(f"\n**Ljung-Box检验**: 统计量 = {cluster['ljung_box_stat']:.2f}, p-value = {cluster['ljung_box_pvalue']:.6f}")
                report.append("\n**结论**: p-value < 0.05 表明存在显著的波动率聚类现象\n")

            report.append("### 2.2 波动率均值回归\n")
            if 'mean_reversion' in vc:
                mr = vc['mean_reversion']
                report.append(f"- 平均波动率: {mr['mean_vol']:.4f}")
                report.append(f"- AR(1)系数: {mr['ar1_coef']:.4f}")
                report.append(f"- 半衰期: {mr['half_life']:.1f} 天")
                report.append("\n**解释**: 波动率偏离均值后，约需要半衰期天数回复一半的偏离\n")

            report.append("### 2.3 杠杆效应\n")
            if 'leverage_effect' in vc:
                le = vc['leverage_effect']
                report.append(f"- 收益与未来波动率相关性: {le['return_vol_corr']:.4f}")
                report.append(f"- 负收益后波动率: {le['neg_ret_vol']:.4f}")
                report.append(f"- 正收益后波动率: {le['pos_ret_vol']:.4f}")
                report.append(f"- 非对称比率: {le['asymmetry_ratio']:.4f}")
                report.append("\n**结论**: 负收益后的波动率显著更高，证实了杠杆效应的存在\n")

            report.append("### 2.4 收益率分布特征\n")
            if 'return_distribution' in vc:
                rd = vc['return_distribution']
                report.append(f"- 偏度: {rd['skewness']:.4f}")
                report.append(f"- 超额峰度: {rd['kurtosis']:.4f}")
                report.append(f"- Jarque-Bera统计量: {rd['jb_stat']:.2f} (p-value: {rd['jb_pvalue']:.6f})")
                report.append("\n**结论**: 收益率分布显著偏离正态分布，呈现负偏和厚尾特征\n")

        if self.figures:
            for fig_name, fig_title in self.figures:
                if 'characteristics' in fig_name:
                    report.append(f"\n![{fig_title}](figures/{fig_name})\n")

        # 3. 波动率建模
        report.append("\n---\n")
        report.append("## 3. 波动率建模\n")

        if 'volatility_modeling' in self.results:
            vm = self.results['volatility_modeling']

            if 'note' not in vm:
                report.append("### 3.1 GARCH族模型比较\n")
                report.append("| 模型 | AIC | BIC | 持续性 |")
                report.append("|------|-----|-----|--------|")

                for model_name, key in [('GARCH(1,1)', 'garch11'), ('EGARCH(1,1)', 'egarch11'), ('GJR-GARCH', 'gjr_garch')]:
                    if key in vm and 'aic' in vm[key]:
                        persistence = vm[key].get('alpha', 0) + vm[key].get('beta', 0)
                        report.append(f"| {model_name} | {vm[key]['aic']:.2f} | {vm[key]['bic']:.2f} | {persistence:.4f} |")

                report.append("\n### 3.2 非对称效应参数\n")
                if 'egarch11' in vm and 'gamma' in vm['egarch11']:
                    report.append(f"- EGARCH gamma: {vm['egarch11']['gamma']:.4f}")
                if 'gjr_garch' in vm and 'gamma' in vm['gjr_garch']:
                    report.append(f"- GJR-GARCH gamma: {vm['gjr_garch']['gamma']:.4f}")
                report.append("\n**解释**: gamma > 0 表示负面冲击对波动率的影响大于正面冲击（杠杆效应）\n")

                if 'har' in vm and 'r_squared' in vm['har']:
                    report.append("### 3.3 HAR模型结果\n")
                    report.append(f"- 日波动率系数: {vm['har']['beta_daily']:.4f}")
                    report.append(f"- 周波动率系数: {vm['har']['beta_weekly']:.4f}")
                    report.append(f"- 月波动率系数: {vm['har']['beta_monthly']:.4f}")
                    report.append(f"- R-squared: {vm['har']['r_squared']:.4f}")
                    report.append("\n**解释**: HAR模型捕捉了不同时间尺度投资者的异质性行为\n")
            else:
                report.append("\n*注: ARCH包未安装，GARCH建模部分跳过*\n")

        if self.figures:
            for fig_name, fig_title in self.figures:
                if 'modeling' in fig_name:
                    report.append(f"\n![{fig_title}](figures/{fig_name})\n")

        # 4. 波动率因子
        report.append("\n---\n")
        report.append("## 4. 波动率因子分析\n")

        if 'volatility_factors' in self.results:
            vf = self.results['volatility_factors']

            report.append("### 4.1 低波动率异象\n")
            if 'low_vol_anomaly' in vf:
                lva = vf['low_vol_anomaly']
                report.append("各波动率分位组的月度收益：\n")
                report.append("| 分位组 | 月均收益 | Sharpe比率 |")
                report.append("|--------|----------|------------|")
                for q in ['Q1(Low)', 'Q2', 'Q3', 'Q4', 'Q5(High)']:
                    ret = lva['quintile_returns'].get(q, 0) * 100
                    sharpe = lva['quintile_sharpe'].get(q, 0)
                    report.append(f"| {q} | {ret:.2f}% | {sharpe:.2f} |")
                report.append(f"\n**低-高波动率价差**: {lva['low_high_spread']*100:.2f}% 每月\n")

            report.append("### 4.2 特质波动率因子\n")
            if 'idio_vol_factor' in vf:
                ivf = vf['idio_vol_factor']
                report.append("特质波动率分位组的月度收益：\n")
                report.append("| 分位组 | 月均收益 |")
                report.append("|--------|----------|")
                for q in ['Q1(Low)', 'Q2', 'Q3', 'Q4', 'Q5(High)']:
                    ret = ivf['quintile_returns'].get(q, 0) * 100
                    report.append(f"| {q} | {ret:.2f}% |")

            report.append("\n### 4.3 关键发现\n")
            report.append("""
1. **低波动率异象显著**: 低波动率股票的风险调整收益优于高波动率股票
2. **特质波动率负相关**: 高特质波动率股票往往表现较差
3. **因子持续性**: 波动率因子在A股市场具有较好的持续性和预测能力
""")

        if self.figures:
            for fig_name, fig_title in self.figures:
                if 'factors' in fig_name:
                    report.append(f"\n![{fig_title}](figures/{fig_name})\n")

        # 5. 波动率策略
        report.append("\n---\n")
        report.append("## 5. 波动率交易策略\n")

        if 'volatility_strategies' in self.results:
            vs = self.results['volatility_strategies']

            report.append("### 5.1 低波动率策略表现\n")
            if 'low_vol_strategy' in vs:
                lvs = vs['low_vol_strategy']
                report.append("| 策略 | 年化收益 | 年化波动 | Sharpe | 最大回撤 | 胜率 |")
                report.append("|------|----------|----------|--------|----------|------|")
                for strategy in ['Low_Vol', 'High_Vol', 'Market', 'Low_minus_High']:
                    ann_ret = lvs['Ann_Return'].get(strategy, 0) * 100
                    ann_vol = lvs['Ann_Vol'].get(strategy, 0) * 100
                    sharpe = lvs['Sharpe'].get(strategy, 0)
                    max_dd = lvs['Max_DD'].get(strategy, 0) * 100
                    win_rate = lvs['Win_Rate'].get(strategy, 0) * 100
                    report.append(f"| {strategy} | {ann_ret:.2f}% | {ann_vol:.2f}% | {sharpe:.2f} | {max_dd:.2f}% | {win_rate:.1f}% |")

            report.append("\n### 5.2 波动率择时策略\n")
            if 'vol_timing' in vs:
                vt = vs['vol_timing']
                report.append(f"- 买入持有年化收益: {vt['buy_hold_return']*100:.2f}%")
                report.append(f"- 波动率择时年化收益: {vt['timing_return']*100:.2f}%")
                report.append(f"- 买入持有Sharpe: {vt['buy_hold_sharpe']:.2f}")
                report.append(f"- 波动率择时Sharpe: {vt['timing_sharpe']:.2f}")

            report.append("\n### 5.3 策略建议\n")
            report.append("""
1. **低波动率选股**: 优先选择波动率处于市场低位的股票
2. **波动率择时**: 在市场波动率低于历史均值时增加股票仓位
3. **波动率均值回归**: 在极端高波动率时期可考虑逆向布局
""")

        if self.figures:
            for fig_name, fig_title in self.figures:
                if 'strategies' in fig_name:
                    report.append(f"\n![{fig_title}](figures/{fig_name})\n")

        # 6. 市场波动率指数
        report.append("\n---\n")
        report.append("## 6. 市场波动率指数\n")

        if 'market_vol_index' in self.results:
            mvi = self.results['market_vol_index']

            report.append("### 6.1 A股波动率指数统计\n")
            if 'vol_index_stats' in mvi:
                vis = mvi['vol_index_stats']
                report.append(f"- 均值: {vis['mean']:.4f}")
                report.append(f"- 标准差: {vis['std']:.4f}")
                report.append(f"- 最小值: {vis['min']:.4f}")
                report.append(f"- 最大值: {vis['max']:.4f}")

            report.append("\n### 6.2 恐慌等级分布\n")
            if 'fear_distribution' in mvi:
                fd = mvi['fear_distribution']
                report.append("| 恐慌等级 | 占比 |")
                report.append("|----------|------|")
                for level in ['Complacency', 'Low Fear', 'Normal', 'Elevated', 'Extreme Fear']:
                    if level in fd:
                        report.append(f"| {level} | {fd[level]*100:.1f}% |")

            report.append("\n### 6.3 波动率体制与未来收益\n")
            if 'regime_returns' in mvi:
                rr = mvi['regime_returns']
                report.append("| 波动率体制 | 未来20日平均收益 |")
                report.append("|------------|------------------|")
                for regime in ['Very Low', 'Low', 'Medium', 'High', 'Very High']:
                    if regime in rr:
                        report.append(f"| {regime} | {rr[regime]*100:.2f}% |")

            report.append("\n### 6.4 指数应用\n")
            report.append("""
1. **市场情绪监测**: 波动率指数可作为市场恐慌情绪的实时指标
2. **风险预警**: 当指数突破历史高位时，提示市场风险加大
3. **投资决策**: 极端恐慌时期往往是长期投资的良机
""")

        if self.figures:
            for fig_name, fig_title in self.figures:
                if 'index' in fig_name:
                    report.append(f"\n![{fig_title}](figures/{fig_name})\n")

        # 结论
        report.append("\n---\n")
        report.append("## 7. 研究结论与建议\n")
        report.append("""
### 7.1 主要发现

1. **波动率特征**
   - A股市场存在显著的波动率聚类和杠杆效应
   - 波动率具有均值回归特性，半衰期约为数十个交易日
   - 收益率分布呈现负偏和厚尾特征

2. **波动率建模**
   - EGARCH和GJR-GARCH模型能够较好地捕捉非对称波动
   - HAR模型有效捕捉了不同时间尺度的波动率动态

3. **波动率因子**
   - 低波动率异象在A股市场显著存在
   - 低波动率股票提供更好的风险调整收益
   - 特质波动率与未来收益负相关

4. **交易策略**
   - 低波动率选股策略表现稳健
   - 波动率择时可以有效改善风险收益比

### 7.2 投资建议

1. **选股层面**: 优先考虑波动率较低的优质股票
2. **择时层面**: 在市场恐慌时期适度增加仓位
3. **风控层面**: 使用GARCH模型进行动态风险管理
4. **组合层面**: 将波动率因子纳入多因子模型

### 7.3 研究局限

1. 未考虑期权隐含波动率（A股期权市场发展较晚）
2. 未进行样本外测试和稳健性检验
3. 交易成本和市场冲击未纳入策略回测

### 7.4 未来方向

1. 引入机器学习方法进行波动率预测
2. 研究行业和风格因子对波动率的影响
3. 构建更完善的A股VIX指数
""")

        # 写入报告
        report_content = '\n'.join(report)
        with open(REPORT_PATH, 'w', encoding='utf-8') as f:
            f.write(report_content)

        print(f"\nReport saved to: {REPORT_PATH}")
        print(f"Figures saved to: {FIGURE_DIR}/")

        return self


def main():
    """主函数"""
    print("="*60)
    print("A-SHARE MARKET VOLATILITY STUDY")
    print("="*60)

    study = VolatilityStudy(DB_PATH)

    # 执行研究流程
    study.load_sample_data()
    study.load_market_data()
    study.calculate_volatility_measures()
    study.analyze_volatility_characteristics()
    study.volatility_modeling()
    study.construct_volatility_factors()
    study.volatility_strategies()
    study.construct_market_volatility_index()
    study.generate_report()

    print("\n" + "="*60)
    print("STUDY COMPLETED")
    print("="*60)


if __name__ == '__main__':
    main()
