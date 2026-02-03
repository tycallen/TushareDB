#!/usr/bin/env python3
"""
股票收益分布特征研究
=====================
研究A股市场收益率的统计分布特征，包括：
1. 分布统计：正态性检验、偏度/峰度、厚尾特征
2. 极端收益：涨跌停统计、极端收益概率、极端收益聚集
3. 建模应用：风险度量、VaR/CVaR计算、压力测试

Author: Claude Code Assistant
Date: 2026-02-01
"""

import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm, t, jarque_bera, shapiro, kstest, skew, kurtosis
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False

# 配置路径
DB_PATH = '/Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db'
REPORT_PATH = '/Users/allen/workspace/python/stock/Tushare-DuckDB/reports/research'


def load_return_data(start_date='20200101', end_date='20260130'):
    """加载收益率数据"""
    conn = duckdb.connect(DB_PATH, read_only=True)

    query = f"""
    SELECT
        ts_code,
        trade_date,
        pct_chg,
        close,
        pre_close,
        vol,
        amount
    FROM daily
    WHERE trade_date >= '{start_date}'
      AND trade_date <= '{end_date}'
      AND pct_chg IS NOT NULL
    ORDER BY trade_date, ts_code
    """

    df = conn.execute(query).fetchdf()
    conn.close()

    print(f"加载数据: {len(df):,} 条记录")
    print(f"日期范围: {df['trade_date'].min()} - {df['trade_date'].max()}")
    print(f"股票数量: {df['ts_code'].nunique()}")

    return df


def analyze_distribution_statistics(returns):
    """
    分布统计分析
    包括：正态性检验、偏度/峰度、厚尾特征
    """
    results = {}

    # 基本统计量
    results['count'] = len(returns)
    results['mean'] = returns.mean()
    results['std'] = returns.std()
    results['median'] = returns.median()
    results['min'] = returns.min()
    results['max'] = returns.max()

    # 偏度和峰度
    results['skewness'] = skew(returns)
    results['kurtosis'] = kurtosis(returns)  # 超额峰度
    results['kurtosis_fisher'] = kurtosis(returns, fisher=True)

    # 正态性检验
    # 1. Jarque-Bera检验
    jb_stat, jb_pvalue = jarque_bera(returns)
    results['jb_statistic'] = jb_stat
    results['jb_pvalue'] = jb_pvalue

    # 2. Shapiro-Wilk检验（样本过大时取子样本）
    sample_size = min(5000, len(returns))
    sample_returns = np.random.choice(returns, size=sample_size, replace=False)
    sw_stat, sw_pvalue = shapiro(sample_returns)
    results['sw_statistic'] = sw_stat
    results['sw_pvalue'] = sw_pvalue

    # 3. Kolmogorov-Smirnov检验
    standardized = (returns - returns.mean()) / returns.std()
    ks_stat, ks_pvalue = kstest(standardized, 'norm')
    results['ks_statistic'] = ks_stat
    results['ks_pvalue'] = ks_pvalue

    # 厚尾特征分析
    # 计算尾部概率（超过2σ、3σ、4σ的概率）
    std = returns.std()
    mean = returns.mean()

    for sigma in [2, 3, 4]:
        threshold = sigma * std
        actual_prob = ((returns > mean + threshold) | (returns < mean - threshold)).mean()
        normal_prob = 2 * (1 - norm.cdf(sigma))
        results[f'tail_prob_{sigma}sigma'] = actual_prob
        results[f'tail_ratio_{sigma}sigma'] = actual_prob / normal_prob if normal_prob > 0 else np.inf

    # 分位数比较（实际 vs 正态分布）
    quantiles = [0.01, 0.05, 0.95, 0.99]
    for q in quantiles:
        actual_q = np.percentile(returns, q * 100)
        normal_q = norm.ppf(q, loc=mean, scale=std)
        results[f'quantile_{int(q*100):02d}'] = actual_q
        results[f'normal_quantile_{int(q*100):02d}'] = normal_q

    return results


def analyze_extreme_returns(df):
    """
    极端收益分析
    包括：涨跌停统计、极端收益概率、极端收益聚集
    """
    results = {}

    # 涨跌停统计（考虑不同板块的涨跌幅限制）
    # 主板: 10%, 科创板/创业板: 20%, ST股票: 5%

    # 简化处理：统计接近涨跌停的情况
    returns = df['pct_chg']

    # 涨停统计（接近10%和20%）
    results['limit_up_10pct'] = (returns >= 9.9).sum()
    results['limit_up_10pct_ratio'] = (returns >= 9.9).mean() * 100
    results['limit_up_20pct'] = (returns >= 19.9).sum()
    results['limit_up_20pct_ratio'] = (returns >= 19.9).mean() * 100

    # 跌停统计
    results['limit_down_10pct'] = (returns <= -9.9).sum()
    results['limit_down_10pct_ratio'] = (returns <= -9.9).mean() * 100
    results['limit_down_20pct'] = (returns <= -19.9).sum()
    results['limit_down_20pct_ratio'] = (returns <= -19.9).mean() * 100

    # 极端收益概率（超过5%、8%的日收益）
    for threshold in [3, 5, 8]:
        results[f'extreme_up_{threshold}pct'] = (returns >= threshold).mean() * 100
        results[f'extreme_down_{threshold}pct'] = (returns <= -threshold).mean() * 100

    # 极端收益聚集分析（按日期统计）
    daily_stats = df.groupby('trade_date').agg({
        'pct_chg': ['mean', 'std', 'min', 'max'],
        'ts_code': 'count'
    }).reset_index()
    daily_stats.columns = ['trade_date', 'mean_return', 'std_return', 'min_return', 'max_return', 'stock_count']

    # 极端日统计（市场整体波动大的日子）
    extreme_days_up = daily_stats[daily_stats['mean_return'] > 2]
    extreme_days_down = daily_stats[daily_stats['mean_return'] < -2]

    results['extreme_days_up'] = len(extreme_days_up)
    results['extreme_days_down'] = len(extreme_days_down)
    results['total_trading_days'] = len(daily_stats)

    # 涨跌停股票数量统计
    daily_limit_up = df[df['pct_chg'] >= 9.9].groupby('trade_date').size()
    daily_limit_down = df[df['pct_chg'] <= -9.9].groupby('trade_date').size()

    results['avg_daily_limit_up'] = daily_limit_up.mean() if len(daily_limit_up) > 0 else 0
    results['max_daily_limit_up'] = daily_limit_up.max() if len(daily_limit_up) > 0 else 0
    results['avg_daily_limit_down'] = daily_limit_down.mean() if len(daily_limit_down) > 0 else 0
    results['max_daily_limit_down'] = daily_limit_down.max() if len(daily_limit_down) > 0 else 0

    # 极端收益的自相关性（聚集效应）
    daily_extreme_count = df.groupby('trade_date').apply(
        lambda x: ((x['pct_chg'] >= 5) | (x['pct_chg'] <= -5)).sum()
    )
    if len(daily_extreme_count) > 1:
        autocorr_1 = daily_extreme_count.autocorr(lag=1)
        autocorr_5 = daily_extreme_count.autocorr(lag=5)
        results['extreme_autocorr_1day'] = autocorr_1
        results['extreme_autocorr_5day'] = autocorr_5

    return results, daily_stats


def calculate_var_cvar(returns, confidence_levels=[0.95, 0.99]):
    """
    计算VaR和CVaR
    使用历史模拟法、参数法和修正参数法
    """
    results = {}

    for conf in confidence_levels:
        alpha = 1 - conf

        # 1. 历史模拟法 VaR
        var_hist = np.percentile(returns, alpha * 100)
        results[f'VaR_{int(conf*100)}_hist'] = var_hist

        # 历史模拟法 CVaR (Expected Shortfall)
        cvar_hist = returns[returns <= var_hist].mean()
        results[f'CVaR_{int(conf*100)}_hist'] = cvar_hist

        # 2. 参数法 VaR（假设正态分布）
        mean = returns.mean()
        std = returns.std()
        var_param = mean + norm.ppf(alpha) * std
        results[f'VaR_{int(conf*100)}_param'] = var_param

        # 参数法 CVaR
        cvar_param = mean - std * norm.pdf(norm.ppf(alpha)) / alpha
        results[f'CVaR_{int(conf*100)}_param'] = cvar_param

        # 3. 修正参数法（考虑偏度和峰度的Cornish-Fisher展开）
        s = skew(returns)
        k = kurtosis(returns)
        z = norm.ppf(alpha)
        z_cf = (z + (z**2 - 1) * s / 6 +
                (z**3 - 3*z) * (k - 3) / 24 -
                (2*z**3 - 5*z) * s**2 / 36)
        var_cf = mean + z_cf * std
        results[f'VaR_{int(conf*100)}_cf'] = var_cf

        # 4. t分布法 VaR（更好地拟合厚尾）
        try:
            # 拟合t分布
            df_t, loc_t, scale_t = t.fit(returns)
            var_t = t.ppf(alpha, df_t, loc=loc_t, scale=scale_t)
            results[f'VaR_{int(conf*100)}_t'] = var_t
            results[f't_dist_df'] = df_t
        except:
            results[f'VaR_{int(conf*100)}_t'] = np.nan

    return results


def stress_testing(returns, scenarios=None):
    """
    压力测试
    分析极端市场条件下的潜在损失
    """
    results = {}

    if scenarios is None:
        scenarios = {
            'worst_day': returns.min(),
            'worst_1pct': np.percentile(returns, 1),
            'worst_5pct': np.percentile(returns, 5),
            '2sigma_shock': returns.mean() - 2 * returns.std(),
            '3sigma_shock': returns.mean() - 3 * returns.std(),
            '4sigma_shock': returns.mean() - 4 * returns.std(),
        }

    # 历史极端场景
    results['scenarios'] = scenarios

    # 最大回撤分析（需要按股票计算）
    # 这里计算整体市场的平均收益序列

    # 连续下跌天数分析
    negative_returns = (returns < 0).astype(int)

    # 极端收益的历史分布
    extreme_returns = returns[returns < np.percentile(returns, 5)]
    results['extreme_return_mean'] = extreme_returns.mean()
    results['extreme_return_std'] = extreme_returns.std()
    results['extreme_return_min'] = extreme_returns.min()

    # 条件概率：大跌后继续下跌的概率
    # 需要按股票时间序列计算

    return results


def fit_distributions(returns):
    """
    拟合不同分布并比较
    """
    results = {}

    # 1. 正态分布
    mu, sigma = norm.fit(returns)
    results['normal'] = {'mu': mu, 'sigma': sigma}

    # 2. t分布
    try:
        df_t, loc_t, scale_t = t.fit(returns)
        results['t'] = {'df': df_t, 'loc': loc_t, 'scale': scale_t}
    except:
        results['t'] = None

    # 使用AIC/BIC比较拟合优度
    n = len(returns)

    # 正态分布的log-likelihood
    ll_norm = norm.logpdf(returns, mu, sigma).sum()
    aic_norm = -2 * ll_norm + 2 * 2
    bic_norm = -2 * ll_norm + 2 * np.log(n)
    results['normal']['ll'] = ll_norm
    results['normal']['aic'] = aic_norm
    results['normal']['bic'] = bic_norm

    # t分布的log-likelihood
    if results['t']:
        ll_t = t.logpdf(returns, df_t, loc_t, scale_t).sum()
        aic_t = -2 * ll_t + 2 * 3
        bic_t = -2 * ll_t + 3 * np.log(n)
        results['t']['ll'] = ll_t
        results['t']['aic'] = aic_t
        results['t']['bic'] = bic_t

    return results


def plot_return_distribution(returns, save_path):
    """
    绘制收益分布图
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. 收益分布直方图
    ax1 = axes[0, 0]
    ax1.hist(returns, bins=200, density=True, alpha=0.7, color='steelblue', label='实际分布')

    # 叠加正态分布
    x = np.linspace(returns.min(), returns.max(), 1000)
    mu, sigma = norm.fit(returns)
    ax1.plot(x, norm.pdf(x, mu, sigma), 'r-', lw=2, label=f'正态分布 (μ={mu:.2f}, σ={sigma:.2f})')

    ax1.set_xlabel('日收益率 (%)')
    ax1.set_ylabel('概率密度')
    ax1.set_title('收益率分布 vs 正态分布')
    ax1.legend()
    ax1.set_xlim(-15, 15)

    # 2. Q-Q图
    ax2 = axes[0, 1]
    stats.probplot(returns, dist="norm", plot=ax2)
    ax2.set_title('Q-Q图 (正态分布)')
    ax2.get_lines()[0].set_markerfacecolor('steelblue')
    ax2.get_lines()[0].set_markersize(3)

    # 3. 尾部分布对比（对数坐标）
    ax3 = axes[0, 2]
    sorted_returns = np.sort(returns)
    n = len(sorted_returns)
    empirical_cdf = np.arange(1, n + 1) / n

    # 左尾
    left_mask = sorted_returns < 0
    ax3.semilogy(-sorted_returns[left_mask], 1 - empirical_cdf[left_mask], 'b.',
                 alpha=0.3, markersize=1, label='实际左尾')

    # 理论正态分布
    x_theory = np.linspace(0.01, -returns.min(), 100)
    normal_tail = norm.sf(x_theory / sigma)
    ax3.semilogy(x_theory, normal_tail, 'r-', lw=2, label='正态分布')

    ax3.set_xlabel('|收益率| (%)')
    ax3.set_ylabel('尾部概率 (对数)')
    ax3.set_title('尾部分布比较')
    ax3.legend()
    ax3.set_xlim(0, 15)

    # 4. 极端收益频率
    ax4 = axes[1, 0]
    thresholds = [3, 5, 7, 10]
    actual_probs_up = [(returns >= t).mean() * 100 for t in thresholds]
    actual_probs_down = [(returns <= -t).mean() * 100 for t in thresholds]
    normal_probs = [norm.sf(t / sigma) * 100 for t in thresholds]

    x_pos = np.arange(len(thresholds))
    width = 0.25

    ax4.bar(x_pos - width, actual_probs_up, width, label='实际上涨', color='red', alpha=0.7)
    ax4.bar(x_pos, actual_probs_down, width, label='实际下跌', color='green', alpha=0.7)
    ax4.bar(x_pos + width, normal_probs, width, label='正态分布预测', color='gray', alpha=0.7)

    ax4.set_xlabel('阈值 (%)')
    ax4.set_ylabel('概率 (%)')
    ax4.set_title('极端收益概率 vs 正态分布预测')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([f'>={t}%' for t in thresholds])
    ax4.legend()
    ax4.set_yscale('log')

    # 5. 收益率时间序列波动
    ax5 = axes[1, 1]
    returns_series = returns.values
    rolling_std = pd.Series(returns_series).rolling(window=250, min_periods=50).std()
    ax5.plot(rolling_std.values, alpha=0.7, color='steelblue')
    ax5.set_xlabel('样本序号')
    ax5.set_ylabel('滚动标准差 (250期)')
    ax5.set_title('波动率聚集效应')

    # 6. VaR回测
    ax6 = axes[1, 2]
    var_95 = np.percentile(returns, 5)
    var_99 = np.percentile(returns, 1)

    ax6.hist(returns, bins=100, density=True, alpha=0.7, color='steelblue')
    ax6.axvline(var_95, color='orange', linestyle='--', lw=2, label=f'VaR 95% = {var_95:.2f}%')
    ax6.axvline(var_99, color='red', linestyle='--', lw=2, label=f'VaR 99% = {var_99:.2f}%')
    ax6.set_xlabel('日收益率 (%)')
    ax6.set_ylabel('概率密度')
    ax6.set_title('VaR风险阈值')
    ax6.legend()
    ax6.set_xlim(-15, 15)

    plt.tight_layout()
    plt.savefig(f'{save_path}/return_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"图表已保存: {save_path}/return_distribution.png")


def plot_extreme_analysis(df, daily_stats, save_path):
    """
    绘制极端收益分析图
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    returns = df['pct_chg']

    # 1. 涨跌停分布
    ax1 = axes[0, 0]
    limit_up = (returns >= 9.9).sum()
    limit_down = (returns <= -9.9).sum()
    normal_range = len(returns) - limit_up - limit_down

    sizes = [limit_up, limit_down, normal_range]
    labels = [f'涨停 ({limit_up:,})', f'跌停 ({limit_down:,})', f'正常 ({normal_range:,})']
    colors = ['red', 'green', 'steelblue']
    explode = (0.05, 0.05, 0)

    ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.2f%%',
            shadow=True, startangle=90)
    ax1.set_title('涨跌停占比统计')

    # 2. 每日极端股票数量
    ax2 = axes[0, 1]
    daily_limit_up = df[df['pct_chg'] >= 9.9].groupby('trade_date').size()
    daily_limit_down = df[df['pct_chg'] <= -9.9].groupby('trade_date').size()

    # 取最近500个交易日
    recent_dates = sorted(df['trade_date'].unique())[-500:]

    limit_up_recent = daily_limit_up.reindex(recent_dates, fill_value=0)
    limit_down_recent = daily_limit_down.reindex(recent_dates, fill_value=0)

    ax2.fill_between(range(len(limit_up_recent)), limit_up_recent.values,
                     alpha=0.7, color='red', label='涨停数')
    ax2.fill_between(range(len(limit_down_recent)), -limit_down_recent.values,
                     alpha=0.7, color='green', label='跌停数')
    ax2.set_xlabel('交易日序号 (近500日)')
    ax2.set_ylabel('涨停/跌停股票数')
    ax2.set_title('每日涨跌停股票数量')
    ax2.legend()
    ax2.axhline(y=0, color='black', linestyle='-', lw=0.5)

    # 3. 市场收益分布
    ax3 = axes[1, 0]
    market_returns = daily_stats['mean_return']
    ax3.hist(market_returns, bins=50, density=True, alpha=0.7, color='steelblue')
    ax3.axvline(x=0, color='black', linestyle='--', lw=1)
    ax3.axvline(x=market_returns.mean(), color='red', linestyle='--', lw=2,
                label=f'均值 = {market_returns.mean():.3f}%')
    ax3.set_xlabel('市场平均收益率 (%)')
    ax3.set_ylabel('概率密度')
    ax3.set_title('每日市场平均收益分布')
    ax3.legend()

    # 4. 极端收益的聚集性热力图
    ax4 = axes[1, 1]

    # 按月统计极端收益
    df_copy = df.copy()
    df_copy['year_month'] = df_copy['trade_date'].str[:6]
    df_copy['is_extreme'] = (df_copy['pct_chg'].abs() >= 5).astype(int)

    monthly_extreme = df_copy.groupby('year_month')['is_extreme'].mean() * 100

    # 重塑为年-月矩阵
    monthly_extreme_df = monthly_extreme.reset_index()
    monthly_extreme_df['year'] = monthly_extreme_df['year_month'].str[:4]
    monthly_extreme_df['month'] = monthly_extreme_df['year_month'].str[4:6].astype(int)

    pivot_table = monthly_extreme_df.pivot(index='year', columns='month', values='is_extreme')

    im = ax4.imshow(pivot_table.values[-6:], cmap='YlOrRd', aspect='auto')
    ax4.set_xticks(range(12))
    ax4.set_xticklabels(range(1, 13))
    ax4.set_yticks(range(len(pivot_table.index[-6:])))
    ax4.set_yticklabels(pivot_table.index[-6:])
    ax4.set_xlabel('月份')
    ax4.set_ylabel('年份')
    ax4.set_title('极端收益占比热力图 (>=5%)')
    plt.colorbar(im, ax=ax4, label='极端收益占比 (%)')

    plt.tight_layout()
    plt.savefig(f'{save_path}/extreme_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"图表已保存: {save_path}/extreme_analysis.png")


def plot_var_analysis(returns, var_results, save_path):
    """
    绘制VaR分析图
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. VaR方法比较
    ax1 = axes[0, 0]
    methods = ['历史模拟', '参数法', 'Cornish-Fisher', 't分布']
    var_95 = [var_results['VaR_95_hist'], var_results['VaR_95_param'],
              var_results['VaR_95_cf'], var_results.get('VaR_95_t', np.nan)]
    var_99 = [var_results['VaR_99_hist'], var_results['VaR_99_param'],
              var_results['VaR_99_cf'], var_results.get('VaR_99_t', np.nan)]

    x = np.arange(len(methods))
    width = 0.35

    ax1.bar(x - width/2, var_95, width, label='VaR 95%', color='orange', alpha=0.8)
    ax1.bar(x + width/2, var_99, width, label='VaR 99%', color='red', alpha=0.8)
    ax1.set_xlabel('计算方法')
    ax1.set_ylabel('VaR (%)')
    ax1.set_title('不同方法的VaR估计')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods)
    ax1.legend()
    ax1.axhline(y=0, color='gray', linestyle='--', lw=0.5)

    # 2. VaR vs CVaR比较
    ax2 = axes[0, 1]
    var_vals = [var_results['VaR_95_hist'], var_results['VaR_99_hist']]
    cvar_vals = [var_results['CVaR_95_hist'], var_results['CVaR_99_hist']]

    x = np.arange(2)
    width = 0.35

    ax2.bar(x - width/2, var_vals, width, label='VaR', color='orange', alpha=0.8)
    ax2.bar(x + width/2, cvar_vals, width, label='CVaR', color='darkred', alpha=0.8)
    ax2.set_xlabel('置信水平')
    ax2.set_ylabel('风险值 (%)')
    ax2.set_title('VaR vs CVaR (历史模拟法)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['95%', '99%'])
    ax2.legend()

    # 3. 收益分布与VaR阈值
    ax3 = axes[1, 0]
    ax3.hist(returns, bins=150, density=True, alpha=0.7, color='steelblue', label='收益分布')

    # 标记VaR阈值
    colors = ['yellow', 'orange', 'red', 'darkred']
    var_levels = [
        (var_results['VaR_95_hist'], 'VaR 95%'),
        (var_results['VaR_99_hist'], 'VaR 99%'),
        (var_results['CVaR_95_hist'], 'CVaR 95%'),
        (var_results['CVaR_99_hist'], 'CVaR 99%'),
    ]

    for (var_val, label), color in zip(var_levels, colors):
        ax3.axvline(var_val, color=color, linestyle='--', lw=2, label=f'{label}: {var_val:.2f}%')

    ax3.set_xlabel('日收益率 (%)')
    ax3.set_ylabel('概率密度')
    ax3.set_title('收益分布与风险阈值')
    ax3.set_xlim(-15, 5)
    ax3.legend(fontsize=8)

    # 4. VaR突破统计
    ax4 = axes[1, 1]

    var_95 = var_results['VaR_95_hist']
    var_99 = var_results['VaR_99_hist']

    # 计算VaR突破次数
    breach_95 = (returns < var_95).sum()
    breach_99 = (returns < var_99).sum()
    expected_95 = len(returns) * 0.05
    expected_99 = len(returns) * 0.01

    categories = ['VaR 95%\n突破', 'VaR 99%\n突破']
    actual = [breach_95, breach_99]
    expected = [expected_95, expected_99]

    x = np.arange(len(categories))
    width = 0.35

    ax4.bar(x - width/2, actual, width, label='实际突破', color='red', alpha=0.8)
    ax4.bar(x + width/2, expected, width, label='预期突破', color='gray', alpha=0.8)
    ax4.set_ylabel('突破次数')
    ax4.set_title('VaR回测：实际vs预期突破')
    ax4.set_xticks(x)
    ax4.set_xticklabels(categories)
    ax4.legend()

    # 添加文本说明
    for i, (a, e) in enumerate(zip(actual, expected)):
        ratio = a / e if e > 0 else np.inf
        ax4.annotate(f'比率: {ratio:.2f}', (i, max(a, e) + 100),
                    ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{save_path}/var_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"图表已保存: {save_path}/var_analysis.png")


def generate_report(dist_stats, extreme_stats, var_results, stress_results, fit_results, save_path):
    """
    生成研究报告
    """
    report = []
    report.append("=" * 80)
    report.append("                    股票收益分布特征研究报告")
    report.append("=" * 80)
    report.append(f"\n报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"数据样本量: {dist_stats['count']:,} 条日收益记录")

    # 第一部分：分布统计
    report.append("\n" + "=" * 80)
    report.append("第一部分：分布统计特征")
    report.append("=" * 80)

    report.append("\n1.1 基本统计量")
    report.append("-" * 40)
    report.append(f"  均值:     {dist_stats['mean']:.4f}%")
    report.append(f"  中位数:   {dist_stats['median']:.4f}%")
    report.append(f"  标准差:   {dist_stats['std']:.4f}%")
    report.append(f"  最小值:   {dist_stats['min']:.2f}%")
    report.append(f"  最大值:   {dist_stats['max']:.2f}%")

    report.append("\n1.2 偏度和峰度")
    report.append("-" * 40)
    report.append(f"  偏度 (Skewness):        {dist_stats['skewness']:.4f}")
    report.append(f"  超额峰度 (Kurtosis):    {dist_stats['kurtosis']:.4f}")
    report.append("\n  解读:")
    if dist_stats['skewness'] > 0.5:
        report.append("  - 分布呈现显著右偏，正极端收益更常见")
    elif dist_stats['skewness'] < -0.5:
        report.append("  - 分布呈现显著左偏，负极端收益更常见")
    else:
        report.append("  - 分布相对对称")

    if dist_stats['kurtosis'] > 3:
        report.append(f"  - 超额峰度为{dist_stats['kurtosis']:.2f}，显著高于正态分布(0)")
        report.append("  - 存在明显的厚尾特征，极端事件概率远高于正态分布预测")

    report.append("\n1.3 正态性检验")
    report.append("-" * 40)
    report.append(f"  Jarque-Bera检验:")
    report.append(f"    统计量: {dist_stats['jb_statistic']:.2f}")
    report.append(f"    p值:    {dist_stats['jb_pvalue']:.2e}")
    report.append(f"    结论:   {'拒绝正态性假设' if dist_stats['jb_pvalue'] < 0.05 else '无法拒绝正态性假设'}")

    report.append(f"\n  Shapiro-Wilk检验 (5000样本):")
    report.append(f"    统计量: {dist_stats['sw_statistic']:.4f}")
    report.append(f"    p值:    {dist_stats['sw_pvalue']:.2e}")
    report.append(f"    结论:   {'拒绝正态性假设' if dist_stats['sw_pvalue'] < 0.05 else '无法拒绝正态性假设'}")

    report.append(f"\n  Kolmogorov-Smirnov检验:")
    report.append(f"    统计量: {dist_stats['ks_statistic']:.4f}")
    report.append(f"    p值:    {dist_stats['ks_pvalue']:.2e}")
    report.append(f"    结论:   {'拒绝正态性假设' if dist_stats['ks_pvalue'] < 0.05 else '无法拒绝正态性假设'}")

    report.append("\n1.4 厚尾特征分析")
    report.append("-" * 40)
    report.append("  超过n个标准差的概率 (实际 vs 正态分布理论值):")
    for sigma in [2, 3, 4]:
        actual = dist_stats[f'tail_prob_{sigma}sigma'] * 100
        ratio = dist_stats[f'tail_ratio_{sigma}sigma']
        normal_prob = 2 * (1 - norm.cdf(sigma)) * 100
        report.append(f"    ±{sigma}σ: 实际 {actual:.4f}% vs 理论 {normal_prob:.4f}% (比率: {ratio:.1f}倍)")

    report.append("\n  分位数比较 (实际 vs 正态分布):")
    for q in [1, 5, 95, 99]:
        actual = dist_stats[f'quantile_{q:02d}']
        normal = dist_stats[f'normal_quantile_{q:02d}']
        report.append(f"    {q}%分位: 实际 {actual:.2f}% vs 正态 {normal:.2f}%")

    # 第二部分：极端收益分析
    report.append("\n" + "=" * 80)
    report.append("第二部分：极端收益分析")
    report.append("=" * 80)

    report.append("\n2.1 涨跌停统计")
    report.append("-" * 40)
    report.append(f"  涨停 (>=9.9%):  {extreme_stats['limit_up_10pct']:,} 次 ({extreme_stats['limit_up_10pct_ratio']:.4f}%)")
    report.append(f"  涨停 (>=19.9%): {extreme_stats['limit_up_20pct']:,} 次 ({extreme_stats['limit_up_20pct_ratio']:.4f}%)")
    report.append(f"  跌停 (<=-9.9%): {extreme_stats['limit_down_10pct']:,} 次 ({extreme_stats['limit_down_10pct_ratio']:.4f}%)")
    report.append(f"  跌停 (<=-19.9%):{extreme_stats['limit_down_20pct']:,} 次 ({extreme_stats['limit_down_20pct_ratio']:.4f}%)")

    report.append("\n2.2 极端收益概率")
    report.append("-" * 40)
    for threshold in [3, 5, 8]:
        up = extreme_stats[f'extreme_up_{threshold}pct']
        down = extreme_stats[f'extreme_down_{threshold}pct']
        report.append(f"  >={threshold}%: {up:.4f}%,  <=-{threshold}%: {down:.4f}%")

    report.append("\n2.3 涨跌停聚集特征")
    report.append("-" * 40)
    report.append(f"  总交易日:         {extreme_stats['total_trading_days']} 天")
    report.append(f"  极端上涨日 (>2%): {extreme_stats['extreme_days_up']} 天")
    report.append(f"  极端下跌日 (<-2%):{extreme_stats['extreme_days_down']} 天")
    report.append(f"\n  每日平均涨停数:   {extreme_stats['avg_daily_limit_up']:.1f}")
    report.append(f"  单日最多涨停数:   {extreme_stats['max_daily_limit_up']}")
    report.append(f"  每日平均跌停数:   {extreme_stats['avg_daily_limit_down']:.1f}")
    report.append(f"  单日最多跌停数:   {extreme_stats['max_daily_limit_down']}")

    if 'extreme_autocorr_1day' in extreme_stats:
        report.append(f"\n  极端收益自相关性:")
        report.append(f"    1日滞后:  {extreme_stats['extreme_autocorr_1day']:.4f}")
        report.append(f"    5日滞后:  {extreme_stats['extreme_autocorr_5day']:.4f}")
        if extreme_stats['extreme_autocorr_1day'] > 0.3:
            report.append("    结论: 极端收益存在显著聚集效应")

    # 第三部分：风险度量
    report.append("\n" + "=" * 80)
    report.append("第三部分：风险度量与建模应用")
    report.append("=" * 80)

    report.append("\n3.1 VaR计算 (在险价值)")
    report.append("-" * 40)
    report.append("  95%置信水平:")
    report.append(f"    历史模拟法:     {var_results['VaR_95_hist']:.2f}%")
    report.append(f"    参数法(正态):   {var_results['VaR_95_param']:.2f}%")
    report.append(f"    Cornish-Fisher: {var_results['VaR_95_cf']:.2f}%")
    if 'VaR_95_t' in var_results and not np.isnan(var_results['VaR_95_t']):
        report.append(f"    t分布法:        {var_results['VaR_95_t']:.2f}%")

    report.append("\n  99%置信水平:")
    report.append(f"    历史模拟法:     {var_results['VaR_99_hist']:.2f}%")
    report.append(f"    参数法(正态):   {var_results['VaR_99_param']:.2f}%")
    report.append(f"    Cornish-Fisher: {var_results['VaR_99_cf']:.2f}%")
    if 'VaR_99_t' in var_results and not np.isnan(var_results['VaR_99_t']):
        report.append(f"    t分布法:        {var_results['VaR_99_t']:.2f}%")

    report.append("\n3.2 CVaR计算 (条件在险价值/期望损失)")
    report.append("-" * 40)
    report.append(f"  CVaR 95% (历史): {var_results['CVaR_95_hist']:.2f}%")
    report.append(f"  CVaR 99% (历史): {var_results['CVaR_99_hist']:.2f}%")
    report.append(f"  CVaR 95% (参数): {var_results['CVaR_95_param']:.2f}%")
    report.append(f"  CVaR 99% (参数): {var_results['CVaR_99_param']:.2f}%")

    report.append("\n  解读:")
    report.append(f"  - 在95%置信水平下，VaR表示有5%的概率日亏损超过{abs(var_results['VaR_95_hist']):.2f}%")
    report.append(f"  - CVaR表示当损失超过VaR时，平均损失为{abs(var_results['CVaR_95_hist']):.2f}%")

    report.append("\n3.3 分布拟合比较")
    report.append("-" * 40)
    if fit_results['normal']:
        report.append(f"  正态分布:")
        report.append(f"    μ = {fit_results['normal']['mu']:.4f}, σ = {fit_results['normal']['sigma']:.4f}")
        report.append(f"    AIC = {fit_results['normal']['aic']:.2f}, BIC = {fit_results['normal']['bic']:.2f}")

    if fit_results['t']:
        report.append(f"  t分布:")
        report.append(f"    df = {fit_results['t']['df']:.2f}, loc = {fit_results['t']['loc']:.4f}, scale = {fit_results['t']['scale']:.4f}")
        report.append(f"    AIC = {fit_results['t']['aic']:.2f}, BIC = {fit_results['t']['bic']:.2f}")

        if fit_results['t']['aic'] < fit_results['normal']['aic']:
            report.append("  结论: t分布拟合优于正态分布(AIC更低)")

    report.append("\n3.4 压力测试")
    report.append("-" * 40)
    scenarios = stress_results['scenarios']
    report.append(f"  历史最大单日跌幅: {scenarios['worst_day']:.2f}%")
    report.append(f"  最差1%分位损失:   {scenarios['worst_1pct']:.2f}%")
    report.append(f"  最差5%分位损失:   {scenarios['worst_5pct']:.2f}%")
    report.append(f"  2σ冲击情景:       {scenarios['2sigma_shock']:.2f}%")
    report.append(f"  3σ冲击情景:       {scenarios['3sigma_shock']:.2f}%")
    report.append(f"  4σ冲击情景:       {scenarios['4sigma_shock']:.2f}%")

    report.append(f"\n  极端损失特征:")
    report.append(f"    极端收益均值:   {stress_results['extreme_return_mean']:.2f}%")
    report.append(f"    极端收益标准差: {stress_results['extreme_return_std']:.2f}%")
    report.append(f"    极端收益最小值: {stress_results['extreme_return_min']:.2f}%")

    # 第四部分：投资启示
    report.append("\n" + "=" * 80)
    report.append("第四部分：投资与风险管理启示")
    report.append("=" * 80)

    report.append("\n4.1 主要发现")
    report.append("-" * 40)
    report.append("  1. A股收益率分布显著偏离正态分布，呈现典型的厚尾特征")
    report.append(f"     - 超额峰度达{dist_stats['kurtosis']:.1f}，表明极端事件远比正态分布预测的更频繁")
    report.append(f"     - 3σ事件的实际发生概率是正态预测的{dist_stats['tail_ratio_3sigma']:.1f}倍")

    report.append("\n  2. 涨跌停制度影响显著")
    report.append(f"     - 每日平均有{extreme_stats['avg_daily_limit_up']:.0f}只股票涨停")
    report.append(f"     - 极端收益存在明显的聚集效应")

    report.append("\n  3. 正态分布假设会严重低估风险")
    param_var = abs(var_results['VaR_99_param'])
    hist_var = abs(var_results['VaR_99_hist'])
    report.append(f"     - 99% VaR: 参数法({param_var:.2f}%) vs 历史法({hist_var:.2f}%)")
    report.append(f"     - 参数法低估风险约{((hist_var/param_var)-1)*100:.1f}%")

    report.append("\n4.2 风险管理建议")
    report.append("-" * 40)
    report.append("  1. 使用历史模拟法或t分布法计算VaR，避免使用正态分布假设")
    report.append("  2. 关注CVaR/Expected Shortfall，更好地刻画尾部风险")
    report.append("  3. 在压力测试中使用历史极端场景，而非仅依赖标准差")
    report.append("  4. 注意极端收益的聚集效应，在市场波动时期加强风控")
    report.append("  5. 考虑使用波动率聚集模型(GARCH)进行动态风险管理")

    report.append("\n" + "=" * 80)
    report.append("                              报告结束")
    report.append("=" * 80)

    # 保存报告
    report_text = "\n".join(report)

    with open(f'{save_path}/return_distribution_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"\n报告已保存: {save_path}/return_distribution_report.txt")

    return report_text


def main():
    """主函数"""
    print("=" * 60)
    print("        股票收益分布特征研究")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1/7] 加载数据...")
    df = load_return_data(start_date='20200101', end_date='20260130')
    returns = df['pct_chg'].dropna()

    # 2. 分布统计分析
    print("\n[2/7] 进行分布统计分析...")
    dist_stats = analyze_distribution_statistics(returns)

    # 3. 极端收益分析
    print("\n[3/7] 进行极端收益分析...")
    extreme_stats, daily_stats = analyze_extreme_returns(df)

    # 4. VaR/CVaR计算
    print("\n[4/7] 计算VaR和CVaR...")
    var_results = calculate_var_cvar(returns)

    # 5. 压力测试
    print("\n[5/7] 进行压力测试...")
    stress_results = stress_testing(returns)

    # 6. 分布拟合
    print("\n[6/7] 拟合分布模型...")
    fit_results = fit_distributions(returns)

    # 7. 生成报告和图表
    print("\n[7/7] 生成报告和图表...")

    # 绘制图表
    plot_return_distribution(returns, REPORT_PATH)
    plot_extreme_analysis(df, daily_stats, REPORT_PATH)
    plot_var_analysis(returns, var_results, REPORT_PATH)

    # 生成文字报告
    report = generate_report(dist_stats, extreme_stats, var_results,
                           stress_results, fit_results, REPORT_PATH)

    print("\n" + "=" * 60)
    print("分析完成！")
    print("=" * 60)
    print(f"\n输出文件:")
    print(f"  - 报告: {REPORT_PATH}/return_distribution_report.txt")
    print(f"  - 分布图: {REPORT_PATH}/return_distribution.png")
    print(f"  - 极端分析图: {REPORT_PATH}/extreme_analysis.png")
    print(f"  - VaR分析图: {REPORT_PATH}/var_analysis.png")

    # 打印报告摘要
    print("\n" + "=" * 60)
    print("报告预览:")
    print("=" * 60)
    print(report)


if __name__ == "__main__":
    main()
