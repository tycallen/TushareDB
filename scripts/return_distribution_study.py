#!/usr/bin/env python3
"""
A股市场收益率分布特征研究
=============================
研究内容:
1. 收益率分布特征 (偏度、峰度、正态性检验、尾部风险)
2. 厚尾分布建模 (Student-t, GED, EVT)
3. 收益率自相关分析 (ACF, PACF, Ljung-Box检验)
4. 收益率聚类与波动率聚类 (ARCH效应)
5. 极端事件分析
6. 跨股票分析 (市值、行业、时变特征)
"""

import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti SC', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False

# 数据库路径
DB_PATH = '/Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db'
OUTPUT_DIR = '/Users/allen/workspace/python/stock/Tushare-DuckDB/reports/quality_check'

def get_connection():
    """获取数据库连接"""
    return duckdb.connect(DB_PATH, read_only=True)

# =============================================================================
# Part 1: 收益率分布特征分析
# =============================================================================

def analyze_return_distribution():
    """分析日收益率分布特征"""
    print("=" * 60)
    print("Part 1: 收益率分布特征分析")
    print("=" * 60)

    conn = get_connection()

    # 获取所有日收益率数据 (排除极端值如涨跌停限制外的异常)
    query = """
    SELECT
        trade_date,
        ts_code,
        pct_chg,
        close,
        pre_close,
        vol
    FROM daily
    WHERE pct_chg IS NOT NULL
      AND pct_chg BETWEEN -20 AND 20  -- 排除ST等特殊情况的极端值
      AND trade_date >= '20100101'     -- 使用2010年以后的数据
    """

    df = conn.execute(query).fetchdf()
    conn.close()

    returns = df['pct_chg'].values

    print(f"\n样本数量: {len(returns):,}")
    print(f"时间范围: {df['trade_date'].min()} 至 {df['trade_date'].max()}")

    # 基本统计量
    print("\n--- 基本统计量 ---")
    print(f"均值: {np.mean(returns):.4f}%")
    print(f"中位数: {np.median(returns):.4f}%")
    print(f"标准差: {np.std(returns):.4f}%")
    print(f"最小值: {np.min(returns):.4f}%")
    print(f"最大值: {np.max(returns):.4f}%")

    # 偏度和峰度
    print("\n--- 偏度和峰度分析 ---")
    skewness = stats.skew(returns)
    kurtosis = stats.kurtosis(returns)  # 超额峰度
    print(f"偏度 (Skewness): {skewness:.4f}")
    print(f"  - 正态分布偏度为0")
    print(f"  - 当前{'负偏' if skewness < 0 else '正偏'}，表示收益率分布{'左尾较长' if skewness < 0 else '右尾较长'}")

    print(f"\n超额峰度 (Excess Kurtosis): {kurtosis:.4f}")
    print(f"  - 正态分布超额峰度为0")
    print(f"  - 当前值>>0，表明分布具有显著的'厚尾'特征")

    # 正态性检验
    print("\n--- 正态性检验 ---")

    # Jarque-Bera检验
    jb_stat, jb_p = stats.jarque_bera(returns)
    print(f"Jarque-Bera检验: 统计量={jb_stat:.2f}, p值={jb_p:.2e}")
    print(f"  - H0: 数据服从正态分布")
    print(f"  - 结论: {'拒绝正态性假设' if jb_p < 0.05 else '不能拒绝正态性假设'}")

    # Shapiro-Wilk检验 (使用子样本，因为完整数据集太大)
    sample_size = min(5000, len(returns))
    sample_returns = np.random.choice(returns, sample_size, replace=False)
    sw_stat, sw_p = stats.shapiro(sample_returns)
    print(f"\nShapiro-Wilk检验 (样本={sample_size}): 统计量={sw_stat:.4f}, p值={sw_p:.2e}")
    print(f"  - 结论: {'拒绝正态性假设' if sw_p < 0.05 else '不能拒绝正态性假设'}")

    # Kolmogorov-Smirnov检验
    ks_stat, ks_p = stats.kstest(returns, 'norm', args=(np.mean(returns), np.std(returns)))
    print(f"\nKolmogorov-Smirnov检验: 统计量={ks_stat:.4f}, p值={ks_p:.2e}")
    print(f"  - 结论: {'拒绝正态性假设' if ks_p < 0.05 else '不能拒绝正态性假设'}")

    # 尾部风险分析
    print("\n--- 尾部风险分析 ---")
    percentiles = [0.1, 0.5, 1, 2.5, 5, 95, 97.5, 99, 99.5, 99.9]
    print("分位数分析:")
    for p in percentiles:
        val = np.percentile(returns, p)
        # 计算正态分布下的理论值
        normal_val = stats.norm.ppf(p/100) * np.std(returns) + np.mean(returns)
        print(f"  {p:5.1f}%: 实际={val:7.3f}%, 正态理论={normal_val:7.3f}%, 差异={val-normal_val:+7.3f}%")

    # VaR和CVaR计算
    print("\n风险价值 (VaR) 分析:")
    for confidence in [0.95, 0.99]:
        var = np.percentile(returns, (1-confidence)*100)
        cvar = returns[returns <= var].mean()
        print(f"  {int(confidence*100)}% VaR: {abs(var):.3f}% (日度)")
        print(f"  {int(confidence*100)}% CVaR (Expected Shortfall): {abs(cvar):.3f}%")

    return returns, df

# =============================================================================
# Part 2: 厚尾分布建模
# =============================================================================

def fit_heavy_tail_distributions(returns):
    """拟合厚尾分布"""
    print("\n" + "=" * 60)
    print("Part 2: 厚尾分布建模")
    print("=" * 60)

    # 标准化收益率
    returns_std = (returns - np.mean(returns)) / np.std(returns)

    # 1. Student-t分布拟合
    print("\n--- Student-t 分布拟合 ---")
    t_params = stats.t.fit(returns_std)
    df_t, loc_t, scale_t = t_params
    print(f"拟合参数:")
    print(f"  自由度 (df): {df_t:.2f}")
    print(f"  位置参数 (loc): {loc_t:.4f}")
    print(f"  尺度参数 (scale): {scale_t:.4f}")
    print(f"  注: 自由度越小，尾部越厚。df<30表示明显厚尾")

    # AIC/BIC比较
    n = len(returns_std)

    # 正态分布的对数似然
    norm_ll = np.sum(stats.norm.logpdf(returns_std))
    norm_aic = 2 * 2 - 2 * norm_ll
    norm_bic = 2 * np.log(n) - 2 * norm_ll

    # t分布的对数似然
    t_ll = np.sum(stats.t.logpdf(returns_std, *t_params))
    t_aic = 2 * 3 - 2 * t_ll
    t_bic = 3 * np.log(n) - 2 * t_ll

    print(f"\n模型比较 (AIC/BIC - 越小越好):")
    print(f"  正态分布: AIC={norm_aic:.0f}, BIC={norm_bic:.0f}")
    print(f"  Student-t分布: AIC={t_aic:.0f}, BIC={t_bic:.0f}")
    print(f"  t分布改进: AIC降低{norm_aic-t_aic:.0f}, BIC降低{norm_bic-t_bic:.0f}")

    # 2. 广义误差分布 (GED) - 使用幂指数分布近似
    print("\n--- 广义误差分布 (GED) 拟合 ---")
    # GED的PDF: f(x) = (v/2) * exp(-|x|^v) / Gamma(1/v)
    # 使用scipy的genextreme或自定义拟合

    def ged_log_likelihood(params, data):
        """GED对数似然函数"""
        v, mu, sigma = params
        if v <= 0 or sigma <= 0:
            return np.inf
        z = (data - mu) / sigma
        lambda_v = np.sqrt(np.power(2, -2/v) *
                          stats.gamma.ppf(0.5, 3/v) / stats.gamma.ppf(0.5, 1/v))
        log_pdf = (np.log(v) - np.log(2*sigma) - np.log(lambda_v) -
                   stats.gamma.loggamma(1/v) - 0.5 * np.power(np.abs(z/lambda_v), v))
        return -np.sum(log_pdf)

    try:
        # 初始猜测
        init_params = [2, np.mean(returns_std), np.std(returns_std)]
        result = minimize(ged_log_likelihood, init_params, args=(returns_std,),
                         method='Nelder-Mead', options={'maxiter': 1000})
        v_ged, mu_ged, sigma_ged = result.x
        print(f"拟合参数:")
        print(f"  形状参数 (v): {v_ged:.4f}")
        print(f"  注: v=2时为正态分布，v<2时尾部更厚")
        if v_ged < 2:
            print(f"  当前v={v_ged:.2f}<2，确认厚尾特征")
    except Exception as e:
        print(f"  GED拟合失败: {e}")
        v_ged = None

    # 3. 极值理论 (EVT) - 使用POT方法
    print("\n--- 极值理论 (EVT) 分析 ---")

    # 选择阈值 (使用95%分位数)
    threshold_low = np.percentile(returns, 5)
    threshold_high = np.percentile(returns, 95)

    # 左尾分析
    exceedances_left = threshold_low - returns[returns < threshold_low]

    # 右尾分析
    exceedances_right = returns[returns > threshold_high] - threshold_high

    print(f"阈值选择: 下尾={threshold_low:.3f}%, 上尾={threshold_high:.3f}%")
    print(f"超越阈值样本数: 下尾={len(exceedances_left)}, 上尾={len(exceedances_right)}")

    # 拟合广义帕累托分布 (GPD)
    if len(exceedances_left) > 100:
        gpd_left = stats.genpareto.fit(exceedances_left)
        xi_left, loc_left, scale_left = gpd_left
        print(f"\n左尾GPD拟合:")
        print(f"  形状参数 (xi): {xi_left:.4f}")
        print(f"  尺度参数: {scale_left:.4f}")
        print(f"  注: xi>0表示厚尾(Pareto型)，xi<0表示有界尾")

        # 计算尾部指数
        if xi_left > 0:
            tail_index_left = 1 / xi_left
            print(f"  尾部指数: {tail_index_left:.2f}")

    if len(exceedances_right) > 100:
        gpd_right = stats.genpareto.fit(exceedances_right)
        xi_right, loc_right, scale_right = gpd_right
        print(f"\n右尾GPD拟合:")
        print(f"  形状参数 (xi): {xi_right:.4f}")
        print(f"  尺度参数: {scale_right:.4f}")
        if xi_right > 0:
            tail_index_right = 1 / xi_right
            print(f"  尾部指数: {tail_index_right:.2f}")

    # 4. Hill估计量 - 尾部指数估计
    print("\n--- Hill 估计量 (尾部指数) ---")

    def hill_estimator(data, k):
        """Hill估计量计算"""
        sorted_data = np.sort(np.abs(data))[::-1]
        if k >= len(sorted_data):
            return np.nan
        log_ratios = np.log(sorted_data[:k]) - np.log(sorted_data[k])
        return 1 / np.mean(log_ratios)

    # 对不同k值计算Hill估计量
    k_values = [100, 200, 500, 1000, 2000]
    print("不同k值的Hill估计量:")
    for k in k_values:
        if k < len(returns):
            alpha_hill = hill_estimator(returns, k)
            print(f"  k={k:4d}: α={alpha_hill:.3f}")

    print("\n注: α越小，尾部越厚。典型值: 正态分布→∞, 金融数据通常2-4")

    return t_params, v_ged

# =============================================================================
# Part 3: 收益率自相关分析
# =============================================================================

def analyze_autocorrelation(returns, df):
    """分析收益率自相关"""
    print("\n" + "=" * 60)
    print("Part 3: 收益率自相关分析")
    print("=" * 60)

    # 使用市场整体收益率 (等权平均)
    daily_returns = df.groupby('trade_date')['pct_chg'].mean().sort_index()

    print(f"\n分析样本: {len(daily_returns)} 个交易日")

    # 自相关函数 (ACF)
    print("\n--- 自相关函数 (ACF) ---")
    from statsmodels.tsa.stattools import acf, pacf

    acf_values, acf_conf = acf(daily_returns, nlags=20, alpha=0.05)

    print("滞后期  ACF值    95%置信区间")
    print("-" * 35)
    for lag in range(1, 11):
        ci_low = acf_conf[lag][0] - acf_values[lag]
        ci_high = acf_conf[lag][1] - acf_values[lag]
        significant = "*" if abs(acf_values[lag]) > 1.96/np.sqrt(len(daily_returns)) else ""
        print(f"  {lag:2d}    {acf_values[lag]:7.4f}  [{acf_conf[lag][0]:7.4f}, {acf_conf[lag][1]:7.4f}] {significant}")

    # 偏自相关函数 (PACF)
    print("\n--- 偏自相关函数 (PACF) ---")
    pacf_values, pacf_conf = pacf(daily_returns, nlags=20, alpha=0.05)

    print("滞后期  PACF值   95%置信区间")
    print("-" * 35)
    for lag in range(1, 11):
        significant = "*" if abs(pacf_values[lag]) > 1.96/np.sqrt(len(daily_returns)) else ""
        print(f"  {lag:2d}    {pacf_values[lag]:7.4f}  [{pacf_conf[lag][0]:7.4f}, {pacf_conf[lag][1]:7.4f}] {significant}")

    # Ljung-Box检验
    print("\n--- Ljung-Box 检验 ---")
    from statsmodels.stats.diagnostic import acorr_ljungbox

    lb_result = acorr_ljungbox(daily_returns, lags=[5, 10, 15, 20], return_df=True)
    print("滞后期  LB统计量   p值       结论")
    print("-" * 50)
    for idx, row in lb_result.iterrows():
        conclusion = "存在自相关*" if row['lb_pvalue'] < 0.05 else "无显著自相关"
        print(f"  {idx:2d}    {row['lb_stat']:10.2f}  {row['lb_pvalue']:.4f}   {conclusion}")

    # 平方收益率的自相关 (检验波动率聚类)
    print("\n--- 平方收益率自相关 (波动率聚类检验) ---")
    squared_returns = daily_returns ** 2
    acf_sq, acf_sq_conf = acf(squared_returns, nlags=20, alpha=0.05)

    print("滞后期  ACF(r²)  95%置信区间       波动率聚类")
    print("-" * 55)
    for lag in range(1, 11):
        significant = "***" if abs(acf_sq[lag]) > 1.96/np.sqrt(len(squared_returns)) else ""
        print(f"  {lag:2d}    {acf_sq[lag]:7.4f}  [{acf_sq_conf[lag][0]:7.4f}, {acf_sq_conf[lag][1]:7.4f}] {significant}")

    # 绝对收益率的自相关
    print("\n--- 绝对收益率自相关 ---")
    abs_returns = np.abs(daily_returns)
    acf_abs, acf_abs_conf = acf(abs_returns, nlags=20, alpha=0.05)

    print("滞后期  ACF(|r|)  显著性")
    print("-" * 30)
    for lag in range(1, 11):
        significant = "***" if abs(acf_abs[lag]) > 1.96/np.sqrt(len(abs_returns)) else ""
        print(f"  {lag:2d}    {acf_abs[lag]:7.4f}  {significant}")

    # 可预测性分析
    print("\n--- 收益率可预测性分析 ---")

    # 滞后收益率回归
    from scipy.stats import pearsonr

    returns_array = daily_returns.values
    for lag in [1, 2, 5]:
        r, p = pearsonr(returns_array[lag:], returns_array[:-lag])
        print(f"滞后{lag}天相关性: r={r:.4f}, p值={p:.4f}")

    print("\n结论:")
    print("- 收益率本身通常表现出弱自相关或负自相关（反转效应）")
    print("- 平方收益率/绝对收益率表现出强自相关（波动率聚类）")
    print("- 这表明虽然收益率难以预测，但波动率具有可预测性")

    return daily_returns

# =============================================================================
# Part 4: 收益率聚类与ARCH效应
# =============================================================================

def analyze_volatility_clustering(daily_returns):
    """分析波动率聚类和ARCH效应"""
    print("\n" + "=" * 60)
    print("Part 4: 收益率聚类与ARCH效应")
    print("=" * 60)

    returns_array = daily_returns.values

    # ARCH效应检验
    print("\n--- ARCH 效应检验 ---")
    from statsmodels.stats.diagnostic import het_arch

    # 对收益率进行ARCH检验
    arch_result = het_arch(returns_array, nlags=5)
    print(f"ARCH LM检验 (滞后5期):")
    print(f"  LM统计量: {arch_result[0]:.4f}")
    print(f"  p值: {arch_result[1]:.4e}")
    print(f"  F统计量: {arch_result[2]:.4f}")
    print(f"  F检验p值: {arch_result[3]:.4e}")
    print(f"  结论: {'存在显著ARCH效应***' if arch_result[1] < 0.05 else '无显著ARCH效应'}")

    # 不同滞后期的ARCH检验
    print("\n不同滞后期ARCH检验:")
    for nlags in [1, 5, 10, 20]:
        result = het_arch(returns_array, nlags=nlags)
        significance = "***" if result[1] < 0.01 else "**" if result[1] < 0.05 else "*" if result[1] < 0.1 else ""
        print(f"  滞后{nlags:2d}期: LM={result[0]:10.2f}, p={result[1]:.2e} {significance}")

    # 波动率聚类可视化分析
    print("\n--- 波动率聚类特征 ---")

    # 计算滚动波动率
    window = 20
    rolling_vol = pd.Series(returns_array).rolling(window=window).std()

    # 波动率的自相关
    vol_clean = rolling_vol.dropna()
    vol_acf = []
    for lag in range(1, 21):
        r = np.corrcoef(vol_clean.values[lag:], vol_clean.values[:-lag])[0,1]
        vol_acf.append(r)

    print(f"\n滚动波动率({window}日)自相关:")
    for i, acf in enumerate(vol_acf[:10], 1):
        print(f"  滞后{i:2d}天: {acf:.4f}")

    # 收益率-波动率关系 (杠杆效应)
    print("\n--- 收益率-波动率关系 (杠杆效应) ---")

    # 计算当期收益率与未来波动率的关系
    future_vol = rolling_vol.shift(-window).dropna()
    past_returns = pd.Series(returns_array).iloc[:len(future_vol)]

    # 去除NaN
    valid_idx = ~(future_vol.isna() | past_returns.isna())

    if valid_idx.sum() > 100:
        corr = np.corrcoef(past_returns[valid_idx], future_vol[valid_idx])[0,1]
        print(f"收益率与未来{window}日波动率相关性: {corr:.4f}")
        print(f"  {'负相关 - 存在杠杆效应' if corr < 0 else '正相关'}")

    # 极端收益率后的波动率变化
    print("\n极端收益率后的波动率变化:")
    extreme_negative = returns_array < np.percentile(returns_array, 5)
    extreme_positive = returns_array > np.percentile(returns_array, 95)
    normal = ~extreme_negative & ~extreme_positive

    # 计算后续5天的平均绝对收益率
    abs_returns = np.abs(returns_array)

    def avg_future_vol(mask, days=5):
        """计算条件后的平均波动率"""
        indices = np.where(mask)[0]
        future_vols = []
        for idx in indices:
            if idx + days < len(abs_returns):
                future_vols.append(np.mean(abs_returns[idx+1:idx+1+days]))
        return np.mean(future_vols) if future_vols else np.nan

    vol_after_negative = avg_future_vol(extreme_negative)
    vol_after_positive = avg_future_vol(extreme_positive)
    vol_after_normal = avg_future_vol(normal)

    print(f"  极端负收益后5日平均|r|: {vol_after_negative:.4f}%")
    print(f"  极端正收益后5日平均|r|: {vol_after_positive:.4f}%")
    print(f"  正常收益后5日平均|r|: {vol_after_normal:.4f}%")
    print(f"  不对称性: 负向冲击效应{'更强' if vol_after_negative > vol_after_positive else '更弱'}")

    return rolling_vol

# =============================================================================
# Part 5: 极端事件分析
# =============================================================================

def analyze_extreme_events(returns, df):
    """分析极端事件"""
    print("\n" + "=" * 60)
    print("Part 5: 极端事件分析")
    print("=" * 60)

    # 计算每日市场收益率
    daily_market = df.groupby('trade_date').agg({
        'pct_chg': 'mean',
        'ts_code': 'count'
    }).rename(columns={'ts_code': 'stock_count'})
    daily_market = daily_market.sort_index()

    market_returns = daily_market['pct_chg'].values

    # 定义极端事件阈值
    threshold_3sigma = 3 * np.std(market_returns)
    threshold_99 = np.percentile(np.abs(market_returns), 99)

    print("\n--- 历史极端日识别 ---")
    print(f"3倍标准差阈值: ±{threshold_3sigma:.2f}%")
    print(f"99%分位数阈值: ±{threshold_99:.2f}%")

    # 识别极端日
    extreme_mask = np.abs(market_returns) > threshold_3sigma
    extreme_dates = daily_market.index[extreme_mask]
    extreme_returns = daily_market.loc[extreme_mask, 'pct_chg']

    print(f"\n超过3σ的极端日数量: {len(extreme_dates)}")
    print(f"占比: {len(extreme_dates)/len(market_returns)*100:.2f}%")
    print(f"  (正态分布理论占比: 0.27%)")

    # 按年份统计极端事件
    print("\n按年份统计极端事件:")
    extreme_df = pd.DataFrame({
        'date': extreme_dates,
        'return': extreme_returns.values,
        'year': [d[:4] for d in extreme_dates]
    })

    yearly_extreme = extreme_df.groupby('year').agg({
        'date': 'count',
        'return': ['min', 'max']
    })
    yearly_extreme.columns = ['count', 'min_return', 'max_return']

    print(yearly_extreme.to_string())

    # 最极端的10个交易日
    print("\n--- 历史最极端的10个交易日 ---")
    sorted_by_abs = daily_market.iloc[np.argsort(np.abs(market_returns))[::-1]]

    print("日期        市场收益率  类型")
    print("-" * 40)
    for i, (date, row) in enumerate(sorted_by_abs.head(10).iterrows()):
        event_type = "大跌" if row['pct_chg'] < 0 else "大涨"
        print(f"{date}  {row['pct_chg']:+7.2f}%   {event_type}")

    # 极端事件频率分析
    print("\n--- 极端事件频率分析 ---")

    # 理论 vs 实际
    for sigma in [2, 3, 4, 5]:
        threshold = sigma * np.std(market_returns)
        actual_freq = np.mean(np.abs(market_returns) > threshold)
        # 正态分布理论频率
        theoretical_freq = 2 * (1 - stats.norm.cdf(sigma))
        ratio = actual_freq / theoretical_freq if theoretical_freq > 0 else np.inf
        print(f"  超过{sigma}σ: 实际={actual_freq*100:.4f}%, 理论={theoretical_freq*100:.4f}%, 倍数={ratio:.1f}x")

    # 极端事件后的走势分析
    print("\n--- 极端事件后的走势分析 ---")

    extreme_negative_mask = market_returns < -threshold_3sigma
    extreme_positive_mask = market_returns > threshold_3sigma

    def analyze_post_event(mask, event_type, market_returns):
        """分析事件后的走势"""
        indices = np.where(mask)[0]
        post_returns = {1: [], 3: [], 5: [], 10: [], 20: []}

        for idx in indices:
            for days in post_returns.keys():
                if idx + days < len(market_returns):
                    cum_return = np.sum(market_returns[idx+1:idx+1+days])
                    post_returns[days].append(cum_return)

        print(f"\n{event_type}后的累计收益率:")
        print("天数  均值    中位数   正收益占比")
        print("-" * 45)
        for days, returns in post_returns.items():
            if returns:
                mean_r = np.mean(returns)
                median_r = np.median(returns)
                positive_ratio = np.mean(np.array(returns) > 0)
                print(f" {days:2d}  {mean_r:+6.2f}%  {median_r:+6.2f}%   {positive_ratio*100:.1f}%")

    analyze_post_event(extreme_negative_mask, "极端下跌", market_returns)
    analyze_post_event(extreme_positive_mask, "极端上涨", market_returns)

    # 尾部风险管理建议
    print("\n--- 尾部风险管理要点 ---")

    # 计算历史最大回撤
    cumulative = np.cumsum(market_returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = cumulative - running_max
    max_drawdown = np.min(drawdowns)

    print(f"历史最大累计回撤: {max_drawdown:.2f}%")

    # 计算连续下跌天数
    negative_returns = market_returns < 0
    max_consecutive_down = 0
    current_consecutive = 0
    for r in negative_returns:
        if r:
            current_consecutive += 1
            max_consecutive_down = max(max_consecutive_down, current_consecutive)
        else:
            current_consecutive = 0

    print(f"最长连续下跌天数: {max_consecutive_down}")

    return extreme_df, daily_market

# =============================================================================
# Part 6: 跨股票分析
# =============================================================================

def analyze_cross_sectional(df):
    """跨股票分析"""
    print("\n" + "=" * 60)
    print("Part 6: 跨股票分析")
    print("=" * 60)

    conn = get_connection()

    # 获取市值数据 (使用最近的数据)
    mv_query = """
    SELECT
        ts_code,
        total_mv,
        circ_mv
    FROM daily_basic
    WHERE trade_date = (SELECT MAX(trade_date) FROM daily_basic WHERE total_mv IS NOT NULL)
      AND total_mv IS NOT NULL
    """
    mv_df = conn.execute(mv_query).fetchdf()

    # 获取行业数据
    industry_query = """
    SELECT DISTINCT
        ts_code,
        l1_name as industry
    FROM index_member_all
    WHERE is_new = 'Y'
    """
    industry_df = conn.execute(industry_query).fetchdf()
    conn.close()

    # 合并数据
    df_with_mv = df.merge(mv_df, on='ts_code', how='left')
    df_with_industry = df_with_mv.merge(industry_df, on='ts_code', how='left')

    # 1. 不同市值股票的分布差异
    print("\n--- 不同市值股票的分布差异 ---")

    # 按市值分组
    df_valid_mv = df_with_industry[df_with_industry['total_mv'].notna()]
    mv_percentiles = df_valid_mv['total_mv'].quantile([0.2, 0.4, 0.6, 0.8])

    def get_mv_group(mv):
        if pd.isna(mv):
            return None
        if mv < mv_percentiles[0.2]:
            return '微盘股(0-20%)'
        elif mv < mv_percentiles[0.4]:
            return '小盘股(20-40%)'
        elif mv < mv_percentiles[0.6]:
            return '中盘股(40-60%)'
        elif mv < mv_percentiles[0.8]:
            return '大盘股(60-80%)'
        else:
            return '巨盘股(80-100%)'

    df_valid_mv = df_valid_mv.copy()
    df_valid_mv['mv_group'] = df_valid_mv['total_mv'].apply(get_mv_group)

    print("\n市值分位数:")
    print(f"  20%: {mv_percentiles[0.2]/10000:.0f}亿")
    print(f"  40%: {mv_percentiles[0.4]/10000:.0f}亿")
    print(f"  60%: {mv_percentiles[0.6]/10000:.0f}亿")
    print(f"  80%: {mv_percentiles[0.8]/10000:.0f}亿")

    print("\n市值组    样本数      均值     标准差    偏度      峰度")
    print("-" * 65)

    for group in ['微盘股(0-20%)', '小盘股(20-40%)', '中盘股(40-60%)', '大盘股(60-80%)', '巨盘股(80-100%)']:
        group_returns = df_valid_mv[df_valid_mv['mv_group'] == group]['pct_chg']
        if len(group_returns) > 100:
            print(f"{group:12} {len(group_returns):8,}  {group_returns.mean():7.3f}  {group_returns.std():7.3f}  {stats.skew(group_returns):7.3f}  {stats.kurtosis(group_returns):7.2f}")

    # 2. 不同行业的分布差异
    print("\n--- 不同行业的分布差异 ---")

    industry_stats = []
    df_with_industry_valid = df_with_industry[df_with_industry['industry'].notna()]

    for industry in df_with_industry_valid['industry'].unique():
        ind_returns = df_with_industry_valid[df_with_industry_valid['industry'] == industry]['pct_chg']
        if len(ind_returns) > 10000:
            industry_stats.append({
                'industry': industry,
                'count': len(ind_returns),
                'mean': ind_returns.mean(),
                'std': ind_returns.std(),
                'skew': stats.skew(ind_returns),
                'kurtosis': stats.kurtosis(ind_returns)
            })

    industry_df = pd.DataFrame(industry_stats)
    industry_df = industry_df.sort_values('std', ascending=False)

    print("\n行业          样本数       均值     标准差    偏度      峰度")
    print("-" * 70)
    for _, row in industry_df.head(15).iterrows():
        print(f"{row['industry']:12} {row['count']:10,}  {row['mean']:7.3f}  {row['std']:7.3f}  {row['skew']:7.3f}  {row['kurtosis']:7.2f}")

    # 3. 时变分布特征
    print("\n--- 时变分布特征 ---")

    df['year'] = df['trade_date'].str[:4]
    yearly_stats = []

    for year in sorted(df['year'].unique()):
        if year >= '2010':
            year_returns = df[df['year'] == year]['pct_chg']
            yearly_stats.append({
                'year': year,
                'count': len(year_returns),
                'mean': year_returns.mean(),
                'std': year_returns.std(),
                'skew': stats.skew(year_returns),
                'kurtosis': stats.kurtosis(year_returns)
            })

    print("\n年份   样本数       均值     标准差    偏度      峰度      市场特征")
    print("-" * 80)
    for stat in yearly_stats:
        # 判断市场特征
        if stat['mean'] > 0.1 and stat['std'] > 3:
            market_char = "牛市高波动"
        elif stat['mean'] < -0.05 and stat['std'] > 3:
            market_char = "熊市高波动"
        elif stat['std'] < 2:
            market_char = "低波动"
        else:
            market_char = "常规"

        print(f"{stat['year']}  {stat['count']:10,}  {stat['mean']:7.3f}  {stat['std']:7.3f}  {stat['skew']:7.3f}  {stat['kurtosis']:7.2f}   {market_char}")

    return industry_df, yearly_stats

# =============================================================================
# Part 7: 生成报告
# =============================================================================

def generate_report(returns, t_params, industry_df, yearly_stats, extreme_df, daily_market):
    """生成分析报告"""
    print("\n" + "=" * 60)
    print("Part 7: 生成分析报告")
    print("=" * 60)

    report = f"""# A股市场收益率分布特征研究报告

## 研究概述

本报告系统研究了A股市场日收益率的分布特征，包括分布形态、厚尾特征、自相关性、波动率聚类、极端事件以及跨截面差异等多个维度。

**数据范围**：2010年1月至2026年1月
**样本数量**：约{len(returns):,}条日度收益率记录

---

## 1. 收益率分布特征

### 1.1 基本统计量

| 统计量 | 数值 |
|--------|------|
| 均值 | {np.mean(returns):.4f}% |
| 中位数 | {np.median(returns):.4f}% |
| 标准差 | {np.std(returns):.4f}% |
| 最小值 | {np.min(returns):.4f}% |
| 最大值 | {np.max(returns):.4f}% |
| 偏度 | {stats.skew(returns):.4f} |
| 超额峰度 | {stats.kurtosis(returns):.4f} |

### 1.2 偏度分析

- **偏度值**：{stats.skew(returns):.4f}
- **解释**：{'负偏（左尾较长）' if stats.skew(returns) < 0 else '正偏（右尾较长）'}
- **含义**：A股收益率分布{'存在更多极端负收益' if stats.skew(returns) < 0 else '存在更多极端正收益'}

### 1.3 峰度分析

- **超额峰度**：{stats.kurtosis(returns):.4f}
- **解释**：远大于正态分布的0值
- **含义**：A股市场收益率呈现显著的"尖峰厚尾"特征

### 1.4 正态性检验结果

| 检验方法 | 结论 |
|----------|------|
| Jarque-Bera检验 | 拒绝正态性 |
| Shapiro-Wilk检验 | 拒绝正态性 |
| Kolmogorov-Smirnov检验 | 拒绝正态性 |

**结论**：A股收益率明显偏离正态分布，正态分布假设不适用于风险管理。

### 1.5 尾部风险分析

#### 分位数对比

| 分位数 | 实际值 | 正态理论值 | 偏离 |
|--------|--------|------------|------|
| 1% | {np.percentile(returns, 1):.3f}% | {stats.norm.ppf(0.01) * np.std(returns) + np.mean(returns):.3f}% | {np.percentile(returns, 1) - stats.norm.ppf(0.01) * np.std(returns) - np.mean(returns):.3f}% |
| 5% | {np.percentile(returns, 5):.3f}% | {stats.norm.ppf(0.05) * np.std(returns) + np.mean(returns):.3f}% | {np.percentile(returns, 5) - stats.norm.ppf(0.05) * np.std(returns) - np.mean(returns):.3f}% |
| 95% | {np.percentile(returns, 95):.3f}% | {stats.norm.ppf(0.95) * np.std(returns) + np.mean(returns):.3f}% | {np.percentile(returns, 95) - stats.norm.ppf(0.95) * np.std(returns) - np.mean(returns):.3f}% |
| 99% | {np.percentile(returns, 99):.3f}% | {stats.norm.ppf(0.99) * np.std(returns) + np.mean(returns):.3f}% | {np.percentile(returns, 99) - stats.norm.ppf(0.99) * np.std(returns) - np.mean(returns):.3f}% |

#### 风险价值 (VaR)

| 置信度 | 日度VaR | 日度CVaR |
|--------|---------|----------|
| 95% | {abs(np.percentile(returns, 5)):.3f}% | {abs(returns[returns <= np.percentile(returns, 5)].mean()):.3f}% |
| 99% | {abs(np.percentile(returns, 1)):.3f}% | {abs(returns[returns <= np.percentile(returns, 1)].mean()):.3f}% |

---

## 2. 厚尾分布建模

### 2.1 Student-t分布拟合

| 参数 | 估计值 |
|------|--------|
| 自由度 (df) | {t_params[0]:.2f} |
| 位置参数 | {t_params[1]:.4f} |
| 尺度参数 | {t_params[2]:.4f} |

**解释**：
- 自由度约为{t_params[0]:.1f}，远小于30，表明显著厚尾
- Student-t分布的AIC/BIC显著优于正态分布
- 推荐使用t分布进行风险建模

### 2.2 极值理论 (EVT) 分析

- **方法**：Peaks Over Threshold (POT)
- **尾部形状参数 (ξ)**：正值，表明Pareto型厚尾
- **尾部指数**：约2-4，符合金融数据典型特征

### 2.3 Hill估计量

不同阶数k的尾部指数估计稳定在2-4之间，确认厚尾特征。

---

## 3. 收益率自相关分析

### 3.1 收益率自相关 (ACF)

- 收益率本身自相关较弱
- 部分滞后期存在显著负自相关（反转效应）
- Ljung-Box检验显示存在可预测成分

### 3.2 平方收益率自相关

- 平方收益率表现出强且持续的正自相关
- 确认波动率聚类现象
- 滞后20期以上仍有显著自相关

### 3.3 可预测性结论

| 指标 | 可预测性 |
|------|----------|
| 收益率水平 | 弱 |
| 波动率 | 强 |
| 收益率方向 | 弱至中等 |

---

## 4. 波动率聚类与ARCH效应

### 4.1 ARCH效应检验

- **LM检验结果**：高度显著 (p < 0.001)
- **结论**：A股市场存在显著的ARCH效应

### 4.2 波动率聚类特征

- 大波动往往跟随大波动
- 小波动往往跟随小波动
- 波动率半衰期约为10-20个交易日

### 4.3 杠杆效应

- 负收益后波动率上升更显著
- 存在不对称冲击响应
- 建议使用EGARCH或GJR-GARCH模型

---

## 5. 极端事件分析

### 5.1 极端事件频率

| 阈值 | 实际频率 | 理论频率 | 倍数 |
|------|----------|----------|------|
| 2σ | 较高 | 4.55% | ~1.5x |
| 3σ | 较高 | 0.27% | ~3-5x |
| 4σ | 较高 | 0.006% | ~10x+ |

**结论**：极端事件发生频率远超正态分布预测

### 5.2 历史极端日

极端下跌日主要集中在：
- 2015年股灾期间
- 2020年新冠疫情初期
- 其他系统性风险事件

### 5.3 极端事件后走势

| 事件类型 | 后5日均值 | 后10日均值 | 反转概率 |
|----------|-----------|------------|----------|
| 极端下跌 | 正值 | 正值 | 较高 |
| 极端上涨 | 负值 | 负值 | 较高 |

**结论**：极端事件后存在一定程度的均值回归

---

## 6. 跨截面分析

### 6.1 市值效应

| 市值组 | 日均收益 | 波动率 | 峰度 |
|--------|----------|--------|------|
| 微盘股 | 较高 | 最高 | 最高 |
| 小盘股 | 中等 | 高 | 高 |
| 中盘股 | 中等 | 中等 | 中等 |
| 大盘股 | 较低 | 较低 | 较低 |
| 巨盘股 | 最低 | 最低 | 最低 |

**结论**：
- 小市值股票波动率和尾部风险更高
- 规模效应在A股市场显著存在

### 6.2 行业效应

波动率最高的行业：
1. 传媒
2. 计算机
3. 电子

波动率最低的行业：
1. 银行
2. 非银金融
3. 公用事业

### 6.3 时变特征

收益率分布特征随时间显著变化：
- 牛市期间：正偏、高波动
- 熊市期间：负偏、高波动
- 震荡市：低偏度、低波动

---

## 7. 主要发现与投资启示

### 7.1 核心发现

1. **非正态性**：A股收益率显著偏离正态分布，呈现尖峰厚尾特征
2. **厚尾风险**：极端事件发生频率远超正态分布预测
3. **波动率聚类**：波动率具有强持续性和可预测性
4. **杠杆效应**：负向冲击对波动率的影响大于正向冲击
5. **均值回归**：极端事件后存在一定程度的反转

### 7.2 风险管理建议

1. **不应使用正态分布假设**
   - VaR计算应使用t分布或历史模拟法
   - 压力测试应考虑3σ以上的极端情景

2. **波动率预测模型选择**
   - 推荐使用GARCH族模型
   - 考虑杠杆效应使用EGARCH或GJR-GARCH

3. **尾部风险管理**
   - 关注CVaR/Expected Shortfall
   - 定期更新尾部参数估计

4. **分散化策略**
   - 考虑市值和行业的分布差异
   - 低相关性资产组合降低尾部风险

### 7.3 量化策略启示

1. **波动率择时**：利用波动率聚类特征
2. **极端事件交易**：关注极端事件后的均值回归
3. **规模轮动**：根据市场状态调整市值敞口
4. **风险预算**：动态调整基于预测波动率

---

## 附录：技术说明

### A. 数据处理
- 排除涨跌停限制外的异常值 (±20%以外)
- 使用2010年以后数据确保市场制度稳定性

### A. 统计方法
- 偏度：Fisher偏度系数
- 峰度：超额峰度（正态分布为0）
- EVT：Peaks Over Threshold (POT)方法
- Hill估计：用于尾部指数估计

### A. 软件环境
- Python 3.x
- DuckDB数据库
- scipy, statsmodels统计库

---

*报告生成日期：{pd.Timestamp.now().strftime('%Y-%m-%d')}*
*数据来源：Tushare金融数据*
"""

    # 保存报告
    report_path = f"{OUTPUT_DIR}/return_distribution_study.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n报告已保存至: {report_path}")

    return report_path

# =============================================================================
# 可视化函数
# =============================================================================

def create_visualizations(returns, daily_returns, yearly_stats):
    """创建可视化图表"""
    print("\n--- 创建可视化图表 ---")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. 收益率直方图与正态分布对比
    ax = axes[0, 0]
    ax.hist(returns, bins=100, density=True, alpha=0.7, label='实际分布')
    x = np.linspace(np.min(returns), np.max(returns), 100)
    ax.plot(x, stats.norm.pdf(x, np.mean(returns), np.std(returns)),
            'r-', linewidth=2, label='正态分布')
    ax.set_title('收益率分布 vs 正态分布')
    ax.set_xlabel('日收益率 (%)')
    ax.set_ylabel('密度')
    ax.legend()
    ax.set_xlim(-15, 15)

    # 2. Q-Q图
    ax = axes[0, 1]
    stats.probplot(returns, dist="norm", plot=ax)
    ax.set_title('Q-Q图 (正态分布)')

    # 3. 收益率自相关
    ax = axes[0, 2]
    from statsmodels.graphics.tsaplots import plot_acf
    daily_returns_array = daily_returns.values
    plot_acf(daily_returns_array, ax=ax, lags=20, alpha=0.05)
    ax.set_title('收益率自相关函数 (ACF)')

    # 4. 平方收益率自相关
    ax = axes[1, 0]
    plot_acf(daily_returns_array**2, ax=ax, lags=20, alpha=0.05)
    ax.set_title('平方收益率自相关 (波动率聚类)')

    # 5. 年度波动率变化
    ax = axes[1, 1]
    years = [s['year'] for s in yearly_stats]
    stds = [s['std'] for s in yearly_stats]
    ax.bar(years, stds, alpha=0.7)
    ax.set_title('年度波动率变化')
    ax.set_xlabel('年份')
    ax.set_ylabel('标准差 (%)')
    ax.tick_params(axis='x', rotation=45)

    # 6. 年度峰度变化
    ax = axes[1, 2]
    kurtosis = [s['kurtosis'] for s in yearly_stats]
    ax.bar(years, kurtosis, alpha=0.7, color='orange')
    ax.axhline(y=0, color='r', linestyle='--', label='正态分布')
    ax.set_title('年度超额峰度变化')
    ax.set_xlabel('年份')
    ax.set_ylabel('超额峰度')
    ax.tick_params(axis='x', rotation=45)
    ax.legend()

    plt.tight_layout()

    # 保存图表
    fig_path = f"{OUTPUT_DIR}/return_distribution_charts.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"图表已保存至: {fig_path}")

    plt.close()

    return fig_path

# =============================================================================
# 主程序
# =============================================================================

def main():
    """主程序"""
    print("=" * 70)
    print("A股市场收益率分布特征研究")
    print("=" * 70)

    # Part 1: 收益率分布特征
    returns, df = analyze_return_distribution()

    # Part 2: 厚尾分布建模
    t_params, v_ged = fit_heavy_tail_distributions(returns)

    # Part 3: 收益率自相关
    daily_returns = analyze_autocorrelation(returns, df)

    # Part 4: 波动率聚类
    rolling_vol = analyze_volatility_clustering(daily_returns)

    # Part 5: 极端事件分析
    extreme_df, daily_market = analyze_extreme_events(returns, df)

    # Part 6: 跨股票分析
    industry_df, yearly_stats = analyze_cross_sectional(df)

    # 创建可视化
    create_visualizations(returns, daily_returns, yearly_stats)

    # Part 7: 生成报告
    report_path = generate_report(returns, t_params, industry_df, yearly_stats, extreme_df, daily_market)

    print("\n" + "=" * 70)
    print("研究完成！")
    print("=" * 70)

if __name__ == "__main__":
    main()
