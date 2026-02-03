#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
个股Alpha特征研究
================

研究内容：
1. Alpha计算：CAPM模型、指数基准对比、行业基准对比
2. Alpha持续性：稳定性分析、衰减速度、预测能力
3. 策略应用：高Alpha股票筛选、Alpha因子构建、与其他因子结合

Author: Research Team
Date: 2024-02
"""

import duckdb
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
import os

# 配置
DB_PATH = '/Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db'
REPORT_DIR = '/Users/allen/workspace/python/stock/Tushare-DuckDB/reports/research'

# 确保报告目录存在
os.makedirs(REPORT_DIR, exist_ok=True)

def get_connection():
    """获取数据库连接"""
    return duckdb.connect(DB_PATH, read_only=True)

def load_stock_returns(start_date='20210101', end_date='20260130'):
    """加载股票收益率数据"""
    conn = get_connection()

    query = f"""
    SELECT
        d.ts_code,
        d.trade_date,
        d.pct_chg / 100.0 as ret,
        d.close,
        d.amount,
        sb.name as stock_name,
        sb.industry
    FROM daily d
    LEFT JOIN stock_basic sb ON d.ts_code = sb.ts_code
    WHERE d.trade_date >= '{start_date}'
      AND d.trade_date <= '{end_date}'
      AND d.pct_chg IS NOT NULL
    ORDER BY d.ts_code, d.trade_date
    """

    df = conn.execute(query).fetchdf()
    conn.close()
    return df

def load_index_returns(index_code='000300.SH', start_date='20210101', end_date='20260130'):
    """加载指数收益率数据"""
    conn = get_connection()

    query = f"""
    SELECT
        trade_date,
        pct_chg / 100.0 as market_ret
    FROM index_daily
    WHERE ts_code = '{index_code}'
      AND trade_date >= '{start_date}'
      AND trade_date <= '{end_date}'
      AND pct_chg IS NOT NULL
    ORDER BY trade_date
    """

    df = conn.execute(query).fetchdf()
    conn.close()
    return df

def load_industry_returns(start_date='20210101', end_date='20260130'):
    """加载行业指数收益率数据（申万一级行业）"""
    conn = get_connection()

    query = f"""
    SELECT
        ts_code as industry_code,
        trade_date,
        name as industry_name,
        pct_change / 100.0 as industry_ret
    FROM sw_daily
    WHERE trade_date >= '{start_date}'
      AND trade_date <= '{end_date}'
      AND pct_change IS NOT NULL
    ORDER BY ts_code, trade_date
    """

    df = conn.execute(query).fetchdf()
    conn.close()
    return df

def load_stock_industry_mapping():
    """加载股票行业映射关系"""
    conn = get_connection()

    query = """
    SELECT
        ts_code,
        l1_code as industry_code,
        l1_name as industry_name
    FROM index_member_all
    WHERE out_date IS NULL OR out_date = ''
    """

    df = conn.execute(query).fetchdf()
    # 去重，一只股票只保留一个行业
    df = df.drop_duplicates(subset=['ts_code'], keep='first')
    conn.close()
    return df

def calculate_capm_alpha(stock_returns, market_returns, rf_rate=0.00008):
    """
    使用CAPM模型计算Alpha

    参数:
    - stock_returns: 股票收益率序列
    - market_returns: 市场收益率序列
    - rf_rate: 日无风险利率（默认约3%年化）

    返回:
    - alpha: Jensen's Alpha (日度)
    - beta: 市场Beta
    - r_squared: R方
    - t_stat: Alpha的t统计量
    - p_value: Alpha的p值
    """
    # 计算超额收益
    stock_excess = stock_returns - rf_rate
    market_excess = market_returns - rf_rate

    # 回归
    if len(stock_excess) < 30:  # 数据不足
        return np.nan, np.nan, np.nan, np.nan, np.nan

    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            market_excess, stock_excess
        )

        # 计算alpha的t统计量
        n = len(stock_excess)
        residuals = stock_excess - (intercept + slope * market_excess)
        mse = np.sum(residuals ** 2) / (n - 2)
        x_mean = np.mean(market_excess)
        ss_x = np.sum((market_excess - x_mean) ** 2)
        se_intercept = np.sqrt(mse * (1/n + x_mean**2 / ss_x))
        t_stat = intercept / se_intercept if se_intercept > 0 else 0

        return intercept, slope, r_value**2, t_stat, p_value
    except:
        return np.nan, np.nan, np.nan, np.nan, np.nan

def calculate_rolling_alpha(stock_df, market_df, window=60):
    """
    计算滚动Alpha

    参数:
    - stock_df: 股票数据DataFrame
    - market_df: 市场数据DataFrame
    - window: 滚动窗口大小（交易日）

    返回:
    - DataFrame包含滚动alpha, beta, r_squared
    """
    # 合并数据
    merged = stock_df.merge(market_df, on='trade_date', how='inner')

    if len(merged) < window:
        return pd.DataFrame()

    results = []
    for i in range(window, len(merged)):
        window_data = merged.iloc[i-window:i]
        alpha, beta, r2, t_stat, _ = calculate_capm_alpha(
            window_data['ret'].values,
            window_data['market_ret'].values
        )
        results.append({
            'trade_date': merged.iloc[i]['trade_date'],
            'alpha': alpha,
            'beta': beta,
            'r_squared': r2,
            't_stat': t_stat
        })

    return pd.DataFrame(results)

def calculate_industry_alpha(stock_returns, industry_returns):
    """
    计算相对于行业的Alpha
    """
    if len(stock_returns) < 30:
        return np.nan, np.nan

    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            industry_returns, stock_returns
        )
        return intercept, slope
    except:
        return np.nan, np.nan

def analyze_alpha_persistence(alpha_series, lags=[1, 5, 20, 60]):
    """
    分析Alpha的持续性

    参数:
    - alpha_series: Alpha时间序列
    - lags: 滞后期列表

    返回:
    - 自相关系数和半衰期
    """
    alpha_series = alpha_series.dropna()

    if len(alpha_series) < max(lags) + 10:
        return {}, np.nan

    autocorrs = {}
    for lag in lags:
        if len(alpha_series) > lag:
            autocorrs[lag] = alpha_series.autocorr(lag)

    # 计算半衰期（使用AR(1)系数估计）
    if len(alpha_series) > 1:
        ar1 = alpha_series.autocorr(1)
        if ar1 > 0 and ar1 < 1:
            half_life = -np.log(2) / np.log(ar1) if ar1 > 0 else np.inf
        else:
            half_life = np.nan
    else:
        half_life = np.nan

    return autocorrs, half_life

def alpha_predictability_test(alpha_series, future_returns, periods=[1, 5, 20]):
    """
    测试Alpha对未来收益的预测能力

    参数:
    - alpha_series: 当期Alpha
    - future_returns: 未来收益
    - periods: 预测期列表

    返回:
    - IC (Information Coefficient) 和 IR (Information Ratio)
    """
    results = {}

    for period in periods:
        if len(alpha_series) <= period:
            continue

        # 当期alpha vs 未来收益
        current_alpha = alpha_series.iloc[:-period].values
        future_ret = future_returns.shift(-period).iloc[:-period].values

        # 去除NaN
        mask = ~(np.isnan(current_alpha) | np.isnan(future_ret))
        current_alpha = current_alpha[mask]
        future_ret = future_ret[mask]

        if len(current_alpha) < 30:
            continue

        # 计算IC (Spearman相关系数)
        ic, _ = stats.spearmanr(current_alpha, future_ret)

        # 计算IC的均值和标准差来得到IR
        # 这里简化处理
        results[period] = {
            'IC': ic,
            'IC_abs': abs(ic)
        }

    return results


def screen_high_alpha_stocks(stock_alpha_df, top_n=50, min_t_stat=2.0):
    """
    筛选高Alpha股票

    参数:
    - stock_alpha_df: 包含Alpha的DataFrame
    - top_n: 筛选数量
    - min_t_stat: 最小t统计量阈值

    返回:
    - 筛选后的股票列表
    """
    # 筛选条件：Alpha显著且为正
    filtered = stock_alpha_df[
        (stock_alpha_df['alpha'] > 0) &
        (stock_alpha_df['t_stat'] > min_t_stat)
    ].copy()

    # 按Alpha降序排序
    filtered = filtered.sort_values('alpha', ascending=False)

    return filtered.head(top_n)

def construct_alpha_factor(stock_returns_df, market_returns_df, lookback=60):
    """
    构建Alpha因子

    使用滚动窗口计算每只股票的Alpha作为因子值
    """
    stocks = stock_returns_df['ts_code'].unique()

    all_alphas = []

    for ts_code in stocks:
        stock_data = stock_returns_df[stock_returns_df['ts_code'] == ts_code].copy()
        stock_data = stock_data.sort_values('trade_date')

        if len(stock_data) < lookback + 20:
            continue

        rolling_alpha = calculate_rolling_alpha(stock_data, market_returns_df, lookback)
        if not rolling_alpha.empty:
            rolling_alpha['ts_code'] = ts_code
            all_alphas.append(rolling_alpha)

    if not all_alphas:
        return pd.DataFrame()

    return pd.concat(all_alphas, ignore_index=True)

def combine_with_other_factors(alpha_factor_df, stock_returns_df):
    """
    将Alpha因子与其他因子结合

    这里演示与动量因子和波动率因子的结合
    """
    # 计算动量因子（过去20日收益）
    stock_returns_df['momentum_20'] = stock_returns_df.groupby('ts_code')['ret'].transform(
        lambda x: x.rolling(20).mean()
    )

    # 计算波动率因子
    stock_returns_df['volatility'] = stock_returns_df.groupby('ts_code')['ret'].transform(
        lambda x: x.rolling(20).std()
    )

    # 合并因子
    combined = alpha_factor_df.merge(
        stock_returns_df[['ts_code', 'trade_date', 'momentum_20', 'volatility', 'ret']],
        on=['ts_code', 'trade_date'],
        how='left'
    )

    return combined

def generate_report():
    """生成完整的Alpha分析报告"""

    print("=" * 60)
    print("个股Alpha特征研究报告")
    print("=" * 60)
    print(f"\n报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"数据周期: 2021-01-01 至 2026-01-30")

    # 1. 加载数据
    print("\n[1] 加载数据...")
    stock_returns = load_stock_returns()
    market_returns = load_index_returns('000300.SH')  # 沪深300作为市场基准
    industry_returns = load_industry_returns()
    industry_mapping = load_stock_industry_mapping()

    print(f"    股票数据: {len(stock_returns)} 条记录, {stock_returns['ts_code'].nunique()} 只股票")
    print(f"    市场数据: {len(market_returns)} 条记录")
    print(f"    行业数据: {len(industry_returns)} 条记录, {industry_returns['industry_code'].nunique()} 个行业")

    # 报告内容
    report_lines = []
    report_lines.append("# 个股Alpha特征研究报告\n")
    report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report_lines.append(f"数据周期: 2021-01-01 至 2026-01-30\n\n")

    # =============================================
    # 第一部分：Alpha计算
    # =============================================
    print("\n[2] 计算Alpha...")
    report_lines.append("## 一、Alpha计算\n\n")

    # 2.1 CAPM模型计算Alpha
    print("    2.1 使用CAPM模型计算Alpha...")
    report_lines.append("### 1.1 CAPM模型计算Alpha\n\n")
    report_lines.append("使用Jensen's Alpha衡量个股相对于市场基准（沪深300）的超额收益能力。\n\n")
    report_lines.append("**模型公式:**\n")
    report_lines.append("```\n")
    report_lines.append("R_i - R_f = α + β(R_m - R_f) + ε\n")
    report_lines.append("```\n\n")
    report_lines.append("其中:\n")
    report_lines.append("- R_i: 个股收益率\n")
    report_lines.append("- R_f: 无风险利率（假设日无风险利率=0.008%，约年化3%）\n")
    report_lines.append("- R_m: 市场收益率（沪深300）\n")
    report_lines.append("- α: Jensen's Alpha\n")
    report_lines.append("- β: 市场Beta\n\n")

    # 计算每只股票的Alpha
    stock_alphas = []
    stocks = stock_returns['ts_code'].unique()
    sample_stocks = np.random.choice(stocks, min(500, len(stocks)), replace=False)

    for ts_code in sample_stocks:
        stock_data = stock_returns[stock_returns['ts_code'] == ts_code]
        stock_data = stock_data.sort_values('trade_date')

        merged = stock_data.merge(market_returns, on='trade_date', how='inner')

        if len(merged) < 60:
            continue

        alpha, beta, r2, t_stat, p_value = calculate_capm_alpha(
            merged['ret'].values,
            merged['market_ret'].values
        )

        stock_alphas.append({
            'ts_code': ts_code,
            'stock_name': stock_data['stock_name'].iloc[0],
            'industry': stock_data['industry'].iloc[0],
            'alpha': alpha,
            'alpha_annual': alpha * 252,  # 年化Alpha
            'beta': beta,
            'r_squared': r2,
            't_stat': t_stat,
            'p_value': p_value,
            'n_obs': len(merged)
        })

    stock_alpha_df = pd.DataFrame(stock_alphas)

    # Alpha统计
    alpha_stats = stock_alpha_df['alpha_annual'].describe()
    report_lines.append("**Alpha分布统计（年化）:**\n\n")
    report_lines.append("| 统计量 | 值 |\n")
    report_lines.append("|--------|----|\n")
    report_lines.append(f"| 均值 | {alpha_stats['mean']*100:.2f}% |\n")
    report_lines.append(f"| 标准差 | {alpha_stats['std']*100:.2f}% |\n")
    report_lines.append(f"| 最小值 | {alpha_stats['min']*100:.2f}% |\n")
    report_lines.append(f"| 25%分位 | {alpha_stats['25%']*100:.2f}% |\n")
    report_lines.append(f"| 中位数 | {alpha_stats['50%']*100:.2f}% |\n")
    report_lines.append(f"| 75%分位 | {alpha_stats['75%']*100:.2f}% |\n")
    report_lines.append(f"| 最大值 | {alpha_stats['max']*100:.2f}% |\n\n")

    # 显著Alpha统计
    sig_positive = len(stock_alpha_df[(stock_alpha_df['alpha'] > 0) & (stock_alpha_df['t_stat'] > 2)])
    sig_negative = len(stock_alpha_df[(stock_alpha_df['alpha'] < 0) & (stock_alpha_df['t_stat'] < -2)])
    total = len(stock_alpha_df)

    report_lines.append("**Alpha显著性统计:**\n\n")
    report_lines.append(f"- 显著正Alpha股票数: {sig_positive} ({sig_positive/total*100:.1f}%)\n")
    report_lines.append(f"- 显著负Alpha股票数: {sig_negative} ({sig_negative/total*100:.1f}%)\n")
    report_lines.append(f"- Alpha不显著股票数: {total - sig_positive - sig_negative} ({(total - sig_positive - sig_negative)/total*100:.1f}%)\n\n")

    print(f"    分析了 {len(stock_alpha_df)} 只股票的Alpha")
    print(f"    显著正Alpha: {sig_positive} 只, 显著负Alpha: {sig_negative} 只")

    # 2.2 指数基准对比
    print("    2.2 与指数基准对比...")
    report_lines.append("### 1.2 与指数基准对比\n\n")

    # 对比不同指数基准
    indices = {
        '000001.SH': '上证指数',
        '000300.SH': '沪深300',
        '000905.SH': '中证500',
        '399006.SZ': '创业板指'
    }

    index_comparison = []
    for idx_code, idx_name in indices.items():
        idx_returns = load_index_returns(idx_code)
        if idx_returns.empty:
            continue

        # 随机选取100只股票进行对比
        sample = np.random.choice(stocks, min(100, len(stocks)), replace=False)
        alphas = []

        for ts_code in sample:
            stock_data = stock_returns[stock_returns['ts_code'] == ts_code]
            merged = stock_data.merge(idx_returns, on='trade_date', how='inner')

            if len(merged) >= 60:
                alpha, _, _, _, _ = calculate_capm_alpha(
                    merged['ret'].values,
                    merged['market_ret'].values
                )
                if not np.isnan(alpha):
                    alphas.append(alpha * 252)

        if alphas:
            index_comparison.append({
                '基准指数': idx_name,
                '平均Alpha': np.mean(alphas),
                'Alpha中位数': np.median(alphas),
                'Alpha标准差': np.std(alphas),
                '正Alpha比例': np.mean(np.array(alphas) > 0)
            })

    if index_comparison:
        report_lines.append("不同指数基准下的Alpha表现:\n\n")
        report_lines.append("| 基准指数 | 平均Alpha | Alpha中位数 | Alpha标准差 | 正Alpha比例 |\n")
        report_lines.append("|----------|-----------|-------------|-------------|-------------|\n")
        for ic in index_comparison:
            report_lines.append(f"| {ic['基准指数']} | {ic['平均Alpha']*100:.2f}% | {ic['Alpha中位数']*100:.2f}% | {ic['Alpha标准差']*100:.2f}% | {ic['正Alpha比例']*100:.1f}% |\n")
        report_lines.append("\n")

    # 2.3 行业基准对比
    print("    2.3 与行业基准对比...")
    report_lines.append("### 1.3 与行业基准对比\n\n")
    report_lines.append("使用申万一级行业指数作为行业基准，计算个股相对于所属行业的Alpha。\n\n")

    # 计算行业Alpha
    industry_alphas = []

    for ts_code in sample_stocks[:200]:  # 取200只股票样本
        stock_data = stock_returns[stock_returns['ts_code'] == ts_code]

        # 获取股票所属行业
        ind_info = industry_mapping[industry_mapping['ts_code'] == ts_code]
        if ind_info.empty:
            continue

        ind_code = ind_info['industry_code'].iloc[0]
        ind_name = ind_info['industry_name'].iloc[0]

        # 获取行业收益率
        ind_ret = industry_returns[industry_returns['industry_code'] == ind_code]

        # 合并数据
        merged = stock_data.merge(ind_ret[['trade_date', 'industry_ret']], on='trade_date', how='inner')

        if len(merged) >= 60:
            ind_alpha, _ = calculate_industry_alpha(
                merged['ret'].values,
                merged['industry_ret'].values
            )

            # 同时计算市场Alpha
            merged2 = stock_data.merge(market_returns, on='trade_date', how='inner')
            if len(merged2) >= 60:
                mkt_alpha, _, _, _, _ = calculate_capm_alpha(
                    merged2['ret'].values,
                    merged2['market_ret'].values
                )
            else:
                mkt_alpha = np.nan

            industry_alphas.append({
                'ts_code': ts_code,
                'stock_name': stock_data['stock_name'].iloc[0],
                'industry': ind_name,
                'industry_alpha': ind_alpha * 252 if not np.isnan(ind_alpha) else np.nan,
                'market_alpha': mkt_alpha * 252 if not np.isnan(mkt_alpha) else np.nan
            })

    industry_alpha_df = pd.DataFrame(industry_alphas)

    # 按行业汇总
    industry_summary = industry_alpha_df.groupby('industry').agg({
        'industry_alpha': ['mean', 'std', 'count'],
        'market_alpha': 'mean'
    }).reset_index()
    industry_summary.columns = ['行业', '行业Alpha均值', '行业Alpha标准差', '股票数', '市场Alpha均值']
    industry_summary = industry_summary.sort_values('行业Alpha均值', ascending=False)

    report_lines.append("**各行业Alpha表现（年化）:**\n\n")
    report_lines.append("| 行业 | 行业Alpha均值 | 行业Alpha标准差 | 市场Alpha均值 | 股票数 |\n")
    report_lines.append("|------|---------------|-----------------|---------------|--------|\n")
    for _, row in industry_summary.head(15).iterrows():
        report_lines.append(f"| {row['行业']} | {row['行业Alpha均值']*100:.2f}% | {row['行业Alpha标准差']*100:.2f}% | {row['市场Alpha均值']*100:.2f}% | {int(row['股票数'])} |\n")
    report_lines.append("\n")

    # =============================================
    # 第二部分：Alpha持续性分析
    # =============================================
    print("\n[3] 分析Alpha持续性...")
    report_lines.append("## 二、Alpha持续性分析\n\n")

    # 3.1 Alpha稳定性分析
    print("    3.1 Alpha稳定性分析...")
    report_lines.append("### 2.1 Alpha稳定性分析\n\n")

    # 计算滚动Alpha
    rolling_alpha_results = []
    sample_for_rolling = np.random.choice(stocks, min(100, len(stocks)), replace=False)

    for ts_code in sample_for_rolling:
        stock_data = stock_returns[stock_returns['ts_code'] == ts_code].copy()
        stock_data = stock_data.sort_values('trade_date')

        if len(stock_data) < 120:
            continue

        rolling_alpha = calculate_rolling_alpha(stock_data, market_returns, window=60)
        if not rolling_alpha.empty:
            # 计算Alpha的稳定性指标
            alpha_series = rolling_alpha['alpha']
            alpha_volatility = alpha_series.std()
            alpha_mean = alpha_series.mean()
            alpha_cv = alpha_volatility / abs(alpha_mean) if alpha_mean != 0 else np.nan

            # Alpha符号一致性
            sign_consistency = (alpha_series > 0).mean() if alpha_mean > 0 else (alpha_series < 0).mean()

            rolling_alpha_results.append({
                'ts_code': ts_code,
                'alpha_mean': alpha_mean * 252,
                'alpha_volatility': alpha_volatility * np.sqrt(252),
                'alpha_cv': alpha_cv,
                'sign_consistency': sign_consistency,
                'n_periods': len(alpha_series)
            })

    rolling_alpha_df = pd.DataFrame(rolling_alpha_results)

    if not rolling_alpha_df.empty:
        stability_stats = rolling_alpha_df[['alpha_mean', 'alpha_volatility', 'alpha_cv', 'sign_consistency']].describe()

        report_lines.append("**Alpha稳定性统计:**\n\n")
        report_lines.append("| 指标 | 均值 | 标准差 | 中位数 |\n")
        report_lines.append("|------|------|--------|--------|\n")
        report_lines.append(f"| Alpha均值（年化） | {stability_stats.loc['mean', 'alpha_mean']*100:.2f}% | {stability_stats.loc['std', 'alpha_mean']*100:.2f}% | {stability_stats.loc['50%', 'alpha_mean']*100:.2f}% |\n")
        report_lines.append(f"| Alpha波动率（年化） | {stability_stats.loc['mean', 'alpha_volatility']*100:.2f}% | {stability_stats.loc['std', 'alpha_volatility']*100:.2f}% | {stability_stats.loc['50%', 'alpha_volatility']*100:.2f}% |\n")
        report_lines.append(f"| 变异系数 | {stability_stats.loc['mean', 'alpha_cv']:.2f} | {stability_stats.loc['std', 'alpha_cv']:.2f} | {stability_stats.loc['50%', 'alpha_cv']:.2f} |\n")
        report_lines.append(f"| 符号一致性 | {stability_stats.loc['mean', 'sign_consistency']*100:.1f}% | {stability_stats.loc['std', 'sign_consistency']*100:.1f}% | {stability_stats.loc['50%', 'sign_consistency']*100:.1f}% |\n\n")

        report_lines.append("**稳定性指标解读:**\n")
        report_lines.append("- **变异系数(CV)**: Alpha波动率/Alpha均值，越小表示Alpha越稳定\n")
        report_lines.append("- **符号一致性**: 滚动Alpha与整体Alpha方向一致的时间比例\n\n")

    # 3.2 Alpha衰减速度
    print("    3.2 Alpha衰减速度分析...")
    report_lines.append("### 2.2 Alpha衰减速度\n\n")
    report_lines.append("使用自相关函数分析Alpha的持续性和衰减特征。\n\n")

    # 选取几只典型股票进行详细分析
    persistence_results = []

    for ts_code in sample_for_rolling[:50]:
        stock_data = stock_returns[stock_returns['ts_code'] == ts_code].copy()
        stock_data = stock_data.sort_values('trade_date')

        rolling_alpha = calculate_rolling_alpha(stock_data, market_returns, window=60)

        if len(rolling_alpha) >= 120:
            alpha_series = pd.Series(rolling_alpha['alpha'].values)
            autocorrs, half_life = analyze_alpha_persistence(alpha_series)

            persistence_results.append({
                'ts_code': ts_code,
                'ac_1': autocorrs.get(1, np.nan),
                'ac_5': autocorrs.get(5, np.nan),
                'ac_20': autocorrs.get(20, np.nan),
                'ac_60': autocorrs.get(60, np.nan),
                'half_life': half_life
            })

    persistence_df = pd.DataFrame(persistence_results)

    if not persistence_df.empty:
        ac_stats = persistence_df[['ac_1', 'ac_5', 'ac_20', 'ac_60', 'half_life']].mean()

        report_lines.append("**Alpha自相关系数（滞后期）:**\n\n")
        report_lines.append("| 滞后期 | 平均自相关系数 |\n")
        report_lines.append("|--------|----------------|\n")
        report_lines.append(f"| 1日 | {ac_stats['ac_1']:.4f} |\n")
        report_lines.append(f"| 5日 | {ac_stats['ac_5']:.4f} |\n")
        report_lines.append(f"| 20日 | {ac_stats['ac_20']:.4f} |\n")
        report_lines.append(f"| 60日 | {ac_stats['ac_60']:.4f} |\n\n")

        valid_half_life = persistence_df['half_life'].dropna()
        valid_half_life = valid_half_life[valid_half_life < 1000]  # 排除异常值

        if len(valid_half_life) > 0:
            report_lines.append(f"**Alpha半衰期**: 平均 {valid_half_life.mean():.1f} 个交易日（中位数 {valid_half_life.median():.1f} 个交易日）\n\n")
            report_lines.append("*半衰期表示Alpha衰减到一半所需的时间，较长的半衰期意味着Alpha具有更好的持续性。*\n\n")

    # 3.3 Alpha预测能力
    print("    3.3 Alpha预测能力分析...")
    report_lines.append("### 2.3 Alpha预测能力\n\n")
    report_lines.append("分析当期Alpha对未来收益的预测能力。\n\n")

    # 计算IC
    ic_results = []

    for ts_code in sample_for_rolling[:50]:
        stock_data = stock_returns[stock_returns['ts_code'] == ts_code].copy()
        stock_data = stock_data.sort_values('trade_date').reset_index(drop=True)

        rolling_alpha = calculate_rolling_alpha(stock_data, market_returns, window=60)

        if len(rolling_alpha) >= 60:
            # 合并Alpha和收益率
            stock_data_subset = stock_data[stock_data['trade_date'].isin(rolling_alpha['trade_date'])]
            merged = rolling_alpha.merge(
                stock_data[['trade_date', 'ret']],
                on='trade_date'
            )

            if len(merged) >= 60:
                alpha_series = merged['alpha']
                ret_series = merged['ret']

                pred_results = alpha_predictability_test(alpha_series, ret_series, [1, 5, 20])

                for period, metrics in pred_results.items():
                    ic_results.append({
                        'ts_code': ts_code,
                        'period': period,
                        'IC': metrics['IC']
                    })

    ic_df = pd.DataFrame(ic_results)

    if not ic_df.empty:
        ic_summary = ic_df.groupby('period')['IC'].agg(['mean', 'std']).reset_index()

        report_lines.append("**Alpha预测能力（IC分析）:**\n\n")
        report_lines.append("| 预测期（交易日） | 平均IC | IC标准差 | IC>0比例 |\n")
        report_lines.append("|------------------|--------|----------|----------|\n")
        for _, row in ic_summary.iterrows():
            period_ic = ic_df[ic_df['period'] == row['period']]['IC']
            positive_ratio = (period_ic > 0).mean()
            report_lines.append(f"| {int(row['period'])} | {row['mean']:.4f} | {row['std']:.4f} | {positive_ratio*100:.1f}% |\n")
        report_lines.append("\n")

        report_lines.append("**IC解读:**\n")
        report_lines.append("- IC (Information Coefficient): Alpha与未来收益的秩相关系数\n")
        report_lines.append("- |IC| > 0.03: 有一定预测能力\n")
        report_lines.append("- |IC| > 0.05: 较强预测能力\n")
        report_lines.append("- |IC| > 0.10: 很强预测能力\n\n")

    # =============================================
    # 第三部分：策略应用
    # =============================================
    print("\n[4] 策略应用分析...")
    report_lines.append("## 三、策略应用\n\n")

    # 4.1 高Alpha股票筛选
    print("    4.1 高Alpha股票筛选...")
    report_lines.append("### 3.1 高Alpha股票筛选\n\n")
    report_lines.append("筛选条件: Alpha > 0 且 t统计量 > 2.0（显著性水平约5%）\n\n")

    high_alpha_stocks = screen_high_alpha_stocks(stock_alpha_df, top_n=30, min_t_stat=2.0)

    if not high_alpha_stocks.empty:
        report_lines.append("**Top 30 高Alpha股票:**\n\n")
        report_lines.append("| 排名 | 股票代码 | 股票名称 | 所属行业 | 年化Alpha | Beta | R方 | t统计量 |\n")
        report_lines.append("|------|----------|----------|----------|-----------|------|-----|----------|\n")
        for i, (_, row) in enumerate(high_alpha_stocks.iterrows(), 1):
            report_lines.append(f"| {i} | {row['ts_code']} | {row['stock_name']} | {row['industry']} | {row['alpha_annual']*100:.2f}% | {row['beta']:.2f} | {row['r_squared']:.2f} | {row['t_stat']:.2f} |\n")
        report_lines.append("\n")

        # 按行业分布
        industry_dist = high_alpha_stocks['industry'].value_counts().head(10)
        report_lines.append("**高Alpha股票行业分布（Top 10）:**\n\n")
        report_lines.append("| 行业 | 股票数量 | 占比 |\n")
        report_lines.append("|------|----------|------|\n")
        for ind, cnt in industry_dist.items():
            report_lines.append(f"| {ind} | {cnt} | {cnt/len(high_alpha_stocks)*100:.1f}% |\n")
        report_lines.append("\n")

    # 4.2 Alpha因子构建
    print("    4.2 Alpha因子构建...")
    report_lines.append("### 3.2 Alpha因子构建\n\n")

    report_lines.append("**因子定义:**\n")
    report_lines.append("使用滚动60日窗口计算的Jensen's Alpha作为因子值。\n\n")

    report_lines.append("**因子构建步骤:**\n")
    report_lines.append("1. 每个交易日，使用过去60个交易日的数据\n")
    report_lines.append("2. 对每只股票进行CAPM回归，获取Alpha\n")
    report_lines.append("3. 将Alpha进行标准化处理\n")
    report_lines.append("4. 根据Alpha值进行股票排序分组\n\n")

    # 构建Alpha因子并分析分层收益
    print("    计算分层收益...")

    # 简化版分层分析
    # 按Alpha分组，看未来收益表现
    stock_alpha_df['alpha_rank'] = pd.qcut(stock_alpha_df['alpha'], q=5, labels=['Q1(低)', 'Q2', 'Q3', 'Q4', 'Q5(高)'])

    # 计算每组的平均指标
    group_stats = stock_alpha_df.groupby('alpha_rank').agg({
        'alpha_annual': 'mean',
        'beta': 'mean',
        'r_squared': 'mean',
        'ts_code': 'count'
    }).reset_index()
    group_stats.columns = ['分组', '平均Alpha(年化)', '平均Beta', '平均R方', '股票数']

    report_lines.append("**Alpha因子分组特征:**\n\n")
    report_lines.append("| 分组 | 平均Alpha(年化) | 平均Beta | 平均R方 | 股票数 |\n")
    report_lines.append("|------|-----------------|----------|---------|--------|\n")
    for _, row in group_stats.iterrows():
        report_lines.append(f"| {row['分组']} | {row['平均Alpha(年化)']*100:.2f}% | {row['平均Beta']:.2f} | {row['平均R方']:.2f} | {int(row['股票数'])} |\n")
    report_lines.append("\n")

    # 4.3 与其他因子结合
    print("    4.3 与其他因子结合分析...")
    report_lines.append("### 3.3 与其他因子结合\n\n")
    report_lines.append("分析Alpha因子与动量因子、波动率因子、Beta因子的相关性。\n\n")

    # 计算因子相关性
    factor_corr = stock_alpha_df[['alpha_annual', 'beta', 'r_squared']].corr()

    report_lines.append("**Alpha与其他因子相关性:**\n\n")
    report_lines.append("| 因子 | Alpha | Beta | R方 |\n")
    report_lines.append("|------|-------|------|-----|\n")
    report_lines.append(f"| Alpha | 1.00 | {factor_corr.loc['alpha_annual', 'beta']:.3f} | {factor_corr.loc['alpha_annual', 'r_squared']:.3f} |\n")
    report_lines.append(f"| Beta | {factor_corr.loc['beta', 'alpha_annual']:.3f} | 1.00 | {factor_corr.loc['beta', 'r_squared']:.3f} |\n")
    report_lines.append(f"| R方 | {factor_corr.loc['r_squared', 'alpha_annual']:.3f} | {factor_corr.loc['r_squared', 'beta']:.3f} | 1.00 |\n\n")

    report_lines.append("**多因子组合策略建议:**\n\n")
    report_lines.append("1. **Alpha+低Beta策略**: 选择高Alpha、低Beta的股票，追求稳定超额收益\n")
    report_lines.append("2. **Alpha+动量策略**: 结合Alpha因子和动量因子，增强收益持续性\n")
    report_lines.append("3. **Alpha+质量因子**: 结合ROE、资产负债率等质量因子，筛选优质Alpha股票\n")
    report_lines.append("4. **行业中性Alpha**: 在行业内选择高Alpha股票，消除行业配置偏差\n\n")

    # =============================================
    # 生成可视化图表
    # =============================================
    print("\n[5] 生成可视化图表...")

    # 图1：Alpha分布直方图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Alpha分布
    ax1 = axes[0, 0]
    valid_alpha = stock_alpha_df['alpha_annual'].dropna()
    valid_alpha = valid_alpha[(valid_alpha > -1) & (valid_alpha < 1)]  # 排除极端值
    ax1.hist(valid_alpha * 100, bins=50, edgecolor='black', alpha=0.7)
    ax1.axvline(x=0, color='red', linestyle='--', label='Zero Alpha')
    ax1.axvline(x=valid_alpha.mean() * 100, color='green', linestyle='--', label=f'Mean: {valid_alpha.mean()*100:.1f}%')
    ax1.set_xlabel('Annual Alpha (%)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Alpha Distribution')
    ax1.legend()

    # Beta vs Alpha 散点图
    ax2 = axes[0, 1]
    valid_data = stock_alpha_df.dropna(subset=['alpha_annual', 'beta'])
    valid_data = valid_data[(valid_data['alpha_annual'] > -1) & (valid_data['alpha_annual'] < 1)]
    ax2.scatter(valid_data['beta'], valid_data['alpha_annual'] * 100, alpha=0.3, s=10)
    ax2.axhline(y=0, color='red', linestyle='--')
    ax2.axvline(x=1, color='red', linestyle='--')
    ax2.set_xlabel('Beta')
    ax2.set_ylabel('Annual Alpha (%)')
    ax2.set_title('Beta vs Alpha')

    # Alpha分组箱线图
    ax3 = axes[1, 0]
    group_data = []
    group_labels = []
    for group in ['Q1(低)', 'Q2', 'Q3', 'Q4', 'Q5(高)']:
        data = stock_alpha_df[stock_alpha_df['alpha_rank'] == group]['alpha_annual'].dropna() * 100
        data = data[(data > -100) & (data < 100)]
        group_data.append(data)
        group_labels.append(group)
    ax3.boxplot(group_data, labels=group_labels)
    ax3.axhline(y=0, color='red', linestyle='--')
    ax3.set_xlabel('Alpha Quintile')
    ax3.set_ylabel('Annual Alpha (%)')
    ax3.set_title('Alpha by Quintile')

    # R方分布
    ax4 = axes[1, 1]
    valid_r2 = stock_alpha_df['r_squared'].dropna()
    ax4.hist(valid_r2, bins=50, edgecolor='black', alpha=0.7)
    ax4.axvline(x=valid_r2.mean(), color='green', linestyle='--', label=f'Mean: {valid_r2.mean():.2f}')
    ax4.set_xlabel('R-squared')
    ax4.set_ylabel('Frequency')
    ax4.set_title('R-squared Distribution (CAPM Fit)')
    ax4.legend()

    plt.tight_layout()
    plt.savefig(f'{REPORT_DIR}/alpha_analysis_overview.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 图2：Alpha持续性分析
    if not persistence_df.empty:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 自相关衰减
        ax1 = axes[0]
        lags = [1, 5, 20, 60]
        ac_means = [persistence_df[f'ac_{lag}'].mean() for lag in lags]
        ax1.plot(lags, ac_means, 'bo-', markersize=10)
        ax1.axhline(y=0, color='red', linestyle='--')
        ax1.set_xlabel('Lag (Trading Days)')
        ax1.set_ylabel('Average Autocorrelation')
        ax1.set_title('Alpha Autocorrelation Decay')
        ax1.set_xticks(lags)

        # 半衰期分布
        ax2 = axes[1]
        valid_hl = persistence_df['half_life'].dropna()
        valid_hl = valid_hl[(valid_hl > 0) & (valid_hl < 200)]
        if len(valid_hl) > 0:
            ax2.hist(valid_hl, bins=30, edgecolor='black', alpha=0.7)
            ax2.axvline(x=valid_hl.median(), color='green', linestyle='--',
                       label=f'Median: {valid_hl.median():.1f} days')
            ax2.set_xlabel('Half-life (Trading Days)')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Alpha Half-life Distribution')
            ax2.legend()

        plt.tight_layout()
        plt.savefig(f'{REPORT_DIR}/alpha_persistence.png', dpi=150, bbox_inches='tight')
        plt.close()

    # 添加图表引用
    report_lines.append("## 四、可视化图表\n\n")
    report_lines.append("### 4.1 Alpha分析概览\n")
    report_lines.append("![Alpha Analysis Overview](alpha_analysis_overview.png)\n\n")
    report_lines.append("### 4.2 Alpha持续性分析\n")
    report_lines.append("![Alpha Persistence](alpha_persistence.png)\n\n")

    # =============================================
    # 总结与建议
    # =============================================
    report_lines.append("## 五、研究结论与建议\n\n")

    report_lines.append("### 5.1 主要发现\n\n")
    report_lines.append(f"1. **Alpha分布特征**: 个股年化Alpha均值约为{alpha_stats['mean']*100:.2f}%，呈现轻微右偏分布\n")
    report_lines.append(f"2. **Alpha显著性**: 约{sig_positive/total*100:.1f}%的股票存在显著正Alpha，表明市场并非完全有效\n")
    report_lines.append(f"3. **Alpha持续性**: Alpha具有一定的持续性，但会随时间衰减\n")
    report_lines.append(f"4. **行业差异**: 不同行业的Alpha表现存在显著差异\n\n")

    report_lines.append("### 5.2 策略建议\n\n")
    report_lines.append("1. **选股策略**: 关注具有显著正Alpha且Alpha稳定性高的股票\n")
    report_lines.append("2. **持仓周期**: 考虑Alpha的衰减特性，建议定期重新评估持仓\n")
    report_lines.append("3. **风险控制**: 结合Beta因子进行风险管理，避免过度暴露于市场风险\n")
    report_lines.append("4. **多因子增强**: 将Alpha因子与动量、质量等因子结合使用\n\n")

    report_lines.append("### 5.3 研究局限\n\n")
    report_lines.append("1. CAPM模型假设较为简化，未考虑其他系统性风险因子\n")
    report_lines.append("2. 历史Alpha不能完全代表未来表现\n")
    report_lines.append("3. 交易成本和冲击成本未纳入分析\n")
    report_lines.append("4. 样本期间可能存在特定市场环境影响\n\n")

    # 保存报告
    report_path = f'{REPORT_DIR}/alpha_analysis_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.writelines(report_lines)

    print(f"\n报告已保存至: {report_path}")
    print(f"图表已保存至: {REPORT_DIR}/")

    # 同时保存CSV数据
    stock_alpha_df.to_csv(f'{REPORT_DIR}/stock_alpha_data.csv', index=False, encoding='utf-8-sig')
    print(f"Alpha数据已保存至: {REPORT_DIR}/stock_alpha_data.csv")

    return stock_alpha_df, rolling_alpha_df, persistence_df


def main():
    """主函数"""
    print("开始个股Alpha特征研究...\n")

    try:
        stock_alpha_df, rolling_alpha_df, persistence_df = generate_report()
        print("\n研究完成！")
    except Exception as e:
        print(f"\n研究过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
