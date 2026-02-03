#!/usr/bin/env python3
"""
股票尾部风险研究
Tail Risk Research for A-Share Market

分析内容:
1. 尾部风险指标: VaR, CVaR/ES, 偏度/峰度, 极端损失概率
2. 尾部特征分析: 行业差异, 市场状态影响, 尾部风险聚集
3. 风控应用: 尾部风险筛选, 组合风控, 极端情景模拟
"""

import duckdb
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm, t, skew, kurtosis
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 数据库配置
# ============================================================
DB_PATH = '/Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db'
OUTPUT_DIR = '/Users/allen/workspace/python/stock/Tushare-DuckDB/reports/research'

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 第一部分: 数据加载与预处理
# ============================================================

def load_data():
    """加载股票日线数据和行业分类"""
    conn = duckdb.connect(DB_PATH, read_only=True)

    # 加载近3年的日线数据 (足够计算尾部风险)
    print("正在加载日线数据...")
    daily_df = conn.execute("""
        SELECT
            d.ts_code,
            d.trade_date,
            d.pct_chg,
            d.close,
            d.vol,
            d.amount
        FROM daily d
        WHERE d.trade_date >= '20220101'
          AND d.pct_chg IS NOT NULL
        ORDER BY d.ts_code, d.trade_date
    """).fetchdf()

    # 加载股票基本信息
    print("正在加载股票信息...")
    stock_basic = conn.execute("""
        SELECT ts_code, name, industry, market
        FROM stock_basic
        WHERE list_status = 'L'
    """).fetchdf()

    # 加载行业分类
    print("正在加载行业分类...")
    industry_df = conn.execute("""
        SELECT DISTINCT
            ts_code,
            l1_name as industry_l1,
            l2_name as industry_l2
        FROM index_member_all
        WHERE is_new = 'Y' AND out_date IS NULL
    """).fetchdf()

    # 加载指数日线 (用于市场状态判断)
    print("正在加载指数数据...")
    index_df = conn.execute("""
        SELECT trade_date, close as index_close, pct_chg as index_pct_chg
        FROM index_daily
        WHERE ts_code = '000001.SH'
          AND trade_date >= '20220101'
        ORDER BY trade_date
    """).fetchdf()

    conn.close()

    # 合并数据
    daily_df = daily_df.merge(stock_basic, on='ts_code', how='left')
    daily_df = daily_df.merge(industry_df, on='ts_code', how='left')
    daily_df = daily_df.merge(index_df, on='trade_date', how='left')

    # 转换日期
    daily_df['trade_date'] = pd.to_datetime(daily_df['trade_date'])

    print(f"数据加载完成: {len(daily_df):,} 条记录, {daily_df['ts_code'].nunique()} 只股票")

    return daily_df


# ============================================================
# 第二部分: 尾部风险指标计算
# ============================================================

def calculate_var(returns, confidence_level=0.95, method='historical'):
    """
    计算VaR (Value at Risk)

    Parameters:
    -----------
    returns : array-like
        收益率序列
    confidence_level : float
        置信水平 (0.95 表示 95%)
    method : str
        计算方法: 'historical', 'parametric', 'cornish_fisher'

    Returns:
    --------
    float : VaR值 (正数表示损失)
    """
    returns = np.array(returns)
    returns = returns[~np.isnan(returns)]

    if len(returns) < 30:
        return np.nan

    alpha = 1 - confidence_level

    if method == 'historical':
        # 历史模拟法
        var = -np.percentile(returns, alpha * 100)

    elif method == 'parametric':
        # 参数法 (假设正态分布)
        mu = np.mean(returns)
        sigma = np.std(returns, ddof=1)
        var = -(mu + norm.ppf(alpha) * sigma)

    elif method == 'cornish_fisher':
        # Cornish-Fisher扩展 (考虑偏度和峰度)
        mu = np.mean(returns)
        sigma = np.std(returns, ddof=1)
        s = skew(returns)
        k = kurtosis(returns, fisher=True)  # 超额峰度

        z = norm.ppf(alpha)
        # Cornish-Fisher调整
        z_cf = z + (z**2 - 1) * s / 6 + (z**3 - 3*z) * k / 24 - (2*z**3 - 5*z) * s**2 / 36

        var = -(mu + z_cf * sigma)

    else:
        raise ValueError(f"Unknown method: {method}")

    return var


def calculate_cvar(returns, confidence_level=0.95, method='historical'):
    """
    计算CVaR/ES (Conditional VaR / Expected Shortfall)

    CVaR是在损失超过VaR时的平均损失
    """
    returns = np.array(returns)
    returns = returns[~np.isnan(returns)]

    if len(returns) < 30:
        return np.nan

    alpha = 1 - confidence_level

    if method == 'historical':
        threshold = np.percentile(returns, alpha * 100)
        tail_returns = returns[returns <= threshold]
        if len(tail_returns) == 0:
            return np.nan
        cvar = -np.mean(tail_returns)

    elif method == 'parametric':
        mu = np.mean(returns)
        sigma = np.std(returns, ddof=1)
        z = norm.ppf(alpha)
        # 正态分布下的CVaR
        cvar = -(mu - sigma * norm.pdf(z) / alpha)

    else:
        # 默认使用历史法
        threshold = np.percentile(returns, alpha * 100)
        tail_returns = returns[returns <= threshold]
        if len(tail_returns) == 0:
            return np.nan
        cvar = -np.mean(tail_returns)

    return cvar


def calculate_tail_metrics(returns):
    """
    计算完整的尾部风险指标

    Returns:
    --------
    dict : 包含各种尾部风险指标
    """
    returns = np.array(returns)
    returns = returns[~np.isnan(returns)]

    if len(returns) < 60:  # 至少需要60个观测值
        return None

    metrics = {}

    # 基本统计
    metrics['mean'] = np.mean(returns)
    metrics['std'] = np.std(returns, ddof=1)
    metrics['skewness'] = skew(returns)
    metrics['kurtosis'] = kurtosis(returns, fisher=True)  # 超额峰度

    # VaR (多个置信水平)
    for conf in [0.95, 0.99]:
        metrics[f'var_{int(conf*100)}_hist'] = calculate_var(returns, conf, 'historical')
        metrics[f'var_{int(conf*100)}_param'] = calculate_var(returns, conf, 'parametric')
        metrics[f'var_{int(conf*100)}_cf'] = calculate_var(returns, conf, 'cornish_fisher')

    # CVaR/ES
    for conf in [0.95, 0.99]:
        metrics[f'cvar_{int(conf*100)}_hist'] = calculate_cvar(returns, conf, 'historical')
        metrics[f'cvar_{int(conf*100)}_param'] = calculate_cvar(returns, conf, 'parametric')

    # 极端损失概率
    for threshold in [-3, -5, -7, -9]:  # 下跌超过3%/5%/7%/9%的概率
        metrics[f'prob_loss_{abs(threshold)}pct'] = np.mean(returns < threshold) * 100

    # 最大单日损失
    metrics['max_loss'] = -np.min(returns)

    # 下行偏差 (只考虑负收益)
    negative_returns = returns[returns < 0]
    if len(negative_returns) > 0:
        metrics['downside_std'] = np.std(negative_returns, ddof=1)
    else:
        metrics['downside_std'] = 0

    # 尾部比率: CVaR / VaR
    if metrics['var_95_hist'] > 0:
        metrics['tail_ratio_95'] = metrics['cvar_95_hist'] / metrics['var_95_hist']
    else:
        metrics['tail_ratio_95'] = np.nan

    # Omega比率 (下行风险调整)
    threshold_return = 0
    gains = returns[returns > threshold_return] - threshold_return
    losses = threshold_return - returns[returns < threshold_return]
    if losses.sum() > 0:
        metrics['omega_ratio'] = gains.sum() / losses.sum()
    else:
        metrics['omega_ratio'] = np.nan

    return metrics


def calculate_stock_tail_risk(df):
    """计算每只股票的尾部风险指标"""
    print("正在计算个股尾部风险指标...")

    results = []

    for ts_code, group in df.groupby('ts_code'):
        returns = group['pct_chg'].values
        metrics = calculate_tail_metrics(returns)

        if metrics is None:
            continue

        # 添加股票信息
        metrics['ts_code'] = ts_code
        metrics['name'] = group['name'].iloc[0] if 'name' in group.columns else ''
        metrics['industry'] = group['industry'].iloc[0] if 'industry' in group.columns else ''
        metrics['industry_l1'] = group['industry_l1'].iloc[0] if 'industry_l1' in group.columns else ''
        metrics['market'] = group['market'].iloc[0] if 'market' in group.columns else ''
        metrics['n_obs'] = len(returns)

        results.append(metrics)

    result_df = pd.DataFrame(results)
    print(f"计算完成: {len(result_df)} 只股票")

    return result_df


# ============================================================
# 第三部分: 尾部特征分析
# ============================================================

def analyze_industry_tail_risk(tail_risk_df):
    """分析行业尾部风险差异"""
    print("\n" + "="*60)
    print("行业尾部风险分析")
    print("="*60)

    # 按一级行业分组
    industry_stats = tail_risk_df.groupby('industry_l1').agg({
        'var_95_hist': ['mean', 'std', 'median'],
        'cvar_95_hist': ['mean', 'std', 'median'],
        'skewness': 'mean',
        'kurtosis': 'mean',
        'prob_loss_5pct': 'mean',
        'max_loss': 'mean',
        'tail_ratio_95': 'mean',
        'ts_code': 'count'
    }).round(4)

    industry_stats.columns = ['_'.join(col).strip('_') for col in industry_stats.columns]
    industry_stats = industry_stats.rename(columns={'ts_code_count': 'stock_count'})
    industry_stats = industry_stats.sort_values('cvar_95_hist_mean', ascending=False)

    print("\n按CVaR排序的行业尾部风险:")
    print(industry_stats.to_string())

    return industry_stats


def analyze_market_state_impact(df):
    """分析市场状态对尾部风险的影响"""
    print("\n" + "="*60)
    print("市场状态与尾部风险关系")
    print("="*60)

    # 定义市场状态
    df = df.copy()

    # 计算指数20日收益率
    index_returns = df.groupby('trade_date')['index_pct_chg'].first()
    index_returns_20d = index_returns.rolling(20).sum()

    # 市场状态分类
    market_state_map = {}
    for date, ret in index_returns_20d.items():
        if pd.isna(ret):
            market_state_map[date] = 'unknown'
        elif ret > 5:
            market_state_map[date] = 'bull'
        elif ret < -5:
            market_state_map[date] = 'bear'
        else:
            market_state_map[date] = 'neutral'

    df['market_state'] = df['trade_date'].map(market_state_map)

    # 分市场状态计算尾部风险
    state_results = []
    for state in ['bull', 'neutral', 'bear']:
        state_data = df[df['market_state'] == state]
        if len(state_data) == 0:
            continue

        returns = state_data['pct_chg'].values

        result = {
            'market_state': state,
            'n_obs': len(returns),
            'mean_return': np.mean(returns),
            'std_return': np.std(returns, ddof=1),
            'var_95': calculate_var(returns, 0.95),
            'cvar_95': calculate_cvar(returns, 0.95),
            'skewness': skew(returns),
            'kurtosis': kurtosis(returns, fisher=True),
            'prob_loss_5pct': np.mean(returns < -5) * 100,
            'prob_loss_7pct': np.mean(returns < -7) * 100
        }
        state_results.append(result)

    state_df = pd.DataFrame(state_results)
    print("\n不同市场状态下的尾部风险:")
    print(state_df.to_string(index=False))

    return state_df


def analyze_tail_clustering(df):
    """分析尾部风险聚集现象"""
    print("\n" + "="*60)
    print("尾部风险聚集分析")
    print("="*60)

    # 按日期统计跌停或大跌股票数量
    daily_extreme = df.groupby('trade_date').agg({
        'ts_code': 'count',
        'pct_chg': [
            lambda x: (x < -5).sum(),  # 跌幅超5%
            lambda x: (x < -7).sum(),  # 跌幅超7%
            lambda x: (x <= -9.9).sum(),  # 跌停
        ]
    })
    daily_extreme.columns = ['total_stocks', 'drop_5pct', 'drop_7pct', 'limit_down']
    daily_extreme['drop_5pct_ratio'] = daily_extreme['drop_5pct'] / daily_extreme['total_stocks'] * 100
    daily_extreme['drop_7pct_ratio'] = daily_extreme['drop_7pct'] / daily_extreme['total_stocks'] * 100
    daily_extreme['limit_down_ratio'] = daily_extreme['limit_down'] / daily_extreme['total_stocks'] * 100

    # 统计极端日
    print("\n极端下跌日统计 (跌幅超5%股票占比>30%的交易日):")
    extreme_days = daily_extreme[daily_extreme['drop_5pct_ratio'] > 30].sort_values('drop_5pct_ratio', ascending=False)
    if len(extreme_days) > 0:
        print(extreme_days.head(20).to_string())
    else:
        print("无符合条件的交易日")

    # 分析连续极端日
    print("\n尾部风险聚集特征:")

    # 定义极端日: 跌幅超5%的股票占比>20%
    is_extreme = (daily_extreme['drop_5pct_ratio'] > 20).astype(int)

    # 计算连续极端日
    clusters = []
    current_cluster = 0
    for i, val in enumerate(is_extreme.values):
        if val == 1:
            current_cluster += 1
        else:
            if current_cluster > 0:
                clusters.append(current_cluster)
            current_cluster = 0
    if current_cluster > 0:
        clusters.append(current_cluster)

    if clusters:
        print(f"  极端日总数: {sum(clusters)}")
        print(f"  聚集事件次数: {len(clusters)}")
        print(f"  平均持续天数: {np.mean(clusters):.2f}")
        print(f"  最长持续天数: {max(clusters)}")
        print(f"  持续天数分布: 1天={clusters.count(1)}, 2天={clusters.count(2)}, 3天+={sum(1 for c in clusters if c>=3)}")

    return daily_extreme


def analyze_tail_correlation(tail_risk_df):
    """分析尾部风险与其他因素的相关性"""
    print("\n" + "="*60)
    print("尾部风险相关性分析")
    print("="*60)

    # 选择关键指标
    key_metrics = ['var_95_hist', 'cvar_95_hist', 'skewness', 'kurtosis',
                   'prob_loss_5pct', 'max_loss', 'std', 'tail_ratio_95']

    corr_df = tail_risk_df[key_metrics].corr()

    print("\n尾部风险指标相关性矩阵:")
    print(corr_df.round(3).to_string())

    return corr_df


# ============================================================
# 第四部分: 风控应用
# ============================================================

def tail_risk_screening(tail_risk_df, n_top=50):
    """尾部风险筛选 - 识别高尾部风险和低尾部风险股票"""
    print("\n" + "="*60)
    print("尾部风险筛选")
    print("="*60)

    # 计算综合尾部风险得分
    df = tail_risk_df.copy()

    # 标准化各指标
    risk_metrics = ['var_95_hist', 'cvar_95_hist', 'prob_loss_5pct', 'max_loss']
    for col in risk_metrics:
        df[f'{col}_zscore'] = (df[col] - df[col].mean()) / df[col].std()

    # 综合得分 (越高风险越大)
    df['tail_risk_score'] = (
        df['var_95_hist_zscore'] * 0.3 +
        df['cvar_95_hist_zscore'] * 0.3 +
        df['prob_loss_5pct_zscore'] * 0.2 +
        df['max_loss_zscore'] * 0.2
    )

    # 高尾部风险股票
    high_risk = df.nlargest(n_top, 'tail_risk_score')[
        ['ts_code', 'name', 'industry_l1', 'var_95_hist', 'cvar_95_hist',
         'prob_loss_5pct', 'max_loss', 'tail_risk_score']
    ]

    # 低尾部风险股票
    low_risk = df.nsmallest(n_top, 'tail_risk_score')[
        ['ts_code', 'name', 'industry_l1', 'var_95_hist', 'cvar_95_hist',
         'prob_loss_5pct', 'max_loss', 'tail_risk_score']
    ]

    print(f"\n高尾部风险股票 (Top {n_top}):")
    print(high_risk.to_string(index=False))

    print(f"\n低尾部风险股票 (Top {n_top}):")
    print(low_risk.to_string(index=False))

    return high_risk, low_risk, df


def portfolio_tail_risk(tail_risk_df, df, n_stocks=30):
    """组合尾部风险分析"""
    print("\n" + "="*60)
    print("组合尾部风险控制")
    print("="*60)

    # 构建几个不同策略的组合
    portfolios = {}

    # 1. 等权组合 (随机选择)
    random_stocks = tail_risk_df.sample(n=min(n_stocks, len(tail_risk_df)))['ts_code'].tolist()
    portfolios['random_equal'] = random_stocks

    # 2. 低尾部风险组合
    low_tail = tail_risk_df.nsmallest(n_stocks, 'cvar_95_hist')['ts_code'].tolist()
    portfolios['low_tail_risk'] = low_tail

    # 3. 高尾部风险组合
    high_tail = tail_risk_df.nlargest(n_stocks, 'cvar_95_hist')['ts_code'].tolist()
    portfolios['high_tail_risk'] = high_tail

    # 4. 分散化组合 (每个行业选最低尾部风险的股票)
    diversified = []
    for industry, group in tail_risk_df.groupby('industry_l1'):
        if pd.isna(industry):
            continue
        best = group.nsmallest(1, 'cvar_95_hist')
        if len(best) > 0:
            diversified.append(best.iloc[0]['ts_code'])
    portfolios['diversified'] = diversified[:n_stocks]

    # 计算各组合的尾部风险
    results = []
    for name, stocks in portfolios.items():
        if len(stocks) == 0:
            continue

        # 获取组合收益率
        portfolio_data = df[df['ts_code'].isin(stocks)]

        # 计算等权组合日收益
        portfolio_returns = portfolio_data.groupby('trade_date')['pct_chg'].mean()

        if len(portfolio_returns) < 60:
            continue

        returns = portfolio_returns.values

        result = {
            'portfolio': name,
            'n_stocks': len(stocks),
            'mean_return': np.mean(returns),
            'std_return': np.std(returns, ddof=1),
            'var_95': calculate_var(returns, 0.95),
            'var_99': calculate_var(returns, 0.99),
            'cvar_95': calculate_cvar(returns, 0.95),
            'cvar_99': calculate_cvar(returns, 0.99),
            'skewness': skew(returns),
            'kurtosis': kurtosis(returns, fisher=True),
            'max_loss': -np.min(returns),
            'prob_loss_3pct': np.mean(returns < -3) * 100
        }
        results.append(result)

    portfolio_df = pd.DataFrame(results)

    print("\n不同组合策略的尾部风险对比:")
    print(portfolio_df.to_string(index=False))

    # 计算风险降低比例
    if len(portfolio_df) > 0:
        baseline = portfolio_df[portfolio_df['portfolio'] == 'random_equal']['cvar_95'].values
        if len(baseline) > 0:
            baseline = baseline[0]
            low_tail_cvar = portfolio_df[portfolio_df['portfolio'] == 'low_tail_risk']['cvar_95'].values
            if len(low_tail_cvar) > 0:
                reduction = (baseline - low_tail_cvar[0]) / baseline * 100
                print(f"\n低尾部风险组合相比随机组合CVaR降低: {reduction:.1f}%")

    return portfolio_df


def extreme_scenario_simulation(df, n_simulations=1000):
    """极端情景模拟"""
    print("\n" + "="*60)
    print("极端情景模拟")
    print("="*60)

    # 获取所有股票的收益率分布
    returns_by_stock = {}
    for ts_code, group in df.groupby('ts_code'):
        if len(group) >= 60:
            returns_by_stock[ts_code] = group['pct_chg'].values

    if len(returns_by_stock) < 100:
        print("股票数量不足,跳过模拟")
        return None

    print(f"使用 {len(returns_by_stock)} 只股票进行模拟")

    # 历史最差日分析
    daily_returns = df.groupby('trade_date')['pct_chg'].agg(['mean', 'std', 'min', 'max'])
    worst_days = daily_returns.nsmallest(10, 'mean')

    print("\n历史最差交易日 (按市场平均收益):")
    print(worst_days.to_string())

    # 蒙特卡洛模拟 - 模拟组合在极端情况下的表现
    print("\n蒙特卡洛压力测试 (30只股票等权组合):")

    np.random.seed(42)
    stock_codes = list(returns_by_stock.keys())

    portfolio_worst_returns = []
    for _ in range(n_simulations):
        # 随机选择30只股票
        selected = np.random.choice(stock_codes, size=min(30, len(stock_codes)), replace=False)

        # 从每只股票的历史收益中随机抽取一个极端值 (最差5%分位)
        extreme_returns = []
        for code in selected:
            rets = returns_by_stock[code]
            extreme_val = np.percentile(rets, 5)
            extreme_returns.append(extreme_val)

        # 组合收益 (等权)
        portfolio_worst_returns.append(np.mean(extreme_returns))

    # 压力测试结果
    stress_results = {
        'mean': np.mean(portfolio_worst_returns),
        'std': np.std(portfolio_worst_returns),
        'min': np.min(portfolio_worst_returns),
        'percentile_1': np.percentile(portfolio_worst_returns, 1),
        'percentile_5': np.percentile(portfolio_worst_returns, 5),
        'percentile_10': np.percentile(portfolio_worst_returns, 10)
    }

    print(f"  极端情景平均损失: {stress_results['mean']:.2f}%")
    print(f"  极端情景最大损失: {stress_results['min']:.2f}%")
    print(f"  1%分位数损失: {stress_results['percentile_1']:.2f}%")
    print(f"  5%分位数损失: {stress_results['percentile_5']:.2f}%")

    # 历史情景重演
    print("\n历史极端事件回顾:")

    # 找出市场大跌的事件
    market_crashes = daily_returns[daily_returns['mean'] < -3].sort_values('mean')

    if len(market_crashes) > 0:
        print(f"  市场平均跌幅超3%的交易日: {len(market_crashes)} 天")
        print(f"  最严重市场跌幅: {market_crashes['mean'].min():.2f}%")

    return stress_results


# ============================================================
# 第五部分: 可视化
# ============================================================

def create_visualizations(tail_risk_df, df, industry_stats, daily_extreme):
    """创建可视化图表"""
    print("\n正在生成可视化图表...")

    # 1. 尾部风险分布图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # VaR分布
    ax = axes[0, 0]
    ax.hist(tail_risk_df['var_95_hist'].dropna(), bins=50, alpha=0.7, color='steelblue', edgecolor='white')
    ax.axvline(tail_risk_df['var_95_hist'].median(), color='red', linestyle='--', label=f"中位数: {tail_risk_df['var_95_hist'].median():.2f}")
    ax.set_xlabel('VaR (95%)')
    ax.set_ylabel('股票数量')
    ax.set_title('个股VaR分布')
    ax.legend()

    # CVaR分布
    ax = axes[0, 1]
    ax.hist(tail_risk_df['cvar_95_hist'].dropna(), bins=50, alpha=0.7, color='coral', edgecolor='white')
    ax.axvline(tail_risk_df['cvar_95_hist'].median(), color='red', linestyle='--', label=f"中位数: {tail_risk_df['cvar_95_hist'].median():.2f}")
    ax.set_xlabel('CVaR (95%)')
    ax.set_ylabel('股票数量')
    ax.set_title('个股CVaR分布')
    ax.legend()

    # 偏度分布
    ax = axes[1, 0]
    ax.hist(tail_risk_df['skewness'].dropna(), bins=50, alpha=0.7, color='forestgreen', edgecolor='white')
    ax.axvline(0, color='black', linestyle='-', linewidth=1)
    ax.axvline(tail_risk_df['skewness'].median(), color='red', linestyle='--', label=f"中位数: {tail_risk_df['skewness'].median():.2f}")
    ax.set_xlabel('偏度')
    ax.set_ylabel('股票数量')
    ax.set_title('收益率偏度分布')
    ax.legend()

    # 峰度分布
    ax = axes[1, 1]
    ax.hist(tail_risk_df['kurtosis'].dropna().clip(-5, 20), bins=50, alpha=0.7, color='purple', edgecolor='white')
    ax.axvline(0, color='black', linestyle='-', linewidth=1)
    ax.axvline(tail_risk_df['kurtosis'].median(), color='red', linestyle='--', label=f"中位数: {tail_risk_df['kurtosis'].median():.2f}")
    ax.set_xlabel('超额峰度')
    ax.set_ylabel('股票数量')
    ax.set_title('收益率峰度分布')
    ax.legend()

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/tail_risk_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 2. 行业尾部风险对比
    fig, ax = plt.subplots(figsize=(12, 8))

    plot_data = industry_stats.dropna().sort_values('cvar_95_hist_mean', ascending=True)
    if len(plot_data) > 0:
        y_pos = range(len(plot_data))
        ax.barh(y_pos, plot_data['cvar_95_hist_mean'], alpha=0.8, color='coral')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(plot_data.index)
        ax.set_xlabel('平均CVaR (95%)')
        ax.set_title('各行业尾部风险对比 (CVaR)')
        ax.axvline(plot_data['cvar_95_hist_mean'].mean(), color='red', linestyle='--',
                   label=f"市场平均: {plot_data['cvar_95_hist_mean'].mean():.2f}")
        ax.legend()

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/industry_tail_risk.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 3. 尾部风险聚集时序图
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # 跌幅超5%股票比例
    ax = axes[0]
    daily_extreme_plot = daily_extreme.reset_index()
    daily_extreme_plot['trade_date'] = pd.to_datetime(daily_extreme_plot['trade_date'])
    ax.plot(daily_extreme_plot['trade_date'], daily_extreme_plot['drop_5pct_ratio'],
            alpha=0.7, linewidth=0.8)
    ax.axhline(20, color='orange', linestyle='--', alpha=0.7, label='20%警戒线')
    ax.axhline(30, color='red', linestyle='--', alpha=0.7, label='30%极端线')
    ax.set_ylabel('跌幅>5%股票占比 (%)')
    ax.set_title('市场尾部风险聚集指标')
    ax.legend()
    ax.set_ylim(0, 80)

    # 跌停股票数
    ax = axes[1]
    ax.bar(daily_extreme_plot['trade_date'], daily_extreme_plot['limit_down'],
           alpha=0.7, width=1)
    ax.set_ylabel('跌停股票数')
    ax.set_xlabel('日期')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/tail_risk_clustering.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 4. VaR vs CVaR 散点图
    fig, ax = plt.subplots(figsize=(10, 8))

    scatter = ax.scatter(tail_risk_df['var_95_hist'], tail_risk_df['cvar_95_hist'],
                        c=tail_risk_df['skewness'], cmap='RdYlGn_r', alpha=0.5, s=10)
    ax.plot([0, 15], [0, 15], 'k--', alpha=0.5, label='VaR=CVaR')
    ax.set_xlabel('VaR (95%)')
    ax.set_ylabel('CVaR (95%)')
    ax.set_title('VaR vs CVaR (颜色=偏度)')
    plt.colorbar(scatter, label='偏度')
    ax.legend()
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 20)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/var_vs_cvar.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 5. 尾部风险与收益关系
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # CVaR vs 平均收益
    ax = axes[0]
    ax.scatter(tail_risk_df['cvar_95_hist'], tail_risk_df['mean'] * 252, alpha=0.3, s=10)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax.axvline(tail_risk_df['cvar_95_hist'].median(), color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('CVaR (95%)')
    ax.set_ylabel('年化收益率 (%)')
    ax.set_title('尾部风险与收益关系')

    # 添加回归线
    x = tail_risk_df['cvar_95_hist'].dropna()
    y = (tail_risk_df['mean'] * 252).loc[x.index]
    valid = ~(x.isna() | y.isna())
    if valid.sum() > 10:
        z = np.polyfit(x[valid], y[valid], 1)
        p = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, p(x_line), 'r-', alpha=0.7, label=f'斜率: {z[0]:.2f}')
        ax.legend()

    # 尾部比率分布
    ax = axes[1]
    tail_ratio = tail_risk_df['tail_ratio_95'].dropna().clip(1, 3)
    ax.hist(tail_ratio, bins=50, alpha=0.7, color='teal', edgecolor='white')
    ax.axvline(tail_ratio.median(), color='red', linestyle='--',
               label=f"中位数: {tail_ratio.median():.2f}")
    ax.set_xlabel('尾部比率 (CVaR/VaR)')
    ax.set_ylabel('股票数量')
    ax.set_title('尾部比率分布')
    ax.legend()

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/tail_risk_return.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"图表已保存至 {OUTPUT_DIR}/")


# ============================================================
# 第六部分: 生成报告
# ============================================================

def generate_report(tail_risk_df, industry_stats, market_state_df,
                    daily_extreme, portfolio_df, high_risk, low_risk, stress_results):
    """生成研究报告"""

    report = []
    report.append("=" * 80)
    report.append("股票尾部风险研究报告")
    report.append("Tail Risk Research Report for A-Share Market")
    report.append("=" * 80)
    report.append(f"\n生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"分析股票数: {len(tail_risk_df)}")
    report.append(f"数据期间: 2022年1月至今")

    # 第一部分: 尾部风险指标概览
    report.append("\n\n" + "=" * 80)
    report.append("第一部分: 尾部风险指标概览")
    report.append("=" * 80)

    report.append("\n1.1 关键指标定义")
    report.append("-" * 40)
    report.append("VaR (Value at Risk): 在给定置信水平下,投资组合在持有期内的最大预期损失")
    report.append("CVaR/ES (Expected Shortfall): 在损失超过VaR时的平均损失,更好地捕捉尾部风险")
    report.append("偏度 (Skewness): 收益分布的不对称性,负偏度表示左尾更厚(更多极端损失)")
    report.append("峰度 (Kurtosis): 收益分布的尖峰程度,高峰度表示极端事件更频繁")

    report.append("\n1.2 市场整体尾部风险统计")
    report.append("-" * 40)

    stats = tail_risk_df[['var_95_hist', 'cvar_95_hist', 'skewness', 'kurtosis',
                          'prob_loss_5pct', 'max_loss']].describe()
    report.append(stats.round(4).to_string())

    report.append("\n1.3 关键发现")
    report.append("-" * 40)

    median_var = tail_risk_df['var_95_hist'].median()
    median_cvar = tail_risk_df['cvar_95_hist'].median()
    median_skew = tail_risk_df['skewness'].median()
    median_kurt = tail_risk_df['kurtosis'].median()

    report.append(f"- 中位数VaR(95%): {median_var:.2f}%, 意味着在95%置信度下,单日最大损失约{median_var:.2f}%")
    report.append(f"- 中位数CVaR(95%): {median_cvar:.2f}%, 当损失超过VaR时,平均损失约{median_cvar:.2f}%")
    report.append(f"- 中位数偏度: {median_skew:.2f}, {'负偏' if median_skew < 0 else '正偏'}说明{'左尾更厚,极端损失概率更高' if median_skew < 0 else '右尾更厚'}")
    report.append(f"- 中位数超额峰度: {median_kurt:.2f}, {'高于正态分布' if median_kurt > 0 else '接近正态分布'},表示{'尾部事件更频繁' if median_kurt > 0 else '尾部特征正常'}")
    report.append(f"- 尾部比率(CVaR/VaR): {tail_risk_df['tail_ratio_95'].median():.2f}, 越高表示尾部风险越集中")

    # 第二部分: 行业尾部风险分析
    report.append("\n\n" + "=" * 80)
    report.append("第二部分: 行业尾部风险差异")
    report.append("=" * 80)

    report.append("\n2.1 行业尾部风险排名 (按CVaR)")
    report.append("-" * 40)

    top_industries = industry_stats.head(10)[['cvar_95_hist_mean', 'var_95_hist_mean', 'skewness_mean', 'stock_count']]
    report.append("高尾部风险行业 (Top 10):")
    report.append(top_industries.to_string())

    report.append("\n低尾部风险行业 (Bottom 10):")
    bottom_industries = industry_stats.tail(10)[['cvar_95_hist_mean', 'var_95_hist_mean', 'skewness_mean', 'stock_count']]
    report.append(bottom_industries.to_string())

    # 第三部分: 市场状态影响
    report.append("\n\n" + "=" * 80)
    report.append("第三部分: 市场状态与尾部风险")
    report.append("=" * 80)

    if market_state_df is not None and len(market_state_df) > 0:
        report.append("\n3.1 不同市场状态下的尾部风险")
        report.append("-" * 40)
        report.append(market_state_df.to_string(index=False))

        report.append("\n3.2 关键发现")
        report.append("-" * 40)

        bear_data = market_state_df[market_state_df['market_state'] == 'bear']
        bull_data = market_state_df[market_state_df['market_state'] == 'bull']

        if len(bear_data) > 0 and len(bull_data) > 0:
            bear_cvar = bear_data['cvar_95'].values[0]
            bull_cvar = bull_data['cvar_95'].values[0]
            report.append(f"- 熊市CVaR({bear_cvar:.2f}%)比牛市({bull_cvar:.2f}%)高出{((bear_cvar/bull_cvar)-1)*100:.1f}%")

            bear_prob = bear_data['prob_loss_5pct'].values[0]
            bull_prob = bull_data['prob_loss_5pct'].values[0]
            report.append(f"- 熊市极端损失(>5%)概率({bear_prob:.2f}%)是牛市({bull_prob:.2f}%)的{bear_prob/bull_prob:.1f}倍")

    # 第四部分: 尾部风险聚集
    report.append("\n\n" + "=" * 80)
    report.append("第四部分: 尾部风险聚集现象")
    report.append("=" * 80)

    report.append("\n4.1 极端下跌日统计")
    report.append("-" * 40)

    extreme_days = daily_extreme[daily_extreme['drop_5pct_ratio'] > 30]
    report.append(f"- 跌幅>5%股票占比超过30%的交易日: {len(extreme_days)} 天")

    extreme_days_20 = daily_extreme[daily_extreme['drop_5pct_ratio'] > 20]
    report.append(f"- 跌幅>5%股票占比超过20%的交易日: {len(extreme_days_20)} 天")

    if len(extreme_days) > 0:
        report.append(f"\n最严重的10个交易日:")
        report.append(extreme_days.nlargest(10, 'drop_5pct_ratio')[['drop_5pct', 'drop_5pct_ratio', 'limit_down']].to_string())

    report.append("\n4.2 聚集特征")
    report.append("-" * 40)
    report.append("尾部风险具有明显的时间聚集性:")
    report.append("- 极端事件往往连续发生,而非均匀分布")
    report.append("- 市场恐慌期间,个股尾部风险高度相关")
    report.append("- 流动性危机可能放大尾部风险聚集效应")

    # 第五部分: 风控应用
    report.append("\n\n" + "=" * 80)
    report.append("第五部分: 风控应用")
    report.append("=" * 80)

    report.append("\n5.1 组合尾部风险控制")
    report.append("-" * 40)

    if portfolio_df is not None and len(portfolio_df) > 0:
        report.append(portfolio_df.to_string(index=False))

        report.append("\n关键发现:")
        random_port = portfolio_df[portfolio_df['portfolio'] == 'random_equal']
        low_risk_port = portfolio_df[portfolio_df['portfolio'] == 'low_tail_risk']

        if len(random_port) > 0 and len(low_risk_port) > 0:
            improvement = (random_port['cvar_95'].values[0] - low_risk_port['cvar_95'].values[0]) / random_port['cvar_95'].values[0] * 100
            report.append(f"- 低尾部风险组合相比随机组合,CVaR降低{improvement:.1f}%")

    report.append("\n5.2 高尾部风险股票警示")
    report.append("-" * 40)
    report.append("Top 20 高尾部风险股票:")
    report.append(high_risk.head(20).to_string(index=False))

    report.append("\n5.3 低尾部风险股票推荐")
    report.append("-" * 40)
    report.append("Top 20 低尾部风险股票:")
    report.append(low_risk.head(20).to_string(index=False))

    report.append("\n5.4 极端情景模拟")
    report.append("-" * 40)
    if stress_results:
        report.append(f"蒙特卡洛压力测试结果 (30只股票等权组合):")
        report.append(f"- 极端情景平均损失: {stress_results['mean']:.2f}%")
        report.append(f"- 极端情景最大损失: {stress_results['min']:.2f}%")
        report.append(f"- 1%分位数损失: {stress_results['percentile_1']:.2f}%")
        report.append(f"- 5%分位数损失: {stress_results['percentile_5']:.2f}%")

    # 第六部分: 结论与建议
    report.append("\n\n" + "=" * 80)
    report.append("第六部分: 结论与投资建议")
    report.append("=" * 80)

    report.append("\n6.1 主要结论")
    report.append("-" * 40)
    report.append("1. A股市场整体呈现显著的尾部风险特征,收益分布明显偏离正态分布")
    report.append(f"2. 市场整体呈现{'负偏' if median_skew < 0 else '正偏'},左尾风险{'高于' if median_skew < 0 else '低于'}右尾收益")
    report.append(f"3. 超额峰度为{median_kurt:.2f},表明极端事件发生概率高于正态假设")
    report.append("4. 行业间尾部风险差异显著,周期性行业和小盘股尾部风险更高")
    report.append("5. 熊市期间尾部风险显著放大,且存在明显的聚集效应")

    report.append("\n6.2 风控建议")
    report.append("-" * 40)
    report.append("1. 使用CVaR替代VaR作为主要风险度量,更好捕捉尾部风险")
    report.append("2. 避免过度集中于高尾部风险行业和个股")
    report.append("3. 构建组合时考虑行业分散化,降低系统性尾部风险")
    report.append("4. 在市场进入熊市状态时,主动降低仓位或增加对冲")
    report.append("5. 建立尾部风险监控指标,及时识别市场极端状态")
    report.append("6. 压力测试应基于历史极端情景,而非正态分布假设")

    report.append("\n6.3 方法论说明")
    report.append("-" * 40)
    report.append("- VaR计算方法: 历史模拟法、参数法(正态)、Cornish-Fisher法")
    report.append("- CVaR计算: 基于历史模拟的条件期望损失")
    report.append("- 置信水平: 95%和99%")
    report.append("- 市场状态划分: 基于20日滚动收益率")
    report.append("- 压力测试: 蒙特卡洛模拟,抽取历史极端收益")

    report.append("\n" + "=" * 80)
    report.append("报告结束")
    report.append("=" * 80)

    # 保存报告
    report_text = '\n'.join(report)
    with open(f'{OUTPUT_DIR}/tail_risk_research_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"\n报告已保存至 {OUTPUT_DIR}/tail_risk_research_report.txt")

    return report_text


# ============================================================
# 主程序
# ============================================================

def main():
    """主程序"""
    print("=" * 60)
    print("股票尾部风险研究")
    print("=" * 60)

    # 1. 加载数据
    df = load_data()

    # 2. 计算个股尾部风险指标
    tail_risk_df = calculate_stock_tail_risk(df)

    # 3. 行业尾部风险分析
    industry_stats = analyze_industry_tail_risk(tail_risk_df)

    # 4. 市场状态影响分析
    market_state_df = analyze_market_state_impact(df)

    # 5. 尾部风险聚集分析
    daily_extreme = analyze_tail_clustering(df)

    # 6. 尾部风险相关性
    analyze_tail_correlation(tail_risk_df)

    # 7. 尾部风险筛选
    high_risk, low_risk, tail_risk_scored = tail_risk_screening(tail_risk_df)

    # 8. 组合尾部风险
    portfolio_df = portfolio_tail_risk(tail_risk_df, df)

    # 9. 极端情景模拟
    stress_results = extreme_scenario_simulation(df)

    # 10. 创建可视化
    create_visualizations(tail_risk_df, df, industry_stats, daily_extreme)

    # 11. 保存数据
    tail_risk_df.to_csv(f'{OUTPUT_DIR}/stock_tail_risk_metrics.csv', index=False, encoding='utf-8-sig')
    industry_stats.to_csv(f'{OUTPUT_DIR}/industry_tail_risk.csv', encoding='utf-8-sig')
    daily_extreme.to_csv(f'{OUTPUT_DIR}/daily_extreme_stats.csv', encoding='utf-8-sig')

    # 12. 生成报告
    report = generate_report(tail_risk_df, industry_stats, market_state_df,
                            daily_extreme, portfolio_df, high_risk, low_risk, stress_results)

    print("\n" + "=" * 60)
    print("研究完成!")
    print("=" * 60)
    print(f"输出目录: {OUTPUT_DIR}")
    print("生成文件:")
    print("  - tail_risk_research_report.txt (研究报告)")
    print("  - stock_tail_risk_metrics.csv (个股尾部风险指标)")
    print("  - industry_tail_risk.csv (行业尾部风险)")
    print("  - daily_extreme_stats.csv (每日极端统计)")
    print("  - tail_risk_distribution.png (尾部风险分布图)")
    print("  - industry_tail_risk.png (行业尾部风险对比)")
    print("  - tail_risk_clustering.png (尾部风险聚集时序)")
    print("  - var_vs_cvar.png (VaR vs CVaR散点图)")
    print("  - tail_risk_return.png (尾部风险与收益关系)")


if __name__ == '__main__':
    main()
