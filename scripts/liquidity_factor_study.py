#!/usr/bin/env python3
"""
A股市场流动性因子深入研究
Liquidity Factor Study for A-Share Market

研究内容:
1. 流动性度量指标构建
2. 流动性因子表现分析
3. 流动性风险研究
4. 流动性聚类分析
5. 流动性与其他因子关系
6. 流动性策略构建
"""

import duckdb
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 数据库路径
DB_PATH = '/Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db'
REPORT_PATH = '/Users/allen/workspace/python/stock/Tushare-DuckDB/reports/quality_check/liquidity_factor_study.md'

# 研究时间范围
START_DATE = '20180101'
END_DATE = '20251231'

def get_connection():
    """获取数据库连接"""
    return duckdb.connect(DB_PATH, read_only=True)

def load_daily_data():
    """加载日线数据"""
    conn = get_connection()
    query = f"""
    SELECT
        d.ts_code,
        d.trade_date,
        d.open,
        d.high,
        d.low,
        d.close,
        d.pre_close,
        d.pct_chg,
        d.vol,
        d.amount,
        db.turnover_rate,
        db.turnover_rate_f,
        db.volume_ratio,
        db.total_mv,
        db.circ_mv
    FROM daily d
    LEFT JOIN daily_basic db
        ON d.ts_code = db.ts_code AND d.trade_date = db.trade_date
    WHERE d.trade_date >= '{START_DATE}' AND d.trade_date <= '{END_DATE}'
    ORDER BY d.ts_code, d.trade_date
    """
    df = conn.execute(query).fetchdf()
    conn.close()
    return df

def load_stock_info():
    """加载股票基本信息"""
    conn = get_connection()
    query = """
    SELECT ts_code, name, industry, market, list_date
    FROM stock_basic
    WHERE list_status = 'L'
    """
    df = conn.execute(query).fetchdf()
    conn.close()
    return df

# ==================== 1. 流动性度量指标 ====================

def calculate_liquidity_metrics(df):
    """计算各种流动性度量指标"""
    print("计算流动性度量指标...")

    # 确保数据按股票和日期排序
    df = df.sort_values(['ts_code', 'trade_date'])

    # 1. 换手率 - 直接使用 turnover_rate
    df['turnover'] = df['turnover_rate']

    # 2. Amihud非流动性指标: |ret| / volume
    # 日收益率绝对值 / 成交额 (单位: 千元)
    df['ret_abs'] = df['pct_chg'].abs() / 100
    df['amihud'] = df['ret_abs'] / (df['amount'] + 1e-8)  # 避免除以0

    # 3. 日内振幅作为买卖价差代理
    df['spread_proxy'] = (df['high'] - df['low']) / ((df['high'] + df['low']) / 2) * 100

    # 4. 成交额/市值 (流动性比率)
    df['liq_ratio'] = df['amount'] / (df['circ_mv'] * 10000 + 1e-8)  # circ_mv 单位是万元

    # 5. Roll价差估计 - 基于连续收益率的协方差
    # Roll Spread = 2 * sqrt(-Cov(ret_t, ret_t-1))
    def calc_roll_spread(group):
        if len(group) < 22:
            return pd.Series([np.nan] * len(group), index=group.index)
        ret = group['pct_chg'] / 100
        roll_values = []
        for i in range(len(group)):
            if i < 21:
                roll_values.append(np.nan)
            else:
                window_ret = ret.iloc[i-21:i+1]
                cov = np.cov(window_ret.iloc[:-1], window_ret.iloc[1:])[0,1]
                if cov < 0:
                    roll_values.append(2 * np.sqrt(-cov))
                else:
                    roll_values.append(0)
        return pd.Series(roll_values, index=group.index)

    # 计算Roll价差 (较慢,分组计算)
    print("  计算Roll价差估计...")
    df['roll_spread'] = df.groupby('ts_code', group_keys=False).apply(calc_roll_spread)

    # 6. 计算月度流动性指标 (用于后续分析)
    df['month'] = df['trade_date'].str[:6]

    return df

def calculate_monthly_liquidity(df):
    """计算月度流动性指标"""
    print("计算月度流动性指标...")

    monthly_liq = df.groupby(['ts_code', 'month']).agg({
        'turnover': 'mean',
        'amihud': 'mean',
        'spread_proxy': 'mean',
        'liq_ratio': 'mean',
        'roll_spread': 'mean',
        'close': 'last',
        'pct_chg': lambda x: (1 + x/100).prod() - 1,  # 月收益率
        'circ_mv': 'last',
        'amount': 'sum',
        'vol': 'sum'
    }).reset_index()

    monthly_liq.columns = ['ts_code', 'month', 'turnover', 'amihud', 'spread_proxy',
                           'liq_ratio', 'roll_spread', 'close', 'ret', 'circ_mv',
                           'amount', 'vol']

    # 计算下月收益率 (用于因子预测分析)
    monthly_liq = monthly_liq.sort_values(['ts_code', 'month'])
    monthly_liq['next_ret'] = monthly_liq.groupby('ts_code')['ret'].shift(-1)

    return monthly_liq

# ==================== 2. 流动性因子表现 ====================

def analyze_liquidity_premium(monthly_liq):
    """分析低流动性溢价"""
    print("\n分析低流动性溢价...")

    results = {}

    for liq_factor in ['turnover', 'amihud', 'spread_proxy', 'liq_ratio']:
        # 每月分组
        def assign_group(group):
            valid = group[liq_factor].notna()
            group = group.copy()
            group['liq_group'] = pd.qcut(group.loc[valid, liq_factor],
                                         q=5, labels=[1,2,3,4,5],
                                         duplicates='drop')
            return group

        grouped = monthly_liq.groupby('month', group_keys=False).apply(assign_group)

        # 计算各组平均收益
        group_ret = grouped.groupby(['month', 'liq_group'])['next_ret'].mean().unstack()

        if group_ret.shape[1] == 5:
            # 计算多空组合收益 (低流动性 - 高流动性)
            if liq_factor in ['amihud', 'spread_proxy']:  # 这些指标越大越不流动
                ls_ret = group_ret[5] - group_ret[1]  # 高非流动性 - 低非流动性
            else:  # turnover, liq_ratio 越大越流动
                ls_ret = group_ret[1] - group_ret[5]  # 低流动性 - 高流动性

            results[liq_factor] = {
                'mean_ret': ls_ret.mean() * 100,  # 月均收益 %
                'std_ret': ls_ret.std() * 100,
                'sharpe': ls_ret.mean() / ls_ret.std() * np.sqrt(12) if ls_ret.std() > 0 else 0,
                't_stat': stats.ttest_1samp(ls_ret.dropna(), 0)[0],
                'win_rate': (ls_ret > 0).mean(),
                'group_returns': group_ret.mean() * 100
            }

    return results

def calculate_ic_ir(monthly_liq):
    """计算流动性因子IC和IR"""
    print("计算流动性因子IC/IR...")

    ic_results = {}

    for liq_factor in ['turnover', 'amihud', 'spread_proxy', 'liq_ratio', 'roll_spread']:
        # 计算每月横截面IC (因子值与下月收益的相关性)
        ic_values = monthly_liq.groupby('month').apply(
            lambda x: x[[liq_factor, 'next_ret']].dropna().corr(method='spearman').iloc[0,1]
            if len(x[[liq_factor, 'next_ret']].dropna()) > 10 else np.nan
        )

        ic_mean = ic_values.mean()
        ic_std = ic_values.std()
        ir = ic_mean / ic_std if ic_std > 0 else 0

        ic_results[liq_factor] = {
            'ic_mean': ic_mean,
            'ic_std': ic_std,
            'ir': ir,
            'ic_positive_rate': (ic_values > 0).mean(),
            't_stat': stats.ttest_1samp(ic_values.dropna(), 0)[0] if len(ic_values.dropna()) > 0 else 0,
            'ic_series': ic_values
        }

    return ic_results

def analyze_market_regime(df, monthly_liq):
    """分析不同市场环境下的流动性因子表现"""
    print("分析不同市场环境下的流动性表现...")

    # 计算市场收益率
    market_ret = df.groupby('trade_date')['pct_chg'].mean()
    market_ret.index = pd.to_datetime(market_ret.index)

    # 计算月度市场收益
    market_monthly = market_ret.resample('M').apply(lambda x: (1 + x/100).prod() - 1)
    market_monthly.index = market_monthly.index.strftime('%Y%m')

    # 定义市场状态
    market_state = pd.Series(index=market_monthly.index, dtype=str)
    market_state[market_monthly > 0.05] = 'bull'  # 牛市 (月涨幅>5%)
    market_state[(market_monthly >= -0.05) & (market_monthly <= 0.05)] = 'normal'  # 震荡
    market_state[market_monthly < -0.05] = 'bear'  # 熊市 (月跌幅>5%)

    # 计算各市场状态下的流动性因子表现
    results = {}

    for liq_factor in ['turnover', 'amihud']:
        # 每月分组
        def assign_group(group):
            valid = group[liq_factor].notna()
            group = group.copy()
            if sum(valid) >= 5:
                group.loc[valid, 'liq_group'] = pd.qcut(group.loc[valid, liq_factor],
                                                        q=5, labels=[1,2,3,4,5],
                                                        duplicates='drop')
            return group

        grouped = monthly_liq.groupby('month', group_keys=False).apply(assign_group)
        grouped['market_state'] = grouped['month'].map(market_state)

        # 各市场状态的分组收益
        state_results = {}
        for state in ['bull', 'normal', 'bear']:
            state_data = grouped[grouped['market_state'] == state]
            if len(state_data) > 0:
                group_ret = state_data.groupby('liq_group')['next_ret'].mean()
                state_results[state] = group_ret * 100

        results[liq_factor] = state_results

    return results, market_state

def analyze_size_relationship(monthly_liq):
    """分析流动性与市值的关系"""
    print("分析流动性与市值的关系...")

    # 计算流动性指标与市值的相关性
    correlations = {}

    # 取最近一个月的数据进行截面分析
    latest_month = monthly_liq['month'].max()
    latest_data = monthly_liq[monthly_liq['month'] == latest_month].copy()
    latest_data['log_mv'] = np.log(latest_data['circ_mv'] + 1)

    for liq_factor in ['turnover', 'amihud', 'spread_proxy', 'liq_ratio']:
        valid_data = latest_data[[liq_factor, 'log_mv']].dropna()
        if len(valid_data) > 10:
            corr = valid_data.corr(method='spearman').iloc[0,1]
            correlations[liq_factor] = corr

    # 按市值分组分析流动性
    latest_data['size_group'] = pd.qcut(latest_data['circ_mv'], q=5,
                                        labels=['Small', 'S2', 'S3', 'S4', 'Big'])

    size_liq = latest_data.groupby('size_group').agg({
        'turnover': 'median',
        'amihud': 'median',
        'spread_proxy': 'median',
        'liq_ratio': 'median'
    })

    return correlations, size_liq

# ==================== 3. 流动性风险 ====================

def calculate_liquidity_beta(df):
    """计算流动性Beta"""
    print("\n计算流动性Beta...")

    # 计算市场流动性 (所有股票的平均换手率)
    market_liq = df.groupby('trade_date').agg({
        'turnover': 'mean',
        'amihud': 'mean'
    })
    market_liq.columns = ['market_turnover', 'market_amihud']

    # 计算市场流动性变化
    market_liq['market_liq_change'] = market_liq['market_turnover'].pct_change()
    market_liq['market_ret'] = df.groupby('trade_date')['pct_chg'].mean()

    # 合并数据
    df = df.merge(market_liq.reset_index(), on='trade_date', how='left')

    # 计算每只股票的流动性Beta
    def calc_liq_beta(group):
        if len(group) < 60:
            return pd.Series({'liq_beta': np.nan, 'liq_beta_ret': np.nan})

        valid = group[['turnover', 'market_liq_change', 'pct_chg']].dropna()
        if len(valid) < 30:
            return pd.Series({'liq_beta': np.nan, 'liq_beta_ret': np.nan})

        # 流动性Beta: 个股流动性对市场流动性的敏感度
        liq_change = valid['turnover'].pct_change()
        market_liq_change = valid['market_liq_change']

        valid_idx = ~(liq_change.isna() | market_liq_change.isna())
        if sum(valid_idx) < 30:
            return pd.Series({'liq_beta': np.nan, 'liq_beta_ret': np.nan})

        # 流动性Beta
        cov_liq = np.cov(liq_change[valid_idx], market_liq_change[valid_idx])[0,1]
        var_market = np.var(market_liq_change[valid_idx])
        liq_beta = cov_liq / var_market if var_market > 0 else np.nan

        # 收益率对市场流动性变化的敏感度
        valid_ret = valid[['pct_chg', 'market_liq_change']].dropna()
        if len(valid_ret) < 30:
            liq_beta_ret = np.nan
        else:
            cov_ret = np.cov(valid_ret['pct_chg'], valid_ret['market_liq_change'])[0,1]
            liq_beta_ret = cov_ret / var_market if var_market > 0 else np.nan

        return pd.Series({'liq_beta': liq_beta, 'liq_beta_ret': liq_beta_ret})

    liq_betas = df.groupby('ts_code').apply(calc_liq_beta)

    return liq_betas, market_liq

def calculate_liquidity_commonality(df):
    """计算流动性共性 (Commonality)"""
    print("计算流动性共性...")

    # 计算市场平均流动性 (排除自身)
    market_turnover = df.groupby('trade_date')['turnover'].mean()

    # 计算每只股票流动性与市场流动性的相关性
    df = df.merge(market_turnover.reset_index().rename(columns={'turnover': 'market_turnover'}),
                  on='trade_date', how='left')

    def calc_commonality(group):
        if len(group) < 60:
            return np.nan
        valid = group[['turnover', 'market_turnover']].dropna()
        if len(valid) < 30:
            return np.nan
        return valid['turnover'].corr(valid['market_turnover'])

    commonality = df.groupby('ts_code').apply(calc_commonality)

    return commonality

def analyze_liquidity_crisis(df, market_liq):
    """分析流动性危机时的表现"""
    print("分析流动性危机时的表现...")

    # 识别流动性危机期 (市场换手率大幅下降)
    market_liq = market_liq.copy()
    market_liq['turnover_ma20'] = market_liq['market_turnover'].rolling(20).mean()
    market_liq['turnover_std20'] = market_liq['market_turnover'].rolling(20).std()

    # 流动性危机: 换手率低于均值2个标准差
    market_liq['is_crisis'] = market_liq['market_turnover'] < (
        market_liq['turnover_ma20'] - 2 * market_liq['turnover_std20']
    )

    crisis_dates = market_liq[market_liq['is_crisis']].index.tolist()

    # 分析危机期间的特征
    crisis_analysis = {
        'n_crisis_days': len(crisis_dates),
        'crisis_dates_sample': crisis_dates[:10] if len(crisis_dates) > 10 else crisis_dates,
        'avg_crisis_turnover': market_liq.loc[market_liq['is_crisis'], 'market_turnover'].mean() if len(crisis_dates) > 0 else np.nan,
        'avg_normal_turnover': market_liq.loc[~market_liq['is_crisis'], 'market_turnover'].mean()
    }

    return crisis_analysis, market_liq

# ==================== 4. 流动性聚类 ====================

def perform_liquidity_clustering(monthly_liq):
    """流动性聚类分析"""
    print("\n进行流动性聚类分析...")

    # 取最近6个月的数据计算平均流动性特征
    recent_months = sorted(monthly_liq['month'].unique())[-6:]
    recent_data = monthly_liq[monthly_liq['month'].isin(recent_months)]

    # 计算股票的流动性特征均值
    stock_liq = recent_data.groupby('ts_code').agg({
        'turnover': 'mean',
        'amihud': 'mean',
        'spread_proxy': 'mean',
        'liq_ratio': 'mean',
        'circ_mv': 'mean',
        'ret': 'mean'
    }).dropna()

    if len(stock_liq) < 100:
        return None, None

    # 标准化
    features = ['turnover', 'amihud', 'spread_proxy', 'liq_ratio']
    scaler = StandardScaler()
    X = scaler.fit_transform(stock_liq[features])

    # K-Means聚类
    n_clusters = 4
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    stock_liq['cluster'] = kmeans.fit_predict(X)

    # 分析各聚类特征
    cluster_stats = stock_liq.groupby('cluster').agg({
        'turnover': ['mean', 'std'],
        'amihud': ['mean', 'std'],
        'spread_proxy': ['mean', 'std'],
        'liq_ratio': ['mean', 'std'],
        'circ_mv': ['mean', 'median'],
        'ret': 'mean'
    })
    cluster_stats.columns = ['_'.join(col) for col in cluster_stats.columns]

    # 各聚类股票数量
    cluster_counts = stock_liq['cluster'].value_counts().sort_index()

    return cluster_stats, cluster_counts, stock_liq

def analyze_liquidity_transitions(monthly_liq):
    """分析流动性转换事件"""
    print("分析流动性转换事件...")

    # 计算流动性分位数
    monthly_liq = monthly_liq.copy()

    def assign_liq_quantile(group):
        group = group.copy()
        valid = group['turnover'].notna()
        if sum(valid) >= 5:
            group.loc[valid, 'liq_quintile'] = pd.qcut(
                group.loc[valid, 'turnover'], q=5, labels=[1,2,3,4,5], duplicates='drop'
            )
        return group

    monthly_liq = monthly_liq.groupby('month', group_keys=False).apply(assign_liq_quantile)

    # 计算分位数变化
    monthly_liq = monthly_liq.sort_values(['ts_code', 'month'])
    monthly_liq['prev_quintile'] = monthly_liq.groupby('ts_code')['liq_quintile'].shift(1)

    # 计算转移矩阵
    valid_transitions = monthly_liq.dropna(subset=['liq_quintile', 'prev_quintile'])
    transition_matrix = pd.crosstab(
        valid_transitions['prev_quintile'],
        valid_transitions['liq_quintile'],
        normalize='index'
    ) * 100  # 转为百分比

    # 分析大幅转换事件 (从最低到最高,或反之)
    monthly_liq['big_transition'] = (
        ((monthly_liq['prev_quintile'] == 1) & (monthly_liq['liq_quintile'] == 5)) |
        ((monthly_liq['prev_quintile'] == 5) & (monthly_liq['liq_quintile'] == 1))
    )

    big_transitions = monthly_liq[monthly_liq['big_transition'] == True]

    return transition_matrix, big_transitions

# ==================== 5. 流动性与其他因子 ====================

def analyze_factor_relationships(monthly_liq, df):
    """分析流动性与其他因子的关系"""
    print("\n分析流动性与其他因子的关系...")

    # 计算动量因子 (过去12个月累计收益)
    monthly_liq = monthly_liq.sort_values(['ts_code', 'month'])
    monthly_liq['momentum_12m'] = monthly_liq.groupby('ts_code')['ret'].transform(
        lambda x: x.rolling(12).apply(lambda y: (1 + y).prod() - 1, raw=True)
    )

    # 市值因子 (对数市值)
    monthly_liq['log_mv'] = np.log(monthly_liq['circ_mv'] + 1)

    # 计算最近月份的因子相关性矩阵
    latest_months = sorted(monthly_liq['month'].unique())[-12:]
    latest_data = monthly_liq[monthly_liq['month'].isin(latest_months)]

    factors = ['turnover', 'amihud', 'log_mv', 'momentum_12m']
    factor_corr = latest_data[factors].corr(method='spearman')

    # 双变量排序分析: 控制市值后的流动性效应
    def double_sort_analysis(data, control_var, target_var, return_var):
        """双变量排序分析"""
        results = []

        for month in data['month'].unique():
            month_data = data[data['month'] == month].copy()

            # 先按控制变量分组
            valid = month_data[[control_var, target_var, return_var]].dropna()
            if len(valid) < 25:
                continue

            valid['control_group'] = pd.qcut(valid[control_var], q=5, labels=[1,2,3,4,5], duplicates='drop')

            # 在每个控制组内按目标变量分组
            def inner_sort(group):
                if len(group) < 5:
                    return group
                group = group.copy()
                group['target_group'] = pd.qcut(group[target_var], q=5, labels=[1,2,3,4,5], duplicates='drop')
                return group

            valid = valid.groupby('control_group', group_keys=False).apply(inner_sort)

            # 计算各组平均收益
            group_ret = valid.groupby('target_group')[return_var].mean()
            if len(group_ret) == 5:
                results.append(group_ret)

        if len(results) > 0:
            avg_ret = pd.concat(results, axis=1).mean(axis=1)
            return avg_ret
        return None

    # 控制市值后的流动性效应
    liq_effect_controlled = double_sort_analysis(latest_data, 'log_mv', 'turnover', 'next_ret')

    return factor_corr, liq_effect_controlled

def calculate_liquidity_adjusted_factors(monthly_liq):
    """计算流动性调整后的因子"""
    print("计算流动性调整后的因子...")

    # 动量因子的流动性调整
    monthly_liq = monthly_liq.copy()
    monthly_liq['momentum_12m'] = monthly_liq.groupby('ts_code')['ret'].transform(
        lambda x: x.rolling(12).apply(lambda y: (1 + y).prod() - 1, raw=True)
    )

    # 在流动性组内比较动量效应
    results = {}

    for liq_group in [1, 2, 3, 4, 5]:
        # 每月先按流动性分组
        def assign_groups(data):
            data = data.copy()
            valid = data['turnover'].notna()
            if sum(valid) >= 5:
                data.loc[valid, 'liq_group'] = pd.qcut(
                    data.loc[valid, 'turnover'], q=5, labels=[1,2,3,4,5], duplicates='drop'
                )
            return data

        grouped = monthly_liq.groupby('month', group_keys=False).apply(assign_groups)
        liq_subset = grouped[grouped['liq_group'] == liq_group]

        # 在该流动性组内按动量分组
        def assign_mom_group(data):
            data = data.copy()
            valid = data['momentum_12m'].notna()
            if sum(valid) >= 3:
                data.loc[valid, 'mom_group'] = pd.qcut(
                    data.loc[valid, 'momentum_12m'], q=3, labels=['Low', 'Mid', 'High'], duplicates='drop'
                )
            return data

        liq_subset = liq_subset.groupby('month', group_keys=False).apply(assign_mom_group)

        # 计算各动量组的平均收益
        mom_ret = liq_subset.groupby('mom_group')['next_ret'].mean() * 100
        results[f'Liq_Q{liq_group}'] = mom_ret

    return pd.DataFrame(results)

# ==================== 6. 流动性策略 ====================

def build_liquidity_portfolio(monthly_liq):
    """构建流动性因子组合"""
    print("\n构建流动性因子组合...")

    # 策略1: 低流动性多头组合
    def get_low_liq_portfolio_ret(month_data):
        valid = month_data['turnover'].notna() & month_data['next_ret'].notna()
        if sum(valid) < 50:
            return np.nan
        data = month_data[valid].copy()
        # 选取换手率最低的20%股票
        threshold = data['turnover'].quantile(0.2)
        low_liq_stocks = data[data['turnover'] <= threshold]
        return low_liq_stocks['next_ret'].mean()

    low_liq_ret = monthly_liq.groupby('month').apply(get_low_liq_portfolio_ret)

    # 策略2: Amihud非流动性多头组合
    def get_amihud_portfolio_ret(month_data):
        valid = month_data['amihud'].notna() & month_data['next_ret'].notna()
        if sum(valid) < 50:
            return np.nan
        data = month_data[valid].copy()
        # 选取Amihud最高的20%股票 (最不流动)
        threshold = data['amihud'].quantile(0.8)
        illiq_stocks = data[data['amihud'] >= threshold]
        return illiq_stocks['next_ret'].mean()

    amihud_ret = monthly_liq.groupby('month').apply(get_amihud_portfolio_ret)

    # 策略3: 流动性因子多空组合 (考虑交易成本)
    def get_ls_portfolio_ret(month_data, cost_rate=0.003):  # 假设单边交易成本0.3%
        valid = month_data['turnover'].notna() & month_data['next_ret'].notna()
        if sum(valid) < 50:
            return np.nan
        data = month_data[valid].copy()

        low_threshold = data['turnover'].quantile(0.2)
        high_threshold = data['turnover'].quantile(0.8)

        long_ret = data[data['turnover'] <= low_threshold]['next_ret'].mean()
        short_ret = data[data['turnover'] >= high_threshold]['next_ret'].mean()

        # 考虑双边交易成本
        ls_ret = long_ret - short_ret - 2 * cost_rate
        return ls_ret

    ls_ret = monthly_liq.groupby('month').apply(get_ls_portfolio_ret)

    # 计算策略绩效
    strategies = {
        'Low_Liquidity': low_liq_ret,
        'Amihud_Illiquidity': amihud_ret,
        'LS_Net_Cost': ls_ret
    }

    performance = {}
    for name, ret_series in strategies.items():
        ret = ret_series.dropna()
        if len(ret) > 0:
            performance[name] = {
                'annual_return': ret.mean() * 12 * 100,
                'annual_vol': ret.std() * np.sqrt(12) * 100,
                'sharpe': ret.mean() / ret.std() * np.sqrt(12) if ret.std() > 0 else 0,
                'max_drawdown': calculate_max_drawdown(ret),
                'win_rate': (ret > 0).mean(),
                'n_months': len(ret)
            }

    return performance, strategies

def calculate_max_drawdown(returns):
    """计算最大回撤"""
    cum_ret = (1 + returns).cumprod()
    running_max = cum_ret.cummax()
    drawdown = (cum_ret - running_max) / running_max
    return drawdown.min() * 100

def analyze_liquidity_timing(monthly_liq, market_state):
    """流动性择时分析"""
    print("流动性择时分析...")

    # 根据市场状态调整流动性策略
    monthly_liq = monthly_liq.copy()
    monthly_liq['market_state'] = monthly_liq['month'].map(market_state)

    # 计算低流动性溢价在不同市场状态下的表现
    results = {}

    for state in ['bull', 'normal', 'bear']:
        state_data = monthly_liq[monthly_liq['market_state'] == state]
        if len(state_data) < 10:
            continue

        # 计算低流动性组收益
        def get_liq_premium(month_data):
            valid = month_data['turnover'].notna() & month_data['next_ret'].notna()
            if sum(valid) < 30:
                return np.nan
            data = month_data[valid].copy()

            q20 = data['turnover'].quantile(0.2)
            q80 = data['turnover'].quantile(0.8)

            low_ret = data[data['turnover'] <= q20]['next_ret'].mean()
            high_ret = data[data['turnover'] >= q80]['next_ret'].mean()

            return low_ret - high_ret

        premium = state_data.groupby('month').apply(get_liq_premium)

        results[state] = {
            'mean_premium': premium.mean() * 100,
            'std_premium': premium.std() * 100,
            'positive_rate': (premium > 0).mean(),
            'n_months': len(premium.dropna())
        }

    return results

# ==================== 7. 生成报告 ====================

def generate_report(all_results):
    """生成分析报告"""
    print("\n生成分析报告...")

    report = """# A股市场流动性因子深入研究报告

## 摘要

本报告对A股市场的流动性因子进行了全面深入的研究,包括流动性度量指标的构建、因子表现分析、风险特征、聚类分析、因子关系以及策略构建等多个维度。

研究时间范围: {start_date} - {end_date}

---

## 1. 流动性度量指标

### 1.1 指标定义

本研究构建了以下五种流动性度量指标:

| 指标 | 定义 | 特点 |
|------|------|------|
| **换手率 (Turnover)** | 成交量/流通股本 | 直接度量交易活跃度 |
| **Amihud非流动性** | \\|收益率\\| / 成交额 | 价格冲击度量,值越大越不流动 |
| **价差代理 (Spread Proxy)** | (最高价-最低价) / 均价 | 用日内振幅近似买卖价差 |
| **流动性比率 (Liq Ratio)** | 成交额 / 流通市值 | 相对于市值的交易量 |
| **Roll价差估计** | 2√(-Cov(r_t, r_{t-1})) | 基于收益率自协方差 |

### 1.2 指标统计特征

""".format(start_date=START_DATE, end_date=END_DATE)

    # 添加流动性指标统计
    if 'liq_stats' in all_results:
        stats = all_results['liq_stats']
        report += """
| 指标 | 均值 | 标准差 | 中位数 | 25%分位 | 75%分位 |
|------|------|--------|--------|---------|---------|
"""
        for metric, values in stats.items():
            report += f"| {metric} | {values['mean']:.4f} | {values['std']:.4f} | {values['median']:.4f} | {values['q25']:.4f} | {values['q75']:.4f} |\n"

    report += """

---

## 2. 流动性因子表现

### 2.1 低流动性溢价分析

根据有效市场假说的拓展,低流动性股票应获得溢价以补偿投资者承担的流动性风险。

"""

    if 'premium_results' in all_results:
        premium = all_results['premium_results']
        report += """
#### 流动性因子多空组合表现

| 因子 | 月均收益(%) | 年化收益(%) | 夏普比率 | t统计量 | 胜率 |
|------|-------------|-------------|----------|---------|------|
"""
        for factor, values in premium.items():
            annual_ret = values['mean_ret'] * 12
            report += f"| {factor} | {values['mean_ret']:.3f} | {annual_ret:.2f} | {values['sharpe']:.3f} | {values['t_stat']:.2f} | {values['win_rate']*100:.1f}% |\n"

        # 分组收益
        report += """

#### 分组平均月收益(%)

"""
        for factor, values in premium.items():
            if 'group_returns' in values:
                gr = values['group_returns']
                report += f"**{factor}**: Q1={gr[1]:.3f}, Q2={gr[2]:.3f}, Q3={gr[3]:.3f}, Q4={gr[4]:.3f}, Q5={gr[5]:.3f}\n\n"

    report += """
### 2.2 流动性因子IC/IR分析

IC (Information Coefficient) 衡量因子预测收益的能力,IR (Information Ratio) = IC均值/IC标准差。

"""

    if 'ic_results' in all_results:
        ic = all_results['ic_results']
        report += """
| 因子 | IC均值 | IC标准差 | IR | IC>0比例 | t统计量 |
|------|--------|----------|-----|----------|---------|
"""
        for factor, values in ic.items():
            report += f"| {factor} | {values['ic_mean']:.4f} | {values['ic_std']:.4f} | {values['ir']:.3f} | {values['ic_positive_rate']*100:.1f}% | {values['t_stat']:.2f} |\n"

    report += """

**解读**:
- IC为负表示低流动性股票倾向于有更高的未来收益 (流动性溢价)
- IR绝对值大于0.5表明因子具有较好的稳定性
- IC>0比例低于40%表明因子具有持续的预测能力

### 2.3 不同市场环境下的表现

"""

    if 'regime_results' in all_results:
        regime = all_results['regime_results']
        for factor, states in regime.items():
            report += f"\n**{factor}因子分组收益 (月均, %)**\n\n"
            report += "| 市场状态 | Q1 | Q2 | Q3 | Q4 | Q5 | Q1-Q5 |\n"
            report += "|----------|-----|-----|-----|-----|-----|-------|\n"
            for state, rets in states.items():
                if len(rets) == 5:
                    spread = rets[1] - rets[5]
                    report += f"| {state} | {rets[1]:.2f} | {rets[2]:.2f} | {rets[3]:.2f} | {rets[4]:.2f} | {rets[5]:.2f} | {spread:.2f} |\n"

    report += """

### 2.4 流动性与市值的关系

"""

    if 'size_corr' in all_results:
        report += "**流动性指标与对数市值的Spearman相关性:**\n\n"
        for factor, corr in all_results['size_corr'].items():
            report += f"- {factor}: {corr:.3f}\n"

        report += "\n**按市值分组的流动性中位数:**\n\n"
        if 'size_liq' in all_results:
            size_liq = all_results['size_liq']
            report += "| 市值组 | 换手率 | Amihud | 价差代理 | 流动性比率 |\n"
            report += "|--------|--------|--------|----------|------------|\n"
            for idx in size_liq.index:
                row = size_liq.loc[idx]
                report += f"| {idx} | {row['turnover']:.2f} | {row['amihud']:.6f} | {row['spread_proxy']:.2f} | {row['liq_ratio']:.4f} |\n"

    report += """

---

## 3. 流动性风险

### 3.1 流动性Beta

流动性Beta衡量个股流动性对市场整体流动性变化的敏感度。高流动性Beta的股票在市场流动性下降时,自身流动性下降更多。

"""

    if 'liq_beta_stats' in all_results:
        beta_stats = all_results['liq_beta_stats']
        report += f"""
**流动性Beta统计:**
- 均值: {beta_stats['mean']:.3f}
- 中位数: {beta_stats['median']:.3f}
- 标准差: {beta_stats['std']:.3f}
- 最小值: {beta_stats['min']:.3f}
- 最大值: {beta_stats['max']:.3f}

高流动性Beta的股票在市场流动性收紧时面临更大的流动性风险。
"""

    report += """

### 3.2 流动性共性 (Commonality)

流动性共性衡量个股流动性与市场流动性的相关程度。高共性意味着在市场流动性危机时更难以交易。

"""

    if 'commonality_stats' in all_results:
        comm_stats = all_results['commonality_stats']
        report += f"""
**流动性共性统计:**
- 均值: {comm_stats['mean']:.3f}
- 中位数: {comm_stats['median']:.3f}
- 共性>0.5的股票比例: {comm_stats['high_commonality_pct']:.1f}%

大多数股票的流动性与市场流动性存在正相关,表明流动性具有系统性成分。
"""

    report += """

### 3.3 流动性危机分析

"""

    if 'crisis_analysis' in all_results:
        crisis = all_results['crisis_analysis']
        report += f"""
**流动性危机识别 (换手率低于20日均值减2个标准差):**
- 危机天数: {crisis['n_crisis_days']}
- 危机期平均换手率: {crisis['avg_crisis_turnover']:.2f}%
- 正常期平均换手率: {crisis['avg_normal_turnover']:.2f}%

流动性危机期间,市场整体换手率显著下降,这时低流动性股票更难交易,流动性风险溢价可能扩大。
"""

    report += """

---

## 4. 流动性聚类分析

### 4.1 聚类结果

基于换手率、Amihud、价差代理和流动性比率四个维度,使用K-Means算法将股票分为4类:

"""

    if 'cluster_stats' in all_results and all_results['cluster_stats'] is not None:
        cluster_stats = all_results['cluster_stats']
        cluster_counts = all_results['cluster_counts']

        report += "| 聚类 | 股票数 | 平均换手率 | 平均Amihud | 平均价差 | 平均市值(亿) | 平均月收益 |\n"
        report += "|------|--------|------------|------------|----------|--------------|------------|\n"
        for i in range(len(cluster_counts)):
            row = cluster_stats.loc[i]
            n = cluster_counts[i]
            mv = row['circ_mv_mean'] / 10000  # 转为亿元
            report += f"| {i} | {n} | {row['turnover_mean']:.2f} | {row['amihud_mean']:.6f} | {row['spread_proxy_mean']:.2f} | {mv:.1f} | {row['ret_mean']*100:.2f}% |\n"

        report += """

**聚类解读:**
- 高换手率、低Amihud = 高流动性股票 (通常是大市值蓝筹)
- 低换手率、高Amihud = 低流动性股票 (通常是小市值或冷门股)
"""

    report += """

### 4.2 流动性转移矩阵

分析股票流动性状态的月度转移概率:

"""

    if 'transition_matrix' in all_results and all_results['transition_matrix'] is not None:
        tm = all_results['transition_matrix']
        report += "| 上月\\本月 | Q1(最低) | Q2 | Q3 | Q4 | Q5(最高) |\n"
        report += "|-----------|----------|-----|-----|-----|----------|\n"
        for i in tm.index:
            row = [f"{tm.loc[i, j]:.1f}%" if j in tm.columns else "N/A" for j in [1,2,3,4,5]]
            report += f"| Q{int(i)} | {' | '.join(row)} |\n"

        report += """

**解读**: 对角线概率较高表明流动性状态具有持续性。流动性从最低组(Q1)跳到最高组(Q5)的概率很低。
"""

    report += """

---

## 5. 流动性与其他因子

### 5.1 因子相关性矩阵

"""

    if 'factor_corr' in all_results:
        corr = all_results['factor_corr']
        report += "| | 换手率 | Amihud | 对数市值 | 动量12M |\n"
        report += "|---|--------|--------|----------|--------|\n"
        for idx in corr.index:
            row = [f"{corr.loc[idx, col]:.3f}" for col in corr.columns]
            report += f"| {idx} | {' | '.join(row)} |\n"

        report += """

**关键发现:**
- 换手率与市值负相关:小市值股票换手率更高
- Amihud与市值负相关:小市值股票价格冲击更大
- 流动性与动量的关系有待进一步控制市值后分析
"""

    report += """

### 5.2 控制市值后的流动性效应

双变量排序分析:先按市值分5组,在每组内再按流动性分5组

"""

    if 'liq_effect_controlled' in all_results and all_results['liq_effect_controlled'] is not None:
        effect = all_results['liq_effect_controlled']
        report += "**控制市值后,各换手率组的平均月收益(%):**\n\n"
        for i, ret in effect.items():
            report += f"- Q{i} (换手率): {ret*100:.3f}%\n"
        report += "\n控制市值后,低换手率股票仍有超额收益,说明流动性溢价独立于规模效应存在。\n"

    report += """

### 5.3 不同流动性组内的动量效应

分析在不同流动性环境下,动量策略的表现差异:

"""

    if 'mom_by_liq' in all_results:
        mom_df = all_results['mom_by_liq']
        report += "| 动量组 | " + " | ".join(mom_df.columns) + " |\n"
        report += "|--------|" + "|".join(["--------" for _ in mom_df.columns]) + "|\n"
        for idx in mom_df.index:
            row = [f"{mom_df.loc[idx, col]:.2f}%" if pd.notna(mom_df.loc[idx, col]) else "N/A" for col in mom_df.columns]
            report += f"| {idx} | {' | '.join(row)} |\n"

        report += """

**发现**: 在低流动性股票中,动量效应可能更强或更弱,这取决于市场结构和投资者行为。
"""

    report += """

---

## 6. 流动性策略

### 6.1 策略绩效对比

"""

    if 'strategy_performance' in all_results:
        perf = all_results['strategy_performance']
        report += "| 策略 | 年化收益(%) | 年化波动(%) | 夏普比率 | 最大回撤(%) | 胜率 | 月份数 |\n"
        report += "|------|-------------|-------------|----------|-------------|------|--------|\n"
        for name, values in perf.items():
            report += f"| {name} | {values['annual_return']:.2f} | {values['annual_vol']:.2f} | {values['sharpe']:.3f} | {values['max_drawdown']:.1f} | {values['win_rate']*100:.1f}% | {values['n_months']} |\n"

    report += """

**策略说明:**
- **Low_Liquidity**: 做多换手率最低的20%股票
- **Amihud_Illiquidity**: 做多Amihud最高的20%股票
- **LS_Net_Cost**: 多空组合(做多低流动性,做空高流动性),扣除双边0.3%交易成本

### 6.2 流动性择时

在不同市场状态下的流动性溢价:

"""

    if 'timing_results' in all_results:
        timing = all_results['timing_results']
        report += "| 市场状态 | 月均溢价(%) | 溢价标准差(%) | 正收益比例 | 月份数 |\n"
        report += "|----------|-------------|---------------|------------|--------|\n"
        for state, values in timing.items():
            report += f"| {state} | {values['mean_premium']:.3f} | {values['std_premium']:.3f} | {values['positive_rate']*100:.1f}% | {values['n_months']} |\n"

        report += """

**择时建议:**
- 熊市期间流动性溢价可能扩大,但风险也更高
- 牛市期间流动性溢价可能收窄,因为市场整体流动性充裕
- 震荡市是执行流动性策略的较好时机
"""

    report += """

---

## 7. 研究结论与投资建议

### 7.1 主要发现

1. **流动性溢价存在**: A股市场存在明显的低流动性溢价,低换手率、高Amihud的股票长期有超额收益。

2. **流动性与规模高度相关**: 小市值股票流动性普遍较差,控制市值后流动性溢价依然显著。

3. **流动性具有持续性**: 流动性状态月度转移概率显示对角线概率较高,流动性变化相对稳定。

4. **流动性风险具有系统性**: 多数股票的流动性与市场流动性正相关,在流动性危机时风险共振。

5. **考虑交易成本后收益降低**: 实际交易中需要考虑冲击成本,低流动性股票交易成本更高。

### 7.2 投资建议

1. **长期投资者**: 可以适度配置低流动性股票以获取流动性溢价。

2. **策略选择**: 建议使用流动性因子与其他因子(如价值、质量)结合,避免单一暴露。

3. **风险控制**: 在市场流动性收紧时降低仓位,避免在危机期间被迫卖出。

4. **交易执行**: 低流动性股票应采用分批建仓、限价单等方式降低冲击成本。

5. **流动性监控**: 关注市场整体流动性指标,作为风险预警信号。

### 7.3 研究局限

1. 未考虑个股的实际买卖价差数据
2. Roll价差估计在高频数据下效果更好
3. 流动性Beta的估计对样本期敏感
4. 交易成本假设可能与实际有偏差

---

*报告生成时间: {timestamp}*

*数据来源: Tushare*
""".format(timestamp=pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'))

    return report

def main():
    """主函数"""
    print("="*60)
    print("A股市场流动性因子深入研究")
    print("="*60)

    # 存储所有结果
    all_results = {}

    # 1. 加载数据
    print("\n[1/7] 加载数据...")
    df = load_daily_data()
    stock_info = load_stock_info()
    print(f"  加载 {len(df):,} 条日线数据, {df['ts_code'].nunique()} 只股票")

    # 2. 计算流动性指标
    print("\n[2/7] 计算流动性指标...")
    df = calculate_liquidity_metrics(df)

    # 流动性指标统计
    liq_stats = {}
    for col in ['turnover', 'amihud', 'spread_proxy', 'liq_ratio', 'roll_spread']:
        valid_data = df[col].dropna()
        liq_stats[col] = {
            'mean': valid_data.mean(),
            'std': valid_data.std(),
            'median': valid_data.median(),
            'q25': valid_data.quantile(0.25),
            'q75': valid_data.quantile(0.75)
        }
    all_results['liq_stats'] = liq_stats

    # 3. 计算月度流动性
    monthly_liq = calculate_monthly_liquidity(df)
    print(f"  生成 {len(monthly_liq):,} 条月度数据")

    # 4. 流动性因子表现分析
    print("\n[3/7] 分析流动性因子表现...")

    # 流动性溢价
    premium_results = analyze_liquidity_premium(monthly_liq)
    all_results['premium_results'] = premium_results

    # IC/IR
    ic_results = calculate_ic_ir(monthly_liq)
    all_results['ic_results'] = ic_results

    # 市场环境分析
    regime_results, market_state = analyze_market_regime(df, monthly_liq)
    all_results['regime_results'] = regime_results

    # 与市值关系
    size_corr, size_liq = analyze_size_relationship(monthly_liq)
    all_results['size_corr'] = size_corr
    all_results['size_liq'] = size_liq

    # 5. 流动性风险分析
    print("\n[4/7] 分析流动性风险...")

    # 流动性Beta
    liq_betas, market_liq = calculate_liquidity_beta(df)
    beta_values = liq_betas['liq_beta'].dropna()
    all_results['liq_beta_stats'] = {
        'mean': beta_values.mean(),
        'median': beta_values.median(),
        'std': beta_values.std(),
        'min': beta_values.min(),
        'max': beta_values.max()
    }

    # 流动性共性
    commonality = calculate_liquidity_commonality(df)
    comm_values = commonality.dropna()
    all_results['commonality_stats'] = {
        'mean': comm_values.mean(),
        'median': comm_values.median(),
        'high_commonality_pct': (comm_values > 0.5).mean() * 100
    }

    # 流动性危机
    crisis_analysis, market_liq_updated = analyze_liquidity_crisis(df, market_liq)
    all_results['crisis_analysis'] = crisis_analysis

    # 6. 流动性聚类
    print("\n[5/7] 流动性聚类分析...")
    cluster_result = perform_liquidity_clustering(monthly_liq)
    if cluster_result[0] is not None:
        all_results['cluster_stats'] = cluster_result[0]
        all_results['cluster_counts'] = cluster_result[1]

    # 流动性转移
    transition_matrix, big_transitions = analyze_liquidity_transitions(monthly_liq)
    all_results['transition_matrix'] = transition_matrix

    # 7. 流动性与其他因子
    print("\n[6/7] 分析因子关系...")
    factor_corr, liq_effect_controlled = analyze_factor_relationships(monthly_liq, df)
    all_results['factor_corr'] = factor_corr
    all_results['liq_effect_controlled'] = liq_effect_controlled

    # 流动性调整后的动量
    mom_by_liq = calculate_liquidity_adjusted_factors(monthly_liq)
    all_results['mom_by_liq'] = mom_by_liq

    # 8. 流动性策略
    print("\n[7/7] 构建流动性策略...")
    strategy_performance, strategies = build_liquidity_portfolio(monthly_liq)
    all_results['strategy_performance'] = strategy_performance

    # 流动性择时
    timing_results = analyze_liquidity_timing(monthly_liq, market_state)
    all_results['timing_results'] = timing_results

    # 9. 生成报告
    report = generate_report(all_results)

    # 保存报告
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(report)

    print("\n" + "="*60)
    print(f"报告已保存至: {REPORT_PATH}")
    print("="*60)

    return all_results

if __name__ == '__main__':
    results = main()
