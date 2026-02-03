#!/usr/bin/env python3
"""
A股市场 Beta 与系统性风险研究
================================
研究内容：
1. Beta估计: 市场模型回归、不同窗口、滚动Beta、收缩估计
2. Beta特征分析: 分布、行业差异、市值关系、时变性
3. 低Beta异象: 低Beta溢价、彩票效应、行为金融解释
4. Beta预测: 均值回归、预测模型、Blume调整
5. Beta策略: Beta中性、低Beta组合、市场择时
"""

import duckdb
import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# 数据库路径
DB_PATH = '/Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db'

def get_connection():
    """获取只读数据库连接"""
    return duckdb.connect(DB_PATH, read_only=True)

def load_market_data(start_date='20150101', end_date='20260130'):
    """加载市场数据和个股数据"""
    conn = get_connection()

    # 加载沪深300指数作为市场基准
    market_query = f"""
    SELECT trade_date, pct_chg as mkt_ret
    FROM index_daily
    WHERE ts_code = '000300.SH'
    AND trade_date >= '{start_date}' AND trade_date <= '{end_date}'
    ORDER BY trade_date
    """
    market_df = conn.execute(market_query).fetchdf()
    market_df['trade_date'] = pd.to_datetime(market_df['trade_date'])

    # 加载个股收益率数据
    stock_query = f"""
    SELECT d.ts_code, d.trade_date, d.pct_chg as ret,
           db.total_mv, db.circ_mv, db.turnover_rate, db.pb, db.pe_ttm,
           s.industry, s.name
    FROM daily d
    LEFT JOIN daily_basic db ON d.ts_code = db.ts_code AND d.trade_date = db.trade_date
    LEFT JOIN stock_basic s ON d.ts_code = s.ts_code
    WHERE d.trade_date >= '{start_date}' AND d.trade_date <= '{end_date}'
    AND d.pct_chg IS NOT NULL
    AND d.ts_code NOT LIKE 'BJ%'
    AND d.ts_code NOT LIKE '68%'
    ORDER BY d.ts_code, d.trade_date
    """
    stock_df = conn.execute(stock_query).fetchdf()
    stock_df['trade_date'] = pd.to_datetime(stock_df['trade_date'])

    conn.close()
    return market_df, stock_df

def estimate_beta_ols(stock_returns, market_returns, min_obs=20):
    """OLS估计Beta"""
    valid_idx = ~(np.isnan(stock_returns) | np.isnan(market_returns))
    if valid_idx.sum() < min_obs:
        return np.nan, np.nan, np.nan, np.nan

    y = stock_returns[valid_idx]
    x = market_returns[valid_idx]

    # OLS回归
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    cov_xy = np.sum((x - x_mean) * (y - y_mean))
    var_x = np.sum((x - x_mean) ** 2)

    if var_x == 0:
        return np.nan, np.nan, np.nan, np.nan

    beta = cov_xy / var_x
    alpha = y_mean - beta * x_mean

    # 计算R2
    y_pred = alpha + beta * x
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # 计算标准误
    n = len(y)
    if n > 2:
        mse = ss_res / (n - 2)
        se_beta = np.sqrt(mse / var_x)
    else:
        se_beta = np.nan

    return beta, alpha, r2, se_beta

def calculate_rolling_beta(stock_ret, market_ret, window=60):
    """计算滚动Beta"""
    n = len(stock_ret)
    betas = np.full(n, np.nan)

    for i in range(window, n):
        y = stock_ret[i-window:i]
        x = market_ret[i-window:i]
        beta, _, _, _ = estimate_beta_ols(y, x, min_obs=int(window * 0.8))
        betas[i] = beta

    return betas

def shrinkage_beta(beta_ols, shrinkage_target=1.0, shrinkage_factor=0.3):
    """收缩估计Beta (Vasicek调整)"""
    if np.isnan(beta_ols):
        return np.nan
    return shrinkage_factor * shrinkage_target + (1 - shrinkage_factor) * beta_ols

def blume_adjustment(beta_t, prior_coef=0.343, slope=0.677):
    """Blume调整 (基于历史均值回归)"""
    if np.isnan(beta_t):
        return np.nan
    return prior_coef + slope * beta_t

print("="*80)
print("A股市场 Beta 与系统性风险研究")
print("="*80)

# 加载数据
print("\n[1] 加载数据...")
market_df, stock_df = load_market_data()
print(f"市场数据: {len(market_df)} 天")
print(f"个股数据: {len(stock_df):,} 条记录")
print(f"股票数量: {stock_df['ts_code'].nunique()}")
print(f"数据范围: {market_df['trade_date'].min()} 至 {market_df['trade_date'].max()}")

# 合并数据
print("\n[2] 合并市场与个股数据...")
merged_df = stock_df.merge(market_df, on='trade_date', how='inner')
print(f"合并后记录数: {len(merged_df):,}")

# ============================================================
# Part 1: Beta 估计
# ============================================================
print("\n" + "="*80)
print("Part 1: Beta 估计")
print("="*80)

# 选择分析时间点 (2025年底)
analysis_date = '20251230'
latest_dates = merged_df[merged_df['trade_date'] <= pd.to_datetime(analysis_date)]['trade_date'].unique()
latest_dates = sorted(latest_dates)[-1]
print(f"\n分析基准日期: {latest_dates}")

# 计算不同窗口的Beta
windows = [60, 120, 250]
beta_results = {}

print("\n计算不同窗口的Beta...")
for window in windows:
    print(f"  窗口 {window} 天...")

    # 获取窗口内的交易日
    trading_dates = sorted(merged_df['trade_date'].unique())
    end_idx = trading_dates.index(latest_dates) if latest_dates in trading_dates else len(trading_dates) - 1
    start_idx = max(0, end_idx - window + 1)
    window_dates = trading_dates[start_idx:end_idx+1]

    window_data = merged_df[merged_df['trade_date'].isin(window_dates)]

    betas = []
    for ts_code in window_data['ts_code'].unique():
        stock_data = window_data[window_data['ts_code'] == ts_code].sort_values('trade_date')
        if len(stock_data) >= int(window * 0.6):  # 至少60%数据
            beta, alpha, r2, se = estimate_beta_ols(
                stock_data['ret'].values,
                stock_data['mkt_ret'].values,
                min_obs=int(window * 0.6)
            )
            if not np.isnan(beta):
                betas.append({
                    'ts_code': ts_code,
                    f'beta_{window}d': beta,
                    f'alpha_{window}d': alpha,
                    f'r2_{window}d': r2,
                    f'se_{window}d': se,
                    'industry': stock_data['industry'].iloc[0],
                    'name': stock_data['name'].iloc[0],
                    'total_mv': stock_data['total_mv'].iloc[-1],
                    'turnover_rate': stock_data['turnover_rate'].mean()
                })

    beta_results[window] = pd.DataFrame(betas)
    print(f"    有效股票数: {len(betas)}")

# 合并不同窗口的结果
beta_df = beta_results[60][['ts_code', 'beta_60d', 'alpha_60d', 'r2_60d', 'se_60d',
                             'industry', 'name', 'total_mv', 'turnover_rate']]
for w in [120, 250]:
    beta_df = beta_df.merge(
        beta_results[w][['ts_code', f'beta_{w}d', f'alpha_{w}d', f'r2_{w}d', f'se_{w}d']],
        on='ts_code', how='outer'
    )

print(f"\n最终样本量: {len(beta_df)} 只股票")

# 收缩估计
print("\n计算收缩Beta和Blume调整...")
beta_df['beta_shrink'] = beta_df['beta_250d'].apply(lambda x: shrinkage_beta(x, 1.0, 0.3))
beta_df['beta_blume'] = beta_df['beta_250d'].apply(lambda x: blume_adjustment(x))

# Beta统计描述
print("\n" + "-"*60)
print("Beta 统计描述")
print("-"*60)
for col in ['beta_60d', 'beta_120d', 'beta_250d', 'beta_shrink', 'beta_blume']:
    valid_data = beta_df[col].dropna()
    print(f"\n{col}:")
    print(f"  均值: {valid_data.mean():.4f}")
    print(f"  中位数: {valid_data.median():.4f}")
    print(f"  标准差: {valid_data.std():.4f}")
    print(f"  最小值: {valid_data.min():.4f}")
    print(f"  最大值: {valid_data.max():.4f}")
    print(f"  偏度: {valid_data.skew():.4f}")
    print(f"  峰度: {valid_data.kurtosis():.4f}")

# ============================================================
# Part 2: Beta 特征分析
# ============================================================
print("\n" + "="*80)
print("Part 2: Beta 特征分析")
print("="*80)

# 2.1 行业Beta差异
print("\n" + "-"*60)
print("2.1 行业Beta差异")
print("-"*60)
industry_beta = beta_df.groupby('industry').agg({
    'beta_250d': ['mean', 'std', 'count'],
    'total_mv': 'mean'
}).round(4)
industry_beta.columns = ['beta_mean', 'beta_std', 'count', 'avg_mv']
industry_beta = industry_beta[industry_beta['count'] >= 10].sort_values('beta_mean', ascending=False)

print("\n高Beta行业 (Top 10):")
print(industry_beta.head(10).to_string())
print("\n低Beta行业 (Bottom 10):")
print(industry_beta.tail(10).to_string())

# 2.2 市值与Beta关系
print("\n" + "-"*60)
print("2.2 市值与Beta关系")
print("-"*60)
beta_df['mv_group'] = pd.qcut(beta_df['total_mv'].rank(method='first'),
                               q=10, labels=[f'D{i}' for i in range(1, 11)])
mv_beta = beta_df.groupby('mv_group').agg({
    'beta_250d': ['mean', 'std'],
    'total_mv': 'mean'
}).round(4)
mv_beta.columns = ['beta_mean', 'beta_std', 'avg_mv_万']
mv_beta['avg_mv_亿'] = (mv_beta['avg_mv_万'] / 10000).round(2)
print(mv_beta.to_string())

# 相关系数
corr_mv_beta = beta_df[['total_mv', 'beta_250d']].dropna().corr().iloc[0, 1]
print(f"\n市值与Beta相关系数: {corr_mv_beta:.4f}")

# 2.3 Beta时变性分析
print("\n" + "-"*60)
print("2.3 Beta时变性分析 (不同窗口Beta相关性)")
print("-"*60)
beta_corr = beta_df[['beta_60d', 'beta_120d', 'beta_250d']].corr()
print(beta_corr.round(4).to_string())

# 窗口间差异
print("\n窗口间Beta差异:")
print(f"  60d vs 120d 平均差异: {(beta_df['beta_60d'] - beta_df['beta_120d']).mean():.4f}")
print(f"  60d vs 250d 平均差异: {(beta_df['beta_60d'] - beta_df['beta_250d']).mean():.4f}")
print(f"  120d vs 250d 平均差异: {(beta_df['beta_120d'] - beta_df['beta_250d']).mean():.4f}")

# ============================================================
# Part 3: 低Beta异象研究
# ============================================================
print("\n" + "="*80)
print("Part 3: 低Beta异象研究")
print("="*80)

# 按Beta分组
print("\n" + "-"*60)
print("3.1 Beta分组收益分析")
print("-"*60)

# 计算过去一年的收益率
one_year_ago = pd.to_datetime(analysis_date) - pd.Timedelta(days=365)
recent_returns = merged_df[merged_df['trade_date'] >= one_year_ago].groupby('ts_code')['ret'].sum().reset_index()
recent_returns.columns = ['ts_code', 'annual_ret']

beta_with_ret = beta_df.merge(recent_returns, on='ts_code', how='inner')
beta_with_ret['beta_group'] = pd.qcut(beta_with_ret['beta_250d'].rank(method='first'),
                                       q=5, labels=['低Beta', 'Q2', 'Q3', 'Q4', '高Beta'])

beta_group_stats = beta_with_ret.groupby('beta_group').agg({
    'beta_250d': 'mean',
    'annual_ret': ['mean', 'std'],
    'total_mv': 'mean',
    'ts_code': 'count'
}).round(4)
beta_group_stats.columns = ['avg_beta', 'avg_return', 'ret_std', 'avg_mv', 'count']
beta_group_stats['sharpe'] = (beta_group_stats['avg_return'] / beta_group_stats['ret_std']).round(4)
print(beta_group_stats.to_string())

# 低Beta溢价
low_beta_ret = beta_group_stats.loc['低Beta', 'avg_return']
high_beta_ret = beta_group_stats.loc['高Beta', 'avg_return']
print(f"\n低Beta组合年化收益: {low_beta_ret:.2f}%")
print(f"高Beta组合年化收益: {high_beta_ret:.2f}%")
print(f"低Beta溢价 (BAB): {low_beta_ret - high_beta_ret:.2f}%")

# 3.2 彩票效应分析
print("\n" + "-"*60)
print("3.2 彩票效应分析")
print("-"*60)

# 计算收益率偏度和最大收益
ret_stats = merged_df[merged_df['trade_date'] >= one_year_ago].groupby('ts_code').agg({
    'ret': ['skew', 'max', 'std']
}).reset_index()
ret_stats.columns = ['ts_code', 'ret_skew', 'ret_max', 'ret_vol']

lottery_df = beta_df.merge(ret_stats, on='ts_code', how='inner')

# 高偏度（彩票型）股票的Beta特征
lottery_df['lottery_group'] = pd.qcut(lottery_df['ret_skew'].rank(method='first'),
                                       q=5, labels=['低偏度', 'Q2', 'Q3', 'Q4', '高偏度'])
lottery_stats = lottery_df.groupby('lottery_group').agg({
    'beta_250d': 'mean',
    'ret_skew': 'mean',
    'ret_vol': 'mean',
    'total_mv': 'mean'
}).round(4)
print(lottery_stats.to_string())

print(f"\n高偏度(彩票型)股票平均Beta: {lottery_stats.loc['高偏度', 'beta_250d']:.4f}")
print(f"低偏度股票平均Beta: {lottery_stats.loc['低偏度', 'beta_250d']:.4f}")

# ============================================================
# Part 4: Beta预测
# ============================================================
print("\n" + "="*80)
print("Part 4: Beta预测与均值回归")
print("="*80)

# 4.1 Beta均值回归检验
print("\n" + "-"*60)
print("4.1 Beta均值回归检验")
print("-"*60)

# 使用更早的数据作为历史Beta
historical_date = '20240630'
hist_trading_dates = sorted(merged_df['trade_date'].unique())
hist_end_idx = None
for i, d in enumerate(hist_trading_dates):
    if d >= pd.to_datetime(historical_date):
        hist_end_idx = i
        break
if hist_end_idx is None:
    hist_end_idx = len(hist_trading_dates) // 2

hist_window = 250
hist_start_idx = max(0, hist_end_idx - hist_window + 1)
hist_window_dates = hist_trading_dates[hist_start_idx:hist_end_idx+1]

hist_data = merged_df[merged_df['trade_date'].isin(hist_window_dates)]
hist_betas = []
for ts_code in hist_data['ts_code'].unique():
    stock_data = hist_data[hist_data['ts_code'] == ts_code].sort_values('trade_date')
    if len(stock_data) >= int(hist_window * 0.6):
        beta, _, _, _ = estimate_beta_ols(
            stock_data['ret'].values,
            stock_data['mkt_ret'].values,
            min_obs=int(hist_window * 0.6)
        )
        if not np.isnan(beta):
            hist_betas.append({'ts_code': ts_code, 'beta_hist': beta})

hist_beta_df = pd.DataFrame(hist_betas)
print(f"历史Beta样本量: {len(hist_beta_df)}")

# 合并历史和当前Beta
reversion_df = beta_df[['ts_code', 'beta_250d']].merge(hist_beta_df, on='ts_code', how='inner')
reversion_df = reversion_df.dropna()
print(f"配对样本量: {len(reversion_df)}")

# 回归分析: beta_t+1 = a + b * beta_t
slope, intercept, r_value, p_value, std_err = stats.linregress(
    reversion_df['beta_hist'], reversion_df['beta_250d']
)
print(f"\nBeta预测回归: Beta(t+1) = {intercept:.4f} + {slope:.4f} * Beta(t)")
print(f"R²: {r_value**2:.4f}")
print(f"p值: {p_value:.6f}")

# Blume参数估计
print(f"\nBlume调整参数估计:")
print(f"  截距 (向均值收缩): {intercept:.4f}")
print(f"  斜率 (持续性): {slope:.4f}")
print(f"  隐含均值目标: {intercept / (1 - slope):.4f}")

# 4.2 分组均值回归
print("\n" + "-"*60)
print("4.2 分组Beta均值回归")
print("-"*60)
reversion_df['hist_beta_group'] = pd.qcut(reversion_df['beta_hist'].rank(method='first'),
                                           q=5, labels=['低Beta', 'Q2', 'Q3', 'Q4', '高Beta'])
group_reversion = reversion_df.groupby('hist_beta_group').agg({
    'beta_hist': 'mean',
    'beta_250d': 'mean'
}).round(4)
group_reversion['change'] = group_reversion['beta_250d'] - group_reversion['beta_hist']
print(group_reversion.to_string())

# ============================================================
# Part 5: Beta策略
# ============================================================
print("\n" + "="*80)
print("Part 5: Beta策略回测")
print("="*80)

# 5.1 低Beta组合策略
print("\n" + "-"*60)
print("5.1 低Beta组合策略")
print("-"*60)

# 选择2024年数据进行回测
backtest_start = '20240101'
backtest_end = '20241231'
backtest_data = merged_df[(merged_df['trade_date'] >= pd.to_datetime(backtest_start)) &
                          (merged_df['trade_date'] <= pd.to_datetime(backtest_end))]

# 月度调仓
backtest_data['month'] = backtest_data['trade_date'].dt.to_period('M')
months = sorted(backtest_data['month'].unique())

portfolio_returns = {'low_beta': [], 'high_beta': [], 'market': [], 'beta_neutral': []}

for i, month in enumerate(months[1:], 1):  # 从第二个月开始
    prev_month = months[i-1]

    # 使用上月数据计算Beta
    prev_data = backtest_data[backtest_data['month'] == prev_month]
    if len(prev_data['ts_code'].unique()) < 100:
        continue

    # 简化：使用上月收益率与市场收益率的协方差作为Beta近似
    monthly_stats = prev_data.groupby('ts_code').agg({
        'ret': 'mean',
        'mkt_ret': 'mean'
    }).reset_index()

    # 计算月度Beta排序（使用与市场的协方差/方差）
    stock_returns = prev_data.pivot(index='trade_date', columns='ts_code', values='ret')
    market_returns = prev_data.groupby('trade_date')['mkt_ret'].first()

    monthly_betas = {}
    for col in stock_returns.columns:
        valid = ~(stock_returns[col].isna() | market_returns.isna())
        if valid.sum() >= 10:
            cov = np.cov(stock_returns[col][valid], market_returns[valid])[0, 1]
            var = market_returns[valid].var()
            if var > 0:
                monthly_betas[col] = cov / var

    if len(monthly_betas) < 100:
        continue

    beta_ranking = pd.Series(monthly_betas).sort_values()
    n_stocks = len(beta_ranking) // 5

    low_beta_stocks = beta_ranking.head(n_stocks).index.tolist()
    high_beta_stocks = beta_ranking.tail(n_stocks).index.tolist()

    # 计算本月收益
    curr_data = backtest_data[backtest_data['month'] == month]

    # 低Beta组合收益
    low_beta_ret = curr_data[curr_data['ts_code'].isin(low_beta_stocks)].groupby('trade_date')['ret'].mean()
    high_beta_ret = curr_data[curr_data['ts_code'].isin(high_beta_stocks)].groupby('trade_date')['ret'].mean()
    market_ret = curr_data.groupby('trade_date')['mkt_ret'].first()

    portfolio_returns['low_beta'].extend(low_beta_ret.values)
    portfolio_returns['high_beta'].extend(high_beta_ret.values)
    portfolio_returns['market'].extend(market_ret.values)

    # Beta中性组合: 做多低Beta，做空高Beta
    neutral_ret = (low_beta_ret.values - high_beta_ret.values) / 2
    portfolio_returns['beta_neutral'].extend(neutral_ret)

# 计算策略统计
print("\n2024年策略表现:")
for strategy, returns in portfolio_returns.items():
    if len(returns) > 0:
        returns = np.array(returns)
        cum_ret = (1 + returns/100).prod() - 1
        ann_ret = cum_ret  # 约一年
        ann_vol = returns.std() * np.sqrt(250)
        sharpe = (ann_ret * 100) / ann_vol if ann_vol > 0 else 0
        max_dd = 0
        peak = 1
        cum = 1
        for r in returns:
            cum = cum * (1 + r/100)
            if cum > peak:
                peak = cum
            dd = (peak - cum) / peak
            if dd > max_dd:
                max_dd = dd

        print(f"\n{strategy}:")
        print(f"  累计收益: {cum_ret*100:.2f}%")
        print(f"  年化波动: {ann_vol:.2f}%")
        print(f"  Sharpe: {sharpe:.4f}")
        print(f"  最大回撤: {max_dd*100:.2f}%")

# 5.2 市场择时策略
print("\n" + "-"*60)
print("5.2 基于市场状态的Beta择时")
print("-"*60)

# 判断市场状态（使用20日收益率）
market_state = market_df.copy()
market_state['ret_20d'] = market_state['mkt_ret'].rolling(20).sum()
market_state['state'] = market_state['ret_20d'].apply(
    lambda x: '牛市' if x > 5 else ('熊市' if x < -5 else '震荡')
)

# 不同市场状态下的Beta表现
print("\n不同市场状态下的Beta组合表现:")
state_perf = market_state.groupby('state').agg({
    'mkt_ret': ['mean', 'std', 'count']
}).round(4)
state_perf.columns = ['avg_ret', 'std_ret', 'days']
print(state_perf.to_string())

# ============================================================
# Part 6: 滚动Beta分析
# ============================================================
print("\n" + "="*80)
print("Part 6: 滚动Beta时间序列分析")
print("="*80)

# 选择代表性股票进行滚动Beta分析
sample_stocks = ['000001.SZ', '600519.SH', '000858.SZ', '601318.SH', '000333.SZ']
sample_names = {'000001.SZ': '平安银行', '600519.SH': '贵州茅台',
                '000858.SZ': '五粮液', '601318.SH': '中国平安', '000333.SZ': '美的集团'}

print("\n代表性股票滚动Beta (60日窗口) 统计:")
for ts_code in sample_stocks:
    stock_data = merged_df[merged_df['ts_code'] == ts_code].sort_values('trade_date')
    if len(stock_data) > 100:
        rolling_betas = calculate_rolling_beta(
            stock_data['ret'].values,
            stock_data['mkt_ret'].values,
            window=60
        )
        valid_betas = rolling_betas[~np.isnan(rolling_betas)]
        if len(valid_betas) > 0:
            print(f"\n{sample_names.get(ts_code, ts_code)}:")
            print(f"  当前Beta: {valid_betas[-1]:.4f}")
            print(f"  平均Beta: {valid_betas.mean():.4f}")
            print(f"  Beta波动: {valid_betas.std():.4f}")
            print(f"  最高Beta: {valid_betas.max():.4f}")
            print(f"  最低Beta: {valid_betas.min():.4f}")

# ============================================================
# Part 7: R² 和系统性风险占比
# ============================================================
print("\n" + "="*80)
print("Part 7: R² 和系统性风险占比分析")
print("="*80)

print("\n" + "-"*60)
print("7.1 R² 分布统计")
print("-"*60)
for col in ['r2_60d', 'r2_120d', 'r2_250d']:
    valid_r2 = beta_df[col].dropna()
    print(f"\n{col}:")
    print(f"  均值: {valid_r2.mean():.4f}")
    print(f"  中位数: {valid_r2.median():.4f}")
    print(f"  标准差: {valid_r2.std():.4f}")
    print(f"  <10%占比: {(valid_r2 < 0.1).mean()*100:.2f}%")
    print(f"  >30%占比: {(valid_r2 > 0.3).mean()*100:.2f}%")

print("\n" + "-"*60)
print("7.2 行业R²差异")
print("-"*60)
industry_r2 = beta_df.groupby('industry')['r2_250d'].agg(['mean', 'count'])
industry_r2 = industry_r2[industry_r2['count'] >= 10].sort_values('mean', ascending=False)
print("\n高R²行业 (系统性风险占比高):")
print(industry_r2.head(10).round(4).to_string())
print("\n低R²行业 (特质风险占比高):")
print(industry_r2.tail(10).round(4).to_string())

# ============================================================
# 保存结果
# ============================================================
print("\n" + "="*80)
print("保存分析结果")
print("="*80)

# 保存Beta数据
beta_df.to_csv('/Users/allen/workspace/python/stock/Tushare-DuckDB/reports/quality_check/beta_estimates.csv', index=False)
print("Beta估计数据已保存到 beta_estimates.csv")

# 保存行业Beta
industry_beta.to_csv('/Users/allen/workspace/python/stock/Tushare-DuckDB/reports/quality_check/industry_beta.csv')
print("行业Beta数据已保存到 industry_beta.csv")

print("\n分析完成！")
