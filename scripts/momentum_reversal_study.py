#!/usr/bin/env python3
"""
A股动量与反转效应深度研究
============================
研究内容:
1. 动量效应研究 - 不同形成期和持有期
2. 反转效应研究 - 短期和长期
3. 横截面动量 vs 时序动量
4. 动量策略改进 - 残差动量、52周高点、成交量加权
5. 风险调整 - 波动率调整、行业中性化
6. 策略回测与分析

Author: Momentum Research
Date: 2026-01
"""

import duckdb
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 数据库路径
DB_PATH = '/Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db'
REPORT_PATH = '/Users/allen/workspace/python/stock/Tushare-DuckDB/reports/quality_check/momentum_reversal_study.md'

def get_connection():
    """获取数据库连接"""
    return duckdb.connect(DB_PATH, read_only=True)

def load_daily_data(conn, start_date='20100101', end_date='20251231'):
    """加载日线数据并计算复权价格"""
    print(f"Loading daily data from {start_date} to {end_date}...")

    query = """
    WITH adj_data AS (
        SELECT
            d.ts_code,
            d.trade_date,
            d.close,
            d.pct_chg,
            d.vol,
            d.amount,
            a.adj_factor,
            d.close * a.adj_factor AS adj_close,
            db.total_mv,
            db.circ_mv,
            db.turnover_rate
        FROM daily d
        LEFT JOIN adj_factor a ON d.ts_code = a.ts_code AND d.trade_date = a.trade_date
        LEFT JOIN daily_basic db ON d.ts_code = db.ts_code AND d.trade_date = db.trade_date
        WHERE d.trade_date >= ? AND d.trade_date <= ?
            AND d.ts_code NOT LIKE '%BJ%'  -- 排除北交所
            AND d.close > 0
            AND a.adj_factor IS NOT NULL
    )
    SELECT * FROM adj_data
    ORDER BY ts_code, trade_date
    """

    df = conn.execute(query, [start_date, end_date]).fetchdf()
    print(f"Loaded {len(df):,} rows, {df['ts_code'].nunique():,} stocks")
    return df

def load_industry_data(conn):
    """加载行业分类数据"""
    query = """
    SELECT ts_code, l1_name as industry
    FROM index_member_all
    WHERE is_new = 'Y'
    """
    return conn.execute(query).fetchdf()

def get_trade_dates(conn, start_date='20100101', end_date='20251231'):
    """获取交易日历"""
    query = """
    SELECT DISTINCT trade_date
    FROM daily
    WHERE trade_date >= ? AND trade_date <= ?
    ORDER BY trade_date
    """
    dates = conn.execute(query, [start_date, end_date]).fetchdf()
    return dates['trade_date'].tolist()

def calculate_returns(df, periods=[5, 21, 63, 126, 252]):
    """计算不同周期的收益率"""
    print("Calculating returns for different periods...")

    # 按股票分组计算
    df = df.sort_values(['ts_code', 'trade_date'])

    for period in periods:
        # 过去收益（动量形成期）
        df[f'ret_{period}d'] = df.groupby('ts_code')['adj_close'].pct_change(period)
        # 未来收益（持有期）
        df[f'fwd_ret_{period}d'] = df.groupby('ts_code')['adj_close'].pct_change(period).shift(-period)

    # 计算日收益率
    df['daily_ret'] = df.groupby('ts_code')['adj_close'].pct_change()

    return df

def calculate_volatility(df, windows=[21, 63]):
    """计算波动率"""
    print("Calculating volatility...")

    for window in windows:
        df[f'vol_{window}d'] = df.groupby('ts_code')['daily_ret'].transform(
            lambda x: x.rolling(window, min_periods=window//2).std() * np.sqrt(252)
        )

    return df

def calculate_52week_high(df):
    """计算52周高点动量"""
    print("Calculating 52-week high momentum...")

    df['high_252d'] = df.groupby('ts_code')['adj_close'].transform(
        lambda x: x.rolling(252, min_periods=126).max()
    )
    df['dist_to_52wk_high'] = df['adj_close'] / df['high_252d'] - 1

    return df

def calculate_volume_weighted_momentum(df, period=21):
    """计算成交量加权动量"""
    print("Calculating volume-weighted momentum...")

    df['vol_weight'] = df.groupby('ts_code')['amount'].transform(
        lambda x: x / x.rolling(period, min_periods=period//2).sum()
    )
    df['vw_momentum'] = df.groupby('ts_code').apply(
        lambda x: (x['daily_ret'] * x['vol_weight']).rolling(period, min_periods=period//2).sum()
    ).reset_index(level=0, drop=True)

    return df

def momentum_portfolio_analysis(df, formation_period, holding_period, n_quantiles=5):
    """
    动量组合分析

    Parameters:
    -----------
    df : DataFrame
        包含收益率数据的DataFrame
    formation_period : int
        形成期天数
    holding_period : int
        持有期天数
    n_quantiles : int
        分组数量

    Returns:
    --------
    dict : 包含分析结果的字典
    """
    print(f"Analyzing momentum portfolio: formation={formation_period}d, holding={holding_period}d")

    formation_col = f'ret_{formation_period}d'
    holding_col = f'fwd_ret_{holding_period}d'

    if formation_col not in df.columns or holding_col not in df.columns:
        return None

    # 筛选有效数据
    valid_df = df[[
        'ts_code', 'trade_date', formation_col, holding_col, 'total_mv'
    ]].dropna()

    if len(valid_df) < 1000:
        return None

    # 按日期分组，计算分位数
    results = []

    for date, group in valid_df.groupby('trade_date'):
        if len(group) < 50:  # 至少50只股票
            continue

        try:
            # 按动量分组
            group['quantile'] = pd.qcut(
                group[formation_col],
                n_quantiles,
                labels=range(1, n_quantiles + 1),
                duplicates='drop'
            )

            # 计算每组的平均收益
            for q in range(1, n_quantiles + 1):
                q_data = group[group['quantile'] == q]
                if len(q_data) > 0:
                    results.append({
                        'trade_date': date,
                        'quantile': q,
                        'avg_ret': q_data[holding_col].mean(),
                        'n_stocks': len(q_data)
                    })
        except:
            continue

    if not results:
        return None

    result_df = pd.DataFrame(results)

    # 计算各分位数的平均收益
    quantile_returns = result_df.groupby('quantile')['avg_ret'].agg(['mean', 'std', 'count'])
    quantile_returns.columns = ['mean_ret', 'std_ret', 'n_periods']
    quantile_returns['sharpe'] = quantile_returns['mean_ret'] / quantile_returns['std_ret'] * np.sqrt(252 / holding_period)

    # 计算多空收益
    winner_ret = result_df[result_df['quantile'] == n_quantiles]['avg_ret'].values
    loser_ret = result_df[result_df['quantile'] == 1]['avg_ret'].values

    # 对齐长度
    min_len = min(len(winner_ret), len(loser_ret))
    if min_len > 0:
        long_short_ret = winner_ret[:min_len] - loser_ret[:min_len]
        long_short_mean = np.mean(long_short_ret)
        long_short_std = np.std(long_short_ret)
        long_short_sharpe = long_short_mean / long_short_std * np.sqrt(252 / holding_period) if long_short_std > 0 else 0
        win_rate = np.mean(long_short_ret > 0)
    else:
        long_short_mean = 0
        long_short_sharpe = 0
        win_rate = 0

    return {
        'formation_period': formation_period,
        'holding_period': holding_period,
        'quantile_returns': quantile_returns,
        'long_short_mean': long_short_mean,
        'long_short_sharpe': long_short_sharpe,
        'win_rate': win_rate,
        'n_periods': min_len
    }

def analyze_momentum_by_year(df, formation_period=126, holding_period=21, n_quantiles=5):
    """分年度动量分析"""
    print("Analyzing momentum by year...")

    formation_col = f'ret_{formation_period}d'
    holding_col = f'fwd_ret_{holding_period}d'

    df['year'] = df['trade_date'].str[:4]

    yearly_results = []

    for year in sorted(df['year'].unique()):
        year_df = df[df['year'] == year]
        valid_df = year_df[['ts_code', 'trade_date', formation_col, holding_col]].dropna()

        if len(valid_df) < 500:
            continue

        # 计算分位数收益
        results = []
        for date, group in valid_df.groupby('trade_date'):
            if len(group) < 30:
                continue
            try:
                group['quantile'] = pd.qcut(
                    group[formation_col],
                    n_quantiles,
                    labels=range(1, n_quantiles + 1),
                    duplicates='drop'
                )

                for q in [1, n_quantiles]:  # 只看极端组
                    q_data = group[group['quantile'] == q]
                    if len(q_data) > 0:
                        results.append({
                            'quantile': q,
                            'avg_ret': q_data[holding_col].mean()
                        })
            except:
                continue

        if results:
            result_df = pd.DataFrame(results)
            winner_ret = result_df[result_df['quantile'] == n_quantiles]['avg_ret'].mean()
            loser_ret = result_df[result_df['quantile'] == 1]['avg_ret'].mean()

            yearly_results.append({
                'year': year,
                'winner_ret': winner_ret,
                'loser_ret': loser_ret,
                'long_short_ret': winner_ret - loser_ret,
                'momentum_works': winner_ret > loser_ret
            })

    return pd.DataFrame(yearly_results)

def analyze_momentum_crash(df, formation_period=126, holding_period=21):
    """分析动量崩溃"""
    print("Analyzing momentum crash...")

    formation_col = f'ret_{formation_period}d'
    holding_col = f'fwd_ret_{holding_period}d'

    # 计算市场收益
    market_ret = df.groupby('trade_date')[holding_col].mean().reset_index()
    market_ret.columns = ['trade_date', 'market_ret']

    # 计算动量策略收益
    results = []
    for date, group in df.groupby('trade_date'):
        valid = group[[formation_col, holding_col]].dropna()
        if len(valid) < 50:
            continue

        try:
            valid['quantile'] = pd.qcut(valid[formation_col], 5, labels=[1,2,3,4,5], duplicates='drop')
            winner_ret = valid[valid['quantile'] == 5][holding_col].mean()
            loser_ret = valid[valid['quantile'] == 1][holding_col].mean()
            results.append({
                'trade_date': date,
                'mom_ret': winner_ret - loser_ret
            })
        except:
            continue

    mom_df = pd.DataFrame(results)
    mom_df = mom_df.merge(market_ret, on='trade_date')

    # 识别市场大跌后的反弹（动量崩溃典型场景）
    mom_df['market_ret_lag'] = mom_df['market_ret'].shift(1)

    # 市场下跌后反弹的情况
    crash_condition = (mom_df['market_ret_lag'] < -0.05) & (mom_df['market_ret'] > 0.03)
    crash_periods = mom_df[crash_condition]

    normal_periods = mom_df[~crash_condition]

    return {
        'crash_mom_ret': crash_periods['mom_ret'].mean() if len(crash_periods) > 0 else 0,
        'normal_mom_ret': normal_periods['mom_ret'].mean() if len(normal_periods) > 0 else 0,
        'n_crash_periods': len(crash_periods),
        'n_normal_periods': len(normal_periods)
    }

def analyze_short_term_reversal(df, formation_period=5, holding_period=5):
    """短期反转分析"""
    print("Analyzing short-term reversal...")

    formation_col = f'ret_{formation_period}d'
    holding_col = f'fwd_ret_{holding_period}d'

    if formation_col not in df.columns:
        df[formation_col] = df.groupby('ts_code')['adj_close'].pct_change(formation_period)
    if holding_col not in df.columns:
        df[holding_col] = df.groupby('ts_code')['adj_close'].pct_change(holding_period).shift(-holding_period)

    valid_df = df[[formation_col, holding_col, 'trade_date']].dropna()

    results = []
    for date, group in valid_df.groupby('trade_date'):
        if len(group) < 50:
            continue

        try:
            group['quantile'] = pd.qcut(group[formation_col], 5, labels=[1,2,3,4,5], duplicates='drop')
            loser_ret = group[group['quantile'] == 1][holding_col].mean()
            winner_ret = group[group['quantile'] == 5][holding_col].mean()
            results.append({
                'trade_date': date,
                'reversal_ret': loser_ret - winner_ret  # 买输家卖赢家
            })
        except:
            continue

    if results:
        result_df = pd.DataFrame(results)
        return {
            'mean_reversal_ret': result_df['reversal_ret'].mean(),
            'std_reversal_ret': result_df['reversal_ret'].std(),
            'sharpe': result_df['reversal_ret'].mean() / result_df['reversal_ret'].std() * np.sqrt(252 / holding_period),
            'win_rate': (result_df['reversal_ret'] > 0).mean(),
            'n_periods': len(result_df)
        }
    return None

def analyze_time_series_momentum(df, lookback=252, holding=21):
    """时序动量分析"""
    print("Analyzing time-series momentum...")

    formation_col = f'ret_{lookback}d' if lookback in [5, 21, 63, 126, 252] else None
    holding_col = f'fwd_ret_{holding}d'

    if formation_col is None:
        df[f'ret_{lookback}d'] = df.groupby('ts_code')['adj_close'].pct_change(lookback)
        formation_col = f'ret_{lookback}d'

    valid_df = df[[formation_col, holding_col, 'trade_date', 'ts_code']].dropna()

    # 时序动量: 过去收益为正则做多，否则做空
    valid_df['ts_signal'] = np.sign(valid_df[formation_col])
    valid_df['ts_ret'] = valid_df['ts_signal'] * valid_df[holding_col]

    # 按日期汇总
    ts_results = valid_df.groupby('trade_date').agg({
        'ts_ret': 'mean',
        'ts_code': 'count'
    }).reset_index()
    ts_results.columns = ['trade_date', 'ts_mom_ret', 'n_stocks']

    return {
        'mean_ret': ts_results['ts_mom_ret'].mean(),
        'std_ret': ts_results['ts_mom_ret'].std(),
        'sharpe': ts_results['ts_mom_ret'].mean() / ts_results['ts_mom_ret'].std() * np.sqrt(252 / holding),
        'win_rate': (ts_results['ts_mom_ret'] > 0).mean(),
        'n_periods': len(ts_results)
    }

def analyze_residual_momentum(df, industry_df, formation_period=126, holding_period=21):
    """残差动量分析（剥离行业效应）"""
    print("Analyzing residual momentum...")

    # 合并行业数据
    df_with_ind = df.merge(industry_df, on='ts_code', how='left')
    df_with_ind = df_with_ind.dropna(subset=['industry'])

    formation_col = f'ret_{formation_period}d'
    holding_col = f'fwd_ret_{holding_period}d'

    valid_df = df_with_ind[['ts_code', 'trade_date', 'industry', formation_col, holding_col]].dropna()

    # 计算残差动量
    results = []
    for date, group in valid_df.groupby('trade_date'):
        if len(group) < 100:
            continue

        # 计算行业平均动量
        ind_mom = group.groupby('industry')[formation_col].mean()
        group['ind_mom'] = group['industry'].map(ind_mom)
        group['residual_mom'] = group[formation_col] - group['ind_mom']

        try:
            # 按残差动量分组
            group['quantile'] = pd.qcut(group['residual_mom'], 5, labels=[1,2,3,4,5], duplicates='drop')
            winner_ret = group[group['quantile'] == 5][holding_col].mean()
            loser_ret = group[group['quantile'] == 1][holding_col].mean()

            results.append({
                'trade_date': date,
                'residual_mom_ret': winner_ret - loser_ret
            })
        except:
            continue

    if results:
        result_df = pd.DataFrame(results)
        return {
            'mean_ret': result_df['residual_mom_ret'].mean(),
            'std_ret': result_df['residual_mom_ret'].std(),
            'sharpe': result_df['residual_mom_ret'].mean() / result_df['residual_mom_ret'].std() * np.sqrt(252 / holding_period),
            'win_rate': (result_df['residual_mom_ret'] > 0).mean(),
            'n_periods': len(result_df)
        }
    return None

def analyze_52week_high_momentum(df, holding_period=21):
    """52周高点动量分析"""
    print("Analyzing 52-week high momentum...")

    holding_col = f'fwd_ret_{holding_period}d'

    valid_df = df[['ts_code', 'trade_date', 'dist_to_52wk_high', holding_col]].dropna()

    results = []
    for date, group in valid_df.groupby('trade_date'):
        if len(group) < 50:
            continue

        try:
            group['quantile'] = pd.qcut(group['dist_to_52wk_high'], 5, labels=[1,2,3,4,5], duplicates='drop')
            winner_ret = group[group['quantile'] == 5][holding_col].mean()  # 接近52周高点
            loser_ret = group[group['quantile'] == 1][holding_col].mean()   # 远离52周高点

            results.append({
                'trade_date': date,
                'high52_ret': winner_ret - loser_ret
            })
        except:
            continue

    if results:
        result_df = pd.DataFrame(results)
        return {
            'mean_ret': result_df['high52_ret'].mean(),
            'std_ret': result_df['high52_ret'].std(),
            'sharpe': result_df['high52_ret'].mean() / result_df['high52_ret'].std() * np.sqrt(252 / holding_period),
            'win_rate': (result_df['high52_ret'] > 0).mean(),
            'n_periods': len(result_df)
        }
    return None

def analyze_volatility_adjusted_momentum(df, formation_period=126, holding_period=21):
    """波动率调整动量分析"""
    print("Analyzing volatility-adjusted momentum...")

    formation_col = f'ret_{formation_period}d'
    holding_col = f'fwd_ret_{holding_period}d'
    vol_col = 'vol_63d'

    valid_df = df[['ts_code', 'trade_date', formation_col, holding_col, vol_col]].dropna()
    valid_df = valid_df[valid_df[vol_col] > 0]

    # 波动率调整动量
    valid_df['vol_adj_mom'] = valid_df[formation_col] / valid_df[vol_col]

    results = []
    for date, group in valid_df.groupby('trade_date'):
        if len(group) < 50:
            continue

        try:
            group['quantile'] = pd.qcut(group['vol_adj_mom'], 5, labels=[1,2,3,4,5], duplicates='drop')
            winner_ret = group[group['quantile'] == 5][holding_col].mean()
            loser_ret = group[group['quantile'] == 1][holding_col].mean()

            results.append({
                'trade_date': date,
                'vol_adj_mom_ret': winner_ret - loser_ret
            })
        except:
            continue

    if results:
        result_df = pd.DataFrame(results)
        return {
            'mean_ret': result_df['vol_adj_mom_ret'].mean(),
            'std_ret': result_df['vol_adj_mom_ret'].std(),
            'sharpe': result_df['vol_adj_mom_ret'].mean() / result_df['vol_adj_mom_ret'].std() * np.sqrt(252 / holding_period),
            'win_rate': (result_df['vol_adj_mom_ret'] > 0).mean(),
            'n_periods': len(result_df)
        }
    return None

def analyze_industry_neutral_momentum(df, industry_df, formation_period=126, holding_period=21):
    """行业中性动量分析"""
    print("Analyzing industry-neutral momentum...")

    # 合并行业数据
    df_with_ind = df.merge(industry_df, on='ts_code', how='left')
    df_with_ind = df_with_ind.dropna(subset=['industry'])

    formation_col = f'ret_{formation_period}d'
    holding_col = f'fwd_ret_{holding_period}d'

    valid_df = df_with_ind[['ts_code', 'trade_date', 'industry', formation_col, holding_col]].dropna()

    results = []
    for date, group in valid_df.groupby('trade_date'):
        if len(group) < 100:
            continue

        industry_rets = []

        # 在每个行业内部做动量排序
        for ind, ind_group in group.groupby('industry'):
            if len(ind_group) < 5:
                continue

            try:
                ind_group['quantile'] = pd.qcut(ind_group[formation_col], 3, labels=[1,2,3], duplicates='drop')
                winner_ret = ind_group[ind_group['quantile'] == 3][holding_col].mean()
                loser_ret = ind_group[ind_group['quantile'] == 1][holding_col].mean()
                industry_rets.append(winner_ret - loser_ret)
            except:
                continue

        if industry_rets:
            results.append({
                'trade_date': date,
                'ind_neutral_ret': np.mean(industry_rets)
            })

    if results:
        result_df = pd.DataFrame(results)
        return {
            'mean_ret': result_df['ind_neutral_ret'].mean(),
            'std_ret': result_df['ind_neutral_ret'].std(),
            'sharpe': result_df['ind_neutral_ret'].mean() / result_df['ind_neutral_ret'].std() * np.sqrt(252 / holding_period),
            'win_rate': (result_df['ind_neutral_ret'] > 0).mean(),
            'n_periods': len(result_df)
        }
    return None

def calculate_transaction_cost_impact(base_return, turnover_rate=1.0, cost_bps=30):
    """计算交易成本影响"""
    cost_per_period = turnover_rate * cost_bps / 10000
    net_return = base_return - cost_per_period
    return net_return

def generate_report(results, report_path):
    """生成研究报告"""
    print(f"Generating report to {report_path}...")

    report = []
    report.append("# A股动量与反转效应深度研究报告")
    report.append("")
    report.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    report.append("---")
    report.append("")

    # 1. 研究概述
    report.append("## 1. 研究概述")
    report.append("")
    report.append("本研究对A股市场的动量和反转效应进行了全面深入的分析，主要包括：")
    report.append("")
    report.append("- **动量效应**：不同形成期和持有期的动量策略表现")
    report.append("- **反转效应**：短期反转和长期反转的收益特征")
    report.append("- **横截面动量 vs 时序动量**：两种动量策略的比较")
    report.append("- **动量策略改进**：残差动量、52周高点动量、波动率调整动量")
    report.append("- **风险调整**：行业中性化处理")
    report.append("- **分年度分析**：动量效应的时变性")
    report.append("")

    # 2. 数据说明
    report.append("## 2. 数据说明")
    report.append("")
    if 'data_info' in results:
        info = results['data_info']
        report.append(f"- **数据时间范围**: {info.get('start_date', 'N/A')} - {info.get('end_date', 'N/A')}")
        report.append(f"- **股票数量**: {info.get('n_stocks', 'N/A'):,}")
        report.append(f"- **观测数量**: {info.get('n_obs', 'N/A'):,}")
    report.append("")

    # 3. 动量效应分析
    report.append("## 3. 动量效应分析")
    report.append("")

    # 3.1 不同形成期和持有期的动量策略
    report.append("### 3.1 不同形成期和持有期组合")
    report.append("")

    if 'momentum_matrix' in results:
        report.append("| 形成期 \\ 持有期 | 5天 | 21天 | 63天 | 126天 |")
        report.append("|:---:|:---:|:---:|:---:|:---:|")

        for form_period in [21, 63, 126, 252]:
            row = f"| {form_period}天 |"
            for hold_period in [5, 21, 63, 126]:
                key = f"{form_period}_{hold_period}"
                if key in results['momentum_matrix']:
                    val = results['momentum_matrix'][key]
                    sharpe = val.get('long_short_sharpe', 0)
                    row += f" {sharpe:.2f} |"
                else:
                    row += " N/A |"
            report.append(row)

        report.append("")
        report.append("*表格数值为多空组合的年化夏普比率*")
        report.append("")

    # 3.2 最佳动量策略详情
    report.append("### 3.2 最佳动量策略详情")
    report.append("")

    if 'best_momentum' in results:
        best = results['best_momentum']
        report.append(f"**最佳配置**: 形成期 {best.get('formation_period', 'N/A')} 天，持有期 {best.get('holding_period', 'N/A')} 天")
        report.append("")
        report.append("| 指标 | 数值 |")
        report.append("|:---|:---:|")
        report.append(f"| 多空收益(日均) | {best.get('long_short_mean', 0)*100:.4f}% |")
        report.append(f"| 年化夏普比率 | {best.get('long_short_sharpe', 0):.2f} |")
        report.append(f"| 胜率 | {best.get('win_rate', 0)*100:.1f}% |")
        report.append(f"| 观测期数 | {best.get('n_periods', 0):,} |")
        report.append("")

        if 'quantile_returns' in best and best['quantile_returns'] is not None:
            report.append("**各分位数组合表现**:")
            report.append("")
            report.append("| 分位数 | 平均收益 | 夏普比率 |")
            report.append("|:---:|:---:|:---:|")
            qr = best['quantile_returns']
            for q in qr.index:
                report.append(f"| Q{q} | {qr.loc[q, 'mean_ret']*100:.4f}% | {qr.loc[q, 'sharpe']:.2f} |")
            report.append("")

    # 3.3 动量效应时变性
    report.append("### 3.3 动量效应的时变性（分年度）")
    report.append("")

    if 'yearly_momentum' in results and results['yearly_momentum'] is not None:
        yearly = results['yearly_momentum']
        report.append("| 年份 | 赢家收益 | 输家收益 | 多空收益 | 动量有效 |")
        report.append("|:---:|:---:|:---:|:---:|:---:|")

        for _, row in yearly.iterrows():
            momentum_flag = "是" if row['momentum_works'] else "否"
            report.append(f"| {row['year']} | {row['winner_ret']*100:.2f}% | {row['loser_ret']*100:.2f}% | {row['long_short_ret']*100:.2f}% | {momentum_flag} |")

        report.append("")

        # 统计动量有效的年份比例
        valid_years = yearly['momentum_works'].sum()
        total_years = len(yearly)
        report.append(f"**动量有效年份**: {valid_years}/{total_years} ({valid_years/total_years*100:.1f}%)")
        report.append("")

    # 3.4 动量崩溃分析
    report.append("### 3.4 动量崩溃分析")
    report.append("")

    if 'momentum_crash' in results:
        crash = results['momentum_crash']
        report.append("动量策略在市场从大跌快速反弹时容易遭受损失（动量崩溃）：")
        report.append("")
        report.append("| 市场状态 | 动量策略收益 | 期数 |")
        report.append("|:---|:---:|:---:|")
        report.append(f"| 正常时期 | {crash.get('normal_mom_ret', 0)*100:.4f}% | {crash.get('n_normal_periods', 0)} |")
        report.append(f"| 崩溃时期（市场急跌后反弹） | {crash.get('crash_mom_ret', 0)*100:.4f}% | {crash.get('n_crash_periods', 0)} |")
        report.append("")

    # 4. 反转效应分析
    report.append("## 4. 反转效应分析")
    report.append("")

    # 4.1 短期反转
    report.append("### 4.1 短期反转效应")
    report.append("")

    if 'short_term_reversal' in results:
        for period, reversal in results['short_term_reversal'].items():
            if reversal:
                report.append(f"**{period}天形成期-{period}天持有期**:")
                report.append("")
                report.append("| 指标 | 数值 |")
                report.append("|:---|:---:|")
                report.append(f"| 反转收益(日均) | {reversal.get('mean_reversal_ret', 0)*100:.4f}% |")
                report.append(f"| 年化夏普比率 | {reversal.get('sharpe', 0):.2f} |")
                report.append(f"| 胜率 | {reversal.get('win_rate', 0)*100:.1f}% |")
                report.append("")

    report.append("**结论**: 短期反转策略（买入近期跌幅最大的股票，卖出涨幅最大的股票）")
    if 'short_term_reversal' in results and '5' in results['short_term_reversal']:
        reversal = results['short_term_reversal']['5']
        if reversal and reversal.get('mean_reversal_ret', 0) > 0:
            report.append("在A股市场显示出一定的有效性，特别是在周度频率上。")
        else:
            report.append("在A股市场效果有限，需要结合其他因素使用。")
    report.append("")

    # 5. 横截面动量 vs 时序动量
    report.append("## 5. 横截面动量 vs 时序动量")
    report.append("")

    if 'time_series_momentum' in results:
        report.append("| 动量类型 | 年化收益 | 夏普比率 | 胜率 |")
        report.append("|:---|:---:|:---:|:---:|")

        if 'best_momentum' in results:
            best = results['best_momentum']
            cs_ret = best.get('long_short_mean', 0) * 252 / best.get('holding_period', 21)
            report.append(f"| 横截面动量 | {cs_ret*100:.2f}% | {best.get('long_short_sharpe', 0):.2f} | {best.get('win_rate', 0)*100:.1f}% |")

        ts = results['time_series_momentum']
        ts_ret = ts.get('mean_ret', 0) * 252 / 21
        report.append(f"| 时序动量 | {ts_ret*100:.2f}% | {ts.get('sharpe', 0):.2f} | {ts.get('win_rate', 0)*100:.1f}% |")
        report.append("")

        report.append("**横截面动量**：基于股票间的相对表现，买入相对强势股，卖出相对弱势股")
        report.append("")
        report.append("**时序动量**：基于股票自身的历史趋势，过去上涨则继续做多，过去下跌则做空")
        report.append("")

    # 6. 动量策略改进
    report.append("## 6. 动量策略改进")
    report.append("")

    report.append("### 6.1 策略对比")
    report.append("")
    report.append("| 策略类型 | 年化收益 | 夏普比率 | 胜率 |")
    report.append("|:---|:---:|:---:|:---:|")

    strategies = [
        ('原始动量', 'best_momentum'),
        ('残差动量', 'residual_momentum'),
        ('52周高点动量', 'high52_momentum'),
        ('波动率调整动量', 'vol_adj_momentum'),
        ('行业中性动量', 'industry_neutral_momentum'),
    ]

    for name, key in strategies:
        if key in results and results[key]:
            strat = results[key]
            if key == 'best_momentum':
                ann_ret = strat.get('long_short_mean', 0) * 252 / strat.get('holding_period', 21)
                sharpe = strat.get('long_short_sharpe', 0)
            else:
                ann_ret = strat.get('mean_ret', 0) * 252 / 21
                sharpe = strat.get('sharpe', 0)
            win_rate = strat.get('win_rate', 0)
            report.append(f"| {name} | {ann_ret*100:.2f}% | {sharpe:.2f} | {win_rate*100:.1f}% |")

    report.append("")

    # 6.2 策略说明
    report.append("### 6.2 策略说明")
    report.append("")
    report.append("- **残差动量**: 剥离行业效应后的个股动量，降低行业轮动的噪音")
    report.append("- **52周高点动量**: 股价距离52周最高点的距离，捕捉锚定效应")
    report.append("- **波动率调整动量**: 用波动率标准化收益，降低高波动股票的权重")
    report.append("- **行业中性动量**: 在每个行业内部做动量排序，避免行业偏离")
    report.append("")

    # 7. 交易成本敏感性分析
    report.append("## 7. 交易成本敏感性分析")
    report.append("")

    if 'best_momentum' in results:
        best = results['best_momentum']
        base_ret = best.get('long_short_mean', 0) * 252 / best.get('holding_period', 21)

        report.append("| 单边成本(bps) | 净年化收益 | 相对原收益 |")
        report.append("|:---:|:---:|:---:|")

        for cost in [10, 20, 30, 50, 100]:
            turnover = 2  # 假设双边换手
            net_ret = base_ret - (turnover * cost / 10000 * 12)  # 假设月度调仓
            report.append(f"| {cost} | {net_ret*100:.2f}% | {net_ret/base_ret*100:.1f}% |")

        report.append("")
        report.append("*假设月度调仓，双边换手*")
        report.append("")

    # 8. 研究结论
    report.append("## 8. 研究结论")
    report.append("")
    report.append("### 8.1 主要发现")
    report.append("")

    conclusions = []

    # 动量效应结论
    if 'best_momentum' in results:
        best = results['best_momentum']
        if best.get('long_short_sharpe', 0) > 0.5:
            conclusions.append("1. **A股动量效应显著**: 中期动量（3-6个月形成期）在A股市场表现良好")
        else:
            conclusions.append("1. **A股动量效应较弱**: 传统动量策略在A股的有效性有限")

    # 反转效应结论
    if 'short_term_reversal' in results and '5' in results['short_term_reversal']:
        reversal = results['short_term_reversal']['5']
        if reversal and reversal.get('sharpe', 0) > 0.5:
            conclusions.append("2. **短期反转效应明显**: 周度反转策略在A股具有较好的预测能力")
        else:
            conclusions.append("2. **短期反转效应有限**: 短期反转策略需要结合其他因素")

    # 时序动量结论
    if 'time_series_momentum' in results:
        ts = results['time_series_momentum']
        if ts.get('sharpe', 0) > 0.5:
            conclusions.append("3. **时序动量有效**: 绝对趋势跟踪策略在A股具有一定有效性")
        else:
            conclusions.append("3. **时序动量效果有限**: 单纯趋势跟踪在A股市场效果不佳")

    # 策略改进结论
    improved_strategies = []
    for key in ['residual_momentum', 'high52_momentum', 'vol_adj_momentum', 'industry_neutral_momentum']:
        if key in results and results[key]:
            if results[key].get('sharpe', 0) > results.get('best_momentum', {}).get('long_short_sharpe', 0):
                improved_strategies.append(key)

    if improved_strategies:
        conclusions.append(f"4. **策略改进有效**: 残差动量、52周高点等改进策略能提升表现")
    else:
        conclusions.append("4. **改进策略效果有限**: 各类改进策略未能显著提升基准动量表现")

    for c in conclusions:
        report.append(c)

    report.append("")

    # 8.2 投资建议
    report.append("### 8.2 投资建议")
    report.append("")
    report.append("1. **动量策略配置**: 建议采用中期动量（3-6个月），结合行业中性化处理")
    report.append("2. **反转策略补充**: 可用短期反转策略对冲动量崩溃风险")
    report.append("3. **风险控制**: 关注市场拐点，在市场急跌后反弹时降低动量敞口")
    report.append("4. **交易成本**: 控制换手率，优化交易执行，降低冲击成本")
    report.append("5. **组合构建**: 建议将动量因子与价值、质量等因子组合使用")
    report.append("")

    # 8.3 风险提示
    report.append("### 8.3 风险提示")
    report.append("")
    report.append("- 动量策略存在尾部风险，在市场反转时可能遭受较大损失")
    report.append("- 历史回测结果不代表未来表现")
    report.append("- 策略容量有限，资金规模较大时可能面临流动性约束")
    report.append("- 交易成本和市场冲击可能显著降低策略收益")
    report.append("")

    report.append("---")
    report.append("")
    report.append("*本报告由动量策略研究系统自动生成*")

    # 写入文件
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    print(f"Report saved to {report_path}")

def main():
    """主函数"""
    print("="*60)
    print("A股动量与反转效应深度研究")
    print("="*60)
    print()

    # 连接数据库
    conn = get_connection()

    # 加载数据
    df = load_daily_data(conn, start_date='20100101', end_date='20251231')
    industry_df = load_industry_data(conn)

    # 存储结果
    results = {}

    # 数据信息
    results['data_info'] = {
        'start_date': df['trade_date'].min(),
        'end_date': df['trade_date'].max(),
        'n_stocks': df['ts_code'].nunique(),
        'n_obs': len(df)
    }

    print(f"\nData range: {results['data_info']['start_date']} - {results['data_info']['end_date']}")
    print(f"Stocks: {results['data_info']['n_stocks']:,}, Observations: {results['data_info']['n_obs']:,}")

    # 计算收益率
    print("\n" + "="*60)
    print("Step 1: Calculating returns and features")
    print("="*60)
    df = calculate_returns(df, periods=[5, 21, 63, 126, 252])
    df = calculate_volatility(df, windows=[21, 63])
    df = calculate_52week_high(df)

    # 1. 动量效应分析 - 不同形成期和持有期
    print("\n" + "="*60)
    print("Step 2: Analyzing momentum portfolios")
    print("="*60)

    momentum_matrix = {}
    best_sharpe = -999
    best_config = None

    formation_periods = [21, 63, 126, 252]
    holding_periods = [5, 21, 63, 126]

    for form in formation_periods:
        for hold in holding_periods:
            result = momentum_portfolio_analysis(df, form, hold)
            if result:
                key = f"{form}_{hold}"
                momentum_matrix[key] = result

                if result['long_short_sharpe'] > best_sharpe:
                    best_sharpe = result['long_short_sharpe']
                    best_config = result

    results['momentum_matrix'] = momentum_matrix
    results['best_momentum'] = best_config

    if best_config:
        print(f"\nBest momentum config: Formation={best_config['formation_period']}d, Holding={best_config['holding_period']}d")
        print(f"Sharpe: {best_config['long_short_sharpe']:.2f}, Win Rate: {best_config['win_rate']:.1%}")

    # 2. 分年度动量分析
    print("\n" + "="*60)
    print("Step 3: Analyzing momentum by year")
    print("="*60)

    if best_config:
        yearly_mom = analyze_momentum_by_year(
            df,
            formation_period=best_config['formation_period'],
            holding_period=best_config['holding_period']
        )
        results['yearly_momentum'] = yearly_mom

        if yearly_mom is not None and len(yearly_mom) > 0:
            print(f"\nYearly momentum summary: {len(yearly_mom)} years analyzed")
            print(f"Momentum works in {yearly_mom['momentum_works'].sum()}/{len(yearly_mom)} years")

    # 3. 动量崩溃分析
    print("\n" + "="*60)
    print("Step 4: Analyzing momentum crash")
    print("="*60)

    if best_config:
        crash_result = analyze_momentum_crash(
            df,
            formation_period=best_config['formation_period'],
            holding_period=best_config['holding_period']
        )
        results['momentum_crash'] = crash_result

        if crash_result:
            print(f"\nNormal periods return: {crash_result['normal_mom_ret']*100:.4f}%")
            print(f"Crash periods return: {crash_result['crash_mom_ret']*100:.4f}%")

    # 4. 短期反转效应
    print("\n" + "="*60)
    print("Step 5: Analyzing short-term reversal")
    print("="*60)

    reversal_results = {}
    for period in [5, 21]:
        result = analyze_short_term_reversal(df, formation_period=period, holding_period=period)
        if result:
            reversal_results[str(period)] = result
            print(f"\n{period}-day reversal: Sharpe={result['sharpe']:.2f}, Win Rate={result['win_rate']:.1%}")

    results['short_term_reversal'] = reversal_results

    # 5. 时序动量分析
    print("\n" + "="*60)
    print("Step 6: Analyzing time-series momentum")
    print("="*60)

    ts_mom = analyze_time_series_momentum(df, lookback=252, holding=21)
    results['time_series_momentum'] = ts_mom

    if ts_mom:
        print(f"\nTime-series momentum: Sharpe={ts_mom['sharpe']:.2f}, Win Rate={ts_mom['win_rate']:.1%}")

    # 6. 残差动量分析
    print("\n" + "="*60)
    print("Step 7: Analyzing residual momentum")
    print("="*60)

    residual_mom = analyze_residual_momentum(df, industry_df, formation_period=126, holding_period=21)
    results['residual_momentum'] = residual_mom

    if residual_mom:
        print(f"\nResidual momentum: Sharpe={residual_mom['sharpe']:.2f}, Win Rate={residual_mom['win_rate']:.1%}")

    # 7. 52周高点动量
    print("\n" + "="*60)
    print("Step 8: Analyzing 52-week high momentum")
    print("="*60)

    high52_mom = analyze_52week_high_momentum(df, holding_period=21)
    results['high52_momentum'] = high52_mom

    if high52_mom:
        print(f"\n52-week high momentum: Sharpe={high52_mom['sharpe']:.2f}, Win Rate={high52_mom['win_rate']:.1%}")

    # 8. 波动率调整动量
    print("\n" + "="*60)
    print("Step 9: Analyzing volatility-adjusted momentum")
    print("="*60)

    vol_adj_mom = analyze_volatility_adjusted_momentum(df, formation_period=126, holding_period=21)
    results['vol_adj_momentum'] = vol_adj_mom

    if vol_adj_mom:
        print(f"\nVol-adjusted momentum: Sharpe={vol_adj_mom['sharpe']:.2f}, Win Rate={vol_adj_mom['win_rate']:.1%}")

    # 9. 行业中性动量
    print("\n" + "="*60)
    print("Step 10: Analyzing industry-neutral momentum")
    print("="*60)

    ind_neutral_mom = analyze_industry_neutral_momentum(df, industry_df, formation_period=126, holding_period=21)
    results['industry_neutral_momentum'] = ind_neutral_mom

    if ind_neutral_mom:
        print(f"\nIndustry-neutral momentum: Sharpe={ind_neutral_mom['sharpe']:.2f}, Win Rate={ind_neutral_mom['win_rate']:.1%}")

    # 10. 生成报告
    print("\n" + "="*60)
    print("Step 11: Generating report")
    print("="*60)

    generate_report(results, REPORT_PATH)

    print("\n" + "="*60)
    print("Analysis completed!")
    print("="*60)

    conn.close()

    return results

if __name__ == "__main__":
    results = main()
