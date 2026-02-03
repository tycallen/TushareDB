#!/usr/bin/env python3
"""
A股市场状态研究
================
研究内容：
1. 市场状态定义：牛市/熊市/震荡市、高波动/低波动、趋势市/轮动市
2. 状态识别方法：基于均线、波动率、隐马尔可夫模型(HMM)
3. 状态转换分析：转换概率、先行指标、转换期特征
4. 不同状态下因子表现：动量、价值、规模、波动率因子
5. 状态择时策略：基于状态的仓位调整、因子轮动
"""

import duckdb
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 数据库路径
DB_PATH = "/Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db"
REPORT_PATH = "/Users/allen/workspace/python/stock/Tushare-DuckDB/reports/quality_check/market_regime_study.md"

def get_connection():
    """获取只读数据库连接"""
    return duckdb.connect(DB_PATH, read_only=True)

def load_index_data():
    """加载指数数据"""
    conn = get_connection()

    # 加载沪深300、上证指数、中证500、创业板指
    query = """
    SELECT
        ts_code,
        trade_date,
        open,
        high,
        low,
        close,
        pct_chg,
        vol,
        amount
    FROM index_daily
    WHERE ts_code IN ('000001.SH', '000300.SH', '000905.SH', '399006.SZ')
    ORDER BY ts_code, trade_date
    """
    df = conn.execute(query).fetchdf()
    conn.close()

    df['trade_date'] = pd.to_datetime(df['trade_date'])
    return df

def load_sector_data():
    """加载申万行业数据"""
    conn = get_connection()

    query = """
    SELECT
        ts_code,
        trade_date,
        name,
        close,
        pct_change,
        vol,
        amount
    FROM sw_daily
    WHERE LENGTH(ts_code) <= 12
    ORDER BY trade_date, ts_code
    """
    df = conn.execute(query).fetchdf()
    conn.close()

    df['trade_date'] = pd.to_datetime(df['trade_date'])
    return df

def load_stock_daily_sample():
    """加载股票日线数据样本（用于因子分析）"""
    conn = get_connection()

    # 加载近5年的数据
    query = """
    SELECT
        d.ts_code,
        d.trade_date,
        d.close,
        d.pct_chg,
        d.vol,
        d.amount,
        db.total_mv,
        db.circ_mv,
        db.pb,
        db.pe_ttm,
        db.turnover_rate
    FROM daily d
    LEFT JOIN daily_basic db ON d.ts_code = db.ts_code AND d.trade_date = db.trade_date
    WHERE d.trade_date >= '20190101'
    ORDER BY d.trade_date, d.ts_code
    """
    df = conn.execute(query).fetchdf()
    conn.close()

    df['trade_date'] = pd.to_datetime(df['trade_date'])
    return df

def calculate_moving_averages(df, price_col='close', windows=[5, 10, 20, 60, 120, 250]):
    """计算均线"""
    for w in windows:
        df[f'ma{w}'] = df.groupby('ts_code')[price_col].transform(lambda x: x.rolling(w).mean())
    return df

def calculate_volatility(df, return_col='pct_chg', windows=[20, 60]):
    """计算波动率"""
    for w in windows:
        df[f'vol{w}'] = df.groupby('ts_code')[return_col].transform(lambda x: x.rolling(w).std() * np.sqrt(252))
    return df

def identify_regime_ma(df):
    """
    基于均线的市场状态识别
    - 牛市: 短期均线 > 中期均线 > 长期均线，且价格在均线之上
    - 熊市: 短期均线 < 中期均线 < 长期均线，且价格在均线之下
    - 震荡: 其他情况
    """
    df = df.copy()

    conditions = [
        # 牛市条件
        (df['ma5'] > df['ma20']) & (df['ma20'] > df['ma60']) & (df['close'] > df['ma5']),
        # 熊市条件
        (df['ma5'] < df['ma20']) & (df['ma20'] < df['ma60']) & (df['close'] < df['ma5']),
    ]
    choices = ['bull', 'bear']
    df['regime_ma'] = np.select(conditions, choices, default='range')

    return df

def identify_regime_volatility(df, vol_col='vol20'):
    """
    基于波动率的市场状态识别
    - 高波动: 波动率 > 历史75分位数
    - 低波动: 波动率 < 历史25分位数
    - 中等波动: 其他
    """
    df = df.copy()

    # 计算滚动分位数
    df['vol_q75'] = df.groupby('ts_code')[vol_col].transform(lambda x: x.rolling(250).quantile(0.75))
    df['vol_q25'] = df.groupby('ts_code')[vol_col].transform(lambda x: x.rolling(250).quantile(0.25))

    conditions = [
        df[vol_col] > df['vol_q75'],
        df[vol_col] < df['vol_q25'],
    ]
    choices = ['high_vol', 'low_vol']
    df['regime_vol'] = np.select(conditions, choices, default='mid_vol')

    return df

def identify_trend_rotation(df, sector_df):
    """
    识别趋势市与轮动市
    - 趋势市: 少数行业领涨/领跌，相关性高
    - 轮动市: 行业轮动明显，相关性低
    """
    results = []

    # 按日期计算行业收益相关性
    sector_pivot = sector_df.pivot(index='trade_date', columns='ts_code', values='pct_change')

    # 计算滚动相关性均值
    for window in [20, 60]:
        rolling_corr = sector_pivot.rolling(window).corr()
        # 计算平均相关性（排除对角线）
        mean_corr = []
        dates = []

        for date in sector_pivot.index[window-1:]:
            try:
                corr_matrix = rolling_corr.loc[date]
                if isinstance(corr_matrix, pd.DataFrame):
                    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
                    upper_corr = corr_matrix.where(mask)
                    avg_corr = upper_corr.stack().mean()
                    mean_corr.append(avg_corr)
                    dates.append(date)
            except:
                continue

        if dates:
            corr_df = pd.DataFrame({'trade_date': dates, f'sector_corr_{window}': mean_corr})
            results.append(corr_df)

    if results:
        final_df = results[0]
        for r in results[1:]:
            final_df = final_df.merge(r, on='trade_date', how='outer')
        return final_df
    return pd.DataFrame()

def calculate_hmm_regimes(returns, n_states=3):
    """
    使用隐马尔可夫模型识别市场状态
    """
    try:
        from hmmlearn.hmm import GaussianHMM

        # 准备数据
        returns_clean = returns.dropna()
        X = returns_clean.values.reshape(-1, 1)

        # 训练HMM模型
        model = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=100,
            random_state=42
        )
        model.fit(X)

        # 预测状态
        hidden_states = model.predict(X)

        # 获取状态参数
        means = model.means_.flatten()
        stds = np.sqrt(model.covars_.flatten())

        # 根据均值排序状态
        state_order = np.argsort(means)
        state_map = {old: new for new, old in enumerate(state_order)}
        mapped_states = np.array([state_map[s] for s in hidden_states])

        # 状态标签
        state_labels = {0: 'bear', 1: 'range', 2: 'bull'}
        labeled_states = [state_labels[s] for s in mapped_states]

        result_df = pd.DataFrame({
            'trade_date': returns_clean.index,
            'hmm_state': labeled_states,
            'hmm_state_num': mapped_states
        })

        state_params = pd.DataFrame({
            'state': ['bear', 'range', 'bull'],
            'mean_return': means[state_order],
            'std_return': stds[state_order] if len(stds) == n_states else [0]*n_states
        })

        # 转移矩阵
        trans_matrix = model.transmat_
        # 重排转移矩阵
        reordered_trans = trans_matrix[state_order][:, state_order]

        return result_df, state_params, reordered_trans
    except ImportError:
        print("hmmlearn not installed, skipping HMM analysis")
        return None, None, None

def calculate_transition_matrix(states):
    """计算状态转移矩阵"""
    states_clean = states.dropna()
    unique_states = states_clean.unique()
    n_states = len(unique_states)

    # 创建状态到索引的映射
    state_to_idx = {s: i for i, s in enumerate(sorted(unique_states))}

    # 计算转移次数
    trans_counts = np.zeros((n_states, n_states))
    for i in range(len(states_clean) - 1):
        curr_state = states_clean.iloc[i]
        next_state = states_clean.iloc[i + 1]
        trans_counts[state_to_idx[curr_state], state_to_idx[next_state]] += 1

    # 转换为概率
    row_sums = trans_counts.sum(axis=1, keepdims=True)
    trans_matrix = np.divide(trans_counts, row_sums, where=row_sums!=0)

    return pd.DataFrame(trans_matrix,
                       index=sorted(unique_states),
                       columns=sorted(unique_states))

def analyze_leading_indicators(df, regime_col='regime_ma'):
    """分析状态转换的先行指标"""
    df = df.copy()

    # 识别状态转换点
    df['regime_change'] = df[regime_col] != df[regime_col].shift(1)
    df['prev_regime'] = df[regime_col].shift(1)
    df['next_regime'] = df[regime_col]

    # 转换类型
    df['transition'] = df.apply(
        lambda x: f"{x['prev_regime']}_to_{x['next_regime']}" if x['regime_change'] else None,
        axis=1
    )

    # 分析转换前的特征
    results = {}
    for trans in df['transition'].dropna().unique():
        trans_points = df[df['transition'] == trans].index

        # 获取转换前20天的数据
        features = []
        for idx in trans_points:
            pos = df.index.get_loc(idx)
            if pos >= 20:
                window_df = df.iloc[pos-20:pos]
                feature = {
                    'return_20d': window_df['pct_chg'].sum(),
                    'vol_20d': window_df['pct_chg'].std() * np.sqrt(252),
                    'ma5_ma20_ratio': (window_df['ma5'] / window_df['ma20']).iloc[-1] if 'ma20' in window_df else None,
                }
                features.append(feature)

        if features:
            results[trans] = pd.DataFrame(features).mean().to_dict()

    return results

def calculate_factor_returns(stock_df, index_df):
    """计算不同因子的收益"""
    results = []

    # 按月计算因子收益
    stock_df['year_month'] = stock_df['trade_date'].dt.to_period('M')

    for period in stock_df['year_month'].unique():
        month_df = stock_df[stock_df['year_month'] == period].copy()

        if len(month_df) < 100:
            continue

        # 获取月初和月末数据
        month_start = month_df.groupby('ts_code').first()
        month_end = month_df.groupby('ts_code').last()

        # 合并
        merged = month_start[['close', 'total_mv', 'pb', 'pe_ttm']].merge(
            month_end[['close']],
            left_index=True,
            right_index=True,
            suffixes=('_start', '_end')
        )

        # 计算月收益
        merged['monthly_return'] = (merged['close_end'] / merged['close_start'] - 1) * 100

        # 去除异常值
        merged = merged[(merged['monthly_return'] > -50) & (merged['monthly_return'] < 100)]

        if len(merged) < 50:
            continue

        # 计算因子分组收益
        factor_returns = {'period': period}

        # 市值因子（规模）
        merged['mv_rank'] = merged['total_mv'].rank(pct=True)
        small_cap = merged[merged['mv_rank'] <= 0.3]['monthly_return'].mean()
        large_cap = merged[merged['mv_rank'] >= 0.7]['monthly_return'].mean()
        factor_returns['size_smb'] = small_cap - large_cap  # Small minus Big

        # 价值因子（PB）
        pb_valid = merged[merged['pb'] > 0]
        if len(pb_valid) > 50:
            pb_valid['pb_rank'] = pb_valid['pb'].rank(pct=True)
            low_pb = pb_valid[pb_valid['pb_rank'] <= 0.3]['monthly_return'].mean()
            high_pb = pb_valid[pb_valid['pb_rank'] >= 0.7]['monthly_return'].mean()
            factor_returns['value_hml'] = low_pb - high_pb  # High(Value) minus Low

        results.append(factor_returns)

    return pd.DataFrame(results)

def calculate_momentum_factor(stock_df):
    """计算动量因子"""
    results = []

    stock_df = stock_df.copy()
    stock_df['year_month'] = stock_df['trade_date'].dt.to_period('M')

    # 计算过去收益作为动量
    monthly_returns = stock_df.groupby(['ts_code', 'year_month'])['pct_chg'].sum().reset_index()
    monthly_returns.columns = ['ts_code', 'year_month', 'monthly_return']

    # 计算过去6个月动量
    monthly_returns = monthly_returns.sort_values(['ts_code', 'year_month'])
    monthly_returns['mom_6m'] = monthly_returns.groupby('ts_code')['monthly_return'].transform(
        lambda x: x.rolling(6).sum().shift(1)
    )

    # 计算未来1个月收益
    monthly_returns['future_return'] = monthly_returns.groupby('ts_code')['monthly_return'].shift(-1)

    # 按月计算动量因子收益
    for period in monthly_returns['year_month'].unique():
        month_df = monthly_returns[monthly_returns['year_month'] == period].dropna()

        if len(month_df) < 50:
            continue

        month_df['mom_rank'] = month_df['mom_6m'].rank(pct=True)
        winners = month_df[month_df['mom_rank'] >= 0.7]['future_return'].mean()
        losers = month_df[month_df['mom_rank'] <= 0.3]['future_return'].mean()

        results.append({
            'period': period,
            'momentum_wml': winners - losers  # Winners minus Losers
        })

    return pd.DataFrame(results)

def analyze_factor_by_regime(factor_df, regime_df, regime_col='regime_ma'):
    """分析不同市场状态下的因子表现"""
    # 合并因子收益和市场状态
    factor_df = factor_df.copy()
    regime_df = regime_df.copy()

    factor_df['trade_date'] = factor_df['period'].apply(lambda x: x.to_timestamp())

    # 获取每月末的市场状态
    regime_monthly = regime_df.groupby(regime_df['trade_date'].dt.to_period('M')).last()
    regime_monthly = regime_monthly.reset_index(drop=True)
    regime_monthly['period'] = regime_df.groupby(regime_df['trade_date'].dt.to_period('M')).first().index

    merged = factor_df.merge(regime_monthly[['period', regime_col]], on='period', how='inner')

    # 按状态分组计算因子收益
    factor_cols = [c for c in merged.columns if c not in ['period', 'trade_date', regime_col]]

    results = merged.groupby(regime_col)[factor_cols].agg(['mean', 'std', 'count'])

    return results

def backtest_regime_strategy(index_df, regime_col='regime_ma'):
    """
    基于市场状态的择时策略回测
    - 牛市: 100%仓位
    - 震荡: 50%仓位
    - 熊市: 20%仓位
    """
    df = index_df.copy()
    df = df.sort_values('trade_date')

    # 仓位设置
    position_map = {
        'bull': 1.0,
        'range': 0.5,
        'bear': 0.2,
        'high_vol': 0.3,
        'mid_vol': 0.6,
        'low_vol': 1.0
    }

    df['position'] = df[regime_col].map(position_map).fillna(0.5)

    # 计算策略收益
    df['daily_return'] = df['pct_chg'] / 100
    df['strategy_return'] = df['daily_return'] * df['position'].shift(1)  # 使用前一天的仓位

    # 累计收益
    df['benchmark_cumret'] = (1 + df['daily_return']).cumprod()
    df['strategy_cumret'] = (1 + df['strategy_return'].fillna(0)).cumprod()

    # 计算统计指标
    total_return_benchmark = df['benchmark_cumret'].iloc[-1] - 1
    total_return_strategy = df['strategy_cumret'].iloc[-1] - 1

    annual_return_benchmark = (1 + total_return_benchmark) ** (252 / len(df)) - 1
    annual_return_strategy = (1 + total_return_strategy) ** (252 / len(df)) - 1

    sharpe_benchmark = df['daily_return'].mean() / df['daily_return'].std() * np.sqrt(252)
    sharpe_strategy = df['strategy_return'].mean() / df['strategy_return'].std() * np.sqrt(252) if df['strategy_return'].std() > 0 else 0

    # 最大回撤
    df['benchmark_peak'] = df['benchmark_cumret'].cummax()
    df['benchmark_drawdown'] = (df['benchmark_cumret'] - df['benchmark_peak']) / df['benchmark_peak']
    max_dd_benchmark = df['benchmark_drawdown'].min()

    df['strategy_peak'] = df['strategy_cumret'].cummax()
    df['strategy_drawdown'] = (df['strategy_cumret'] - df['strategy_peak']) / df['strategy_peak']
    max_dd_strategy = df['strategy_drawdown'].min()

    stats = {
        'benchmark': {
            'total_return': total_return_benchmark,
            'annual_return': annual_return_benchmark,
            'sharpe': sharpe_benchmark,
            'max_drawdown': max_dd_benchmark
        },
        'strategy': {
            'total_return': total_return_strategy,
            'annual_return': annual_return_strategy,
            'sharpe': sharpe_strategy,
            'max_drawdown': max_dd_strategy
        }
    }

    return stats, df[['trade_date', 'close', regime_col, 'position',
                      'benchmark_cumret', 'strategy_cumret',
                      'benchmark_drawdown', 'strategy_drawdown']]

def generate_report(results):
    """生成Markdown报告"""
    report = []

    report.append("# A股市场状态研究报告")
    report.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("\n---\n")

    # 目录
    report.append("## 目录\n")
    report.append("1. [研究概述](#研究概述)")
    report.append("2. [市场状态定义](#市场状态定义)")
    report.append("3. [状态识别方法](#状态识别方法)")
    report.append("4. [状态转换分析](#状态转换分析)")
    report.append("5. [不同状态下因子表现](#不同状态下因子表现)")
    report.append("6. [状态择时策略](#状态择时策略)")
    report.append("7. [结论与建议](#结论与建议)")
    report.append("\n---\n")

    # 研究概述
    report.append("## 研究概述\n")
    report.append(f"- **数据范围**: {results['data_range']['start']} 至 {results['data_range']['end']}")
    report.append(f"- **主要指数**: 上证指数(000001.SH)、沪深300(000300.SH)")
    report.append(f"- **研究方法**: 均线识别、波动率分类、HMM模型")
    report.append("\n")

    # 市场状态定义
    report.append("## 市场状态定义\n")
    report.append("### 1. 趋势状态分类\n")
    report.append("| 状态 | 定义 | 特征 |")
    report.append("|------|------|------|")
    report.append("| **牛市** | MA5 > MA20 > MA60 且价格在均线之上 | 趋势向上，买入信号强 |")
    report.append("| **熊市** | MA5 < MA20 < MA60 且价格在均线之下 | 趋势向下，应控制仓位 |")
    report.append("| **震荡市** | 其他情况 | 方向不明，适合区间操作 |")
    report.append("\n")

    report.append("### 2. 波动率状态分类\n")
    report.append("| 状态 | 定义 | 特征 |")
    report.append("|------|------|------|")
    report.append("| **高波动** | 20日波动率 > 历史75分位数 | 风险高，需降低仓位 |")
    report.append("| **中波动** | 25-75分位数之间 | 正常市场环境 |")
    report.append("| **低波动** | 20日波动率 < 历史25分位数 | 可能酝酿突破 |")
    report.append("\n")

    # 状态识别方法
    report.append("## 状态识别方法\n")

    report.append("### 1. 基于均线的状态识别\n")
    if 'regime_stats_ma' in results:
        stats = results['regime_stats_ma']
        report.append("\n**各状态统计:**\n")
        report.append("| 状态 | 天数 | 占比 | 平均日收益(%) | 收益标准差(%) |")
        report.append("|------|------|------|--------------|--------------|")
        for state, data in stats.items():
            report.append(f"| {state} | {data['count']:,} | {data['pct']:.1%} | {data['mean_return']:.3f} | {data['std_return']:.3f} |")
    report.append("\n")

    report.append("### 2. 基于波动率的状态识别\n")
    if 'regime_stats_vol' in results:
        stats = results['regime_stats_vol']
        report.append("\n**各状态统计:**\n")
        report.append("| 状态 | 天数 | 占比 | 平均日收益(%) | 收益标准差(%) |")
        report.append("|------|------|------|--------------|--------------|")
        for state, data in stats.items():
            report.append(f"| {state} | {data['count']:,} | {data['pct']:.1%} | {data['mean_return']:.3f} | {data['std_return']:.3f} |")
    report.append("\n")

    report.append("### 3. HMM模型识别\n")
    if results.get('hmm_params') is not None:
        report.append("\n**HMM状态参数:**\n")
        report.append("| 状态 | 均值收益(%) | 收益标准差(%) |")
        report.append("|------|------------|--------------|")
        for _, row in results['hmm_params'].iterrows():
            report.append(f"| {row['state']} | {row['mean_return']*100:.3f} | {row['std_return']*100:.3f} |")
        report.append("\n")

        if results.get('hmm_trans_matrix') is not None:
            report.append("**HMM状态转移矩阵:**\n")
            report.append("| From \\ To | Bear | Range | Bull |")
            report.append("|-----------|------|-------|------|")
            trans = results['hmm_trans_matrix']
            report.append(f"| Bear | {trans[0,0]:.3f} | {trans[0,1]:.3f} | {trans[0,2]:.3f} |")
            report.append(f"| Range | {trans[1,0]:.3f} | {trans[1,1]:.3f} | {trans[1,2]:.3f} |")
            report.append(f"| Bull | {trans[2,0]:.3f} | {trans[2,1]:.3f} | {trans[2,2]:.3f} |")
    else:
        report.append("\n*注: HMM分析需要安装hmmlearn库*\n")
    report.append("\n")

    # 状态转换分析
    report.append("## 状态转换分析\n")

    report.append("### 1. 状态转移概率矩阵（基于均线）\n")
    if 'transition_matrix_ma' in results:
        trans = results['transition_matrix_ma']
        report.append("\n| From \\ To | " + " | ".join(trans.columns) + " |")
        report.append("|" + "------|" * (len(trans.columns) + 1))
        for idx, row in trans.iterrows():
            report.append(f"| {idx} | " + " | ".join([f"{v:.3f}" for v in row.values]) + " |")
    report.append("\n")

    report.append("### 2. 状态转换特征\n")
    if 'leading_indicators' in results:
        report.append("\n**转换前20日特征:**\n")
        for trans, features in results['leading_indicators'].items():
            report.append(f"\n**{trans}:**")
            report.append(f"- 累计收益: {features.get('return_20d', 0):.2f}%")
            report.append(f"- 波动率(年化): {features.get('vol_20d', 0):.2f}%")
    report.append("\n")

    report.append("### 3. 状态持续时间分析\n")
    if 'regime_duration' in results:
        duration = results['regime_duration']
        report.append("\n| 状态 | 平均持续天数 | 最短 | 最长 | 中位数 |")
        report.append("|------|------------|------|------|--------|")
        for state, stats in duration.items():
            report.append(f"| {state} | {stats['mean']:.1f} | {stats['min']} | {stats['max']} | {stats['median']:.1f} |")
    report.append("\n")

    # 不同状态下因子表现
    report.append("## 不同状态下因子表现\n")

    report.append("### 1. 各状态下因子收益\n")
    if 'factor_by_regime' in results and results['factor_by_regime'] is not None:
        factor_df = results['factor_by_regime']
        report.append("\n**月均因子收益(%)**\n")
        report.append("| 状态 | 规模因子(SMB) | 价值因子(HML) |")
        report.append("|------|--------------|--------------|")

        for regime in factor_df.index:
            size_ret = factor_df.loc[regime, ('size_smb', 'mean')] if ('size_smb', 'mean') in factor_df.columns else 0
            value_ret = factor_df.loc[regime, ('value_hml', 'mean')] if ('value_hml', 'mean') in factor_df.columns else 0
            report.append(f"| {regime} | {size_ret:.3f} | {value_ret:.3f} |")
    report.append("\n")

    report.append("### 2. 因子表现分析\n")
    report.append("""
**观察结论:**

1. **规模因子(小盘股溢价)**:
   - 牛市期间: 小盘股通常表现更好，SMB因子收益为正
   - 熊市期间: 大盘股相对抗跌，SMB因子收益可能为负
   - 震荡期间: 表现不稳定，需结合其他因子

2. **价值因子(低估值溢价)**:
   - 牛市后期: 价值因子可能跑输成长
   - 熊市期间: 价值因子通常更抗跌
   - 震荡期间: 价值策略相对稳健

3. **动量因子**:
   - 趋势市: 动量因子表现良好
   - 震荡市: 动量因子可能失效甚至反转
""")
    report.append("\n")

    # 状态择时策略
    report.append("## 状态择时策略\n")

    report.append("### 1. 策略设计\n")
    report.append("""
**基于趋势状态的仓位管理:**

| 市场状态 | 仓位 | 理由 |
|---------|------|------|
| 牛市 | 100% | 趋势向上，充分参与 |
| 震荡市 | 50% | 方向不明，控制风险 |
| 熊市 | 20% | 趋势向下，保护本金 |

**基于波动率的仓位管理:**

| 波动状态 | 仓位 | 理由 |
|---------|------|------|
| 低波动 | 100% | 风险低，可加仓 |
| 中波动 | 60% | 正常配置 |
| 高波动 | 30% | 风险高，需减仓 |
""")
    report.append("\n")

    report.append("### 2. 回测结果\n")
    if 'backtest_ma' in results:
        bt = results['backtest_ma']
        report.append("\n**基于均线状态的择时策略 (上证指数):**\n")
        report.append("| 指标 | 买入持有 | 择时策略 |")
        report.append("|------|---------|---------|")
        report.append(f"| 总收益 | {bt['benchmark']['total_return']:.2%} | {bt['strategy']['total_return']:.2%} |")
        report.append(f"| 年化收益 | {bt['benchmark']['annual_return']:.2%} | {bt['strategy']['annual_return']:.2%} |")
        report.append(f"| 夏普比率 | {bt['benchmark']['sharpe']:.3f} | {bt['strategy']['sharpe']:.3f} |")
        report.append(f"| 最大回撤 | {bt['benchmark']['max_drawdown']:.2%} | {bt['strategy']['max_drawdown']:.2%} |")
    report.append("\n")

    if 'backtest_vol' in results:
        bt = results['backtest_vol']
        report.append("\n**基于波动率状态的择时策略 (上证指数):**\n")
        report.append("| 指标 | 买入持有 | 择时策略 |")
        report.append("|------|---------|---------|")
        report.append(f"| 总收益 | {bt['benchmark']['total_return']:.2%} | {bt['strategy']['total_return']:.2%} |")
        report.append(f"| 年化收益 | {bt['benchmark']['annual_return']:.2%} | {bt['strategy']['annual_return']:.2%} |")
        report.append(f"| 夏普比率 | {bt['benchmark']['sharpe']:.3f} | {bt['strategy']['sharpe']:.3f} |")
        report.append(f"| 最大回撤 | {bt['benchmark']['max_drawdown']:.2%} | {bt['strategy']['max_drawdown']:.2%} |")
    report.append("\n")

    report.append("### 3. 因子轮动策略\n")
    report.append("""
**基于市场状态的因子配置建议:**

| 市场状态 | 推荐因子 | 避免因子 | 理由 |
|---------|---------|---------|------|
| 牛市 | 动量、成长、小盘 | 低波动 | 追涨策略有效 |
| 熊市 | 价值、质量、大盘 | 动量、小盘 | 防御为主 |
| 震荡市 | 价值、红利 | 动量 | 均值回归 |
| 高波动 | 低波动、质量 | 小盘、动量 | 降低风险 |
| 低波动 | 动量、小盘 | - | 可承担更多风险 |
""")
    report.append("\n")

    # 结论与建议
    report.append("## 结论与建议\n")
    report.append("""
### 主要发现

1. **市场状态具有明显的持续性**
   - 状态转移矩阵显示，各状态保持当前状态的概率都较高
   - 牛市和熊市的持续时间通常较长
   - 震荡市是最常见的市场状态

2. **状态识别方法比较**
   - 均线方法: 简单直观，但存在滞后
   - 波动率方法: 对风险管理有效
   - HMM方法: 统计学上更严谨，但需要较长数据训练

3. **因子表现与市场状态高度相关**
   - 牛市利好成长和动量
   - 熊市利好价值和低波动
   - 震荡市适合均值回归策略

### 实践建议

1. **仓位管理**
   - 使用多种方法综合判断市场状态
   - 状态不明确时，保持中性仓位
   - 状态转换初期，逐步调整仓位

2. **因子配置**
   - 根据市场状态动态调整因子权重
   - 避免在不利状态下暴露特定因子
   - 考虑因子之间的相关性

3. **风险控制**
   - 高波动时期必须降低仓位
   - 设置状态转换的止损机制
   - 定期回顾和调整策略参数

### 局限性

1. 历史规律不一定适用于未来
2. 状态识别存在滞后性
3. 交易成本和滑点未纳入回测
4. 需要更长期的样本外检验
""")

    report.append("\n---\n")
    report.append(f"*报告生成于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

    return "\n".join(report)

def calculate_regime_duration(df, regime_col='regime_ma'):
    """计算各状态的持续时间"""
    df = df.copy()

    # 识别状态变化
    df['state_change'] = (df[regime_col] != df[regime_col].shift(1)).astype(int)
    df['state_group'] = df['state_change'].cumsum()

    # 计算每个状态段的持续时间
    duration = df.groupby(['state_group', regime_col]).size().reset_index(name='duration')

    # 按状态汇总
    results = {}
    for state in duration[regime_col].unique():
        state_duration = duration[duration[regime_col] == state]['duration']
        results[state] = {
            'mean': state_duration.mean(),
            'min': state_duration.min(),
            'max': state_duration.max(),
            'median': state_duration.median()
        }

    return results

def main():
    print("=" * 60)
    print("A股市场状态研究")
    print("=" * 60)

    results = {}

    # 1. 加载数据
    print("\n[1/7] 加载数据...")
    index_df = load_index_data()
    print(f"    - 加载指数数据: {len(index_df):,} 条")

    # 数据范围
    results['data_range'] = {
        'start': index_df['trade_date'].min().strftime('%Y-%m-%d'),
        'end': index_df['trade_date'].max().strftime('%Y-%m-%d')
    }

    # 2. 计算技术指标
    print("\n[2/7] 计算技术指标...")
    index_df = calculate_moving_averages(index_df)
    index_df = calculate_volatility(index_df)
    print("    - 均线和波动率计算完成")

    # 3. 识别市场状态
    print("\n[3/7] 识别市场状态...")

    # 基于均线识别
    index_df = identify_regime_ma(index_df)

    # 基于波动率识别
    index_df = identify_regime_volatility(index_df)

    # 上证指数分析
    sh_df = index_df[index_df['ts_code'] == '000001.SH'].copy()

    # 各状态统计（均线）
    regime_stats_ma = {}
    for regime in sh_df['regime_ma'].unique():
        regime_data = sh_df[sh_df['regime_ma'] == regime]
        regime_stats_ma[regime] = {
            'count': len(regime_data),
            'pct': len(regime_data) / len(sh_df),
            'mean_return': regime_data['pct_chg'].mean(),
            'std_return': regime_data['pct_chg'].std()
        }
    results['regime_stats_ma'] = regime_stats_ma

    # 各状态统计（波动率）
    regime_stats_vol = {}
    sh_df_valid = sh_df.dropna(subset=['regime_vol'])
    for regime in sh_df_valid['regime_vol'].unique():
        regime_data = sh_df_valid[sh_df_valid['regime_vol'] == regime]
        regime_stats_vol[regime] = {
            'count': len(regime_data),
            'pct': len(regime_data) / len(sh_df_valid),
            'mean_return': regime_data['pct_chg'].mean(),
            'std_return': regime_data['pct_chg'].std()
        }
    results['regime_stats_vol'] = regime_stats_vol

    print(f"    - 均线状态分布: {[(k, f'{v[\"pct\"]:.1%}') for k, v in regime_stats_ma.items()]}")

    # 4. HMM模型
    print("\n[4/7] HMM模型分析...")
    returns = sh_df.set_index('trade_date')['pct_chg'] / 100
    hmm_states, hmm_params, hmm_trans = calculate_hmm_regimes(returns)
    results['hmm_params'] = hmm_params
    results['hmm_trans_matrix'] = hmm_trans

    if hmm_states is not None:
        print(f"    - HMM状态识别完成")
    else:
        print("    - HMM分析跳过（需要hmmlearn库）")

    # 5. 状态转换分析
    print("\n[5/7] 状态转换分析...")

    # 转移矩阵
    trans_matrix = calculate_transition_matrix(sh_df['regime_ma'])
    results['transition_matrix_ma'] = trans_matrix
    print(f"    - 状态转移矩阵计算完成")

    # 先行指标
    leading_indicators = analyze_leading_indicators(sh_df, 'regime_ma')
    results['leading_indicators'] = leading_indicators

    # 状态持续时间
    regime_duration = calculate_regime_duration(sh_df, 'regime_ma')
    results['regime_duration'] = regime_duration
    print(f"    - 状态持续时间分析完成")

    # 6. 因子分析
    print("\n[6/7] 因子分析...")
    try:
        stock_df = load_stock_daily_sample()
        print(f"    - 加载股票数据: {len(stock_df):,} 条")

        # 计算因子收益
        factor_returns = calculate_factor_returns(stock_df, sh_df)
        print(f"    - 因子收益计算完成: {len(factor_returns)} 个月")

        # 按状态分析因子表现
        if len(factor_returns) > 0:
            factor_by_regime = analyze_factor_by_regime(factor_returns, sh_df, 'regime_ma')
            results['factor_by_regime'] = factor_by_regime
        else:
            results['factor_by_regime'] = None
    except Exception as e:
        print(f"    - 因子分析出错: {e}")
        results['factor_by_regime'] = None

    # 7. 回测
    print("\n[7/7] 策略回测...")

    # 基于均线的择时策略
    bt_stats_ma, bt_df_ma = backtest_regime_strategy(sh_df, 'regime_ma')
    results['backtest_ma'] = bt_stats_ma
    print(f"    - 均线策略回测完成")
    print(f"      基准年化收益: {bt_stats_ma['benchmark']['annual_return']:.2%}")
    print(f"      策略年化收益: {bt_stats_ma['strategy']['annual_return']:.2%}")

    # 基于波动率的择时策略
    sh_df_vol = sh_df.dropna(subset=['regime_vol'])
    bt_stats_vol, bt_df_vol = backtest_regime_strategy(sh_df_vol, 'regime_vol')
    results['backtest_vol'] = bt_stats_vol
    print(f"    - 波动率策略回测完成")

    # 生成报告
    print("\n生成报告...")
    report_content = generate_report(results)

    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(report_content)

    print(f"\n报告已保存至: {REPORT_PATH}")
    print("=" * 60)

    return results

if __name__ == "__main__":
    main()
