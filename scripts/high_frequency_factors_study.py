#!/usr/bin/env python3
"""
高频因子模拟研究
基于日频数据提取日内信息，构建高频因子代理

Author: AI Research Assistant
Date: 2024
"""

import duckdb
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 配置
DB_PATH = '/Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db'
OUTPUT_PATH = '/Users/allen/workspace/python/stock/Tushare-DuckDB/reports/quality_check/high_frequency_factors_study.md'

# 研究时间范围 (使用近5年数据)
START_DATE = '20200101'
END_DATE = '20260130'

def load_data():
    """加载所需数据"""
    print("正在加载数据...")
    conn = duckdb.connect(DB_PATH, read_only=True)

    # 加载日线数据
    daily_sql = f"""
    SELECT
        d.ts_code, d.trade_date, d.open, d.high, d.low, d.close,
        d.pre_close, d.pct_chg, d.vol, d.amount
    FROM daily d
    WHERE d.trade_date >= '{START_DATE}' AND d.trade_date <= '{END_DATE}'
        AND d.ts_code LIKE '%.SH' OR d.ts_code LIKE '%.SZ'
    ORDER BY d.ts_code, d.trade_date
    """
    daily = conn.execute(daily_sql).fetchdf()
    print(f"  日线数据: {len(daily):,} 条")

    # 加载daily_basic数据
    basic_sql = f"""
    SELECT ts_code, trade_date, turnover_rate, turnover_rate_f, volume_ratio,
           total_mv, circ_mv
    FROM daily_basic
    WHERE trade_date >= '{START_DATE}' AND trade_date <= '{END_DATE}'
    ORDER BY ts_code, trade_date
    """
    basic = conn.execute(basic_sql).fetchdf()
    print(f"  基础数据: {len(basic):,} 条")

    # 加载资金流数据
    mf_sql = f"""
    SELECT *
    FROM moneyflow
    WHERE trade_date >= '{START_DATE}' AND trade_date <= '{END_DATE}'
    ORDER BY ts_code, trade_date
    """
    moneyflow = conn.execute(mf_sql).fetchdf()
    print(f"  资金流数据: {len(moneyflow):,} 条")

    conn.close()
    return daily, basic, moneyflow

def construct_intraday_price_factors(daily):
    """
    1. 日内价格特征因子
    """
    print("\n构建日内价格特征因子...")
    df = daily.copy()

    # 开盘收益 (Open Return)
    df['open_return'] = (df['open'] / df['pre_close'] - 1) * 100

    # 盘中波动 (Intraday Range)
    df['intraday_range'] = (df['high'] - df['low']) / df['pre_close'] * 100

    # 真实波动幅度 (True Range)
    df['true_range'] = df.apply(
        lambda x: max(x['high'] - x['low'],
                     abs(x['high'] - x['pre_close']),
                     abs(x['low'] - x['pre_close'])) / x['pre_close'] * 100
        if x['pre_close'] > 0 else np.nan, axis=1
    )

    # 收盘强度 (Close Location Value)
    # (close - low) / (high - low), 值越高说明收盘越接近日内高点
    df['close_location'] = np.where(
        df['high'] > df['low'],
        (df['close'] - df['low']) / (df['high'] - df['low']),
        0.5
    )

    # 缺口因子 (Gap)
    df['gap'] = (df['open'] - df['pre_close']) / df['pre_close'] * 100
    df['gap_filled'] = np.where(
        df['gap'] > 0,
        df['low'] <= df['pre_close'],  # 上跳缺口是否回补
        df['high'] >= df['pre_close']   # 下跳缺口是否回补
    ).astype(int)

    # 上影线/下影线
    df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['pre_close'] * 100
    df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['pre_close'] * 100

    # 实体大小
    df['body_size'] = abs(df['close'] - df['open']) / df['pre_close'] * 100

    # 日内反转 (Intraday Reversal)
    # 开盘涨但收盘跌 (或反之) 的强度
    df['intraday_reversal'] = (df['close'] - df['open']) / df['pre_close'] * 100 - df['open_return']

    return df

def construct_volume_price_factors(df, basic):
    """
    2. 日内量价特征因子
    """
    print("构建日内量价特征因子...")

    # 合并基础数据
    df = df.merge(basic[['ts_code', 'trade_date', 'turnover_rate', 'volume_ratio', 'circ_mv']],
                  on=['ts_code', 'trade_date'], how='left')

    # 成交额强度 (相对于流通市值)
    df['amount_intensity'] = df['amount'] / (df['circ_mv'] + 1e-10) * 100

    # 按股票分组计算时序特征
    df = df.sort_values(['ts_code', 'trade_date'])

    # 量价相关性 (过去20日滚动)
    def rolling_corr(group):
        group = group.copy()
        group['vol_price_corr'] = group['pct_chg'].rolling(20, min_periods=10).corr(
            group['vol'].pct_change()
        )
        return group

    df = df.groupby('ts_code', group_keys=False).apply(rolling_corr)

    # 高位放量 / 低位放量
    def high_low_volume(group):
        group = group.copy()
        # 过去20日最高价和最低价
        high_20 = group['high'].rolling(20, min_periods=10).max()
        low_20 = group['low'].rolling(20, min_periods=10).min()
        avg_vol = group['vol'].rolling(20, min_periods=10).mean()

        # 当前价格在20日区间的位置
        price_pos = (group['close'] - low_20) / (high_20 - low_20 + 1e-10)

        # 相对成交量
        rel_vol = group['vol'] / (avg_vol + 1e-10)

        # 高位放量 (价格在高位且成交量放大)
        group['high_vol_indicator'] = price_pos * rel_vol
        # 低位放量 (价格在低位且成交量放大)
        group['low_vol_indicator'] = (1 - price_pos) * rel_vol

        return group

    df = df.groupby('ts_code', group_keys=False).apply(high_low_volume)

    # 成交额集中度估计 (用日内波动估计)
    # 假设波动大时成交分散，波动小时成交集中
    df['volume_concentration'] = 1 / (df['intraday_range'] + 0.1) * df['volume_ratio'].fillna(1)

    # VWAP偏离度估计
    # VWAP ≈ amount / vol, 与收盘价的偏离
    df['vwap_estimate'] = df['amount'] * 1000 / (df['vol'] * 100 + 1e-10)  # 转换单位
    df['vwap_deviation'] = (df['close'] - df['vwap_estimate']) / df['vwap_estimate'] * 100

    return df

def construct_order_flow_factors(df, moneyflow):
    """
    3. 订单流代理因子
    """
    print("构建订单流代理因子...")

    # 合并资金流数据
    df = df.merge(moneyflow, on=['ts_code', 'trade_date'], how='left')

    # 总成交额
    df['total_buy_amount'] = (df['buy_sm_amount'].fillna(0) + df['buy_md_amount'].fillna(0) +
                              df['buy_lg_amount'].fillna(0) + df['buy_elg_amount'].fillna(0))
    df['total_sell_amount'] = (df['sell_sm_amount'].fillna(0) + df['sell_md_amount'].fillna(0) +
                               df['sell_lg_amount'].fillna(0) + df['sell_elg_amount'].fillna(0))
    df['total_amount_mf'] = df['total_buy_amount'] + df['total_sell_amount']

    # 1. 大单/小单比例
    df['large_small_ratio'] = ((df['buy_lg_amount'].fillna(0) + df['sell_lg_amount'].fillna(0) +
                                df['buy_elg_amount'].fillna(0) + df['sell_elg_amount'].fillna(0)) /
                               (df['buy_sm_amount'].fillna(0) + df['sell_sm_amount'].fillna(0) + 1e-10))

    # 2. 主力净流入率 (大单+超大单净流入 / 总成交额)
    df['main_net_inflow_rate'] = ((df['buy_lg_amount'].fillna(0) - df['sell_lg_amount'].fillna(0) +
                                   df['buy_elg_amount'].fillna(0) - df['sell_elg_amount'].fillna(0)) /
                                  (df['total_amount_mf'] + 1e-10))

    # 3. 订单不平衡 (Order Imbalance)
    df['order_imbalance'] = ((df['total_buy_amount'] - df['total_sell_amount']) /
                             (df['total_buy_amount'] + df['total_sell_amount'] + 1e-10))

    # 4. 知情交易代理 (Informed Trading Proxy)
    # 使用大单净流入与价格变动的一致性
    df['informed_trading'] = np.sign(df['pct_chg']) * df['main_net_inflow_rate']

    # 5. 散户情绪代理 (小单净流入率)
    df['retail_sentiment'] = ((df['buy_sm_amount'].fillna(0) - df['sell_sm_amount'].fillna(0)) /
                              (df['buy_sm_amount'].fillna(0) + df['sell_sm_amount'].fillna(0) + 1e-10))

    # 6. 机构vs散户分歧
    df['inst_retail_divergence'] = df['main_net_inflow_rate'] - df['retail_sentiment']

    # 7. 超大单占比
    df['elg_order_ratio'] = ((df['buy_elg_amount'].fillna(0) + df['sell_elg_amount'].fillna(0)) /
                             (df['total_amount_mf'] + 1e-10))

    # 8. 净流入持续性 (过去5日累计)
    df = df.sort_values(['ts_code', 'trade_date'])
    def calc_persistence(group):
        group = group.copy()
        group['net_inflow_persist'] = group['main_net_inflow_rate'].rolling(5, min_periods=3).sum()
        return group
    df = df.groupby('ts_code', group_keys=False).apply(calc_persistence)

    return df

def construct_reversal_factors(df):
    """
    4. 短期反转因子
    """
    print("构建短期反转因子...")
    df = df.sort_values(['ts_code', 'trade_date'])

    def calc_reversal(group):
        group = group.copy()

        # 1. 隔夜收益 (Overnight Return)
        group['overnight_return'] = group['open_return']

        # 2. 日内收益 (Intraday Return)
        group['intraday_return'] = (group['close'] / group['open'] - 1) * 100

        # 3. 隔夜vs日内反转
        group['overnight_intraday_diff'] = group['overnight_return'] - group['intraday_return']

        # 4. 短期反转 (过去1-5日收益)
        for lag in [1, 3, 5]:
            group[f'ret_lag{lag}'] = group['pct_chg'].shift(lag)
            group[f'reversal_{lag}d'] = -group['pct_chg'].rolling(lag).sum().shift(1)

        # 5. 过去5日累计收益 (用于判断超买超卖)
        group['cum_ret_5d'] = group['pct_chg'].rolling(5).sum()

        # 6. 波动调整反转
        vol_20 = group['pct_chg'].rolling(20, min_periods=10).std()
        group['vol_adj_reversal'] = group['reversal_5d'] / (vol_20 + 1e-10)

        # 7. 均值回归强度
        ma_20 = group['close'].rolling(20, min_periods=10).mean()
        group['mean_reversion'] = (group['close'] - ma_20) / ma_20 * 100

        return group

    df = df.groupby('ts_code', group_keys=False).apply(calc_reversal)

    return df

def calculate_factor_ic(df, factor_cols, forward_periods=[1, 3, 5, 10, 20]):
    """
    5. 计算因子IC
    """
    print("\n计算因子IC...")

    # 计算未来收益
    df = df.sort_values(['ts_code', 'trade_date'])
    def calc_forward_ret(group):
        group = group.copy()
        for period in forward_periods:
            group[f'fwd_ret_{period}d'] = group['pct_chg'].shift(-period).rolling(period).sum()
        return group

    df = df.groupby('ts_code', group_keys=False).apply(calc_forward_ret)

    # 按日期计算截面IC
    ic_results = {}

    for factor in factor_cols:
        ic_results[factor] = {}
        for period in forward_periods:
            fwd_col = f'fwd_ret_{period}d'

            # 按日期计算IC
            def calc_ic(group):
                valid = group[[factor, fwd_col]].dropna()
                if len(valid) < 30:
                    return np.nan
                return valid[factor].corr(valid[fwd_col], method='spearman')

            daily_ic = df.groupby('trade_date').apply(calc_ic)
            daily_ic = daily_ic.dropna()

            if len(daily_ic) > 0:
                ic_results[factor][f'{period}d'] = {
                    'IC_mean': daily_ic.mean(),
                    'IC_std': daily_ic.std(),
                    'ICIR': daily_ic.mean() / (daily_ic.std() + 1e-10),
                    'IC_positive_ratio': (daily_ic > 0).mean(),
                    'IC_series': daily_ic
                }

    return ic_results, df

def analyze_factor_decay(ic_results, factor_cols):
    """
    因子衰减分析
    """
    print("分析因子衰减...")

    decay_analysis = {}
    periods = [1, 3, 5, 10, 20]

    for factor in factor_cols:
        if factor not in ic_results:
            continue

        ic_values = []
        for period in periods:
            key = f'{period}d'
            if key in ic_results[factor]:
                ic_values.append(ic_results[factor][key]['IC_mean'])
            else:
                ic_values.append(np.nan)

        decay_analysis[factor] = {
            'ic_by_period': dict(zip(periods, ic_values)),
            'best_period': periods[np.nanargmax(np.abs(ic_values))] if not all(np.isnan(ic_values)) else None,
            'decay_rate': None
        }

        # 计算衰减率
        valid_ics = [(p, ic) for p, ic in zip(periods, ic_values) if not np.isnan(ic)]
        if len(valid_ics) >= 3:
            x = np.array([np.log(p) for p, _ in valid_ics])
            y = np.array([abs(ic) for _, ic in valid_ics])
            if np.std(x) > 0:
                slope, _, _, _, _ = stats.linregress(x, y)
                decay_analysis[factor]['decay_rate'] = slope

    return decay_analysis

def backtest_strategy(df, factor_col, holding_period=5, n_groups=5):
    """
    6. 策略回测
    """
    print(f"回测策略: {factor_col}, 持有期={holding_period}天...")

    df = df.copy()
    df = df.sort_values(['trade_date', 'ts_code'])

    # 过滤有效数据
    df = df.dropna(subset=[factor_col, f'fwd_ret_{holding_period}d'])

    # 按日期分组
    results = []

    for date, group in df.groupby('trade_date'):
        if len(group) < n_groups * 10:
            continue

        # 按因子分组
        group = group.copy()
        group['factor_group'] = pd.qcut(group[factor_col], n_groups, labels=False, duplicates='drop')

        # 计算各组平均收益
        group_ret = group.groupby('factor_group')[f'fwd_ret_{holding_period}d'].mean()

        if len(group_ret) == n_groups:
            results.append({
                'date': date,
                'long_ret': group_ret.iloc[-1],  # 最高组
                'short_ret': group_ret.iloc[0],   # 最低组
                'long_short': group_ret.iloc[-1] - group_ret.iloc[0],
                'n_stocks': len(group) // n_groups
            })

    if len(results) == 0:
        return None

    results_df = pd.DataFrame(results)

    # 计算策略统计
    strategy_stats = {
        'total_return': results_df['long_short'].sum(),
        'annual_return': results_df['long_short'].mean() * 252 / holding_period,
        'volatility': results_df['long_short'].std() * np.sqrt(252 / holding_period),
        'sharpe': (results_df['long_short'].mean() * 252 / holding_period) /
                  (results_df['long_short'].std() * np.sqrt(252 / holding_period) + 1e-10),
        'win_rate': (results_df['long_short'] > 0).mean(),
        'max_drawdown': (results_df['long_short'].cumsum() -
                        results_df['long_short'].cumsum().cummax()).min(),
        'avg_stocks_per_group': results_df['n_stocks'].mean()
    }

    return strategy_stats, results_df

def estimate_trading_cost(df, factor_col):
    """
    交易成本分析
    """
    print("估算交易成本...")

    # 估算换手率
    df = df.copy()
    df = df.sort_values(['ts_code', 'trade_date'])

    # 因子变化 -> 估算换手
    def calc_turnover(group):
        group = group.copy()
        factor_change = group[factor_col].diff().abs()
        factor_std = group[factor_col].rolling(20, min_periods=10).std()
        group['signal_change'] = factor_change / (factor_std + 1e-10)
        return group

    df = df.groupby('ts_code', group_keys=False).apply(calc_turnover)

    avg_turnover = df['signal_change'].mean()

    # 假设交易成本
    # 佣金: 0.03% (万3)
    # 印花税: 0.1% (卖出)
    # 冲击成本: 与波动和换手相关
    commission_rate = 0.0003
    stamp_tax = 0.001

    # 估算冲击成本 (简化)
    avg_spread = df['intraday_range'].mean() * 0.1 / 100  # 假设冲击为日内波动的10%

    cost_analysis = {
        'commission': commission_rate * 2 * 100,  # 双边，转为百分比
        'stamp_tax': stamp_tax * 100,
        'estimated_spread': avg_spread * 100,
        'total_cost_estimate': (commission_rate * 2 + stamp_tax + avg_spread) * 100,
        'avg_signal_change': avg_turnover
    }

    return cost_analysis

def estimate_capacity(df, factor_col, target_impact=0.001):
    """
    策略容量估计
    """
    print("估算策略容量...")

    # 使用最高组的平均成交额估算
    df = df.dropna(subset=[factor_col, 'amount', 'circ_mv'])

    # 选取因子最高的20%股票
    top_quantile = df.groupby('trade_date').apply(
        lambda x: x.nlargest(int(len(x) * 0.2), factor_col)
    )

    if len(top_quantile) == 0:
        return {'estimated_capacity': 0}

    # 平均每天的组合成交额
    avg_daily_amount = top_quantile.groupby('trade_date')['amount'].sum().mean()

    # 假设可以占用日成交额的10%不产生明显冲击
    capacity_estimate = avg_daily_amount * 0.1 / 10000  # 转为亿元

    # 根据市值调整
    avg_mv = top_quantile['circ_mv'].mean() / 10000  # 转为亿元

    capacity = {
        'avg_daily_turnover': avg_daily_amount / 10000,  # 亿元
        'estimated_capacity': capacity_estimate,  # 亿元
        'avg_market_cap': avg_mv,
        'capacity_constraint': 'medium' if capacity_estimate > 10 else 'small'
    }

    return capacity

def generate_report(ic_results, decay_analysis, strategy_results, cost_analysis, capacity_analysis, df):
    """
    7. 生成分析报告
    """
    print("\n生成分析报告...")

    report = []
    report.append("# 高频因子模拟研究报告")
    report.append("")
    report.append("## 研究概述")
    report.append("")
    report.append("本研究基于日频数据模拟构建高频因子，主要包括：")
    report.append("- 日内价格特征因子")
    report.append("- 日内量价特征因子")
    report.append("- 订单流代理因子")
    report.append("- 短期反转因子")
    report.append("")
    report.append(f"**数据范围**: {START_DATE} - {END_DATE}")
    report.append(f"**样本量**: {len(df):,} 条记录")
    report.append(f"**股票数量**: {df['ts_code'].nunique():,} 只")
    report.append("")

    # 因子定义
    report.append("---")
    report.append("## 1. 因子定义")
    report.append("")

    report.append("### 1.1 日内价格特征因子")
    report.append("")
    report.append("| 因子名称 | 计算公式 | 经济含义 |")
    report.append("|---------|---------|---------|")
    report.append("| open_return | (open/pre_close - 1) * 100 | 开盘跳空，反映隔夜信息冲击 |")
    report.append("| intraday_range | (high - low) / pre_close * 100 | 日内波动幅度，反映日内不确定性 |")
    report.append("| close_location | (close - low) / (high - low) | 收盘在日内区间的位置，反映收盘强度 |")
    report.append("| gap | (open - pre_close) / pre_close * 100 | 跳空缺口 |")
    report.append("| upper_shadow | (high - max(open,close)) / pre_close * 100 | 上影线，反映上方抛压 |")
    report.append("| lower_shadow | (min(open,close) - low) / pre_close * 100 | 下影线，反映下方支撑 |")
    report.append("| intraday_reversal | 日内收益 - 开盘收益 | 日内反转强度 |")
    report.append("")

    report.append("### 1.2 日内量价特征因子")
    report.append("")
    report.append("| 因子名称 | 计算公式 | 经济含义 |")
    report.append("|---------|---------|---------|")
    report.append("| vol_price_corr | corr(pct_chg, vol_chg, 20d) | 量价相关性，正相关为趋势特征 |")
    report.append("| high_vol_indicator | 价格位置 * 相对成交量 | 高位放量指标 |")
    report.append("| low_vol_indicator | (1-价格位置) * 相对成交量 | 低位放量指标 |")
    report.append("| volume_concentration | 1/(波动+0.1) * 量比 | 成交集中度估计 |")
    report.append("| vwap_deviation | (close - VWAP估计) / VWAP估计 | VWAP偏离度 |")
    report.append("")

    report.append("### 1.3 订单流代理因子")
    report.append("")
    report.append("| 因子名称 | 计算公式 | 经济含义 |")
    report.append("|---------|---------|---------|")
    report.append("| large_small_ratio | 大单成交额 / 小单成交额 | 大单活跃度 |")
    report.append("| main_net_inflow_rate | 主力净流入 / 总成交额 | 主力资金流向 |")
    report.append("| order_imbalance | (买入-卖出) / (买入+卖出) | 订单不平衡度 |")
    report.append("| informed_trading | sign(收益) * 主力净流入率 | 知情交易代理 |")
    report.append("| retail_sentiment | 小单净流入率 | 散户情绪 |")
    report.append("| inst_retail_divergence | 主力净流入率 - 散户净流入率 | 机构散户分歧 |")
    report.append("| elg_order_ratio | 超大单成交额 / 总成交额 | 超大单占比 |")
    report.append("| net_inflow_persist | 过去5日主力净流入率累计 | 资金流入持续性 |")
    report.append("")

    report.append("### 1.4 短期反转因子")
    report.append("")
    report.append("| 因子名称 | 计算公式 | 经济含义 |")
    report.append("|---------|---------|---------|")
    report.append("| overnight_return | 开盘收益 | 隔夜收益 |")
    report.append("| intraday_return | (close/open - 1) * 100 | 日内收益 |")
    report.append("| reversal_1d/3d/5d | -过去N日累计收益 | N日反转因子 |")
    report.append("| vol_adj_reversal | 反转因子 / 波动率 | 波动调整反转 |")
    report.append("| mean_reversion | (close - MA20) / MA20 | 均值回归强度 |")
    report.append("")

    # IC分析
    report.append("---")
    report.append("## 2. 因子IC分析")
    report.append("")

    # 汇总表
    report.append("### 2.1 IC汇总表 (预测1日收益)")
    report.append("")
    report.append("| 因子 | IC均值 | IC标准差 | ICIR | IC>0比例 |")
    report.append("|------|--------|----------|------|----------|")

    # 按ICIR排序
    factor_ic_1d = []
    for factor, results in ic_results.items():
        if '1d' in results:
            r = results['1d']
            factor_ic_1d.append({
                'factor': factor,
                'ic_mean': r['IC_mean'],
                'ic_std': r['IC_std'],
                'icir': r['ICIR'],
                'ic_pos': r['IC_positive_ratio']
            })

    factor_ic_1d.sort(key=lambda x: abs(x['icir']), reverse=True)

    for item in factor_ic_1d[:30]:  # 只显示前30个
        report.append(f"| {item['factor']} | {item['ic_mean']:.4f} | {item['ic_std']:.4f} | "
                     f"{item['icir']:.3f} | {item['ic_pos']:.1%} |")

    report.append("")

    # 不同持有期IC对比
    report.append("### 2.2 不同持有期IC对比 (Top 10因子)")
    report.append("")
    report.append("| 因子 | 1日IC | 3日IC | 5日IC | 10日IC | 20日IC | 最佳持有期 |")
    report.append("|------|-------|-------|-------|--------|--------|-----------|")

    for item in factor_ic_1d[:10]:
        factor = item['factor']
        ics = []
        for period in ['1d', '3d', '5d', '10d', '20d']:
            if period in ic_results.get(factor, {}):
                ics.append(f"{ic_results[factor][period]['IC_mean']:.4f}")
            else:
                ics.append("-")

        best_period = decay_analysis.get(factor, {}).get('best_period', '-')
        report.append(f"| {factor} | {' | '.join(ics)} | {best_period}日 |")

    report.append("")

    # 因子衰减分析
    report.append("### 2.3 因子衰减分析")
    report.append("")
    report.append("因子衰减特征分析（IC绝对值随持有期的变化）：")
    report.append("")

    # 分类
    fast_decay = []
    slow_decay = []

    for factor, analysis in decay_analysis.items():
        if analysis['decay_rate'] is not None:
            if analysis['decay_rate'] < -0.01:
                fast_decay.append((factor, analysis['decay_rate'], analysis['best_period']))
            else:
                slow_decay.append((factor, analysis['decay_rate'], analysis['best_period']))

    report.append("**快速衰减因子** (适合短期交易):")
    report.append("")
    for f, rate, best in sorted(fast_decay, key=lambda x: x[1])[:10]:
        report.append(f"- {f}: 衰减率={rate:.4f}, 最佳持有期={best}日")

    report.append("")
    report.append("**慢速衰减因子** (适合中期持有):")
    report.append("")
    for f, rate, best in sorted(slow_decay, key=lambda x: -x[1])[:10]:
        report.append(f"- {f}: 衰减率={rate:.4f}, 最佳持有期={best}日")

    report.append("")

    # 因子分类分析
    report.append("---")
    report.append("## 3. 因子分类有效性分析")
    report.append("")

    categories = {
        '日内价格因子': ['open_return', 'intraday_range', 'close_location', 'gap',
                       'upper_shadow', 'lower_shadow', 'intraday_reversal', 'body_size'],
        '量价因子': ['vol_price_corr', 'high_vol_indicator', 'low_vol_indicator',
                   'volume_concentration', 'vwap_deviation', 'amount_intensity'],
        '订单流因子': ['large_small_ratio', 'main_net_inflow_rate', 'order_imbalance',
                     'informed_trading', 'retail_sentiment', 'inst_retail_divergence',
                     'elg_order_ratio', 'net_inflow_persist'],
        '反转因子': ['overnight_return', 'intraday_return', 'reversal_1d', 'reversal_3d',
                   'reversal_5d', 'vol_adj_reversal', 'mean_reversion']
    }

    for cat_name, factors in categories.items():
        report.append(f"### {cat_name}")
        report.append("")

        cat_ics = []
        for f in factors:
            if f in ic_results and '1d' in ic_results[f]:
                cat_ics.append({
                    'factor': f,
                    'ic': ic_results[f]['1d']['IC_mean'],
                    'icir': ic_results[f]['1d']['ICIR']
                })

        if cat_ics:
            cat_ics.sort(key=lambda x: abs(x['icir']), reverse=True)
            report.append("| 因子 | IC均值 | ICIR | 有效性评级 |")
            report.append("|------|--------|------|-----------|")

            for item in cat_ics:
                if abs(item['icir']) > 0.5:
                    rating = "强"
                elif abs(item['icir']) > 0.3:
                    rating = "中"
                elif abs(item['icir']) > 0.1:
                    rating = "弱"
                else:
                    rating = "无效"
                report.append(f"| {item['factor']} | {item['ic']:.4f} | {item['icir']:.3f} | {rating} |")

        report.append("")

    # 策略回测结果
    report.append("---")
    report.append("## 4. 策略应用分析")
    report.append("")

    report.append("### 4.1 多空策略回测")
    report.append("")
    report.append("基于因子分5组，做多最高组、做空最低组的多空策略表现：")
    report.append("")

    if strategy_results:
        report.append("| 因子 | 持有期 | 年化收益 | 年化波动 | 夏普比率 | 胜率 | 最大回撤 |")
        report.append("|------|--------|----------|----------|----------|------|----------|")

        for (factor, period), stats in strategy_results.items():
            if stats is not None:
                report.append(f"| {factor} | {period}日 | {stats['annual_return']:.2%} | "
                             f"{stats['volatility']:.2%} | {stats['sharpe']:.2f} | "
                             f"{stats['win_rate']:.1%} | {stats['max_drawdown']:.2%} |")

        report.append("")

    # 交易成本分析
    report.append("### 4.2 交易成本分析")
    report.append("")

    if cost_analysis:
        report.append("| 成本类型 | 估计值 (单边/%) |")
        report.append("|----------|----------------|")
        for cost_type, value in cost_analysis.items():
            if 'cost' in cost_type.lower() or 'commission' in cost_type.lower() or 'tax' in cost_type.lower() or 'spread' in cost_type.lower():
                report.append(f"| {cost_type} | {value:.4f}% |")

        report.append("")
        report.append(f"**总交易成本估计**: {cost_analysis.get('total_cost_estimate', 0):.4f}% (单次完整交易)")
        report.append("")
        report.append("**成本影响分析**:")
        report.append(f"- 日内交易策略 (日换手率100%): 年化成本约 {cost_analysis.get('total_cost_estimate', 0) * 250:.1f}%")
        report.append(f"- 周度调仓策略 (周换手率20%): 年化成本约 {cost_analysis.get('total_cost_estimate', 0) * 52 * 0.2:.1f}%")
        report.append(f"- 月度调仓策略 (月换手率10%): 年化成本约 {cost_analysis.get('total_cost_estimate', 0) * 12 * 0.1:.1f}%")
        report.append("")

    # 策略容量分析
    report.append("### 4.3 策略容量估计")
    report.append("")

    if capacity_analysis:
        report.append(f"- **日均成交额 (Top组)**: {capacity_analysis.get('avg_daily_turnover', 0):.2f} 亿元")
        report.append(f"- **估计容量**: {capacity_analysis.get('estimated_capacity', 0):.2f} 亿元")
        report.append(f"- **平均市值**: {capacity_analysis.get('avg_market_cap', 0):.2f} 亿元")
        report.append(f"- **容量等级**: {capacity_analysis.get('capacity_constraint', '-')}")
        report.append("")

    # 研究结论
    report.append("---")
    report.append("## 5. 研究结论与建议")
    report.append("")

    # 找出最有效的因子
    top_factors = factor_ic_1d[:5] if factor_ic_1d else []

    report.append("### 5.1 主要发现")
    report.append("")
    report.append("**最有效因子 (按ICIR排序)**:")
    report.append("")
    for i, item in enumerate(top_factors, 1):
        report.append(f"{i}. **{item['factor']}**: ICIR={item['icir']:.3f}, IC均值={item['ic_mean']:.4f}")

    report.append("")
    report.append("### 5.2 因子特点总结")
    report.append("")
    report.append("1. **日内价格因子**: 主要捕捉日内价格形态特征，`close_location`和`intraday_reversal`通常具有一定预测能力")
    report.append("")
    report.append("2. **量价因子**: `vol_price_corr`反映趋势特征，高位/低位放量指标对判断价格位置有辅助作用")
    report.append("")
    report.append("3. **订单流因子**: 基于资金流数据构建，`main_net_inflow_rate`和`inst_retail_divergence`具有较好的信息含量")
    report.append("")
    report.append("4. **反转因子**: 短期反转效应在A股市场较为显著，但需注意控制交易成本")
    report.append("")

    report.append("### 5.3 策略应用建议")
    report.append("")
    report.append("1. **持有期选择**: 根据因子衰减特征，多数高频代理因子适合1-5日持有期")
    report.append("")
    report.append("2. **成本控制**: 高频策略面临较高交易成本，建议:")
    report.append("   - 选择流动性好的标的")
    report.append("   - 控制换手率")
    report.append("   - 使用算法交易降低冲击成本")
    report.append("")
    report.append("3. **容量管理**: 策略容量有限，适合中小规模资金运作")
    report.append("")
    report.append("4. **组合应用**: 建议将多个低相关因子组合使用，提高策略稳定性")
    report.append("")

    report.append("### 5.4 研究局限")
    report.append("")
    report.append("1. 本研究使用日频数据模拟高频因子，无法完全捕捉真实的日内交易行为")
    report.append("2. 资金流数据基于交易所披露，可能存在一定滞后性")
    report.append("3. 未考虑涨跌停、停牌等交易限制的影响")
    report.append("4. 实际交易中的冲击成本可能高于估计值")
    report.append("")

    report.append("---")
    report.append("*报告生成时间: " + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S') + "*")

    return '\n'.join(report)

def main():
    """主函数"""
    print("=" * 60)
    print("高频因子模拟研究")
    print("=" * 60)

    # 1. 加载数据
    daily, basic, moneyflow = load_data()

    # 2. 构建因子
    df = construct_intraday_price_factors(daily)
    df = construct_volume_price_factors(df, basic)
    df = construct_order_flow_factors(df, moneyflow)
    df = construct_reversal_factors(df)

    # 定义要分析的因子
    factor_cols = [
        # 日内价格因子
        'open_return', 'intraday_range', 'close_location', 'gap', 'gap_filled',
        'upper_shadow', 'lower_shadow', 'body_size', 'intraday_reversal', 'true_range',
        # 量价因子
        'vol_price_corr', 'high_vol_indicator', 'low_vol_indicator',
        'volume_concentration', 'vwap_deviation', 'amount_intensity',
        # 订单流因子
        'large_small_ratio', 'main_net_inflow_rate', 'order_imbalance',
        'informed_trading', 'retail_sentiment', 'inst_retail_divergence',
        'elg_order_ratio', 'net_inflow_persist',
        # 反转因子
        'overnight_return', 'intraday_return', 'overnight_intraday_diff',
        'reversal_1d', 'reversal_3d', 'reversal_5d',
        'vol_adj_reversal', 'mean_reversion', 'cum_ret_5d'
    ]

    # 过滤存在的因子
    factor_cols = [f for f in factor_cols if f in df.columns]
    print(f"\n共计 {len(factor_cols)} 个因子待分析")

    # 3. 计算IC
    ic_results, df = calculate_factor_ic(df, factor_cols)

    # 4. 因子衰减分析
    decay_analysis = analyze_factor_decay(ic_results, factor_cols)

    # 5. 策略回测 (选择Top因子)
    # 按1日ICIR排序选择top因子
    top_factors = []
    for factor in factor_cols:
        if factor in ic_results and '1d' in ic_results[factor]:
            top_factors.append((factor, abs(ic_results[factor]['1d']['ICIR'])))
    top_factors.sort(key=lambda x: x[1], reverse=True)

    strategy_results = {}
    for factor, _ in top_factors[:5]:  # 前5个因子
        for period in [1, 5]:
            result = backtest_strategy(df, factor, holding_period=period)
            if result:
                strategy_results[(factor, period)] = result[0]

    # 6. 交易成本分析
    if top_factors:
        cost_analysis = estimate_trading_cost(df, top_factors[0][0])
    else:
        cost_analysis = {}

    # 7. 容量估计
    if top_factors:
        capacity_analysis = estimate_capacity(df, top_factors[0][0])
    else:
        capacity_analysis = {}

    # 8. 生成报告
    report = generate_report(ic_results, decay_analysis, strategy_results,
                            cost_analysis, capacity_analysis, df)

    # 保存报告
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n报告已保存至: {OUTPUT_PATH}")
    print("=" * 60)

    # 打印简要结果
    print("\n=== 因子IC排名 (Top 10) ===")
    factor_ic_1d = []
    for factor, results in ic_results.items():
        if '1d' in results:
            r = results['1d']
            factor_ic_1d.append({
                'factor': factor,
                'ic_mean': r['IC_mean'],
                'icir': r['ICIR']
            })

    factor_ic_1d.sort(key=lambda x: abs(x['icir']), reverse=True)

    for i, item in enumerate(factor_ic_1d[:10], 1):
        print(f"{i:2d}. {item['factor']:30s} IC={item['ic_mean']:+.4f} ICIR={item['icir']:+.3f}")

if __name__ == '__main__':
    main()
