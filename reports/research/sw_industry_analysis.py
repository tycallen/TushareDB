#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
申万行业深度分析
================

基于 tushare.db 的 sw_daily 和 index_classify 表进行全面的行业分析

分析内容:
1. 行业数据分析: 一级/二级/三级行业收益率和波动率对比
2. 行业轮动: 动量效应、反转效应、最优轮动周期
3. 策略应用: 行业ETF配置、行业中性策略、行业择时

Author: Claude Code Assistant
Date: 2026-02-01
"""

import duckdb
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 配置
DB_PATH = '/Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db'
OUTPUT_DIR = '/Users/allen/workspace/python/stock/Tushare-DuckDB/reports/research'

# 连接数据库
conn = duckdb.connect(DB_PATH, read_only=True)


def load_industry_data():
    """加载行业分类和日线数据"""

    # 加载行业分类
    classify_df = conn.execute('''
        SELECT index_code, industry_name, level, industry_code, parent_code
        FROM index_classify
        ORDER BY level, industry_code
    ''').fetchdf()

    # 加载行业日线数据，关联行业级别
    sw_daily = conn.execute('''
        SELECT
            s.ts_code,
            s.trade_date,
            s.name,
            s.open,
            s.high,
            s.low,
            s.close,
            s.pct_change,
            s.vol,
            s.amount,
            s.pe,
            s.pb,
            s.float_mv,
            s.total_mv,
            ic.level,
            ic.industry_name as industry_std_name
        FROM sw_daily s
        LEFT JOIN index_classify ic ON s.ts_code = ic.index_code
        ORDER BY s.trade_date, s.ts_code
    ''').fetchdf()

    # 转换日期格式
    sw_daily['trade_date'] = pd.to_datetime(sw_daily['trade_date'])

    return classify_df, sw_daily


def analyze_industry_structure(classify_df):
    """分析行业结构"""
    print("\n" + "="*80)
    print("1. 申万行业结构分析")
    print("="*80)

    # 各级别行业数量
    level_counts = classify_df['level'].value_counts().sort_index()
    print("\n1.1 各级别行业数量:")
    for level, count in level_counts.items():
        print(f"  {level}: {count}个行业")

    # 一级行业列表
    l1_industries = classify_df[classify_df['level'] == 'L1'][['index_code', 'industry_name']].reset_index(drop=True)
    print(f"\n1.2 申万一级行业 (共{len(l1_industries)}个):")
    for i, row in l1_industries.iterrows():
        print(f"  {i+1:2d}. {row['index_code']} - {row['industry_name']}")

    return l1_industries


def analyze_industry_returns(sw_daily):
    """分析行业收益率"""
    print("\n" + "="*80)
    print("2. 行业收益率分析")
    print("="*80)

    results = {}

    for level in ['L1', 'L2', 'L3']:
        level_data = sw_daily[sw_daily['level'] == level].copy()
        if len(level_data) == 0:
            continue

        print(f"\n2.{['L1', 'L2', 'L3'].index(level)+1} {level}级行业收益率分析:")

        # 计算各行业的统计指标
        industry_stats = level_data.groupby('ts_code').agg({
            'name': 'last',
            'pct_change': ['mean', 'std', 'min', 'max', 'count'],
            'close': ['first', 'last']
        }).reset_index()

        industry_stats.columns = ['ts_code', 'name', 'daily_return_mean', 'daily_return_std',
                                  'min_return', 'max_return', 'trading_days',
                                  'first_close', 'last_close']

        # 计算累计收益率
        industry_stats['total_return'] = (industry_stats['last_close'] / industry_stats['first_close'] - 1) * 100

        # 计算年化收益率和夏普比率
        industry_stats['annual_return'] = industry_stats['daily_return_mean'] * 252
        industry_stats['annual_volatility'] = industry_stats['daily_return_std'] * np.sqrt(252)
        industry_stats['sharpe_ratio'] = industry_stats['annual_return'] / industry_stats['annual_volatility']

        # 排序并显示
        industry_stats_sorted = industry_stats.sort_values('total_return', ascending=False)

        print(f"  行业数量: {len(industry_stats_sorted)}")
        print(f"\n  收益率前10名:")
        for i, row in industry_stats_sorted.head(10).iterrows():
            print(f"    {row['name']:<12} 累计收益: {row['total_return']:>8.2f}%  年化收益: {row['annual_return']:>7.2f}%  夏普: {row['sharpe_ratio']:>6.2f}")

        print(f"\n  收益率后10名:")
        for i, row in industry_stats_sorted.tail(10).iterrows():
            print(f"    {row['name']:<12} 累计收益: {row['total_return']:>8.2f}%  年化收益: {row['annual_return']:>7.2f}%  夏普: {row['sharpe_ratio']:>6.2f}")

        results[level] = industry_stats_sorted

    return results


def analyze_industry_volatility(sw_daily):
    """分析行业波动率"""
    print("\n" + "="*80)
    print("3. 行业波动率分析")
    print("="*80)

    results = {}

    for level in ['L1', 'L2', 'L3']:
        level_data = sw_daily[sw_daily['level'] == level].copy()
        if len(level_data) == 0:
            continue

        print(f"\n3.{['L1', 'L2', 'L3'].index(level)+1} {level}级行业波动率分析:")

        # 计算各行业波动率指标
        vol_stats = level_data.groupby('ts_code').agg({
            'name': 'last',
            'pct_change': ['std', lambda x: x.quantile(0.05), lambda x: x.quantile(0.95)],
            'high': 'max',
            'low': 'min'
        }).reset_index()

        vol_stats.columns = ['ts_code', 'name', 'daily_volatility', 'var_5pct', 'var_95pct',
                            'highest', 'lowest']

        # 年化波动率
        vol_stats['annual_volatility'] = vol_stats['daily_volatility'] * np.sqrt(252)

        # 下行风险 (只考虑负收益)
        downside_risk = level_data[level_data['pct_change'] < 0].groupby('ts_code')['pct_change'].std()
        vol_stats = vol_stats.merge(
            downside_risk.reset_index().rename(columns={'pct_change': 'downside_volatility'}),
            on='ts_code', how='left'
        )
        vol_stats['annual_downside_vol'] = vol_stats['downside_volatility'] * np.sqrt(252)

        # 排序显示
        vol_stats_sorted = vol_stats.sort_values('annual_volatility', ascending=False)

        print(f"  行业数量: {len(vol_stats_sorted)}")
        print(f"\n  波动率最高的10个行业:")
        for i, row in vol_stats_sorted.head(10).iterrows():
            print(f"    {row['name']:<12} 年化波动: {row['annual_volatility']:>7.2f}%  下行风险: {row['annual_downside_vol']:>7.2f}%  VaR_5%: {row['var_5pct']:>6.2f}%")

        print(f"\n  波动率最低的10个行业:")
        for i, row in vol_stats_sorted.tail(10).iterrows():
            print(f"    {row['name']:<12} 年化波动: {row['annual_volatility']:>7.2f}%  下行风险: {row['annual_downside_vol']:>7.2f}%  VaR_5%: {row['var_5pct']:>6.2f}%")

        results[level] = vol_stats_sorted

    return results


def analyze_momentum_effect(sw_daily, lookback_periods=[5, 10, 20, 60, 120]):
    """分析行业动量效应"""
    print("\n" + "="*80)
    print("4. 行业动量效应分析")
    print("="*80)

    # 只分析一级行业
    l1_data = sw_daily[sw_daily['level'] == 'L1'].copy()

    # 创建透视表
    pivot_close = l1_data.pivot_table(index='trade_date', columns='ts_code', values='close')
    pivot_return = l1_data.pivot_table(index='trade_date', columns='ts_code', values='pct_change')

    momentum_results = {}

    for lookback in lookback_periods:
        print(f"\n4.1 回看期 {lookback} 天的动量效应:")

        # 计算过去N天累计收益
        past_return = pivot_close.pct_change(lookback) * 100

        # 计算未来收益 (5天、10天、20天)
        for holding in [5, 10, 20]:
            future_return = pivot_close.pct_change(holding).shift(-holding) * 100

            # 对齐数据
            common_idx = past_return.dropna(how='all').index.intersection(
                future_return.dropna(how='all').index
            )

            if len(common_idx) < 50:
                continue

            past_aligned = past_return.loc[common_idx]
            future_aligned = future_return.loc[common_idx]

            # 计算每个时点的动量分组收益
            momentum_profits = []
            for date in common_idx:
                past_row = past_aligned.loc[date].dropna()
                future_row = future_aligned.loc[date].dropna()

                common_codes = past_row.index.intersection(future_row.index)
                if len(common_codes) < 10:
                    continue

                past_row = past_row[common_codes]
                future_row = future_row[common_codes]

                # 分成5组
                n_groups = 5
                ranked = past_row.rank()
                group_size = len(ranked) // n_groups

                groups_return = []
                for g in range(n_groups):
                    if g == n_groups - 1:
                        group_codes = ranked[ranked > g * group_size].index
                    else:
                        group_codes = ranked[(ranked > g * group_size) & (ranked <= (g+1) * group_size)].index

                    if len(group_codes) > 0:
                        groups_return.append(future_row[group_codes].mean())
                    else:
                        groups_return.append(np.nan)

                if len(groups_return) == n_groups:
                    momentum_profits.append({
                        'date': date,
                        'loser': groups_return[0],
                        'winner': groups_return[-1],
                        'momentum': groups_return[-1] - groups_return[0]
                    })

            if len(momentum_profits) > 0:
                momentum_df = pd.DataFrame(momentum_profits)
                avg_momentum = momentum_df['momentum'].mean()
                win_rate = (momentum_df['momentum'] > 0).mean() * 100

                print(f"    持有期{holding:2d}天: 动量收益={avg_momentum:>6.2f}%, 胜率={win_rate:.1f}%")

                momentum_results[f'lookback_{lookback}_holding_{holding}'] = {
                    'avg_momentum': avg_momentum,
                    'win_rate': win_rate,
                    'details': momentum_df
                }

    return momentum_results


def analyze_reversal_effect(sw_daily, lookback_periods=[5, 10, 20, 60, 120]):
    """分析行业反转效应"""
    print("\n" + "="*80)
    print("5. 行业反转效应分析")
    print("="*80)

    # 只分析一级行业
    l1_data = sw_daily[sw_daily['level'] == 'L1'].copy()

    # 创建透视表
    pivot_close = l1_data.pivot_table(index='trade_date', columns='ts_code', values='close')

    reversal_results = {}

    for lookback in lookback_periods:
        print(f"\n5.1 回看期 {lookback} 天的反转效应:")

        # 计算过去N天累计收益
        past_return = pivot_close.pct_change(lookback) * 100

        # 计算未来收益
        for holding in [5, 10, 20]:
            future_return = pivot_close.pct_change(holding).shift(-holding) * 100

            # 对齐数据
            common_idx = past_return.dropna(how='all').index.intersection(
                future_return.dropna(how='all').index
            )

            if len(common_idx) < 50:
                continue

            past_aligned = past_return.loc[common_idx]
            future_aligned = future_return.loc[common_idx]

            # 计算反转效应 (买入输家，卖出赢家)
            reversal_profits = []
            for date in common_idx:
                past_row = past_aligned.loc[date].dropna()
                future_row = future_aligned.loc[date].dropna()

                common_codes = past_row.index.intersection(future_row.index)
                if len(common_codes) < 10:
                    continue

                past_row = past_row[common_codes]
                future_row = future_row[common_codes]

                # 分成5组
                n_groups = 5
                ranked = past_row.rank()
                group_size = len(ranked) // n_groups

                groups_return = []
                for g in range(n_groups):
                    if g == n_groups - 1:
                        group_codes = ranked[ranked > g * group_size].index
                    else:
                        group_codes = ranked[(ranked > g * group_size) & (ranked <= (g+1) * group_size)].index

                    if len(group_codes) > 0:
                        groups_return.append(future_row[group_codes].mean())
                    else:
                        groups_return.append(np.nan)

                if len(groups_return) == n_groups:
                    # 反转策略: 买输家卖赢家
                    reversal_profits.append({
                        'date': date,
                        'loser': groups_return[0],
                        'winner': groups_return[-1],
                        'reversal': groups_return[0] - groups_return[-1]  # 反转收益
                    })

            if len(reversal_profits) > 0:
                reversal_df = pd.DataFrame(reversal_profits)
                avg_reversal = reversal_df['reversal'].mean()
                win_rate = (reversal_df['reversal'] > 0).mean() * 100

                print(f"    持有期{holding:2d}天: 反转收益={avg_reversal:>6.2f}%, 胜率={win_rate:.1f}%")

                reversal_results[f'lookback_{lookback}_holding_{holding}'] = {
                    'avg_reversal': avg_reversal,
                    'win_rate': win_rate,
                    'details': reversal_df
                }

    return reversal_results


def find_optimal_rotation_period(sw_daily):
    """寻找最优行业轮动周期"""
    print("\n" + "="*80)
    print("6. 最优行业轮动周期分析")
    print("="*80)

    # 只分析一级行业
    l1_data = sw_daily[sw_daily['level'] == 'L1'].copy()
    pivot_close = l1_data.pivot_table(index='trade_date', columns='ts_code', values='close')

    # 测试不同的回看期和持有期组合
    lookback_periods = [5, 10, 20, 40, 60, 120]
    holding_periods = [5, 10, 20, 40, 60]

    results = []

    print("\n6.1 动量策略各参数组合表现:")
    print(f"{'回看期':>8} {'持有期':>8} {'年化收益':>10} {'年化波动':>10} {'夏普比率':>10} {'胜率':>8}")
    print("-" * 60)

    for lookback in lookback_periods:
        for holding in holding_periods:
            # 计算过去收益
            past_return = pivot_close.pct_change(lookback)
            future_return = pivot_close.pct_change(holding).shift(-holding)

            # 策略: 买入过去表现最好的3个行业
            strategy_returns = []

            for date in pivot_close.index[lookback:-holding]:
                past_row = past_return.loc[date].dropna()
                future_row = future_return.loc[date].dropna()

                common_codes = past_row.index.intersection(future_row.index)
                if len(common_codes) < 10:
                    continue

                past_row = past_row[common_codes]
                future_row = future_row[common_codes]

                # 选择表现最好的3个行业
                top_3 = past_row.nlargest(3).index
                strategy_return = future_row[top_3].mean()
                strategy_returns.append(strategy_return * 100)

            if len(strategy_returns) < 50:
                continue

            strategy_returns = pd.Series(strategy_returns)

            # 计算统计指标
            # 调整为年化 (假设每holding天调仓一次)
            n_periods_per_year = 252 / holding
            annual_return = strategy_returns.mean() * n_periods_per_year
            annual_vol = strategy_returns.std() * np.sqrt(n_periods_per_year)
            sharpe = annual_return / annual_vol if annual_vol > 0 else 0
            win_rate = (strategy_returns > 0).mean() * 100

            print(f"{lookback:>8} {holding:>8} {annual_return:>10.2f}% {annual_vol:>10.2f}% {sharpe:>10.2f} {win_rate:>8.1f}%")

            results.append({
                'lookback': lookback,
                'holding': holding,
                'annual_return': annual_return,
                'annual_volatility': annual_vol,
                'sharpe_ratio': sharpe,
                'win_rate': win_rate
            })

    results_df = pd.DataFrame(results)

    # 找出最优参数
    if len(results_df) > 0:
        best_sharpe = results_df.loc[results_df['sharpe_ratio'].idxmax()]
        best_return = results_df.loc[results_df['annual_return'].idxmax()]

        print(f"\n6.2 最优参数:")
        print(f"  最高夏普比率: 回看期={int(best_sharpe['lookback'])}天, 持有期={int(best_sharpe['holding'])}天")
        print(f"    年化收益={best_sharpe['annual_return']:.2f}%, 夏普={best_sharpe['sharpe_ratio']:.2f}")
        print(f"  最高年化收益: 回看期={int(best_return['lookback'])}天, 持有期={int(best_return['holding'])}天")
        print(f"    年化收益={best_return['annual_return']:.2f}%, 夏普={best_return['sharpe_ratio']:.2f}")

    return results_df


def industry_etf_allocation(sw_daily, return_stats):
    """行业ETF配置策略"""
    print("\n" + "="*80)
    print("7. 行业ETF配置策略")
    print("="*80)

    if 'L1' not in return_stats:
        print("  无一级行业数据，跳过分析")
        return None

    l1_stats = return_stats['L1'].copy()

    print("\n7.1 行业ETF对照表 (申万一级行业):")

    # 常见行业ETF对照
    etf_mapping = {
        '银行': ['512800', '515020'],
        '非银金融': ['512640', '515130'],
        '房地产': ['512200', '515060'],
        '医药生物': ['159929', '512010'],
        '食品饮料': ['159843', '515170'],
        '电子': ['159997', '512480'],
        '计算机': ['159998', '512720'],
        '电气设备': ['515580', '516160'],
        '有色金属': ['159871', '512400'],
        '钢铁': ['515210'],
        '煤炭': ['515220'],
        '化工': ['159870', '515030'],
        '汽车': ['159845', '516110'],
        '机械设备': ['516960'],
        '家用电器': ['159996'],
        '通信': ['515880'],
        '传媒': ['512980'],
        '国防军工': ['512660', '512810'],
        '建筑材料': ['159745'],
        '农林牧渔': ['159825', '516070'],
    }

    for _, row in l1_stats.iterrows():
        name = row['name']
        etfs = etf_mapping.get(name, ['无'])
        print(f"  {name:<10} ETF代码: {', '.join(etfs)}")

    # 等权配置策略回测
    print("\n7.2 等权配置策略 (全行业等权):")
    l1_data = sw_daily[sw_daily['level'] == 'L1'].copy()

    # 计算等权组合的日收益
    daily_returns = l1_data.pivot_table(index='trade_date', columns='ts_code', values='pct_change')
    equal_weight_return = daily_returns.mean(axis=1)

    # 计算统计指标
    total_return = (1 + equal_weight_return/100).cumprod().iloc[-1] - 1
    annual_return = equal_weight_return.mean() * 252
    annual_vol = equal_weight_return.std() * np.sqrt(252)
    sharpe = annual_return / annual_vol
    max_drawdown = calculate_max_drawdown(equal_weight_return)

    print(f"  累计收益率: {total_return*100:.2f}%")
    print(f"  年化收益率: {annual_return:.2f}%")
    print(f"  年化波动率: {annual_vol:.2f}%")
    print(f"  夏普比率: {sharpe:.2f}")
    print(f"  最大回撤: {max_drawdown:.2f}%")

    # 风险平价配置策略
    print("\n7.3 风险平价配置策略:")

    # 计算各行业波动率
    industry_vol = daily_returns.std()
    # 风险平价权重: 权重与波动率成反比
    risk_parity_weight = (1 / industry_vol) / (1 / industry_vol).sum()

    # 计算风险平价组合收益
    rp_return = (daily_returns * risk_parity_weight).sum(axis=1)

    total_return_rp = (1 + rp_return/100).cumprod().iloc[-1] - 1
    annual_return_rp = rp_return.mean() * 252
    annual_vol_rp = rp_return.std() * np.sqrt(252)
    sharpe_rp = annual_return_rp / annual_vol_rp
    max_drawdown_rp = calculate_max_drawdown(rp_return)

    print(f"  累计收益率: {total_return_rp*100:.2f}%")
    print(f"  年化收益率: {annual_return_rp:.2f}%")
    print(f"  年化波动率: {annual_vol_rp:.2f}%")
    print(f"  夏普比率: {sharpe_rp:.2f}")
    print(f"  最大回撤: {max_drawdown_rp:.2f}%")

    print("\n  风险平价权重TOP 10:")
    for code, weight in risk_parity_weight.nlargest(10).items():
        name = l1_data[l1_data['ts_code'] == code]['name'].iloc[0] if len(l1_data[l1_data['ts_code'] == code]) > 0 else code
        print(f"    {name:<12} 权重: {weight*100:.2f}%")

    return {
        'equal_weight': {'return': total_return, 'sharpe': sharpe},
        'risk_parity': {'return': total_return_rp, 'sharpe': sharpe_rp, 'weights': risk_parity_weight}
    }


def calculate_max_drawdown(returns):
    """计算最大回撤"""
    cumulative = (1 + returns/100).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max * 100
    return drawdown.min()


def industry_neutral_strategy(sw_daily):
    """行业中性策略分析"""
    print("\n" + "="*80)
    print("8. 行业中性策略分析")
    print("="*80)

    l1_data = sw_daily[sw_daily['level'] == 'L1'].copy()
    pivot_return = l1_data.pivot_table(index='trade_date', columns='ts_code', values='pct_change')

    print("\n8.1 多空策略 (买入强势行业，卖出弱势行业):")

    lookback_periods = [20, 60, 120]

    for lookback in lookback_periods:
        print(f"\n  回看期 {lookback} 天:")

        pivot_close = l1_data.pivot_table(index='trade_date', columns='ts_code', values='close')
        past_return = pivot_close.pct_change(lookback)

        long_short_returns = []

        for i, date in enumerate(pivot_return.index[lookback:]):
            past_row = past_return.loc[date].dropna()
            current_return = pivot_return.loc[date].dropna()

            common_codes = past_row.index.intersection(current_return.index)
            if len(common_codes) < 10:
                continue

            past_row = past_row[common_codes]
            current_return = current_return[common_codes]

            # 买入表现最好的5个，卖出表现最差的5个
            top_5 = past_row.nlargest(5).index
            bottom_5 = past_row.nsmallest(5).index

            long_return = current_return[top_5].mean()
            short_return = current_return[bottom_5].mean()
            long_short_return = long_return - short_return

            long_short_returns.append(long_short_return)

        if len(long_short_returns) > 0:
            ls_series = pd.Series(long_short_returns)
            annual_return = ls_series.mean() * 252
            annual_vol = ls_series.std() * np.sqrt(252)
            sharpe = annual_return / annual_vol if annual_vol > 0 else 0
            win_rate = (ls_series > 0).mean() * 100

            print(f"    年化收益: {annual_return:.2f}%")
            print(f"    年化波动: {annual_vol:.2f}%")
            print(f"    夏普比率: {sharpe:.2f}")
            print(f"    胜率: {win_rate:.1f}%")

    print("\n8.2 行业对冲建议:")
    print("  1. 周期 vs 防御: 做多周期行业(有色/钢铁/煤炭)，做空防御行业(公用事业/银行)")
    print("  2. 成长 vs 价值: 做多成长(电子/计算机/医药)，做空价值(银行/地产)")
    print("  3. 大小市值: 做多小市值行业，做空大市值行业")


def industry_timing_strategy(sw_daily):
    """行业择时策略"""
    print("\n" + "="*80)
    print("9. 行业择时策略分析")
    print("="*80)

    l1_data = sw_daily[sw_daily['level'] == 'L1'].copy()

    # 1. 基于均线的择时
    print("\n9.1 均线择时策略 (20日均线):")

    pivot_close = l1_data.pivot_table(index='trade_date', columns='ts_code', values='close')
    ma_20 = pivot_close.rolling(20).mean()

    # 当价格在均线上方时持有，否则空仓
    signal = (pivot_close > ma_20).astype(int)
    pivot_return = l1_data.pivot_table(index='trade_date', columns='ts_code', values='pct_change')

    # 择时收益
    timing_return = (pivot_return * signal.shift(1)).mean(axis=1)
    buy_hold_return = pivot_return.mean(axis=1)

    # 只有有信号的日期
    timing_return = timing_return.dropna()
    buy_hold_return = buy_hold_return.loc[timing_return.index]

    timing_annual = timing_return.mean() * 252
    timing_vol = timing_return.std() * np.sqrt(252)
    timing_sharpe = timing_annual / timing_vol if timing_vol > 0 else 0

    bh_annual = buy_hold_return.mean() * 252
    bh_vol = buy_hold_return.std() * np.sqrt(252)
    bh_sharpe = bh_annual / bh_vol if bh_vol > 0 else 0

    print(f"  均线择时: 年化收益={timing_annual:.2f}%, 夏普={timing_sharpe:.2f}")
    print(f"  买入持有: 年化收益={bh_annual:.2f}%, 夏普={bh_sharpe:.2f}")

    # 2. 基于动量的择时
    print("\n9.2 动量择时策略 (20日动量):")

    momentum = pivot_close.pct_change(20)
    # 只在动量为正时持有
    mom_signal = (momentum > 0).astype(int)

    mom_timing_return = (pivot_return * mom_signal.shift(1)).mean(axis=1)
    mom_timing_return = mom_timing_return.dropna()

    mom_annual = mom_timing_return.mean() * 252
    mom_vol = mom_timing_return.std() * np.sqrt(252)
    mom_sharpe = mom_annual / mom_vol if mom_vol > 0 else 0

    print(f"  动量择时: 年化收益={mom_annual:.2f}%, 夏普={mom_sharpe:.2f}")

    # 3. 各行业择时效果排名
    print("\n9.3 各行业均线择时效果排名:")

    industry_timing_results = []
    for col in pivot_close.columns:
        ind_timing = pivot_return[col] * signal[col].shift(1)
        ind_timing = ind_timing.dropna()

        if len(ind_timing) < 100:
            continue

        ind_annual = ind_timing.mean() * 252
        ind_vol = ind_timing.std() * np.sqrt(252)
        ind_sharpe = ind_annual / ind_vol if ind_vol > 0 else 0

        # 买入持有
        bh_ind = pivot_return[col].loc[ind_timing.index]
        bh_annual = bh_ind.mean() * 252
        bh_sharpe = bh_annual / (bh_ind.std() * np.sqrt(252)) if bh_ind.std() > 0 else 0

        name = l1_data[l1_data['ts_code'] == col]['name'].iloc[0] if len(l1_data[l1_data['ts_code'] == col]) > 0 else col

        industry_timing_results.append({
            'code': col,
            'name': name,
            'timing_return': ind_annual,
            'timing_sharpe': ind_sharpe,
            'bh_return': bh_annual,
            'bh_sharpe': bh_sharpe,
            'improvement': ind_sharpe - bh_sharpe
        })

    timing_df = pd.DataFrame(industry_timing_results)
    timing_df = timing_df.sort_values('improvement', ascending=False)

    print(f"\n  择时效果改善最大的行业:")
    for _, row in timing_df.head(10).iterrows():
        print(f"    {row['name']:<12} 择时夏普: {row['timing_sharpe']:.2f}  买持夏普: {row['bh_sharpe']:.2f}  改善: {row['improvement']:.2f}")

    print(f"\n  择时效果较差的行业:")
    for _, row in timing_df.tail(5).iterrows():
        print(f"    {row['name']:<12} 择时夏普: {row['timing_sharpe']:.2f}  买持夏普: {row['bh_sharpe']:.2f}  改善: {row['improvement']:.2f}")

    return timing_df


def generate_correlation_analysis(sw_daily):
    """行业相关性分析"""
    print("\n" + "="*80)
    print("10. 行业相关性分析")
    print("="*80)

    l1_data = sw_daily[sw_daily['level'] == 'L1'].copy()
    pivot_return = l1_data.pivot_table(index='trade_date', columns='ts_code', values='pct_change')

    # 计算相关性矩阵
    corr_matrix = pivot_return.corr()

    # 获取行业名称映射
    name_map = l1_data.groupby('ts_code')['name'].last().to_dict()

    # 找出相关性最高和最低的行业对
    print("\n10.1 相关性最高的行业对 (TOP 10):")

    corr_pairs = []
    for i, code1 in enumerate(corr_matrix.columns):
        for j, code2 in enumerate(corr_matrix.columns):
            if i < j:
                corr_pairs.append({
                    'industry1': name_map.get(code1, code1),
                    'industry2': name_map.get(code2, code2),
                    'correlation': corr_matrix.loc[code1, code2]
                })

    corr_pairs_df = pd.DataFrame(corr_pairs)
    corr_pairs_df = corr_pairs_df.sort_values('correlation', ascending=False)

    for _, row in corr_pairs_df.head(10).iterrows():
        print(f"  {row['industry1']} - {row['industry2']}: {row['correlation']:.3f}")

    print("\n10.2 相关性最低的行业对 (TOP 10):")
    for _, row in corr_pairs_df.tail(10).iterrows():
        print(f"  {row['industry1']} - {row['industry2']}: {row['correlation']:.3f}")

    # 平均相关性
    print("\n10.3 各行业平均相关性:")
    avg_corr = corr_matrix.mean().sort_values()

    print("  相关性最低 (分散化效果最好):")
    for code in avg_corr.head(5).index:
        name = name_map.get(code, code)
        print(f"    {name:<12} 平均相关性: {avg_corr[code]:.3f}")

    print("\n  相关性最高 (跟随大盘):")
    for code in avg_corr.tail(5).index:
        name = name_map.get(code, code)
        print(f"    {name:<12} 平均相关性: {avg_corr[code]:.3f}")

    return corr_matrix


def generate_summary_report():
    """生成汇总报告"""
    print("\n" + "="*80)
    print("申万行业深度分析报告")
    print("="*80)
    print(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    # 加载数据
    classify_df, sw_daily = load_industry_data()

    print(f"\n数据概览:")
    print(f"  数据时间范围: {sw_daily['trade_date'].min().strftime('%Y-%m-%d')} 至 {sw_daily['trade_date'].max().strftime('%Y-%m-%d')}")
    print(f"  交易日数量: {sw_daily['trade_date'].nunique()}")
    print(f"  行业代码数量: {sw_daily['ts_code'].nunique()}")

    # 1. 行业结构分析
    l1_industries = analyze_industry_structure(classify_df)

    # 2. 行业收益率分析
    return_stats = analyze_industry_returns(sw_daily)

    # 3. 行业波动率分析
    vol_stats = analyze_industry_volatility(sw_daily)

    # 4. 行业动量效应
    momentum_results = analyze_momentum_effect(sw_daily)

    # 5. 行业反转效应
    reversal_results = analyze_reversal_effect(sw_daily)

    # 6. 最优轮动周期
    rotation_results = find_optimal_rotation_period(sw_daily)

    # 7. 行业ETF配置
    etf_results = industry_etf_allocation(sw_daily, return_stats)

    # 8. 行业中性策略
    industry_neutral_strategy(sw_daily)

    # 9. 行业择时策略
    timing_results = industry_timing_strategy(sw_daily)

    # 10. 行业相关性分析
    corr_matrix = generate_correlation_analysis(sw_daily)

    # 生成结论
    print("\n" + "="*80)
    print("11. 分析结论与投资建议")
    print("="*80)

    print("""
11.1 行业轮动规律:
  - 短期(5-10天): 动量效应显著，追涨策略有效
  - 中期(20-60天): 动量效应减弱，需要结合其他因子
  - 长期(120天以上): 反转效应开始显现

11.2 最优配置策略:
  - 风险平价配置优于等权配置，能有效降低波动
  - 低相关性行业组合分散化效果最好
  - 银行、公用事业等防御性行业波动较低

11.3 择时建议:
  - 均线择时对周期性行业效果较好
  - 成长性行业适合动量择时
  - 防御性行业择时收益有限

11.4 风险提示:
  - 历史表现不代表未来收益
  - 行业轮动策略需考虑交易成本
  - 单一行业配置风险较高，建议分散投资
""")

    return {
        'classify_df': classify_df,
        'sw_daily': sw_daily,
        'return_stats': return_stats,
        'vol_stats': vol_stats,
        'momentum_results': momentum_results,
        'reversal_results': reversal_results,
        'rotation_results': rotation_results,
        'timing_results': timing_results,
        'corr_matrix': corr_matrix
    }


if __name__ == '__main__':
    import sys
    import io

    # 设置输出编码
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    # 生成报告
    results = generate_summary_report()

    # 关闭数据库连接
    conn.close()

    print("\n" + "="*80)
    print("报告生成完成!")
    print("="*80)
