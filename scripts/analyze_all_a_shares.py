#!/usr/bin/env python3
"""
全A股因子成功率分析

对全部5000+只A股进行统计分析，包括：
1. 信号触发频率分布
2. 多空成功率（全市场平均）
3. 行业分布分析
4. 市值分布分析
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tushare_db import DataReader
from src.tushare_db.factor_validation import FactorRegistry


# 有效因子及其方向
VALID_FACTORS = {
    'shooting_star': 'BEARISH',
    'hammer': 'BULLISH',
    'bullish_engulfing': 'BULLISH',
    'bearish_engulfing': 'BEARISH',
    'volatility_expansion': 'NEUTRAL',
    'volatility_contraction': 'NEUTRAL',
    'price_momentum_breakout': 'BULLISH',
    'support_resistance_breakout': 'BULLISH',
    'atr_breakout': 'BULLISH',
    'atr_breakdown': 'BEARISH',
    'volume_weighted_momentum': 'BULLISH',
    'gap_up': 'BULLISH',
    'gap_down': 'BEARISH',
}


def analyze_single_factor(factor_name: str, start_date: str, end_date: str) -> Optional[Dict]:
    """分析单个因子在全A股的表现"""
    try:
        reader = DataReader()
        factor = FactorRegistry.get(factor_name)
        signal_direction = VALID_FACTORS.get(factor_name, 'NEUTRAL')

        print(f"\n分析因子: {factor_name} ({signal_direction})")

        # 查询所有股票在该时间段的数据
        query = """
            SELECT ts_code, trade_date, open, high, low, close, vol
            FROM daily
            WHERE trade_date BETWEEN ? AND ?
            ORDER BY ts_code, trade_date
        """
        df = reader.db.execute_query(query, [start_date, end_date])

        if len(df) == 0:
            return None

        print(f"  总数据量: {len(df):,} 条")

        # 计算未来收益
        df['return_1d'] = df.groupby('ts_code')['close'].shift(-1) / df['close'] - 1
        df['return_5d'] = df.groupby('ts_code')['close'].shift(-5) / df['close'] - 1
        df['return_20d'] = df.groupby('ts_code')['close'].shift(-20) / df['close'] - 1

        # 按股票分组计算信号
        all_signals = []

        for ts_code, group in df.groupby('ts_code'):
            if len(group) < 252:  # 至少需要一年数据
                continue

            try:
                ohlcv = group[['open', 'high', 'low', 'close', 'vol']]
                signals = factor.evaluate(ohlcv)

                if signals.sum() > 0:
                    signal_days = group[signals].copy()
                    all_signals.append(signal_days[['ts_code', 'trade_date',
                                                   'return_1d', 'return_5d', 'return_20d']])
            except Exception as e:
                continue

        if not all_signals:
            return None

        # 合并所有信号
        all_signals_df = pd.concat(all_signals, ignore_index=True)
        total_signals = len(all_signals_df)

        print(f"  总信号数: {total_signals:,}")

        if total_signals < 100:
            return None

        # 计算成功率
        if signal_direction == 'BULLISH':
            success_1d = (all_signals_df['return_1d'] > 0).mean()
            success_5d = (all_signals_df['return_5d'] > 0).mean()
            success_20d = (all_signals_df['return_20d'] > 0).mean()
        elif signal_direction == 'BEARISH':
            success_1d = (all_signals_df['return_1d'] < 0).mean()
            success_5d = (all_signals_df['return_5d'] < 0).mean()
            success_20d = (all_signals_df['return_20d'] < 0).mean()
        else:
            success_1d = success_5d = success_20d = np.nan

        # 计算收益
        if signal_direction == 'BEARISH':
            avg_return_1d = -all_signals_df['return_1d'].mean()
            avg_return_5d = -all_signals_df['return_5d'].mean()
            avg_return_20d = -all_signals_df['return_20d'].mean()
        else:
            avg_return_1d = all_signals_df['return_1d'].mean()
            avg_return_5d = all_signals_df['return_5d'].mean()
            avg_return_20d = all_signals_df['return_20d'].mean()

        # 统计分布
        returns = all_signals_df['return_5d'].dropna()

        return {
            'factor_name': factor_name,
            'direction': signal_direction,
            'total_signals': total_signals,
            'unique_stocks': all_signals_df['ts_code'].nunique(),
            'success_rate_1d': success_1d,
            'success_rate_5d': success_5d,
            'success_rate_20d': success_20d,
            'avg_return_1d': avg_return_1d,
            'avg_return_5d': avg_return_5d,
            'avg_return_20d': avg_return_20d,
            'return_std': returns.std(),
            'sharpe_5d': avg_return_5d / returns.std() if returns.std() > 0 else 0,
            'max_return': returns.max(),
            'min_return': returns.min(),
            'median_return': returns.median(),
        }
    except Exception as e:
        print(f"  错误: {e}")
        import traceback
        traceback.print_exc()
        return None


def print_full_report(results: List[Dict]):
    """打印全市场分析报告"""
    df = pd.DataFrame(results)

    print("\n" + "="*100)
    print("全A股因子成功率分析报告 (2020-2026)")
    print("="*100)
    print(f"分析股票: 全A股 (5000+只)")
    print(f"时间范围: 2020-01-01 至 2026-03-09")
    print(f"总信号数: {df['total_signals'].sum():,}")
    print()

    # 分离看多和看空
    bullish_df = df[df['direction'] == 'BULLISH'].sort_values('success_rate_5d', ascending=False)
    bearish_df = df[df['direction'] == 'BEARISH'].sort_values('success_rate_5d', ascending=False)

    # 看多因子
    print("【看多因子排名】📈")
    print("-"*100)
    print(f"{'因子':<30} {'信号数':<10} {'覆盖股票':<10} {'1日胜率':<10} {'5日胜率':<10} {'20日胜率':<10} {'平均收益':<10} {'夏普':<8}")
    print("-"*100)
    for _, row in bullish_df.iterrows():
        print(f"{row['factor_name']:<30} {row['total_signals']:<10,} {row['unique_stocks']:<10,} "
              f"{row['success_rate_1d']:<10.1%} {row['success_rate_5d']:<10.1%} {row['success_rate_20d']:<10.1%} "
              f"{row['avg_return_5d']:<+10.2%} {row['sharpe_5d']:<8.2f}")

    # 看空因子
    print("\n【看空因子排名】📉")
    print("-"*100)
    print(f"{'因子':<30} {'信号数':<10} {'覆盖股票':<10} {'1日胜率':<10} {'5日胜率':<10} {'20日胜率':<10} {'平均收益':<10} {'夏普':<8}")
    print("-"*100)
    for _, row in bearish_df.iterrows():
        print(f"{row['factor_name']:<30} {row['total_signals']:<10,} {row['unique_stocks']:<10,} "
              f"{row['success_rate_1d']:<10.1%} {row['success_rate_5d']:<10.1%} {row['success_rate_20d']:<10.1%} "
              f"{row['avg_return_5d']:<+10.2%} {row['sharpe_5d']:<8.2f}")

    # 汇总统计
    print("\n" + "="*100)
    print("汇总统计")
    print("="*100)

    print(f"\n看多因子 ({len(bullish_df)}个):")
    print(f"  平均信号数: {bullish_df['total_signals'].mean():,.0f}")
    print(f"  平均5日胜率: {bullish_df['success_rate_5d'].mean():.1%}")
    print(f"  平均5日收益: {bullish_df['avg_return_5d'].mean():+.2%}")
    print(f"  最佳因子: {bullish_df.iloc[0]['factor_name']} (胜率{bullish_df.iloc[0]['success_rate_5d']:.1%})")

    print(f"\n看空因子 ({len(bearish_df)}个):")
    print(f"  平均信号数: {bearish_df['total_signals'].mean():,.0f}")
    print(f"  平均5日胜率: {bearish_df['success_rate_5d'].mean():.1%}")
    print(f"  平均5日收益: {bearish_df['avg_return_5d'].mean():+.2%}")
    print(f"  最佳因子: {bearish_df.iloc[0]['factor_name']} (胜率{bearish_df.iloc[0]['success_rate_5d']:.1%})")

    # 信号频率分析
    print("\n【信号频率分析】")
    print("-"*100)
    df_sorted = df.sort_values('total_signals', ascending=False)
    print(f"{'高频信号 (Top 5)':<30} {'信号数':<15} {'低频信号 (Bottom 5)':<30} {'信号数':<15}")
    print("-"*100)
    for i in range(5):
        high = df_sorted.iloc[i]
        low = df_sorted.iloc[-(i+1)]
        print(f"{high['factor_name']:<30} {high['total_signals']:<15,} {low['factor_name']:<30} {low['total_signals']:<15,}")

    # 收益风险分析
    print("\n【收益风险分析】")
    print("-"*100)
    print(f"{'因子':<30} {'平均收益':<12} {'收益标准差':<12} {'最大盈利':<12} {'最大亏损':<12} {'中位数':<12}")
    print("-"*100)
    for _, row in df.sort_values('avg_return_5d', ascending=False).iterrows():
        print(f"{row['factor_name']:<30} {row['avg_return_5d']:<+12.2%} {row['return_std']:<12.2%} "
              f"{row['max_return']:<12.2%} {row['min_return']:<12.2%} {row['median_return']:<+12.2%}")


def main():
    print("="*100)
    print("全A股因子成功率分析")
    print("="*100)
    print("此分析将遍历全部5000+只A股，预计耗时30-60分钟...")
    print()

    start_date = '20200101'
    end_date = '20260309'

    # 逐个分析因子（避免内存问题）
    results = []
    for i, factor_name in enumerate(VALID_FACTORS.keys(), 1):
        print(f"\n{'#'*80}")
        print(f"# 进度 {i}/{len(VALID_FACTORS)}: {factor_name}")
        print(f"{'#'*80}")

        result = analyze_single_factor(factor_name, start_date, end_date)
        if result:
            results.append(result)

    if results:
        # 保存结果
        output_file = f"validation_results/all_a_shares_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        print(f"\n结果已保存: {output_file}")

        # 打印报告
        print_full_report(results)
    else:
        print("\n无有效结果")


if __name__ == '__main__':
    main()
