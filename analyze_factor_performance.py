#!/usr/bin/env python
"""分析所有因子的模拟计算表现"""
import sys
sys.path.insert(0, '/Users/allen/workspace/python/stock/Tushare-DuckDB/.worktrees/monte-carlo-factor-filter')

import pandas as pd
import numpy as np
from src.tushare_db.factor_validation import FactorRegistry
from src.tushare_db.factor_validation.simulator import GBMSimulator
from src.tushare_db.factor_validation.detector import SignalDetector


def analyze_all_factors():
    """分析所有因子的模拟表现"""
    print("=" * 80)
    print("因子模拟计算表现分析")
    print("=" * 80)

    # 创建模拟数据
    np.random.seed(42)
    simulator = GBMSimulator(n_paths=5000, n_steps=252, random_seed=42)
    price_matrix = simulator.simulate(s0=100, mu=0.1, sigma=0.2)

    # 创建 OHLC 数据（模拟）
    n_paths, n_steps = price_matrix.shape

    # 模拟 high/low/open 基于 close
    high_matrix = price_matrix * (1 + np.abs(np.random.randn(n_paths, n_steps) * 0.01))
    low_matrix = price_matrix * (1 - np.abs(np.random.randn(n_paths, n_steps) * 0.01))
    open_matrix = np.roll(price_matrix, 1, axis=1)
    open_matrix[:, 0] = price_matrix[:, 0] * 0.99
    vol_matrix = np.random.randint(1000000, 10000000, size=(n_paths, n_steps))

    print(f"\n模拟参数:")
    print(f"  路径数: {n_paths:,}")
    print(f"  天数: {n_steps}")
    print(f"  总样本数: {n_paths * n_steps:,}")

    detector = SignalDetector()
    factor_names = FactorRegistry.list_builtin()

    results = []

    print("\n" + "=" * 80)
    print("各因子触发概率统计")
    print("=" * 80)

    print(f"\n{'因子名称':<30} {'触发次数':>12} {'触发概率':>12} {'信号/年':>10}")
    print("-" * 80)

    for factor_name in sorted(factor_names):
        try:
            factor = FactorRegistry.get(factor_name)

            # 根据因子类型准备数据
            if factor_name in ['macd_golden_cross', 'macd_death_cross', 'macd_zero_golden_cross',
                             'rsi_oversold', 'rsi_overbought', 'golden_cross', 'death_cross',
                             'price_above_sma', 'price_below_sma', 'bollinger_lower_break',
                             'bollinger_upper_break']:
                # 这些因子可以使用向量化计算
                signals = detector.detect_signals(price_matrix, factor)
            else:
                # 其他因子需要 DataFrame 格式
                # 使用第一条路径作为示例
                df = pd.DataFrame({
                    'open': open_matrix[0],
                    'high': high_matrix[0],
                    'low': low_matrix[0],
                    'close': price_matrix[0],
                    'vol': vol_matrix[0]
                })
                signals = factor.evaluate(df)
                # 扩展到所有路径（近似）
                total_signals = signals.sum() * n_paths
                p_random = total_signals / (n_paths * n_steps)
                results.append({
                    'factor_name': factor_name,
                    'total_signals': total_signals,
                    'p_random': p_random,
                    'signals_per_year': p_random * 252
                })
                print(f"{factor_name:<30} {total_signals:>12,} {p_random:>11.2%} {p_random*252:>10.1f}")
                continue

            total_signals = signals.sum()
            p_random = total_signals / (n_paths * n_steps)

            results.append({
                'factor_name': factor_name,
                'total_signals': total_signals,
                'p_random': p_random,
                'signals_per_year': p_random * 252
            })

            print(f"{factor_name:<30} {total_signals:>12,} {p_random:>11.2%} {p_random*252:>10.1f}")

        except Exception as e:
            print(f"{factor_name:<30} {'错误':>12} {str(e)[:30]:>30}")
            results.append({
                'factor_name': factor_name,
                'total_signals': 0,
                'p_random': 0,
                'signals_per_year': 0,
                'error': str(e)
            })

    # 统计分析
    df_results = pd.DataFrame(results)
    df_valid = df_results[df_results['p_random'] > 0]

    print("\n" + "=" * 80)
    print("统计摘要")
    print("=" * 80)

    if len(df_valid) > 0:
        print(f"\n触发概率分布:")
        print(f"  平均触发概率: {df_valid['p_random'].mean():.2%}")
        print(f"  中位数触发概率: {df_valid['p_random'].median():.2%}")
        print(f"  最高触发概率: {df_valid['p_random'].max():.2%} ({df_valid.loc[df_valid['p_random'].idxmax(), 'factor_name']})")
        print(f"  最低触发概率: {df_valid['p_random'].min():.2%} ({df_valid.loc[df_valid['p_random'].idxmin(), 'factor_name']})")

        print(f"\n信号频率（每年）:")
        print(f"  平均: {df_valid['signals_per_year'].mean():.1f} 次/年")
        print(f"  最高: {df_valid['signals_per_year'].max():.1f} 次/年 ({df_valid.loc[df_valid['signals_per_year'].idxmax(), 'factor_name']})")
        print(f"  最低: {df_valid['signals_per_year'].min():.1f} 次/年 ({df_valid.loc[df_valid['signals_per_year'].idxmin(), 'factor_name']})")

        # 分类统计
        print("\n" + "=" * 80)
        print("按类别统计平均触发概率")
        print("=" * 80)

        categories = {
            'MACD': ['macd_golden_cross', 'macd_death_cross', 'macd_zero_golden_cross'],
            'RSI': ['rsi_oversold', 'rsi_overbought'],
            'KDJ': ['kdj_golden_cross', 'kdj_death_cross'],
            'Williams %R': ['williams_r_oversold', 'williams_r_overbought'],
            'CCI': ['cci_oversold', 'cci_overbought'],
            '均线': ['golden_cross', 'death_cross', 'price_above_sma', 'price_below_sma'],
            '布林带': ['bollinger_lower_break', 'bollinger_upper_break'],
            'ATR': ['atr_breakout', 'atr_breakdown'],
            '成交量': ['volume_breakout'],
            '蜡烛图': ['bullish_engulfing', 'bearish_engulfing', 'hammer', 'shooting_star',
                      'doji', 'three_white_soldiers', 'three_black_crows'],
            '价格形态': ['close_gt_open', 'gap_up', 'gap_down']
        }

        for cat_name, cat_factors in categories.items():
            cat_df = df_results[df_results['factor_name'].isin(cat_factors)]
            if len(cat_df) > 0:
                avg_p = cat_df['p_random'].mean()
                avg_signals = cat_df['signals_per_year'].mean()
                print(f"{cat_name:12s}: 平均触发概率={avg_p:6.2%}, 平均{avg_signals:5.1f}次/年")

        # 按触发概率排序
        print("\n" + "=" * 80)
        print("触发概率排名（Top 10）")
        print("=" * 80)
        df_sorted = df_valid.sort_values('p_random', ascending=False)
        for idx, row in df_sorted.head(10).iterrows():
            print(f"{row['factor_name']:30s}: {row['p_random']:6.2%} ({row['signals_per_year']:5.1f}次/年)")

        print("\n" + "=" * 80)
        print("触发概率排名（Bottom 10）")
        print("=" * 80)
        for idx, row in df_sorted.tail(10).iterrows():
            print(f"{row['factor_name']:30s}: {row['p_random']:6.2%} ({row['signals_per_year']:5.1f}次/年)")

    # 保存结果
    output_file = 'factor_simulation_analysis.csv'
    df_results.to_csv(output_file, index=False)
    print(f"\n详细结果已保存到: {output_file}")

    return df_results


if __name__ == "__main__":
    analyze_all_factors()
