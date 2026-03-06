#!/usr/bin/env python
"""
批量测试所有内置因子的效果
"""
import sys
sys.path.insert(0, '/Users/allen/workspace/python/stock/Tushare-DuckDB/.worktrees/monte-carlo-factor-filter')

from src.tushare_db.factor_validation import FactorFilter, FactorRegistry
import pandas as pd


def test_all_factors():
    """测试所有内置因子"""
    print("=" * 80)
    print("蒙特卡洛因子质检系统 - 批量因子测试")
    print("=" * 80)

    # 初始化过滤器
    filter_obj = FactorFilter(
        db_path=":memory:",
        n_simulations=2000,  # 减少模拟次数以加快测试
        simulation_days=100,
        alpha_threshold=1.5
    )

    # 获取所有内置因子
    factor_names = FactorRegistry.list_builtin()

    print(f"\n测试因子数: {len(factor_names)}")
    print(f"模拟路径: {filter_obj.n_simulations}")
    print(f"模拟天数: {filter_obj.simulation_days}")
    print(f"Alpha阈值: {filter_obj.alpha_threshold}")
    print()

    # 存储结果
    results = []

    # 批量测试
    for i, factor_name in enumerate(factor_names, 1):
        print(f"[{i}/{len(factor_names)}] 测试因子: {factor_name}...", end=" ")

        try:
            report = filter_obj.filter(
                factor=factor_name,
                ts_codes=[],  # 空列表，使用模拟数据
                use_sample=True  # 使用模拟数据
            )

            summary = report.summary
            results.append({
                'factor_name': factor_name,
                'alpha_ratio_median': summary.get('alpha_ratio_median', 0),
                'alpha_ratio_mean': summary.get('alpha_ratio_mean', 0),
                'p_actual_median': summary.get('p_actual_median', 0),
                'p_random_median': summary.get('p_random_median', 0),
                'significant_ratio': summary.get('significant_ratio', 0),
                'recommendation': 'KEEP' if summary.get('alpha_ratio_median', 0) >= 1.5 else 'DISCARD'
            })
            print(f"✓ Alpha={summary.get('alpha_ratio_median', 0):.2f}")

        except Exception as e:
            print(f"✗ 错误: {e}")
            results.append({
                'factor_name': factor_name,
                'alpha_ratio_median': 0,
                'alpha_ratio_mean': 0,
                'p_actual_median': 0,
                'p_random_median': 0,
                'significant_ratio': 0,
                'recommendation': 'ERROR'
            })

    # 创建结果DataFrame
    df = pd.DataFrame(results)

    # 按Alpha Ratio排序
    df_sorted = df.sort_values('alpha_ratio_median', ascending=False)

    print("\n" + "=" * 80)
    print("测试结果汇总")
    print("=" * 80)

    print("\n| 排名 | 因子名称 | Alpha中位数 | P_actual | P_random | 显著比例 | 建议 |")
    print("|------|----------|-------------|----------|----------|----------|------|")

    for idx, row in df_sorted.iterrows():
        rank = df_sorted.index.get_loc(idx) + 1
        print(f"| {rank:2d} | {row['factor_name']:22s} | "
              f"{row['alpha_ratio_median']:11.2f} | "
              f"{row['p_actual_median']:8.2%} | "
              f"{row['p_random_median']:8.2%} | "
              f"{row['significant_ratio']:7.1%} | "
              f"{row['recommendation']:4s} |")

    # 统计
    keep_count = (df['recommendation'] == 'KEEP').sum()
    discard_count = (df['recommendation'] == 'DISCARD').sum()
    error_count = (df['recommendation'] == 'ERROR').sum()

    print("\n" + "=" * 80)
    print("统计摘要")
    print("=" * 80)
    print(f"总因子数: {len(df)}")
    print(f"建议保留 (Alpha >= 1.5): {keep_count} ({keep_count/len(df)*100:.1f}%)")
    print(f"建议废弃 (Alpha < 1.5): {discard_count} ({discard_count/len(df)*100:.1f}%)")
    if error_count > 0:
        print(f"测试错误: {error_count}")

    # 分类统计
    print("\n" + "=" * 80)
    print("按类别统计")
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
        cat_df = df[df['factor_name'].isin(cat_factors)]
        if len(cat_df) > 0:
            avg_alpha = cat_df['alpha_ratio_median'].mean()
            keep_pct = (cat_df['recommendation'] == 'KEEP').mean() * 100
            print(f"{cat_name:12s}: 平均Alpha={avg_alpha:.2f}, 保留率={keep_pct:.0f}%")

    # 保存结果
    output_file = 'factor_validation_results.csv'
    df_sorted.to_csv(output_file, index=False)
    print(f"\n结果已保存到: {output_file}")

    return df_sorted


if __name__ == "__main__":
    results = test_all_factors()
