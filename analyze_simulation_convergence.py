#!/usr/bin/env python
"""
模拟规模收敛性分析

测试不同模拟规模下的结果稳定性，帮助确定最优参数
"""
import sys
sys.path.insert(0, '/Users/allen/workspace/python/stock/Tushare-DuckDB/.worktrees/monte-carlo-factor-filter')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from src.tushare_db.factor_validation import FactorRegistry
from src.tushare_db.factor_validation.simulator import GBMSimulator
from src.tushare_db.factor_validation.detector import SignalDetector


def test_convergence(factor_name, n_paths_list=[100, 500, 1000, 2000, 5000, 10000], n_steps=252):
    """测试不同路径数下的收敛性"""
    print(f"\n测试因子: {factor_name}")
    print("=" * 80)

    factor = FactorRegistry.get(factor_name)

    # 固定参数
    np.random.seed(42)
    s0 = 100
    mu = 0.1
    sigma = 0.2

    results = []

    for n_paths in n_paths_list:
        print(f"  测试 {n_paths:6,} 路径...", end=" ")

        # GBM模拟
        simulator = GBMSimulator(
            n_paths=n_paths,
            n_steps=n_steps,
            random_seed=42
        )
        price_matrix = simulator.simulate(s0=s0, mu=mu, sigma=sigma)

        # 计算信号
        detector = SignalDetector()
        signals = detector.detect_signals(price_matrix, factor)

        # 计算概率
        p_random = signals.sum() / (n_paths * n_steps)

        # 计算95%置信区间
        n = n_paths * n_steps
        ci_half_width = 1.96 * np.sqrt(p_random * (1 - p_random) / n)
        relative_error = ci_half_width / p_random if p_random > 0 else 0

        results.append({
            'n_paths': n_paths,
            'total_samples': n,
            'p_random': p_random,
            'ci_half_width': ci_half_width,
            'relative_error': relative_error,
            'signals_count': signals.sum()
        })

        print(f"P={p_random:.4%}, 相对误差={relative_error:.2%}")

    return pd.DataFrame(results)


def plot_convergence(results_dict):
    """绘制收敛性图表"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 图1: P_random 收敛
    ax1 = axes[0, 0]
    for factor_name, df in results_dict.items():
        ax1.plot(df['n_paths'], df['p_random'], marker='o', label=factor_name, linewidth=2)
    ax1.set_xscale('log')
    ax1.set_xlabel('Number of Paths (log scale)')
    ax1.set_ylabel('P_random')
    ax1.set_title('P_random Convergence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 图2: 相对误差
    ax2 = axes[0, 1]
    for factor_name, df in results_dict.items():
        ax2.plot(df['n_paths'], df['relative_error'] * 100, marker='s', label=factor_name, linewidth=2)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Number of Paths (log scale)')
    ax2.set_ylabel('Relative Error (%)')
    ax2.set_title('Relative Error vs Sample Size')
    ax2.axhline(y=5, color='r', linestyle='--', label='5% target')
    ax2.axhline(y=2, color='g', linestyle='--', label='2% target')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 图3: 置信区间半宽
    ax3 = axes[1, 0]
    for factor_name, df in results_dict.items():
        ax3.plot(df['n_paths'], df['ci_half_width'] * 100, marker='^', label=factor_name, linewidth=2)
    ax3.set_xscale('log')
    ax3.set_xlabel('Number of Paths (log scale)')
    ax3.set_ylabel('95% CI Half-Width (%)')
    ax3.set_title('Confidence Interval Width')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 图4: 理论 vs 实际误差
    ax4 = axes[1, 1]
    n_paths_ref = 10000
    for factor_name, df in results_dict.items():
        # 理论误差: 1/sqrt(n)
        theoretical = 1 / np.sqrt(df['total_samples'])
        ax4.scatter(df['relative_error'] * 100, theoretical * 100,
                   s=df['n_paths']/50, alpha=0.6, label=factor_name)

    ax4.plot([0, 10], [0, 10], 'k--', label='Perfect match')
    ax4.set_xlabel('Actual Relative Error (%)')
    ax4.set_ylabel('Theoretical 1/sqrt(n) (%)')
    ax4.set_title('Actual vs Theoretical Error')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('simulation_convergence_analysis.png', dpi=150, bbox_inches='tight')
    print("\n✓ 图表已保存: simulation_convergence_analysis.png")


def print_recommendations(results_dict):
    """打印推荐配置"""
    print("\n" + "=" * 80)
    print("推荐配置")
    print("=" * 80)

    # 找出使相对误差 < 5% 的最小规模
    print("\n使相对误差 < 5% 的最小规模:")
    for factor_name, df in results_dict.items():
        df_valid = df[df['relative_error'] < 0.05]
        if not df_valid.empty:
            min_paths = df_valid['n_paths'].min()
            print(f"  {factor_name:25s}: {min_paths:6,} 路径")
        else:
            print(f"  {factor_name:25s}: 即使 10,000 路径也无法达到 (P_random 太小)")

    # 找出使相对误差 < 2% 的最小规模
    print("\n使相对误差 < 2% 的最小规模:")
    for factor_name, df in results_dict.items():
        df_valid = df[df['relative_error'] < 0.02]
        if not df_valid.empty:
            min_paths = df_valid['n_paths'].min()
            print(f"  {factor_name:25s}: {min_paths:6,} 路径")
        else:
            print(f"  {factor_name:25s}: 即使 10,000 路径也无法达到")


def main():
    """主函数"""
    print("=" * 80)
    print("模拟规模收敛性分析")
    print("=" * 80)
    print(f"\n开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 选择代表性因子（不同触发频率）
    test_factors = [
        'macd_golden_cross',  # ~4% 触发
        'rsi_oversold',       # ~3% 触发
        'doji',               # ~10% 触发
        'close_gt_open',      # ~50% 触发
    ]

    n_paths_list = [100, 500, 1000, 2000, 5000, 10000]

    print(f"\n测试规模: {n_paths_list}")
    print(f"测试因子: {', '.join(test_factors)}")

    # 运行测试
    results_dict = {}
    for factor_name in test_factors:
        df = test_convergence(factor_name, n_paths_list)
        results_dict[factor_name] = df

    # 打印详细结果
    print("\n" + "=" * 80)
    print("详细结果")
    print("=" * 80)

    for factor_name, df in results_dict.items():
        print(f"\n{factor_name}:")
        print(df[['n_paths', 'p_random', 'relative_error']].to_string(index=False))

    # 打印推荐
    print_recommendations(results_dict)

    # 生成图表
    print("\n生成收敛性图表...")
    try:
        plot_convergence(results_dict)
    except Exception as e:
        print(f"  图表生成失败: {e}")
        print("  请确保已安装 matplotlib: pip install matplotlib")

    # 保存数据
    output_file = 'simulation_convergence_data.csv'
    all_results = []
    for factor_name, df in results_dict.items():
        df['factor_name'] = factor_name
        all_results.append(df)

    pd.concat(all_results).to_csv(output_file, index=False)
    print(f"\n✓ 数据已保存: {output_file}")

    print("\n" + "=" * 80)
    print("分析完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
