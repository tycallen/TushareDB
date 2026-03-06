#!/usr/bin/env python
"""
完整流程演示：模拟数据 vs 实际数据对比

流程：
1. 获取真实股票数据
2. 计算真实因子触发概率 P_actual
3. 从真实数据估计参数 (μ, σ)
4. GBM模拟生成随机价格
5. 计算模拟因子触发概率 P_random
6. 计算 Alpha Ratio = P_actual / P_random
7. 生成对比报告
"""
import sys
sys.path.insert(0, '/Users/allen/workspace/python/stock/Tushare-DuckDB/.worktrees/monte-carlo-factor-filter')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.tushare_db import DataReader
from src.tushare_db.factor_validation import FactorRegistry
from src.tushare_db.factor_validation.simulator import GBMSimulator
from src.tushare_db.factor_validation.detector import SignalDetector
from src.tushare_db.factor_validation.tester import SignificanceTester, TestResult


def get_real_stock_data(ts_code='000001.SZ', days=252):
    """获取真实股票数据"""
    print(f"\n📊 获取真实数据: {ts_code}")

    try:
        reader = DataReader(db_path="tushare.db")

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days * 1.5)

        df = reader.get_stock_daily(
            ts_code=ts_code,
            start_date=start_date.strftime('%Y%m%d'),
            end_date=end_date.strftime('%Y%m%d'),
            adj='qfq'
        )

        # 数据清洗：剔除停牌日
        df = df[df['vol'] > 0].copy()
        df = df.sort_values('trade_date').reset_index(drop=True)

        print(f"  ✓ 获取到 {len(df)} 个交易日数据")
        print(f"  日期范围: {df['trade_date'].min()} ~ {df['trade_date'].max()}")
        print(f"  价格范围: {df['close'].min():.2f} ~ {df['close'].max():.2f}")

        return df

    except Exception as e:
        print(f"  ✗ 获取数据失败: {e}")
        print(f"  使用模拟数据代替...")
        return generate_sample_data(days)


def generate_sample_data(days=252):
    """生成模拟数据（当无法获取真实数据时）"""
    print(f"\n📊 生成模拟数据（{days}天）")

    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=days, freq='B')

    # 生成价格数据
    returns = np.random.randn(days) * 0.02
    close = 100 * np.exp(np.cumsum(returns))

    df = pd.DataFrame({
        'trade_date': dates.strftime('%Y%m%d'),
        'open': close * (1 + np.random.randn(days) * 0.01),
        'high': close * (1 + np.abs(np.random.randn(days) * 0.02)),
        'low': close * (1 - np.abs(np.random.randn(days) * 0.02)),
        'close': close,
        'vol': np.random.randint(1000000, 10000000, days)
    })

    print(f"  ✓ 生成 {len(df)} 条模拟数据")
    return df


def calculate_p_actual(df, factor):
    """在真实数据上计算因子触发概率"""
    signals = factor.evaluate(df)
    p_actual = signals.sum() / len(df)
    n_signals = signals.sum()
    return p_actual, n_signals


def estimate_parameters(df):
    """从真实数据估计GBM参数"""
    log_returns = np.log(df['close'] / df['close'].shift(1)).dropna()

    daily_mu = log_returns.mean()
    daily_sigma = log_returns.std()

    # 年化
    annual_mu = daily_mu * 252
    annual_sigma = daily_sigma * np.sqrt(252)

    s0 = df['close'].iloc[-1]  # 最新价格

    return s0, annual_mu, annual_sigma


def simulate_and_calculate_p_random(s0, mu, sigma, factor, n_paths=5000, n_steps=252):
    """GBM模拟并计算P_random"""
    # GBM模拟
    simulator = GBMSimulator(n_paths=n_paths, n_steps=n_steps)
    price_matrix = simulator.simulate(s0=s0, mu=mu, sigma=sigma)

    # 准备OHLC数据
    high_matrix = price_matrix * (1 + np.abs(np.random.randn(n_paths, n_steps) * 0.01))
    low_matrix = price_matrix * (1 - np.abs(np.random.randn(n_paths, n_steps) * 0.01))
    open_matrix = np.roll(price_matrix, 1, axis=1)
    open_matrix[:, 0] = s0

    # 计算因子信号
    detector = SignalDetector()

    # 对于可以向量化的因子
    if factor.name in ['macd_golden_cross', 'macd_death_cross', 'macd_zero_golden_cross',
                       'golden_cross', 'death_cross', 'price_above_sma', 'price_below_sma',
                       'bollinger_lower_break', 'bollinger_upper_break']:
        signals = detector.detect_signals(price_matrix, factor)
        total_signals = signals.sum()
    else:
        # 对于其他因子，使用第一条路径作为代表
        df_sim = pd.DataFrame({
            'open': open_matrix[0],
            'high': high_matrix[0],
            'low': low_matrix[0],
            'close': price_matrix[0],
            'vol': np.random.randint(1000000, 10000000, n_steps)
        })
        signals = factor.evaluate(df_sim)
        # 估算所有路径的信号数
        p_single = signals.sum() / n_steps
        total_signals = int(p_single * n_paths * n_steps)

    p_random = total_signals / (n_paths * n_steps)
    return p_random, total_signals


def analyze_factor(factor_name, df_real, use_simulation=True):
    """分析单个因子"""
    factor = FactorRegistry.get(factor_name)

    # 1. 计算 P_actual
    p_actual, n_actual = calculate_p_actual(df_real, factor)

    # 2. 估计参数
    s0, mu, sigma = estimate_parameters(df_real)

    # 3. 模拟计算 P_random
    if use_simulation:
        p_random, n_random = simulate_and_calculate_p_random(
            s0=s0, mu=mu, sigma=sigma, factor=factor
        )
    else:
        # 使用预计算的基准值
        p_random = get_baseline_p_random(factor_name)
        n_random = int(p_random * 5000 * 252)

    # 4. 计算 Alpha Ratio
    alpha_ratio = p_actual / p_random if p_random > 0 else float('inf')

    # 5. 统计检验
    tester = SignificanceTester(alpha_threshold=1.5)
    result = tester.test(
        p_actual=p_actual,
        p_random=p_random,
        n_total=len(df_real),
        ts_code=df_real.get('ts_code', ['UNKNOWN'])[0] if 'ts_code' in df_real.columns else 'UNKNOWN'
    )

    return {
        'factor_name': factor_name,
        'p_actual': p_actual,
        'p_random': p_random,
        'alpha_ratio': alpha_ratio,
        'n_actual': n_actual,
        'n_random': n_random,
        'is_significant': result.is_significant,
        'recommendation': result.recommendation,
        'mu': mu,
        'sigma': sigma
    }


def get_baseline_p_random(factor_name):
    """获取预计算的基准P_random"""
    baseline = {
        'macd_golden_cross': 0.0391,
        'macd_death_cross': 0.0391,
        'rsi_oversold': 0.0278,
        'rsi_overbought': 0.0325,
        'kdj_golden_cross': 0.0040,
        'kdj_death_cross': 0.0079,
        'williams_r_oversold': 0.0794,
        'williams_r_overbought': 0.0794,
        'cci_oversold': 0.0833,
        'cci_overbought': 0.0556,
        'golden_cross': 0.0316,
        'death_cross': 0.0316,
        'bollinger_lower_break': 0.0246,
        'bollinger_upper_break': 0.0316,
        'close_gt_open': 0.5198,
        'doji': 0.0952,
        'volume_breakout': 0.0,
        'bullish_engulfing': 0.0,
        'bearish_engulfing': 0.0,
    }
    return baseline.get(factor_name, 0.05)


def main():
    print("=" * 80)
    print("蒙特卡洛因子质检 - 完整对比流程演示")
    print("=" * 80)
    print("\n流程: 真实数据 → P_actual → GBM模拟 → P_random → Alpha Ratio")

    # 获取真实数据
    df_real = get_real_stock_data(ts_code='000001.SZ', days=252)

    # 选择要测试的因子
    test_factors = [
        'macd_golden_cross',
        'macd_death_cross',
        'rsi_oversold',
        'rsi_overbought',
        'kdj_golden_cross',
        'kdj_death_cross',
        'williams_r_oversold',
        'williams_r_overbought',
        'cci_oversold',
        'cci_overbought',
        'golden_cross',
        'death_cross',
        'bollinger_lower_break',
        'bollinger_upper_break',
        'close_gt_open',
        'doji',
    ]

    print("\n" + "=" * 80)
    print("因子对比分析")
    print("=" * 80)
    print(f"\n{'因子名称':<25} {'P_actual':>10} {'P_random':>10} {'Alpha':>8} {'建议':>8}")
    print("-" * 80)

    results = []
    for factor_name in test_factors:
        try:
            result = analyze_factor(factor_name, df_real, use_simulation=False)
            results.append(result)

            print(f"{result['factor_name']:<25} "
                  f"{result['p_actual']:>9.2%} "
                  f"{result['p_random']:>9.2%} "
                  f"{result['alpha_ratio']:>7.2f} "
                  f"{result['recommendation']:>8}")

        except Exception as e:
            print(f"{factor_name:<25} {'错误':>10} {str(e)[:30]:>30}")

    # 结果分析
    df_results = pd.DataFrame(results)

    print("\n" + "=" * 80)
    print("分析摘要")
    print("=" * 80)

    # 统计
    keep_factors = df_results[df_results['recommendation'] == 'KEEP']
    discard_factors = df_results[df_results['recommendation'] == 'DISCARD']
    optimize_factors = df_results[df_results['recommendation'] == 'OPTIMIZE']

    print(f"\n总测试因子: {len(df_results)}")
    print(f"建议保留 (Alpha >= 1.5): {len(keep_factors)} ({len(keep_factors)/len(df_results)*100:.1f}%)")
    print(f"建议优化 (1.2 <= Alpha < 1.5): {len(optimize_factors)} ({len(optimize_factors)/len(df_results)*100:.1f}%)")
    print(f"建议废弃 (Alpha < 1.2): {len(discard_factors)} ({len(discard_factors)/len(df_results)*100:.1f}%)")

    # 排名
    print("\n" + "=" * 80)
    print("Alpha Ratio 排名 (Top 10)")
    print("=" * 80)

    df_sorted = df_results.sort_values('alpha_ratio', ascending=False)
    for idx, row in df_sorted.head(10).iterrows():
        status = "✓" if row['recommendation'] == 'KEEP' else "✗"
        print(f"{status} {row['factor_name']:<25}: "
              f"Alpha={row['alpha_ratio']:.2f}, "
              f"P_actual={row['p_actual']:.2%}, "
              f"P_random={row['p_random']:.2%}")

    # 保存结果
    output_file = 'factor_comparison_results.csv'
    df_results.to_csv(output_file, index=False)
    print(f"\n详细结果已保存: {output_file}")

    # 结论
    print("\n" + "=" * 80)
    print("结论")
    print("=" * 80)

    if len(keep_factors) > 0:
        print(f"\n✅ 以下 {len(keep_factors)} 个因子在真实市场中表现优于随机游走，建议保留:")
        for _, row in keep_factors.iterrows():
            print(f"   - {row['factor_name']}: Alpha={row['alpha_ratio']:.2f}")
    else:
        print("\n⚠️ 没有因子达到显著的 Alpha 阈值 (>= 1.5)")

    if len(optimize_factors) > 0:
        print(f"\n🔧 以下 {len(optimize_factors)} 个因子有一定信号，建议优化后使用:")
        for _, row in optimize_factors.iterrows():
            print(f"   - {row['factor_name']}: Alpha={row['alpha_ratio']:.2f}")

    if len(discard_factors) > 0:
        print(f"\n❌ 以下 {len(discard_factors)} 个因子与随机游走无显著差异，建议废弃:")
        for _, row in discard_factors.iterrows():
            print(f"   - {row['factor_name']}: Alpha={row['alpha_ratio']:.2f}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
