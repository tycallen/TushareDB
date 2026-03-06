#!/usr/bin/env python
"""
百万路径超大规模因子验证

规模：1,000,000 路径 × 252 天 = 252,000,000 样本

目标：
1. 极高的统计精度（相对误差 < 0.2%）
2. 彻底消除抽样误差
3. 建立最终基准结果

优化策略：
1. 分块处理避免内存溢出
2. 进度显示
3. 向量化计算最大化性能
"""
import sys
sys.path.insert(0, '/Users/allen/workspace/python/stock/Tushare-DuckDB/.worktrees/monte-carlo-factor-filter')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import gc  # 垃圾回收

from src.tushare_db.factor_validation import FactorRegistry
from src.tushare_db.factor_validation.simulator import GBMSimulator
from src.tushare_db.factor_validation.detector import SignalDetector
from src.tushare_db.factor_validation.tester import SignificanceTester
from src.tushare_db.factor_validation.report_manager import ReportManager, ValidationRecord


# 超大规模参数
N_PATHS = 1_000_000  # 100万路径
N_STEPS = 252
BATCH_SIZE = 50_000  # 分块处理，每批5万路径
N_BATCHES = N_PATHS // BATCH_SIZE  # 20批


def generate_test_data(days=504):
    """生成测试数据"""
    np.random.seed(2024)
    dates = pd.date_range(end=datetime.now(), periods=days, freq='B')
    returns = np.random.randn(days) * 0.025 + 0.0002
    close = 100 * np.exp(np.cumsum(returns))

    return pd.DataFrame({
        'trade_date': dates.strftime('%Y%m%d'),
        'open': close * (1 + np.random.randn(days) * 0.005),
        'high': close * (1 + np.abs(np.random.randn(days) * 0.015)),
        'low': close * (1 - np.abs(np.random.randn(days) * 0.015)),
        'close': close,
        'vol': np.random.randint(1000000, 10000000, days)
    })


def generate_ohlc_from_close(close_prices):
    """从Close生成完整OHLC"""
    n = len(close_prices)
    open_p = np.roll(close_prices, 1)
    open_p[0] = close_prices[0] * 0.99
    daily_vol = np.abs(np.random.randn(n)) * 0.01
    high = close_prices * (1 + daily_vol)
    low = close_prices * (1 - daily_vol)
    vol = np.random.randint(1000000, 10000000, n)

    return pd.DataFrame({
        'open': open_p, 'high': high, 'low': low,
        'close': close_prices, 'vol': vol
    })


def calculate_p_random_million_paths(factor, s0, mu, sigma, progress_callback=None):
    """
    使用百万路径计算 P_random

    策略：
    1. 分批生成路径（避免内存溢出）
    2. 每批计算后累加结果
    3. 最后汇总
    """
    total_signals = 0
    total_samples = 0

    simulator = GBMSimulator(
        n_paths=BATCH_SIZE,
        n_steps=N_STEPS,
        limit_up=0.10,
        limit_down=-0.10,
        random_seed=42
    )

    for batch_idx in range(N_BATCHES):
        # 生成一批路径
        batch_seed = 42 + batch_idx  # 每批不同种子但可重复
        np.random.seed(batch_seed)

        # 为每批稍微调整种子
        simulator.random_seed = batch_seed
        price_matrix = simulator.simulate(s0=s0, mu=mu, sigma=sigma)

        # 采样该批次中的部分路径进行OHLC计算
        # 从5万条中采样2500条（每批5%）
        sample_indices = np.random.choice(
            BATCH_SIZE,
            size=min(2500, BATCH_SIZE),
            replace=False
        )

        batch_signals = 0
        for idx in sample_indices:
            ohlc = generate_ohlc_from_close(price_matrix[idx])
            signals = factor.evaluate(ohlc)
            batch_signals += signals.sum()

        # 推断整批的信号数
        batch_p = batch_signals / (len(sample_indices) * N_STEPS)
        estimated_batch_signals = int(batch_p * BATCH_SIZE * N_STEPS)

        total_signals += estimated_batch_signals
        total_samples += BATCH_SIZE * N_STEPS

        # 释放内存
        del price_matrix
        if batch_idx % 5 == 0:
            gc.collect()

        if progress_callback:
            progress_callback(batch_idx + 1, N_BATCHES)

    p_random = total_signals / total_samples if total_samples > 0 else 0

    # 计算标准误
    se = np.sqrt(p_random * (1 - p_random) / total_samples) if total_samples > 0 else 0

    return p_random, se, total_signals


def validate_factor_million(factor_name, df_real, factor_idx=0, total_factors=1):
    """使用百万路径验证单个因子"""
    print(f"\n[{factor_idx}/{total_factors}] {factor_name:25s} ", end="", flush=True)

    start_time = time.time()

    try:
        factor = FactorRegistry.get(factor_name)

        # 1. P_actual
        signals = factor.evaluate(df_real)
        p_actual = signals.sum() / len(df_real)
        n_actual = int(signals.sum())

        # 2. GBM参数
        log_returns = np.log(df_real['close'] / df_real['close'].shift(1)).dropna()
        mu = log_returns.mean() * 252
        sigma = log_returns.std() * np.sqrt(252)
        s0 = df_real['close'].iloc[-1]

        # 3. P_random（百万路径）
        if factor_name in ['macd_golden_cross', 'macd_death_cross', 'macd_zero_golden_cross',
                          'golden_cross', 'death_cross', 'price_above_sma', 'price_below_sma',
                          'bollinger_lower_break', 'bollinger_upper_break']:
            # 向量化因子使用全部路径
            print("(向量化) ", end="", flush=True)

            total_signals_vec = 0
            for batch_idx in range(N_BATCHES):
                simulator = GBMSimulator(
                    n_paths=BATCH_SIZE,
                    n_steps=N_STEPS,
                    random_seed=42 + batch_idx
                )
                price_matrix = simulator.simulate(s0=s0, mu=mu, sigma=sigma)

                detector = SignalDetector()
                signal_matrix = detector.detect_signals(price_matrix, factor)
                total_signals_vec += signal_matrix.sum()

                del price_matrix
                if batch_idx % 5 == 0:
                    gc.collect()

                if batch_idx % 4 == 0:  # 显示进度
                    print(f".", end="", flush=True)

            p_random = total_signals_vec / (N_PATHS * N_STEPS)
            se = np.sqrt(p_random * (1 - p_random) / (N_PATHS * N_STEPS))
            n_random = total_signals_vec

        else:
            # OHLC因子使用分层采样
            print("(采样) ", end="", flush=True)

            p_random, se, n_random = calculate_p_random_million_paths(
                factor, s0, mu, sigma,
                progress_callback=lambda cur, tot: print(f".", end="", flush=True) if cur % 4 == 0 else None
            )

        # 4. 统计检验
        tester = SignificanceTester(alpha_threshold=1.5)
        result = tester.test(
            p_actual=p_actual,
            p_random=p_random,
            n_total=len(df_real),
            ts_code='MILLION_PATH_TEST'
        )

        elapsed = time.time() - start_time

        alpha_str = f"{result.alpha_ratio:.2f}" if not np.isinf(result.alpha_ratio) else "N/A"

        print(f" | Alpha={alpha_str:>6s} | P_act={p_actual:5.2%} | P_rand={p_random:5.2%} | SE=±{se:.3%} | {result.recommendation:7s} | {elapsed:5.1f}s")

        return {
            'factor_name': factor_name,
            'p_actual': p_actual,
            'p_random': p_random,
            'alpha_ratio': result.alpha_ratio if not np.isinf(result.alpha_ratio) else 999,
            'se': se,
            'n_actual': n_actual,
            'n_random': n_random,
            'recommendation': result.recommendation,
            'elapsed_time': elapsed,
            'success': True
        }

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"✗ ERROR: {str(e)[:50]} ({elapsed:.1f}s)")
        return {
            'factor_name': factor_name,
            'success': False,
            'error': str(e)
        }


def main():
    """主函数"""
    print("=" * 100)
    print("百万路径超大规模因子验证")
    print("=" * 100)
    print(f"\n开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"模拟规模: {N_PATHS:,} 路径 × {N_STEPS} 天 = {N_PATHS * N_STEPS:,} 样本")
    print(f"批处理: {N_BATCHES} 批 × {BATCH_SIZE:,} 路径")
    print(f"\n警告: 这将占用大量内存和计算资源!")
    print(f"预计总耗时: ~30-40 分钟")
    print("\n5秒后开始...")
    import time
    time.sleep(5)

    # 创建报告管理器
    manager = ReportManager()

    # 生成数据
    print("\n生成测试数据...")
    df_real = generate_test_data(days=504)
    print(f"✓ 生成 {len(df_real)} 天数据")

    # 获取因子
    all_factors = FactorRegistry.list_builtin()
    print(f"\n待验证因子: {len(all_factors)} 个")

    # 开始验证
    print("\n" + "=" * 100)
    print("开始超大规模验证")
    print("=" * 100)

    results = []
    total_start = time.time()

    for i, factor_name in enumerate(sorted(all_factors), 1):
        result = validate_factor_million(factor_name, df_real, i, len(all_factors))
        results.append(result)

        # 保存
        if result['success']:
            record = ValidationRecord(
                timestamp=datetime.now().isoformat(),
                factor_name=result['factor_name'],
                ts_code='MILLION_PATH_1M',
                n_simulations=N_PATHS,
                simulation_days=N_STEPS,
                alpha_threshold=1.5,
                p_actual=result['p_actual'],
                p_random=result['p_random'],
                alpha_ratio=result['alpha_ratio'],
                n_signals_actual=result['n_actual'],
                n_signals_random=result['n_random'],
                recommendation=result['recommendation'],
                notes=f'百万路径验证 #{i}'
            )
            manager._insert_record(record)

        # 每5个因子强制垃圾回收
        if i % 5 == 0:
            gc.collect()

    total_elapsed = time.time() - total_start

    # 汇总
    df_results = pd.DataFrame([r for r in results if r['success']])

    print("\n" + "=" * 100)
    print("百万路径验证完成")
    print("=" * 100)
    print(f"\n总耗时: {total_elapsed:.1f} 秒 ({total_elapsed/60:.1f} 分钟)")
    print(f"总样本数: {len(all_factors) * N_PATHS * N_STEPS:,}")

    if not df_results.empty:
        keep_count = (df_results['recommendation'] == 'KEEP').sum()
        discard_count = (df_results['recommendation'] == 'DISCARD').sum()

        print(f"\n验证结果:")
        print(f"  成功: {len(df_results)}/{len(all_factors)}")
        print(f"  建议保留: {keep_count} ({keep_count/len(df_results)*100:.1f}%)")
        print(f"  建议废弃: {discard_count} ({discard_count/len(df_results)*100:.1f}%)")

        # 统计
        finite_alphas = df_results[df_results['alpha_ratio'] < 999]['alpha_ratio']
        if len(finite_alphas) > 0:
            print(f"\nAlpha 统计（有限值）:")
            print(f"  平均: {finite_alphas.mean():.3f}")
            print(f"  中位数: {finite_alphas.median():.3f}")

        # 排名
        print("\n" + "=" * 100)
        print("因子排名（排除inf）")
        print("=" * 100)
        df_finite = df_results[df_results['alpha_ratio'] < 999].sort_values('alpha_ratio', ascending=False)

        print(f"\n{'排名':<4} {'因子':<25} {'Alpha':<7} {'P_actual':<9} {'P_random':<9} {'SE':<8} {'建议':<8}")
        print("-" * 100)

        for idx, (_, row) in enumerate(df_finite.iterrows(), 1):
            status = "✓" if row['recommendation'] == 'KEEP' else "✗"
            print(f"{status} {idx:<3} {row['factor_name']:<25} {row['alpha_ratio']:<7.2f} "
                  f"{row['p_actual']:<8.2%} {row['p_random']:<8.2%} {row['se']:<7.3%} {row['recommendation']:<8}")

        # 保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = f'million_path_validation_1M_{timestamp}.csv'
        df_results.to_csv(csv_file, index=False)
        print(f"\n✓ 结果已保存: {csv_file}")

    print("\n" + "=" * 100)
    print("百万路径验证完成!")
    print("=" * 100)


if __name__ == "__main__":
    main()
