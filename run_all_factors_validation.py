#!/usr/bin/env python
"""
批量验证所有内置因子

为所有30个内置因子运行完整的蒙特卡洛验证流程：
1. 模拟真实数据
2. 计算 P_actual
3. GBM 模拟
4. 计算 P_random
5. 统计检验
6. 保存结果
"""
import sys
sys.path.insert(0, '/Users/allen/workspace/python/stock/Tushare-DuckDB/.worktrees/monte-carlo-factor-filter')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.tushare_db.factor_validation import FactorRegistry
from src.tushare_db.factor_validation.simulator import GBMSimulator
from src.tushare_db.factor_validation.detector import SignalDetector
from src.tushare_db.factor_validation.tester import SignificanceTester, TestResult
from src.tushare_db.factor_validation.report_manager import ReportManager, ValidationRecord


def generate_mock_data(days=252, trend='random', volatility='medium'):
    """生成模拟股票数据"""
    np.random.seed(42)

    # 设置波动率
    vol_map = {'low': 0.015, 'medium': 0.025, 'high': 0.04}
    daily_vol = vol_map.get(volatility, 0.025)

    # 设置趋势
    trend_map = {'down': -0.0003, 'random': 0.0001, 'up': 0.0005}
    daily_return = trend_map.get(trend, 0.0001)

    dates = pd.date_range(end=datetime.now(), periods=days, freq='B')

    # 生成价格
    returns = np.random.randn(days) * daily_vol + daily_return
    close = 100 * np.exp(np.cumsum(returns))

    # 生成 OHLC
    df = pd.DataFrame({
        'trade_date': dates.strftime('%Y%m%d'),
        'open': close * (1 + np.random.randn(days) * 0.005),
        'high': close * (1 + np.abs(np.random.randn(days) * 0.015)),
        'low': close * (1 - np.abs(np.random.randn(days) * 0.015)),
        'close': close,
        'vol': np.random.randint(1000000, 10000000, days)
    })

    return df


def validate_factor(factor_name, df_real, n_simulations=5000, n_steps=252):
    """验证单个因子"""
    print(f"  正在验证: {factor_name}...", end=" ")

    try:
        factor = FactorRegistry.get(factor_name)

        # 1. 计算 P_actual
        signals = factor.evaluate(df_real)
        p_actual = signals.sum() / len(df_real)
        n_actual = int(signals.sum())

        # 2. 估计参数
        log_returns = np.log(df_real['close'] / df_real['close'].shift(1)).dropna()
        mu = log_returns.mean() * 252
        sigma = log_returns.std() * np.sqrt(252)
        s0 = df_real['close'].iloc[-1]

        # 3. GBM 模拟
        simulator = GBMSimulator(
            n_paths=n_simulations,
            n_steps=n_steps,
            limit_up=0.10,
            limit_down=-0.10,
            random_seed=42
        )
        price_matrix = simulator.simulate(s0=s0, mu=mu, sigma=sigma)

        # 4. 计算 P_random
        detector = SignalDetector()

        # 对于可以向量化的因子
        if factor_name in ['macd_golden_cross', 'macd_death_cross', 'macd_zero_golden_cross',
                          'golden_cross', 'death_cross', 'price_above_sma', 'price_below_sma',
                          'bollinger_lower_break', 'bollinger_upper_break']:
            signal_matrix = detector.detect_signals(price_matrix, factor)
            total_signals = signal_matrix.sum()
        else:
            # 使用代表性路径估算
            df_sim = pd.DataFrame({
                'open': np.roll(price_matrix[0], 1),
                'high': price_matrix[0] * 1.01,
                'low': price_matrix[0] * 0.99,
                'close': price_matrix[0],
                'vol': np.random.randint(1000000, 10000000, n_steps)
            })
            df_sim['open'].iloc[0] = price_matrix[0][0]
            signals_sim = factor.evaluate(df_sim)
            p_single = signals_sim.sum() / n_steps
            total_signals = int(p_single * n_simulations * n_steps)

        p_random = total_signals / (n_simulations * n_steps)

        # 5. 统计检验
        tester = SignificanceTester(alpha_threshold=1.5)
        result = tester.test(
            p_actual=p_actual,
            p_random=p_random,
            n_total=len(df_real),
            ts_code='MOCK'
        )

        print(f"✓ Alpha={result.alpha_ratio:.2f} ({result.recommendation})")

        return {
            'factor_name': factor_name,
            'factor_description': factor.description,
            'p_actual': p_actual,
            'p_random': p_random,
            'alpha_ratio': result.alpha_ratio,
            'n_actual': n_actual,
            'n_random': total_signals,
            'p_value': result.p_value,
            'is_significant': result.is_significant,
            'recommendation': result.recommendation,
            'mu': mu,
            'sigma': sigma,
            'success': True
        }

    except Exception as e:
        print(f"✗ 错误: {str(e)[:40]}")
        return {
            'factor_name': factor_name,
            'factor_description': '',
            'p_actual': 0,
            'p_random': 0,
            'alpha_ratio': 0,
            'recommendation': 'ERROR',
            'error': str(e),
            'success': False
        }


def main():
    """主函数"""
    print("=" * 80)
    print("批量因子验证 - 所有内置因子")
    print("=" * 80)
    print(f"\n开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 创建报告管理器
    manager = ReportManager()

    # 生成模拟数据
    print("\n生成模拟数据...")
    df_real = generate_mock_data(days=252, trend='random', volatility='medium')
    print(f"✓ 生成 {len(df_real)} 天数据")
    print(f"  价格范围: {df_real['close'].min():.2f} ~ {df_real['close'].max():.2f}")
    print(f"  平均收益率: {np.log(df_real['close']/df_real['close'].shift(1)).dropna().mean()*252:.2%}")
    print(f"  年化波动率: {np.log(df_real['close']/df_real['close'].shift(1)).dropna().std()*np.sqrt(252):.2%}")

    # 获取所有因子
    all_factors = FactorRegistry.list_builtin()
    print(f"\n共 {len(all_factors)} 个因子需要验证")

    # 验证所有因子
    results = []

    print("\n" + "=" * 80)
    print("开始验证")
    print("=" * 80)

    start_time = datetime.now()

    for i, factor_name in enumerate(sorted(all_factors), 1):
        result = validate_factor(factor_name, df_real)
        results.append(result)

        # 保存到数据库
        if result['success']:
            record = ValidationRecord(
                timestamp=datetime.now().isoformat(),
                factor_name=result['factor_name'],
                factor_description=result['factor_description'],
                ts_code='MOCK001',
                stock_name='模拟股票',
                start_date='20240101',
                end_date='20241231',
                n_days=len(df_real),
                n_simulations=5000,
                simulation_days=252,
                alpha_threshold=1.5,
                p_actual=result['p_actual'],
                p_random=result['p_random'],
                alpha_ratio=result['alpha_ratio'],
                n_signals_actual=result['n_actual'],
                n_signals_random=result['n_random'],
                p_value=result['p_value'],
                is_significant=result['is_significant'],
                recommendation=result['recommendation'],
                mu=result['mu'],
                sigma=result['sigma'],
                notes=f'批量验证 #{i}'
            )
            manager._insert_record(record)

    elapsed = (datetime.now() - start_time).total_seconds()

    # 结果汇总
    print("\n" + "=" * 80)
    print("验证完成")
    print("=" * 80)
    print(f"\n总耗时: {elapsed:.1f} 秒")
    print(f"平均每个因子: {elapsed/len(all_factors):.2f} 秒")

    # 统计
    df_results = pd.DataFrame([r for r in results if r['success']])

    if not df_results.empty:
        print(f"\n成功验证: {len(df_results)}/{len(all_factors)}")

        keep_count = (df_results['recommendation'] == 'KEEP').sum()
        discard_count = (df_results['recommendation'] == 'DISCARD').sum()

        print(f"\n结果分布:")
        print(f"  建议保留 (KEEP): {keep_count} ({keep_count/len(df_results)*100:.1f}%)")
        print(f"  建议废弃 (DISCARD): {discard_count} ({discard_count/len(df_results)*100:.1f}%)")

        print(f"\nAlpha Ratio 统计:")
        print(f"  平均值: {df_results['alpha_ratio'].mean():.3f}")
        print(f"  中位数: {df_results['alpha_ratio'].median():.3f}")
        print(f"  最大值: {df_results['alpha_ratio'].max():.3f}")
        print(f"  最小值: {df_results['alpha_ratio'].min():.3f}")

        # 排名
        print("\n" + "=" * 80)
        print("因子排名 (按 Alpha Ratio)")
        print("=" * 80)

        df_sorted = df_results.sort_values('alpha_ratio', ascending=False)

        print(f"\n{'排名':<4} {'因子名称':<25} {'Alpha':<8} {'P_actual':<10} {'P_random':<10} {'建议':<8}")
        print("-" * 80)

        for idx, (_, row) in enumerate(df_sorted.iterrows(), 1):
            status = "✓" if row['recommendation'] == 'KEEP' else "✗"
            print(f"{status} {idx:<3} {row['factor_name']:<25} {row['alpha_ratio']:<8.2f} "
                  f"{row['p_actual']:<9.2%} {row['p_random']:<9.2%} {row['recommendation']:<8}")

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
            cat_df = df_results[df_results['factor_name'].isin(cat_factors)]
            if len(cat_df) > 0:
                avg_alpha = cat_df['alpha_ratio'].mean()
                keep_rate = (cat_df['recommendation'] == 'KEEP').mean() * 100
                print(f"{cat_name:12s}: 平均Alpha={avg_alpha:6.2f}, 保留率={keep_rate:5.1f}%")

        # 保存结果
        output_file = f'all_factors_validation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        df_sorted.to_csv(output_file, index=False)
        print(f"\n✓ 结果已保存: {output_file}")

        # 生成报告
        report = f"""
# 批量因子验证报告

## 验证信息
- **验证时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **总因子数**: {len(all_factors)}
- **成功验证**: {len(df_results)}
- **总耗时**: {elapsed:.1f} 秒

## 数据信息
- **数据天数**: {len(df_real)}
- **模拟路径**: 5,000
- **模拟天数**: 252
- **Alpha阈值**: 1.5

## 统计摘要

### 结果分布
- 建议保留 (KEEP): {keep_count} ({keep_count/len(df_results)*100:.1f}%)
- 建议废弃 (DISCARD): {discard_count} ({discard_count/len(df_results)*100:.1f}%)

### Alpha Ratio
- 平均值: {df_results['alpha_ratio'].mean():.3f}
- 中位数: {df_results['alpha_ratio'].median():.3f}
- 最大值: {df_results['alpha_ratio'].max():.3f}
- 最小值: {df_results['alpha_ratio'].min():.3f}

## Top 10 因子

"""
        for idx, (_, row) in enumerate(df_sorted.head(10).iterrows(), 1):
            report += f"{idx}. **{row['factor_name']}**: Alpha={row['alpha_ratio']:.2f}, {row['recommendation']}\n"

        report_file = f'report_all_factors_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"✓ 报告已保存: {report_file}")

    print("\n" + "=" * 80)
    print("所有结果已保存到数据库:")
    print(f"  {manager.db_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
