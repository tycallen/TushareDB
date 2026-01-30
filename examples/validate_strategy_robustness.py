"""
严格的样本外测试：验证策略是否过拟合

测试方案：
1. 训练期（2020-2022）：参数优化
2. 测试期（2023-2024）：样本外验证
3. 对比训练期和测试期的表现差异
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from backtest_money_signal import MoneySignalBacktest


def optimize_on_training_data(backtester, sector_data_train):
    """在训练数据上优化参数"""

    param_grid = {
        'money_threshold': [1.5, 2.0, 2.5, 3.0, 3.5],
        'price_threshold': [0.0, 0.5, 1.0, 1.5],
        'lookback': [10, 20, 30]
    }

    results = []
    total_tests = len(param_grid['money_threshold']) * len(param_grid['price_threshold']) * len(param_grid['lookback'])
    test_count = 0

    print(f"\n[训练期参数优化] 测试 {total_tests} 组参数...")

    for lookback in param_grid['lookback']:
        for money_th in param_grid['money_threshold']:
            for price_th in param_grid['price_threshold']:
                test_count += 1

                signals_data = backtester.detect_signals(
                    sector_data_train,
                    lookback=lookback,
                    money_threshold=money_th,
                    price_threshold=price_th
                )

                signal_count = signals_data['signal'].sum()

                # 至少需要5个信号
                if signal_count < 5:
                    continue

                returns_df = backtester.calculate_forward_returns(
                    signals_data, sector_data_train, periods=[5, 10, 20]
                )

                if len(returns_df) == 0:
                    continue

                # 计算指标
                metrics = {}
                for period in [5, 10, 20]:
                    col = f'return_{period}d'
                    if col in returns_df.columns:
                        valid_returns = returns_df[col].dropna()
                        if len(valid_returns) > 0:
                            win_rate = (valid_returns > 0).sum() / len(valid_returns) * 100
                            avg_return = valid_returns.mean()
                            sharpe = avg_return / (valid_returns.std() + 1e-6)

                            metrics[f'win_rate_{period}d'] = win_rate
                            metrics[f'avg_return_{period}d'] = avg_return
                            metrics[f'sharpe_{period}d'] = sharpe

                result = {
                    'lookback': lookback,
                    'money_threshold': money_th,
                    'price_threshold': price_th,
                    'signal_count': signal_count,
                    **metrics
                }

                results.append(result)

                if test_count % 10 == 0:
                    print(f"  进度: {test_count}/{total_tests}")

    if not results:
        return None

    results_df = pd.DataFrame(results)

    # 找出最佳参数（基于20日平均收益）
    best_idx = results_df['avg_return_20d'].idxmax()
    best_params = results_df.loc[best_idx]

    return {
        'lookback': int(best_params['lookback']),
        'money_threshold': float(best_params['money_threshold']),
        'price_threshold': float(best_params['price_threshold']),
        'metrics': best_params.to_dict()
    }


def test_on_validation_data(backtester, sector_data_test, params):
    """在测试数据上验证参数"""

    signals_data = backtester.detect_signals(
        sector_data_test,
        lookback=params['lookback'],
        money_threshold=params['money_threshold'],
        price_threshold=params['price_threshold']
    )

    signal_count = signals_data['signal'].sum()

    returns_df = backtester.calculate_forward_returns(
        signals_data, sector_data_test, periods=[1, 3, 5, 10, 20]
    )

    metrics = backtester.analyze_performance(returns_df, periods=[1, 3, 5, 10, 20])

    return {
        'signal_count': signal_count,
        'metrics': metrics
    }


def analyze_parameter_sensitivity(backtester, sector_data, base_params):
    """分析参数敏感性"""

    print("\n" + "=" * 80)
    print("参数敏感性分析（全量数据2020-2024）")
    print("=" * 80)

    # 测试不同阈值
    threshold_variations = [
        {'money_threshold': 2.0},
        {'money_threshold': 2.5},
        {'money_threshold': 3.0},
        {'money_threshold': 3.5},
        {'money_threshold': 4.0},
    ]

    results = []

    for variation in threshold_variations:
        test_params = base_params.copy()
        test_params.update(variation)

        signals_data = backtester.detect_signals(
            sector_data,
            lookback=test_params['lookback'],
            money_threshold=test_params['money_threshold'],
            price_threshold=test_params['price_threshold']
        )

        signal_count = signals_data['signal'].sum()

        if signal_count < 5:
            continue

        returns_df = backtester.calculate_forward_returns(
            signals_data, sector_data, periods=[20]
        )

        if len(returns_df) == 0 or 'return_20d' not in returns_df.columns:
            continue

        valid_returns = returns_df['return_20d'].dropna()

        if len(valid_returns) > 0:
            results.append({
                'money_threshold': test_params['money_threshold'],
                'signal_count': signal_count,
                'win_rate': (valid_returns > 0).sum() / len(valid_returns) * 100,
                'avg_return': valid_returns.mean(),
                'sharpe': valid_returns.mean() / (valid_returns.std() + 1e-6)
            })

    if results:
        df = pd.DataFrame(results)
        print("\n资金阈值敏感性（20日持有期）：")
        print(df.to_string(index=False))


def main():
    print("=" * 80)
    print("策略鲁棒性验证：严格样本外测试")
    print("=" * 80)

    backtester = MoneySignalBacktest('tushare.db')

    # 1. 获取训练数据（2020-2022）
    print("\n[1/6] 加载训练期数据（2020-2022）...")
    sector_data_train = backtester.get_sector_data('20200101', '20221231', 'L1')
    print(f"      训练期数据量: {len(sector_data_train)} 条")

    # 2. 获取测试数据（2023-2024）
    print("\n[2/6] 加载测试期数据（2023-2024）...")
    sector_data_test = backtester.get_sector_data('20230101', '20241231', 'L1')
    print(f"      测试期数据量: {len(sector_data_test)} 条")

    # 3. 在训练期优化参数
    print("\n[3/6] 在训练期优化参数...")
    best_params = optimize_on_training_data(backtester, sector_data_train)

    if best_params is None:
        print("未找到有效参数组合")
        backtester.close()
        return

    print("\n最佳参数（基于训练期）：")
    print(f"  回看周期: {best_params['lookback']}")
    print(f"  资金阈值: {best_params['money_threshold']}")
    print(f"  价格阈值: {best_params['price_threshold']}")
    print(f"  训练期信号数: {int(best_params['metrics']['signal_count'])}")

    # 4. 在测试期验证
    print("\n[4/6] 在测试期验证（样本外）...")
    test_results = test_on_validation_data(backtester, sector_data_test, best_params)

    print(f"\n测试期信号数: {test_results['signal_count']}")

    # 5. 对比训练期和测试期表现
    print("\n" + "=" * 80)
    print("训练期 vs 测试期 表现对比")
    print("=" * 80)

    comparison_data = []

    for period in ['5d', '10d', '20d']:
        train_metrics = best_params['metrics']
        test_metrics = test_results['metrics'].get(period, {})

        comparison_data.append({
            '持有期': period,
            '训练期胜率': f"{train_metrics.get(f'win_rate_{period}', 0):.1f}%",
            '测试期胜率': f"{test_metrics.get('胜率', 0):.1f}%",
            '训练期收益': f"{train_metrics.get(f'avg_return_{period}', 0):.2f}%",
            '测试期收益': f"{test_metrics.get('平均收益', 0):.2f}%",
            '训练期夏普': f"{train_metrics.get(f'sharpe_{period}', 0):.3f}",
            '测试期夏普': f"{test_metrics.get('夏普比率', 0):.3f}",
        })

    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))

    # 6. 参数敏感性分析
    print("\n[5/6] 参数敏感性分析...")
    sector_data_all = backtester.get_sector_data('20200101', '20241231', 'L1')
    analyze_parameter_sensitivity(backtester, sector_data_all, best_params)

    # 7. 结论
    print("\n" + "=" * 80)
    print("结论")
    print("=" * 80)

    # 检查过拟合迹象
    train_return_20d = best_params['metrics'].get('avg_return_20d', 0)
    test_return_20d = test_results['metrics'].get('20d', {}).get('平均收益', 0)

    if test_return_20d < train_return_20d * 0.5:
        print("\n⚠️  严重过拟合风险：")
        print(f"   - 测试期收益显著低于训练期（{test_return_20d:.2f}% vs {train_return_20d:.2f}%）")
        print(f"   - 收益衰减: {(1 - test_return_20d/train_return_20d)*100:.1f}%")
    elif test_return_20d < train_return_20d * 0.8:
        print("\n⚠️  轻度过拟合：")
        print(f"   - 测试期收益略低于训练期（{test_return_20d:.2f}% vs {train_return_20d:.2f}%）")
    else:
        print("\n✅ 策略表现稳定：")
        print(f"   - 测试期收益与训练期接近（{test_return_20d:.2f}% vs {train_return_20d:.2f}%）")

    if test_results['signal_count'] < 10:
        print(f"\n⚠️  信号数量过少：测试期仅{test_results['signal_count']}个信号，统计不显著")

    print("\n建议：")
    if test_return_20d < train_return_20d * 0.5 or test_results['signal_count'] < 10:
        print("  1. 降低阈值以获取更多信号")
        print("  2. 尝试不同的特征组合")
        print("  3. 考虑使用滚动窗口回测")
    else:
        print("  1. 策略在样本外数据表现可接受")
        print("  2. 建议进一步测试更长时间周期")
        print("  3. 可以考虑小资金实盘验证")

    backtester.close()


if __name__ == '__main__':
    main()
