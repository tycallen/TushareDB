"""
使用优化后的参数进行完整回测
"""

import sys
from pathlib import Path
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

# 导入回测模块
from backtest_money_signal import MoneySignalBacktest

def run_optimized_backtest():
    """使用优化参数运行回测"""

    print("=" * 80)
    print("优化参数策略回测（2020-2024完整数据）")
    print("=" * 80)

    backtester = MoneySignalBacktest('tushare.db')

    # 获取数据
    print("\n[1/4] 获取数据...")
    sector_data = backtester.get_sector_data('20200101', '20241231', 'L1')
    print(f"      数据量: {len(sector_data)} 条")

    # 测试3组最佳参数
    param_sets = [
        {
            'name': '激进型（最高收益）',
            'lookback': 30,
            'money_threshold': 3.0,
            'price_threshold': 1.0
        },
        {
            'name': '平衡型（高胜率）',
            'lookback': 20,
            'money_threshold': 2.5,
            'price_threshold': 1.0
        },
        {
            'name': '稳健型',
            'lookback': 10,
            'money_threshold': 2.0,
            'price_threshold': 0.5
        }
    ]

    all_results = []

    for param_set in param_sets:
        print(f"\n" + "=" * 80)
        print(f"测试参数: {param_set['name']}")
        print("=" * 80)
        print(f"  回看周期: {param_set['lookback']}天")
        print(f"  资金阈值: {param_set['money_threshold']}σ")
        print(f"  价格阈值: {param_set['price_threshold']}σ")

        # 检测信号
        signals_data = backtester.detect_signals(
            sector_data,
            lookback=param_set['lookback'],
            money_threshold=param_set['money_threshold'],
            price_threshold=param_set['price_threshold']
        )

        signal_count = signals_data['signal'].sum()
        print(f"\n  信号数量: {signal_count}")

        if signal_count < 5:
            print("  ⚠️  信号过少，跳过")
            continue

        # 计算收益
        returns_df = backtester.calculate_forward_returns(
            signals_data,
            sector_data,
            periods=[1, 3, 5, 10, 20]
        )

        # 分析表现
        metrics = backtester.analyze_performance(returns_df, periods=[1, 3, 5, 10, 20])

        # 输出结果
        print("\n  回测结果:")
        for period, data in metrics.items():
            print(f"\n  【{period}持有期】")
            print(f"    胜率: {data['胜率']:.2f}%")
            print(f"    平均收益: {data['平均收益']:.2f}%")
            print(f"    夏普比率: {data['夏普比率']:.3f}")
            print(f"    盈亏比: {data['盈亏比']:.2f}")

        # 保存结果
        result = {
            'param_name': param_set['name'],
            'lookback': param_set['lookback'],
            'money_th': param_set['money_threshold'],
            'price_th': param_set['price_threshold'],
            'signal_count': signal_count,
            **{f"{k}_{period}": v for period, data in metrics.items() for k, v in data.items()}
        }
        all_results.append(result)

    # 保存对比结果
    if all_results:
        results_df = pd.DataFrame(all_results)
        output_dir = Path('output/optimized_backtest')
        output_dir.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_dir / 'param_comparison.csv', index=False, encoding='utf-8-sig')

        print("\n" + "=" * 80)
        print("参数对比总结")
        print("=" * 80)

        # 显示5日和10日收益对比
        comparison = results_df[[
            'param_name', 'signal_count',
            '胜率_5d', '平均收益_5d', '夏普比率_5d',
            '胜率_10d', '平均收益_10d', '夏普比率_10d'
        ]]
        print(comparison.to_string(index=False))

        print(f"\n详细结果已保存至: {output_dir}")

    backtester.close()


if __name__ == '__main__':
    run_optimized_backtest()
