"""
更稳健的资金流策略：
1. 降低阈值，增加样本量
2. 添加更多过滤条件
3. 多特征组合
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from backtest_money_signal import MoneySignalBacktest


def detect_robust_signals(backtester, sector_data, lookback=20):
    """
    更稳健的信号检测：多特征组合

    信号条件：
    1. 大单净流入 > 1.5σ（降低阈值，增加样本）
    2. 涨跌幅 < 0.5σ（价格未明显上涨）
    3. 大单占比 > 历史中位数（确保资金质量）
    4. 连续性：前3日内至少2日大单净流入为正
    """
    df = sector_data.copy().sort_values(['sector_code', 'trade_date'])

    # 计算统计量
    for sector in df['sector_code'].unique():
        mask = df['sector_code'] == sector

        # 大单净流入的统计量
        df.loc[mask, 'lg_net_ma'] = df.loc[mask, 'lg_net_amount'].rolling(
            lookback, min_periods=5
        ).mean()
        df.loc[mask, 'lg_net_std'] = df.loc[mask, 'lg_net_amount'].rolling(
            lookback, min_periods=5
        ).std()

        # 涨跌幅统计量
        df.loc[mask, 'pct_chg_ma'] = df.loc[mask, 'pct_chg'].rolling(
            lookback, min_periods=5
        ).mean()
        df.loc[mask, 'pct_chg_std'] = df.loc[mask, 'pct_chg'].rolling(
            lookback, min_periods=5
        ).std()

        # 大单占比统计量
        df.loc[mask, 'lg_ratio_median'] = df.loc[mask, 'lg_ratio'].rolling(
            lookback, min_periods=5
        ).median()

        # 连续性：前3日大单净流入为正的天数
        df.loc[mask, 'lg_positive_count'] = (
            (df.loc[mask, 'lg_net_amount'] > 0).rolling(3, min_periods=1).sum()
        )

    # Z-score
    df['lg_net_zscore'] = (df['lg_net_amount'] - df['lg_net_ma']) / (df['lg_net_std'] + 1e-6)
    df['pct_chg_zscore'] = (df['pct_chg'] - df['pct_chg_ma']) / (df['pct_chg_std'] + 0.01)

    # 信号
    df['signal'] = (
        (df['lg_net_zscore'] > 1.5) &  # 降低到1.5σ
        (df['pct_chg_zscore'] < 0.5) &
        (df['lg_net_amount'] > 0) &
        (df['lg_ratio'] > df['lg_ratio_median']) &  # 大单占比高于历史
        (df['lg_positive_count'] >= 2) &  # 连续性
        (df['pct_chg'].notna())
    )

    return df


def test_with_walk_forward(backtester):
    """
    滚动窗口验证：更真实的样本外测试

    方案：
    - 每年作为一个测试期
    - 使用前2年数据优化参数
    - 在当年验证
    """

    periods = [
        {'train_start': '20200101', 'train_end': '20211231', 'test_start': '20220101', 'test_end': '20221231', 'name': '2022'},
        {'train_start': '20210101', 'train_end': '20221231', 'test_start': '20230101', 'test_end': '20231231', 'name': '2023'},
        {'train_start': '20220101', 'train_end': '20231231', 'test_start': '20240101', 'test_end': '20241231', 'name': '2024'},
    ]

    all_results = []

    for period in periods:
        print(f"\n{'='*80}")
        print(f"测试年份: {period['name']}")
        print(f"训练期: {period['train_start']} ~ {period['train_end']}")
        print(f"测试期: {period['test_start']} ~ {period['test_end']}")
        print('='*80)

        # 训练数据
        train_data = backtester.get_sector_data(
            period['train_start'], period['train_end'], 'L1'
        )

        # 测试数据
        test_data = backtester.get_sector_data(
            period['test_start'], period['test_end'], 'L1'
        )

        # 检测信号（使用固定参数）
        train_signals = detect_robust_signals(backtester, train_data, lookback=20)
        test_signals = detect_robust_signals(backtester, test_data, lookback=20)

        train_count = train_signals['signal'].sum()
        test_count = test_signals['signal'].sum()

        print(f"\n训练期信号数: {train_count}")
        print(f"测试期信号数: {test_count}")

        if test_count < 3:
            print("⚠️  测试期信号过少，跳过")
            continue

        # 计算收益
        test_returns = backtester.calculate_forward_returns(
            test_signals, test_data, periods=[5, 10, 20]
        )

        if len(test_returns) == 0:
            print("⚠️  无有效收益数据")
            continue

        # 分析表现
        for hold_period in [5, 10, 20]:
            col = f'return_{hold_period}d'
            if col in test_returns.columns:
                valid_returns = test_returns[col].dropna()

                if len(valid_returns) > 0:
                    win_rate = (valid_returns > 0).sum() / len(valid_returns) * 100
                    avg_return = valid_returns.mean()
                    sharpe = avg_return / (valid_returns.std() + 1e-6)

                    all_results.append({
                        'year': period['name'],
                        'hold_period': f'{hold_period}d',
                        'signal_count': test_count,
                        'win_rate': win_rate,
                        'avg_return': avg_return,
                        'sharpe': sharpe
                    })

                    print(f"\n【{hold_period}日持有】")
                    print(f"  胜率: {win_rate:.1f}%")
                    print(f"  平均收益: {avg_return:.2f}%")
                    print(f"  夏普比率: {sharpe:.3f}")

    return pd.DataFrame(all_results)


def main():
    print("=" * 80)
    print("稳健策略验证：滚动窗口回测")
    print("=" * 80)
    print("\n策略改进：")
    print("  1. 降低阈值（1.5σ）增加样本量")
    print("  2. 添加大单占比过滤")
    print("  3. 添加连续性要求")
    print("  4. 使用滚动窗口验证")

    backtester = MoneySignalBacktest('tushare.db')

    # 滚动窗口测试
    results_df = test_with_walk_forward(backtester)

    if len(results_df) > 0:
        # 按持有期汇总
        print("\n" + "=" * 80)
        print("各年度表现汇总")
        print("=" * 80)

        for hold_period in ['5d', '10d', '20d']:
            period_data = results_df[results_df['hold_period'] == hold_period]

            if len(period_data) > 0:
                print(f"\n【{hold_period}持有期】")
                print(period_data[['year', 'signal_count', 'win_rate', 'avg_return', 'sharpe']].to_string(index=False))

                print(f"\n  均值:")
                print(f"    平均信号数: {period_data['signal_count'].mean():.0f}")
                print(f"    平均胜率: {period_data['win_rate'].mean():.1f}%")
                print(f"    平均收益: {period_data['avg_return'].mean():.2f}%")
                print(f"    平均夏普: {period_data['sharpe'].mean():.3f}")

                print(f"\n  稳定性:")
                print(f"    收益标准差: {period_data['avg_return'].std():.2f}%")
                print(f"    胜率标准差: {period_data['win_rate'].std():.1f}%")

        # 保存结果
        output_dir = Path('output/robust_strategy')
        output_dir.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_dir / 'walk_forward_results.csv', index=False, encoding='utf-8-sig')
        print(f"\n详细结果已保存至: {output_dir}")

        # 结论
        print("\n" + "=" * 80)
        print("结论")
        print("=" * 80)

        overall_avg_return = results_df[results_df['hold_period'] == '20d']['avg_return'].mean()
        overall_avg_signals = results_df[results_df['hold_period'] == '20d']['signal_count'].mean()
        return_stability = results_df[results_df['hold_period'] == '20d']['avg_return'].std()

        print(f"\n20日持有期总体表现:")
        print(f"  平均年度信号数: {overall_avg_signals:.0f}")
        print(f"  平均收益: {overall_avg_return:.2f}%")
        print(f"  收益波动: {return_stability:.2f}%")

        if overall_avg_signals >= 15 and overall_avg_return > 1.0 and return_stability < 3.0:
            print("\n✅ 策略表现稳定:")
            print("  - 信号数量充足")
            print("  - 收益为正且稳定")
            print("  - 可以考虑进一步优化或小规模实盘")
        elif overall_avg_signals < 10:
            print("\n⚠️  信号数量仍然不足:")
            print("  - 需要进一步降低阈值或增加特征")
        elif return_stability > 3.0:
            print("\n⚠️  收益不稳定:")
            print("  - 不同年份表现差异大")
            print("  - 可能受市场环境影响较大")
        else:
            print("\n⚠️  收益不足:")
            print("  - 策略优势不明显")
            print("  - 需要重新设计信号逻辑")

    backtester.close()


if __name__ == '__main__':
    main()
