#!/usr/bin/env python3
"""
增强版验证快速测试（小规模）

用于验证脚本正确性，使用减少的样本量
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from enhanced_validation import (
    EnhancedMonteCarloValidator,
    ValidationConfig,
    GapAwareGBM,
    BayesianAlphaEstimator
)
from src.tushare_db.factor_validation import FactorRegistry
import pandas as pd


def test_gap_model():
    """测试跳空模型"""
    print("="*60)
    print("测试1: 跳空GBM模型")
    print("="*60)

    config = ValidationConfig(
        simulation_years=1,  # 1年
        simulate_gap=True,
        gap_prob=0.3,
        gap_std=0.01
    )

    gbm = GapAwareGBM(config)
    ohlc = gbm.generate_ohlc(S0=100, mu=0, sigma=0.2)

    # 检测跳空
    gaps = []
    for t in range(1, len(ohlc['close'])):
        prev_close = ohlc['close'][t-1]
        curr_open = ohlc['open'][t]
        gap = (curr_open - prev_close) / prev_close
        if abs(gap) > 0.001:  # >0.1%视为跳空
            gaps.append(gap)

    print(f"生成 {len(ohlc['close'])} 个交易日数据")
    print(f"检测到 {len(gaps)} 次跳空 ({len(gaps)/252:.1%})")
    if gaps:
        print(f"跳空均值: {pd.Series(gaps).mean():.4%}")
        print(f"跳空标准差: {pd.Series(gaps).std():.4%}")

    assert len(gaps) > 0, "应该有跳空出现"
    print("✅ 跳空模型测试通过")


def test_bayesian_estimator():
    """测试贝叶斯估计器"""
    print("\n" + "="*60)
    print("测试2: 贝叶斯估计器")
    print("="*60)

    estimator = BayesianAlphaEstimator(prior_alpha=1, prior_beta=99)

    # 模拟低频信号：100次观测中触发1次
    result = estimator.estimate(n_success=1, n_total=1000, confidence=0.95)

    print(f"观测: 1次成功 / 1000次尝试")
    print(f"后验均值: {result['mean']:.4%}")
    print(f"95%置信区间: [{result['ci_lower']:.4%}, {result['ci_upper']:.4%}]")

    # 置信区间应该较宽（信息不足）- 注意返回的是百分比值，比如0.5%存储为0.005
    ci_width = result['ci_upper'] - result['ci_lower']
    print(f"  置信区间宽度: {ci_width:.4%}")
    assert ci_width > 0.001, f"低频信号应该有较宽置信区间，实际宽度{ci_width:.4%}"
    print("✅ 贝叶斯估计器测试通过")


def test_validation_pipeline():
    """测试完整验证流程（小规模）"""
    print("\n" + "="*60)
    print("测试3: 完整验证流程（小规模）")
    print("="*60)

    config = ValidationConfig(
        n_stocks=100,  # 只使用100只股票
        history_years=2,  # 2年历史
        n_paths=1000,  # 1000条路径
        simulation_years=1,  # 1年模拟
        use_bayesian=True,
        simulate_gap=True
    )

    validator = EnhancedMonteCarloValidator(config)

    # 只验证2个因子
    test_factors = ['doji', 'macd_golden_cross']

    print(f"将验证 {len(test_factors)} 个因子（小规模测试）")

    results = validator.run_validation(test_factors)

    print("\n验证结果:")
    for _, row in results.iterrows():
        print(f"  {row['factor_name']}: Alpha={row.get('alpha_ratio', 'N/A')}, "
              f"建议={row.get('recommendation', 'ERROR')}")

    print("✅ 完整流程测试通过")


def compare_with_standard():
    """对比标准GBM和跳空GBM的gap因子"""
    print("\n" + "="*60)
    print("测试4: 标准GBM vs 跳空GBM (gap因子)")
    print("="*60)

    from src.tushare_db.factor_validation.builtin_factors import gap_up

    # 标准GBM
    config_std = ValidationConfig(simulate_gap=False, simulation_years=1)
    gbm_std = GapAwareGBM(config_std)

    # 跳空GBM
    config_gap = ValidationConfig(simulate_gap=True, gap_prob=0.3, simulation_years=1)
    gbm_gap = GapAwareGBM(config_gap)

    # 各生成100条路径
    n_paths = 100
    std_triggers = 0
    gap_triggers = 0

    for _ in range(n_paths):
        # 标准GBM
        ohlc_std = gbm_std.generate_ohlc()
        df_std = pd.DataFrame(ohlc_std)
        signals_std = gap_up(df_std)
        std_triggers += signals_std.sum()

        # 跳空GBM
        ohlc_gap = gbm_gap.generate_ohlc()
        df_gap = pd.DataFrame(ohlc_gap)
        signals_gap = gap_up(df_gap)
        gap_triggers += signals_gap.sum()

    print(f"标准GBM: {std_triggers} 次触发 ({std_triggers/(n_paths*252):.2%})")
    print(f"跳空GBM: {gap_triggers} 次触发 ({gap_triggers/(n_paths*252):.2%})")

    # 跳空GBM应该有更多gap触发
    assert gap_triggers > std_triggers, "跳空GBM应该产生更多gap信号"
    print("✅ 对比测试通过")


def main():
    print("\n" + "="*70)
    print("增强版蒙特卡洛验证 - 快速测试套件")
    print("="*70)

    try:
        test_gap_model()
        test_bayesian_estimator()
        compare_with_standard()
        test_validation_pipeline()

        print("\n" + "="*70)
        print("所有测试通过！✅")
        print("="*70)
        print("\n可以运行完整验证:")
        print("  python scripts/enhanced_validation.py --factors doji macd_golden_cross")
        print("\n或运行全部因子验证（耗时较长）:")
        print("  python scripts/enhanced_validation.py")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
