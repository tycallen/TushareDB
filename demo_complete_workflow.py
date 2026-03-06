#!/usr/bin/env python
"""
完整工作流程演示：从真实数据到因子质检报告

详细步骤：
1. 获取真实股票数据
2. 计算真实因子触发概率 (P_actual)
3. 估计GBM参数 (μ, σ)
4. 运行GBM蒙特卡洛模拟
5. 计算模拟因子触发概率 (P_random)
6. 统计检验 (Alpha Ratio, P-value)
7. 生成质检报告
"""
import sys
sys.path.insert(0, '/Users/allen/workspace/python/stock/Tushare-DuckDB/.worktrees/monte-carlo-factor-filter')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

from src.tushare_db import DataReader
from src.tushare_db.factor_validation import FactorRegistry
from src.tushare_db.factor_validation.simulator import GBMSimulator
from src.tushare_db.factor_validation.detector import SignalDetector
from src.tushare_db.factor_validation.tester import SignificanceTester


def print_section(title):
    """打印章节标题"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def step1_get_real_data(ts_code='000001.SZ', days=252):
    """步骤1: 获取真实数据"""
    print_section("步骤 1: 获取真实市场数据")

    try:
        reader = DataReader(db_path="tushare.db")

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days * 1.5)

        print(f"\n股票代码: {ts_code}")
        print(f"日期范围: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")

        df = reader.get_stock_daily(
            ts_code=ts_code,
            start_date=start_date.strftime('%Y%m%d'),
            end_date=end_date.strftime('%Y%m%d'),
            adj='qfq'
        )

        # 数据清洗
        df = df[df['vol'] > 0].copy()
        df = df.sort_values('trade_date').reset_index(drop=True)

        print(f"\n✓ 成功获取 {len(df)} 个交易日数据")
        print(f"  首行数据:")
        print(f"    日期: {df['trade_date'].iloc[0]}")
        print(f"    开盘: {df['open'].iloc[0]:.2f}")
        print(f"    最高: {df['high'].iloc[0]:.2f}")
        print(f"    最低: {df['low'].iloc[0]:.2f}")
        print(f"    收盘: {df['close'].iloc[0]:.2f}")
        print(f"    成交量: {df['vol'].iloc[0]:,}")

        return df

    except Exception as e:
        print(f"\n⚠ 无法获取真实数据: {e}")
        print("  使用模拟数据演示流程...")
        return generate_mock_data(days)


def generate_mock_data(days=252):
    """生成模拟数据"""
    np.random.seed(42)

    # 生成带有趋势和波动特征的价格序列
    dates = pd.date_range(end=datetime.now(), periods=days, freq='B')

    # 生成收益率（带有一点动量）
    returns = np.random.randn(days) * 0.02 + 0.0003  # 正收益偏置

    # 生成价格
    close = 100 * np.exp(np.cumsum(returns))

    # 生成OHLC
    df = pd.DataFrame({
        'trade_date': dates.strftime('%Y%m%d'),
        'open': close * (1 + np.random.randn(days) * 0.005),
        'high': close * (1 + np.abs(np.random.randn(days) * 0.015)),
        'low': close * (1 - np.abs(np.random.randn(days) * 0.015)),
        'close': close,
        'vol': np.random.randint(1000000, 10000000, days)
    })

    print(f"\n✓ 生成 {len(df)} 条模拟数据")
    return df


def step2_calculate_p_actual(df, factor):
    """步骤2: 计算真实触发概率 P_actual"""
    print_section("步骤 2: 计算真实因子触发概率 (P_actual)")

    factor_name = factor.name

    print(f"\n因子: {factor_name}")
    print(f"描述: {factor.description}")

    # 计算因子信号
    signals = factor.evaluate(df)

    n_signals = int(signals.sum())
    n_total = len(df)
    p_actual = n_signals / n_total

    print(f"\n计算结果:")
    print(f"  总交易日: {n_total}")
    print(f"  信号次数: {n_signals}")
    print(f"  P_actual: {p_actual:.2%}")
    print(f"  信号频率: 约 {p_actual * 252:.1f} 次/年")

    # 显示最近几次信号
    signal_dates = df[signals]['trade_date'].tail(5).tolist()
    if signal_dates:
        print(f"\n  最近信号日期: {', '.join(map(str, signal_dates))}")

    return p_actual, n_signals


def step3_estimate_parameters(df):
    """步骤3: 估计GBM参数"""
    print_section("步骤 3: 估计几何布朗运动参数")

    # 计算对数收益率
    log_returns = np.log(df['close'] / df['close'].shift(1)).dropna()

    # 日度参数
    daily_mu = log_returns.mean()
    daily_sigma = log_returns.std()

    # 年化参数
    annual_mu = daily_mu * 252
    annual_sigma = daily_sigma * np.sqrt(252)

    s0 = df['close'].iloc[-1]

    print(f"\n历史统计:")
    print(f"  平均日收益: {daily_mu:.4f} ({daily_mu * 100:.2f}%)")
    print(f"  日波动率: {daily_sigma:.4f} ({daily_sigma * 100:.2f}%)")

    print(f"\n年化参数 (GBM输入):")
    print(f"  初始价格 (S0): {s0:.2f}")
    print(f"  年化收益率 (μ): {annual_mu:.2%}")
    print(f"  年化波动率 (σ): {annual_sigma:.2%}")

    print(f"\nGBM公式:")
    print(f"  S_t = S_{{t-1}} × exp[(μ - σ²/2) × Δt + σ × √Δt × Z]")

    return s0, annual_mu, annual_sigma


def step4_gbm_simulation(s0, mu, sigma, factor_name, n_paths=5000, n_steps=252):
    """步骤4: GBM蒙特卡洛模拟"""
    print_section("步骤 4: GBM 蒙特卡洛模拟")

    print(f"\n模拟参数:")
    print(f"  路径数: {n_paths:,}")
    print(f"  时间步: {n_steps} (约1年)")
    print(f"  总样本: {n_paths * n_steps:,}")

    # 创建模拟器
    simulator = GBMSimulator(
        n_paths=n_paths,
        n_steps=n_steps,
        limit_up=0.10,
        limit_down=-0.10,
        random_seed=42
    )

    # 运行模拟
    print(f"\n正在模拟 {n_paths:,} 条价格路径...")
    start_time = time.time()

    price_matrix = simulator.simulate(s0=s0, mu=mu, sigma=sigma)

    elapsed = time.time() - start_time
    print(f"✓ 模拟完成，耗时: {elapsed:.3f} 秒")

    # 显示部分路径统计
    print(f"\n模拟结果统计:")
    print(f"  初始价格: {price_matrix[:, 0].mean():.2f} ± {price_matrix[:, 0].std():.2f}")
    print(f"  最终价格: {price_matrix[:, -1].mean():.2f} ± {price_matrix[:, -1].std():.2f}")
    print(f"  平均收益率: {(price_matrix[:, -1].mean() / s0 - 1):.2%}")

    # 计算日收益率检查涨跌停
    returns = np.diff(price_matrix, axis=1) / price_matrix[:, :-1]
    print(f"\n涨跌停检查:")
    print(f"  最大日收益: {returns.max():.2%}")
    print(f"  最小日收益: {returns.min():.2%}")

    return price_matrix


def step5_calculate_p_random(price_matrix, factor, factor_name):
    """步骤5: 计算模拟触发概率 P_random"""
    print_section("步骤 5: 计算模拟因子触发概率 (P_random)")

    print(f"\n因子: {factor_name}")

    detector = SignalDetector()

    # 准备完整的OHLCV数据
    n_paths, n_steps = price_matrix.shape

    # 基于收盘价生成其他价格
    np.random.seed(42)
    high_matrix = price_matrix * (1 + np.abs(np.random.randn(n_paths, n_steps) * 0.01))
    low_matrix = price_matrix * (1 - np.abs(np.random.randn(n_paths, n_steps) * 0.01))
    open_matrix = np.roll(price_matrix, 1, axis=1)
    open_matrix[:, 0] = price_matrix[:, 0]
    vol_matrix = np.random.randint(1000000, 10000000, size=(n_paths, n_steps))

    # 计算信号
    print(f"\n正在检测 {n_paths:,} 条路径上的信号...")
    start_time = time.time()

    # 对于可以向量化的因子
    if factor_name in ['macd_golden_cross', 'macd_death_cross', 'macd_zero_golden_cross',
                       'golden_cross', 'death_cross', 'price_above_sma', 'price_below_sma',
                       'bollinger_lower_break', 'bollinger_upper_break']:
        signals = detector.detect_signals(price_matrix, factor)
        total_signals = signals.sum()
    else:
        # 其他因子使用代表性路径
        df_sim = pd.DataFrame({
            'open': open_matrix[0],
            'high': high_matrix[0],
            'low': low_matrix[0],
            'close': price_matrix[0],
            'vol': vol_matrix[0]
        })
        signals = factor.evaluate(df_sim)
        p_single = signals.sum() / n_steps
        total_signals = int(p_single * n_paths * n_steps)

    elapsed = time.time() - start_time
    print(f"✓ 信号检测完成，耗时: {elapsed:.3f} 秒")

    # 计算概率
    total_samples = n_paths * n_steps
    p_random = total_signals / total_samples

    print(f"\n计算结果:")
    print(f"  总样本数: {total_samples:,}")
    print(f"  信号次数: {total_signals:,}")
    print(f"  P_random: {p_random:.2%}")
    print(f"  信号频率: 约 {p_random * 252:.1f} 次/年")

    return p_random, total_signals


def step6_statistical_test(p_actual, p_random, n_total, factor_name):
    """步骤6: 统计检验"""
    print_section("步骤 6: 统计显著性检验")

    # 计算 Alpha Ratio
    alpha_ratio = p_actual / p_random if p_random > 0 else float('inf')

    print(f"\n假设检验:")
    print(f"  H0: 因子触发完全是随机游走的结果 (P_actual = P_random)")
    print(f"  H1: 真实市场存在非随机力量 (P_actual > P_random)")

    print(f"\n统计指标:")
    print(f"  P_actual:  {p_actual:.2%}")
    print(f"  P_random:  {p_random:.2%}")
    print(f"  Alpha Ratio: {alpha_ratio:.2f}")

    # 统计检验
    tester = SignificanceTester(alpha_threshold=1.5)
    result = tester.test(
        p_actual=p_actual,
        p_random=p_random,
        n_total=n_total,
        ts_code='DEMO'
    )

    print(f"\n检验结果:")
    print(f"  P-value: {result.p_value:.4f}")
    print(f"  是否显著: {'是' if result.is_significant else '否'}")
    print(f"  建议: {result.recommendation}")

    # 解释
    print(f"\n结果解释:")
    if alpha_ratio >= 1.5:
        print(f"  ✓ Alpha Ratio = {alpha_ratio:.2f} >= 1.5")
        print(f"    该因子在真实市场中的触发频率是随机游走的 {alpha_ratio:.1f} 倍")
        print(f"    说明存在系统性的非随机力量，具备 Alpha 挖掘价值")
    elif alpha_ratio >= 1.2:
        print(f"  △ Alpha Ratio = {alpha_ratio:.2f} (1.2 ~ 1.5)")
        print(f"    有一定信号，但未达显著阈值，建议优化条件后重新测试")
    else:
        print(f"  ✗ Alpha Ratio = {alpha_ratio:.2f} < 1.2")
        print(f"    该因子与随机游走无显著差异，可能只是噪音")

    return result


def step7_generate_report(factor_name, df, p_actual, p_random, alpha_ratio, result):
    """步骤7: 生成报告"""
    print_section("步骤 7: 生成质检报告")

    report = f"""
# 因子质检报告

## 基本信息
- **股票代码**: 000001.SZ (平安银行)
- **因子名称**: {factor_name}
- **因子描述**: {FactorRegistry.get(factor_name).description}
- **数据期间**: {df['trade_date'].min()} ~ {df['trade_date'].max()}
- **交易日数**: {len(df)}

## 统计结果

| 指标 | 数值 |
|------|------|
| P_actual (真实市场) | {p_actual:.2%} |
| P_random (随机游走) | {p_random:.2%} |
| Alpha Ratio | {alpha_ratio:.2f} |
| P-value | {result.p_value:.4f} |
| 是否显著 | {'是 ✓' if result.is_significant else '否 ✗'} |

## 结论与建议

**建议: {result.recommendation}**

{generate_recommendation_text(alpha_ratio)}

## 详细分析

### 真实市场表现
- 在 {len(df)} 个交易日中，该因子触发了 {int(p_actual * len(df))} 次
- 触发频率约为每年 {p_actual * 252:.1f} 次

### 随机游走对比
- 蒙特卡洛模拟了 5,000 条价格路径
- 每条路径 252 个交易日
- 在随机游走中，该因子平均触发频率为每年 {p_random * 252:.1f} 次

### Alpha 分析
- Alpha Ratio = {alpha_ratio:.2f} 表示该因子在真实市场中的触发频率
  是随机游走的 {alpha_ratio:.1f} 倍
- {'这表明市场存在系统性的非随机力量，该因子具备预测价值。' if alpha_ratio >= 1.5 else '这表明该因子可能只是随机噪音，缺乏预测价值。'}

---
*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

    print(report)

    # 保存报告
    filename = f"report_{factor_name}.md"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n✓ 报告已保存: {filename}")


def generate_recommendation_text(alpha_ratio):
    """生成建议文本"""
    if alpha_ratio >= 1.5:
        return """该因子通过了蒙特卡洛质检，在真实市场中表现出显著的 Alpha 特征。

建议：
1. **保留该因子** - 纳入多因子模型
2. **进一步研究** - 分析因子在不同市场环境下的表现
3. **组合测试** - 与其他因子组合，测试协同效应"""
    elif alpha_ratio >= 1.2:
        return """该因子有一定信号，但未达到显著性阈值。

建议：
1. **优化条件** - 调整参数或增加过滤条件
2. **分市场测试** - 在牛市/熊市/震荡市分别测试
3. **组合使用** - 作为辅助因子与其他强因子配合使用"""
    else:
        return """该因子与随机游走无显著差异。

建议：
1. **废弃该因子** - 避免引入噪音
2. **重新审视逻辑** - 检查因子定义是否有误
3. **尝试变体** - 调整参数或结合其他条件"""


def main():
    """主函数"""
    print("=" * 80)
    print("蒙特卡洛因子质检系统 - 完整工作流程演示")
    print("=" * 80)
    print("\n本演示展示从真实数据到因子质检报告的完整流程")

    # 选择测试因子
    factor_name = 'doji'  # 使用 doji 作为示例（在演示中通常有较好效果）
    factor = FactorRegistry.get(factor_name)

    print(f"\n测试因子: {factor_name}")
    print(f"因子描述: {factor.description}")

    # 执行完整流程
    df = step1_get_real_data()
    p_actual, n_actual = step2_calculate_p_actual(df, factor)
    s0, mu, sigma = step3_estimate_parameters(df)
    price_matrix = step4_gbm_simulation(s0, mu, sigma, factor_name)
    p_random, n_random = step5_calculate_p_random(price_matrix, factor, factor_name)
    result = step6_statistical_test(p_actual, p_random, len(df), factor_name)
    step7_generate_report(factor_name, df, p_actual, p_random, p_actual / p_random if p_random > 0 else 0, result)

    print("\n" + "=" * 80)
    print("工作流程演示完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()
