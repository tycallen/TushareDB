#!/usr/bin/env python
"""验证新添加的因子"""
import sys
sys.path.insert(0, '/Users/allen/workspace/python/stock/Tushare-DuckDB/.worktrees/monte-carlo-factor-filter')

import pandas as pd
import numpy as np
from src.tushare_db.factor_validation import FactorRegistry


def test_factor_function(factor_name, df):
    """测试单个因子函数"""
    try:
        factor = FactorRegistry.get(factor_name)
        result = factor.evaluate(df)
        signal_count = result.sum()
        signal_pct = signal_count / len(df) * 100
        return True, signal_count, signal_pct
    except Exception as e:
        return False, 0, str(e)


def main():
    print("=" * 80)
    print("验证新添加的因子")
    print("=" * 80)

    # 创建模拟数据
    np.random.seed(42)
    n = 100
    dates = pd.date_range('2024-01-01', periods=n)

    # 生成价格数据
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.5)
    low = close - np.abs(np.random.randn(n) * 0.5)
    open_price = close + np.random.randn(n) * 0.3
    vol = np.random.randint(1000000, 10000000, n)

    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'vol': vol
    }, index=dates)

    print(f"\n测试数据: {n} 天")
    print(f"价格范围: {df['close'].min():.2f} - {df['close'].max():.2f}")

    # 新添加的因子
    new_factors = {
        'KDJ': ['kdj_golden_cross', 'kdj_death_cross'],
        'Williams %R': ['williams_r_oversold', 'williams_r_overbought'],
        'CCI': ['cci_oversold', 'cci_overbought'],
        'ATR': ['atr_breakout', 'atr_breakdown'],
        '蜡烛图': ['bullish_engulfing', 'bearish_engulfing', 'hammer',
                   'shooting_star', 'doji', 'three_white_soldiers', 'three_black_crows']
    }

    print("\n" + "=" * 80)
    print("因子验证结果")
    print("=" * 80)

    all_passed = True
    for category, factors in new_factors.items():
        print(f"\n【{category}】")
        for factor_name in factors:
            success, count, pct = test_factor_function(factor_name, df)
            if success:
                print(f"  ✓ {factor_name:25s}: 信号数={count:3d} ({pct:5.2f}%)")
            else:
                print(f"  ✗ {factor_name:25s}: 错误={pct}")
                all_passed = False

    print("\n" + "=" * 80)
    if all_passed:
        print("✓ 所有新因子验证通过！")
    else:
        print("✗ 部分因子验证失败")
    print("=" * 80)

    # 验证总因子数
    total_factors = FactorRegistry.list_builtin()
    print(f"\n总因子数: {len(total_factors)}")


if __name__ == "__main__":
    main()
