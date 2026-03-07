import pandas as pd
import numpy as np
import pytest


def test_multi_timeframe_alignment_factor():
    """测试多周期共振因子"""
    from src.tushare_db.factor_validation.builtin_factors import multi_timeframe_alignment

    # 价格同时突破短期、中期、长期均线
    # 短期均线 < 中期均线 < 长期均线，价格突破所有均线
    df = pd.DataFrame({
        'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                  110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
                  125]  # 最后一日大幅突破所有均线
    })

    signals = multi_timeframe_alignment(df, short_period=5, medium_period=10, long_period=20)

    # 最后一天应该有共振信号
    assert signals.sum() > 0
    assert signals.dtype == bool


def test_higher_high_lower_low_sequence_factor():
    """测试HHLL序列因子（Higher High, Lower Low）"""
    from src.tushare_db.factor_validation.builtin_factors import higher_high_lower_low_sequence

    # 构建HHLL序列：需要连续两个更高的高点和更高的低点
    # 数据设计：
    # index 0-2: 横盘整理 (100->100->100)，建立基础
    # index 3: 第一次上涨 high=102 (higher_high=True, higher_high_prev=False)
    # index 4: 第二次上涨 high=104 (higher_high=True, higher_high_prev=True) -> HH sequence!
    # index 3-4 同时满足 higher_low 条件

    high = [100, 100, 100, 102, 104, 106, 108]  # 0-2横盘，3-6连续HH
    low = [98, 98, 98, 100, 102, 104, 106]      # 0-2横盘，3-6连续HL
    close = [(h + l) / 2 for h, l in zip(high, low)]

    df = pd.DataFrame({
        'high': high,
        'low': low,
        'close': close
    })

    signals = higher_high_lower_low_sequence(df, lookback=3)

    # 应该有HHLL序列信号（在index 4首次触发）
    assert signals.sum() > 0
    assert signals.dtype == bool


def test_support_resistance_breakout_factor():
    """测试支撑阻力突破因子"""
    from src.tushare_db.factor_validation.builtin_factors import support_resistance_breakout

    # 价格长期盘整后突破阻力位
    close = [100 + (i % 5) * 0.5 for i in range(30)]  # 横盘震荡
    close[-1] = 110  # 最后一日突破阻力位

    df = pd.DataFrame({
        'close': close,
        'high': [c + 1 for c in close],
        'low': [c - 1 for c in close]
    })

    signals = support_resistance_breakout(df, lookback=20, breakout_threshold=0.03)

    # 最后一日应该有突破信号
    assert signals.sum() > 0
    assert signals.dtype == bool
