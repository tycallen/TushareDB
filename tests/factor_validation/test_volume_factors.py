import pandas as pd
import numpy as np
import pytest


def test_volume_price_divergence_factor():
    """测试量价背离因子"""
    from src.tushare_db.factor_validation.builtin_factors import volume_price_divergence

    # 价格持续下跌但成交量显著放大（底背离）
    # 需要足够的数据点（lookback=10需要至少11行数据）
    close = [100, 99, 98, 97, 96, 95, 94, 93, 92, 91,  # 10期下跌 -5%
             90, 88, 86, 84, 82, 80, 78, 76, 74, 72]   # 再跌 -20%
    vol = [1000] * 10 + [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500]  # 后期大幅放量

    df = pd.DataFrame({'close': close, 'vol': vol})

    signals = volume_price_divergence(df, lookback=10, divergence_threshold=0.05)

    # 后期应该有底背离信号
    assert signals.sum() > 0
    assert signals.dtype == bool


def test_on_balance_volume_breakout_factor():
    """测试OBV突破因子"""
    from src.tushare_db.factor_validation.builtin_factors import on_balance_volume_breakout

    # 价格上涨伴随成交量放大
    df = pd.DataFrame({
        'close': [100, 101, 102, 101, 103, 104, 103, 105, 106, 107,
                  106, 108, 109, 110, 111, 112, 113, 114, 115, 116],
        'vol': [1000] * 15 + [2000, 2200, 2500, 2800, 3000]  # 后期放量突破
    })

    signals = on_balance_volume_breakout(df, lookback=10)

    # 后期应该有OBV突破信号
    assert signals.sum() > 0
    assert signals.dtype == bool


def test_volume_weighted_momentum_factor():
    """测试成交量加权动量因子"""
    from src.tushare_db.factor_validation.builtin_factors import volume_weighted_momentum

    # 价格先横盘震荡，然后突破新高，伴随成交量放大
    # momentum_cross 需要价格从低momentum区域突破到高momentum区域
    close = [100, 101, 100, 101, 100, 101, 100, 101, 100, 101,  # 前期横盘，momentum接近0
             112, 115, 118]  # 后期突然突破，momentum大幅上升
    vol = [1000] * 10 + [2000, 2500, 3000]  # 后期放量

    df = pd.DataFrame({'close': close, 'vol': vol})

    signals = volume_weighted_momentum(df, lookback=5, threshold=0.03)

    # 突破时应该有信号
    assert signals.sum() > 0
    assert signals.dtype == bool


def test_accumulation_distribution_breakout_factor():
    """测试集散指标(ADL)突破因子"""
    from src.tushare_db.factor_validation.builtin_factors import accumulation_distribution_breakout

    # 模拟吸筹后突破的数据
    # AD Line需要money flow multiplier为正（收盘在区间上半部分）才能上升
    close = []
    high = []
    low = []
    vol = []

    # 前期缓慢吸筹 - 收盘接近最高点（positive multiplier）
    for i in range(20):
        c = 100 + i * 0.1
        close.append(c + 0.4)  # 接近最高点
        high.append(c + 0.5)
        low.append(c - 0.5)
        vol.append(1000)

    # 中期震荡整理 - AD Line横盘（multiplier接近0）
    for i in range(10):
        c = 102
        close.append(c)  # 收盘在中间位置
        high.append(c + 1.0)
        low.append(c - 1.0)
        vol.append(1000)

    # 后期强势突破：收盘接近最高点，成交量显著放大
    for i in range(15):
        c = 102 + i * 0.5
        close.append(c + 0.4)  # 收盘接近最高点，保证positive multiplier
        high.append(c + 0.5)
        low.append(c - 0.5)
        vol.append(5000 + i * 500)  # 大幅放量

    df = pd.DataFrame({
        'close': close,
        'high': high,
        'low': low,
        'vol': vol
    })

    signals = accumulation_distribution_breakout(df, lookback=10, breakout_threshold=1.0)

    # 应该有ADL突破信号
    assert signals.sum() > 0
    assert signals.dtype == bool
