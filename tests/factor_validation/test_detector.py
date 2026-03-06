import numpy as np
import pandas as pd
import pytest


def test_detector_import():
    """测试模块可以导入"""
    from src.tushare_db.factor_validation.detector import SignalDetector
    assert SignalDetector is not None


def test_detector_init():
    """测试初始化"""
    from src.tushare_db.factor_validation.detector import SignalDetector

    detector = SignalDetector()
    assert detector.vectorized is True


def test_macd_golden_cross_detection():
    """测试 MACD 金叉检测"""
    from src.tushare_db.factor_validation.detector import SignalDetector

    detector = SignalDetector()

    # 创建一个价格序列（100条路径，50天）
    np.random.seed(42)
    price_matrix = np.cumsum(np.random.randn(100, 50) * 0.01, axis=1) + 100

    # 检测 MACD 金叉
    signals = detector.macd_golden_cross(price_matrix)

    # 检查形状
    assert signals.shape == (100, 50)
    # 检查是布尔类型
    assert signals.dtype == bool
    # 检查至少有一些信号（随机数据应该有约 5% 的交叉）
    assert signals.sum() > 0


def test_calculate_p_random():
    """测试计算随机概率"""
    from src.tushare_db.factor_validation.detector import SignalDetector

    detector = SignalDetector()

    # 创建信号矩阵
    signal_matrix = np.array([
        [False, True, False],
        [True, False, True],
        [False, False, False]
    ])

    p_random = detector.calculate_p_random(signal_matrix)

    # 3个 True / 9个总位置 = 0.333
    assert abs(p_random - 3/9) < 1e-6


def test_detect_signals_with_simple_factor():
    """测试使用简单因子检测信号"""
    from src.tushare_db.factor_validation.detector import SignalDetector
    from src.tushare_db.factor_validation.factor import Factor, FactorType

    detector = SignalDetector()

    # 创建一个简单因子（价格上涨）
    def rising_factor(df):
        return df['close'] > df['close'].shift(1)

    factor = Factor(
        name="rising",
        description="价格上涨",
        definition=rising_factor,
        type=FactorType.FUNCTION
    )

    # 创建价格矩阵并转换为 DataFrame
    np.random.seed(42)
    price_matrix = np.cumsum(np.random.randn(10, 20) * 0.01, axis=1) + 100

    # 检测信号
    signals = detector.detect_signals(price_matrix, factor)

    assert signals.shape == (10, 20)
    assert signals.dtype == bool


def test_detector_with_builtin_factor():
    """测试使用内置因子"""
    from src.tushare_db.factor_validation.detector import SignalDetector
    from src.tushare_db.factor_validation.factor import FactorRegistry

    detector = SignalDetector()

    # 获取内置因子
    factor = FactorRegistry.get("close_gt_open")

    # 创建价格矩阵（模拟 open = close * 0.99）
    np.random.seed(42)
    close = np.cumsum(np.random.randn(10, 20) * 0.01, axis=1) + 100
    open_price = close * 0.99

    # 检测信号
    signals = detector.detect_signals(close, factor, open_price=open_price)

    assert signals.shape == (10, 20)
    # 因为 close > open (close > close*0.99)，应该全部为 True
    assert signals.sum() > 0
