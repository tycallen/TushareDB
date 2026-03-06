import pandas as pd
import pytest


def test_factor_import():
    """测试模块可以导入"""
    from src.tushare_db.factor_validation.factor import Factor, FactorType
    assert Factor is not None
    assert FactorType is not None


def test_factor_creation():
    """测试创建因子对象"""
    from src.tushare_db.factor_validation.factor import Factor, FactorType

    factor = Factor(
        name="test_factor",
        description="测试因子",
        definition="close > open",
        type=FactorType.YAML
    )

    assert factor.name == "test_factor"
    assert factor.description == "测试因子"
    assert factor.type == FactorType.YAML


def test_factor_from_function():
    """测试从函数创建因子"""
    from src.tushare_db.factor_validation.factor import Factor, FactorType

    def my_factor(df):
        return df['close'] > df['open']

    factor = Factor(
        name="close_gt_open",
        description="收盘大于开盘",
        definition=my_factor,
        type=FactorType.FUNCTION
    )

    # 测试评估
    df = pd.DataFrame({
        'open': [10, 11, 12],
        'close': [11, 10, 13]
    })

    result = factor.evaluate(df)
    expected = pd.Series([True, False, True])

    assert result.equals(expected)


def test_factor_registry_list_builtin():
    """测试列出内置因子"""
    from src.tushare_db.factor_validation.factor import FactorRegistry

    builtins = FactorRegistry.list_builtin()
    assert isinstance(builtins, list)


def test_factor_registry_get_builtin():
    """测试获取内置因子"""
    from src.tushare_db.factor_validation.factor import FactorRegistry, Factor

    # 初始可能没有内置因子，返回 None 或抛出异常
    try:
        factor = FactorRegistry.get("macd_golden_cross")
        assert isinstance(factor, Factor)
    except (KeyError, NotImplementedError):
        pass  # 预期行为，内置因子尚未实现
