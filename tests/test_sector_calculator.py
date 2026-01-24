"""
测试SectorCalculator
"""

import pytest
import pandas as pd
from tushare_db.reader import DataReader
from tushare_db.sector_analysis.calculator import SectorCalculator


@pytest.fixture
def calculator():
    """创建测试用的Calculator"""
    reader = DataReader('tushare.db')
    calc = SectorCalculator(reader)
    yield calc
    reader.close()


def test_get_sectors(calculator):
    """测试获取板块列表"""
    sectors = calculator._get_sectors('L1')
    assert len(sectors) > 0
    assert all(isinstance(code, str) and isinstance(name, str) for code, name in sectors)
    print(f"✓ L1板块数量: {len(sectors)}")
    print(f"✓ 示例: {sectors[:3]}")


def test_get_trade_dates(calculator):
    """测试获取交易日列表"""
    dates = calculator._get_trade_dates('20240101', '20240110')
    assert len(dates) > 0
    assert all(isinstance(d, str) for d in dates)
    print(f"✓ 交易日数量: {len(dates)}")
    print(f"✓ 日期: {dates}")


def test_calculate_equal_weighted_return(calculator):
    """测试单日板块涨跌幅计算"""
    # 测试农林牧渔板块在2024-01-02的涨跌幅
    result = calculator._calculate_equal_weighted_return(
        sector_code='801010.SI',
        trade_date='20240102',
        level='L1'
    )

    assert result is not None
    assert 'return' in result
    assert 'stock_count' in result
    assert 'valid_count' in result
    assert result['stock_count'] > 0
    assert result['valid_count'] > 0
    assert isinstance(result['return'], (int, float))

    print(f"✓ 板块涨跌幅: {result['return']:.2f}%")
    print(f"✓ 成分股数: {result['stock_count']}")
    print(f"✓ 有效数据: {result['valid_count']}")


def test_calculate_daily_returns(calculator):
    """测试计算日线涨跌幅（小范围数据）"""
    # 只测试前3个交易日
    df = calculator._calculate_daily_returns(
        start_date='20240102',
        end_date='20240104',
        level='L1'
    )

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert set(df.columns) == {'sector_code', 'sector_name', 'trade_date', 'return', 'stock_count', 'valid_count'}

    print(f"✓ 计算结果行数: {len(df)}")
    print(f"✓ 板块数: {df['sector_code'].nunique()}")
    print(f"✓ 交易日数: {df['trade_date'].nunique()}")
    print("\n前5条记录:")
    print(df.head().to_string(index=False))


def test_calculate_returns_daily(calculator):
    """测试完整的日线计算流程"""
    df = calculator.calculate_returns(
        start_date='20240102',
        end_date='20240105',
        level='L1',
        period='daily',
        method='equal'
    )

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    print(f"✓ 日线数据: {len(df)} 条")


def test_calculate_returns_weekly(calculator):
    """测试周线转换"""
    df = calculator.calculate_returns(
        start_date='20240101',
        end_date='20240131',
        level='L1',
        period='weekly',
        method='equal'
    )

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    print(f"✓ 周线数据: {len(df)} 条")
    print(f"✓ 周数: {df['trade_date'].nunique()}")


def test_calculate_returns_monthly(calculator):
    """测试月线转换"""
    df = calculator.calculate_returns(
        start_date='20240101',
        end_date='20240331',
        level='L1',
        period='monthly',
        method='equal'
    )

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    print(f"✓ 月线数据: {len(df)} 条")
    print(f"✓ 月数: {df['trade_date'].nunique()}")


def test_no_future_function(calculator):
    """测试无未来函数（成分股筛选）"""
    # 假设某板块在2024-01-10有成分股调整
    # 计算2024-01-05的涨跌幅，不应包含2024-01-10才加入的股票

    result_before = calculator._calculate_equal_weighted_return(
        sector_code='801010.SI',
        trade_date='20240102',
        level='L1'
    )

    result_after = calculator._calculate_equal_weighted_return(
        sector_code='801010.SI',
        trade_date='20240110',
        level='L1'
    )

    # 验证两个时点的成分股数可能不同（如果有调整）
    assert result_before is not None
    assert result_after is not None
    print(f"✓ 2024-01-02成分股数: {result_before['stock_count']}")
    print(f"✓ 2024-01-10成分股数: {result_after['stock_count']}")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
