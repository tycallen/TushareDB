"""
测试SectorAnalyzer
"""

import pytest
import pandas as pd
import numpy as np
from tushare_db.sector_analysis import SectorAnalyzer


@pytest.fixture
def analyzer():
    """创建测试用的Analyzer"""
    ana = SectorAnalyzer('tushare.db')
    yield ana
    ana.close()


def test_calculate_sector_returns(analyzer):
    """测试板块涨跌幅计算（通过Analyzer调用）"""
    df = analyzer.calculate_sector_returns(
        start_date='20240102',
        end_date='20240105',
        level='L1',
        period='daily'
    )

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert 'sector_code' in df.columns
    assert 'return' in df.columns
    print(f"✓ 计算了 {len(df)} 条板块涨跌数据")


def test_calculate_correlation_matrix(analyzer):
    """测试相关性矩阵计算"""
    corr_matrix = analyzer.calculate_correlation_matrix(
        start_date='20240101',
        end_date='20240131',
        level='L1',
        period='daily'
    )

    assert isinstance(corr_matrix, pd.DataFrame)
    assert corr_matrix.shape[0] == corr_matrix.shape[1]  # 方阵
    assert corr_matrix.shape[0] > 0

    # 检查对角线为1（自相关）
    diagonal = pd.Series([corr_matrix.iloc[i, i] for i in range(len(corr_matrix))])
    assert (diagonal - 1.0).abs().max() < 0.01

    # 检查对称性
    assert (corr_matrix - corr_matrix.T).abs().max().max() < 0.01

    # 检查数值范围 [-1, 1]
    assert corr_matrix.min().min() >= -1
    assert corr_matrix.max().max() <= 1

    print(f"✓ 相关性矩阵: {corr_matrix.shape[0]}×{corr_matrix.shape[1]}")

    # 获取非对角线元素的最大值
    mask = ~(pd.DataFrame(True, index=corr_matrix.index, columns=corr_matrix.columns).values == pd.DataFrame(True, index=corr_matrix.index, columns=corr_matrix.columns).T.values)
    np.fill_diagonal(mask, False)
    non_diag_values = corr_matrix.values[mask]
    if len(non_diag_values) > 0:
        print(f"✓ 最大相关系数（非对角线）: {non_diag_values.max():.3f}")
        print(f"✓ 最小相关系数（非对角线）: {non_diag_values.min():.3f}")


def test_calculate_correlation_with_pvalue(analyzer):
    """测试带p值的相关性计算"""
    df = analyzer.calculate_correlation_with_pvalue(
        start_date='20240101',
        end_date='20240131',
        level='L1',
        period='daily',
        min_correlation=0.5  # 只看相关性>0.5的
    )

    assert isinstance(df, pd.DataFrame)

    if len(df) > 0:
        assert 'sector_a' in df.columns
        assert 'sector_b' in df.columns
        assert 'correlation' in df.columns
        assert 'p_value' in df.columns
        assert 'sample_size' in df.columns

        # 检查相关系数范围
        assert df['correlation'].abs().min() >= 0.5
        assert df['correlation'].abs().max() <= 1

        # 检查p值范围
        assert (df['p_value'] >= 0).all()
        assert (df['p_value'] <= 1).all()

        print(f"✓ 找到 {len(df)} 对高相关板块")
        print("\n相关性最强的前5对:")
        print(df.head().to_string(index=False))
    else:
        print("✓ 未找到相关性>0.5的板块对（数据周期较短）")


def test_correlation_weekly(analyzer):
    """测试周线相关性"""
    corr_matrix = analyzer.calculate_correlation_matrix(
        start_date='20240101',
        end_date='20240331',
        level='L1',
        period='weekly'
    )

    assert isinstance(corr_matrix, pd.DataFrame)
    assert corr_matrix.shape[0] > 0
    print(f"✓ 周线相关性矩阵: {corr_matrix.shape}")


def test_correlation_monthly(analyzer):
    """测试月线相关性"""
    corr_matrix = analyzer.calculate_correlation_matrix(
        start_date='20240101',
        end_date='20241231',
        level='L1',
        period='monthly'
    )

    assert isinstance(corr_matrix, pd.DataFrame)
    assert corr_matrix.shape[0] > 0
    print(f"✓ 月线相关性矩阵: {corr_matrix.shape}")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
