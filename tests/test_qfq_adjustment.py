# -*- coding: utf-8 -*-
"""
前复权 (qfq) / 后复权 (hfq) 计算的回归测试。

重点验证 `_adjust_prices` 永远使用「真正的最新复权因子」（来自对整张
adj_factor 表的查询），而不是查询窗口里最后一行的因子——后者在查询历史区间
时会得到错误的复权价。
"""
import pandas as pd
import pytest

from tushare_db.reader import DataReader


@pytest.fixture
def reader():
    r = DataReader()  # 只读打开 tushare.db
    yield r
    r.close()


def _patch_latest_factor(reader, mapping):
    """
    让 `_adjust_prices` 内部「查询最新复权因子」的调用返回受控结果。

    在 `_adjust_prices` 的 qfq 分支中，`self.db.execute_query` 只会被调用一次
    （即 latest_adj_query），因此可以安全地整体替换。
    """
    def fake_query(query, params=None):
        codes = list(mapping.keys())
        return pd.DataFrame({'ts_code': codes,
                             'adj_factor': [mapping[c] for c in codes]})
    reader.db.execute_query = fake_query


def test_qfq_multi_stock_uses_true_latest_factor(reader):
    """多股票 qfq 必须用每只股票的真实最新因子，而不是窗口最后一行的因子。"""
    # 真实最新因子：A=2.0, B=4.0（与窗口最后一行的 1.1 / 2.0 不同）
    _patch_latest_factor(reader, {'A.SZ': 2.0, 'B.SZ': 4.0})
    df = pd.DataFrame({
        'ts_code': ['A.SZ', 'A.SZ', 'B.SZ'],
        'trade_date': ['20200101', '20200102', '20200101'],
        'close': [10.0, 11.0, 20.0],
        'open': [10.0, 11.0, 20.0],
        'adj_factor': [1.0, 1.1, 2.0],
    })
    out = reader._adjust_prices(df.copy(), 'qfq', ['A.SZ', 'B.SZ'])
    # qfq = raw * adj_factor / true_latest
    assert out.loc[0, 'close'] == pytest.approx(10.0 * 1.0 / 2.0)   # 5.00
    assert out.loc[1, 'close'] == pytest.approx(11.0 * 1.1 / 2.0)   # 6.05
    assert out.loc[2, 'close'] == pytest.approx(20.0 * 2.0 / 4.0)   # 10.0
    # open 列同样被复权
    assert out.loc[1, 'open'] == pytest.approx(11.0 * 1.1 / 2.0)


def test_qfq_single_stock_uses_true_latest_factor(reader):
    """单股票（df 无 ts_code 列）qfq 也必须用真实最新因子。"""
    _patch_latest_factor(reader, {'A.SZ': 2.0})
    df = pd.DataFrame({
        'trade_date': ['20200101', '20200102'],
        'close': [10.0, 11.0],
        'adj_factor': [1.0, 1.1],
    })
    out = reader._adjust_prices(df.copy(), 'qfq', 'A.SZ')
    assert out.loc[0, 'close'] == pytest.approx(10.0 * 1.0 / 2.0)
    assert out.loc[1, 'close'] == pytest.approx(11.0 * 1.1 / 2.0)


def test_qfq_single_stock_fallback_prefers_query_latest_not_window_last(reader):
    """
    当 latest_factors 查不到目标代码（key 不匹配）时，回退应使用查询返回的
    真实最新因子，而不是窗口最后一行的因子。
    """
    # 查询返回了一行，但 ts_code 与目标不一致，触发 .get() 回退
    _patch_latest_factor(reader, {'WRONG.SZ': 3.0})
    df = pd.DataFrame({
        'trade_date': ['20200101', '20200102'],
        'close': [10.0, 11.0],
        'adj_factor': [1.0, 1.1],
    })
    out = reader._adjust_prices(df.copy(), 'qfq', 'A.SZ')
    # 修复后：用查询的真实最新因子 3.0，而非窗口最后一行 1.1
    assert out.loc[1, 'close'] == pytest.approx(11.0 * 1.1 / 3.0)


def test_qfq_without_tscode_raises_in_strict_mode():
    """无法确定股票代码时，严格模式应抛错而不是静默返回可能错误的复权价。"""
    r = DataReader(strict_mode=True)
    try:
        df = pd.DataFrame({
            'trade_date': ['20200101', '20200102'],
            'close': [10.0, 11.0],
            'adj_factor': [1.0, 1.1],
        })
        with pytest.raises(Exception):
            r._adjust_prices(df, 'qfq', None)
    finally:
        r.close()


def test_hfq_multiplies_by_factor(reader):
    """后复权：价格 × 复权因子。"""
    df = pd.DataFrame({
        'trade_date': ['20200101'],
        'close': [10.0],
        'adj_factor': [2.0],
    })
    out = reader._adjust_prices(df.copy(), 'hfq', 'A.SZ')
    assert out.loc[0, 'close'] == pytest.approx(20.0)
