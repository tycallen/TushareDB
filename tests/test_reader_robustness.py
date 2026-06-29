# -*- coding: utf-8 -*-
"""
reader 健壮性小修的回归测试：
- holder_name LIKE 元字符转义
- get_sw_daily / get_index_daily 的 trade_date 与 start/end 互斥语义
- get_sw_daily_pivot 的 field 白名单 + 重复键容忍
"""
import pandas as pd
import pytest

from tushare_db.reader import DataReader, DataReaderError


@pytest.fixture
def reader():
    r = DataReader()  # 只读打开 tushare.db
    yield r
    r.close()


def _capture_query(reader):
    """把 reader.db.execute_query 替换为捕获 SQL/params 的桩，返回空结果。"""
    cap = {}

    def fake(query, params=None):
        cap["query"] = query
        cap["params"] = params
        return pd.DataFrame()

    reader.db.execute_query = fake
    return cap


def test_holder_name_like_is_escaped(reader):
    cap = _capture_query(reader)
    reader.get_top10_floatholders("000001.SZ", holder_name="中央_结算%")
    assert "ESCAPE '\\'" in cap["query"]
    joined = " ".join(str(p) for p in cap["params"])
    assert "\\_" in joined and "\\%" in joined  # 通配符已被转义


def test_sw_daily_trade_date_excludes_range(reader):
    cap = _capture_query(reader)
    reader.get_sw_daily(trade_date="20240101", start_date="20240101", end_date="20240131")
    assert "trade_date = ?" in cap["query"]
    assert ">=" not in cap["query"] and "<=" not in cap["query"]
    assert cap["params"] == ["20240101"]


def test_index_daily_trade_date_excludes_range(reader):
    cap = _capture_query(reader)
    reader.get_index_daily(trade_date="20240101", start_date="20240101")
    assert ">=" not in cap["query"]
    assert cap["params"] == ["20240101"]


def test_sw_daily_pivot_field_whitelist(reader):
    with pytest.raises(DataReaderError):
        reader.get_sw_daily_pivot("20240101", "20240131", field="close; DROP TABLE x")


def test_sw_daily_pivot_tolerates_duplicate_keys(reader):
    def fake(query, params=None):
        return pd.DataFrame({
            "trade_date": ["20240101", "20240101"],
            "name": ["农林牧渔", "农林牧渔"],   # 重复 (trade_date, name)
            "pct_change": [1.0, 2.0],
        })

    reader.db.execute_query = fake
    out = reader.get_sw_daily_pivot("20240101", "20240101", field="pct_change")
    assert not out.empty                       # 不再抛 ValueError
    assert out.loc["20240101", "农林牧渔"] == 2.0  # aggfunc='last'
