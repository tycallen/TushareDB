# -*- coding: utf-8 -*-
"""
TushareFetcher 对「接口不可用」的自适应优雅降级测试。

不联网：mock 掉底层 DataApi.query，根据 api_name 返回不同结果/抛不同错误。
验证：无权限/接口不存在 → 记录并跳过（默认返空）；token 错误不被误判；
缓存命中后不再请求；raise_on_unavailable=True 时改为抛异常。
"""
import pandas as pd
import tushare as ts
import pytest

from tushare_db.tushare_fetcher import (
    TushareFetcher,
    TushareUnavailableError,
    TushareClientError,
)

CFG = {"default": {"limit": 1000, "period": "minute"}}


def _fake_query(self, api_name, fields="", **kw):
    if api_name == "no_perm":
        raise Exception("抱歉，您没有访问该接口的权限")
    if api_name == "missing":
        raise Exception("接口不存在")
    if api_name == "bad_token":
        raise Exception("您的token不对，请确认。")
    if api_name == "good":
        return pd.DataFrame({"ts_code": ["x"], "v": [1]})
    return pd.DataFrame()  # user() 及其它


@pytest.fixture
def fetcher(monkeypatch):
    monkeypatch.setattr(ts.pro.client.DataApi, "query", _fake_query)
    return TushareFetcher("dummytoken1234567890", CFG)


def test_no_permission_returns_empty_and_is_cached(fetcher):
    df = fetcher.fetch("no_perm", ts_code="x")
    assert df.empty
    assert "no_perm" in fetcher.unavailable_apis


def test_missing_interface_also_unavailable(fetcher):
    df = fetcher.fetch("missing")
    assert df.empty
    assert "missing" in fetcher.unavailable_apis


def test_cached_unavailable_skips_request(fetcher, monkeypatch):
    fetcher.fetch("no_perm")  # 首次：识别并缓存
    calls = {"n": 0}

    def counting(self, api_name, fields="", **kw):
        calls["n"] += 1
        return _fake_query(self, api_name, fields, **kw)

    monkeypatch.setattr(ts.pro.client.DataApi, "query", counting)
    df = fetcher.fetch("no_perm")          # 第二次：应直接跳过，不再请求
    assert df.empty and calls["n"] == 0


def test_raise_on_unavailable_flag(fetcher):
    with pytest.raises(TushareUnavailableError):
        fetcher.fetch("no_perm", raise_on_unavailable=True)


def test_token_error_is_not_treated_as_unavailable(fetcher):
    with pytest.raises(TushareClientError) as ei:
        fetcher.fetch("bad_token")
    assert not isinstance(ei.value, TushareUnavailableError)
    assert "bad_token" not in fetcher.unavailable_apis


def test_available_api_still_works(fetcher):
    df = fetcher.fetch("good")
    assert not df.empty and list(df["ts_code"]) == ["x"]
