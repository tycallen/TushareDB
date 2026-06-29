# -*- coding: utf-8 -*-
"""
TushareFetcher 接入自定义代理（TUSHARE_API_URL）的回归测试。

全部不联网：mock 掉 DataApi.user，ts.pro_api/ts.set_token 本身不发网络请求。
"""
import tushare as ts
import pytest

from tushare_db.tushare_fetcher import TushareFetcher

_RATE_CFG = {"default": {"limit": 100, "period": "minute"}}


@pytest.fixture(autouse=True)
def _no_network_user(monkeypatch):
    """构造 TushareFetcher 时不要真的联网。

    tushare 的 pro.user()/pro.daily() 等都不是真实方法，而是经 DataApi.__getattr__
    动态分发到底层 query()，因此 mock query 即可拦截所有接口（含 user）。
    """
    monkeypatch.setattr(ts.pro.client.DataApi, "query",
                        lambda self, api_name='', fields='', **kwargs: None)


def test_proxy_url_applied_when_env_set(monkeypatch):
    monkeypatch.setenv("TUSHARE_API_URL", "https://fast.xiaodefa.cn/")  # 故意带尾斜杠
    f = TushareFetcher("dummytoken1234567890", _RATE_CFG)
    # 尾斜杠应被去掉，请求地址指向代理
    assert f.pro._DataApi__http_url == "https://fast.xiaodefa.cn"


def test_official_url_when_env_unset(monkeypatch):
    monkeypatch.delenv("TUSHARE_API_URL", raising=False)
    f = TushareFetcher("dummytoken1234567890", _RATE_CFG)
    # 未配置代理时保持官方默认地址
    url = f.pro._DataApi__http_url
    assert "waditu" in url or "tushare" in url


def test_init_survives_user_failure(monkeypatch):
    """user() 抛错（代理不支持/ token 仅对代理有效）不应阻断初始化。"""
    def boom(self, api_name='', fields='', **kwargs):
        raise RuntimeError("40101 token error")

    monkeypatch.setattr(ts.pro.client.DataApi, "query", boom)
    f = TushareFetcher("dummytoken1234567890", _RATE_CFG)
    assert f is not None
