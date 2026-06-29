# -*- coding: utf-8 -*-
"""
限速配置 profile 的回归测试，重点验证第三方代理档（proxy）。
"""
import pytest

from tushare_db.rate_limit_config import (
    PRESET_PROFILES,
    PROXY_PROFILE,
    PROXY_MINUTE_LIMIT,
)


def test_proxy_profile_registered():
    assert "proxy" in PRESET_PROFILES
    assert PRESET_PROFILES["proxy"] is PROXY_PROFILE


def test_proxy_profile_has_valid_default():
    assert "default" in PROXY_PROFILE
    d = PROXY_PROFILE["default"]
    assert d["period"] in ("minute", "day")
    assert isinstance(d["limit"], int) and d["limit"] > 0


def test_proxy_profile_is_conservative():
    # 默认每分钟限制应保守（≤120，对应请求间隔 ≥0.5s），避免触发代理冷却
    assert PROXY_PROFILE["default"]["limit"] <= 120
    assert PROXY_PROFILE["default"]["limit"] == PROXY_MINUTE_LIMIT


def test_stk_mins_more_conservative_than_default():
    # 历史分钟数据量大，限制应不高于默认档
    assert PROXY_PROFILE["stk_mins"]["limit"] <= PROXY_PROFILE["default"]["limit"]


@pytest.mark.parametrize("name", ["trial", "standard", "pro", "proxy"])
def test_all_profiles_have_required_shape(name):
    """每个档位都必须有合法的 default（TushareFetcher 初始化的硬性要求）。"""
    profile = PRESET_PROFILES[name]
    assert "default" in profile
    for api, cfg in profile.items():
        assert "limit" in cfg and "period" in cfg
        assert cfg["period"] in ("minute", "day")
        assert cfg["limit"] > 0
