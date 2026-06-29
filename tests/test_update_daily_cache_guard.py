# -*- coding: utf-8 -*-
"""
update_daily._advance_cache_if_progress 的回归测试：

不可用接口经优雅降级后返回 0 行；此时不应推进 cache 时间戳（否则配合周/月跳过守卫
会静默跳过该接口数天~数周）。但接口可用、本期genuinely 0 行时仍应正常推进（不破坏节流）。
"""
import importlib.util
from pathlib import Path

_PATH = Path(__file__).resolve().parent.parent / "scripts" / "update_daily.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("update_daily_mod_under_test", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _StubDB:
    def __init__(self):
        self.advanced = []

    def update_cache_metadata(self, key, ts):
        self.advanced.append(key)


class _StubFetcher:
    def __init__(self, unavailable):
        self._u = set(unavailable)

    @property
    def unavailable_apis(self):
        return set(self._u)


class _StubDownloader:
    def __init__(self, unavailable=()):
        self.fetcher = _StubFetcher(unavailable)
        self.db = _StubDB()


def test_no_advance_when_unavailable_and_zero_rows():
    mod = _load_module()
    dl = _StubDownloader(unavailable={"income_vip"})
    mod._advance_cache_if_progress(dl, "fin", 0, "income_vip", "cashflow_vip")
    assert dl.db.advanced == []  # 接口不可用且 0 行 → 不推进，下次重试


def test_advance_when_rows_downloaded():
    mod = _load_module()
    dl = _StubDownloader(unavailable={"income_vip"})
    mod._advance_cache_if_progress(dl, "fin", 5, "income_vip")
    assert dl.db.advanced == ["fin"]  # 有进展 → 推进


def test_advance_when_zero_rows_but_interface_available():
    mod = _load_module()
    dl = _StubDownloader(unavailable=set())
    mod._advance_cache_if_progress(dl, "fin", 0, "income_vip")
    assert dl.db.advanced == ["fin"]  # 0 行但接口可用(本期无新数据) → 照常推进，不破坏节流
