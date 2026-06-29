# -*- coding: utf-8 -*-
"""
业绩预告(forecast_vip) / 业绩快报(express_vip) 下载方法的单元测试。

不联网、不建库：用 __new__ 绕过 DataDownloader.__init__，注入 stub 的
fetcher 与 db，只验证「取数 -> 写正确的表 -> 返回行数」的下载逻辑。
"""
import pandas as pd

from tushare_db.downloader import DataDownloader
from tushare_db.duckdb_manager import TABLE_PRIMARY_KEYS


class _StubDB:
    def __init__(self):
        self.calls = []

    def write_dataframe(self, df, table_name, mode="append"):
        self.calls.append((table_name, len(df), mode))


class _StubFetcher:
    def __init__(self, mapping):
        self.mapping = mapping

    def fetch(self, api_name, **kwargs):
        return self.mapping.get(api_name, pd.DataFrame())


def _make_downloader(mapping):
    dl = DataDownloader.__new__(DataDownloader)  # 跳过 __init__（不联网/不需 token）
    dl.fetcher = _StubFetcher(mapping)
    dl.db = _StubDB()
    return dl


def test_pk_registered():
    assert TABLE_PRIMARY_KEYS["forecast"] == ["ts_code", "ann_date", "end_date", "type"]
    assert TABLE_PRIMARY_KEYS["express"] == ["ts_code", "ann_date", "end_date"]


def test_download_forecast_vip_writes_forecast_table():
    fake = pd.DataFrame({
        "ts_code": ["A.SZ"], "ann_date": ["20250401"],
        "end_date": ["20250331"], "type": ["预增"],
    })
    dl = _make_downloader({"forecast_vip": fake})
    n = dl.download_forecast_vip("20250331")
    assert n == 1
    assert dl.db.calls == [("forecast", 1, "append")]


def test_download_express_vip_writes_express_table():
    fake = pd.DataFrame({
        "ts_code": ["A.SZ"], "ann_date": ["20250401"],
        "end_date": ["20250331"], "revenue": [100.0],
    })
    dl = _make_downloader({"express_vip": fake})
    n = dl.download_express_vip("20250331")
    assert n == 1
    assert dl.db.calls == [("express", 1, "append")]


def test_empty_response_writes_nothing():
    # fetch 返回空（如代理无权限时 fetcher 优雅返空）→ 不写表、返回 0
    dl = _make_downloader({})
    assert dl.download_forecast_vip("20250331") == 0
    assert dl.download_express_vip("20250331") == 0
    assert dl.db.calls == []


def test_drops_rows_with_null_pk_to_avoid_whole_batch_failure():
    # 主键列含 NULL/空串的坏行应被丢弃（仅丢坏行），避免整批 INSERT 因 NOT NULL 约束失败
    fake = pd.DataFrame({
        "ts_code": ["A.SZ", "B.SZ", "C.SZ"],
        "ann_date": ["20250401", None, "20250402"],   # B 行 ann_date 为 NULL
        "end_date": ["20250331", "20250331", "20250331"],
        "type": ["预增", "预增", ""],                   # C 行 type 为空串
    })
    dl = _make_downloader({"forecast_vip": fake})
    n = dl.download_forecast_vip("20250331")
    assert n == 1                                      # 仅 A 行有效
    assert dl.db.calls == [("forecast", 1, "append")]


def test_all_rows_null_pk_writes_nothing():
    fake = pd.DataFrame({
        "ts_code": ["A.SZ"], "ann_date": [None],
        "end_date": ["20250331"], "type": ["预增"],
    })
    dl = _make_downloader({"forecast_vip": fake})
    assert dl.download_forecast_vip("20250331") == 0
    assert dl.db.calls == []
