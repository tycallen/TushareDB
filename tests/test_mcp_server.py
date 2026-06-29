# -*- coding: utf-8 -*-
"""
MCP server 工具的单元测试。

只测不依赖 tushare.db 的逻辑：只读 SQL 校验、白名单、文本格式化，以及用
mock reader 验证 SELECT 路径。需要 mcp SDK；未装则跳过整个模块。
"""
import pandas as pd
import pytest

pytest.importorskip("mcp", reason="需要安装 mcp SDK: pip install -e '.[mcp]'")

from tushare_db import mcp_server as m  # noqa: E402


def test_run_sql_rejects_write_statements():
    for sql in ["DROP TABLE daily", "UPDATE daily SET close=0",
                "INSERT INTO daily VALUES (1)", "DELETE FROM daily"]:
        assert "只读" in m.run_sql(sql)


def test_run_sql_rejects_multi_statement():
    assert "多语句" in m.run_sql("SELECT 1; SELECT 2")


def test_get_financial_rejects_unknown_statement():
    assert "必须是" in m.get_financial("000001.SZ", statement="evil_table")


def test_describe_table_rejects_illegal_name():
    assert "非法表名" in m.describe_table("daily; DROP TABLE x")


def test_df_to_text_empty():
    assert m._df_to_text(pd.DataFrame()) == "（无数据）"


def test_df_to_text_truncates():
    out = m._df_to_text(pd.DataFrame({"a": range(300)}), max_rows=10)
    assert "共 300 行" in out


def test_run_sql_select_uses_reader(monkeypatch):
    class _FakeReader:
        def query(self, sql, params=None):
            assert sql.lower().startswith("select")
            return pd.DataFrame({"ts_code": ["000001.SZ"], "close": [9.2]})

    monkeypatch.setattr(m, "_get_reader", lambda: _FakeReader())
    out = m.run_sql("SELECT ts_code, close FROM daily LIMIT 1")
    assert "000001.SZ" in out and "9.2" in out


def test_get_financial_valid_statement_uses_reader(monkeypatch):
    captured = {}

    class _FakeReader:
        def query(self, sql, params=None):
            captured["sql"] = sql
            captured["params"] = params
            return pd.DataFrame({"end_date": ["20241231"], "roe": [12.3]})

    monkeypatch.setattr(m, "_get_reader", lambda: _FakeReader())
    out = m.get_financial("000001.SZ", statement="fina_indicator_vip", start_date="20240101")
    assert "FROM fina_indicator_vip" in captured["sql"]
    assert "000001.SZ" in captured["params"]
    assert "12.3" in out
