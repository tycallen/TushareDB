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


def test_library_logger_redirected_to_stderr_not_stdout():
    """MCP stdio：库日志绝不能写 stdout（会破坏 JSON-RPC 协议），必须在 stderr。"""
    import logging
    import sys
    lg = logging.getLogger("tushare_db")
    streams = [h.stream for h in lg.handlers if isinstance(h, logging.StreamHandler)]
    assert streams, "tushare_db logger 应有 StreamHandler"
    assert all(s is sys.stderr for s in streams)
    assert all(s is not sys.stdout for s in streams)
    assert lg.propagate is False  # 不向 root 传播，避免 root 的 stdout handler 污染


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


def test_run_sql_allows_semicolon_inside_string_literal(monkeypatch):
    class _FakeReader:
        def query(self, sql, params=None):
            return pd.DataFrame({"sep": [";"]})
    monkeypatch.setattr(m, "_get_reader", lambda: _FakeReader())
    out = m.run_sql("SELECT ';' AS sep")
    assert "不支持多语句" not in out and ";" in out


def test_df_to_text_truncation_note_separated_from_csv():
    out = m._df_to_text(pd.DataFrame({"a": range(300)}), max_rows=10)
    assert out.startswith("（共 300 行")   # 提示在前
    assert "\n\na\n" in out                # CSV 表头 'a' 在空行分隔后，主体未被污染


def test_get_financial_rejects_empty_ts_code():
    assert "ts_code 不能为空" in m.get_financial("", statement="income")


def test_get_financial_rejects_bad_date_format():
    assert "YYYYMMDD" in m.get_financial("000001.SZ", statement="income", start_date="2024")


def test_live_fetch_uses_fetcher(monkeypatch):
    class _FakeFetcher:
        def fetch(self, api_name, **kwargs):
            assert api_name == "daily"
            return pd.DataFrame({"ts_code": ["000001.SZ"], "close": [9.2]})
    monkeypatch.setattr(m, "_get_fetcher", lambda: _FakeFetcher())
    out = m.live_fetch("daily", {"ts_code": "000001.SZ", "trade_date": "20240102"})
    assert "000001.SZ" in out and "9.2" in out


def test_live_fetch_rejects_empty_api_name():
    assert "api_name 不能为空" in m.live_fetch("", {})


def test_live_stock_daily_qfq_hfq_math(monkeypatch):
    import io
    daily = pd.DataFrame({
        "ts_code": ["A", "A"], "trade_date": ["20240102", "20240103"],
        "open": [10.0, 11.0], "high": [10.0, 11.0], "low": [10.0, 11.0],
        "close": [10.0, 11.0], "pre_close": [10.0, 10.0], "vol": [1, 1],
    })
    factors = pd.DataFrame({
        "ts_code": ["A", "A", "A"],
        "trade_date": ["20240102", "20240103", "20240201"],  # 最新=20240201 factor=2.0
        "adj_factor": [1.0, 1.1, 2.0],
    })

    class _F:
        def fetch(self, api, **kw):
            return daily.copy() if api == "daily" else factors.copy()

    monkeypatch.setattr(m, "_get_fetcher", lambda: _F())

    # qfq: close × (factor/最新2.0)；20240102:10×0.5=5.0；20240103:11×1.1/2=6.05
    df = pd.read_csv(io.StringIO(m.live_stock_daily("A", "20240102", "20240103", "qfq")),
                     dtype={"trade_date": str})
    assert "adj_factor" not in df.columns  # 中间列已清掉
    assert abs(df.loc[df.trade_date == "20240102", "close"].iloc[0] - 5.0) < 1e-6
    assert abs(df.loc[df.trade_date == "20240103", "close"].iloc[0] - 6.05) < 1e-6

    # hfq: close × factor；20240103:11×1.1=12.1
    df2 = pd.read_csv(io.StringIO(m.live_stock_daily("A", "20240102", "20240103", "hfq")),
                      dtype={"trade_date": str})
    assert abs(df2.loc[df2.trade_date == "20240103", "close"].iloc[0] - 12.1) < 1e-6


def test_live_stock_daily_rejects_bad_adj():
    assert "adj 只能是" in m.live_stock_daily("A", "20240102", "20240103", "xfq")


def test_live_limit_list_rejects_bad_date():
    assert "YYYYMMDD" in m.live_limit_list("2024")


def test_live_top_list_rejects_bad_date():
    assert "YYYYMMDD" in m.live_top_list("not-a-date")


def test_live_moneyflow_validation_and_delegation(monkeypatch):
    assert "ts_code 不能为空" in m.live_moneyflow("")
    assert "YYYYMMDD" in m.live_moneyflow("000001.SZ", "2024")

    cap = {}

    class _F:
        def fetch(self, api, **kw):
            cap["api"], cap["kw"] = api, kw
            return pd.DataFrame({"net_mf_amount": [123.0]})

    monkeypatch.setattr(m, "_get_fetcher", lambda: _F())
    out = m.live_moneyflow("000001.SZ", "20240102")
    assert cap["api"] == "moneyflow"
    assert cap["kw"] == {"ts_code": "000001.SZ", "trade_date": "20240102"}
    assert "123.0" in out


def test_get_concept_stocks_validation_and_delegation(monkeypatch):
    assert "YYYYMMDD" in m.get_concept_stocks("2024", "AI")

    cap = {}

    class _R:
        def get_concept_stocks(self, trade_date, concept_name=None):
            cap["d"], cap["c"] = trade_date, concept_name
            return pd.DataFrame({"ts_code": ["000001.SZ"]})

    monkeypatch.setattr(m, "_get_reader", lambda: _R())
    out = m.get_concept_stocks("20240115", "人工智能")
    assert cap["d"] == "20240115" and cap["c"] == "人工智能"
    assert "000001.SZ" in out


def test_get_stock_concepts_rejects_empty_ts_code():
    assert "ts_code 不能为空" in m.get_stock_concepts("20240115", "")


def test_load_env_from_db_dir(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("TUSHARE_TOKEN='abc123'\nTUSHARE_API_URL='https://x'\n")
    (tmp_path / "tushare.db").write_text("")  # 仅用其目录
    monkeypatch.setenv("DB_PATH", str(tmp_path / "tushare.db"))
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
    m._load_env_from_db_dir()
    import os
    assert os.environ.get("TUSHARE_TOKEN") == "abc123"


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
