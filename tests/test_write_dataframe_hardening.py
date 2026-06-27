# -*- coding: utf-8 -*-
"""
write_dataframe 写入加固的回归测试：
- 显式列名插入（不依赖列顺序，避免列错位）
- replace 模式用事务包裹，失败时回滚、不丢原表
"""
import pandas as pd
import pytest

from tushare_db.duckdb_manager import DuckDBManager


@pytest.fixture
def db(tmp_path):
    mgr = DuckDBManager(str(tmp_path / "t.db"))
    yield mgr
    mgr.close()


def test_append_no_pk_respects_column_names_not_order(db):
    """无主键表 append：列顺序打乱时应按列名写入，而不是按位置错位。"""
    tbl = "my_nopk_table"  # 不在 TABLE_PRIMARY_KEYS 中
    db.write_dataframe(pd.DataFrame({"a": [1], "b": [2], "c": [3]}), tbl, mode="append")
    # 第二批：列顺序故意打乱
    db.write_dataframe(pd.DataFrame({"c": [30], "a": [10], "b": [20]}), tbl, mode="append")

    out = db.execute_query(f"SELECT a, b, c FROM {tbl} ORDER BY a")
    row = out[out["a"] == 10].iloc[0]
    # 按列名：a=10,b=20,c=30；若是旧的 SELECT * 会错位成 a=30,b=10,c=20
    assert row["b"] == 20 and row["c"] == 30


def test_append_with_pk_upsert(db):
    """有主键表 append：相同 PK 更新、不同 PK 插入。"""
    tbl = "daily"  # PK = (ts_code, trade_date)
    db.write_dataframe(
        pd.DataFrame({"ts_code": ["A"], "trade_date": ["20200101"], "close": [1.0]}),
        tbl, mode="append",
    )
    # 同 PK → 更新 close
    db.write_dataframe(
        pd.DataFrame({"ts_code": ["A"], "trade_date": ["20200101"], "close": [9.0]}),
        tbl, mode="append",
    )
    out = db.execute_query(f"SELECT close FROM {tbl} WHERE ts_code='A' AND trade_date='20200101'")
    assert len(out) == 1 and out.iloc[0]["close"] == 9.0


def test_replace_mode_rebuilds_table(db):
    """replace 模式：用新数据整体替换旧表。"""
    tbl = "my_repl"
    db.write_dataframe(pd.DataFrame({"a": [1, 2, 3]}), tbl, mode="replace")
    db.write_dataframe(pd.DataFrame({"a": [9]}), tbl, mode="replace")
    out = db.execute_query(f"SELECT * FROM {tbl}")
    assert len(out) == 1 and out.iloc[0]["a"] == 9


def test_replace_rollback_preserves_table_on_failure(db):
    """
    replace 在 INSERT 阶段失败时，事务回滚必须保留原表数据，
    而不是因为先 DROP 了表导致永久丢失。
    """
    tbl = "my_repl2"
    db.write_dataframe(pd.DataFrame({"a": [1, 2, 3]}), tbl, mode="replace")  # 原表 3 行

    # 注入故障：让本次 replace 的 INSERT 抛错（DROP/CREATE 正常执行）。
    # DuckDB 的 con.execute 是只读属性，无法直接替换，故用代理对象转发。
    real_con = db.con

    class _ConProxy:
        def execute(self, sql, *args, **kwargs):
            if sql.strip().upper().startswith("INSERT"):
                raise RuntimeError("simulated insert failure")
            return real_con.execute(sql, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(real_con, name)

    db.con = _ConProxy()
    try:
        with pytest.raises(Exception):
            db.write_dataframe(pd.DataFrame({"a": [9]}), tbl, mode="replace")
    finally:
        db.con = real_con

    # 回滚后原表应仍在且为 3 行
    out = db.execute_query(f"SELECT * FROM {tbl} ORDER BY a")
    assert len(out) == 3
    assert out["a"].tolist() == [1, 2, 3]
