# -*- coding: utf-8 -*-
"""
Tushare-DuckDB MCP Server —— 通过 Model Context Protocol 把本地 DuckDB 的
历史/大数据查询能力暴露给 Claude Desktop / Claude Code。

定位：
- 本 MCP 负责【历史、批量、大数据】，数据来自本地 tushare.db（只读）。
- 【实时、小数据】请用配套的 `tushare-live` skill，通过 token 直接查询代理 API。

运行：
    python -m tushare_db.mcp_server

Claude Desktop 配置（claude_desktop_config.json）：
    {
      "mcpServers": {
        "tushare-duckdb": {
          "command": "python",
          "args": ["-m", "tushare_db.mcp_server"],
          "env": {"DB_PATH": "/绝对路径/tushare.db"}
        }
      }
    }
"""
import re

from mcp.server.fastmcp import FastMCP

from .reader import DataReader

mcp = FastMCP("tushare-duckdb")

_reader = None
# 财务相关表白名单（供 get_financial 使用，防止表名注入）
_FINANCIAL_TABLES = {
    "income", "balancesheet", "cashflow",
    "fina_indicator_vip", "forecast", "express",
}
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _get_reader() -> DataReader:
    """惰性创建只读 DataReader（DB_PATH 环境变量或默认 tushare.db）。"""
    global _reader
    if _reader is None:
        _reader = DataReader()
    return _reader


def _df_to_text(df, max_rows: int = 200) -> str:
    """把查询结果格式化为紧凑 CSV 文本（无额外依赖，Claude 易解析）。"""
    if df is None or df.empty:
        return "（无数据）"
    truncated = len(df) > max_rows
    body = df.head(max_rows).to_csv(index=False)
    if truncated:
        body += f"\n# 共 {len(df)} 行，仅显示前 {max_rows} 行（如需更多请在 SQL 中聚合或缩小范围）"
    return body


@mcp.tool()
def list_tables() -> str:
    """列出本地 DuckDB 中所有可查询的表名（历史数据全集）。"""
    df = _get_reader().query(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'main' ORDER BY table_name"
    )
    return "\n".join(df["table_name"].tolist()) if not df.empty else "（无表）"


@mcp.tool()
def describe_table(table_name: str) -> str:
    """返回某张表的列清单。先用 list_tables 查看可用表名。"""
    if not _IDENT_RE.match(table_name):
        return "错误：非法表名。"
    r = _get_reader()
    exists = r.query(
        "SELECT 1 FROM information_schema.tables WHERE table_name = ?", [table_name]
    )
    if exists.empty:
        return f"表 {table_name} 不存在（用 list_tables 查看可用表）。"
    cols = r.db.get_table_columns(table_name)
    return f"表 {table_name} 的列（{len(cols)}）:\n" + ", ".join(cols)


@mcp.tool()
def run_sql(sql: str, max_rows: int = 500) -> str:
    """对本地历史数据执行只读 SQL（仅允许 SELECT / WITH / PRAGMA，单条语句）。

    这是最通用的工具，适合历史、批量、聚合分析。示例：
        SELECT trade_date, close FROM daily
        WHERE ts_code='000001.SZ' ORDER BY trade_date DESC LIMIT 30
    """
    s = sql.strip().rstrip(";").strip()
    low = s.lower()
    if not (low.startswith("select") or low.startswith("with") or low.startswith("pragma")):
        return "错误：仅允许只读查询（SELECT / WITH / PRAGMA）。"
    if ";" in s:
        return "错误：不支持多语句查询。"
    try:
        df = _get_reader().query(s)
    except Exception as e:
        return f"查询出错：{e}"
    return _df_to_text(df, max_rows=max_rows)


@mcp.tool()
def get_stock_daily(ts_code: str, start_date: str, end_date: str, adj: str = "") -> str:
    """查询个股日线行情（历史）。

    Args:
        ts_code: 股票代码，如 000001.SZ
        start_date / end_date: YYYYMMDD
        adj: 复权方式，''=不复权，'qfq'=前复权，'hfq'=后复权
    """
    try:
        df = _get_reader().get_stock_daily(
            ts_code, start_date, end_date, adj=(adj or None)
        )
    except Exception as e:
        return f"查询出错：{e}"
    return _df_to_text(df)


@mcp.tool()
def get_financial(ts_code: str, statement: str = "fina_indicator_vip",
                  start_date: str = "", end_date: str = "") -> str:
    """查询个股财务数据（历史）。

    Args:
        ts_code: 股票代码
        statement: income / balancesheet / cashflow / fina_indicator_vip / forecast / express
        start_date / end_date: 可选，按 end_date(报告期) YYYYMMDD 过滤
    """
    if statement not in _FINANCIAL_TABLES:
        return f"错误：statement 必须是 {sorted(_FINANCIAL_TABLES)} 之一。"
    conditions = ["ts_code = ?"]
    params = [ts_code]
    if start_date:
        conditions.append("end_date >= ?")
        params.append(start_date)
    if end_date:
        conditions.append("end_date <= ?")
        params.append(end_date)
    sql = f"SELECT * FROM {statement} WHERE " + " AND ".join(conditions) + " ORDER BY end_date DESC"
    try:
        df = _get_reader().query(sql, params)
    except Exception as e:
        return f"查询出错：{e}"
    return _df_to_text(df)


def main():
    """以 stdio 方式运行 MCP server（Claude Desktop / Code 默认接入方式）。"""
    mcp.run()


if __name__ == "__main__":
    main()
