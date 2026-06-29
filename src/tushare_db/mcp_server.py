# -*- coding: utf-8 -*-
"""
Tushare-DuckDB MCP Server —— 通过 Model Context Protocol 把本地 DuckDB 的
历史/大数据查询能力暴露给 Claude Desktop / Claude Code。

定位：
- 历史/批量/大数据：run_sql / get_stock_daily / get_financial，查本地 tushare.db（只读）。
- 实时/小数据：live_fetch，经 token 直连 Tushare（或所配代理）拿最新数据。
  token 由 MCP 在宿主机侧从项目 .env（DB_PATH 同目录）读取，不进对话、不暴露给沙箱。
  这样 Claude Desktop（沙箱内）无需自身联网/持有 token，调 MCP 即可拿实时数据。

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
import logging
import os
import re
import sys
import threading
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from .reader import DataReader

# MCP 走 stdio 传输：stdout 是 JSON-RPC 通道，任何写入 stdout 的日志都会破坏协议。
# 库的 'tushare_db' 命名 logger 默认写 stdout（见 logger.py），这里把它重定向到
# stderr（MCP 约定 stderr 用于诊断日志），否则首个触发日志的工具调用会污染响应流。
_lib_logger = logging.getLogger("tushare_db")
for _h in list(_lib_logger.handlers):
    _lib_logger.removeHandler(_h)
_stderr_handler = logging.StreamHandler(sys.stderr)
_stderr_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
_lib_logger.addHandler(_stderr_handler)
_lib_logger.propagate = False

mcp = FastMCP("tushare-duckdb")

_reader = None
_reader_lock = threading.Lock()
_fetcher = None
_fetcher_lock = threading.Lock()
# 财务相关表白名单（供 get_financial 使用，防止表名注入）
_FINANCIAL_TABLES = {
    "income", "balancesheet", "cashflow",
    "fina_indicator_vip", "forecast", "express",
}
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# 匹配 SQL 字符串字面量（单/双引号，含 '' 转义），用于多语句检测前剔除字面量
_STRING_LITERAL_RE = re.compile(r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"")
# 复权作用的价格列（与 reader._adjust_prices 保持一致）
_PRICE_COLS = ["open", "high", "low", "close", "pre_close"]


def _get_reader() -> DataReader:
    """惰性创建只读 DataReader（DB_PATH 环境变量或默认 tushare.db），线程安全。"""
    global _reader
    if _reader is None:
        with _reader_lock:
            if _reader is None:  # double-checked locking
                _reader = DataReader()
    return _reader


def _load_env_from_db_dir() -> None:
    """从 DB_PATH 所在目录（项目根）加载 .env，使 MCP 在宿主机上拿到
    TUSHARE_TOKEN / TUSHARE_API_URL（实时查询用）。不覆盖已存在的环境变量。"""
    db = os.getenv("DB_PATH")
    if not db:
        return
    env_path = Path(db).resolve().parent / ".env"
    if not env_path.exists():
        return
    for _ln in env_path.read_text(encoding="utf-8").splitlines():
        _s = _ln.strip()
        if _s and not _s.startswith("#") and "=" in _s:
            _k, _v = _s.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip().strip("'").strip('"'))


def _get_fetcher():
    """惰性创建 TushareFetcher（实时查询，token 直连代理 API），线程安全。

    token 从环境变量或项目 .env（DB_PATH 同目录）读取——这样 token 始终留在宿主机
    侧，不暴露给对话/沙箱。"""
    global _fetcher
    if _fetcher is None:
        with _fetcher_lock:
            if _fetcher is None:
                _load_env_from_db_dir()
                token = os.getenv("TUSHARE_TOKEN")
                if not token:
                    raise RuntimeError(
                        "未配置 TUSHARE_TOKEN（请在项目 .env 或 MCP env 中设置）"
                    )
                from .tushare_fetcher import TushareFetcher
                from .rate_limit_config import PROXY_PROFILE
                _fetcher = TushareFetcher(token, PROXY_PROFILE)
    return _fetcher


def _df_to_text(df, max_rows: int = 200) -> str:
    """把查询结果格式化为紧凑 CSV 文本（无额外依赖，Claude 易解析）。

    截断提示放在 CSV 之前作为独立说明行，避免污染 CSV 主体——若追加到末尾，提示行
    会被按 CSV 解析的下游当成一条畸形数据记录。
    """
    if df is None or df.empty:
        return "（无数据）"
    csv_body = df.head(max_rows).to_csv(index=False)
    if len(df) > max_rows:
        note = (f"（共 {len(df)} 行，仅显示前 {max_rows} 行；"
                f"如需更多请在 SQL 中聚合或缩小范围）\n\n")
        return note + csv_body
    return csv_body


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
    # 多语句检测：先剔除字符串字面量，避免把字面量里的分号误判为语句分隔符
    if ";" in _STRING_LITERAL_RE.sub("", s):
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
    if not ts_code or not ts_code.strip():
        return "错误：ts_code 不能为空。"
    for _d in (start_date, end_date):
        if _d and not (len(_d) == 8 and _d.isdigit()):
            return f"错误：日期需为 YYYYMMDD 格式：{_d}"
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


@mcp.tool()
def live_fetch(api_name: str, params: dict | None = None) -> str:
    """实时查询 Tushare 接口（经 token 直连 API，适合最新/小数据，数据不落库）。

    与本地库工具（run_sql/get_stock_daily 查历史）互补：这里直接打 Tushare（或所配
    第三方代理）拿最新数据。token 由 MCP 在宿主机侧从 .env 读取，不进对话、不暴露。

    Args:
        api_name: Tushare 接口名，如 'daily','daily_basic','moneyflow','top_list',
            'limit_list_d','stk_limit','index_daily' 等。
        params: 接口参数 dict，如 {"ts_code":"000001.SZ","trade_date":"20260627"}。

    Returns:
        CSV 文本结果（无权限/不存在的接口会返回空，并在服务端日志提示）。

    示例：
        live_fetch("daily", {"ts_code":"000001.SZ","trade_date":"20260627"})
        live_fetch("limit_list_d", {"trade_date":"20260627"})
    """
    if not api_name or not api_name.strip():
        return "错误：api_name 不能为空。"
    try:
        df = _get_fetcher().fetch(api_name.strip(), **(params or {}))
    except Exception as e:
        return f"实时查询出错：{e}"
    return _df_to_text(df)


@mcp.tool()
def live_stock_daily(ts_code: str, start_date: str, end_date: str, adj: str = "") -> str:
    """实时查询个股日线，可复权——口径与本地 get_stock_daily 完全一致。

    经 token 直连 Tushare：取原始 daily + adj_factor 后自行复权（不用 pro_bar，因为
    pro_bar 不走自定义代理）。适合把"最新行情"与本地历史(qfq/hfq)拼接而不串口径。

    Args:
        ts_code: 股票代码，如 000001.SZ
        start_date / end_date: YYYYMMDD
        adj: 复权方式，''=不复权，'qfq'=前复权，'hfq'=后复权

    复权公式（与 reader._adjust_prices 一致，最新因子取该股全历史 adj_factor 的最新值）：
        hfq: 价 × adj_factor；  qfq: 价 × (adj_factor / 最新因子)
    """
    if not ts_code or not ts_code.strip():
        return "错误：ts_code 不能为空。"
    if adj not in ("", "qfq", "hfq"):
        return "错误：adj 只能是 '' / 'qfq' / 'hfq'。"
    for _d in (start_date, end_date):
        if _d and not (len(_d) == 8 and _d.isdigit()):
            return f"错误：日期需为 YYYYMMDD 格式：{_d}"
    try:
        f = _get_fetcher()
        raw = f.fetch("daily", ts_code=ts_code, start_date=start_date, end_date=end_date)
        if raw is None or raw.empty:
            return "（无数据）"
        if not adj:
            return _df_to_text(raw)
        factors = f.fetch("adj_factor", ts_code=ts_code)  # 全历史一次取，含真实最新因子
        if factors is None or factors.empty:
            return ("（注：未取到复权因子，以下为不复权原始价）\n\n" + _df_to_text(raw))
        orig_cols = list(raw.columns)  # 复权后只返回原始列，不带出中间的 adj_factor
        raw = raw.merge(factors[["trade_date", "adj_factor"]], on="trade_date", how="left")
        # 最新因子取该股全历史 adj_factor 的最新值（实时，可能比本地库更新）
        latest_factor = factors.sort_values("trade_date")["adj_factor"].iloc[-1]
        for col in _PRICE_COLS:
            if col in raw.columns:
                if adj == "hfq":
                    raw[col] = raw[col] * raw["adj_factor"]
                else:  # qfq
                    raw[col] = raw[col] * (raw["adj_factor"] / latest_factor)
        raw = raw[orig_cols]
    except Exception as e:
        return f"实时查询出错：{e}"
    return _df_to_text(raw)


def _bad_date(d: str) -> bool:
    return not (len(d) == 8 and d.isdigit())


@mcp.tool()
def live_limit_list(trade_date: str) -> str:
    """实时查询某交易日的涨跌停/炸板统计（Tushare limit_list_d）。trade_date: YYYYMMDD。"""
    if _bad_date(trade_date):
        return "错误：trade_date 需为 YYYYMMDD 格式。"
    try:
        df = _get_fetcher().fetch("limit_list_d", trade_date=trade_date)
    except Exception as e:
        return f"实时查询出错：{e}"
    return _df_to_text(df)


@mcp.tool()
def live_moneyflow(ts_code: str, trade_date: str = "") -> str:
    """实时查询个股资金流向（Tushare moneyflow，主力/超大单等）。

    Args:
        ts_code: 股票代码，如 000001.SZ
        trade_date: 可选 YYYYMMDD；不填则取该股最近可得数据。
    """
    if not ts_code or not ts_code.strip():
        return "错误：ts_code 不能为空。"
    if trade_date and _bad_date(trade_date):
        return "错误：trade_date 需为 YYYYMMDD 格式。"
    params = {"ts_code": ts_code.strip()}
    if trade_date:
        params["trade_date"] = trade_date
    try:
        df = _get_fetcher().fetch("moneyflow", **params)
    except Exception as e:
        return f"实时查询出错：{e}"
    return _df_to_text(df)


@mcp.tool()
def live_top_list(trade_date: str) -> str:
    """实时查询某交易日龙虎榜（Tushare top_list）。trade_date: YYYYMMDD。"""
    if _bad_date(trade_date):
        return "错误：trade_date 需为 YYYYMMDD 格式。"
    try:
        df = _get_fetcher().fetch("top_list", trade_date=trade_date)
    except Exception as e:
        return f"实时查询出错：{e}"
    return _df_to_text(df)


@mcp.tool()
def get_concept_stocks(trade_date: str, concept_name: str = "") -> str:
    """查询某日概念板块成分股（本地库 PIT，避免前视偏差）。

    Args:
        trade_date: YYYYMMDD（按此日的成员关系）
        concept_name: 可选，如 '人工智能'；不填返回该日全部概念-股票关系。
    """
    if _bad_date(trade_date):
        return "错误：trade_date 需为 YYYYMMDD 格式。"
    try:
        df = _get_reader().get_concept_stocks(trade_date, concept_name=(concept_name or None))
    except Exception as e:
        return f"查询出错：{e}"
    return _df_to_text(df)


@mcp.tool()
def get_stock_concepts(trade_date: str, ts_code: str) -> str:
    """查询某日个股所属的概念板块（本地库 PIT）。

    Args:
        trade_date: YYYYMMDD
        ts_code: 股票代码，如 000001.SZ
    """
    if _bad_date(trade_date):
        return "错误：trade_date 需为 YYYYMMDD 格式。"
    if not ts_code or not ts_code.strip():
        return "错误：ts_code 不能为空。"
    try:
        df = _get_reader().get_stock_concepts(trade_date, ts_code.strip())
    except Exception as e:
        return f"查询出错：{e}"
    return _df_to_text(df)


def main():
    """以 stdio 方式运行 MCP server（Claude Desktop / Code 默认接入方式）。"""
    mcp.run()


if __name__ == "__main__":
    main()
