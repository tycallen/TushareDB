# Claude 接入：MCP（历史/大数据）+ Skill（实时/小数据）

本项目为 Claude Desktop / Claude Code 提供两条数据通道：

| 通道 | 用途 | 数据来源 |
|------|------|----------|
| **tushare-duckdb MCP** | 历史、批量、大数据、SQL 分析 | 本地 DuckDB（`tushare.db`，只读） |
| **tushare-live Skill** | 实时、单次、小数据（最新行情/涨跌停/资金流…） | Tushare token 直查 API（支持第三方代理） |

---

## 一、MCP Server（历史/大数据）

### 1. 安装依赖
```bash
pip install -e ".[mcp]"     # 安装 mcp SDK
```

### 2. 验证可启动
```bash
DB_PATH=/绝对路径/tushare.db python -m tushare_db.mcp_server
# 正常会以 stdio 方式等待（Ctrl-C 退出）。供 Claude 调用，不需手动交互。
```

### 3. Claude Desktop 配置
编辑 `claude_desktop_config.json`（macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`）：
```json
{
  "mcpServers": {
    "tushare-duckdb": {
      "command": "python",
      "args": ["-m", "tushare_db.mcp_server"],
      "env": { "DB_PATH": "/绝对路径/tushare.db" }
    }
  }
}
```
> 若 `python` 不是项目环境，请用环境的绝对路径（如 miniforge env 的 `python`）。

### 4. Claude Code 配置
项目根目录创建 `.mcp.json`：
```json
{
  "mcpServers": {
    "tushare-duckdb": {
      "command": "python",
      "args": ["-m", "tushare_db.mcp_server"],
      "env": { "DB_PATH": "/绝对路径/tushare.db" }
    }
  }
}
```
或：`claude mcp add tushare-duckdb -- python -m tushare_db.mcp_server`

### 5. 可用工具
| 工具 | 说明 |
|------|------|
| `list_tables()` | 列出本地所有表 |
| `describe_table(table_name)` | 查看某表的列 |
| `run_sql(sql, max_rows=500)` | 执行只读 SQL（最通用，仅 SELECT/WITH/PRAGMA） |
| `get_stock_daily(ts_code, start_date, end_date, adj)` | 个股日线（支持 qfq/hfq 复权） |
| `get_financial(ts_code, statement, start_date, end_date)` | 财务数据（income/balancesheet/cashflow/fina_indicator_vip/forecast/express） |
| `live_fetch(api_name, params)` | **实时**查询任意 Tushare 接口（经 token 直连代理，返回原始数据）。token 由 MCP 在宿主机从 `.env` 读取，不进对话。例：`live_fetch("daily", {"ts_code":"000001.SZ","trade_date":"20260627"})` |
| `live_stock_daily(ts_code, start_date, end_date, adj)` | **实时**个股日线，**可复权**（adj=''/qfq/hfq）。取原始 daily+adj_factor 自行复权，口径与本地 `get_stock_daily` 一致。用于把"最新行情"接到本地历史(qfq/hfq)而不串口径 |

> `live_fetch`/`live_stock_daily` 让 Claude Desktop（沙箱内）也能取实时数据——无需自身联网或持有 token，
> token 全程留在宿主机侧。前置：项目 `.env`（DB_PATH 同目录）配好 `TUSHARE_TOKEN`/`TUSHARE_API_URL`。

> **复权口径注意**：`live_fetch("daily")` 与 `run_sql` 查的 `daily` 都是**不复权原始价**；
> 算涨幅/拼接历史前要统一口径，用 `live_stock_daily(adj=...)` 或 `get_stock_daily(adj=...)`。
> 另：`live_stock_daily` 的 qfq 用**实时最新复权因子**，若本地库 `adj_factor` 滞后（未跑
> `update_daily`），它与本地 `get_stock_daily(qfq)` 会差一个比例——以实时的为准更贴近当前市场价。

示例对话："用 tushare-duckdb 查 000001.SZ 最近 30 个交易日的前复权收盘价" → Claude 调 `get_stock_daily` 或 `run_sql`。

---

## 二、Skill（实时/小数据）

### 安装
```bash
# 全局（所有项目可用）
cp -r docs/skills/tushare-live ~/.claude/skills/
# 或项目级
cp -r docs/skills/tushare-live .claude/skills/
```

### 前置：配置 .env
```
TUSHARE_TOKEN='你的56位key'
TUSHARE_API_URL='https://fast.xiaodefa.cn'
```

详见 `docs/skills/tushare-live/SKILL.md`。示例对话："000001.SZ 今天的资金流" → Claude 走 skill，用 token 直查代理。

---

## 边界与建议
- 「现在/今天/某只股票最新…」→ **Skill**（实时直查）
- 「近 N 年 / 全市场 / 批量统计 / SQL…」→ **MCP**（本地 db）
- 实时直查注意限速（间隔 ≥0.5s，详见 SKILL.md）；本地 MCP 无网络、无限速。
