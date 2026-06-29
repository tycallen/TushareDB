---
name: tushare-live
description: 实时/最新的中国股票小数据查询，通过 Tushare token 直接调用 API（支持第三方代理）。适合单股最新行情、每日指标、当日涨跌停、最新资金流、龙虎榜等小数据的即时查询。历史/批量/大数据请改用 tushare-duckdb MCP（查本地 DuckDB）。
allowed-tools:
  - Bash(python:*)
  - Read
---

# Tushare Live —— 实时小数据直查 Skill

通过 Tushare token **直接查询 API**（支持第三方协议兼容代理），获取最新/实时的小数据。

## 定位与边界
- ✅ **本 skill**：实时、单次、小数据 —— 最新行情、每日指标、当日涨跌停、最新资金流、龙虎榜等，**不落库**
- ❌ **历史 / 批量 / 大数据**：改用 **tushare-duckdb MCP**（查本地 DuckDB `tushare.db`）

> 一句话区分：问"现在/今天/某只股票最新…"→ 本 skill；问"近 N 年 / 全市场 / 批量统计…"→ tushare-duckdb MCP。

> **在 Claude Desktop（沙箱）里**：沙箱无项目环境、读不到本机 `.env`，请直接用
> **tushare-duckdb MCP 的 `live_fetch` 工具**做实时查询——token 由 MCP 在宿主机侧从
> `.env` 读取，**无需在对话中提供 token**。例：`live_fetch("daily", {"ts_code":"000001.SZ","trade_date":"20260627"})`。
> 本 skill 的下列本机代码方式主要面向 Claude Code（本机有项目+`.env`）。

## Token 配置

本项目数据可通过第三方代理（Tushare 协议兼容）获取。配置项目根目录 `.env`：
```
TUSHARE_TOKEN='你的56位key'
TUSHARE_API_URL='https://fast.xiaodefa.cn'   # 第三方代理（电信优选）；不设则走官方 api.tushare.pro
```
- 备用域名：`https://tt.xiaodefa.cn`
- token 有效期 / 冷却 / 今日次数查询：`https://tt.xiaodefa.cn/status`

## 获取方式

### 方式一：用项目库（推荐，自动走 `TUSHARE_API_URL` 代理）
```python
import os, sys
from pathlib import Path
sys.path.insert(0, "src")  # 或已 pip install -e .

# 加载 .env（覆盖式，避免 shell 残留旧 token）
for _ln in Path(".env").read_text(encoding="utf-8").splitlines():
    _s = _ln.strip()
    if _s and not _s.startswith("#") and "=" in _s:
        _k, _v = _s.split("=", 1)
        os.environ[_k.strip()] = _v.strip().strip("'").strip('"')

from tushare_db.tushare_fetcher import TushareFetcher
from tushare_db.rate_limit_config import PROXY_PROFILE

f = TushareFetcher(os.environ["TUSHARE_TOKEN"], PROXY_PROFILE)  # 自带限速 + 优雅降级
df = f.fetch("daily", ts_code="000001.SZ", trade_date="20260627")
print(df)
```
`TushareFetcher` 会自动按 `PROXY_PROFILE` 限速；遇到无权限/不存在的接口会自动返回空、不反复报错。

### 方式二：直接 HTTP POST 代理（不依赖项目库）
```python
import os, requests
resp = requests.post(
    os.environ.get("TUSHARE_API_URL", "https://fast.xiaodefa.cn"),
    json={
        "api_name": "daily",
        "token": os.environ["TUSHARE_TOKEN"],
        "params": {"ts_code": "000001.SZ", "trade_date": "20260627"},
    },
    headers={"Accept-Encoding": "gzip"},   # 非 requests 库务必手动设，提速
    timeout=15,
)
print(resp.json())   # {"code":0,"msg":"","data":{"fields":[...],"items":[...]}}
```

## 常用实时小数据接口速查（均已实测可用）

| 数据 | api_name | 关键参数 | 说明 |
|------|----------|----------|------|
| 最新日线 | `daily` | ts_code, trade_date | 单股某日 OHLCV |
| 每日指标 | `daily_basic` | ts_code, trade_date | PE/PB/换手率/市值 |
| 涨跌停价 | `stk_limit` | ts_code, trade_date | 当日涨/跌停价 |
| 个股资金流 | `moneyflow` | ts_code, trade_date | 主力/超大单等 |
| 沪深股通资金 | `moneyflow_hsgt` | trade_date | 北向资金 |
| 龙虎榜 | `top_list` | trade_date | 当日上榜个股 |
| 龙虎榜机构 | `top_inst` | trade_date | 机构席位明细 |
| 涨跌停统计 | `limit_list_d` | trade_date | 当日涨/跌停/炸板 |
| 股票列表 | `stock_basic` | list_status=L | 全市场股票 |
| 交易日历 | `trade_cal` | start_date, end_date | 是否交易日 |
| 指数日线 | `index_daily` | ts_code | 指数行情 |
| 业绩快报 | `express_vip` | period | 最新业绩快报 |
| 业绩预告 | `forecast_vip` | period | 最新业绩预告 |

> 当前 token 到底能用哪些接口？运行 `python scripts/probe_interface_permissions.py` 实测，
> 输出 ✓可用 / ✗无权限 / ⚠接口不存在 清单。

## 参数格式
- 日期：`YYYYMMDD`（如 20260627）
- 股票代码：`ts_code`（000001.SZ / 600000.SH）
- 返回：pandas DataFrame（方式一）或 JSON（方式二）

## ⚠️ 限速（务必遵守）
- 第三方代理对超速敏感：请求间隔 **≥0.5 秒**，不要超过你购买的频次，否则触发 **3–10 分钟冷却**（表现为超时报错——不是服务不稳，是被冷却了）。
- 每个 token 每日总请求约 **1–2 万次**（按请求次数算，每次最多 8000 条数据）。
- 让 AI 批量调用时务必明确限速，例如："每次请求间隔务必等待 0.5 秒"。
- 尽量把多只股票/多日合并到一次请求，充分利用单次 8000 条额度。
- 冷却 / 剩余次数：`https://tt.xiaodefa.cn/status`

## 无权限 / 不存在的接口
部分接口当前数据源未开放（实测：`stk_mins` 历史分钟、`stk_auction_o` 集合竞价 无权限；
`hs_const`、`kpl_concept` 接口不存在）。用方式一（`TushareFetcher`）时会被**自动识别并优雅跳过**
（返回空、首次告警一次、不反复请求）；不同代理不可用的接口可能不同，以实测为准。

## 参考
- 历史 / 批量 / 大数据：改用 **tushare-duckdb MCP**（本地 DuckDB，见 `docs/MCP_SETUP.md`）
- Tushare 官方接口文档：https://tushare.pro/document/2
