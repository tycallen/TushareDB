#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
探测当前 Tushare token（含第三方代理）实际可调用的接口权限。

用法：
    # 先在 .env 配好 TUSHARE_TOKEN，以及（如走代理）TUSHARE_API_URL
    python scripts/probe_interface_permissions.py
    # 只探测部分分组
    python scripts/probe_interface_permissions.py --only 财务VIP,资金
    # 调整请求间隔（秒），频次低的账号请调大
    python scripts/probe_interface_permissions.py --interval 1.0

输出：逐接口标注 ✓可用 / ✗无权限 / ⚠冷却·其他，并给出分组与总计汇总。
脚本会手动限速（不依赖 fetcher 的滑窗，因为窗口未满时它不会 sleep），
默认每请求间隔 0.7s（>文档要求的 0.5s），尽量避免触发代理冷却。
"""
import os
import sys
import time
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 加载 .env（不覆盖已存在的环境变量；token 不会被打印）
_env = project_root / ".env"
if _env.exists():
    for _ln in _env.read_text(encoding="utf-8").splitlines():
        _ln = _ln.strip()
        if _ln and not _ln.startswith("#") and "=" in _ln:
            _k, _v = _ln.split("=", 1)
            # .env 覆盖已有环境变量：避免 shell 里残留的旧 TUSHARE_TOKEN 盖过 .env 的新值
            os.environ[_k.strip()] = _v.strip().strip("'").strip('"')

from datetime import datetime  # noqa: E402
from tushare_db.tushare_fetcher import TushareFetcher, TushareUnavailableError  # noqa: E402
from tushare_db.rate_limit_config import PROXY_PROFILE  # noqa: E402

# 探测用的通用参数
TS = "000001.SZ"
START, END = "20260601", "20260627"
PERIOD = "20250331"          # 已披露的财报期（2025Q1）
DATE = "20260626"            # 单日类接口默认日期，运行时会用真实最近交易日替换
_DATE_PLACEHOLDER = "<DATE>"

# 候选接口（分组）。每项: (api_name, params)
GROUPS = {
    "基础信息": [
        ("stock_basic", {}),
        ("trade_cal", {}),
        ("namechange", {"ts_code": TS}),
        ("stock_company", {"ts_code": TS}),
        ("new_share", {}),
        ("hs_const", {"hs_type": "SH"}),
    ],
    "行情": [
        ("daily", {"ts_code": TS, "start_date": START, "end_date": END}),
        ("weekly", {"ts_code": TS, "start_date": START, "end_date": END}),
        ("monthly", {"ts_code": TS, "start_date": START, "end_date": END}),
        ("daily_basic", {"ts_code": TS, "start_date": START, "end_date": END}),
        ("adj_factor", {"ts_code": TS, "start_date": START, "end_date": END}),
        ("stk_limit", {"ts_code": TS, "start_date": START, "end_date": END}),
        ("suspend_d", {"ts_code": TS, "start_date": START, "end_date": END}),
        ("bak_daily", {"ts_code": TS, "start_date": START, "end_date": END}),
        ("stk_auction_o", {"trade_date": _DATE_PLACEHOLDER}),
    ],
    "分钟数据": [
        ("stk_mins", {"ts_code": TS, "freq": "1min",
                      "start_date": "2026-06-26 09:30:00",
                      "end_date": "2026-06-26 10:00:00"}),
    ],
    "资金/筹码/龙虎榜/两融": [
        ("moneyflow", {"ts_code": TS, "start_date": START, "end_date": END}),
        ("moneyflow_hsgt", {"start_date": START, "end_date": END}),
        ("hsgt_top10", {"trade_date": _DATE_PLACEHOLDER}),
        ("ggt_top10", {"trade_date": _DATE_PLACEHOLDER}),
        ("hk_hold", {"trade_date": _DATE_PLACEHOLDER}),
        ("top_list", {"trade_date": _DATE_PLACEHOLDER}),
        ("top_inst", {"trade_date": _DATE_PLACEHOLDER}),
        ("margin", {"start_date": START, "end_date": END}),
        ("margin_detail", {"trade_date": _DATE_PLACEHOLDER}),
        ("cyq_perf", {"ts_code": TS, "start_date": START, "end_date": END}),
        ("cyq_chips", {"ts_code": TS, "trade_date": _DATE_PLACEHOLDER}),
        ("stk_factor", {"ts_code": TS, "start_date": START, "end_date": END}),
        ("stk_factor_pro", {"ts_code": TS, "start_date": START, "end_date": END}),
    ],
    "涨跌停/打板/题材": [
        ("limit_list_d", {"trade_date": _DATE_PLACEHOLDER}),
        ("limit_list_ths", {"trade_date": _DATE_PLACEHOLDER}),
        ("kpl_list", {"trade_date": _DATE_PLACEHOLDER}),
        ("kpl_concept", {"trade_date": _DATE_PLACEHOLDER}),
        ("limit_cpt_list", {"trade_date": _DATE_PLACEHOLDER}),
        ("dc_index", {"trade_date": _DATE_PLACEHOLDER}),
        ("dc_member", {"trade_date": _DATE_PLACEHOLDER}),
    ],
    "财务": [
        ("income", {"ts_code": TS, "period": PERIOD}),
        ("balancesheet", {"ts_code": TS, "period": PERIOD}),
        ("cashflow", {"ts_code": TS, "period": PERIOD}),
        ("forecast", {"ts_code": TS, "period": PERIOD}),
        ("express", {"ts_code": TS, "period": PERIOD}),
        ("dividend", {"ts_code": TS}),
        ("fina_indicator", {"ts_code": TS, "period": PERIOD}),
        ("fina_mainbz", {"ts_code": TS, "period": PERIOD}),
        ("disclosure_date", {"ts_code": TS}),
    ],
    "财务VIP(按日期取全市场，高积分)": [
        ("income_vip", {"period": PERIOD}),
        ("balancesheet_vip", {"period": PERIOD}),
        ("cashflow_vip", {"period": PERIOD}),
        ("fina_indicator_vip", {"period": PERIOD}),
        ("forecast_vip", {"period": PERIOD}),
        ("express_vip", {"period": PERIOD}),
    ],
    "指数/行业": [
        ("index_basic", {"market": "SSE"}),
        ("index_daily", {"ts_code": "000300.SH", "start_date": START, "end_date": END}),
        ("index_weight", {"index_code": "399300.SZ", "start_date": START, "end_date": END}),
        ("index_dailybasic", {"trade_date": _DATE_PLACEHOLDER}),
        ("sw_daily", {"ts_code": "801010.SI", "start_date": START, "end_date": END}),
        ("ths_daily", {"ts_code": "885800.TI", "start_date": START, "end_date": END}),
    ],
    "基金": [
        ("fund_basic", {}),
        ("fund_daily", {"ts_code": "515050.SH", "start_date": START, "end_date": END}),
        ("fund_nav", {"ts_code": "515050.SH"}),
        ("fund_portfolio", {"ts_code": "515050.SH"}),
    ],
}


def _short(msg: str, n: int = 70) -> str:
    msg = " ".join(str(msg).split())
    return msg if len(msg) <= n else msg[:n] + "…"


def classify(fetcher, api_name: str, params: dict):
    """返回 (status, detail)。status ∈ OK/NO_PERM/MISSING/TOKEN/RATE/ERR。"""
    try:
        # raise_on_unavailable=True：让无权限/接口不存在抛出来以便分类，
        # 否则会被 fetcher 的优雅降级吞成空结果，被误判为「有权限但空」。
        df = fetcher.fetch(api_name, raise_on_unavailable=True, **params)
        n = 0 if df is None else len(df)
        return ("OK", f"{n} 行" + ("（空，但有权限）" if n == 0 else ""))
    except TushareUnavailableError as e:
        msg = str(e)
        if "接口不存在" in msg:
            return ("MISSING", "接口不存在（当前数据源未实现）")
        return ("NO_PERM", _short(msg))
    except Exception as e:
        msg = str(e)
        low = msg.lower()
        if "token不对" in msg or "40101" in msg:
            return ("TOKEN", _short(msg))
        if any(k in msg for k in ["冷却", "每分钟", "频率", "超过", "频次", "限制"]) \
                or "timeout" in low or "max retries" in low or "timed out" in low:
            return ("RATE", _short(msg))
        return ("ERR", _short(msg))


_ICON = {"OK": "✓", "NO_PERM": "✗", "MISSING": "⚠", "TOKEN": "⛔", "RATE": "⚠", "ERR": "⚠"}


def main():
    parser = argparse.ArgumentParser(description="探测 Tushare token 接口权限")
    parser.add_argument("--only", type=str, help="只探测指定分组（逗号分隔，支持子串匹配）")
    parser.add_argument("--interval", type=float, default=0.7, help="每请求间隔秒数，默认 0.7")
    args = parser.parse_args()

    token = os.getenv("TUSHARE_TOKEN")
    if not token:
        print("错误: 未设置 TUSHARE_TOKEN（请在 .env 填写，或 export）")
        sys.exit(1)
    api_url = os.getenv("TUSHARE_API_URL")
    print("=" * 72)
    print("Tushare 接口权限探测")
    print(f"  请求地址: {api_url or '官方 api.tushare.pro'}")
    print(f"  请求间隔: {args.interval}s")
    print("=" * 72)

    fetcher = TushareFetcher(token=token, rate_limit_config=PROXY_PROFILE)

    # 先用 trade_cal 取最近交易日，替换单日类接口的日期
    global DATE
    try:
        today = datetime.now().strftime("%Y%m%d")
        cal = fetcher.fetch("trade_cal", start_date="20260101", end_date=today, is_open="1")
        if cal is not None and not cal.empty:
            _dates = sorted(d for d in cal["cal_date"].astype(str).tolist() if d <= today)
            if _dates:
                DATE = _dates[-1]
                print(f"  最近交易日: {DATE}\n")
    except Exception as e:
        print(f"  (取最近交易日失败，用默认 {DATE}): {_short(str(e))}\n")

    groups = GROUPS
    if args.only:
        wanted = [w.strip() for w in args.only.split(",")]
        groups = {g: v for g, v in GROUPS.items() if any(w in g for w in wanted)}

    totals = {"OK": 0, "NO_PERM": 0, "MISSING": 0, "TOKEN": 0, "RATE": 0, "ERR": 0}
    available, no_perm = [], []

    for gname, items in groups.items():
        print(f"\n【{gname}】")
        for api_name, params in items:
            params = {k: (DATE if v == _DATE_PLACEHOLDER else v) for k, v in params.items()}
            status, detail = classify(fetcher, api_name, params)
            totals[status] += 1
            print(f"  {_ICON[status]} {api_name:<22} {detail}")
            if status == "OK":
                available.append(api_name)
            elif status == "NO_PERM":
                no_perm.append(api_name)
            if status == "TOKEN":
                print("\n⛔ token 无效或未走代理，后续无意义，已停止。"
                      "请检查 TUSHARE_TOKEN / TUSHARE_API_URL。")
                _summary(totals, available, no_perm)
                return
            time.sleep(args.interval)

    _summary(totals, available, no_perm)


def _summary(totals, available, no_perm):
    total = sum(totals.values())
    print("\n" + "=" * 72)
    print(f"汇总: ✓可用 {totals['OK']} / ✗无权限 {totals['NO_PERM']} / "
          f"⚠接口不存在 {totals['MISSING']} / ⚠其他(冷却·报错) {totals['RATE'] + totals['ERR']} / 共 {total}")
    if no_perm:
        print(f"\n无权限接口: {', '.join(no_perm)}")
    print("=" * 72)


if __name__ == "__main__":
    main()
