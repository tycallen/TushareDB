# -*- coding: utf-8 -*-
"""
This module defines pre-configured rate limit profiles for the Tushare API.
These profiles correspond to different user subscription levels (积分档位)
and provide a convenient way to manage API call frequencies without manual
configuration for every API.

Each profile is a dictionary where keys are API names (or 'default') and
values specify the 'limit' and 'period' ('minute' or 'day').
"""
from typing import Dict, Any

# --- Base settings for different data categories ---
# Metadata APIs are generally less restricted.
META_API_MINUTE_LIMIT_TRIAL = 200
META_API_MINUTE_LIMIT_STANDARD = 500
META_API_MINUTE_LIMIT_PRO = 1000

# Data-intensive APIs have stricter limits.
DATA_API_MINUTE_LIMIT_TRIAL = 200
DATA_API_MINUTE_LIMIT_STANDARD = 500
DATA_API_MINUTE_LIMIT_PRO = 1000

# --- Trial User Profile (e.g., 120 points) ---
# Low limits, suitable for testing and basic usage.
TRIAL_PROFILE: Dict[str, Dict[str, Any]] = {
    "default": {"limit": META_API_MINUTE_LIMIT_TRIAL, "period": "minute"},
    # Metadata APIs
    "stock_basic": {"limit": META_API_MINUTE_LIMIT_TRIAL, "period": "minute"},
    "trade_cal": {"limit": META_API_MINUTE_LIMIT_TRIAL, "period": "minute"},
    "hs_const": {"limit": META_API_MINUTE_LIMIT_TRIAL, "period": "minute"},
    "stock_company": {"limit": META_API_MINUTE_LIMIT_TRIAL, "period": "minute"},
    # Data APIs
    "pro_bar": {"limit": DATA_API_MINUTE_LIMIT_TRIAL, "period": "minute"},
    "dc_member": {"limit": DATA_API_MINUTE_LIMIT_TRIAL, "period": "minute"},
    "dc_index": {"limit": DATA_API_MINUTE_LIMIT_TRIAL, "period": "minute"},
    "cyq_perf": {"limit": DATA_API_MINUTE_LIMIT_TRIAL, "period": "minute"},
    "cyq_chips": {"limit": DATA_API_MINUTE_LIMIT_TRIAL, "period": "minute"},
}

# --- Standard User Profile (e.g., 5000 points) ---
# Increased limits for regular data analysis.
STANDARD_PROFILE: Dict[str, Dict[str, Any]] = {
    "default": {"limit": META_API_MINUTE_LIMIT_STANDARD, "period": "minute"},
    # Metadata APIs
    "stock_basic": {"limit": META_API_MINUTE_LIMIT_STANDARD, "period": "minute"},
    "trade_cal": {"limit": META_API_MINUTE_LIMIT_STANDARD, "period": "minute"},
    "hs_const": {"limit": META_API_MINUTE_LIMIT_STANDARD, "period": "minute"},
    "stock_company": {"limit": META_API_MINUTE_LIMIT_STANDARD, "period": "minute"},
    # Data APIs
    "pro_bar": {"limit": DATA_API_MINUTE_LIMIT_STANDARD, "period": "minute"},
    "dc_member": {"limit": DATA_API_MINUTE_LIMIT_STANDARD, "period": "minute"},
    "dc_index": {"limit": DATA_API_MINUTE_LIMIT_STANDARD, "period": "minute"},
    "cyq_perf": {"limit": 20000, "period": "day"},
    "cyq_chips": {"limit": 200, "period": "minute"},
    "stk_factor_pro": {"limit": 30, "period": "minute"}
}

# --- Professional User Profile (e.g., 10000+ points) ---
# High limits for intensive data processing and applications.
PRO_PROFILE: Dict[str, Dict[str, Any]] = {
    "default": {"limit": META_API_MINUTE_LIMIT_PRO, "period": "minute"},
    # Metadata APIs
    "stock_basic": {"limit": META_API_MINUTE_LIMIT_PRO, "period": "minute"},
    "trade_cal": {"limit": META_API_MINUTE_LIMIT_PRO, "period": "minute"},
    "hs_const": {"limit": META_API_MINUTE_LIMIT_PRO, "period": "minute"},
    "stock_company": {"limit": META_API_MINUTE_LIMIT_PRO, "period": "minute"},
    # Data APIs
    "pro_bar": {"limit": DATA_API_MINUTE_LIMIT_PRO, "period": "minute"},
    "dc_member": {"limit": DATA_API_MINUTE_LIMIT_PRO, "period": "minute"},
    "dc_index": {"limit": DATA_API_MINUTE_LIMIT_PRO, "period": "minute"},
    "cyq_perf": {"limit": 200000, "period": "day"},
    "cyq_chips": {"limit": 200, "period": "minute"},
}

# --- Third-party Proxy Profile ---
# 适配 Tushare 协议兼容的第三方代理（如 xiaodefa，配合环境变量 TUSHARE_API_URL 使用）。
# 代理对超速敏感：超过购买速度过多会触发 3-10 分钟冷却（程序上表现为超时报错）。
# 据其文档建议请求间隔 ≥0.5s（≈120 次/分钟），这里默认取 100 次/分钟（≈0.6s/请求）
# 留出余量。请按你实际购买的频次调整 PROXY_MINUTE_LIMIT。
# 注意：每个 token 每日总请求约 1-2 万次；限速器按 api 名分别计数，没有跨 api 的全局
# 日上限，批量回填时请自行控制总量与并发。冷却/剩余次数可查 https://tt.xiaodefa.cn/status
PROXY_MINUTE_LIMIT = 100

PROXY_PROFILE: Dict[str, Dict[str, Any]] = {
    "default": {"limit": PROXY_MINUTE_LIMIT, "period": "minute"},
    # 历史分钟数据（stk_mins）量巨大且官方另有限速与单日容量限制，取更保守值，
    # 并建议在调用处对官方限速错误加重试。
    # 用 min(default, ...) 兜底：即便用户把 PROXY_MINUTE_LIMIT 调到 <30，
    # stk_mins 也不会比 default 更宽松。
    "stk_mins": {"limit": min(PROXY_MINUTE_LIMIT, max(30, PROXY_MINUTE_LIMIT // 2)),
                 "period": "minute"},
}

# A mapping of profile names to profile configurations for easy selection.
PRESET_PROFILES: Dict[str, Dict[str, Dict[str, Any]]] = {
    "trial": TRIAL_PROFILE,
    "standard": STANDARD_PROFILE,
    "pro": PRO_PROFILE,
    "proxy": PROXY_PROFILE,
}