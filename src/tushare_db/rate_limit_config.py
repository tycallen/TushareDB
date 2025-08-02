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
    "cyq_chips": {"limit": 20000, "period": "day"},
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
    "cyq_chips": {"limit": 200000, "period": "day"},
}

# A mapping of profile names to profile configurations for easy selection.
PRESET_PROFILES: Dict[str, Dict[str, Dict[str, Any]]] = {
    "trial": TRIAL_PROFILE,
    "standard": STANDARD_PROFILE,
    "pro": PRO_PROFILE,
}