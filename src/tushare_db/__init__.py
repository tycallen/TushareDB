# -*- coding: utf-8 -*-
"""
tushare_db - A local cache for Tushare data.
"""

from .client import TushareDBClient
from .duckdb_manager import DuckDBManager
from .api import (
    trade_cal,
    TradeCal,
    stock_basic,
    StockBasic,
    hs_const,
    HsConst,
    stock_company,
    StockCompany,
    pro_bar,
    ProBar,
    ProBarAsset,
    ProBarAdj,
    ProBarFreq,
)

__all__ = [
    "TushareDBClient",
    "DuckDBManager",
    "stock_basic",
    "StockBasic",
    "trade_cal",
    "TradeCal",
    "hs_const",
    "HsConst",
    "stock_company",
    "StockCompany",
    "pro_bar",
    "ProBar",
    "ProBarAsset",
    "ProBarAdj",
    "ProBarFreq",
]
