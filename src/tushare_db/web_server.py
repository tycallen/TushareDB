# -*- coding: utf-8 -*-
"""
FastAPI Web Server for Tushare-DuckDB.

This server provides a RESTful API to access the data
managed by the Tushare-DuckDB library.
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse # Import ORJSONResponse
from typing import Optional, List
import pandas as pd
import orjson # Import orjson
import json

from . import api
from .client import TushareDBClient

# Create a FastAPI app instance
app = FastAPI(
    title="Tushare-DuckDB API",
    description="A web API for accessing Tushare data stored in DuckDB.",
    version="0.1.0",
    default_response_class=ORJSONResponse # Set default response class to ORJSONResponse
)

# Configure CORS (Cross-Origin Resource Sharing)
# 命令：uvicorn src.tushare_db.web_server:app --host 0.0.0.0 --port 8000
# This allows the frontend (running on a different port) to communicate with this backend.
origins = [
    "http://localhost:5173",  # Default Vite dev server port
    "http://localhost:3000",  # Common alternative dev port
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def read_root():
    """
    Root endpoint to check if the server is running.
    """
    return {"message": "Welcome to the Tushare-DuckDB API!"}

# --- API Endpoints will be added below ---

# Create a single, reusable client instance to manage the database connection
# This is more efficient than creating a new client for each request.
try:
    # It's good practice to wrap this in a try...except block
    # in case initialization fails (e.g., missing config, DB connection error)
    client = TushareDBClient()
except Exception as e:
    # If the client fails to initialize, the app can't function.
    # We can log the error and potentially exit or handle it gracefully.
    print(f"FATAL: Failed to initialize TushareDBClient: {e}")
    client = None


def df_to_json_response(df):
    """
    Helper function to convert a pandas DataFrame to a JSON response using orjson.
    Handles potential NaN values by converting them to None (null in JSON).
    """
    if df is None or df.empty:
        return []
    # Convert DataFrame to list of dictionaries, handling NaN values
    # orjson handles NaN by converting to null by default, so no explicit .where() is needed here
    return orjson.loads(orjson.dumps(df.to_dict(orient='records')))


@app.get("/api/stock_basic")
async def get_stock_basic(
    ts_code: Optional[str] = None,
    name: Optional[str] = None,
    market: Optional[str] = None,
    list_status: Optional[str] = 'L',
    exchange: Optional[str] = None,
    is_hs: Optional[str] = None,
    fields: Optional[str] = 'ts_code,symbol,name,area,industry,list_date,market,list_status'
):
    """
    API endpoint for the stock_basic interface.
    """
    if not client:
        raise HTTPException(status_code=503, detail="Database client is not available.")
    try:
        df = api.stock_basic(
            client=client,
            ts_code=ts_code,
            name=name,
            market=market,
            list_status=list_status,
            exchange=exchange,
            is_hs=is_hs,
            fields=fields
        )
        return df_to_json_response(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/cyq_chips")
async def get_cyq_chips(
    ts_code: Optional[str] = None,
    trade_date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    fields: Optional[str] = None
):
    """
    API endpoint for the cyq_chips interface.
    """
    if not client:
        raise HTTPException(status_code=503, detail="Database client is not available.")
    try:
        df = api.cyq_chips(
            client=client,
            ts_code=ts_code,
            trade_date=trade_date,
            start_date=start_date,
            end_date=end_date,
            fields=fields
        )
        return df_to_json_response(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stk_factor_pro")
async def get_stk_factor_pro(
    ts_code: Optional[str] = None,
    trade_date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    fields: Optional[str] = None
):
    """
    API endpoint for the stk_factor_pro interface.
    """
    if not client:
        raise HTTPException(status_code=503, detail="Database client is not available.")
    try:
        df = api.stk_factor_pro(
            client=client,
            ts_code=ts_code,
            trade_date=trade_date,
            start_date=start_date,
            end_date=end_date,
            fields=fields
        )
        return df_to_json_response(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/trade_cal")
async def get_trade_cal(
    exchange: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    is_open: Optional[str] = None,
    fields: Optional[str] = 'exchange,cal_date,is_open,pretrade_date'
):
    """
    API endpoint for the trade_cal interface.
    """
    if not client:
        raise HTTPException(status_code=503, detail="Database client is not available.")
    try:
        df = api.trade_cal(
            client=client,
            exchange=exchange,
            start_date=start_date,
            end_date=end_date,
            is_open=is_open,
            fields=fields
        )
        return df_to_json_response(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/hs_const")
async def get_hs_const(
    hs_type: str,
    is_new: Optional[str] = '1',
    fields: Optional[str] = 'ts_code,hs_type,in_date,out_date,is_new'
):
    """
    API endpoint for the hs_const interface.
    Note: `hs_type` is required for this endpoint.
    """
    if not client:
        raise HTTPException(status_code=503, detail="Database client is not available.")
    try:
        df = api.hs_const(
            client=client,
            hs_type=hs_type,
            is_new=is_new,
            fields=fields
        )
        return df_to_json_response(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stock_company")
async def get_stock_company(
    ts_code: Optional[str] = None,
    exchange: Optional[str] = None,
    fields: Optional[str] = 'ts_code,com_name,com_id,exchange,chairman,manager,secretary,reg_capital,setup_date,province,city,website,email,employees,main_business'
):
    """
    API endpoint for the stock_company interface.
    """
    if not client:
        raise HTTPException(status_code=503, detail="Database client is not available.")
    try:
        df = api.stock_company(
            client=client,
            ts_code=ts_code,
            exchange=exchange,
            fields=fields
        )
        return df_to_json_response(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/index_basic")
async def get_index_basic(
    ts_code: Optional[str] = None,
    name: Optional[str] = None,
    market: Optional[str] = None,
    publisher: Optional[str] = None,
    category: Optional[str] = None,
    fields: Optional[str] = None
):
    """
    API endpoint for the index_basic interface.
    """
    if not client:
        raise HTTPException(status_code=503, detail="Database client is not available.")
    try:
        df = api.index_basic(
            client=client,
            ts_code=ts_code,
            name=name,
            market=market,
            publisher=publisher,
            category=category,
            fields=fields
        )
        return df_to_json_response(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/index_weight")
async def get_index_weight(
    index_code: str,
    year: int,
    month: int,
    fields: Optional[str] = None
):
    """
    API endpoint for the index_weight interface.
    Note: `index_code`, `year`, and `month` are required for this endpoint.
    """
    if not client:
        raise HTTPException(status_code=503, detail="Database client is not available.")
    try:
        df = api.index_weight(
            client=client,
            index_code=index_code,
            year=year,
            month=month,
            fields=fields
        )
        return df_to_json_response(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/daily_basic")
async def get_daily_basic(
    ts_code: Optional[str] = None,
    trade_date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    fields: Optional[str] = None
):
    """
    API endpoint for the daily_basic interface.
    """
    if not client:
        raise HTTPException(status_code=503, detail="Database client is not available.")
    try:
        df = api.daily_basic(
            client=client,
            ts_code=ts_code,
            trade_date=trade_date,
            start_date=start_date,
            end_date=end_date,
            fields=fields
        )
        return df_to_json_response(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/dc_member")
async def get_dc_member(
    ts_code: Optional[str] = None,
    con_code: Optional[str] = None,
    trade_date: Optional[str] = None,
    fields: Optional[str] = None
):
    """
    API endpoint for the dc_member interface.
    """
    if not client:
        raise HTTPException(status_code=503, detail="Database client is not available.")
    try:
        df = api.dc_member(
            client=client,
            ts_code=ts_code,
            con_code=con_code,
            trade_date=trade_date,
            fields=fields
        )
        return df_to_json_response(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/dc_index")
async def get_dc_index(
    ts_code: Optional[str] = None,
    trade_date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    fields: Optional[str] = None
):
    """
    API endpoint for the dc_index interface.
    """
    if not client:
        raise HTTPException(status_code=503, detail="Database client is not available.")
    try:
        df = api.dc_index(
            client=client,
            ts_code=ts_code,
            trade_date=trade_date,
            start_date=start_date,
            end_date=end_date,
            fields=fields
        )
        return df_to_json_response(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/get_top_n_sector_members")
async def get_top_n_sector_members(
    start_date: str,
    end_date: str,
    top_n: int = 5,
    sort_by: str = 'pct_change',
    ascending: bool = False
):
    """
    API endpoint for the get_top_n_sector_members interface.
    Note: `trade_date` and `n` are required for this endpoint.
    """
    if not client:
        raise HTTPException(status_code=503, detail="Database client is not available.")
    try:
        df = api.get_top_n_sector_members(
            client=client,
            start_date=start_date,
            end_date=end_date,
            top_n=top_n,
            sort_by=sort_by,
            ascending=ascending,
        )
        print(type(df))
        return json.dumps(df)#df_to_json_response(df) #
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/cyq_perf")
async def get_cyq_perf(
    ts_code: Optional[str] = None,
    trade_date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    fields: Optional[str] = None
):
    """
    API endpoint for the cyq_perf interface.
    """
    if not client:
        raise HTTPException(status_code=503, detail="Database client is not available.")
    try:
        df = api.cyq_perf(
            client=client,
            ts_code=ts_code,
            trade_date=trade_date,
            start_date=start_date,
            end_date=end_date,
            fields=fields
        )
        return df_to_json_response(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/fina_indicator_vip")
async def get_fina_indicator_vip(
    ts_code: Optional[str] = None,
    ann_date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    period: Optional[str] = None,
    fields: Optional[str] = None
):
    """
    API endpoint for the fina_indicator_vip interface.
    """
    if not client:
        raise HTTPException(status_code=503, detail="Database client is not available.")
    try:
        df = api.fina_indicator_vip(
            client=client,
            ts_code=ts_code,
            ann_date=ann_date,
            start_date=start_date,
            end_date=end_date,
            period=period,
            fields=fields
        )
        return df_to_json_response(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/adj_factor")
async def get_adj_factor(
    ts_code: Optional[str] = None,
    trade_date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    fields: Optional[str] = None
):
    """
    API endpoint for the adj_factor interface.
    """
    if not client:
        raise HTTPException(status_code=503, detail="Database client is not available.")
    try:
        df = api.adj_factor(
            client=client,
            ts_code=ts_code,
            trade_date=trade_date,
            start_date=start_date,
            end_date=end_date,
            fields=fields
        )
        return df_to_json_response(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/pro_bar")
async def get_pro_bar(
    ts_code: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    asset: str = 'E',
    freq: str = 'D',
    # For 'ma', FastAPI can handle list parameters from query strings
    # e.g., /api/pro_bar?ts_code=...&ma=5&ma=10&ma=20
    ma: Optional[List[int]] = Query(None),
    adjfactor: bool = False,
):
    """
    API endpoint for the pro_bar interface.
    Note: `ts_code` is required for this endpoint.
    """
    if not client:
        raise HTTPException(status_code=503, detail="Database client is not available.")
    try:
        # The api.pro_bar function expects a list for 'ma'
        df = api.pro_bar(
            client=client,
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            asset=asset,
            freq=freq,
            ma=ma,
            adjfactor=adjfactor
        )
        return df_to_json_response(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


