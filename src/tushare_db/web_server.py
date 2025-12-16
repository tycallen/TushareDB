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

from .reader import DataReader

# Create a FastAPI app instance
app = FastAPI(
    title="Tushare-DuckDB API",
    description="A web API for accessing Tushare data stored in DuckDB (New Architecture - High Performance).",
    version="2.0.0",  # New architecture version
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

# Create a single, reusable DataReader instance (read-only, high performance)
# This is more efficient than creating a new reader for each request.
try:
    # DataReader is lightweight and read-only, no network overhead
    reader = DataReader()
    print("✓ DataReader initialized successfully (New Architecture - High Performance Mode)")
except Exception as e:
    # If the reader fails to initialize, the app can't function.
    print(f"FATAL: Failed to initialize DataReader: {e}")
    reader = None


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
    list_status: Optional[str] = 'L'
):
    """
    API endpoint for stock_basic - get stock list.
    Note: Simplified parameters in new architecture. Use custom SQL for complex queries.
    """
    if not reader:
        raise HTTPException(status_code=503, detail="Database reader is not available.")
    try:
        df = reader.get_stock_basic(ts_code=ts_code, list_status=list_status)
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
    if not reader:
        raise HTTPException(status_code=503, detail="Database reader is not available.")
    try:
        # Build SQL query
        conditions = []
        params = []
        if ts_code:
            conditions.append("ts_code = ?")
            params.append(ts_code)
        if trade_date:
            conditions.append("trade_date = ?")
            params.append(trade_date)
        if start_date:
            conditions.append("trade_date >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("trade_date <= ?")
            params.append(end_date)

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        field_list = fields if fields else "*"
        sql = f"SELECT {field_list} FROM cyq_chips WHERE {where_clause}"

        df = reader.query(sql, params if params else None)
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
    if not reader:
        raise HTTPException(status_code=503, detail="Database reader is not available.")
    try:
        # Handle trade_date parameter
        if trade_date:
            query_start = trade_date
            query_end = trade_date
        else:
            query_start = start_date
            query_end = end_date

        # If ts_code is not provided, use custom SQL
        if not ts_code:
            conditions = []
            params = []
            if query_start:
                conditions.append("trade_date >= ?")
                params.append(query_start)
            if query_end:
                conditions.append("trade_date <= ?")
                params.append(query_end)

            where_clause = " AND ".join(conditions) if conditions else "1=1"
            field_list = fields if fields else "*"
            sql = f"SELECT {field_list} FROM stk_factor_pro WHERE {where_clause}"
            df = reader.query(sql, params if params else None)
        else:
            # Use DataReader method when ts_code is specified
            if not query_start:
                raise HTTPException(status_code=400, detail="start_date or trade_date is required")
            df = reader.get_stk_factor_pro(
                ts_code=ts_code,
                start_date=query_start,
                end_date=query_end
            )
            # Apply field filtering if specified
            if fields and not df.empty:
                field_list = [f.strip() for f in fields.split(',')]
                available_fields = [f for f in field_list if f in df.columns]
                if available_fields:
                    df = df[available_fields]

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
    if not reader:
        raise HTTPException(status_code=503, detail="Database reader is not available.")
    try:
        df = reader.get_trade_calendar(
            start_date=start_date,
            end_date=end_date,
            is_open=is_open
        )
        # Apply field filtering if specified
        if fields and not df.empty:
            field_list = [f.strip() for f in fields.split(',')]
            available_fields = [f for f in field_list if f in df.columns]
            if available_fields:
                df = df[available_fields]
        # Apply exchange filter if specified
        if exchange and not df.empty and 'exchange' in df.columns:
            df = df[df['exchange'] == exchange]
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
    if not reader:
        raise HTTPException(status_code=503, detail="Database reader is not available.")
    try:
        conditions = ["hs_type = ?"]
        params = [hs_type]
        if is_new:
            conditions.append("is_new = ?")
            params.append(is_new)

        where_clause = " AND ".join(conditions)
        field_list = fields if fields else "*"
        sql = f"SELECT {field_list} FROM hs_const WHERE {where_clause}"

        df = reader.query(sql, params)
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
    if not reader:
        raise HTTPException(status_code=503, detail="Database reader is not available.")
    try:
        # If ts_code is not provided, use custom SQL to query by exchange or get all
        if not ts_code:
            conditions = []
            params = []
            if exchange:
                conditions.append("exchange = ?")
                params.append(exchange)

            where_clause = " AND ".join(conditions) if conditions else "1=1"
            field_list = fields if fields else "*"
            sql = f"SELECT {field_list} FROM stock_company WHERE {where_clause}"
            df = reader.query(sql, params if params else None)
        else:
            # Use DataReader method when ts_code is specified
            df = reader.get_stock_company(ts_code=ts_code)
            # Apply exchange filter if specified
            if exchange and not df.empty and 'exchange' in df.columns:
                df = df[df['exchange'] == exchange]
            # Apply field filtering if specified
            if fields and not df.empty:
                field_list = [f.strip() for f in fields.split(',')]
                available_fields = [f for f in field_list if f in df.columns]
                if available_fields:
                    df = df[available_fields]

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
    if not reader:
        raise HTTPException(status_code=503, detail="Database reader is not available.")
    try:
        conditions = []
        params = []
        if ts_code:
            conditions.append("ts_code = ?")
            params.append(ts_code)
        if name:
            conditions.append("name LIKE ?")
            params.append(f"%{name}%")
        if market:
            conditions.append("market = ?")
            params.append(market)
        if publisher:
            conditions.append("publisher = ?")
            params.append(publisher)
        if category:
            conditions.append("category = ?")
            params.append(category)

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        field_list = fields if fields else "*"
        sql = f"SELECT {field_list} FROM index_basic WHERE {where_clause}"

        df = reader.query(sql, params if params else None)
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
    if not reader:
        raise HTTPException(status_code=503, detail="Database reader is not available.")
    try:
        # Construct trade_date from year and month (first day of the month)
        trade_date = f"{year}{month:02d}01"

        field_list = fields if fields else "*"
        sql = f"""
            SELECT {field_list} FROM index_weight
            WHERE index_code = ?
            AND trade_date LIKE ?
        """

        df = reader.query(sql, [index_code, f"{year}{month:02d}%"])
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
    if not reader:
        raise HTTPException(status_code=503, detail="Database reader is not available.")
    try:
        # Handle trade_date parameter (convert to start_date/end_date)
        if trade_date:
            query_start = trade_date
            query_end = trade_date
        else:
            query_start = start_date
            query_end = end_date

        # If ts_code is not provided, use custom SQL to get all stocks
        if not ts_code:
            conditions = []
            params = []
            if query_start:
                conditions.append("trade_date >= ?")
                params.append(query_start)
            if query_end:
                conditions.append("trade_date <= ?")
                params.append(query_end)

            where_clause = " AND ".join(conditions) if conditions else "1=1"
            field_list = fields if fields else "*"
            sql = f"SELECT {field_list} FROM daily_basic WHERE {where_clause}"
            df = reader.query(sql, params if params else None)
        else:
            # Use DataReader method when ts_code is specified
            if not query_start:
                raise HTTPException(status_code=400, detail="start_date or trade_date is required")
            df = reader.get_daily_basic(
                ts_code=ts_code,
                start_date=query_start,
                end_date=query_end
            )
            # Apply field filtering if specified
            if fields and not df.empty:
                field_list = [f.strip() for f in fields.split(',')]
                available_fields = [f for f in field_list if f in df.columns]
                if available_fields:
                    df = df[available_fields]

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
    if not reader:
        raise HTTPException(status_code=503, detail="Database reader is not available.")
    try:
        conditions = []
        params = []
        if ts_code:
            conditions.append("ts_code = ?")
            params.append(ts_code)
        if con_code:
            conditions.append("con_code = ?")
            params.append(con_code)
        if trade_date:
            conditions.append("trade_date = ?")
            params.append(trade_date)

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        field_list = fields if fields else "*"
        sql = f"SELECT {field_list} FROM dc_member WHERE {where_clause}"

        df = reader.query(sql, params if params else None)
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
    if not reader:
        raise HTTPException(status_code=503, detail="Database reader is not available.")
    try:
        conditions = []
        params = []
        if ts_code:
            conditions.append("ts_code = ?")
            params.append(ts_code)
        if trade_date:
            conditions.append("trade_date = ?")
            params.append(trade_date)
        if start_date:
            conditions.append("trade_date >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("trade_date <= ?")
            params.append(end_date)

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        field_list = fields if fields else "*"
        sql = f"SELECT {field_list} FROM dc_index WHERE {where_clause}"

        df = reader.query(sql, params if params else None)
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
    Note: This endpoint requires custom implementation in DataReader.
    """
    if not reader:
        raise HTTPException(status_code=503, detail="Database reader is not available.")
    try:
        # Check if reader has this method
        if hasattr(reader, 'get_top_n_sector_members'):
            result = reader.get_top_n_sector_members(
                start_date=start_date,
                end_date=end_date,
                top_n=top_n,
                sort_by=sort_by,
                ascending=ascending
            )
            return json.dumps(result)
        else:
            # Fallback: simple custom SQL query
            raise HTTPException(
                status_code=501,
                detail="get_top_n_sector_members not implemented in DataReader yet"
            )
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
    if not reader:
        raise HTTPException(status_code=503, detail="Database reader is not available.")
    try:
        conditions = []
        params = []
        if ts_code:
            conditions.append("ts_code = ?")
            params.append(ts_code)
        if trade_date:
            conditions.append("trade_date = ?")
            params.append(trade_date)
        if start_date:
            conditions.append("trade_date >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("trade_date <= ?")
            params.append(end_date)

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        field_list = fields if fields else "*"
        sql = f"SELECT {field_list} FROM cyq_perf WHERE {where_clause}"

        df = reader.query(sql, params if params else None)
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
    if not reader:
        raise HTTPException(status_code=503, detail="Database reader is not available.")
    try:
        conditions = []
        params = []
        if ts_code:
            conditions.append("ts_code = ?")
            params.append(ts_code)
        if ann_date:
            conditions.append("ann_date = ?")
            params.append(ann_date)
        if start_date:
            conditions.append("ann_date >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("ann_date <= ?")
            params.append(end_date)
        if period:
            conditions.append("period = ?")
            params.append(period)

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        field_list = fields if fields else "*"
        sql = f"SELECT {field_list} FROM fina_indicator_vip WHERE {where_clause}"

        df = reader.query(sql, params if params else None)
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
    if not reader:
        raise HTTPException(status_code=503, detail="Database reader is not available.")
    try:
        # Handle trade_date parameter
        if trade_date:
            query_start = trade_date
            query_end = trade_date
        else:
            query_start = start_date
            query_end = end_date

        # If ts_code is not provided, use custom SQL
        if not ts_code:
            conditions = []
            params = []
            if query_start:
                conditions.append("trade_date >= ?")
                params.append(query_start)
            if query_end:
                conditions.append("trade_date <= ?")
                params.append(query_end)

            where_clause = " AND ".join(conditions) if conditions else "1=1"
            field_list = fields if fields else "*"
            sql = f"SELECT {field_list} FROM adj_factor WHERE {where_clause}"
            df = reader.query(sql, params if params else None)
        else:
            # Use DataReader method when ts_code is specified
            if not query_start:
                raise HTTPException(status_code=400, detail="start_date or trade_date is required")
            df = reader.get_adj_factor(
                ts_code=ts_code,
                start_date=query_start,
                end_date=query_end
            )
            # Apply field filtering if specified
            if fields and not df.empty:
                field_list = [f.strip() for f in fields.split(',')]
                available_fields = [f for f in field_list if f in df.columns]
                if available_fields:
                    df = df[available_fields]

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
    New architecture: Uses DataReader for query-only operations.
    """
    if not reader:
        raise HTTPException(status_code=503, detail="Database reader is not available.")
    try:
        # Determine adjustment type based on adjfactor parameter
        # In old architecture: adjfactor=True meant apply adjustment
        # In new architecture: adj='qfq' for forward adjustment
        adj = 'qfq' if adjfactor else None

        # Get stock daily data using DataReader
        df = reader.get_stock_daily(
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            adj=adj
        )

        # Calculate moving averages if requested
        if ma and not df.empty:
            # Sort by trade_date to ensure correct MA calculation
            df = df.sort_values('trade_date')
            for window in ma:
                df[f'ma{window}'] = df['close'].rolling(window=window).mean()

        return df_to_json_response(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/listing_first_day_info")
async def get_listing_first_day_info(
    ts_code: Optional[str] = None,
    list_status: Optional[str] = None,
    market: Optional[str] = None,
    include_no_data: bool = False
):
    """
    API endpoint to get listing first day information for stocks.
    
    - If ts_code is provided, returns info for that specific stock
    - Otherwise, returns info for all stocks (with optional filtering)
    
    Args:
        ts_code: Stock code (optional). If provided, returns single stock info.
        list_status: Listing status filter ('L'=listed, 'D'=delisted, 'P'=paused)
        market: Market filter (e.g., '主板', '创业板', '科创板', '北交所')
        include_no_data: Whether to include stocks without first day trading data
    
    Returns:
        JSON array containing:
        - ts_code: Stock code
        - name: Stock name
        - list_date: Listing date
        - market: Market type
        - list_status: Listing status
        - open, high, low, close: First day prices
        - vol, amount: First day volume and amount
    """
    if not reader:
        raise HTTPException(status_code=503, detail="Database reader is not available.")
    
    try:
        if ts_code:
            # Get info for specific stock
            df = reader.get_listing_first_day_info(ts_code=ts_code)
        else:
            # Get info for all stocks
            df = reader.get_all_listing_first_day_info(
                list_status=list_status,
                market=market,
                include_no_data=include_no_data
            )
        
        return df_to_json_response(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

