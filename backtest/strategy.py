import vectorbt as vbt
import pandas as pd
from datetime import datetime, timedelta
from tushare_db import DataReader
import tushare as ts

# Initialize the DataReader (new architecture - high performance, read-only)
reader = DataReader()
pro = ts.pro_api()

def get_top_sectors(trade_date: str, top_n: int, moneyflow_ind_dc_df: pd.DataFrame):
    """
    Gets the top N sectors based on net money inflow for a specific date.

    Args:
        trade_date (str): The date for which to calculate money flow (YYYYMMDD).
        top_n (int): The number of top industries to return.
        moneyflow_ind_dc_df (pd.DataFrame): Pre-fetched money flow data for the entire period.

    Returns:
        A list of the top N sector ts_codes.
    """
    try:
        # 1. Filter money flow data for the current trade_date
        daily_money_flow = moneyflow_ind_dc_df[moneyflow_ind_dc_df['trade_date'] == trade_date]
        if daily_money_flow.empty:
            print(f"No money flow data available for {trade_date} in pre-fetched data.")
            return []

        # 2. Sort by net amount and get top N sector codes
        top_sectors = daily_money_flow.sort_values('net_amount', ascending=False).head(top_n)
        top_sector_codes = top_sectors['ts_code'].tolist()
        
        print(f"[DEBUG] Top {top_n} sectors by money inflow on {trade_date}: {top_sectors['name'].tolist()}")
        print(f"[DEBUG] Found {len(top_sector_codes)} top sectors for {trade_date}.")
        return top_sector_codes

    except Exception as e:
        print(f"An error occurred in get_top_sectors: {e}")
        return []


def get_stock_list(trade_date: str, top_sector_codes: list, reader, daily_basic_df: pd.DataFrame, stock_basic_df: pd.DataFrame):
    """
    Filters stocks based on market cap and industry.

    Args:
        trade_date (str): The date for filtering (YYYYMMDD).
        top_sector_codes (list): A list of target sector ts_codes.
        daily_basic_df (pd.DataFrame): Pre-fetched daily basic data for the entire period.
        stock_basic_df (pd.DataFrame): Pre-fetched stock basic information.

    Returns:
        A list of filtered stock ts_codes.
    """
    try:
        # 1. Initial pool: Get all constituent stocks from the top sectors
        initial_pool = set()
        for sector_code in top_sector_codes:
            # DataReader provides fast, read-only access (no network calls)
            members = reader.query(
                "SELECT * FROM dc_member WHERE ts_code = ?",
                [sector_code]
            )
            if not members.empty:
                initial_pool.update(members['con_code'].tolist())
        
        if not initial_pool:
            print(f"[DEBUG] Could not find any constituent stocks for top sectors on {trade_date}.")
            return []
        
        initial_pool = list(initial_pool)
        print(f"[DEBUG] Initial pool size after getting sector constituents: {len(initial_pool)}")

        # 2. First layer of filtering using pre-fetched stock_basic_df
        stock_basic_filtered = stock_basic_df.copy()
        
        # Exclude new stocks (listed less than 375 days)
        today = datetime.strptime(trade_date, '%Y%m%d')
        min_list_date = (today - timedelta(days=180)).strftime('%Y%m%d')
        stock_basic_filtered = stock_basic_filtered[stock_basic_filtered['list_date'] < min_list_date]
        
        # Exclude STAR, ChiNext, BSE stocks
        stock_basic_filtered = stock_basic_filtered[~stock_basic_filtered['market'].isin(['北交所'])]
        
        # Exclude ST stocks
        stock_basic_filtered = stock_basic_filtered[~stock_basic_filtered['name'].str.contains('ST')]
        
        # Apply filters to the initial pool
        filtered_pool = list(set(initial_pool) & set(stock_basic_filtered['ts_code']))
        print(f"[DEBUG] Pool size after basic stock filters (list date, market, ST): {len(filtered_pool)}")

        # 3. Market Cap filtering from pre-fetched daily_basic_df
        current_daily_basic = daily_basic_df[daily_basic_df['trade_date'] == trade_date]
        if current_daily_basic.empty:
            print(f"No daily basic data for {trade_date} in pre-fetched data.")
            return []
            
        current_daily_basic = current_daily_basic[current_daily_basic['ts_code'].isin(filtered_pool)]
        print(f"[DEBUG] Pool size after filtering by daily_basic_df and filtered_pool: {len(current_daily_basic)}")
        
        # Sort by circulating market cap and take top 1000
        small_circ_mv = current_daily_basic.sort_values('circ_mv').head(1000)
        print(f"[DEBUG] Pool size after sorting by circ_mv and taking top 1000: {len(small_circ_mv)}")
        
        # Sort by total market cap and take top 500
        small_total_mv = small_circ_mv.sort_values('total_mv').head(500)
        print(f"[DEBUG] Pool size after sorting by total_mv and taking top 500: {len(small_total_mv)}")
        
        final_stock_list = small_total_mv['ts_code'].tolist()
        
        print(f"[DEBUG] Found {len(final_stock_list)} candidate stocks after market cap filtering.")
        return final_stock_list

    except Exception as e:
        print(f"An error occurred in get_stock_list: {e}")
        return []

def weekly_adjustment(current_date, context, reader, moneyflow_ind_dc_df: pd.DataFrame, daily_basic_df: pd.DataFrame, stock_basic_df: pd.DataFrame):
    """
    The main logic function to be run weekly.
    """
    trade_date_str = current_date.strftime('%Y%m%d')
    print(f"\n----- Running weekly adjustment for {trade_date_str} ----- ")
    
    # Step 1: Get top sectors
    top_sector_codes = get_top_sectors(trade_date_str, 10, moneyflow_ind_dc_df)
    if not top_sector_codes:
        print(f"[DEBUG] No top sectors found for {trade_date_str}. Skipping stock selection.")
        return [] # Return empty list if no sectors found

    # Step 2: Get stock list from those sectors
    candidate_stocks = get_stock_list(trade_date_str, top_sector_codes, reader, daily_basic_df, stock_basic_df)
    if not candidate_stocks:
        print(f"[DEBUG] No candidate stocks found for {trade_date_str} after filtering. Skipping final selection.")
        return []

    # Step 3: Final filtering and selection (up to 6 stocks)
    final_selection = candidate_stocks[:6] # Select up to 6 stocks
    print(f"[DEBUG] Final selection for {trade_date_str}: {final_selection}")
    return final_selection


def run_backtest(start_date_str, end_date_str):
    """
    Sets up and runs the vectorbt backtest.
    """
    # Define the date range for the backtest
    start_date = pd.to_datetime(start_date_str)
    end_date = pd.to_datetime(end_date_str)
    
    # Pre-fetch all necessary data for the entire backtest period
    print(f"Pre-fetching moneyflow_ind_dc data from {start_date_str} to {end_date_str}...")
    moneyflow_ind_dc_df = reader.get_moneyflow_ind_dc(start_date_str, end_date_str)
    if moneyflow_ind_dc_df.empty:
        print("Could not fetch moneyflow_ind_dc data. Exiting.")
        return
    print("moneyflow_ind_dc_df head:\n", moneyflow_ind_dc_df.head())
    print("moneyflow_ind_dc_df tail:\n", moneyflow_ind_dc_df.tail())

    print(f"Pre-fetching daily_basic data from {start_date_str} to {end_date_str}...")
    # Using custom SQL query to select specific fields
    daily_basic_df = reader.query(
        "SELECT ts_code, trade_date, circ_mv, total_mv FROM daily_basic WHERE trade_date BETWEEN ? AND ?",
        [start_date_str, end_date_str]
    )
    if daily_basic_df.empty:
        print("Could not fetch daily_basic data. Exiting.")
        return
    print("daily_basic_df head:\n", daily_basic_df.head())
    print("daily_basic_df tail:\n", daily_basic_df.tail())

    print("Pre-fetching stock_basic data...")
    stock_basic_df = reader.query("SELECT ts_code, list_date, market, name FROM stock_basic")
    if stock_basic_df.empty:
        print("Could not fetch stock_basic data. Exiting.")
        return
    print("stock_basic_df head:\n", stock_basic_df.head())
    print("stock_basic_df tail:\n", stock_basic_df.tail())

    # Fetch price data for the simulation
    # We'll use a broad market index for simulation purposes to keep it simple and fast.
    print(f"Fetching price data for market index from {start_date_str} to {end_date_str}...")
    # Note: For index data, using custom SQL query (DataReader is for stocks by default)
    price_df = reader.query(
        "SELECT * FROM pro_bar WHERE ts_code = ? AND trade_date BETWEEN ? AND ?",
        ['399101.SZ', start_date_str, end_date_str]
    )
    if price_df.empty:
        print("Could not fetch price data for the backtest. Exiting.")
        return

    price = price_df.set_index('trade_date')['close']
    price.index = pd.to_datetime(price.index)

    # Generate signals on a weekly basis
    resampled_dates = price.index
    print(f"[DEBUG] Unique trade dates in resampled_dates (price.index): {[d.strftime('%Y%m%d') for d in resampled_dates.unique().tolist()]}")

    # Filter resampled_dates to only include dates present in daily_basic_df
    daily_basic_trade_dates = pd.to_datetime(daily_basic_df['trade_date'].unique()).normalize()
    resampled_dates = resampled_dates[resampled_dates.normalize().isin(daily_basic_trade_dates)]
    print(f"[DEBUG] Filtered resampled_dates (after daily_basic_df check): {[d.strftime('%Y%m%d') for d in resampled_dates.unique().tolist()]}")
    
    entries = pd.Series(False, index=price.index)
    exits = pd.Series(False, index=price.index)
    
    context = {} # A simple context dict, can be expanded if needed
    
    for date in resampled_dates:
        print(f"[DEBUG] Processing date: {date.strftime('%Y%m%d')}")
        if date in price.index:
            selected_stocks = weekly_adjustment(date, context, reader, moneyflow_ind_dc_df, daily_basic_df, stock_basic_df)
            if selected_stocks:
                print(f"[DEBUG] Buy signal generated for {date.strftime('%Y%m%d')} with stocks: {selected_stocks}")
                entries.loc[date] = True
                # Assume we hold for a week
                exit_date_iloc = price.index.get_loc(date) + 5
                if exit_date_iloc < len(price.index):
                    exits.iloc[exit_date_iloc] = True
                    print(f"[DEBUG] Exit signal generated for {price.index[exit_date_iloc].strftime('%Y%m%d')}")

    # Run vectorbt portfolio simulation
    pf = vbt.Portfolio.from_signals(price, entries, exits, init_cash=100000, freq='D')

    # Print results
    print("\n----- Backtest Results -----")
    print(pf.stats())
    pf.plot().show()

    # Close the DataReader connection
    reader.close()
    print("\nDataReader connection closed.")


if __name__ == "__main__":
    # Define the backtest period
    run_backtest(start_date_str='20251025', end_date_str='20251031')