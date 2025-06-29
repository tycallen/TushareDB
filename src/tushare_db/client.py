
import os
import time
import pandas as pd
import logging
from typing import Any, Dict, Optional, Union

from tushare_db.tushare_client import TushareClient, TushareClientError
from tushare_db.duckdb_manager import DuckDBManager, DuckDBManagerError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TushareDBClientError(Exception):
    """Custom exception for TushareDBClient errors."""
    pass

class TushareDBClient:
    """
    Main client for TushareDB, coordinating Tushare API calls with DuckDB caching.
    """

    def __init__(
        self,
        tushare_token: Optional[str] = None,
        db_path: str = 'tushare.db',
        rate_limit_per_minute: int = 500,
        cache_policy: Optional[Dict[str, Dict[str, Any]]] = None
    ):
        """
        Initializes the TushareDBClient.

        Args:
            tushare_token: Your Tushare Pro API token. If None, it tries to read from
                           the TUSHARE_TOKEN environment variable.
            db_path: Path to the DuckDB database file. Defaults to 'tushare.db'.
            rate_limit_per_minute: Tushare API call limit per minute. Defaults to 500.
            cache_policy: A dictionary defining caching strategies for different API names.
                          Example:
                          {
                              'daily': {'type': 'incremental', 'date_col': 'trade_date'},
                              'stock_basic': {'type': 'full', 'ttl': 60 * 60 * 24 * 7} # 7 days
                          }
        Raises:
            TushareDBClientError: If Tushare token is not provided.
        """
        self.tushare_token = tushare_token or os.getenv('TUSHARE_TOKEN')
        if not self.tushare_token:
            raise TushareDBClientError(
                "Tushare token not provided. Please pass it as an argument or set the TUSHARE_TOKEN environment variable."
            )

        try:
            self.tushare_client = TushareClient(self.tushare_token, rate_limit_per_minute)
            self.duckdb_manager = DuckDBManager(db_path)
        except (TushareClientError, DuckDBManagerError) as e:
            raise TushareDBClientError(f"Initialization failed: {e}") from e

        self.cache_policy = cache_policy or {
            'daily': {'type': 'incremental', 'date_col': 'trade_date'},
            'stock_basic': {'type': 'full', 'ttl': 60 * 60 * 24 * 7}, # 7 days
            'trade_cal': {'type': 'incremental', 'date_col': 'cal_date'}
        }
        logging.info("TushareDBClient initialized.")

    def _build_where_clause(self, api_name: str, params: Dict[str, Any]) -> str:
        """
        Builds a SQL WHERE clause from a dictionary of parameters,
        considering which parameters are actual database columns for a given API.
        """
        conditions = []
        # This needs to be maintained based on Tushare API return fields
        column_mappers = {
            'daily': ['ts_code', 'trade_date'],
            'stock_basic': ['ts_code', 'symbol', 'name', 'area', 'industry', 'market', 'list_status'],
            'trade_cal': ['exchange', 'cal_date', 'is_open']
        }
        
        table_name = api_name
        available_columns = self.duckdb_manager.get_table_columns(table_name)
        
        # Use only parameters that are valid for the API and exist in the table
        valid_params = {k: v for k, v in params.items() if k in column_mappers.get(api_name, []) and k in available_columns}

        for key, value in valid_params.items():
            if isinstance(value, str):
                conditions.append(f"{key} = '{value}'")
            elif isinstance(value, (int, float)):
                conditions.append(f"{key} = {value}")
        return " WHERE " + " AND ".join(conditions) if conditions else ""

    def get_data(self, api_name: str, **params: Any) -> pd.DataFrame:
        """
        Fetches data for a given Tushare API, utilizing a smart caching strategy.

        Args:
            api_name: The name of the Tushare API interface (e.g., 'daily', 'pro_bar').
            **params: Keyword arguments to pass to the Tushare API query method.

        Returns:
            A Pandas DataFrame containing the requested data, potentially from cache or API.

        Raises:
            TushareDBClientError: If data fetching or caching fails.
        """
        cache_info = self.cache_policy.get(api_name)
        table_name = api_name # Use api_name as table name for simplicity
        local_data_df = pd.DataFrame()
        fetch_from_api = True
        updated_params = params.copy()

        # Build a WHERE clause for initial DuckDB query based on known columns for this API
        where_clause_for_db = self._build_where_clause(api_name, params)

        try:
            if self.duckdb_manager.table_exists(table_name):
                local_data_df = self.duckdb_manager.execute_query(f"SELECT * FROM {table_name}{where_clause_for_db}")
                logging.info(f"Loaded {len(local_data_df)} rows from cache for {api_name}.")

                if cache_info and not local_data_df.empty:
                    if cache_info['type'] == 'incremental':
                        date_col = cache_info.get('date_col')
                        if date_col and date_col in local_data_df.columns:
                            latest_local_date = self.duckdb_manager.get_latest_date(table_name, date_col)
                            requested_start_date = params.get('start_date')
                            requested_end_date = params.get('end_date')

                            # If local data covers the requested end date, no need to fetch from API
                            # This check needs to be robust for incremental updates
                            if latest_local_date and requested_end_date and latest_local_date >= requested_end_date:
                                logging.info(f"Incremental cache for {api_name} covers requested range up to {requested_end_date}. Returning cached data.")
                                fetch_from_api = False
                            elif latest_local_date:
                                # If local data exists but doesn't cover the full requested range, fetch incrementally
                                updated_params['start_date'] = latest_local_date
                                logging.info(f"Incremental update: Latest local date for {api_name} is {latest_local_date}. Fetching data from {latest_local_date}.")
                                fetch_from_api = True
                            else:
                                logging.info(f"No latest date found for {api_name} in cache. Fetching all data.")
                                fetch_from_api = True
                        else:
                            logging.warning(f"Incremental policy for {api_name} missing or invalid 'date_col'. Fetching all data.")
                            fetch_from_api = True

                    elif cache_info['type'] == 'full':
                        ttl = cache_info.get('ttl')
                        if ttl is not None:
                            last_updated = self.duckdb_manager.get_cache_metadata(table_name)
                            if last_updated and (time.time() - last_updated) < ttl:
                                logging.info(f"Full cache for {api_name} is fresh. Returning cached data.")
                                fetch_from_api = False
                            else:
                                logging.info(f"Full cache for {api_name} is stale or not found. Fetching new data.")
                                fetch_from_api = True
                        else:
                            logging.warning(f"Full policy for {api_name} missing 'ttl'. Fetching all data.")
                            fetch_from_api = True
                elif local_data_df.empty:
                    logging.info(f"No data found in cache for {api_name}. Fetching from API.")
                    fetch_from_api = True
                else:
                    logging.info(f"No cache policy defined for {api_name}. Fetching from API.")
                    fetch_from_api = True
            else:
                logging.info(f"Table {table_name} does not exist in cache. Fetching from API.")
                fetch_from_api = True

            final_df = pd.DataFrame()
            if fetch_from_api:
                new_data_df = self.tushare_client.fetch(api_name, **updated_params)
                if not new_data_df.empty:
                    if cache_info and cache_info['type'] == 'incremental':
                        date_col = cache_info.get('date_col')
                        if date_col and date_col in new_data_df.columns:
                            # Ensure date column is string for comparison if it's not already
                            if date_col in local_data_df.columns:
                                local_data_df[date_col] = local_data_df[date_col].astype(str)
                            new_data_df[date_col] = new_data_df[date_col].astype(str)

                            # Filter out data from new_data_df that is already in local_data_df
                            # This is crucial for incremental updates to avoid duplicates
                            if not local_data_df.empty:
                                # Get the latest date from local_data_df to filter new_data_df
                                latest_local_date_for_filter = local_data_df[date_col].max()
                                new_data_df = new_data_df[new_data_df[date_col] > latest_local_date_for_filter]
                                logging.info(f"Filtered new data to only include records after {latest_local_date_for_filter}.")

                            if not new_data_df.empty:
                                combined_df = pd.concat([local_data_df, new_data_df]).drop_duplicates(subset=[date_col], keep='last') # Simplified deduplication
                                self.duckdb_manager.write_dataframe(combined_df, table_name, mode='replace')
                                logging.info(f"Appended and deduplicated {len(new_data_df)} new rows to {table_name}.")
                            else:
                                logging.info("No truly new data to append after filtering for incremental update.")
                                # If no new data after filtering, just return existing local data
                                final_df = local_data_df
                                fetch_from_api = False # No need to re-query if nothing new was written
                        else:
                            logging.warning(f"Incremental policy for {api_name} missing or invalid 'date_col'. Performing full replace.")
                            self.duckdb_manager.write_dataframe(new_data_df, table_name, mode='replace')
                            logging.info(f"Replaced cache for {table_name} with {len(new_data_df)} new rows.")
                    else: # Full refresh or no specific policy
                        self.duckdb_manager.write_dataframe(new_data_df, table_name, mode='replace')
                        logging.info(f"Replaced cache for {table_name} with {len(new_data_df)} new rows.")
                    
                    if cache_info and cache_info['type'] == 'full':
                        self.duckdb_manager.update_cache_metadata(table_name, time.time())
                    
                    # After writing, re-query to ensure we return the complete, updated local data
                    final_df = self.duckdb_manager.execute_query(f"SELECT * FROM {table_name}{where_clause_for_db}")
                else:
                    logging.info(f"No new data fetched for {api_name}. Returning existing local data if any.")
                    final_df = local_data_df
            else:
                final_df = local_data_df

            # Apply date and other API-specific filtering to the final DataFrame
            if 'trade_date' in final_df.columns:
                if 'start_date' in params:
                    final_df = final_df[final_df['trade_date'] >= params['start_date']]
                if 'end_date' in params:
                    final_df = final_df[final_df['trade_date'] <= params['end_date']]
            
            # Apply other API-specific filters if they are not part of the DB query
            # These are filters that Tushare API accepts but might not be direct columns in the returned DF
            if api_name == 'stock_basic':
                # Tushare's stock_basic returns 'market' and 'list_status', but not 'exchange' as a column.
                # We need to map 'exchange' parameter to 'market' column for filtering if needed.
                if 'exchange' in params and 'market' in final_df.columns:
                    # This mapping is specific to Tushare's stock_basic API and its returned columns
                    # Example: SSE (Shanghai Stock Exchange) maps to '主板' (Main Board) in 'market' column
                    # This might need a more robust mapping or direct check against Tushare docs
                    # For simplicity, let's assume a direct match for now if 'exchange' is intended to filter 'market'
                    # However, 'exchange' is often a top-level filter for Tushare, not a column in the result.
                    # If 'exchange' is truly not a column, this filter should be removed or handled differently.
                    # Based on the error, 'exchange' is not a column. So, this filter should be removed from here.
                    pass # Removed direct filtering by 'exchange' as it's not a column
                if 'list_status' in params and 'list_status' in final_df.columns:
                    final_df = final_df[final_df['list_status'] == params['list_status']]

            return final_df

        except (TushareClientError, DuckDBManagerError) as e:
            logging.error(f"Error in get_data for {api_name}: {e}")
            raise TushareDBClientError(f"Failed to get data for {api_name}: {e}") from e

    def close(self) -> None:
        """
        Closes the DuckDB database connection.
        """
        try:
            self.duckdb_manager.close()
            logging.info("TushareDBClient closed DuckDB connection.")
        except DuckDBManagerError as e:
            raise TushareDBClientError(f"Failed to close DuckDB connection: {e}") from e


if __name__ == '__main__':
    print("\n=====================================================")
    print("TushareDBClient Quick Start Example")
    print("=====================================================")
    print("Please ensure your TUSHARE_TOKEN environment variable is set.")
    print("Example: export TUSHARE_TOKEN=\"YOUR_TUSHARE_PRO_TOKEN\"")
    print("\nInitializing TushareDBClient...")

    try:
        # Initialize the client
        # The database will be created as 'tushare.db' in the current directory
        # You can specify a different path: db_path='data/my_tushare.db'
        client = TushareDBClient(tushare_token='mock_token')
        print("Client initialized successfully.")

        # --- Example 1: Fetching daily stock data (Incremental Update Policy) ---
        print("\n--- Fetching daily stock data (ts_code='000001.SZ') ---")
        print("First fetch: This might take a moment as data is fetched from Tushare API.")
        daily_df = client.get_data('daily', ts_code='000001.SZ', start_date='20230101', end_date='20230131')
        print(f"First fetch of daily data (000001.SZ): {len(daily_df)} rows")
        if not daily_df.empty:
            print(daily_df.head())
        else:
            print("No daily data fetched.")

        # Second fetch for the same data - should load from cache (incremental update)
        print("\n--- Second fetch of daily stock data (should be from cache) ---")
        print("Second fetch: This should be much faster as data is loaded from DuckDB cache.")
        daily_df_cached = client.get_data('daily', ts_code='000001.SZ', start_date='20230101', end_date='20230131')
        print(f"Second fetch of daily data (000001.SZ): {len(daily_df_cached)} rows (from cache)")

        # 第二次获取 - 如果在缓存有效期 (TTL) 内，将从缓存加载
        print("\n--- 第二次获取股票基础信息 (应从缓存加载) ---")
        print("第二次获取: 数据将从 DuckDB 缓存中加载，速度会快很多。")
        stock_basic_cached = client.get_stock_basic(list_status='L')
        print(f"第二次获取 stock_basic 数据: {len(stock_basic_cached.data)} 行 (来自缓存)")

    except TushareDBClientError as e:
        print(f"\n示例执行过程中发生错误: {e}")
    except Exception as e:
        print(f"\n发生未知错误: {e}")
    finally:
        # Close the database connection when done
        if 'client' in locals() and client.duckdb_manager.con:
            client.close()
            print("\nDatabase connection closed.")
        else:
            print("\nClient not fully initialized or connection already closed.")

