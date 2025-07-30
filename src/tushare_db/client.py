import logging
import os
import datetime
import pandas as pd
from typing import Any, Dict, Optional
from datetime import datetime, timedelta, date
from tqdm import tqdm
import math

from .tushare_fetcher import TushareFetcher, TushareClientError
from .duckdb_manager import DuckDBManager, DuckDBManagerError
from .cache_policies import BaseCachePolicy, FullCachePolicy, IncrementalCachePolicy
from .rate_limit_config import PRESET_PROFILES, STANDARD_PROFILE

# Default cache policies if none are provided by the user.
# This configuration externalizes the caching logic for each API.
DEFAULT_CACHE_POLICY_CONFIG = {
    "stock_basic": {"type": "full", "ttl": 60 * 60 * 24 * 7},  # 7 days
    "trade_cal": {"type": "incremental", "date_col": "cal_date"},
    "daily": {"type": "incremental", "date_col": "trade_date"},
    "pro_bar": {"type": "incremental", "date_col": "trade_date"},
    "hs_const": {"type": "full", "ttl": 60 * 60 * 24 * 30}, # 30 days
    "stock_company": {"type": "full", "ttl": 60 * 60 * 24 * 30}, # 30 days
    "cyq_perf": {"type": "incremental", "date_col": "trade_date"},
    "cyq_chips": {"type": "incremental", "date_col": "trade_date"},
    "dc_member": {"type": "incremental", "date_col": "trade_date"},
    "dc_index": {"type": "incremental", "date_col": "trade_date"},
    "stk_factor_pro": {"type": "incremental", "date_col": "trade_date"},
    # Add other API policies here...
}

# Maps policy 'type' strings to their respective classes.
CACHE_POLICY_MAPPING = {
    "full": FullCachePolicy,
    "incremental": IncrementalCachePolicy,
}

class TushareDBClientError(Exception):
    """Custom exception for TushareDBClient errors."""
    pass

class TushareDBClient:
    """
    A refactored client for TushareDB that coordinates data fetching and caching.
    It uses the Strategy Pattern to handle different caching policies.
    """

    def __init__(
        self,
        tushare_token: Optional[str] = None,
        db_path: str = "tushare.db",
        rate_limit_config: Optional[Dict[str, Any]] = None,
        rate_limit_profile: str = "standard",
        cache_policy_config: Optional[Dict[str, Dict[str, Any]]] = None,
        log_level: int = logging.INFO,
    ):
        """
        Initializes the TushareDBClient.

        Args:
            tushare_token: Your Tushare Pro API token.
            db_path: Path to the DuckDB database file.
            rate_limit_config: A custom dictionary defining detailed rate limits for APIs.
                               This overrides `rate_limit_profile`.
            rate_limit_profile: The name of a preset rate limit profile ('trial', 'standard', 'pro').
                                Defaults to 'standard'.
            cache_policy_config: A dictionary defining caching strategies for APIs.
                                 If None, uses DEFAULT_CACHE_POLICY_CONFIG.
            log_level: The logging level to use.
        """
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
        )

        self.tushare_token = tushare_token or os.getenv("TUSHARE_TOKEN")
        if not self.tushare_token:
            raise TushareDBClientError("Tushare token not provided.")

        # Determine the final rate limit configuration based on priority
        final_rate_limit_config = None
        if rate_limit_config:
            final_rate_limit_config = rate_limit_config
            logging.info("Using custom rate limit configuration.")
        elif rate_limit_profile in PRESET_PROFILES:
            final_rate_limit_config = PRESET_PROFILES[rate_limit_profile]
            logging.info(f"Using '{rate_limit_profile}' preset rate limit profile.")
        else:
            # Fallback to a default profile if an invalid profile name is given
            final_rate_limit_config = STANDARD_PROFILE
            logging.warning(f"Invalid rate_limit_profile '{rate_limit_profile}'. Falling back to 'standard' profile.")

        if "default" not in final_rate_limit_config:
            raise TushareDBClientError("Rate limit configuration must contain a 'default' key.")

        try:
            self.tushare_fetcher = TushareFetcher(self.tushare_token, final_rate_limit_config)
            self.duckdb_manager = DuckDBManager(db_path)
        except (TushareClientError, DuckDBManagerError, ValueError) as e:
            raise TushareDBClientError(f"Initialization failed: {e}") from e

        self.cache_policy_config = cache_policy_config or DEFAULT_CACHE_POLICY_CONFIG
        logging.info("TushareDBClient initialized successfully.")

    def get_data(self, api_name: str, **params: Any) -> pd.DataFrame:
        """
        Fetches data for a given Tushare API, using the configured cache policy.

        This method demonstrates the Strategy Pattern. It selects the appropriate
        caching policy at runtime based on the `api_name` and delegates the
        data fetching and caching logic to that policy object.

        Args:
            api_name: The name of the Tushare API (e.g., 'daily').
            **params: Keyword arguments for the Tushare API call.

        Returns:
            A pandas DataFrame with the requested data.
        """
        policy_info = self.cache_policy_config.get(api_name)
        if not policy_info:
            raise TushareDBClientError(f"No cache policy configured for API: '{api_name}'")

        policy_type = policy_info.get("type")
        policy_class = CACHE_POLICY_MAPPING.get(policy_type)

        if not policy_class:
            raise TushareDBClientError(f"Invalid cache policy type: '{policy_type}' for API: '{api_name}'")

        # Instantiate the selected cache policy strategy
        policy_strategy: BaseCachePolicy = policy_class(
            api_name=api_name,
            duckdb_manager=self.duckdb_manager,
            tushare_fetcher=self.tushare_fetcher,
            **policy_info,
        )

        try:
            # Delegate the work to the policy object
            return policy_strategy.get_data(**params)
            
        except (TushareClientError, DuckDBManagerError, ValueError) as e:
            logging.error(f"Error getting data for '{api_name}': {e}")
            raise TushareDBClientError(f"Failed to get data for '{api_name}': {e}") from e

    def get_all_stock_qfq_daily_bar(
        self, start_date: str, end_date: str, batch_size: int = 1
    ) -> pd.DataFrame:
        """
        Efficiently fetches forward-adjusted daily bar data for ALL stocks.

        This method is optimized to fetch data in batches to avoid hitting API
        rate limits and to improve performance significantly compared to fetching
        one stock at a time.

        Args:
            start_date: Start date (YYYYMMDD).
            end_date: End date (YYYYMMDD).
            batch_size: Number of stocks to fetch in a single API call.

        Returns:
            A DataFrame containing the combined daily bar data for all stocks.
        """
        logging.info(f"Fetching all stock QFQ daily bars from {start_date} to {end_date}.")
        
        # 1. Get the list of all stocks
        all_stocks_df = self.get_data("stock_basic", list_status="L")
        if all_stocks_df.empty:
            logging.warning("No stock basic data found. Cannot fetch pro_bar data.")
            return pd.DataFrame()
        
        ts_codes = all_stocks_df["ts_code"].tolist()
        all_bar_data = []
        
        # 2. Process stocks in batches
        num_batches = math.ceil(len(ts_codes) / batch_size)
        
        for i in tqdm(range(num_batches), desc="Fetching stock bars in batches"):
            batch_start = i * batch_size
            batch_end = batch_start + batch_size
            batch_codes = ts_codes[batch_start:batch_end]
            
            try:
                # 3. Make a single API call for the entire batch
                df = self.get_data(
                    "pro_bar",
                    ts_code=",".join(batch_codes),
                    start_date=start_date,
                    end_date=end_date,
                    adj="qfq",
                    freq="D",
                    asset="E",
                )
                if not df.empty:
                    all_bar_data.append(df)
                    logging.debug(f"Fetched batch {i+1}/{num_batches}: {len(df)} rows.")
            except TushareDBClientError as e:
                logging.error(f"Failed to fetch batch {i+1}/{num_batches}: {e}")

        if not all_bar_data:
            logging.warning("No daily bar data found for any stock.")
            return pd.DataFrame()

        final_df = pd.concat(all_bar_data, ignore_index=True)
        logging.info(f"Successfully fetched QFQ daily bars for {len(ts_codes)} stocks, total {len(final_df)} rows.")
        return final_df

    def get_latest_common_date(self, api_name: str, date_col: str = 'trade_date') -> Optional[str]:
        """
        获取某个接口的本地数据库中,大部分股票的最新数据日期。
        通过计算每个ts_code的最新日期，然后返回出现次数最多的那个日期。

        :param api_name: Tushare接口名称，也对应数据库中的表名。
        :param date_col: 日期列的名称。
        :return: 出现次数最多的最新日期字符串，如果表不存在或无数据则返回None。
        """
        if not self.duckdb_manager.table_exists(api_name):
            logging.warning(f"表 '{api_name}' 不存在。")
            return None

        columns = self.duckdb_manager.get_table_columns(api_name)
        if date_col not in columns:
            # 常见的时间列
            common_date_cols = ["trade_date", "end_date", "ann_date"]
            actual_date_col = next((col for col in common_date_cols if col in columns), None)

            if not actual_date_col:
                logging.warning(f"在表 '{api_name}' 中未找到指定的日期列 '{date_col}' 或常见的日期列。")
                return None
            
            logging.info(f"指定的日期列 '{date_col}' 不存在，将使用找到的日期列 '{actual_date_col}'。")
            date_col = actual_date_col


        query = f"""
        WITH LatestDates AS (
            SELECT MAX({date_col}) AS last_date
            FROM {api_name}
            GROUP BY ts_code
        )
        SELECT last_date
        FROM LatestDates
        WHERE last_date IS NOT NULL
        GROUP BY last_date
        ORDER BY COUNT(*) DESC
        LIMIT 1;
        """
        try:
            result = self.duckdb_manager.execute_query(query)
            if result is not None and not result.empty:
                latest_date = result.iloc[0, 0]
                if isinstance(latest_date, (date, datetime)):
                    return latest_date.strftime('%Y%m%d')
                return str(latest_date)
            else:
                logging.info(f"在表 '{api_name}' 中没有找到符合条件的数据。")
                return None
        except Exception as e:
            logging.error(f"为 '{api_name}' 获取最新通用日期时出错: {e}")
            return None

    def close(self) -> None:
        """Closes the DuckDB database connection."""
        try:
            self.duckdb_manager.close()
            logging.info("TushareDBClient closed DuckDB connection.")
        except DuckDBManagerError as e:
            raise TushareDBClientError(f"Failed to close DuckDB connection: {e}") from e

if __name__ == '__main__':
    print("\n=====================================================")
    print("TushareDBClient Refactored Quick Start Example")
    print("=====================================================")
    
    try:
        # Initialize the client
        client = TushareDBClient(tushare_token='mock_token', log_level=logging.DEBUG)
        print("Client initialized successfully.")

        # --- Example 1: Incremental Cache ---
        print("\n--- Fetching daily data (Incremental Cache) ---")
        daily_df = client.get_data('daily', ts_code='000001.SZ', start_date='20230101', end_date='20230115')
        print(f"Fetched daily data for 000001.SZ: {len(daily_df)} rows")
        
        print("\n--- Fetching more recent daily data (should be incremental) ---")
        daily_df_new = client.get_data('daily', ts_code='000001.SZ', start_date='20230101', end_date='20230131')
        print(f"Fetched updated daily data: {len(daily_df_new)} rows")

        # --- Example 2: Full Cache with TTL ---
        print("\n--- Fetching stock basic data (Full Cache with TTL) ---")
        stock_basic_df = client.get_data('stock_basic', list_status='L')
        print(f"Fetched stock basic data: {len(stock_basic_df)} rows")

        print("\n--- Fetching stock basic data again (should be from cache) ---")
        stock_basic_cached_df = client.get_data('stock_basic', list_status='L')
        print(f"Fetched stock basic data from cache: {len(stock_basic_cached_df)} rows")
        
        # --- Example 3: Efficient Batch Fetching ---
        print("\n--- Fetching all stock bars for a short period (Batch Operation) ---")
        # Note: This will still make multiple API calls depending on batch_size
        all_bars_df = client.get_all_stock_qfq_daily_bar(start_date='20230101', end_date='20230105')
        print(f"Fetched {len(all_bars_df)} total bar records for all stocks.")

    except TushareDBClientError as e:
        print(f"\nAn error occurred: {e}")
    finally:
        if 'client' in locals():
            client.close()
            print("\nDatabase connection closed.")