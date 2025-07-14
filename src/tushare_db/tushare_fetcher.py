

import tushare as ts
import pandas as pd
import time
import threading
import collections
import logging
from typing import Any, Deque, Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s -[%(filename)s:%(lineno)d]- %(message)s')

class TushareClientError(Exception):
    """Custom exception for TushareClient errors."""
    pass

class TushareFetcher:
    """
    A client for interacting with the Tushare Pro API, enforcing rate limits.
    """

    def __init__(self, token: str, rate_limit_config: Dict[str, Any]):
        """
        Initializes the TushareFetcher with a token and a detailed rate limit configuration.

        Args:
            token: Your Tushare Pro API token.
            rate_limit_config: A dictionary defining rate limits for APIs.
                Example:
                {
                    "default": {"limit": 500, "period": "minute"},
                    "daily": {"limit": 200, "period": "day"},
                    "pro_bar": {"limit": 1000, "period": "minute"}
                }
        """
        if not token:
            raise ValueError("Tushare token cannot be empty.")

        self.token: str = token
        self.pro: ts.pro_api = ts.pro_api(self.token)
        ts.set_token(self.token)

        self._lock: threading.Lock = threading.Lock()
        self._api_call_timestamps: Dict[str, Deque[float]] = collections.defaultdict(collections.deque)
        
        # Parse and store rate limit configuration
        self.rate_limit_config = {}
        period_map = {"minute": 60, "day": 86400}
        for api, config in rate_limit_config.items():
            if "limit" not in config or "period" not in config:
                raise ValueError(f"Invalid rate limit config for '{api}'. Must include 'limit' and 'period'.")
            period_seconds = period_map.get(config["period"])
            if not period_seconds:
                raise ValueError(f"Invalid period '{config['period']}' for '{api}'. Use 'minute' or 'day'.")
            self.rate_limit_config[api] = {
                "limit": config["limit"],
                "period_seconds": period_seconds
            }
        
        if "default" not in self.rate_limit_config:
            raise ValueError("rate_limit_config must contain a 'default' key.")

        logging.info(f"TushareFetcher initialized with rate limit config: {self.rate_limit_config}")

    def fetch(self, api_name: str, **params: Any) -> pd.DataFrame:
        """
        Fetches data from the Tushare Pro API, respecting per-API rate limits.
        Includes special handling for 'pro_bar' to iterate over multiple ts_codes.

        Args:
            api_name: The name of the Tushare API interface (e.g., 'daily', 'pro_bar').
            **params: Keyword arguments to pass to the Tushare API query method.

        Returns:
            A Pandas DataFrame containing the fetched data.

        Raises:
            TushareClientError: If there is an error during the API call or data retrieval.
        """
        # Special handling for pro_bar with multiple ts_codes
        if api_name == 'pro_bar' and 'ts_code' in params:
            ts_codes = params['ts_code']
            if isinstance(ts_codes, str) and ',' in ts_codes:
                ts_codes = ts_codes.split(',')
            
            if isinstance(ts_codes, list) and len(ts_codes) > 1:
                logging.info(f"Fetching 'pro_bar' for {len(ts_codes)} stocks in a loop.")
                all_data = []
                base_params = params.copy()
                for code in ts_codes:
                    base_params['ts_code'] = code
                    # We call the main fetch method recursively, which will handle rate limiting for each call
                    df_single = self.fetch(api_name, **base_params)
                    if not df_single.empty:
                        all_data.append(df_single)
                
                if not all_data:
                    return pd.DataFrame()
                return pd.concat(all_data, ignore_index=True)

        # Default behavior for all other APIs or pro_bar with a single ts_code
        self._wait_for_rate_limit(api_name)

        try:
            logging.info(f"Fetching data for API: {api_name} with params: {params}")
            
            if api_name == 'pro_bar':
                api_func = ts.pro_bar
            else:
                api_func = getattr(self.pro, api_name, None)
                if api_func is None:
                    api_func = lambda **p: self.pro.query(api_name, **p)

            df = api_func(**params)

            if df is None:
                raise TushareClientError(f"Tushare API returned None for {api_name}. Check parameters or token.")
            
            if not df.empty and 'code' in df.columns and 'msg' in df.columns:
                error_code = df['code'].iloc[0]
                if error_code != 0 and str(error_code) != '0':
                    error_msg = df['msg'].iloc[0]
                    raise TushareClientError(f"Tushare API error for {api_name}: Code {error_code}, Message: {error_msg}")
            
            with self._lock:
                self._api_call_timestamps[api_name].append(time.time())
            # logging.error(df)
            logging.info(f"Successfully fetched {len(df)} rows for API: {api_name}.")
            return df

        except Exception as e:
            if isinstance(e, TushareClientError):
                raise
            logging.error(f"Error fetching data for {api_name}: {e}")
            raise TushareClientError(f"Failed to fetch data from Tushare API for {api_name}: {e}") from e

    def _wait_for_rate_limit(self, api_name: str) -> None:
        """
        Waits if necessary to ensure the API rate limit for the given API is not exceeded.
        This method is thread-safe.
        """
        with self._lock:
            config = self.rate_limit_config.get(api_name, self.rate_limit_config["default"])
            limit = config["limit"]
            period_seconds = config["period_seconds"]
            
            timestamps = self._api_call_timestamps[api_name]
            current_time = time.time()

            # Remove timestamps older than the period
            while timestamps and timestamps[0] <= current_time - period_seconds:
                timestamps.popleft()

            # If the limit is reached, calculate wait time and sleep
            if len(timestamps) >= limit:
                oldest_request_time = timestamps[0]
                time_to_wait = (oldest_request_time + period_seconds) - current_time

                if time_to_wait > 0:
                    logging.info(f"Rate limit for '{api_name}' reached. Waiting for {time_to_wait:.2f} seconds...")
                    time.sleep(time_to_wait)
                    # After sleeping, re-check and remove expired timestamps
                    new_current_time = time.time()
                    while timestamps and timestamps[0] <= new_current_time - period_seconds:
                        timestamps.popleft()


