

import tushare as ts
import pandas as pd
import time
import threading
import collections
import logging
from typing import Any, Deque

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s -[%(filename)s:%(lineno)d]- %(message)s')

class TushareClientError(Exception):
    """Custom exception for TushareClient errors."""
    pass

class TushareClient:
    """
    A client for interacting with the Tushare Pro API, enforcing rate limits.
    """

    def __init__(self, token: str, rate_limit_per_minute: int = 500):
        """
        Initializes the TushareClient with a token and rate limit.

        Args:
            token: Your Tushare Pro API token.
            rate_limit_per_minute: The maximum number of API calls allowed per minute.
                                   Defaults to 500.
        """
        if not token:
            raise ValueError("Tushare token cannot be empty.")

        self.token: str = token
        self.rate_limit_per_minute: int = rate_limit_per_minute
        self.pro: ts.pro_api = ts.pro_api(self.token)

        # Thread-safe rate limiting data structures
        self._request_timestamps: Deque[float] = collections.deque()
        self._lock: threading.Lock = threading.Lock()
        self._one_minute_in_seconds: int = 60

        logging.info(f"TushareClient initialized with rate limit: {self.rate_limit_per_minute} calls/minute.")

    def _wait_for_rate_limit(self) -> None:
        """
        Waits if necessary to ensure the API rate limit is not exceeded.
        This method is thread-safe.
        """
        with self._lock:
            current_time = time.time()

            # Remove timestamps older than one minute
            while self._request_timestamps and \
                  self._request_timestamps[0] <= current_time - self._one_minute_in_seconds:
                self._request_timestamps.popleft()

            # If the limit is reached, calculate wait time and sleep
            if len(self._request_timestamps) >= self.rate_limit_per_minute:
                # The time when the oldest request will expire
                oldest_request_time = self._request_timestamps[0]
                # Time until the oldest request expires
                time_to_wait = (oldest_request_time + self._one_minute_in_seconds) - current_time

                if time_to_wait > 0:
                    logging.info(f"Rate limit reached. Waiting for {time_to_wait:.2f} seconds...")
                    time.sleep(time_to_wait)
                    # After sleeping, re-check and remove expired timestamps
                    # This is important if multiple threads were waiting
                    while self._request_timestamps and \
                          self._request_timestamps[0] <= time.time() - self._one_minute_in_seconds:
                        self._request_timestamps.popleft()

    def fetch(self, api_name: str, **params: Any) -> pd.DataFrame:
        """
        Fetches data from the Tushare Pro API, respecting the rate limit.

        Args:
            api_name: The name of the Tushare API interface (e.g., 'daily', 'pro_bar').
            **params: Keyword arguments to pass to the Tushare API query method.

        Returns:
            A Pandas DataFrame containing the fetched data.

        Raises:
            TushareClientError: If there is an error during the API call or data retrieval.
        """
        self._wait_for_rate_limit()

        try:
            logging.info(f"Fetching data for API: {api_name} with params: {params}")
            df = self.pro.query(api_name, **params)

            if df is None:
                raise TushareClientError(f"Tushare API returned None for {api_name}. Check parameters or token.")
            
            # Tushare API often returns an empty DataFrame or a DataFrame with an error message
            # if the call was unsuccessful but didn't raise an exception.
            # A common pattern for Tushare errors is a 'code' and 'msg' column in the DataFrame
            # or an empty DataFrame if no data is found.
            if not df.empty and 'code' in df.columns and 'msg' in df.columns:
                error_code = df['code'].iloc[0]
                error_msg = df['msg'].iloc[0]
                if error_code != '0': # Assuming '0' means success
                    raise TushareClientError(f"Tushare API error for {api_name}: Code {error_code}, Message: {error_msg}")
            
            # Record the timestamp only after a successful query
            with self._lock:
                self._request_timestamps.append(time.time())
            
            logging.info(f"Successfully fetched {len(df)} rows for API: {api_name}.")
            return df

        except Exception as e:
            logging.error(f"Error fetching data for {api_name}: {e}")
            raise TushareClientError(f"Failed to fetch data from Tushare API for {api_name}: {e}") from e


