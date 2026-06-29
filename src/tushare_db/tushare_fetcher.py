

import os
import tushare as ts
import pandas as pd
import time
import threading
import collections
from typing import Any, Deque, Dict

from .logger import get_logger

# 使用库统一的命名 logger，避免在库内调用 basicConfig 劫持调用方的 root logger
logger = get_logger("tushare_fetcher")

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

        # 可选：将请求地址指向 Tushare 协议兼容的自定义代理（如第三方 xiaodefa 代理）。
        # 未设 TUSHARE_API_URL 时保持官方默认地址，向后兼容。
        # tushare 的 DataApi 用 name-mangled 类属性 _DataApi__http_url，请求时拼成
        # POST {__http_url}/{api_name}；此类代理在 /{api_name} 路径上接受请求。
        api_url = os.getenv("TUSHARE_API_URL")
        if api_url:
            api_url = api_url.rstrip("/")
            self.pro._DataApi__http_url = api_url
            logger.info(f"Tushare 请求地址已指向自定义代理: {api_url}")

        masked_token = self.token[:4] + "****" + self.token[-4:] if len(self.token) > 8 else "****"
        # 账户信息查询为可选：自定义代理可能不提供 user 接口，或 token 仅对代理有效，
        # 失败不应阻断初始化（数据请求走的是 fetch()，与 user 无关）。
        try:
            df = self.pro.user(token=self.token)
            logger.debug(f"User with token {masked_token}, credit info: {df}")
        except Exception as e:
            logger.debug(f"获取账户信息失败（忽略，不影响数据请求）: {e}")

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

        logger.info(f"TushareFetcher initialized with rate limit config: {self.rate_limit_config}")

    def _truncate_params_for_logging(self, params: dict, max_len: int = 10) -> str:
        """Truncates the 'ts_code' in params for cleaner logging."""
        params_copy = params.copy()
        if 'ts_code' in params_copy:
            ts_code = params_copy['ts_code']
            if isinstance(ts_code, str) and ',' in ts_code:
                ts_code = ts_code.split(',')
            
            if isinstance(ts_code, list) and len(ts_code) > max_len:
                params_copy['ts_code'] = f"[{','.join(ts_code[:max_len])}, ... ({len(ts_code)} total)]"
        return str(params_copy)

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
        # Default behavior for all other APIs
        self._wait_for_rate_limit(api_name)

        try:
            params_for_log = self._truncate_params_for_logging(params)
            logger.info(f"Fetching data for API: {api_name} with params: {params_for_log}")
            
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
            # logger.error(df)
            logger.info(f"Successfully fetched {len(df)} rows for API: {api_name}.")
            return df

        except Exception as e:
            if isinstance(e, TushareClientError):
                raise
            logger.error(f"Error fetching data for {api_name}: {e}")
            raise TushareClientError(f"Failed to fetch data from Tushare API for {api_name}: {e}") from e

    def _wait_for_rate_limit(self, api_name: str) -> None:
        """
        Waits if necessary to ensure the API rate limit for the given API is not exceeded.
        This method is thread-safe. The lock is released during sleep so that
        calls to other APIs are not blocked.
        """
        while True:
            time_to_wait = 0
            with self._lock:
                config = self.rate_limit_config.get(api_name, self.rate_limit_config["default"])
                limit = config["limit"]
                period_seconds = config["period_seconds"]

                timestamps = self._api_call_timestamps[api_name]
                current_time = time.time()

                # Remove timestamps older than the period
                while timestamps and timestamps[0] <= current_time - period_seconds:
                    timestamps.popleft()

                # If the limit is not reached, we can proceed
                if len(timestamps) < limit:
                    return

                # Calculate wait time
                oldest_request_time = timestamps[0]
                time_to_wait = (oldest_request_time + period_seconds) - current_time

            # Sleep OUTSIDE the lock so other API calls are not blocked
            if time_to_wait > 0:
                logger.info(f"Rate limit for '{api_name}' reached. Waiting for {time_to_wait:.2f} seconds...")
                time.sleep(time_to_wait)
            # Loop back to re-check after sleep


