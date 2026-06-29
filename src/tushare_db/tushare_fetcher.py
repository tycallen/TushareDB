

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


class TushareUnavailableError(TushareClientError):
    """接口在当前数据源（官方或第三方代理）不可用：无权限或接口不存在。

    这类错误重试无意义，应跳过该接口而非反复请求。不同中转站不可用的接口各不
    相同，因此由 fetch() 在运行时自动发现并缓存，不维护硬编码黑名单。
    """
    pass


# 判定「接口不可用」（重试无用）的错误关键词。必须排除 token 错误——
# token 错误是全局凭证问题，不是某个接口不可用。
_UNAVAILABLE_MARKERS = (
    "没有权限", "没有访问", "接口不存在", "积分不足", "未开通", "权限不足",
    "no permission", "not have permission",
)

# 看起来是临时/网络/网关错误（重试可能成功）的标志。命中这些就【不】缓存为
# 「接口不可用」——否则第三方代理一次 503/超时/WAF 页面就会让接口被永久跳过。
_TRANSIENT_MARKERS = (
    "timeout", "timed out", "connection", "max retries", "reset by peer",
    "502", "503", "504", "bad gateway", "gateway", "<html", "<!doctype",
    "temporarily", "try again", "稍后", "重试", "繁忙", "网络", "超时",
)


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
            api_url = api_url.strip().rstrip("/")
            # 校验 scheme：漏写 http(s):// 会导致后续每次请求在 requests 层报
            # 难以定位的 'No scheme supplied'。这里补全为 https:// 并告警。
            if not api_url.startswith(("http://", "https://")):
                logger.warning(
                    f"TUSHARE_API_URL 缺少 http(s):// 前缀（{api_url}），已自动按 https:// 处理"
                )
                api_url = "https://" + api_url
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
        # 运行时自动发现的「当前数据源不可用接口」集合（无权限/接口不存在）。
        # 自适应不同中转站：一旦某接口被判定不可用，后续直接跳过，不再请求。
        self._unavailable_apis: set = set()

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

    @property
    def unavailable_apis(self) -> set:
        """本次运行中被判定为「当前数据源不可用」的接口名集合（只读快照）。"""
        with self._lock:
            return set(self._unavailable_apis)

    @staticmethod
    def _is_unavailable_error(msg: str) -> bool:
        """错误信息是否表示「接口不可用」（无权限/不存在）——这类重试无意义。

        必须保守：token 错误（全局凭证问题）和临时/网络/网关错误都不算「接口不可用」，
        以免把可恢复的问题误缓存成永久跳过。
        """
        low = msg.lower()
        if "token" in low or "40101" in msg:  # 全局凭证问题，非接口不可用
            return False
        if any(t in low for t in _TRANSIENT_MARKERS):  # 临时/网络错误，重试可能成功
            return False
        return any(m in msg for m in _UNAVAILABLE_MARKERS)

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

    def fetch(self, api_name: str, raise_on_unavailable: bool = False, **params: Any) -> pd.DataFrame:
        """
        Fetches data from the Tushare Pro API, respecting per-API rate limits.

        对「当前数据源不可用」的接口（无权限/接口不存在）做自适应优雅降级：首次
        遇到时告警并记入 unavailable_apis，之后对该接口直接跳过（不再请求、不再刷
        错误日志）。默认返回空 DataFrame，使下游下载逻辑（df.empty -> 跳过）无需
        改动即可自动跳过不可用接口；探测等需要区分的场景可传 raise_on_unavailable=True
        改为抛 TushareUnavailableError。

        Args:
            api_name: The name of the Tushare API interface (e.g., 'daily').
            raise_on_unavailable: 接口不可用时是否抛 TushareUnavailableError
                （默认 False，即返回空 DataFrame 优雅跳过）。
            **params: Keyword arguments to pass to the Tushare API query method.

        Returns:
            A Pandas DataFrame containing the fetched data（不可用接口返回空 DataFrame）。

        Raises:
            TushareUnavailableError: 接口不可用且 raise_on_unavailable=True。
            TushareClientError: 其它（通常可重试的）请求或数据错误。
        """
        # 已知不可用的接口：直接跳过，不再请求、不再刷日志
        with self._lock:
            already_unavailable = api_name in self._unavailable_apis
        if already_unavailable:
            if raise_on_unavailable:
                raise TushareUnavailableError(f"接口 {api_name} 在当前数据源不可用（已缓存，跳过）")
            logger.debug(f"接口 {api_name} 已知不可用，跳过请求并返回空结果")
            return pd.DataFrame()

        self._wait_for_rate_limit(api_name)

        try:
            params_for_log = self._truncate_params_for_logging(params)
            logger.info(f"Fetching data for API: {api_name} with params: {params_for_log}")

            api_func = getattr(self.pro, api_name, None)
            if api_func is None:
                def api_func(**p):
                    return self.pro.query(api_name, **p)

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
            logger.info(f"Successfully fetched {len(df)} rows for API: {api_name}.")
            return df

        except Exception as e:
            if isinstance(e, TushareUnavailableError):
                raise
            msg = str(e)
            # 接口不可用（无权限/不存在）：记录并自适应跳过，重试无意义
            if self._is_unavailable_error(msg):
                with self._lock:
                    is_new = api_name not in self._unavailable_apis
                    self._unavailable_apis.add(api_name)
                if is_new:
                    logger.warning(
                        f"接口 {api_name} 在当前数据源不可用（{msg}），"
                        f"已记录，本次运行将自动跳过该接口"
                    )
                if raise_on_unavailable:
                    raise TushareUnavailableError(f"接口 {api_name} 不可用: {msg}") from e
                return pd.DataFrame()
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


