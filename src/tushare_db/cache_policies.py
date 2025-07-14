
import time
import logging
import pandas as pd
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from datetime import datetime, timedelta

from .duckdb_manager import DuckDBManager
from .tushare_fetcher import TushareFetcher


class BaseCachePolicy(ABC):
    """Abstract base class for all caching strategies."""

    def __init__(
        self,
        api_name: str,
        duckdb_manager: DuckDBManager,
        tushare_fetcher: TushareFetcher,
        **policy_config: Any,
    ):
        self.api_name = api_name
        self.table_name = api_name  # Use api_name as table name for simplicity
        self.duckdb_manager = duckdb_manager
        self.tushare_fetcher = tushare_fetcher
        self.policy_config = policy_config

    @abstractmethod
    def get_data(self, **params: Any) -> pd.DataFrame:
        """
        Fetches data, applying the specific caching policy.

        Args:
            **params: Keyword arguments for the Tushare API call.

        Returns:
            A pandas DataFrame with the requested data.
        """
        pass

    def _fetch_from_api(self, **params: Any) -> pd.DataFrame:
        """Helper method to fetch data from the Tushare API."""
        logging.debug(f"Fetching '{self.api_name}' from API with params: {params}")
        return self.tushare_fetcher.fetch(self.api_name, **params)


class FullCachePolicy(BaseCachePolicy):
    """
    Implements a full-refresh caching strategy with a Time-To-Live (TTL).
    The entire dataset is replaced if the cache is older than the TTL.
    """

    def get_data(self, **params: Any) -> pd.DataFrame:
        ttl = self.policy_config.get("ttl")
        if ttl is None:
            logging.warning(f"FullCachePolicy for '{self.api_name}' is missing 'ttl'. Fetching fresh data.")
            return self._refresh_cache(**params)

        last_updated = self.duckdb_manager.get_cache_metadata(self.table_name)
        if last_updated and (time.time() - last_updated) < ttl:
            logging.info(f"'{self.api_name}' cache is fresh (TTL: {ttl}s). Loading from DuckDB.")
            return self.duckdb_manager.execute_query(f"SELECT * FROM {self.table_name}")

        logging.info(f"'{self.api_name}' cache is stale or missing. Refreshing from API.")
        return self._refresh_cache(**params)

    def _refresh_cache(self, **params: Any) -> pd.DataFrame:
        """Fetches new data and replaces the old cache."""
        new_data_df = self._fetch_from_api(**params)
        if not new_data_df.empty:
            self.duckdb_manager.write_dataframe(new_data_df, self.table_name, mode="replace")
            self.duckdb_manager.update_cache_metadata(self.table_name, time.time())
        return new_data_df


class IncrementalCachePolicy(BaseCachePolicy):
    """

    Implements an incremental update caching strategy.
    Only fetches data newer than the latest record in the cache.
    """

    def get_data(self, **params: Any) -> pd.DataFrame:
        date_col = self.policy_config.get("date_col")
        if not date_col:
            raise ValueError(f"IncrementalCachePolicy for '{self.api_name}' requires a 'date_col'.")

        updated_params = params.copy()
        requested_end_date = params.get('end_date')
        
        if self.duckdb_manager.table_exists(self.table_name):
            ts_code = params.get("ts_code")
            latest_date = None
            if ts_code:
                # For requests targeting specific stock(s)
                latest_date = self.duckdb_manager.get_latest_date_for_stock(
                    self.table_name, date_col, ts_code
                )
            else:
                # For general requests (e.g., trade_cal)
                latest_date = self.duckdb_manager.get_latest_date(self.table_name, date_col)

            if latest_date:
                logging.info(
                    f"Incremental update for '{self.api_name}': "
                    f"Latest cached date is {latest_date}."
                )
                
                # 检查是否已经是最新数据
                if requested_end_date and latest_date >= requested_end_date:
                    logging.debug(f"Cache already covers requested range up to {requested_end_date}. No API fetch needed.")
                    where_clause = self._build_where_clause(params)
                    return self.duckdb_manager.execute_query(f"SELECT * FROM {self.table_name}{where_clause}")
                
                # 检查是否有新的交易日需要更新
                if self._should_fetch_new_data(latest_date, requested_end_date):
                    # 从最新日期的下一个交易日开始获取
                    next_date = self._get_next_trading_date(latest_date)
                    if next_date:
                        updated_params["start_date"] = next_date
                    else:
                        logging.debug(f"No trading days after {latest_date}. Returning cached data.")
                        where_clause = self._build_where_clause(params)
                        return self.duckdb_manager.execute_query(f"SELECT * FROM {self.table_name}{where_clause}")
                else:
                    logging.debug(f"No new trading days to fetch. Returning cached data.")
                    where_clause = self._build_where_clause(params)
                    return self.duckdb_manager.execute_query(f"SELECT * FROM {self.table_name}{where_clause}")
        else:
            logging.info(f"Table '{self.table_name}' not found. Performing initial full fetch.")

        new_data_df = self._fetch_from_api(**updated_params)

        if not new_data_df.empty:
            # Deduplicate before writing
            if self.duckdb_manager.table_exists(self.table_name) and latest_date:
                # Get the latest date from local_data_df to filter new_data_df
                new_data_df[date_col] = new_data_df[date_col].astype(str)
                new_data_df = new_data_df[new_data_df[date_col] > latest_date]
                logging.debug(f"Filtered new data to exclude dates <= {latest_date}.")

            if not new_data_df.empty:
                self.duckdb_manager.write_dataframe(new_data_df, self.table_name, mode="append")
                logging.info(f"Appended {len(new_data_df)} new rows to '{self.table_name}'.")
            else:
                logging.debug("No new data to append after deduplication.")

        # Always return the full dataset satisfying the original query
        where_clause = self._build_where_clause(params)
        return self.duckdb_manager.execute_query(f"SELECT * FROM {self.table_name}{where_clause}")

    def _build_where_clause(self, params: Dict[str, Any]) -> str:
        """
        Builds a SQL WHERE clause from query parameters, but only for columns
        that actually exist in the target table.
        """
        conditions = []
        # Get the actual columns of the table to prevent errors
        table_columns = self.duckdb_manager.get_table_columns(self.table_name)
        if not table_columns:
            return "" # Return empty if table doesn't exist or has no columns

        for key, value in params.items():
            # Only include parameters that are actual columns in the table
            if key not in table_columns:
                continue

            if value is None:
                continue

            if isinstance(value, str):
                conditions.append(f"{key} = '{value}'")
            elif isinstance(value, list):
                if not value: continue # Skip empty lists
                str_values = [f"'{v}'" for v in value]
                conditions.append(f"{key} IN ({', '.join(str_values)})")
            elif isinstance(value, (int, float)):
                conditions.append(f"{key} = {value}")
        
        # Handle date range separately, as they are for filtering the final result
        if 'start_date' in params and self.policy_config.get('date_col') in table_columns:
            conditions.append(f"{self.policy_config['date_col']} >= '{params['start_date']}'")
        if 'end_date' in params and self.policy_config.get('date_col') in table_columns:
            conditions.append(f"{self.policy_config['date_col']} <= '{params['end_date']}'")

        return " WHERE " + " AND ".join(conditions) if conditions else ""
    
    def _should_fetch_new_data(self, latest_date: str, requested_end_date: Optional[str]) -> bool:
        """
        检查是否需要从API获取新数据。
        考虑交易日历，避免在非交易日进行无效的API调用。
        """
        if not self.duckdb_manager.table_exists('trade_cal'):
            # 如果没有交易日历表，使用简单的日期比较
            if requested_end_date:
                return latest_date < requested_end_date
            else:
                # 没有指定结束日期，假设需要获取到今天
                today = datetime.now().strftime('%Y%m%d')
                return latest_date < today
        
        try:
            # 使用交易日历检查是否有新的交易日
            end_date_for_check = requested_end_date or datetime.now().strftime('%Y%m%d')
            query = f"""
                SELECT COUNT(*) FROM trade_cal 
                WHERE cal_date > '{latest_date}' 
                AND cal_date <= '{end_date_for_check}' 
                AND is_open = '1'
            """
            result = self.duckdb_manager.execute_query(query)
            trading_days_count = result.iloc[0, 0] if not result.empty else 0
            
            logging.debug(f"Found {trading_days_count} trading days between {latest_date} and {end_date_for_check}.")
            return trading_days_count > 0
            
        except Exception as e:
            logging.warning(f"Error checking trading days: {e}. Falling back to simple date comparison.")
            if requested_end_date:
                return latest_date < requested_end_date
            else:
                today = datetime.now().strftime('%Y%m%d')
                return latest_date < today
    
    def _get_next_trading_date(self, latest_date: str) -> Optional[str]:
        """
        获取指定日期之后的下一个交易日。
        如果没有交易日历表，返回下一个自然日。
        """
        if not self.duckdb_manager.table_exists('trade_cal'):
            # 如果没有交易日历表，返回下一个自然日
            try:
                date_obj = datetime.strptime(latest_date, '%Y%m%d')
                next_date = date_obj + timedelta(days=1)
                return next_date.strftime('%Y%m%d')
            except ValueError:
                logging.warning(f"Invalid date format: {latest_date}. Using original date.")
                return latest_date
        
        try:
            # 查找下一个交易日
            query = f"""
                SELECT MIN(cal_date) FROM trade_cal 
                WHERE cal_date > '{latest_date}' 
                AND is_open = '1'
            """
            result = self.duckdb_manager.execute_query(query)
            if not result.empty and result.iloc[0, 0] is not None:
                next_trading_date = str(result.iloc[0, 0])
                logging.debug(f"Next trading date after {latest_date}: {next_trading_date}")
                return next_trading_date
            else:
                logging.debug(f"No trading days found after {latest_date}.")
                return None
                
        except Exception as e:
            logging.warning(f"Error finding next trading date: {e}. Using next calendar day.")
            try:
                date_obj = datetime.strptime(latest_date, '%Y%m%d')
                next_date = date_obj + timedelta(days=1)
                return next_date.strftime('%Y%m%d')
            except ValueError:
                logging.warning(f"Invalid date format: {latest_date}. Using original date.")
                return latest_date

