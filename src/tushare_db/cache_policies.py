import time
import logging
import pandas as pd
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from datetime import datetime, timedelta

from tqdm import tqdm
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
            where_clause = self._build_where_clause(params) # Use the new where clause
            return self.duckdb_manager.execute_query(f"SELECT * FROM {self.table_name}{where_clause}")

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
            latest_date = None
            # Check if the policy is partitioned
            partition_key_col = self.policy_config.get("partition_key_col")
            partition_key_value = params.get(partition_key_col) if partition_key_col else None

            if partition_key_col and partition_key_value:
                # Partitioned request (e.g., for a specific ts_code or index_code)
                latest_date = self.duckdb_manager.get_latest_date_for_partition(
                    self.table_name, date_col, partition_key_col, partition_key_value
                )
            else:
                # Non-partitioned request (e.g., trade_cal)
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
            self.duckdb_manager.write_dataframe(new_data_df, self.table_name, mode="append")
            logging.info(f"Appended {len(new_data_df)} new rows to '{self.table_name}'.")

        # Always return the full dataset satisfying the original query
        if not self.duckdb_manager.table_exists(self.table_name):
            logging.warning(
                f"Table '{self.table_name}' does not exist after fetch attempt. "
                "This can happen if the API returned no data for the given parameters. "
                "Returning empty DataFrame."
            )
            return pd.DataFrame()
        
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


class DiscreteCachePolicy(BaseCachePolicy):
    """
    Implements a caching strategy for APIs that are updated based on discrete values (e.g., quarterly/annual reports).
    It fetches data for missing discrete values or refreshes existing values if the data is older than a specified TTL.
    """

    def get_data(self, **params: Any) -> pd.DataFrame:
        discrete_col = self.policy_config.get("discrete_col") # Renamed from date_col to discrete_col
        if not discrete_col:
            raise ValueError(f"DiscreteCachePolicy for '{self.api_name}' requires a 'discrete_col'.")

        requested_value = params.get(discrete_col)
        if not requested_value:
            logging.warning(f"No '{discrete_col}' specified for '{self.api_name}'. Fetching all available data.")
            return self._refresh_cache(**params)

        if self.duckdb_manager.table_exists(self.table_name):
            cached_data = self.duckdb_manager.execute_query(
                f"SELECT * FROM {self.table_name} WHERE {discrete_col} = '{requested_value}'"
            )
            if not cached_data.empty:
                logging.info(f"'{self.api_name}' data for {discrete_col}={requested_value} found in cache.")
                return cached_data

        logging.info(f"'{self.api_name}' data for {discrete_col}={requested_value} not found. Fetching from API.")
        return self._refresh_cache(**params)

    def _refresh_cache(self, **params: Any) -> pd.DataFrame:
        """Fetches new data and replaces/appends to the cache for the specific discrete value."""
        new_data_df = self._fetch_from_api(**params)
        if not new_data_df.empty:
            # Handle potential type conversion issues for 'extra_item' column
            # for numeric_cols in ['ebitda', 'extra_item', 'profit_dedt']:
            #     if numeric_cols in new_data_df.columns:
            #         try:
            #             # Convert 'extra_item' to float64 to prevent INT32 overflow
            #             new_data_df[numeric_cols] = pd.to_numeric(new_data_df[numeric_cols], errors='coerce').astype('float64')
            #             logging.debug(f"Converted '{numeric_cols}' column to float64 for {self.api_name}.")
            #         except Exception as e:
            #             logging.warning(f"Could not convert '{numeric_cols}' column to float64 for {self.api_name}: {e}")

            discrete_col = self.policy_config.get("discrete_col")
            if self.duckdb_manager.table_exists(self.table_name) and discrete_col and discrete_col in new_data_df.columns:
                existing_values = self.duckdb_manager.execute_query(
                    f"SELECT DISTINCT {discrete_col} FROM {self.table_name}"
                )
                if not existing_values.empty:
                    new_data_df = new_data_df[~new_data_df[discrete_col].isin(existing_values[discrete_col])]

            if not new_data_df.empty:
                self.duckdb_manager.write_dataframe(new_data_df, self.table_name, mode="append")
                logging.info(f"Appended {len(new_data_df)} new rows to '{self.table_name}'.")
            else:
                logging.debug("No new data to append after deduplication for discrete values.")
        return new_data_df


class AdjFactorCachePolicy(BaseCachePolicy):
    """
    专门为'adj_factor'（复权因子）接口设计的特殊缓存策略。

    该策略处理复权因子的独特性质：即某一天因分红、送股等事件导致的变动，
    会追溯性地影响该股票的所有历史复权因子。

    - 当按 `ts_code`（查询一只或多只股票的历史）查询时：
      1. 检查本地数据库中该股票的最新一条记录。
      2. 从Tushare API获取该股票在 *同一天* 的数据进行对比。
      3. 如果最新一日的 `adj_factor` 值发生变化，则认为该股票所有历史数据都已失效，
         随后会删除本地关于该 `ts_code` 的所有记录，并从API重新获取全部历史数据。
      4. 如果复权因子未变，则执行标准的增量更新，只获取本地缓存中缺失的日期的数据。

    - 当按 `trade_date`（查询某一天所有股票）查询时：
      1. 执行简单的增量更新逻辑，只获取本地缓存中不存在的日期的数据。
         这适用于每日更新所有股票的复权因子。
    """

    def get_data(self, **params: Any) -> pd.DataFrame:
        """
        获取复权因子数据的主入口。
        根据参数是 `ts_code` 还是 `trade_date` 来分发到不同的处理逻辑。
        """
        ts_code = params.get("ts_code")
        trade_date = params.get("trade_date")

        if ts_code:
            return self._handle_batch_stock_logic(ts_code, params)
        elif trade_date:
            return self._handle_daily_logic(trade_date, params)
        else:
            logging.warning(
                "adj_factor 接口在没有 'ts_code' 或 'trade_date' 参数的情况下被调用。"
                "将直接从API获取数据而不使用缓存。"
            )
            return self._fetch_from_api(**params)

    def _handle_batch_stock_logic(self, ts_code: Any, params: Dict[str, Any]) -> pd.DataFrame:
        """
        使用高效的批量方式处理按 `ts_code` 的查询请求。
        新增了动态遍历策略：根据请求的股票数和日期范围，自动选择最优的遍历方式（按股票或按日期）。
        """
        # 1. 标准化股票代码输入
        codes = []
        if isinstance(ts_code, str):
            codes = [c.strip() for c in ts_code.split(',') if c.strip()]
        elif isinstance(ts_code, list):
            codes = ts_code
        
        start_date = params.get("start_date")
        end_date = params.get("end_date")
        logging.info(
            f"AdjFactor请求: {len(codes)}只股票, "
            f"日期范围: {start_date or 'N/A'} to {end_date or 'N/A'}."
        )

        if not codes:
            logging.warning("adj_factor 接口调用时 ts_code 列表为空。")
            return pd.DataFrame()

        # --- 智能遍历决策 ---
        if len(codes) > 1 and start_date and end_date:
            try:
                trade_cal_df = self.duckdb_manager.execute_query(
                    f"SELECT count(*) FROM trade_cal WHERE cal_date >= '{start_date}' AND cal_date <= '{end_date}' AND is_open = '1'"
                )
                date_cost = trade_cal_df.iloc[0, 0] if not trade_cal_df.empty else 0
                stock_cost = len(codes)
                logging.debug(f"智能遍历成本估算：按日期({date_cost}次) vs 按股票({stock_cost}次)。")

                if date_cost > 0 and date_cost < stock_cost:
                    logging.info(f"检测到按日期遍历成本更低，将执行按日期 ({date_cost} 天) 的数据获取策略。")
                    return self._execute_date_iteration(start_date, end_date, codes, params)
            except Exception as e:
                logging.warning(f"智能遍历成本估算失败: {e}。将回退到按股票遍历的策略。")

        # --- 默认执行按股票的批量验证逻辑 ---
        return self._execute_stock_iteration(codes, params)

    def _execute_stock_iteration(self, codes: list, params: Dict[str, Any]) -> pd.DataFrame:
        """
        执行按股票为单位的批量验证和更新逻辑。
        """
        # 如果表不存在，则逐个初始化所有股票
        if not self.duckdb_manager.table_exists(self.table_name):
            logging.info(f"数据表 '{self.table_name}' 不存在。将为所有请求的股票执行首次全量数据获取。")
            for code in tqdm(codes, desc="首次初始化"):
                self._perform_full_refresh(code)
            return self._query_final_data_from_db(codes, params)

        # --- 核心验证与更新逻辑 ---
        latest_common_date = self.duckdb_manager.get_latest_date_for_partition(
            self.table_name, "trade_date", "ts_code", codes
        )
        logging.info(f"数据库状态: 本地最新公共日期为 {latest_common_date or 'N/A'}.")

        if not latest_common_date:
            logging.info("请求的股票在本地均无数据，将逐个进行初始化。")
            for code in tqdm(codes, desc="初始化新股票"):
                self._perform_full_refresh(code)
            return self._query_final_data_from_db(codes, params)

        logging.debug(f"使用基准日期 {latest_common_date} 进行批量验证。")
        remote_df = self._fetch_from_api(trade_date=latest_common_date)
        remote_factors = remote_df.set_index('ts_code')['adj_factor'].to_dict()

        placeholders = ','.join(['?'] * len(codes))
        local_df = self.duckdb_manager.execute_query(
            f"SELECT ts_code, adj_factor FROM {self.table_name} WHERE trade_date = '{latest_common_date}' AND ts_code IN ({placeholders})",
            codes
        )
        local_factors = local_df.set_index('ts_code')['adj_factor'].to_dict()

        stocks_to_refresh, stocks_to_increment, new_stocks = [], [], []
        for code in codes:
            if code not in local_factors:
                new_stocks.append(code)
            elif code in remote_factors and local_factors[code] != remote_factors[code]:
                stocks_to_refresh.append(code)
            else:
                stocks_to_increment.append(code)
        
        logging.info(
            f"验证结果: {len(new_stocks)}只新股票将全量获取, "
            f"{len(stocks_to_refresh)}只股票因子已变动将全量刷新, "
            f"{len(stocks_to_increment)}只股票将增量更新."
        )

        stocks_for_full_update = stocks_to_refresh + new_stocks
        if stocks_for_full_update:
            for code in tqdm(stocks_for_full_update, desc="全量更新"):
                self._perform_full_refresh(code)

        if stocks_to_increment:
            placeholders_inc = ','.join(['?'] * len(stocks_to_increment))
            latest_dates_df = self.duckdb_manager.execute_query(
                f"SELECT ts_code, MAX(trade_date) as max_date FROM {self.table_name} WHERE ts_code IN ({placeholders_inc}) GROUP BY ts_code",
                stocks_to_increment
            )
            latest_dates = latest_dates_df.set_index('ts_code')['max_date'].to_dict()
            
            for code in tqdm(stocks_to_increment, desc="增量更新"):
                self._perform_incremental_update(code, latest_dates.get(code, ''), params)

        return self._query_final_data_from_db(codes, params)

    def _execute_date_iteration(self, start_date: str, end_date: str, codes: list, params: Dict[str, Any]) -> pd.DataFrame:
        """
        执行按日期为单位的遍历，并增加前置验证步骤以处理复权因子的追溯性变化。
        """
        placeholders = ','.join(['?'] * len(codes))
        validation_date_df = self.duckdb_manager.execute_query(
            f"SELECT MAX(trade_date) as validation_date FROM {self.table_name} WHERE trade_date < '{start_date}' AND ts_code IN ({placeholders})",
            codes
        )
        validation_date = validation_date_df['validation_date'].iloc[0] if not validation_date_df.empty and pd.notna(validation_date_df['validation_date'].iloc[0]) else None

        if validation_date:
            logging.info(f"按日期遍历前置验证：使用 {validation_date} 作为验证日。")
            remote_df = self._fetch_from_api(trade_date=validation_date)
            if not remote_df.empty:
                remote_factors = remote_df.set_index('ts_code')['adj_factor'].to_dict()
                local_df = self.duckdb_manager.execute_query(
                    f"SELECT ts_code, adj_factor FROM {self.table_name} WHERE trade_date = '{validation_date}' AND ts_code IN ({placeholders})",
                    codes
                )
                local_factors = local_df.set_index('ts_code')['adj_factor'].to_dict()
                stocks_to_refresh = [
                    code for code in codes 
                    if code in local_factors and code in remote_factors and local_factors[code] != remote_factors[code]
                ]
                if stocks_to_refresh:
                    logging.info(f"验证发现 {len(stocks_to_refresh)} 只股票的复权因子已变动，将对它们进行全量刷新。")
                    for code in tqdm(stocks_to_refresh, desc="全量刷新变动股票"):
                        self._perform_full_refresh(code)

        trade_cal_df = self.duckdb_manager.execute_query(
            f"SELECT cal_date FROM trade_cal WHERE cal_date >= '{start_date}' AND cal_date <= '{end_date}' AND is_open = '1' ORDER BY cal_date"
        )
        all_dates = trade_cal_df['cal_date'].tolist()

        if all_dates:
            logging.info(f"开始按日期遍历获取 {len(all_dates)} 天的数据。")
            for trade_date in tqdm(all_dates, desc="按日期获取数据"):
                # --- 核心修正 ---
                # 检查当天数据是否已存在，如果存在则跳过
                if self.duckdb_manager.execute_query(f"SELECT 1 FROM {self.table_name} WHERE trade_date = '{trade_date}' LIMIT 1").empty:
                    daily_data = self._fetch_from_api(trade_date=trade_date)
                    if not daily_data.empty:
                        self.duckdb_manager.write_dataframe(daily_data, self.table_name, mode="append")
                else:
                    logging.debug(f"日期 {trade_date} 的数据已存在，跳过网络请求。")
        
        return self._query_final_data_from_db(codes, params)

    def _query_final_data_from_db(self, codes: list, params: Dict[str, Any]) -> pd.DataFrame:
        """封装了从数据库根据最终参数查询数据的逻辑。"""
        logging.debug("所有数据更新完毕，从数据库查询最终结果。")
        final_placeholders = ','.join(['?'] * len(codes))
        where_clause = f" WHERE ts_code IN ({final_placeholders})"
        
        query_params = codes[:] # 创建副本以安全地修改
        
        start_date = params.get('start_date')
        if start_date:
            where_clause += f" AND trade_date >= ?"
            query_params.append(start_date)
            
        end_date = params.get('end_date')
        if end_date:
            where_clause += f" AND trade_date <= ?"
            query_params.append(end_date)
            
        return self.duckdb_manager.execute_query(f"SELECT * FROM {self.table_name}{where_clause}", query_params)

    def _perform_full_refresh(self, ts_code: str) -> pd.DataFrame:
        """删除某股票的所有本地数据，并从API重新获取全量数据。"""
        self.duckdb_manager.delete_data(self.table_name, where_clause=f"ts_code = '{ts_code}'")
        new_data = self._fetch_from_api(ts_code=ts_code)
        if not new_data.empty:
            self.duckdb_manager.write_dataframe(new_data, self.table_name, mode="append")
        return new_data

    def _perform_incremental_update(self, ts_code: str, latest_date: str, params: Dict[str, Any]) -> pd.DataFrame:
        """获取从本地最新日期之后的新数据并追加到数据库。"""
        today = datetime.now().strftime("%Y%m%d")
        if latest_date < today:
            next_day = (datetime.strptime(latest_date, "%Y%m%d") + timedelta(days=1)).strftime("%Y%m%d")
            logging.debug(f"正在为 {ts_code} 获取从 {next_day} 到 {today} 的新数据。")
            new_data = self._fetch_from_api(ts_code=ts_code, start_date=next_day, end_date=today)
            if not new_data.empty:
                self.duckdb_manager.write_dataframe(new_data, self.table_name, mode="append")
        
        # 注意：此函数仅负责更新，最终数据由调用方（_execute_stock_iteration）统一查询
        return pd.DataFrame() # 返回空DF，因为数据已写入DB

    def _handle_daily_logic(self, trade_date: str, params: Dict[str, Any]) -> pd.DataFrame:
        """处理按天更新所有股票的缓存逻辑。"""
        # 检查该日期的数据是否已存在
        if self.duckdb_manager.table_exists(self.table_name):
            existing_data = self.duckdb_manager.execute_query(
                f"SELECT 1 FROM {self.table_name} WHERE trade_date = '{trade_date}' LIMIT 1"
            )
            if not existing_data.empty:
                logging.info(f"交易日 {trade_date} 的数据已存在。将从缓存返回。")
                return self.duckdb_manager.execute_query(f"SELECT * FROM {self.table_name} WHERE trade_date = '{trade_date}'")

        # 如果不存在，则从API获取
        logging.info(f"正在获取 {trade_date} 的每日复权因子数据。")
        new_data = self._fetch_from_api(**params)
        if not new_data.empty:
            self.duckdb_manager.write_dataframe(new_data, self.table_name, mode="append")
        
        return new_data