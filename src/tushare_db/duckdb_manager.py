import duckdb
import pandas as pd
import logging
import re
import time
from typing import TYPE_CHECKING, Optional, Union, List, Any

# Configure logging
logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(levelname)s -[%(filename)s:%(lineno)d]- %(message)s')

class DuckDBManagerError(Exception):
    """Custom exception for DuckDBManager errors."""
    pass

# A dictionary to define primary keys for tables. This is crucial for UPSERT operations.
TABLE_PRIMARY_KEYS = {
    "daily": ["ts_code", "trade_date"],
    "daily_basic": ["ts_code", "trade_date"],
    "adj_factor": ["ts_code", "trade_date"],
    "cyq_perf": ["ts_code", "trade_date"],
    # "cyq_chips": ["ts_code", "trade_date", "price"],
    "dc_member": ["ts_code", "trade_date", "con_code"],
    "dc_index": ["ts_code", "trade_date"],
    "stk_factor_pro": ["ts_code", "trade_date"],
    "index_weight": ["index_code", "trade_date", "con_code"],
    "trade_cal": ["exchange", "cal_date"],
    "stock_basic": ["ts_code"],
    "hs_const": ["ts_code", "in_date"],
    "stock_company": ["ts_code"],
    "index_basic": ["ts_code"],
    "fina_indicator_vip": ["ts_code", "end_date"],
    "moneyflow_dc": ["ts_code", "trade_date"],
    "moneyflow": ["ts_code", "trade_date"],
    "index_classify": ["industry_code"],
    "index_member_all": ["ts_code", "l3_code", "in_date"],
    # 财务报表
    "income": ["ts_code", "end_date", "report_type"],
    "balancesheet": ["ts_code", "end_date", "report_type"],
    "cashflow": ["ts_code", "end_date", "report_type"],
    # 分红送股
    "dividend": ["ts_code", "end_date"],
    # 融资融券
    "margin_detail": ["ts_code", "trade_date"],
}

class DuckDBManager:
    """
    Manages interactions with a DuckDB database, providing methods for table operations,
    query execution, and data writing with UPSERT capabilities.
    """

    def __init__(self, db_path: str, read_only: bool = False):
        """
        Initializes the DuckDBManager and establishes a connection to the database.

        Args:
            db_path: The path to the DuckDB database file (e.g., 'data/tushare.duckdb').
            read_only: Whether to open the database in read-only mode (allows concurrent readers).
        """
        if not db_path:
            raise ValueError("Database path cannot be empty.")
        try:
            self.db_path = db_path
            self.read_only = read_only
            self.con = duckdb.connect(database=db_path, read_only=read_only)
            
            if not read_only:
                self._create_metadata_table()
                
            logging.info(f"Connected to DuckDB database: {db_path} (read_only={read_only})")
        except Exception as e:
            logging.error(f"Failed to connect to DuckDB at {db_path}: {e}")
            raise DuckDBManagerError(f"Failed to connect to DuckDB: {e}") from e

    def table_exists(self, table_name: str) -> bool:
        """
        Checks if a table exists in the database.

        Args:
            table_name: The name of the table to check.

        Returns:
            True if the table exists, False otherwise.
        """
        try:
            result = self.con.execute(f"PRAGMA table_info('{table_name}')").fetchdf()
            return not result.empty
        except Exception as e:
            logging.error(f"Error checking existence of table {table_name}: {e}")
            return False

    def _truncate_query_for_logging(self, query: str, max_len: int = 200) -> str:
        """Shortens long IN clauses in SQL queries for cleaner logging."""
        in_clause_pattern = re.compile(r"(IN\s*\()([^)]+)(\))", re.IGNORECASE)

        def replacer(match):
            prefix, content, suffix = match.groups()
            if len(content) > max_len:
                items = content.split(',')
                if len(items) > 5:
                    truncated_content = f"{','.join(items[:3])}, ... ,{','.join(items[-2:])} ({len(items)} total)"
                    return f"{prefix}{truncated_content}{suffix}"
            return match.group(0)

        return in_clause_pattern.sub(replacer, query)

    def execute_query(self, query: str, params: Optional[List[Any]] = None) -> pd.DataFrame:
        """
        Executes a SQL query and returns the result as a Pandas DataFrame.

        Args:
            query: The SQL query string to execute.
            params: A list of parameters to bind to the query placeholders.

        Returns:
            A Pandas DataFrame containing the query results.

        Raises:
            DuckDBManagerError: If the query execution fails.
        """
        try:
            log_query = self._truncate_query_for_logging(query)
            logging.info(f"Executing query: {log_query}")
            
            # Execute with parameters if they are provided
            if params:
                df = self.con.execute(query, params).fetchdf()
            else:
                df = self.con.execute(query).fetchdf()
            
            return df
        except Exception as e:
            logging.error(f"Error executing query '{query}' with params {params}: {e}")
            raise DuckDBManagerError(f"Failed to execute query: {e}") from e

    def get_table_columns(self, table_name: str) -> List[str]:
        """
        Retrieves the column names of a table.

        Args:
            table_name: The name of the table.

        Returns:
            A list of column names, or an empty list if the table does not exist.
        """
        if not self.table_exists(table_name):
            return []
        try:
            result = self.con.execute(f"PRAGMA table_info('{table_name}')").fetchdf()
            return result['name'].tolist()
        except Exception as e:
            logging.error(f"Error getting columns for table {table_name}: {e}")
            raise DuckDBManagerError(f"Failed to get columns for table {table_name}: {e}") from e

    def _get_sql_schema_from_df(self, df: pd.DataFrame) -> str:
        """
        Generates a SQL column definition string from a pandas DataFrame's dtypes.
        """
        type_mapping = {
            'int64': 'BIGINT', 'int32': 'INTEGER', 'int16': 'SMALLINT',
            'float64': 'DOUBLE', 'float32': 'REAL', 'object': 'VARCHAR',
            'bool': 'BOOLEAN', 'datetime64[ns]': 'TIMESTAMP', 'category': 'VARCHAR',
        }
        default_type = 'VARCHAR'
        cols = [f'"{col_name}" {type_mapping.get(str(dtype), default_type)}' for col_name, dtype in df.dtypes.items()]
        return ", ".join(cols)

    def write_dataframe(self, df: pd.DataFrame, table_name: str, mode: str = 'append') -> None:
        """
        Writes a Pandas DataFrame to a specified table in the database, with support for
        primary keys and UPSERT operations in 'append' mode.
        """
        if mode not in ['append', 'replace']:
            raise ValueError(f"Invalid write mode: {mode}. Must be 'append' or 'replace'.")
        if df.empty:
            logging.warning(f"DataFrame is empty. No data written to table {table_name}.")
            return

        temp_view_name = f"temp_view_{table_name}_{int(time.time())}"
        try:
            self.con.register(temp_view_name, df)

            if mode == 'replace':
                logging.info(f"Replacing table {table_name} with new data.")
                self.con.execute(f"DROP TABLE IF EXISTS {table_name}")

            if not self.table_exists(table_name):
                logging.info(f"Table {table_name} does not exist. Creating it...")
                schema_str = self._get_sql_schema_from_df(df)
                pk_columns = TABLE_PRIMARY_KEYS.get(table_name)
                create_sql = f"CREATE TABLE {table_name} ({schema_str}"
                if pk_columns:
                    missing_keys = [pk for pk in pk_columns if pk not in df.columns]
                    if missing_keys:
                        raise DuckDBManagerError(f"Primary key columns {missing_keys} not found in DataFrame for table {table_name}.")
                    pk_str = ", ".join([f'"{col}"' for col in pk_columns])
                    create_sql += f", PRIMARY KEY ({pk_str})"
                create_sql += ")"
                logging.error(f"Executing create SQL: {create_sql}")
                self.con.execute(create_sql)
                self.con.execute(f"INSERT INTO {table_name} SELECT * FROM {temp_view_name}")
                logging.info(f"Table '{table_name}' created with {len(df)} rows.")
            elif mode == 'append':
                logging.info(f"Appending/updating data in table {table_name}.")
                pk_columns = TABLE_PRIMARY_KEYS.get(table_name)
                all_columns_str = ", ".join([f'"{c}"' for c in df.columns])
                if pk_columns:
                    update_columns = [c for c in df.columns if c not in pk_columns]
                    update_clause = "DO NOTHING" if not update_columns else "DO UPDATE SET " + ", ".join([f'"{c}"=excluded."{c}"' for c in update_columns])
                    pk_str = ", ".join([f'"{col}"' for col in pk_columns])
                    upsert_sql = f"INSERT INTO {table_name} ({all_columns_str}) SELECT * FROM {temp_view_name} ON CONFLICT ({pk_str}) {update_clause}"
                    self.con.execute(upsert_sql)
                else:
                    self.con.execute(f"INSERT INTO {table_name} SELECT * FROM {temp_view_name}")
            
            logging.info(f"Successfully wrote {len(df)} rows to table {table_name} in {mode} mode.")
        except Exception as e:
            logging.error(f"Error writing DataFrame to table {table_name} in {mode} mode: ", e, exc_info=True)
            raise DuckDBManagerError(f"Failed to write DataFrame to DuckDB: {e}") from e
        finally:
            self.con.unregister(temp_view_name)

    def get_latest_date(self, table_name: str, date_col: str) -> Optional[str]:
        """
        Retrieves the maximum (latest) date from a specified date column in a table.
        """
        if not self.table_exists(table_name):
            logging.info(f"Table {table_name} does not exist. Returning None for latest date.")
            return None
        try:
            query = f"SELECT MAX({date_col}) FROM {table_name}"
            result = self.con.execute(query).fetchone()
            if result and result[0] is not None:
                latest_date = str(result[0])
                logging.info(f"Latest date in {table_name}.{date_col}: {latest_date}")
                return latest_date
            else:
                logging.info(f"Table {table_name} is empty or {date_col} contains no data. Returning None.")
                return None
        except Exception as e:
            logging.error(f"Error getting latest date from {table_name}.{date_col}: {e}")
            raise DuckDBManagerError(f"Failed to get latest date: {e}") from e

    def _truncate_codes_for_logging(self, codes: List[str], max_len: int = 10) -> str:
        """Truncates a list of codes for cleaner logging."""
        if len(codes) > max_len:
            return f"[{', '.join(codes[:max_len])}, ... ({len(codes)} total)]"
        return str(codes)

    def get_latest_date_for_partition(self, table_name: str, date_col: str,
                                      partition_key_col: str,
                                      partition_key_value: Union[str, List[str]]) -> Optional[str]:
        """
        For one or more partition keys, finds the most common latest date among them.
        """
        if not self.table_exists(table_name):
            logging.info(f"Table {table_name} does not exist. Returning None for latest date.")
            return None
        keys = [partition_key_value] if isinstance(partition_key_value, str) else partition_key_value
        if not keys:
            logging.warning(f"{partition_key_col} list is empty. Returning None.")
            return None
        keys_for_log = self._truncate_codes_for_logging(keys)
        try:
            query = f"""
                WITH LatestDates AS (
                    SELECT MAX({date_col}) AS last_date FROM {table_name}
                    WHERE {partition_key_col} IN (SELECT * FROM UNNEST(?)) GROUP BY {partition_key_col}
                )
                SELECT last_date FROM LatestDates WHERE last_date IS NOT NULL
                GROUP BY last_date ORDER BY COUNT(*) DESC LIMIT 1;
            """
            result = self.con.execute(query, [keys]).fetchone()
            if result and result[0] is not None:
                latest_date = str(result[0])
                logging.info(f"Most common latest date for partition '{partition_key_col}' with keys {keys_for_log} in {table_name}.{date_col}: {latest_date}")
                return latest_date
            else:
                logging.info(f"No data found for any of keys {keys_for_log} in {table_name}. Returning None.")
                return None
        except Exception as e:
            logging.error(f"Error getting latest date for partition '{partition_key_col}' with keys {keys_for_log} from {table_name}.{date_col}: {e}")
            raise DuckDBManagerError(f"Failed to get latest date for partition(s): {e}") from e

    def close(self) -> None:
        """Closes the database connection."""
        try:
            self.con.close()
            logging.info(f"Closed connection to DuckDB database: {self.db_path}")
        except Exception as e:
            logging.error(f"Error closing DuckDB connection: {e}")
            raise DuckDBManagerError(f"Failed to close DuckDB connection: {e}") from e

    def _create_metadata_table(self) -> None:
        """Creates a metadata table to store cache information."""
        try:
            self.con.execute("""
                CREATE TABLE IF NOT EXISTS _tushare_cache_metadata (
                    table_name VARCHAR PRIMARY KEY,
                    last_updated_timestamp DOUBLE
                )
            """)
            logging.info("Ensured _tushare_cache_metadata table exists.")
        except Exception as e:
            logging.error(f"Error creating metadata table: {e}")
            raise DuckDBManagerError(f"Failed to create metadata table: {e}") from e

    def get_cache_metadata(self, table_name: str) -> Optional[float]:
        """Retrieves the last updated timestamp for a given table."""
        try:
            result = self.con.execute(f"SELECT last_updated_timestamp FROM _tushare_cache_metadata WHERE table_name = '{table_name}'").fetchone()
            return float(result[0]) if result and result[0] is not None else None
        except Exception as e:
            logging.error(f"Error getting cache metadata for {table_name}: {e}")
            raise DuckDBManagerError(f"Failed to get cache metadata: {e}") from e

    def update_cache_metadata(self, table_name: str, timestamp: float) -> None:
        """Updates or inserts the last updated timestamp for a given table."""
        try:
            self.con.execute("""
                INSERT INTO _tushare_cache_metadata (table_name, last_updated_timestamp) VALUES (?, ?)
                ON CONFLICT (table_name) DO UPDATE SET last_updated_timestamp = EXCLUDED.last_updated_timestamp
            """, [table_name, timestamp])
            logging.info(f"Updated cache metadata for {table_name} with timestamp {timestamp}.")
        except Exception as e:
            logging.error(f"Error updating cache metadata for {table_name}: {e}")
            raise DuckDBManagerError(f"Failed to update cache metadata: {e}") from e

    def delete_data(self, table_name: str, where_clause: str) -> int:
        """
        Deletes data from a table based on a WHERE clause.

        Args:
            table_name: The name of the table from which to delete data.
            where_clause: The SQL WHERE clause to identify rows for deletion.

        Returns:
            The number of rows deleted.

        Raises:
            DuckDBManagerError: If the deletion fails.
        """
        if not self.table_exists(table_name):
            logging.warning(f"Table {table_name} does not exist. No data deleted.")
            return 0
        if not where_clause:
            raise ValueError("A WHERE clause is required to prevent accidental full-table deletion.")

        try:
            # Use a CTE to count the rows to be deleted first
            count_query = f"SELECT count(*) FROM {table_name} WHERE {where_clause}"
            rows_to_delete = self.con.execute(count_query).fetchone()[0]

            if rows_to_delete > 0:
                delete_query = f"DELETE FROM {table_name} WHERE {where_clause}"
                logging.info(f"Executing delete: {delete_query}")
                self.con.execute(delete_query)
                logging.info(f"Successfully deleted {rows_to_delete} rows from {table_name} where {where_clause}.")
            else:
                logging.info(f"No rows matched the criteria for deletion in {table_name} where {where_clause}.")
            
            return rows_to_delete
        except Exception as e:
            logging.error(f"Error deleting data from {table_name} with condition '{where_clause}': {e}")
            raise DuckDBManagerError(f"Failed to delete data: {e}") from e