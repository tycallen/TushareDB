
import duckdb
import pandas as pd
import logging
import re
from typing import TYPE_CHECKING, Optional, Union, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s -[%(filename)s:%(lineno)d]- %(message)s')

class DuckDBManagerError(Exception):
    """Custom exception for DuckDBManager errors."""
    pass

class DuckDBManager:
    """
    Manages interactions with a DuckDB database, providing methods for table operations,
    query execution, and data writing.
    """

    def __init__(self, db_path: str):
        """
        Initializes the DuckDBManager and establishes a connection to the database.

        Args:
            db_path: The path to the DuckDB database file (e.g., 'data/tushare.duckdb').
        """
        if not db_path:
            raise ValueError("Database path cannot be empty.")
        try:
            self.db_path = db_path
            self.con = duckdb.connect(database=db_path, read_only=False)
            self._create_metadata_table()
            logging.info(f"Connected to DuckDB database: {db_path}")
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
            # PRAGMA table_info returns an empty result if the table does not exist
            result = self.con.execute(f"PRAGMA table_info('{table_name}')").fetchdf()
            return not result.empty
        except Exception as e:
            logging.error(f"Error checking existence of table {table_name}: {e}")
            # Depending on the error, it might mean the table doesn't exist or a deeper issue.
            # For simplicity, we'll return False for any error during existence check.
            return False

    def _truncate_query_for_logging(self, query: str, max_len: int = 200) -> str:
        """Shortens long IN clauses in SQL queries for cleaner logging."""
        # This regex looks for "IN (" followed by a list of items, especially quoted strings.
        in_clause_pattern = re.compile(r"(IN\s*\()([^)]+)(\))", re.IGNORECASE)

        def replacer(match):
            prefix = match.group(1)  # "IN ("
            content = match.group(2)
            suffix = match.group(3)  # ")"

            # Only truncate if the content inside parentheses is long
            if len(content) > max_len:
                items = content.split(',')
                num_items = len(items)
                # Heuristic to decide if it's a list of codes
                if num_items > 5:
                    # Take first 3 and last 2 items as a sample
                    truncated_content = f"{','.join(items[:3])}, ... ,{','.join(items[-2:])} ({num_items} total)"
                    return f"{prefix}{truncated_content}{suffix}"
            # Return the original match if it's not long enough to truncate
            return match.group(0)

        return in_clause_pattern.sub(replacer, query)

    def execute_query(self, query: str) -> pd.DataFrame:
        """
        Executes a SQL query and returns the result as a Pandas DataFrame.

        Args:
            query: The SQL query string to execute.

        Returns:
            A Pandas DataFrame containing the query results.

        Raises:
            DuckDBManagerError: If the query execution fails.
        """
        try:
            log_query = self._truncate_query_for_logging(query)
            logging.info(f"Executing query: {log_query}")
            df = self.con.execute(query).fetchdf()
            return df
        except Exception as e:
            logging.error(f"Error executing query '{query}': {e}")
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
            # Use PRAGMA table_info to get column information
            result = self.con.execute(f"PRAGMA table_info('{table_name}')").fetchdf()
            # The 'name' column in the result DataFrame contains the column names
            return result['name'].tolist()
        except Exception as e:
            logging.error(f"Error getting columns for table {table_name}: {e}")
            raise DuckDBManagerError(f"Failed to get columns for table {table_name}: {e}") from e

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
            # Use PRAGMA table_info to get column information
            result = self.con.execute(f"PRAGMA table_info('{table_name}')").fetchdf()
            # The 'name' column in the result DataFrame contains the column names
            return result['name'].tolist()
        except Exception as e:
            logging.error(f"Error getting columns for table {table_name}: {e}")
            raise DuckDBManagerError(f"Failed to get columns for table {table_name}: {e}") from e

    def write_dataframe(self, df: pd.DataFrame, table_name: str, mode: str = 'append') -> None:
        """
        Writes a Pandas DataFrame to a specified table in the database.

        Args:
            df: The Pandas DataFrame to write.
            table_name: The name of the target table.
            mode: The write mode. Can be 'append' (add rows) or 'replace' (overwrite table).
                  Defaults to 'append'.

        Raises:
            ValueError: If an invalid mode is provided.
            DuckDBManagerError: If the write operation fails.
        """
        if mode not in ['append', 'replace']:
            raise ValueError(f"Invalid write mode: {mode}. Must be 'append' or 'replace'.")

        if df.empty:
            logging.warning(f"DataFrame is empty. No data written to table {table_name}.")
            return

        try:
            # Register the DataFrame as a temporary view/table in DuckDB
            self.con.register('df_temp', df)

            if mode == 'replace':
                logging.info(f"Replacing table {table_name} with new data.")
                # DuckDB's CREATE OR REPLACE TABLE handles schema inference from the registered DataFrame
                self.con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM df_temp")
            elif mode == 'append':
                logging.info(f"Appending data to table {table_name}.")
                # If table doesn't exist, create it first, then append.
                if not self.table_exists(table_name):
                    logging.info(f"Table {table_name} does not exist. Creating it before appending.")
                    self.con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df_temp")
                else:
                    self.con.execute(f"INSERT INTO {table_name} SELECT * FROM df_temp")
            
            logging.info(f"Successfully wrote {len(df)} rows to table {table_name} in {mode} mode.")

        except Exception as e:
            logging.error(f"Error writing DataFrame to table {table_name} in {mode} mode: {e}")
            raise DuckDBManagerError(f"Failed to write DataFrame to DuckDB: {e}") from e
        finally:
            # Always unregister the temporary DataFrame
            try:
                self.con.unregister('df_temp')
            except Exception as e:
                logging.warning(f"Failed to unregister temporary DataFrame 'df_temp': {e}")

    def get_latest_date(self, table_name: str, date_col: str) -> Optional[str]:
        """
        Retrieves the maximum (latest) date from a specified date column in a table.
        Useful for incremental updates.

        Args:
            table_name: The name of the table to query.
            date_col: The name of the date column (e.g., 'trade_date').

        Returns:
            The latest date as a string (e.g., 'YYYYMMDD') or None if the table
            does not exist or is empty.

        Raises:
            DuckDBManagerError: If there's an error during the query.
        """
        assert table_name not in ['pro_bar']
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
        This is useful for partitioned data (e.g., by ts_code, index_code).

        Args:
            table_name: The name of the table to query.
            date_col: The name of the date column (e.g., 'trade_date').
            partition_key_col: The name of the column used for partitioning (e.g., 'ts_code').
            partition_key_value: A single key value or a list of key values.

        Returns:
            The most common latest date as a string (e.g., 'YYYYMMDD'),
            or None if no data is found for any of the given keys.

        Raises:
            DuckDBManagerError: If there's an error during the query.
        """
        if not self.table_exists(table_name):
            logging.info(f"Table {table_name} does not exist. Returning None for latest date.")
            return None

        if isinstance(partition_key_value, str):
            keys = [partition_key_value]
        else:
            keys = partition_key_value

        if not keys:
            logging.warning(f"{partition_key_col} list is empty. Returning None.")
            return None

        keys_for_log = self._truncate_codes_for_logging(keys)

        try:
            # This query finds the latest date for each key, then finds the
            # most frequently occurring date among them. This avoids outliers.
            query = f"""
                WITH LatestDates AS (
                    SELECT MAX({date_col}) AS last_date
                    FROM {table_name}
                    WHERE {partition_key_col} IN (SELECT * FROM UNNEST(?))
                    GROUP BY {partition_key_col}
                )
                SELECT last_date
                FROM LatestDates
                WHERE last_date IS NOT NULL
                GROUP BY last_date
                ORDER BY COUNT(*) DESC
                LIMIT 1;
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
        """
        Closes the database connection.
        """
        try:
            self.con.close()
            logging.info(f"Closed connection to DuckDB database: {self.db_path}")
        except Exception as e:
            logging.error(f"Error closing DuckDB connection: {e}")
            raise DuckDBManagerError(f"Failed to close DuckDB connection: {e}") from e

    def _create_metadata_table(self) -> None:
        """
        Creates a metadata table to store cache information like last update timestamps.
        """
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
        """
        Retrieves the last updated timestamp for a given table from the metadata table.

        Args:
            table_name: The name of the table to retrieve metadata for.

        Returns:
            The last updated timestamp as a float, or None if not found.
        """
        try:
            query = f"SELECT last_updated_timestamp FROM _tushare_cache_metadata WHERE table_name = '{table_name}'"
            result = self.con.execute(query).fetchone()
            if result and result[0] is not None:
                return float(result[0])
            return None
        except Exception as e:
            logging.error(f"Error getting cache metadata for {table_name}: {e}")
            raise DuckDBManagerError(f"Failed to get cache metadata: {e}") from e

    def update_cache_metadata(self, table_name: str, timestamp: float) -> None:
        """
        Updates or inserts the last updated timestamp for a given table in the metadata table.

        Args:
            table_name: The name of the table to update metadata for.
            timestamp: The new last updated timestamp (e.g., time.time()).
        """
        try:
            self.con.execute("""
                INSERT INTO _tushare_cache_metadata (table_name, last_updated_timestamp)
                VALUES (?, ?)
                ON CONFLICT (table_name) DO UPDATE SET last_updated_timestamp = EXCLUDED.last_updated_timestamp
            """, [table_name, timestamp])
            logging.info(f"Updated cache metadata for {table_name} with timestamp {timestamp}.")
        except Exception as e:
            logging.error(f"Error updating cache metadata for {table_name}: {e}")
            raise DuckDBManagerError(f"Failed to update cache metadata: {e}") from e

