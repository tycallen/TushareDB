"""
TDD tests for shareholder data tables.

These tests verify the three new Tushare shareholder interfaces:
- top10_floatholders: 前十大流通股东
- stk_holdernumber: 股东户数
- stk_rewards: 管理层薪酬和持股

Tests are designed to FAIL initially (tables don't exist yet) - this is expected for TDD.
"""

import pytest
import tempfile
import os
import pandas as pd
from tushare_db.duckdb_manager import DuckDBManager, TABLE_PRIMARY_KEYS


class TestTop10FloatholdersTable:
    """Test top10_floatholders table - 前十大流通股东"""

    @pytest.fixture
    def db(self):
        """Create a temporary database for testing"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            db = DuckDBManager(db_path)
            yield db
            db.close()

    def test_table_has_primary_key_definition(self):
        """Test that top10_floatholders has primary key defined in TABLE_PRIMARY_KEYS"""
        assert "top10_floatholders" in TABLE_PRIMARY_KEYS
        assert TABLE_PRIMARY_KEYS["top10_floatholders"] == ["ts_code", "end_date", "holder_name"]

    def test_table_creation_with_schema(self, db):
        """Test table creation with correct schema"""
        df = pd.DataFrame({
            'ts_code': ['000001.SZ', '000001.SZ'],
            'end_date': ['20241231', '20241231'],
            'holder_name': ['股东A', '股东B'],
            'hold_amount': [1000000, 500000],
            'hold_ratio': [5.5, 2.8]
        })

        db.write_dataframe(df, 'top10_floatholders', mode='append')

        # Verify table exists
        assert db.table_exists('top10_floatholders') is True

        # Verify schema
        columns = db.get_table_columns('top10_floatholders')
        assert 'ts_code' in columns
        assert 'end_date' in columns
        assert 'holder_name' in columns
        assert 'hold_amount' in columns
        assert 'hold_ratio' in columns

    def test_upsert_functionality(self, db):
        """Test UPSERT with primary keys (ts_code, end_date, holder_name)"""
        # Initial insert
        df1 = pd.DataFrame({
            'ts_code': ['000001.SZ'],
            'end_date': ['20241231'],
            'holder_name': ['股东A'],
            'hold_amount': [1000000],
            'hold_ratio': [5.5]
        })
        db.write_dataframe(df1, 'top10_floatholders', mode='append')

        # Upsert - update existing record
        df2 = pd.DataFrame({
            'ts_code': ['000001.SZ'],
            'end_date': ['20241231'],
            'holder_name': ['股东A'],
            'hold_amount': [1500000],  # Updated
            'hold_ratio': [6.0]  # Updated
        })
        db.write_dataframe(df2, 'top10_floatholders', mode='append')

        # Verify only one record with updated values
        result = db.execute_query("SELECT * FROM top10_floatholders")
        assert len(result) == 1
        assert result.iloc[0]['hold_amount'] == 1500000
        assert result.iloc[0]['hold_ratio'] == 6.0

    def test_data_insertion_and_retrieval(self, db):
        """Test data insertion and retrieval"""
        df = pd.DataFrame({
            'ts_code': ['000001.SZ', '000001.SZ', '000002.SZ'],
            'end_date': ['20241231', '20241231', '20241231'],
            'holder_name': ['股东A', '股东B', '股东C'],
            'hold_amount': [1000000, 500000, 800000],
            'hold_ratio': [5.5, 2.8, 4.2]
        })

        db.write_dataframe(df, 'top10_floatholders', mode='append')

        # Query by ts_code
        result = db.execute_query(
            "SELECT * FROM top10_floatholders WHERE ts_code = '000001.SZ' ORDER BY hold_amount DESC"
        )
        assert len(result) == 2
        assert result.iloc[0]['holder_name'] == '股东A'

        # Query by end_date
        result = db.execute_query(
            "SELECT * FROM top10_floatholders WHERE end_date = '20241231'"
        )
        assert len(result) == 3


class TestStkHoldernumberTable:
    """Test stk_holdernumber table - 股东户数"""

    @pytest.fixture
    def db(self):
        """Create a temporary database for testing"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            db = DuckDBManager(db_path)
            yield db
            db.close()

    def test_table_has_primary_key_definition(self):
        """Test that stk_holdernumber has primary key defined in TABLE_PRIMARY_KEYS"""
        assert "stk_holdernumber" in TABLE_PRIMARY_KEYS
        assert TABLE_PRIMARY_KEYS["stk_holdernumber"] == ["ts_code", "end_date"]

    def test_table_creation_with_schema(self, db):
        """Test table creation with correct schema"""
        df = pd.DataFrame({
            'ts_code': ['000001.SZ', '000002.SZ'],
            'end_date': ['20241231', '20241231'],
            'holder_num': [50000, 30000],
            'holder_ratio': [25.5, 18.2]
        })

        db.write_dataframe(df, 'stk_holdernumber', mode='append')

        # Verify table exists
        assert db.table_exists('stk_holdernumber') is True

        # Verify schema
        columns = db.get_table_columns('stk_holdernumber')
        assert 'ts_code' in columns
        assert 'end_date' in columns
        assert 'holder_num' in columns

    def test_upsert_functionality(self, db):
        """Test UPSERT with primary keys (ts_code, end_date)"""
        # Initial insert
        df1 = pd.DataFrame({
            'ts_code': ['000001.SZ'],
            'end_date': ['20241231'],
            'holder_num': [50000],
            'holder_ratio': [25.5]
        })
        db.write_dataframe(df1, 'stk_holdernumber', mode='append')

        # Upsert - update existing record
        df2 = pd.DataFrame({
            'ts_code': ['000001.SZ'],
            'end_date': ['20241231'],
            'holder_num': [55000],  # Updated
            'holder_ratio': [26.0]  # Updated
        })
        db.write_dataframe(df2, 'stk_holdernumber', mode='append')

        # Verify only one record with updated values
        result = db.execute_query("SELECT * FROM stk_holdernumber")
        assert len(result) == 1
        assert result.iloc[0]['holder_num'] == 55000
        assert result.iloc[0]['holder_ratio'] == 26.0

    def test_data_insertion_and_retrieval(self, db):
        """Test data insertion and retrieval"""
        df = pd.DataFrame({
            'ts_code': ['000001.SZ', '000001.SZ', '000002.SZ'],
            'end_date': ['20241231', '20240930', '20241231'],
            'holder_num': [50000, 48000, 30000],
            'holder_ratio': [25.5, 24.8, 18.2]
        })

        db.write_dataframe(df, 'stk_holdernumber', mode='append')

        # Query by ts_code
        result = db.execute_query(
            "SELECT * FROM stk_holdernumber WHERE ts_code = '000001.SZ' ORDER BY end_date DESC"
        )
        assert len(result) == 2
        assert result.iloc[0]['end_date'] == '20241231'

        # Query latest date
        latest = db.get_latest_date_for_partition(
            'stk_holdernumber', 'end_date', 'ts_code', '000001.SZ'
        )
        assert latest == '20241231'


class TestStkRewardsTable:
    """Test stk_rewards table - 管理层薪酬和持股"""

    @pytest.fixture
    def db(self):
        """Create a temporary database for testing"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            db = DuckDBManager(db_path)
            yield db
            db.close()

    def test_table_has_primary_key_definition(self):
        """Test that stk_rewards has primary key defined in TABLE_PRIMARY_KEYS"""
        assert "stk_rewards" in TABLE_PRIMARY_KEYS
        assert TABLE_PRIMARY_KEYS["stk_rewards"] == ["ts_code", "end_date", "name"]

    def test_table_creation_with_schema(self, db):
        """Test table creation with correct schema"""
        df = pd.DataFrame({
            'ts_code': ['000001.SZ', '000001.SZ'],
            'end_date': ['20241231', '20241231'],
            'name': ['张三', '李四'],
            'title': ['董事长', '总经理'],
            'reward': [5000000.0, 3000000.0],
            'hold_vol': [100000, 50000]
        })

        db.write_dataframe(df, 'stk_rewards', mode='append')

        # Verify table exists
        assert db.table_exists('stk_rewards') is True

        # Verify schema
        columns = db.get_table_columns('stk_rewards')
        assert 'ts_code' in columns
        assert 'end_date' in columns
        assert 'name' in columns
        assert 'title' in columns
        assert 'reward' in columns
        assert 'hold_vol' in columns

    def test_upsert_functionality(self, db):
        """Test UPSERT with primary keys (ts_code, end_date, name)"""
        # Initial insert
        df1 = pd.DataFrame({
            'ts_code': ['000001.SZ'],
            'end_date': ['20241231'],
            'name': ['张三'],
            'title': ['董事长'],
            'reward': [5000000.0],
            'hold_vol': [100000]
        })
        db.write_dataframe(df1, 'stk_rewards', mode='append')

        # Upsert - update existing record
        df2 = pd.DataFrame({
            'ts_code': ['000001.SZ'],
            'end_date': ['20241231'],
            'name': ['张三'],
            'title': ['董事长'],  # Same title
            'reward': [6000000.0],  # Updated
            'hold_vol': [120000]  # Updated
        })
        db.write_dataframe(df2, 'stk_rewards', mode='append')

        # Verify only one record with updated values
        result = db.execute_query("SELECT * FROM stk_rewards")
        assert len(result) == 1
        assert result.iloc[0]['reward'] == 6000000.0
        assert result.iloc[0]['hold_vol'] == 120000

    def test_data_insertion_and_retrieval(self, db):
        """Test data insertion and retrieval"""
        df = pd.DataFrame({
            'ts_code': ['000001.SZ', '000001.SZ', '000002.SZ'],
            'end_date': ['20241231', '20241231', '20241231'],
            'name': ['张三', '李四', '王五'],
            'title': ['董事长', '总经理', '董事长'],
            'reward': [5000000.0, 3000000.0, 4000000.0],
            'hold_vol': [100000, 50000, 80000]
        })

        db.write_dataframe(df, 'stk_rewards', mode='append')

        # Query by ts_code
        result = db.execute_query(
            "SELECT * FROM stk_rewards WHERE ts_code = '000001.SZ' ORDER BY reward DESC"
        )
        assert len(result) == 2
        assert result.iloc[0]['name'] == '张三'

        # Query by name
        result = db.execute_query(
            "SELECT * FROM stk_rewards WHERE name = '张三'"
        )
        assert len(result) == 1
        assert result.iloc[0]['title'] == '董事长'


class TestShareholderTablesIntegration:
    """Integration tests for all three shareholder tables"""

    @pytest.fixture
    def db(self):
        """Create a temporary database for testing"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            db = DuckDBManager(db_path)
            yield db
            db.close()

    def test_all_tables_can_coexist(self, db):
        """Test that all three tables can coexist in the same database"""
        # Insert into top10_floatholders
        df1 = pd.DataFrame({
            'ts_code': ['000001.SZ'],
            'end_date': ['20241231'],
            'holder_name': ['股东A'],
            'hold_amount': [1000000],
            'hold_ratio': [5.5]
        })
        db.write_dataframe(df1, 'top10_floatholders', mode='append')

        # Insert into stk_holdernumber
        df2 = pd.DataFrame({
            'ts_code': ['000001.SZ'],
            'end_date': ['20241231'],
            'holder_num': [50000],
            'holder_ratio': [25.5]
        })
        db.write_dataframe(df2, 'stk_holdernumber', mode='append')

        # Insert into stk_rewards
        df3 = pd.DataFrame({
            'ts_code': ['000001.SZ'],
            'end_date': ['20241231'],
            'name': ['张三'],
            'title': ['董事长'],
            'reward': [5000000.0],
            'hold_vol': [100000]
        })
        db.write_dataframe(df3, 'stk_rewards', mode='append')

        # Verify all tables exist
        assert db.table_exists('top10_floatholders') is True
        assert db.table_exists('stk_holdernumber') is True
        assert db.table_exists('stk_rewards') is True

        # Verify data in each table
        result1 = db.execute_query("SELECT COUNT(*) as cnt FROM top10_floatholders")
        assert result1.iloc[0]['cnt'] == 1

        result2 = db.execute_query("SELECT COUNT(*) as cnt FROM stk_holdernumber")
        assert result2.iloc[0]['cnt'] == 1

        result3 = db.execute_query("SELECT COUNT(*) as cnt FROM stk_rewards")
        assert result3.iloc[0]['cnt'] == 1
