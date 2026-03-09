import pytest
import tempfile
import os
import pandas as pd
from tushare_db.duckdb_manager import DuckDBManager, DuckDBManagerError


class TestDuckDBManager:
    """Test DuckDBManager core functionality"""

    @pytest.fixture
    def db(self):
        """Create a temporary database for testing"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            db = DuckDBManager(db_path)
            yield db
            db.close()

    def test_table_exists(self, db):
        """Test table_exists method"""
        # Create a test table
        db.con.execute("CREATE TABLE test_table (id INTEGER, name VARCHAR)")
        assert db.table_exists('test_table') is True
        assert db.table_exists('non_existent') is False

    def test_write_and_read_dataframe(self, db):
        """Test writing and reading DataFrame"""
        df = pd.DataFrame({
            'ts_code': ['000001.SZ', '000002.SZ'],
            'trade_date': ['20240101', '20240102'],
            'close': [10.5, 20.3]
        })

        db.write_dataframe(df, 'daily', mode='append')

        # Verify data was written
        result = db.execute_query("SELECT * FROM daily")
        assert len(result) == 2
        assert list(result['ts_code']) == ['000001.SZ', '000002.SZ']

    def test_get_cache_metadata_parameterized(self, db):
        """Test get_cache_metadata uses parameterized queries"""
        # The _tushare_cache_metadata table is already created by DuckDBManager.__init__
        # Just insert test data
        db.con.execute("""
            INSERT INTO _tushare_cache_metadata VALUES ('test_table', 12345.0)
        """)

        # Test with normal table name
        result = db.get_cache_metadata('test_table')
        assert result == 12345.0

        # Test with potentially malicious input (should not cause injection)
        result = db.get_cache_metadata("test' OR '1'='1")
        assert result is None  # Should return None, not raise error or return data


class TestRateLimiter:
    """Test TushareFetcher rate limiting"""

    def test_rate_limit_config_parsing(self, monkeypatch):
        """Test rate limit configuration is parsed correctly"""
        from tushare_db.tushare_fetcher import TushareFetcher
        import tushare as ts

        config = {
            "default": {"limit": 100, "period": "minute"},
            "daily": {"limit": 200, "period": "day"}
        }

        # Mock the pro_api and user call to avoid network request
        class MockPro:
            def user(self, **kwargs):
                return pd.DataFrame({"credit": [100]})

        monkeypatch.setattr(ts, "pro_api", lambda token: MockPro())
        monkeypatch.setattr(ts, "set_token", lambda token: None)

        # Mock token for initialization
        fetcher = TushareFetcher("mock_token_for_testing", config)

        assert fetcher.rate_limit_config["default"]["limit"] == 100
        assert fetcher.rate_limit_config["default"]["period_seconds"] == 60
        assert fetcher.rate_limit_config["daily"]["period_seconds"] == 86400
