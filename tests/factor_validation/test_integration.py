"""Integration tests for Monte Carlo Factor Validation System.

These tests verify the complete workflow from data input to report generation.
"""

import os
import tempfile
from datetime import datetime
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.tushare_db.factor_validation import FactorFilter, FactorRegistry
from src.tushare_db.factor_validation.factor import Factor, FactorType
from src.tushare_db.factor_validation.filter import FactorReport
from src.tushare_db.factor_validation.tester import TestResult


def create_mock_reader():
    """Create a mock DataReader for testing."""
    mock_reader = Mock()

    # Create sample daily price data
    dates = pd.date_range("2023-01-01", periods=300, freq="B")
    prices = 100 * (1.0005 ** np.arange(300))

    sample_data = pd.DataFrame({
        "ts_code": ["000001.SZ"] * 300,
        "trade_date": dates,
        "open": prices * 0.99,
        "high": prices * 1.02,
        "low": prices * 0.98,
        "close": prices,
        "vol": [10000] * 300,
    })

    mock_reader.get_stock_daily.return_value = sample_data
    return mock_reader


def create_mock_reader_with_multiple_stocks(ts_codes):
    """Create a mock DataReader with multiple stocks."""
    mock_reader = Mock()

    all_data = []
    for ts_code in ts_codes:
        dates = pd.date_range("2023-01-01", periods=300, freq="B")
        np.random.seed(hash(ts_code) % 2**32)
        returns = np.random.randn(300) * 0.02 + 0.0005
        prices = 100 * np.cumprod(1 + returns)

        df = pd.DataFrame({
            "ts_code": [ts_code] * 300,
            "trade_date": dates,
            "open": prices * 0.99,
            "high": prices * 1.02,
            "low": prices * 0.98,
            "close": prices,
            "vol": np.random.randint(5000, 15000, 300),
        })
        all_data.append(df)

    def get_stock_daily(ts_code, *args, **kwargs):
        for df in all_data:
            if df["ts_code"].iloc[0] == ts_code:
                return df
        return pd.DataFrame()

    mock_reader.get_stock_daily.side_effect = get_stock_daily
    return mock_reader


class TestFullWorkflow:
    """Test complete validation workflow."""

    def test_full_workflow_with_sample_data(self):
        """Test complete workflow with mock data.

        Verifies:
        - FactorFilter initialization
        - Parameter estimation
        - Monte Carlo simulation
        - Signal detection
        - Report generation
        """
        mock_reader = create_mock_reader()

        # Create a simple momentum factor
        def momentum_factor(df):
            return df["close"] > df["close"].shift(1).fillna(df["close"])

        factor = Factor(
            name="momentum_test",
            description="Simple momentum factor",
            definition=momentum_factor,
            type=FactorType.FUNCTION,
        )

        with patch("src.tushare_db.reader.DataReader", return_value=mock_reader):
            filter_obj = FactorFilter(
                db_path=":memory:",
                n_simulations=100,
                simulation_days=60,
                alpha_threshold=1.5,
                window=60,
            )

            report = filter_obj.filter(
                factor=factor,
                ts_codes=["000001.SZ"],
                lookback_days=100,
            )

            # Verify report structure
            assert isinstance(report, FactorReport)
            assert report.factor == "momentum_test"
            assert isinstance(report.timestamp, datetime)
            assert "n_simulations" in report.parameters
            assert "alpha_threshold" in report.parameters

            # Verify results
            assert len(report.results) == 1
            result = report.results[0]
            assert result.ts_code == "000001.SZ"
            assert isinstance(result.p_actual, float)
            assert isinstance(result.p_random, float)
            assert isinstance(result.alpha_ratio, float)
            assert isinstance(result.p_value, float)
            assert isinstance(result.is_significant, bool)
            assert result.recommendation in ["KEEP", "OPTIMIZE", "DISCARD"]

            # Verify summary
            assert report.summary["total_stocks"] == 1
            assert "avg_alpha_ratio" in report.summary
            assert "avg_p_value" in report.summary

            # Verify dataframe
            assert isinstance(report.dataframe, pd.DataFrame)
            assert len(report.dataframe) == 1
            assert "ts_code" in report.dataframe.columns
            assert "alpha_ratio" in report.dataframe.columns

            # Verify markdown
            assert isinstance(report.markdown, str)
            assert "momentum_test" in report.markdown
            assert "000001.SZ" in report.markdown

            filter_obj.close()


class TestBatchFilter:
    """Test batch validation of multiple factors."""

    def test_batch_filter(self):
        """Test batch validation of multiple factors.

        Verifies:
        - Multiple factors can be validated in one call
        - Each factor gets its own report
        - Reports contain correct factor names
        """
        mock_reader = create_mock_reader_with_multiple_stocks(
            ["000001.SZ", "000002.SZ"]
        )

        # Create test factors
        def factor_up(df):
            return df["close"] > df["close"].shift(1).fillna(df["close"])

        def factor_down(df):
            return df["close"] < df["close"].shift(1).fillna(df["close"])

        factor1 = Factor(
            name="up_trend",
            description="Up trend detection",
            definition=factor_up,
            type=FactorType.FUNCTION,
        )
        factor2 = Factor(
            name="down_trend",
            description="Down trend detection",
            definition=factor_down,
            type=FactorType.FUNCTION,
        )

        with patch("src.tushare_db.reader.DataReader", return_value=mock_reader):
            filter_obj = FactorFilter(
                db_path=":memory:",
                n_simulations=50,
                simulation_days=60,
                alpha_threshold=1.5,
                window=60,
            )

            reports = filter_obj.batch_filter(
                factors=[factor1, factor2],
                ts_codes=["000001.SZ", "000002.SZ"],
                lookback_days=100,
            )

            # Verify reports dictionary
            assert isinstance(reports, dict)
            assert len(reports) == 2
            assert "up_trend" in reports
            assert "down_trend" in reports

            # Verify each report
            for name, report in reports.items():
                assert isinstance(report, FactorReport)
                assert report.factor == name
                assert len(report.results) == 2
                assert report.summary["total_stocks"] == 2

            filter_obj.close()

    def test_batch_filter_with_string_names(self):
        """Test batch filter using factor name strings."""
        mock_reader = create_mock_reader()

        # Register test factors
        def test_factor_func(df):
            return df["close"] > df["close"].shift(1).fillna(df["close"])

        FactorRegistry.register_builtin(Factor(
            name="batch_test_factor_1",
            description="Test factor 1",
            definition=test_factor_func,
            type=FactorType.FUNCTION,
        ))
        FactorRegistry.register_builtin(Factor(
            name="batch_test_factor_2",
            description="Test factor 2",
            definition=test_factor_func,
            type=FactorType.FUNCTION,
        ))

        with patch("src.tushare_db.reader.DataReader", return_value=mock_reader):
            filter_obj = FactorFilter(
                db_path=":memory:",
                n_simulations=50,
                simulation_days=60,
                alpha_threshold=1.5,
                window=60,
            )

            reports = filter_obj.batch_filter(
                factors=["batch_test_factor_1", "batch_test_factor_2"],
                ts_codes=["000001.SZ"],
                lookback_days=100,
            )

            assert isinstance(reports, dict)
            assert "batch_test_factor_1" in reports
            assert "batch_test_factor_2" in reports

            filter_obj.close()


class TestReportSave:
    """Test saving reports to files."""

    def test_report_save(self):
        """Test saving reports to markdown and CSV files.

        Verifies:
        - Markdown file is created with correct content
        - CSV file is created with correct data
        - Directory is created if it doesn't exist
        """
        result = TestResult(
            ts_code="000001.SZ",
            p_actual=0.3,
            p_random=0.2,
            alpha_ratio=1.5,
            n_signals_actual=30,
            n_signals_random=20.0,
            p_value=0.01,
            is_significant=True,
            recommendation="KEEP",
        )

        report = FactorReport(
            factor="save_test_factor",
            timestamp=datetime.now(),
            parameters={"n_simulations": 100},
            results=[result],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Test with nested directory
            nested_path = os.path.join(tmpdir, "reports", "validation")
            full_path = os.path.join(nested_path, "test_report")

            report.save(full_path)

            # Verify directory was created
            assert os.path.exists(nested_path)

            # Verify markdown file
            md_path = f"{full_path}.md"
            assert os.path.exists(md_path)
            with open(md_path, "r", encoding="utf-8") as f:
                content = f.read()
                assert "save_test_factor" in content
                assert "000001.SZ" in content
                assert "KEEP" in content
                assert "Alpha Ratio" in content

            # Verify CSV file
            csv_path = f"{full_path}.csv"
            assert os.path.exists(csv_path)
            df = pd.read_csv(csv_path)
            assert len(df) == 1
            assert df.iloc[0]["ts_code"] == "000001.SZ"
            assert df.iloc[0]["alpha_ratio"] == 1.5
            assert df.iloc[0]["recommendation"] == "KEEP"

    def test_report_save_empty_results(self):
        """Test saving report with empty results."""
        report = FactorReport(
            factor="empty_test",
            timestamp=datetime.now(),
            parameters={},
            results=[],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "empty_report")
            report.save(path)

            # Both files should still be created
            assert os.path.exists(f"{path}.md")
            assert os.path.exists(f"{path}.csv")

            # CSV should have headers but no data
            df = pd.read_csv(f"{path}.csv")
            assert len(df) == 0
            assert "ts_code" in df.columns
            assert "alpha_ratio" in df.columns


class TestAlphaRatioCalculation:
    """Test alpha ratio calculation correctness."""

    def test_alpha_ratio_calculation(self):
        """Verify alpha ratio calculation is mathematically correct.

        Alpha ratio = p_actual / p_random
        Should be > 1 for factors with predictive power.
        """
        mock_reader = create_mock_reader()

        def simple_factor(df):
            return df["close"] > df["close"].shift(1).fillna(df["close"])

        factor = Factor(
            name="alpha_test",
            description="Alpha ratio test factor",
            definition=simple_factor,
            type=FactorType.FUNCTION,
        )

        with patch("src.tushare_db.reader.DataReader", return_value=mock_reader):
            filter_obj = FactorFilter(
                db_path=":memory:",
                n_simulations=100,
                simulation_days=60,
                alpha_threshold=1.5,
                window=60,
            )

            report = filter_obj.filter(
                factor=factor,
                ts_codes=["000001.SZ"],
                lookback_days=100,
            )

            if report.results:
                result = report.results[0]
                # Verify alpha ratio calculation
                expected_alpha = result.p_actual / result.p_random if result.p_random > 0 else 0
                assert abs(result.alpha_ratio - expected_alpha) < 0.001

                # Verify p-value is between 0 and 1
                assert 0 <= result.p_value <= 1

                # Verify recommendation logic
                if result.alpha_ratio >= 1.5 and result.p_value < 0.05:
                    assert result.recommendation == "KEEP"
                elif result.alpha_ratio >= 1.2:
                    assert result.recommendation == "OPTIMIZE"
                else:
                    assert result.recommendation == "DISCARD"

            filter_obj.close()

    def test_alpha_ratio_extreme_values(self):
        """Test alpha ratio with extreme probability values."""
        # Test with very high alpha ratio
        result_high = TestResult(
            ts_code="TEST1",
            p_actual=0.5,
            p_random=0.01,
            alpha_ratio=50.0,
            n_signals_actual=50,
            n_signals_random=1.0,
            p_value=0.001,
            is_significant=True,
            recommendation="KEEP",
        )

        # Test with very low alpha ratio
        result_low = TestResult(
            ts_code="TEST2",
            p_actual=0.01,
            p_random=0.5,
            alpha_ratio=0.02,
            n_signals_actual=1,
            n_signals_random=50.0,
            p_value=0.99,
            is_significant=False,
            recommendation="DISCARD",
        )

        report = FactorReport(
            factor="extreme_test",
            timestamp=datetime.now(),
            parameters={},
            results=[result_high, result_low],
        )

        # Verify summary calculations handle extremes
        assert report.summary["total_stocks"] == 2
        assert report.summary["keep_count"] == 1
        assert report.summary["discard_count"] == 1
        assert report.summary["avg_alpha_ratio"] == pytest.approx(25.01, rel=0.01)


@pytest.mark.skipif(
    not os.environ.get("RUN_REAL_DATA_TESTS"),
    reason="Set RUN_REAL_DATA_TESTS=1 to run tests with real database",
)
class TestWithRealData:
    """Optional tests with real database - skipped by default."""

    def test_with_real_data(self):
        """Test with real database if available.

        Requires:
        - Valid database at default path or DB_PATH env var
        - TUSHARE_TOKEN environment variable
        """
        db_path = os.environ.get("DB_PATH", "tushare.db")

        if not os.path.exists(db_path):
            pytest.skip(f"Database not found at {db_path}")

        # Use real FactorFilter without mocking
        filter_obj = FactorFilter(
            db_path=db_path,
            n_simulations=100,  # Reduced for faster testing
            simulation_days=60,
            alpha_threshold=1.5,
            window=252,
        )

        try:
            # Get some stock codes from the database
            from src.tushare_db.reader import DataReader

            reader = DataReader(db_path=db_path)
            stocks = reader.get_stock_basic()
            reader.close()

            if stocks.empty:
                pytest.skip("No stock data in database")

            ts_codes = stocks["ts_code"].head(3).tolist()

            # Test with a built-in factor
            report = filter_obj.filter(
                factor="macd_golden_cross",
                ts_codes=ts_codes,
                lookback_days=252,
                use_sample=True,
            )

            assert isinstance(report, FactorReport)
            assert report.factor == "macd_golden_cross"
            assert len(report.results) > 0

            # Verify all results have valid data
            for result in report.results:
                assert result.ts_code in ts_codes
                assert 0 <= result.p_actual <= 1
                assert 0 <= result.p_random <= 1
                assert result.alpha_ratio >= 0

        finally:
            filter_obj.close()

    def test_batch_with_real_data(self):
        """Test batch filter with real data."""
        db_path = os.environ.get("DB_PATH", "tushare.db")

        if not os.path.exists(db_path):
            pytest.skip(f"Database not found at {db_path}")

        filter_obj = FactorFilter(
            db_path=db_path,
            n_simulations=50,
            simulation_days=60,
            alpha_threshold=1.5,
        )

        try:
            from src.tushare_db.reader import DataReader

            reader = DataReader(db_path=db_path)
            stocks = reader.get_stock_basic()
            reader.close()

            if stocks.empty:
                pytest.skip("No stock data in database")

            ts_codes = stocks["ts_code"].head(2).tolist()

            # Test batch filter with built-in factors
            builtin_factors = FactorRegistry.list_builtin()[:3]  # First 3 factors
            if builtin_factors:
                reports = filter_obj.batch_filter(
                    factors=builtin_factors,
                    ts_codes=ts_codes,
                    lookback_days=252,
                    use_sample=True,
                )

                assert len(reports) == len(builtin_factors)
                for name, report in reports.items():
                    assert isinstance(report, FactorReport)
                    assert report.factor == name

        finally:
            filter_obj.close()


class TestContextManager:
    """Test FactorFilter as context manager."""

    def test_context_manager(self):
        """Test FactorFilter works as context manager."""
        mock_reader = create_mock_reader()

        def simple_factor(df):
            return df["close"] > df["close"].shift(1).fillna(df["close"])

        factor = Factor(
            name="context_test",
            description="Context manager test",
            definition=simple_factor,
            type=FactorType.FUNCTION,
        )

        with patch("src.tushare_db.reader.DataReader", return_value=mock_reader):
            with FactorFilter(
                db_path=":memory:",
                n_simulations=50,
                simulation_days=60,
            ) as filter_obj:
                report = filter_obj.filter(
                    factor=factor,
                    ts_codes=["000001.SZ"],
                    lookback_days=100,
                )
                assert isinstance(report, FactorReport)

            # After exiting context, reader should be closed
            assert filter_obj._reader is None


class TestUseSample:
    """Test use_sample parameter for faster testing."""

    def test_use_sample_limits_stocks(self):
        """Test that use_sample limits number of stocks processed."""
        mock_reader = create_mock_reader_with_multiple_stocks(
            [f"00000{i}.SZ" for i in range(1, 11)]  # 10 stocks
        )

        def simple_factor(df):
            return df["close"] > df["close"].shift(1).fillna(df["close"])

        factor = Factor(
            name="sample_test",
            description="Sample test factor",
            definition=simple_factor,
            type=FactorType.FUNCTION,
        )

        with patch("src.tushare_db.reader.DataReader", return_value=mock_reader):
            filter_obj = FactorFilter(
                db_path=":memory:",
                n_simulations=50,
                simulation_days=60,
            )

            # Without use_sample, should process all 10
            report_full = filter_obj.filter(
                factor=factor,
                ts_codes=[f"00000{i}.SZ" for i in range(1, 11)],
                lookback_days=100,
                use_sample=False,
            )
            assert len(report_full.results) == 10

            # With use_sample, should limit to 5
            report_sample = filter_obj.filter(
                factor=factor,
                ts_codes=[f"00000{i}.SZ" for i in range(1, 11)],
                lookback_days=100,
                use_sample=True,
            )
            assert len(report_sample.results) <= 5

            filter_obj.close()
