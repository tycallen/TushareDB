"""Tests for FactorFilter module."""

import os
import tempfile
from datetime import datetime
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.tushare_db.factor_validation.detector import SignalDetector
from src.tushare_db.factor_validation.factor import Factor, FactorRegistry, FactorType
from src.tushare_db.factor_validation.filter import FactorFilter, FactorReport
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


def test_factor_report_dataclass():
    """Test FactorReport dataclass creation."""
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
        factor="test_factor",
        timestamp=datetime.now(),
        parameters={"n_simulations": 100},
        results=[result],
    )

    assert report.factor == "test_factor"
    assert len(report.results) == 1
    assert report.summary != {}
    assert report.dataframe is not None


def test_factor_report_save():
    """Test FactorReport save method."""
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
        factor="test_factor",
        timestamp=datetime.now(),
        parameters={"n_simulations": 100},
        results=[result],
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_report")
        report.save(path)

        # Check files were created
        assert os.path.exists(f"{path}.md")
        assert os.path.exists(f"{path}.csv")

        # Check markdown content
        with open(f"{path}.md", "r") as f:
            content = f.read()
            assert "test_factor" in content
            assert "000001.SZ" in content

        # Check CSV content
        df = pd.read_csv(f"{path}.csv")
        assert len(df) == 1
        assert df.iloc[0]["ts_code"] == "000001.SZ"


def test_factor_filter_init():
    """Test FactorFilter initialization."""
    filter_obj = FactorFilter(
        db_path="test.db",
        n_simulations=1000,
        simulation_days=252,
        alpha_threshold=1.5,
        window=252,
    )

    assert filter_obj.db_path == "test.db"
    assert filter_obj.n_simulations == 1000
    assert filter_obj.simulation_days == 252
    assert filter_obj.alpha_threshold == 1.5
    assert filter_obj.window == 252


def test_factor_filter_filter_with_mock():
    """Test FactorFilter.filter with mocked reader."""
    mock_reader = create_mock_reader()

    # Create a simple factor that only uses close price
    # (SignalDetector only provides close column)
    def simple_factor(df):
        # Close price greater than previous close (momentum)
        return df["close"] > df["close"].shift(1).fillna(df["close"])

    factor = Factor(
        name="price_momentum",
        description="Close greater than previous close",
        definition=simple_factor,
        type=FactorType.FUNCTION,
    )

    with patch("src.tushare_db.reader.DataReader", return_value=mock_reader):
        filter_obj = FactorFilter(
            db_path="test.db",
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

        assert isinstance(report, FactorReport)
        assert report.factor == "price_momentum"
        assert len(report.results) == 1
        assert report.results[0].ts_code == "000001.SZ"


def test_factor_filter_filter_with_string_factor():
    """Test FactorFilter.filter with factor name string."""
    mock_reader = create_mock_reader()

    # Register a test factor that only uses close price
    def test_factor_func(df):
        return df["close"] > df["close"].shift(1).fillna(df["close"])

    test_factor = Factor(
        name="test_price_momentum",
        description="Test factor using close only",
        definition=test_factor_func,
        type=FactorType.FUNCTION,
    )
    FactorRegistry.register_builtin(test_factor)

    with patch("src.tushare_db.reader.DataReader", return_value=mock_reader):
        filter_obj = FactorFilter(
            db_path="test.db",
            n_simulations=100,
            simulation_days=60,
            alpha_threshold=1.5,
            window=60,
        )

        report = filter_obj.filter(
            factor="test_price_momentum",
            ts_codes=["000001.SZ"],
            lookback_days=100,
        )

        assert isinstance(report, FactorReport)
        assert report.factor == "test_price_momentum"


def test_factor_filter_batch_filter():
    """Test FactorFilter.batch_filter."""
    mock_reader = create_mock_reader()

    # Create test factors that only use close price
    def factor1(df):
        return df["close"] > df["close"].shift(1).fillna(df["close"])

    def factor2(df):
        return df["close"] < df["close"].shift(1).fillna(df["close"])

    factor_a = Factor(name="factor_a", description="Factor A", definition=factor1, type=FactorType.FUNCTION)
    factor_b = Factor(name="factor_b", description="Factor B", definition=factor2, type=FactorType.FUNCTION)

    with patch("src.tushare_db.reader.DataReader", return_value=mock_reader):
        filter_obj = FactorFilter(
            db_path="test.db",
            n_simulations=100,
            simulation_days=60,
            alpha_threshold=1.5,
            window=60,
        )

        reports = filter_obj.batch_filter(
            factors=[factor_a, factor_b],
            ts_codes=["000001.SZ"],
            lookback_days=100,
        )

        assert isinstance(reports, dict)
        assert len(reports) == 2
        assert "factor_a" in reports
        assert "factor_b" in reports
        assert isinstance(reports["factor_a"], FactorReport)
        assert isinstance(reports["factor_b"], FactorReport)


def test_factor_filter_benchmark_all_builtin():
    """Test FactorFilter.benchmark_all_builtin."""
    mock_reader = create_mock_reader()

    # Clear the registry first to avoid factors that need 'open' column
    # The SignalDetector registers 'close_gt_open' which needs 'open'
    original_builtins = FactorRegistry._builtin_factors.copy()
    FactorRegistry._builtin_factors.clear()

    try:
        # Register a test factor that only uses close price
        def test_factor_func(df):
            return df["close"] > df["close"].shift(1).fillna(df["close"])

        test_factor = Factor(
            name="benchmark_test_factor",
            description="Benchmark test factor",
            definition=test_factor_func,
            type=FactorType.FUNCTION,
        )
        FactorRegistry.register_builtin(test_factor)

        # Patch SignalDetector to not register builtin factors
        with patch("src.tushare_db.reader.DataReader", return_value=mock_reader):
            with patch.object(SignalDetector, "_register_builtin_factors"):
                filter_obj = FactorFilter(
                    db_path="test.db",
                    n_simulations=100,
                    simulation_days=60,
                    alpha_threshold=1.5,
                    window=60,
                )

                reports = filter_obj.benchmark_all_builtin(
                    ts_codes=["000001.SZ"],
                    lookback_days=100,
                )

                # Should return at least our test factor
                assert isinstance(reports, dict)
                assert "benchmark_test_factor" in reports
    finally:
        # Restore original registry
        FactorRegistry._builtin_factors = original_builtins


def test_factor_filter_use_sample():
    """Test FactorFilter.filter with use_sample=True."""
    mock_reader = create_mock_reader()

    def simple_factor(df):
        return df["close"] > df["close"].shift(1).fillna(df["close"])

    factor = Factor(
        name="price_momentum",
        description="Close greater than previous close",
        definition=simple_factor,
        type=FactorType.FUNCTION,
    )

    with patch("src.tushare_db.reader.DataReader", return_value=mock_reader):
        filter_obj = FactorFilter(
            db_path="test.db",
            n_simulations=100,
            simulation_days=60,
            alpha_threshold=1.5,
            window=60,
        )

        # Pass 10 stocks but use_sample should limit to 5
        report = filter_obj.filter(
            factor=factor,
            ts_codes=["000001.SZ"] * 10,
            lookback_days=100,
            use_sample=True,
        )

        # Should only process 5 stocks
        assert len(report.results) <= 5


def test_factor_filter_empty_params_df():
    """Test FactorFilter.filter with empty parameter DataFrame."""
    mock_reader = Mock()
    mock_reader.get_stock_daily.return_value = pd.DataFrame()  # Empty data

    def simple_factor(df):
        return df["close"] > df["close"].shift(1).fillna(df["close"])

    factor = Factor(
        name="price_momentum",
        description="Close greater than previous close",
        definition=simple_factor,
        type=FactorType.FUNCTION,
    )

    with patch("src.tushare_db.reader.DataReader", return_value=mock_reader):
        filter_obj = FactorFilter(
            db_path="test.db",
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

        assert isinstance(report, FactorReport)
        assert report.factor == "price_momentum"
        assert len(report.results) == 0


def test_factor_filter_context_manager():
    """Test FactorFilter as context manager."""
    mock_reader = create_mock_reader()

    with patch("src.tushare_db.reader.DataReader", return_value=mock_reader):
        with FactorFilter(
            db_path="test.db",
            n_simulations=100,
            simulation_days=60,
        ) as filter_obj:
            assert filter_obj._reader is None  # Lazy init

        # After exit, reader should be closed
        assert filter_obj._reader is None


def test_factor_report_empty_results():
    """Test FactorReport with empty results."""
    report = FactorReport(
        factor="test_factor",
        timestamp=datetime.now(),
        parameters={},
        results=[],
    )

    assert report.summary == {"total_stocks": 0}
    assert report.dataframe is not None
    assert len(report.dataframe) == 0

    # Should still be able to save
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "empty_report")
        report.save(path)
        assert os.path.exists(f"{path}.md")
        assert os.path.exists(f"{path}.csv")


def test_factor_report_summary_generation():
    """Test FactorReport summary generation."""
    results = [
        TestResult(
            ts_code="000001.SZ",
            p_actual=0.3,
            p_random=0.2,
            alpha_ratio=1.5,
            n_signals_actual=30,
            n_signals_random=20.0,
            p_value=0.01,
            is_significant=True,
            recommendation="KEEP",
        ),
        TestResult(
            ts_code="000002.SZ",
            p_actual=0.25,
            p_random=0.2,
            alpha_ratio=1.25,
            n_signals_actual=25,
            n_signals_random=20.0,
            p_value=0.05,
            is_significant=False,
            recommendation="OPTIMIZE",
        ),
        TestResult(
            ts_code="000003.SZ",
            p_actual=0.15,
            p_random=0.2,
            alpha_ratio=0.75,
            n_signals_actual=15,
            n_signals_random=20.0,
            p_value=0.5,
            is_significant=False,
            recommendation="DISCARD",
        ),
    ]

    report = FactorReport(
        factor="test_factor",
        timestamp=datetime.now(),
        parameters={},
        results=results,
    )

    assert report.summary["total_stocks"] == 3
    assert report.summary["keep_count"] == 1
    assert report.summary["optimize_count"] == 1
    assert report.summary["discard_count"] == 1
    assert report.summary["significant_count"] == 1
    assert report.summary["keep_rate"] == 1 / 3
    assert report.summary["significant_rate"] == 1 / 3
