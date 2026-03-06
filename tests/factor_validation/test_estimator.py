"""Tests for ParameterEstimator module."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock, MagicMock


def create_mock_reader():
    """Create a mock DataReader for testing."""
    mock_reader = Mock()
    return mock_reader


def test_estimator_import():
    """Test module can be imported."""
    from src.tushare_db.factor_validation.estimator import ParameterEstimator
    assert ParameterEstimator is not None


def test_estimator_init():
    """Test ParameterEstimator initialization."""
    from src.tushare_db.factor_validation.estimator import ParameterEstimator

    mock_reader = create_mock_reader()
    estimator = ParameterEstimator(reader=mock_reader, window=252)

    assert estimator.reader is mock_reader
    assert estimator.window == 252


def test_calculate_log_returns():
    """Test log returns calculation."""
    from src.tushare_db.factor_validation.estimator import ParameterEstimator

    mock_reader = create_mock_reader()
    estimator = ParameterEstimator(reader=mock_reader)

    # Create sample price data
    prices = pd.Series([100, 110, 121, 133.1])

    returns = estimator._calculate_log_returns(prices)

    # Expected: ln(110/100), ln(121/110), ln(133.1/121)
    expected = pd.Series([np.nan, np.log(1.1), np.log(1.1), np.log(1.1)])

    assert len(returns) == 4
    assert np.isnan(returns.iloc[0])
    assert np.allclose(returns.iloc[1:].values, expected.iloc[1:].values)


def test_annualize_parameters():
    """Test annualization of daily parameters."""
    from src.tushare_db.factor_validation.estimator import ParameterEstimator

    mock_reader = create_mock_reader()
    estimator = ParameterEstimator(reader=mock_reader)

    daily_mu = 0.001  # 0.1% daily return
    daily_sigma = 0.02  # 2% daily volatility

    annual_mu, annual_sigma = estimator._annualize_parameters(daily_mu, daily_sigma)

    # Expected: mu * 252, sigma * sqrt(252)
    expected_mu = daily_mu * 252
    expected_sigma = daily_sigma * np.sqrt(252)

    assert np.isclose(annual_mu, expected_mu)
    assert np.isclose(annual_sigma, expected_sigma)


def test_calculate_p_actual():
    """Test calculation of actual signal probability."""
    from src.tushare_db.factor_validation.estimator import ParameterEstimator
    from src.tushare_db.factor_validation.factor import Factor, FactorType

    mock_reader = create_mock_reader()
    estimator = ParameterEstimator(reader=mock_reader)

    # Create a simple factor
    def simple_factor(df):
        return df['close'] > df['open']

    factor = Factor(
        name="test_factor",
        description="Test factor",
        definition=simple_factor,
        type=FactorType.FUNCTION
    )

    # Create sample data where 3 out of 5 rows have close > open
    df = pd.DataFrame({
        'open': [10, 11, 12, 13, 14],
        'close': [11, 10, 13, 12, 15]  # True, False, True, False, True
    })

    p_actual = estimator.calculate_p_actual(df, factor)

    # 3 out of 5 signals are True
    assert p_actual == 0.6


def test_estimate_parameters_single_stock():
    """Test parameter estimation for a single stock."""
    from src.tushare_db.factor_validation.estimator import ParameterEstimator
    from src.tushare_db.factor_validation.factor import Factor, FactorType

    # Create mock reader with sample data
    mock_reader = create_mock_reader()

    # Create sample daily price data (consistent 10% growth)
    dates = pd.date_range('2023-01-01', periods=100, freq='B')
    prices = 100 * (1.001 ** np.arange(100))  # Small daily growth

    sample_data = pd.DataFrame({
        'ts_code': ['000001.SZ'] * 100,
        'trade_date': dates,
        'open': prices * 0.99,
        'high': prices * 1.02,
        'low': prices * 0.98,
        'close': prices,
        'vol': [10000] * 100
    })

    mock_reader.get_stock_daily.return_value = sample_data

    estimator = ParameterEstimator(reader=mock_reader, window=60)

    # Create a simple factor
    def simple_factor(df):
        return df['close'] > df['open']

    factor = Factor(
        name="close_gt_open",
        description="Close greater than open",
        definition=simple_factor,
        type=FactorType.FUNCTION
    )

    result = estimator.estimate_parameters(
        ts_codes=['000001.SZ'],
        lookback_days=100,
        factor=factor
    )

    # Check result structure
    assert isinstance(result, pd.DataFrame)
    assert 'ts_code' in result.columns
    assert 'mu' in result.columns
    assert 'sigma' in result.columns
    assert 'p_actual' in result.columns

    # Check values
    assert len(result) == 1
    assert result.iloc[0]['ts_code'] == '000001.SZ'
    assert result.iloc[0]['mu'] > 0  # Positive drift due to price growth
    assert result.iloc[0]['sigma'] > 0  # Positive volatility
    assert 0 <= result.iloc[0]['p_actual'] <= 1  # Probability between 0 and 1


def test_estimate_parameters_multiple_stocks():
    """Test parameter estimation for multiple stocks."""
    from src.tushare_db.factor_validation.estimator import ParameterEstimator
    from src.tushare_db.factor_validation.factor import Factor, FactorType

    mock_reader = create_mock_reader()

    # Create sample data for two stocks
    dates = pd.date_range('2023-01-01', periods=100, freq='B')

    def make_stock_data(ts_code, growth_rate):
        prices = 100 * (growth_rate ** np.arange(100))
        return pd.DataFrame({
            'ts_code': [ts_code] * 100,
            'trade_date': dates,
            'open': prices * 0.99,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'vol': [10000] * 100
        })

    def mock_get_stock_daily(ts_code, **kwargs):
        if ts_code == '000001.SZ':
            return make_stock_data(ts_code, 1.001)
        elif ts_code == '000002.SZ':
            return make_stock_data(ts_code, 1.0005)
        return pd.DataFrame()

    mock_reader.get_stock_daily.side_effect = mock_get_stock_daily

    estimator = ParameterEstimator(reader=mock_reader, window=60)

    def simple_factor(df):
        return df['close'] > df['open']

    factor = Factor(
        name="close_gt_open",
        description="Close greater than open",
        definition=simple_factor,
        type=FactorType.FUNCTION
    )

    result = estimator.estimate_parameters(
        ts_codes=['000001.SZ', '000002.SZ'],
        lookback_days=100,
        factor=factor
    )

    assert len(result) == 2
    assert set(result['ts_code'].values) == {'000001.SZ', '000002.SZ'}


def test_estimate_parameters_empty_data():
    """Test handling of empty data."""
    from src.tushare_db.factor_validation.estimator import ParameterEstimator
    from src.tushare_db.factor_validation.factor import Factor, FactorType

    mock_reader = create_mock_reader()
    mock_reader.get_stock_daily.return_value = pd.DataFrame()

    estimator = ParameterEstimator(reader=mock_reader)

    def simple_factor(df):
        return df['close'] > df['open']

    factor = Factor(
        name="test_factor",
        description="Test factor",
        definition=simple_factor,
        type=FactorType.FUNCTION
    )

    result = estimator.estimate_parameters(
        ts_codes=['000001.SZ'],
        lookback_days=100,
        factor=factor
    )

    # Should return empty DataFrame or handle gracefully
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


def test_estimate_parameters_insufficient_data():
    """Test handling of insufficient data."""
    from src.tushare_db.factor_validation.estimator import ParameterEstimator
    from src.tushare_db.factor_validation.factor import Factor, FactorType

    mock_reader = create_mock_reader()

    # Only 5 rows of data
    sample_data = pd.DataFrame({
        'ts_code': ['000001.SZ'] * 5,
        'trade_date': pd.date_range('2023-01-01', periods=5, freq='B'),
        'open': [10, 11, 12, 13, 14],
        'high': [11, 12, 13, 14, 15],
        'low': [9, 10, 11, 12, 13],
        'close': [10.5, 11.5, 12.5, 13.5, 14.5],
        'vol': [10000] * 5
    })

    mock_reader.get_stock_daily.return_value = sample_data

    estimator = ParameterEstimator(reader=mock_reader, window=20)

    def simple_factor(df):
        return df['close'] > df['open']

    factor = Factor(
        name="test_factor",
        description="Test factor",
        definition=simple_factor,
        type=FactorType.FUNCTION
    )

    result = estimator.estimate_parameters(
        ts_codes=['000001.SZ'],
        lookback_days=10,
        factor=factor
    )

    # Should handle gracefully with NaN values or skip
    assert isinstance(result, pd.DataFrame)


def test_estimate_parameters_with_real_dates():
    """Test parameter estimation with proper date filtering."""
    from src.tushare_db.factor_validation.estimator import ParameterEstimator
    from src.tushare_db.factor_validation.factor import Factor, FactorType

    mock_reader = create_mock_reader()

    # Create data spanning more than lookback period
    dates = pd.date_range('2023-01-01', periods=300, freq='B')
    prices = 100 * (1.0005 ** np.arange(300))

    sample_data = pd.DataFrame({
        'ts_code': ['000001.SZ'] * 300,
        'trade_date': dates,
        'open': prices * 0.99,
        'high': prices * 1.02,
        'low': prices * 0.98,
        'close': prices,
        'vol': [10000] * 300
    })

    mock_reader.get_stock_daily.return_value = sample_data

    estimator = ParameterEstimator(reader=mock_reader, window=60)

    def simple_factor(df):
        return df['close'] > df['open']

    factor = Factor(
        name="close_gt_open",
        description="Close greater than open",
        definition=simple_factor,
        type=FactorType.FUNCTION
    )

    # Request 100 days of lookback
    result = estimator.estimate_parameters(
        ts_codes=['000001.SZ'],
        lookback_days=100,
        factor=factor
    )

    assert len(result) == 1
    # Verify get_stock_daily was called with correct parameters
    mock_reader.get_stock_daily.assert_called_once()
    call_args = mock_reader.get_stock_daily.call_args
    # call_args.kwargs contains the keyword arguments
    assert call_args.kwargs['ts_code'] == '000001.SZ'
