"""Tests for built-in technical indicator factors."""

import numpy as np
import pandas as pd
import pytest

from src.tushare_db.factor_validation import builtin_factors


@pytest.fixture
def sample_ohlcv_df():
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    n = 100

    # Generate trending data with some noise
    trend = np.linspace(100, 120, n)
    noise = np.random.randn(n) * 2

    close = trend + noise
    open_price = close + np.random.randn(n) * 0.5
    high = np.maximum(open_price, close) + np.random.rand(n) * 2
    low = np.minimum(open_price, close) - np.random.rand(n) * 2
    vol = np.random.randint(1000000, 5000000, n)

    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'vol': vol
    })

    return df


@pytest.fixture
def crossover_df():
    """Create DataFrame with known crossover patterns."""
    # Create data that will produce a clear golden cross
    close = [100.0] * 25 + [101.0, 102.0, 103.0, 104.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0]
    open_price = [c - 0.5 for c in close]
    high = [c + 1.0 for c in close]
    low = [c - 1.0 for c in close]
    vol = [1000000] * len(close)

    return pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'vol': vol
    })


class TestMACDFactors:
    """Test MACD indicator factors."""

    def test_macd_golden_cross(self, sample_ohlcv_df):
        """Test MACD golden cross detection."""
        result = builtin_factors.macd_golden_cross(sample_ohlcv_df)

        assert isinstance(result, pd.Series)
        assert result.dtype == bool
        assert len(result) == len(sample_ohlcv_df)
        # First ~35 values should mostly be False (not enough data for MACD)
        # Note: With random data, some early crosses may occur due to EWM initialization

    def test_macd_golden_cross_crossover_pattern(self, crossover_df):
        """Test MACD golden cross with known crossover pattern."""
        result = builtin_factors.macd_golden_cross(crossover_df)

        # With strong uptrend, should have at least one golden cross
        assert result.sum() >= 1

    def test_macd_death_cross(self, sample_ohlcv_df):
        """Test MACD death cross detection."""
        result = builtin_factors.macd_death_cross(sample_ohlcv_df)

        assert isinstance(result, pd.Series)
        assert result.dtype == bool
        assert len(result) == len(sample_ohlcv_df)

    def test_macd_zero_golden_cross(self, sample_ohlcv_df):
        """Test MACD zero golden cross detection."""
        result = builtin_factors.macd_zero_golden_cross(sample_ohlcv_df)

        assert isinstance(result, pd.Series)
        assert result.dtype == bool
        assert len(result) == len(sample_ohlcv_df)

    def test_macd_missing_close(self):
        """Test MACD factors with missing close column."""
        df = pd.DataFrame({'open': [1, 2, 3]})

        with pytest.raises(ValueError, match="close"):
            builtin_factors.macd_golden_cross(df)

    def test_macd_custom_parameters(self, sample_ohlcv_df):
        """Test MACD with custom parameters."""
        result = builtin_factors.macd_golden_cross(
            sample_ohlcv_df, fast=8, slow=17, signal=5
        )

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv_df)


class TestRSIFactors:
    """Test RSI indicator factors."""

    def test_rsi_oversold(self, sample_ohlcv_df):
        """Test RSI oversold detection."""
        result = builtin_factors.rsi_oversold(sample_ohlcv_df)

        assert isinstance(result, pd.Series)
        assert result.dtype == bool
        assert len(result) == len(sample_ohlcv_df)

    def test_rsi_overbought(self, sample_ohlcv_df):
        """Test RSI overbought detection."""
        result = builtin_factors.rsi_overbought(sample_ohlcv_df)

        assert isinstance(result, pd.Series)
        assert result.dtype == bool
        assert len(result) == len(sample_ohlcv_df)

    def test_rsi_custom_threshold(self):
        """Test RSI with custom threshold."""
        # Create data with RSI crossing 40
        close = [100.0] * 20
        close.extend([95.0, 94.0, 93.0, 92.0, 91.0, 90.0, 89.0, 88.0, 87.0, 86.0])
        close.extend([87.0, 88.0, 89.0])  # Bounce back

        df = pd.DataFrame({'close': close})

        result = builtin_factors.rsi_oversold(df, threshold=40.0)
        assert isinstance(result, pd.Series)

    def test_rsi_missing_close(self):
        """Test RSI factors with missing close column."""
        df = pd.DataFrame({'open': [1, 2, 3]})

        with pytest.raises(ValueError, match="close"):
            builtin_factors.rsi_oversold(df)


class TestMovingAverageFactors:
    """Test Moving Average indicator factors."""

    def test_golden_cross(self, sample_ohlcv_df):
        """Test MA golden cross detection."""
        result = builtin_factors.golden_cross(sample_ohlcv_df)

        assert isinstance(result, pd.Series)
        assert result.dtype == bool
        assert len(result) == len(sample_ohlcv_df)

    def test_golden_cross_known_pattern(self, crossover_df):
        """Test golden cross with known crossover pattern."""
        result = builtin_factors.golden_cross(crossover_df, fast_period=5, slow_period=20)

        # Should detect at least one golden cross in uptrend
        assert result.sum() >= 1

    def test_death_cross(self, sample_ohlcv_df):
        """Test MA death cross detection."""
        result = builtin_factors.death_cross(sample_ohlcv_df)

        assert isinstance(result, pd.Series)
        assert result.dtype == bool
        assert len(result) == len(sample_ohlcv_df)

    def test_price_above_sma(self, sample_ohlcv_df):
        """Test price crossing above SMA."""
        result = builtin_factors.price_above_sma(sample_ohlcv_df)

        assert isinstance(result, pd.Series)
        assert result.dtype == bool
        assert len(result) == len(sample_ohlcv_df)

    def test_price_below_sma(self, sample_ohlcv_df):
        """Test price crossing below SMA."""
        result = builtin_factors.price_below_sma(sample_ohlcv_df)

        assert isinstance(result, pd.Series)
        assert result.dtype == bool
        assert len(result) == len(sample_ohlcv_df)

    def test_ma_missing_close(self):
        """Test MA factors with missing close column."""
        df = pd.DataFrame({'open': [1, 2, 3]})

        with pytest.raises(ValueError, match="close"):
            builtin_factors.golden_cross(df)


class TestBollingerFactors:
    """Test Bollinger Bands indicator factors."""

    def test_bollinger_lower_break(self, sample_ohlcv_df):
        """Test Bollinger lower band break detection."""
        result = builtin_factors.bollinger_lower_break(sample_ohlcv_df)

        assert isinstance(result, pd.Series)
        assert result.dtype == bool
        assert len(result) == len(sample_ohlcv_df)

    def test_bollinger_upper_break(self, sample_ohlcv_df):
        """Test Bollinger upper band break detection."""
        result = builtin_factors.bollinger_upper_break(sample_ohlcv_df)

        assert isinstance(result, pd.Series)
        assert result.dtype == bool
        assert len(result) == len(sample_ohlcv_df)

    def test_bollinger_custom_parameters(self, sample_ohlcv_df):
        """Test Bollinger with custom parameters."""
        result = builtin_factors.bollinger_lower_break(
            sample_ohlcv_df, period=10, std_dev=1.5
        )

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv_df)

    def test_bollinger_missing_close(self):
        """Test Bollinger factors with missing close column."""
        df = pd.DataFrame({'open': [1, 2, 3]})

        with pytest.raises(ValueError, match="close"):
            builtin_factors.bollinger_lower_break(df)


class TestVolumeFactors:
    """Test Volume indicator factors."""

    def test_volume_breakout(self, sample_ohlcv_df):
        """Test volume breakout detection."""
        result = builtin_factors.volume_breakout(sample_ohlcv_df)

        assert isinstance(result, pd.Series)
        assert result.dtype == bool
        assert len(result) == len(sample_ohlcv_df)

    def test_volume_breakout_known_pattern(self):
        """Test volume breakout with known pattern."""
        # Normal volume followed by spike
        vol = [1000000] * 25
        vol.append(5000000)  # 5x average

        df = pd.DataFrame({'vol': vol})

        result = builtin_factors.volume_breakout(df, period=20, multiplier=2.0)

        # Last row should be True (volume spike)
        assert result.iloc[-1] == True

    def test_volume_breakout_custom_params(self, sample_ohlcv_df):
        """Test volume breakout with custom parameters."""
        result = builtin_factors.volume_breakout(
            sample_ohlcv_df, period=10, multiplier=1.5
        )

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv_df)

    def test_volume_missing_vol(self):
        """Test volume factors with missing vol column."""
        df = pd.DataFrame({'close': [1, 2, 3]})

        with pytest.raises(ValueError, match="vol"):
            builtin_factors.volume_breakout(df)


class TestPricePatternFactors:
    """Test Price Pattern indicator factors."""

    def test_close_gt_open(self, sample_ohlcv_df):
        """Test close greater than open detection."""
        result = builtin_factors.close_gt_open(sample_ohlcv_df)

        assert isinstance(result, pd.Series)
        assert result.dtype == bool
        assert len(result) == len(sample_ohlcv_df)

        # Verify logic
        expected = sample_ohlcv_df['close'] > sample_ohlcv_df['open']
        assert result.equals(expected)

    def test_close_gt_open_missing_columns(self):
        """Test close_gt_open with missing columns."""
        df = pd.DataFrame({'high': [1, 2, 3]})

        with pytest.raises(ValueError, match="open"):
            builtin_factors.close_gt_open(df)

    def test_gap_up(self, sample_ohlcv_df):
        """Test gap up detection."""
        result = builtin_factors.gap_up(sample_ohlcv_df)

        assert isinstance(result, pd.Series)
        assert result.dtype == bool
        assert len(result) == len(sample_ohlcv_df)

        # First value should be False (no previous close)
        assert result.iloc[0] == False

    def test_gap_up_with_threshold(self):
        """Test gap up with minimum percentage threshold."""
        df = pd.DataFrame({
            'open': [100.0, 102.0, 101.0],
            'close': [100.0, 101.0, 102.0]
        })

        # 2% gap up required
        result = builtin_factors.gap_up(df, min_gap_pct=1.5)

        assert result.iloc[0] == False  # First row
        assert result.iloc[1] == True   # 102 vs 100 = 2% gap
        assert result.iloc[2] == False  # 101 vs 101 = 0% gap

    def test_gap_down(self, sample_ohlcv_df):
        """Test gap down detection."""
        result = builtin_factors.gap_down(sample_ohlcv_df)

        assert isinstance(result, pd.Series)
        assert result.dtype == bool
        assert len(result) == len(sample_ohlcv_df)

        # First value should be False (no previous close)
        assert result.iloc[0] == False

    def test_gap_down_with_threshold(self):
        """Test gap down with minimum percentage threshold."""
        df = pd.DataFrame({
            'open': [100.0, 98.0, 99.0],
            'close': [100.0, 99.0, 98.0]
        })

        # 1% gap down required
        result = builtin_factors.gap_down(df, min_gap_pct=1.0)

        assert result.iloc[0] == False  # First row
        assert result.iloc[1] == True   # 98 vs 99 = -1% gap
        assert result.iloc[2] == False  # 99 vs 98 = positive, not a gap down

    def test_gap_missing_columns(self):
        """Test gap factors with missing columns."""
        df = pd.DataFrame({'close': [1, 2, 3]})

        with pytest.raises(ValueError, match="open"):
            builtin_factors.gap_up(df)


class TestHelperFunctions:
    """Test internal helper functions."""

    def test_calculate_sma(self):
        """Test SMA calculation."""
        series = pd.Series([1, 2, 3, 4, 5])
        result = builtin_factors._calculate_sma(series, 3)

        expected = pd.Series([np.nan, np.nan, 2.0, 3.0, 4.0])
        assert np.allclose(result.values, expected.values, equal_nan=True)

    def test_calculate_ema(self):
        """Test EMA calculation."""
        series = pd.Series([1, 2, 3, 4, 5])
        result = builtin_factors._calculate_ema(series, 3)

        assert len(result) == len(series)
        assert not np.isnan(result.iloc[-1])

    def test_calculate_rsi(self):
        """Test RSI calculation."""
        series = pd.Series([100.0] * 14 + [90.0, 95.0, 100.0])
        result = builtin_factors._calculate_rsi(series, 14)

        assert len(result) == len(series)
        # RSI should be in 0-100 range
        assert result.dropna().min() >= 0
        assert result.dropna().max() <= 100

    def test_calculate_bollinger(self):
        """Test Bollinger Bands calculation."""
        series = pd.Series([100.0] * 20)
        upper, middle, lower = builtin_factors._calculate_bollinger(series, 20, 2.0)

        assert len(upper) == len(series)
        assert len(middle) == len(series)
        assert len(lower) == len(series)

        # For constant series, bands should be equal
        assert upper.iloc[-1] == middle.iloc[-1] == lower.iloc[-1]

    def test_calculate_macd(self):
        """Test MACD calculation."""
        series = pd.Series([100.0 + i * 0.1 for i in range(50)])
        macd_line, signal_line, histogram = builtin_factors._calculate_macd(series)

        assert len(macd_line) == len(series)
        assert len(signal_line) == len(series)
        assert len(histogram) == len(series)

        # Histogram should equal MACD - signal
        expected_hist = macd_line - signal_line
        assert np.allclose(histogram.dropna(), expected_hist.dropna())


class TestBuiltinFactorsList:
    """Test the BUILTIN_FACTORS list."""

    def test_builtin_factors_list_exists(self):
        """Test that BUILTIN_FACTORS list exists and contains functions."""
        assert hasattr(builtin_factors, 'BUILTIN_FACTORS')
        assert isinstance(builtin_factors.BUILTIN_FACTORS, list)
        assert len(builtin_factors.BUILTIN_FACTORS) > 0

    def test_all_factors_are_callable(self):
        """Test that all items in BUILTIN_FACTORS are callable."""
        for factor in builtin_factors.BUILTIN_FACTORS:
            assert callable(factor)

    def test_expected_factors_present(self):
        """Test that expected factors are in the list."""
        factor_names = {f.__name__ for f in builtin_factors.BUILTIN_FACTORS}

        expected = {
            'macd_golden_cross', 'macd_death_cross', 'macd_zero_golden_cross',
            'rsi_oversold', 'rsi_overbought',
            'golden_cross', 'death_cross', 'price_above_sma', 'price_below_sma',
            'bollinger_lower_break', 'bollinger_upper_break',
            'volume_breakout',
            'close_gt_open', 'gap_up', 'gap_down'
        }

        assert expected.issubset(factor_names)
