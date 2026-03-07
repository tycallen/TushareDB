"""Tests for volatility-based technical indicator factors."""

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
def volatility_expansion_df():
    """Create DataFrame with volatility expansion pattern."""
    # Create data with low volatility followed by high volatility
    # Need at least 40 rows for ATR (20) + ATR average (20) calculations
    np.random.seed(42)
    close = []
    high = []
    low = []
    open_price = []

    # 39 days of low volatility (small daily ranges) to establish baseline
    price = 100.0
    for i in range(39):
        open_price.append(price)
        high.append(price + 1.0)  # Range of 2
        low.append(price - 1.0)
        close.append(price + np.random.uniform(-0.3, 0.3))
        price = close[-1]

    # Day 40: dramatic volatility expansion (very large range)
    open_price.append(price)
    high.append(price + 20.0)  # Much larger range (40 vs 2 = 20x)
    low.append(price - 20.0)
    close.append(price + 5.0)

    return pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close
    })


@pytest.fixture
def volatility_contraction_df():
    """Create DataFrame with volatility contraction (squeeze) pattern."""
    # Create data with high volatility followed by sustained low volatility
    # Need at least 60 days for proper ATR (20) + ATR average (20) + crossover detection
    np.random.seed(42)
    close = []
    high = []
    low = []
    open_price = []

    # 40 days of high volatility (large daily ranges) to establish baseline
    # This ensures ATR and ATR average both have valid values
    price = 100.0
    for i in range(40):
        open_price.append(price)
        high.append(price + 5.0)
        low.append(price - 5.0)
        close.append(price + np.random.uniform(-2.0, 2.0))
        price = close[-1]

    # 20 days of low volatility to bring ATR down below threshold
    for i in range(20):
        open_price.append(price)
        high.append(price + 0.1)  # Very small range
        low.append(price - 0.1)
        close.append(price + np.random.uniform(-0.05, 0.05))
        price = close[-1]

    return pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close
    })


@pytest.fixture
def atr_percent_b_df():
    """Create DataFrame for testing ATR %B factor."""
    # Create data where price moves to ATR band extremes
    # ATR %B uses SMA as middle band: upper = SMA + 2*ATR, lower = SMA - 2*ATR
    # %B = (close - lower) / (upper - lower)
    # Need at least 40 rows for proper ATR and SMA calculation
    np.random.seed(42)
    close = []
    high = []
    low = []
    open_price = []

    # 39 days of normal trading with stable price to establish ATR baseline
    # Price oscillates around 100 with small moves
    price = 100.0
    for i in range(39):
        open_price.append(price)
        high.append(price + 2.0)
        low.append(price - 2.0)
        # Small price changes to keep SMA stable around 100
        price_change = np.random.uniform(-0.5, 0.5)
        price = price + price_change
        close.append(price)

    # Calculate approximate SMA and ATR at this point
    # SMA ~98.5 (price has drifted down), ATR ~2 (from the 4-point daily range)
    # Bands: upper = 98.5 + 2*2 = 102.5, lower = 98.5 - 2*2 = 94.5
    # For %B > 0.8: close > lower + 0.8*(upper-lower) = 94.5 + 0.8*8 = 100.9
    # For %B < 0.2: close < lower + 0.2*(upper-lower) = 94.5 + 0.2*8 = 96.1

    # Day 40: price near upper ATR band (high %B > 0.8)
    # Need close > 100.9, use 105 for a clear signal
    open_price.append(100.0)
    high.append(106.0)
    low.append(104.0)
    close.append(105.0)  # This should give high %B

    # Day 41: price near lower ATR band (low %B < 0.2)
    # With SMA ~98.5 and ATR ~4.6, bands are ~89.2 to ~107.7
    # For %B < 0.2: close < lower + 0.2*(upper-lower) = 89.2 + 0.2*18.5 = 92.9
    open_price.append(105.0)
    high.append(90.0)
    low.append(88.0)
    close.append(89.0)  # This should give low %B

    return pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close
    })


@pytest.fixture
def bollinger_squeeze_df():
    """Create DataFrame with Bollinger Bands squeeze pattern."""
    # Create data with narrowing Bollinger Bands
    # Need at least 40 rows for proper Bollinger calculation
    np.random.seed(42)
    close = []
    high = []
    low = []
    open_price = []

    # 39 days of decreasing volatility (bands narrowing)
    price = 100.0
    for i in range(39):
        open_price.append(price)
        # Decreasing range over time
        range_size = max(0.2, 5.0 - (i * 0.12))  # From 5.0 down to ~0.4
        high.append(price + range_size / 2)
        low.append(price - range_size / 2)
        close.append(price + np.random.uniform(-range_size/4, range_size/4))
        price = close[-1]

    # Day 40: extreme squeeze (very narrow bands)
    open_price.append(price)
    high.append(price + 0.15)
    low.append(price - 0.15)
    close.append(price + 0.05)

    return pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close
    })


class TestVolatilityExpansion:
    """Test volatility expansion factor."""

    def test_volatility_expansion_returns_series(self, sample_ohlcv_df):
        """Test that volatility_expansion returns a boolean Series."""
        result = builtin_factors.volatility_expansion(sample_ohlcv_df)

        assert isinstance(result, pd.Series)
        assert result.dtype == bool
        assert len(result) == len(sample_ohlcv_df)

    def test_volatility_expansion_with_expansion_pattern(self, volatility_expansion_df):
        """Test volatility expansion detection with known pattern."""
        result = builtin_factors.volatility_expansion(
            volatility_expansion_df, period=20, expansion_threshold=1.5
        )

        # Should detect expansion on the last day
        assert result.iloc[-1] == True

    def test_volatility_expansion_missing_columns(self):
        """Test volatility_expansion with missing required columns."""
        df = pd.DataFrame({'close': [1, 2, 3]})

        with pytest.raises(ValueError, match="high"):
            builtin_factors.volatility_expansion(df)

    def test_volatility_expansion_custom_parameters(self, sample_ohlcv_df):
        """Test volatility_expansion with custom parameters."""
        result = builtin_factors.volatility_expansion(
            sample_ohlcv_df, period=10, expansion_threshold=2.0
        )

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv_df)

    def test_volatility_expansion_crossover_logic(self):
        """Test that volatility expansion uses crossover logic."""
        # Create data where ATR crosses above threshold
        # Need 40 rows minimum for ATR (20) + ATR average (20) to have valid values
        np.random.seed(42)
        close = []
        high = []
        low = []
        open_price = []

        # 39 days of low volatility to establish baseline
        price = 100.0
        for i in range(39):
            open_price.append(price)
            high.append(price + 1.0)  # Range of 2
            low.append(price - 1.0)
            close.append(price + np.random.uniform(-0.3, 0.3))
            price = close[-1]

        # Day 40: dramatic volatility expansion
        open_price.append(price)
        high.append(price + 20.0)  # Much larger range
        low.append(price - 20.0)
        close.append(price + 5.0)

        df = pd.DataFrame({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close
        })

        result = builtin_factors.volatility_expansion(df, period=20, expansion_threshold=1.5)

        # Last row should be True (crossed above threshold)
        assert result.iloc[-1] == True
        # Earlier rows should be False
        assert result.iloc[-2] == False

    def test_volatility_expansion_no_false_positives(self):
        """Test that stable volatility doesn't trigger expansion."""
        # Create data with stable volatility
        close = [100.0 + i * 0.1 for i in range(30)]
        high = [c + 2.0 for c in close]
        low = [c - 2.0 for c in close]
        open_price = [c - 0.1 for c in close]

        df = pd.DataFrame({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close
        })

        result = builtin_factors.volatility_expansion(df, period=20, expansion_threshold=1.5)

        # Should not trigger on stable volatility
        assert result.sum() == 0


class TestVolatilityContraction:
    """Test volatility contraction factor."""

    def test_volatility_contraction_returns_series(self, sample_ohlcv_df):
        """Test that volatility_contraction returns a boolean Series."""
        result = builtin_factors.volatility_contraction(sample_ohlcv_df)

        assert isinstance(result, pd.Series)
        assert result.dtype == bool
        assert len(result) == len(sample_ohlcv_df)

    def test_volatility_contraction_with_contraction_pattern(self, volatility_contraction_df):
        """Test volatility contraction detection with known pattern."""
        result = builtin_factors.volatility_contraction(
            volatility_contraction_df, period=20, contraction_threshold=0.7
        )

        # Should detect contraction during the low volatility period
        # After sustained low volatility, ATR should drop below threshold
        # Signal should occur when ATR crosses below threshold
        assert result.sum() >= 1

    def test_volatility_contraction_missing_columns(self):
        """Test volatility_contraction with missing required columns."""
        df = pd.DataFrame({'close': [1, 2, 3]})

        with pytest.raises(ValueError, match="high"):
            builtin_factors.volatility_contraction(df)

    def test_volatility_contraction_custom_parameters(self, sample_ohlcv_df):
        """Test volatility_contraction with custom parameters."""
        result = builtin_factors.volatility_contraction(
            sample_ohlcv_df, period=10, contraction_threshold=0.5
        )

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv_df)

    def test_volatility_contraction_crossover_logic(self):
        """Test that volatility contraction uses crossover logic."""
        # Create data where ATR crosses below threshold
        # Need sustained high volatility first, then sustained low volatility
        np.random.seed(42)
        close = []
        high = []
        low = []
        open_price = []

        # 40 days of high volatility to establish baseline
        # This ensures both ATR and ATR average have valid values
        price = 100.0
        for i in range(40):
            open_price.append(price)
            high.append(price + 5.0)  # Range of 10
            low.append(price - 5.0)
            close.append(price + np.random.uniform(-2.0, 2.0))
            price = close[-1]

        # 20 days of low volatility to bring ATR down below threshold
        for i in range(20):
            open_price.append(price)
            high.append(price + 0.1)  # Very small range
            low.append(price - 0.1)
            close.append(price + np.random.uniform(-0.05, 0.05))
            price = close[-1]

        df = pd.DataFrame({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close
        })

        result = builtin_factors.volatility_contraction(df, period=20, contraction_threshold=0.7)

        # Should detect contraction during low volatility period
        # At least one signal should occur
        assert result.sum() >= 1
        # Earlier high volatility period should not trigger
        assert result.iloc[:40].sum() == 0


class TestAtrPercentB:
    """Test ATR %B factor."""

    def test_atr_percent_b_returns_series(self, sample_ohlcv_df):
        """Test that atr_percent_b returns a boolean Series."""
        result = builtin_factors.atr_percent_b(sample_ohlcv_df)

        assert isinstance(result, pd.Series)
        assert result.dtype == bool
        assert len(result) == len(sample_ohlcv_df)

    def test_atr_percent_b_high_extreme(self, atr_percent_b_df):
        """Test ATR %B detection at upper extreme."""
        result = builtin_factors.atr_percent_b(
            atr_percent_b_df, period=20, upper=0.8, lower=0.2
        )

        # Day 40 should trigger (high %B - close near upper ATR band)
        assert result.iloc[39] == True

    def test_atr_percent_b_low_extreme(self, atr_percent_b_df):
        """Test ATR %B detection at lower extreme."""
        result = builtin_factors.atr_percent_b(
            atr_percent_b_df, period=20, upper=0.8, lower=0.2
        )

        # Day 41 should trigger (low %B - close near lower ATR band)
        assert result.iloc[40] == True

    def test_atr_percent_b_missing_columns(self):
        """Test atr_percent_b with missing required columns."""
        df = pd.DataFrame({'close': [1, 2, 3]})

        with pytest.raises(ValueError, match="high"):
            builtin_factors.atr_percent_b(df)

    def test_atr_percent_b_custom_parameters(self, sample_ohlcv_df):
        """Test atr_percent_b with custom parameters."""
        result = builtin_factors.atr_percent_b(
            sample_ohlcv_df, period=10, upper=0.9, lower=0.1
        )

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv_df)

    def test_atr_percent_b_crossover_logic(self):
        """Test that ATR %B uses crossover logic."""
        # Create data where %B crosses above upper threshold
        # Need 40 rows minimum for proper ATR and SMA calculation
        np.random.seed(42)
        close = []
        high = []
        low = []
        open_price = []

        # 39 days of stable price to establish baseline
        price = 100.0
        for i in range(39):
            open_price.append(price)
            high.append(price + 2.0)
            low.append(price - 2.0)
            price_change = np.random.uniform(-0.5, 0.5)
            price = price + price_change
            close.append(price)

        # Day 40: price jumps to upper extreme (high %B)
        # With SMA ~98.5 and ATR ~2, bands are ~94.5 to ~102.5
        # close = 105 gives %B > 0.8
        open_price.append(100.0)
        high.append(106.0)
        low.append(104.0)
        close.append(105.0)

        df = pd.DataFrame({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close
        })

        result = builtin_factors.atr_percent_b(df, period=20, upper=0.8, lower=0.2)

        # Last row should be True (crossed above upper threshold)
        assert result.iloc[-1] == True


class TestBollingerSqueeze:
    """Test Bollinger Bands squeeze factor."""

    def test_bollinger_squeeze_returns_series(self, sample_ohlcv_df):
        """Test that bollinger_squeeze returns a boolean Series."""
        result = builtin_factors.bollinger_squeeze(sample_ohlcv_df)

        assert isinstance(result, pd.Series)
        assert result.dtype == bool
        assert len(result) == len(sample_ohlcv_df)

    def test_bollinger_squeeze_with_squeeze_pattern(self, bollinger_squeeze_df):
        """Test Bollinger squeeze detection with known pattern."""
        result = builtin_factors.bollinger_squeeze(
            bollinger_squeeze_df, period=20, squeeze_threshold=0.1
        )

        # Should detect squeeze on the last day (extreme narrow bands)
        assert result.iloc[-1] == True

    def test_bollinger_squeeze_missing_close(self):
        """Test bollinger_squeeze with missing close column."""
        df = pd.DataFrame({'open': [1, 2, 3]})

        with pytest.raises(ValueError, match="close"):
            builtin_factors.bollinger_squeeze(df)

    def test_bollinger_squeeze_custom_parameters(self, sample_ohlcv_df):
        """Test bollinger_squeeze with custom parameters."""
        result = builtin_factors.bollinger_squeeze(
            sample_ohlcv_df, period=10, squeeze_threshold=0.05
        )

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv_df)

    def test_bollinger_squeeze_calculation(self):
        """Test the Bollinger squeeze calculation logic."""
        # Create data with very narrow Bollinger Bands
        close = [100.0] * 25
        # Small variations to create narrow bands
        for i in range(25):
            close[i] = 100.0 + np.random.uniform(-0.1, 0.1)

        df = pd.DataFrame({'close': close})

        result = builtin_factors.bollinger_squeeze(df, period=20, squeeze_threshold=0.1)

        # Should detect squeeze with very narrow bands
        assert result.iloc[-1] == True

    def test_bollinger_squeeze_no_false_positives(self):
        """Test that wide bands don't trigger squeeze."""
        # Create data with wide Bollinger Bands
        close = []
        for i in range(30):
            close.append(100.0 + np.sin(i) * 10)  # Large oscillations

        df = pd.DataFrame({'close': close})

        result = builtin_factors.bollinger_squeeze(df, period=20, squeeze_threshold=0.1)

        # Should not trigger on wide bands
        assert result.sum() == 0

    def test_bollinger_squeeze_band_width_calculation(self):
        """Test that band width is calculated correctly."""
        # Create data with known band width
        # Flat price = zero standard deviation = zero band width
        close = [100.0] * 25

        df = pd.DataFrame({'close': close})

        result = builtin_factors.bollinger_squeeze(df, period=20, squeeze_threshold=0.1)

        # Zero band width should be below any positive threshold
        assert result.iloc[-1] == True
