"""Tests for momentum-based technical indicator factors."""

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
def momentum_df():
    """Create DataFrame with clear momentum pattern for testing."""
    # Create data with a strong uptrend (momentum > 5%)
    close = [100.0] * 20
    # Strong momentum period: price increases by 10% over 20 days
    for i in range(1, 21):
        close.append(100.0 + i * 0.5)  # 100.5, 101.0, ... 110.0

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


@pytest.fixture
def breakout_df():
    """Create DataFrame with consolidation and breakout pattern."""
    # 20 days of consolidation (flat price)
    close = [100.0] * 20
    # Breakout day
    close.append(104.0)  # 4% breakout above 100

    open_price = [c - 0.5 for c in close]
    high = [c + 1.0 for c in close]
    low = [c - 1.0 for c in close]
    # Normal volume then spike
    vol = [1000000] * 20
    vol.append(2000000)  # 2x volume spike

    return pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'vol': vol
    })


@pytest.fixture
def acceleration_df():
    """Create DataFrame with price acceleration pattern."""
    # Acceleration: rate of price increase is speeding up
    # For acceleration signal: short-term momentum > long-term momentum AND acceleration increasing
    # Pattern: decline (negative long momentum), then explosive rally (positive short momentum)
    # This creates positive acceleration because short > long (which is negative)
    close = []
    price = 200.0

    # Days 0-19: decline (lose 2.0 per day = -40 total, long momentum negative)
    for i in range(20):
        price -= 2.0
        close.append(price)

    # Days 20-24: explosive rally (gain 10.0 per day = +50 total)
    # Short momentum = +50/160 = +31%, Long momentum = +10/200 = +5%
    for i in range(5):
        price += 10.0
        close.append(price)

    # Days 25-29: even faster rally (gain 15.0 per day = +75 total)
    # Short momentum increases, long momentum still catching up
    for i in range(5):
        price += 15.0
        close.append(price)

    # Days 30-34: continue acceleration (gain 20.0 per day = +100 total)
    for i in range(5):
        price += 20.0
        close.append(price)

    # Days 35-39: peak acceleration (gain 25.0 per day = +125 total)
    for i in range(5):
        price += 25.0
        close.append(price)

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


class TestMomentum20Day:
    """Test 20-day momentum factor."""

    def test_momentum_20_day_returns_series(self, sample_ohlcv_df):
        """Test that momentum_20_day returns a boolean Series."""
        result = builtin_factors.momentum_20_day(sample_ohlcv_df)

        assert isinstance(result, pd.Series)
        assert result.dtype == bool
        assert len(result) == len(sample_ohlcv_df)

    def test_momentum_20_day_with_momentum_pattern(self, momentum_df):
        """Test momentum detection with known momentum pattern."""
        result = builtin_factors.momentum_20_day(momentum_df, period=20, threshold=0.05)

        # Should detect momentum signals when price makes new high AND momentum > 5%
        # After day 40, we have 20 days of uptrend with ~10% gain
        assert result.sum() >= 1

    def test_momentum_20_day_missing_close(self):
        """Test momentum_20_day with missing close column."""
        df = pd.DataFrame({'open': [1, 2, 3]})

        with pytest.raises(ValueError, match="close"):
            builtin_factors.momentum_20_day(df)

    def test_momentum_20_day_custom_parameters(self, sample_ohlcv_df):
        """Test momentum_20_day with custom parameters."""
        result = builtin_factors.momentum_20_day(
            sample_ohlcv_df, period=10, threshold=0.03
        )

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv_df)

    def test_momentum_20_day_crossover_logic(self):
        """Test that momentum signal uses crossover logic."""
        # Create data where momentum crosses above threshold
        close = [100.0] * 21  # Flat, no momentum
        close.append(106.0)   # 6% gain over 20 days - crosses 5% threshold

        df = pd.DataFrame({'close': close})

        result = builtin_factors.momentum_20_day(df, period=20, threshold=0.05)

        # Last row should be True (crossed above threshold)
        assert result.iloc[-1] == True
        # Earlier rows should be False
        assert result.iloc[-2] == False


class TestPriceMomentumBreakout:
    """Test price momentum breakout factor."""

    def test_price_momentum_breakout_returns_series(self, sample_ohlcv_df):
        """Test that price_momentum_breakout returns a boolean Series."""
        result = builtin_factors.price_momentum_breakout(sample_ohlcv_df)

        assert isinstance(result, pd.Series)
        assert result.dtype == bool
        assert len(result) == len(sample_ohlcv_df)

    def test_price_momentum_breakout_with_breakout(self, breakout_df):
        """Test breakout detection with known breakout pattern."""
        result = builtin_factors.price_momentum_breakout(
            breakout_df, lookback=20, breakout_threshold=0.03
        )

        # Last row should be True (breakout with volume confirmation)
        assert result.iloc[-1] == True

    def test_price_momentum_breakout_missing_close(self):
        """Test price_momentum_breakout with missing close column."""
        df = pd.DataFrame({'open': [1, 2, 3]})

        with pytest.raises(ValueError, match="close"):
            builtin_factors.price_momentum_breakout(df)

    def test_price_momentum_breakout_without_volume(self):
        """Test breakout detection without volume column."""
        # Create data with price breakout but no volume column
        close = [100.0] * 20
        close.append(104.0)  # 4% breakout

        df = pd.DataFrame({'close': close})

        result = builtin_factors.price_momentum_breakout(
            df, lookback=20, breakout_threshold=0.03
        )

        # Should still work without volume (just price breakout)
        assert isinstance(result, pd.Series)
        assert result.dtype == bool
        assert result.iloc[-1] == True

    def test_price_momentum_breakout_custom_params(self, sample_ohlcv_df):
        """Test price_momentum_breakout with custom parameters."""
        result = builtin_factors.price_momentum_breakout(
            sample_ohlcv_df, lookback=10, breakout_threshold=0.05
        )

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv_df)

    def test_price_momentum_breakout_volume_confirmation(self):
        """Test that volume confirmation is required when volume exists."""
        # Price breakout but no volume spike
        close = [100.0] * 20
        close.append(104.0)  # 4% breakout

        vol = [1000000] * 21  # No volume spike

        df = pd.DataFrame({
            'close': close,
            'vol': vol
        })

        result = builtin_factors.price_momentum_breakout(
            df, lookback=20, breakout_threshold=0.03
        )

        # Without volume spike, should not trigger
        assert result.iloc[-1] == False


class TestPriceAcceleration:
    """Test price acceleration factor."""

    def test_price_acceleration_returns_series(self, sample_ohlcv_df):
        """Test that price_acceleration returns a boolean Series."""
        result = builtin_factors.price_acceleration(sample_ohlcv_df)

        assert isinstance(result, pd.Series)
        assert result.dtype == bool
        assert len(result) == len(sample_ohlcv_df)

    def test_price_acceleration_with_acceleration_pattern(self, acceleration_df):
        """Test acceleration detection with known acceleration pattern."""
        result = builtin_factors.price_acceleration(
            acceleration_df, short_period=5, long_period=20
        )

        # Should detect acceleration signals when short momentum > long momentum
        # and acceleration is increasing (occurs in early rally phase, days 21-24)
        assert result.sum() >= 1

    def test_price_acceleration_missing_close(self):
        """Test price_acceleration with missing close column."""
        df = pd.DataFrame({'open': [1, 2, 3]})

        with pytest.raises(ValueError, match="close"):
            builtin_factors.price_acceleration(df)

    def test_price_acceleration_custom_parameters(self, sample_ohlcv_df):
        """Test price_acceleration with custom parameters."""
        result = builtin_factors.price_acceleration(
            sample_ohlcv_df, short_period=3, long_period=10
        )

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv_df)

    def test_price_acceleration_positive_only(self):
        """Test that acceleration signal requires positive acceleration."""
        # Create data with deceleration (slowing down)
        # Start with fast increase, then slow down
        close = [100.0 + i * 1.0 for i in range(20)]  # Fast increase: +1.0/day
        close.extend([120.0 + i * 0.1 for i in range(1, 21)])  # Slow increase: +0.1/day

        df = pd.DataFrame({'close': close})

        result = builtin_factors.price_acceleration(
            df, short_period=5, long_period=20
        )

        # Should not signal during deceleration
        # When short momentum < long momentum, acceleration is negative
        signals_after_decel = result.iloc[35:].sum()
        assert signals_after_decel == 0

    def test_price_acceleration_calculation(self):
        """Test the acceleration calculation logic."""
        # Create data with clear acceleration
        # Decline (negative long momentum), then explosive rally (positive short momentum)
        close = []
        price = 200.0

        # Phase 1: decline (lose 2.0 per day for 20 days)
        for i in range(20):
            price -= 2.0
            close.append(price)

        # Phase 2: explosive rally (gain 10.0 per day for 5 days)
        for i in range(5):
            price += 10.0
            close.append(price)

        # Phase 3: even faster rally (gain 15.0 per day for 5 days)
        for i in range(5):
            price += 15.0
            close.append(price)

        # Phase 4: continue acceleration (gain 20.0 per day for 5 days)
        for i in range(5):
            price += 20.0
            close.append(price)

        # Phase 5: peak acceleration (gain 25.0 per day for 5 days)
        for i in range(5):
            price += 25.0
            close.append(price)

        df = pd.DataFrame({'close': close})

        result = builtin_factors.price_acceleration(
            df, short_period=5, long_period=20
        )

        # Should have signals in the early rally phase (days 21-24)
        # when short momentum exceeds long momentum and acceleration is increasing
        assert result.iloc[21:25].sum() >= 1
