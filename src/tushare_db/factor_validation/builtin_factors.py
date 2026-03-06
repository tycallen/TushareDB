"""Built-in technical indicator factors for Monte Carlo Factor Validation System.

This module provides common technical indicator factors including:
- MACD indicators
- RSI indicators
- Moving Average indicators
- Bollinger Bands indicators
- Volume indicators
- Price pattern indicators
"""

from typing import Optional

import pandas as pd


def _validate_ohlcv(df: pd.DataFrame) -> None:
    """Validate that DataFrame has required OHLCV columns.

    Args:
        df: Input DataFrame

    Raises:
        ValueError: If required columns are missing
    """
    required = {'open', 'high', 'low', 'close', 'vol'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average.

    Args:
        series: Price series
        period: EMA period

    Returns:
        EMA series
    """
    return series.ewm(span=period, adjust=False).mean()


def _calculate_sma(series: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average.

    Args:
        series: Price series
        period: SMA period

    Returns:
        SMA series
    """
    return series.rolling(window=period).mean()


def _calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index.

    Args:
        series: Price series
        period: RSI period (default 14)

    Returns:
        RSI series (0-100)
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _calculate_bollinger(
    series: pd.Series,
    period: int = 20,
    std_dev: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands.

    Args:
        series: Price series
        period: Moving average period (default 20)
        std_dev: Standard deviation multiplier (default 2.0)

    Returns:
        Tuple of (upper_band, middle_band, lower_band)
    """
    middle = _calculate_sma(series, period)
    std = series.rolling(window=period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    return upper, middle, lower


def _calculate_macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD indicator.

    Args:
        series: Price series
        fast: Fast EMA period (default 12)
        slow: Slow EMA period (default 26)
        signal: Signal line period (default 9)

    Returns:
        Tuple of (macd_line, signal_line, histogram)
    """
    ema_fast = _calculate_ema(series, fast)
    ema_slow = _calculate_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


# =============================================================================
# MACD Factors
# =============================================================================

def macd_golden_cross(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> pd.Series:
    """MACD golden cross: MACD line crosses above signal line.

    This is a bullish signal indicating potential upward momentum.

    Args:
        df: DataFrame with 'close' column
        fast: Fast EMA period (default 12)
        slow: Slow EMA period (default 26)
        signal: Signal line period (default 9)

    Returns:
        Boolean Series indicating golden cross signals
    """
    if 'close' not in df.columns:
        raise ValueError("DataFrame must have 'close' column")

    macd_line, signal_line, _ = _calculate_macd(df['close'], fast, slow, signal)

    # Golden cross: MACD crosses above signal line
    prev_macd = macd_line.shift(1)
    prev_signal = signal_line.shift(1)

    return (macd_line > signal_line) & (prev_macd <= prev_signal)


def macd_death_cross(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> pd.Series:
    """MACD death cross: MACD line crosses below signal line.

    This is a bearish signal indicating potential downward momentum.

    Args:
        df: DataFrame with 'close' column
        fast: Fast EMA period (default 12)
        slow: Slow EMA period (default 26)
        signal: Signal line period (default 9)

    Returns:
        Boolean Series indicating death cross signals
    """
    if 'close' not in df.columns:
        raise ValueError("DataFrame must have 'close' column")

    macd_line, signal_line, _ = _calculate_macd(df['close'], fast, slow, signal)

    # Death cross: MACD crosses below signal line
    prev_macd = macd_line.shift(1)
    prev_signal = signal_line.shift(1)

    return (macd_line < signal_line) & (prev_macd >= prev_signal)


def macd_zero_golden_cross(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> pd.Series:
    """MACD zero golden cross: MACD histogram crosses above zero.

    This is a strong bullish signal indicating momentum shift from negative to positive.

    Args:
        df: DataFrame with 'close' column
        fast: Fast EMA period (default 12)
        slow: Slow EMA period (default 26)
        signal: Signal line period (default 9)

    Returns:
        Boolean Series indicating zero golden cross signals
    """
    if 'close' not in df.columns:
        raise ValueError("DataFrame must have 'close' column")

    _, _, histogram = _calculate_macd(df['close'], fast, slow, signal)

    # Zero golden cross: histogram crosses above zero
    prev_hist = histogram.shift(1)

    return (histogram > 0) & (prev_hist <= 0)


# =============================================================================
# RSI Factors
# =============================================================================

def rsi_oversold(
    df: pd.DataFrame,
    period: int = 14,
    threshold: float = 30.0
) -> pd.Series:
    """RSI oversold: RSI crosses above threshold (default 30).

    This is a bullish reversal signal indicating the asset may be undervalued
    and due for a bounce.

    Args:
        df: DataFrame with 'close' column
        period: RSI calculation period (default 14)
        threshold: Oversold threshold (default 30)

    Returns:
        Boolean Series indicating oversold bounce signals
    """
    if 'close' not in df.columns:
        raise ValueError("DataFrame must have 'close' column")

    rsi = _calculate_rsi(df['close'], period)
    prev_rsi = rsi.shift(1)

    # RSI crosses above threshold (leaving oversold territory)
    return (rsi > threshold) & (prev_rsi <= threshold)


def rsi_overbought(
    df: pd.DataFrame,
    period: int = 14,
    threshold: float = 70.0
) -> pd.Series:
    """RSI overbought: RSI crosses below threshold (default 70).

    This is a bearish reversal signal indicating the asset may be overvalued
    and due for a pullback.

    Args:
        df: DataFrame with 'close' column
        period: RSI calculation period (default 14)
        threshold: Overbought threshold (default 70)

    Returns:
        Boolean Series indicating overbought reversal signals
    """
    if 'close' not in df.columns:
        raise ValueError("DataFrame must have 'close' column")

    rsi = _calculate_rsi(df['close'], period)
    prev_rsi = rsi.shift(1)

    # RSI crosses below threshold (leaving overbought territory)
    return (rsi < threshold) & (prev_rsi >= threshold)


# =============================================================================
# Moving Average Factors
# =============================================================================

def golden_cross(
    df: pd.DataFrame,
    fast_period: int = 5,
    slow_period: int = 20
) -> pd.Series:
    """Moving average golden cross: fast MA crosses above slow MA.

    This is a bullish trend reversal signal.

    Args:
        df: DataFrame with 'close' column
        fast_period: Fast moving average period (default 5)
        slow_period: Slow moving average period (default 20)

    Returns:
        Boolean Series indicating golden cross signals
    """
    if 'close' not in df.columns:
        raise ValueError("DataFrame must have 'close' column")

    fast_ma = _calculate_sma(df['close'], fast_period)
    slow_ma = _calculate_sma(df['close'], slow_period)

    prev_fast = fast_ma.shift(1)
    prev_slow = slow_ma.shift(1)

    return (fast_ma > slow_ma) & (prev_fast <= prev_slow)


def death_cross(
    df: pd.DataFrame,
    fast_period: int = 5,
    slow_period: int = 20
) -> pd.Series:
    """Moving average death cross: fast MA crosses below slow MA.

    This is a bearish trend reversal signal.

    Args:
        df: DataFrame with 'close' column
        fast_period: Fast moving average period (default 5)
        slow_period: Slow moving average period (default 20)

    Returns:
        Boolean Series indicating death cross signals
    """
    if 'close' not in df.columns:
        raise ValueError("DataFrame must have 'close' column")

    fast_ma = _calculate_sma(df['close'], fast_period)
    slow_ma = _calculate_sma(df['close'], slow_period)

    prev_fast = fast_ma.shift(1)
    prev_slow = slow_ma.shift(1)

    return (fast_ma < slow_ma) & (prev_fast >= prev_slow)


def price_above_sma(
    df: pd.DataFrame,
    period: int = 20
) -> pd.Series:
    """Price above SMA: close price crosses above simple moving average.

    This is a bullish signal indicating price momentum above average.

    Args:
        df: DataFrame with 'close' column
        period: SMA period (default 20)

    Returns:
        Boolean Series indicating price crossing above SMA
    """
    if 'close' not in df.columns:
        raise ValueError("DataFrame must have 'close' column")

    sma = _calculate_sma(df['close'], period)
    prev_close = df['close'].shift(1)
    prev_sma = sma.shift(1)

    return (df['close'] > sma) & (prev_close <= prev_sma)


def price_below_sma(
    df: pd.DataFrame,
    period: int = 20
) -> pd.Series:
    """Price below SMA: close price crosses below simple moving average.

    This is a bearish signal indicating price momentum below average.

    Args:
        df: DataFrame with 'close' column
        period: SMA period (default 20)

    Returns:
        Boolean Series indicating price crossing below SMA
    """
    if 'close' not in df.columns:
        raise ValueError("DataFrame must have 'close' column")

    sma = _calculate_sma(df['close'], period)
    prev_close = df['close'].shift(1)
    prev_sma = sma.shift(1)

    return (df['close'] < sma) & (prev_close >= prev_sma)


# =============================================================================
# Bollinger Bands Factors
# =============================================================================

def bollinger_lower_break(
    df: pd.DataFrame,
    period: int = 20,
    std_dev: float = 2.0
) -> pd.Series:
    """Bollinger lower band break: price crosses above lower band.

    This is a potential bullish reversal signal indicating price bounce
    from oversold conditions.

    Args:
        df: DataFrame with 'close' column
        period: Bollinger Bands period (default 20)
        std_dev: Standard deviation multiplier (default 2.0)

    Returns:
        Boolean Series indicating lower band break signals
    """
    if 'close' not in df.columns:
        raise ValueError("DataFrame must have 'close' column")

    upper, middle, lower = _calculate_bollinger(df['close'], period, std_dev)
    prev_close = df['close'].shift(1)
    prev_lower = lower.shift(1)

    # Price crosses above lower band
    return (df['close'] > lower) & (prev_close <= prev_lower)


def bollinger_upper_break(
    df: pd.DataFrame,
    period: int = 20,
    std_dev: float = 2.0
) -> pd.Series:
    """Bollinger upper band break: price crosses below upper band.

    This is a potential bearish reversal signal indicating price pullback
    from overbought conditions.

    Args:
        df: DataFrame with 'close' column
        period: Bollinger Bands period (default 20)
        std_dev: Standard deviation multiplier (default 2.0)

    Returns:
        Boolean Series indicating upper band break signals
    """
    if 'close' not in df.columns:
        raise ValueError("DataFrame must have 'close' column")

    upper, middle, lower = _calculate_bollinger(df['close'], period, std_dev)
    prev_close = df['close'].shift(1)
    prev_upper = upper.shift(1)

    # Price crosses below upper band
    return (df['close'] < upper) & (prev_close >= prev_upper)


# =============================================================================
# Volume Factors
# =============================================================================

def volume_breakout(
    df: pd.DataFrame,
    period: int = 20,
    multiplier: float = 2.0
) -> pd.Series:
    """Volume breakout: volume exceeds average by multiplier.

    This indicates unusual trading activity that may precede price movement.

    Args:
        df: DataFrame with 'vol' column
        period: Volume average period (default 20)
        multiplier: Volume multiplier threshold (default 2.0)

    Returns:
        Boolean Series indicating volume breakout signals
    """
    if 'vol' not in df.columns:
        raise ValueError("DataFrame must have 'vol' column")

    avg_volume = df['vol'].rolling(window=period).mean()

    # Volume exceeds average by multiplier
    return df['vol'] > (avg_volume * multiplier)


# =============================================================================
# Price Pattern Factors
# =============================================================================

def close_gt_open(df: pd.DataFrame) -> pd.Series:
    """Close greater than open: bullish candlestick.

    A simple bullish signal indicating buying pressure.

    Args:
        df: DataFrame with 'open' and 'close' columns

    Returns:
        Boolean Series indicating bullish candles
    """
    if 'open' not in df.columns or 'close' not in df.columns:
        raise ValueError("DataFrame must have 'open' and 'close' columns")

    return df['close'] > df['open']


def gap_up(
    df: pd.DataFrame,
    min_gap_pct: float = 0.0
) -> pd.Series:
    """Gap up: open price higher than previous close.

    This indicates strong buying interest at market open.

    Args:
        df: DataFrame with 'open' and 'close' columns
        min_gap_pct: Minimum gap percentage (default 0.0)

    Returns:
        Boolean Series indicating gap up signals
    """
    if 'open' not in df.columns or 'close' not in df.columns:
        raise ValueError("DataFrame must have 'open' and 'close' columns")

    prev_close = df['close'].shift(1)
    gap_pct = (df['open'] - prev_close) / prev_close * 100

    return gap_pct > min_gap_pct


def gap_down(
    df: pd.DataFrame,
    min_gap_pct: float = 0.0
) -> pd.Series:
    """Gap down: open price lower than previous close.

    This indicates strong selling pressure at market open.

    Args:
        df: DataFrame with 'open' and 'close' columns
        min_gap_pct: Minimum gap percentage (default 0.0, use negative)

    Returns:
        Boolean Series indicating gap down signals
    """
    if 'open' not in df.columns or 'close' not in df.columns:
        raise ValueError("DataFrame must have 'open' and 'close' columns")

    prev_close = df['close'].shift(1)
    gap_pct = (df['open'] - prev_close) / prev_close * 100

    return gap_pct < -min_gap_pct


# List of all built-in factor functions for registration
BUILTIN_FACTORS = [
    # MACD
    macd_golden_cross,
    macd_death_cross,
    macd_zero_golden_cross,
    # RSI
    rsi_oversold,
    rsi_overbought,
    # Moving Average
    golden_cross,
    death_cross,
    price_above_sma,
    price_below_sma,
    # Bollinger
    bollinger_lower_break,
    bollinger_upper_break,
    # Volume
    volume_breakout,
    # Price patterns
    close_gt_open,
    gap_up,
    gap_down,
]
