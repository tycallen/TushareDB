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


# =============================================================================
# KDJ Factors
# =============================================================================

def _calculate_kdj(
    df: pd.DataFrame,
    n: int = 9,
    m1: int = 3,
    m2: int = 3
) -> tuple:
    """Calculate KDJ indicator.

    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        n: RSV period (default 9)
        m1: K smoothing period (default 3)
        m2: D smoothing period (default 3)

    Returns:
        Tuple of (K, D, J) series
    """
    low_list = df['low'].rolling(window=n, min_periods=n).min()
    high_list = df['high'].rolling(window=n, min_periods=n).max()
    rsv = (df['close'] - low_list) / (high_list - low_list) * 100

    K = rsv.ewm(alpha=1/m1, adjust=False).mean()
    D = K.ewm(alpha=1/m2, adjust=False).mean()
    J = 3 * K - 2 * D

    return K, D, J


def kdj_golden_cross(
    df: pd.DataFrame,
    n: int = 9,
    m1: int = 3,
    m2: int = 3
) -> pd.Series:
    """KDJ golden cross: K crosses above D from below 20 (oversold).

    This is a bullish reversal signal in oversold territory.

    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        n: RSV period (default 9)
        m1: K smoothing period (default 3)
        m2: D smoothing period (default 3)

    Returns:
        Boolean Series indicating KDJ golden cross signals
    """
    K, D, _ = _calculate_kdj(df, n, m1, m2)

    prev_K = K.shift(1)
    prev_D = D.shift(1)

    # K crosses above D, and both were in oversold region (< 20)
    return (K > D) & (prev_K <= prev_D) & (K < 20) & (D < 20)


def kdj_death_cross(
    df: pd.DataFrame,
    n: int = 9,
    m1: int = 3,
    m2: int = 3
) -> pd.Series:
    """KDJ death cross: K crosses below D from above 80 (overbought).

    This is a bearish reversal signal in overbought territory.

    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        n: RSV period (default 9)
        m1: K smoothing period (default 3)
        m2: D smoothing period (default 3)

    Returns:
        Boolean Series indicating KDJ death cross signals
    """
    K, D, _ = _calculate_kdj(df, n, m1, m2)

    prev_K = K.shift(1)
    prev_D = D.shift(1)

    # K crosses below D, and both were in overbought region (> 80)
    return (K < D) & (prev_K >= prev_D) & (K > 80) & (D > 80)


# =============================================================================
# Williams %R Factors
# =============================================================================

def _calculate_williams_r(
    df: pd.DataFrame,
    period: int = 14
) -> pd.Series:
    """Calculate Williams %R indicator.

    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        period: Lookback period (default 14)

    Returns:
        Williams %R series (range -100 to 0)
    """
    highest_high = df['high'].rolling(window=period).max()
    lowest_low = df['low'].rolling(window=period).min()

    williams_r = (highest_high - df['close']) / (highest_high - lowest_low) * -100
    return williams_r


def williams_r_oversold(
    df: pd.DataFrame,
    period: int = 14,
    threshold: float = -80.0
) -> pd.Series:
    """Williams %R oversold: crosses above threshold (default -80).

    This is a bullish reversal signal.

    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        period: Lookback period (default 14)
        threshold: Oversold threshold (default -80)

    Returns:
        Boolean Series indicating Williams %R oversold signals
    """
    williams_r = _calculate_williams_r(df, period)
    prev_wr = williams_r.shift(1)

    return (williams_r > threshold) & (prev_wr <= threshold)


def williams_r_overbought(
    df: pd.DataFrame,
    period: int = 14,
    threshold: float = -20.0
) -> pd.Series:
    """Williams %R overbought: crosses below threshold (default -20).

    This is a bearish reversal signal.

    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        period: Lookback period (default 14)
        threshold: Overbought threshold (default -20)

    Returns:
        Boolean Series indicating Williams %R overbought signals
    """
    williams_r = _calculate_williams_r(df, period)
    prev_wr = williams_r.shift(1)

    return (williams_r < threshold) & (prev_wr >= threshold)


# =============================================================================
# CCI Factors (Commodity Channel Index)
# =============================================================================

def _calculate_cci(
    df: pd.DataFrame,
    period: int = 20
) -> pd.Series:
    """Calculate Commodity Channel Index (CCI).

    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        period: Lookback period (default 20)

    Returns:
        CCI series
    """
    tp = (df['high'] + df['low'] + df['close']) / 3
    sma_tp = tp.rolling(window=period).mean()
    mean_deviation = tp.rolling(window=period).apply(lambda x: abs(x - x.mean()).mean(), raw=True)

    cci = (tp - sma_tp) / (0.015 * mean_deviation)
    return cci


def cci_oversold(
    df: pd.DataFrame,
    period: int = 20,
    threshold: float = -100.0
) -> pd.Series:
    """CCI oversold: crosses above threshold (default -100).

    This is a bullish reversal signal.

    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        period: Lookback period (default 20)
        threshold: Oversold threshold (default -100)

    Returns:
        Boolean Series indicating CCI oversold signals
    """
    cci = _calculate_cci(df, period)
    prev_cci = cci.shift(1)

    return (cci > threshold) & (prev_cci <= threshold)


def cci_overbought(
    df: pd.DataFrame,
    period: int = 20,
    threshold: float = 100.0
) -> pd.Series:
    """CCI overbought: crosses below threshold (default 100).

    This is a bearish reversal signal.

    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        period: Lookback period (default 20)
        threshold: Overbought threshold (default 100)

    Returns:
        Boolean Series indicating CCI overbought signals
    """
    cci = _calculate_cci(df, period)
    prev_cci = cci.shift(1)

    return (cci < threshold) & (prev_cci >= threshold)


# =============================================================================
# ATR Factors (Average True Range)
# =============================================================================

def _calculate_atr(
    df: pd.DataFrame,
    period: int = 14
) -> pd.Series:
    """Calculate Average True Range (ATR).

    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        period: ATR period (default 14)

    Returns:
        ATR series
    """
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift(1))
    low_close = abs(df['low'] - df['close'].shift(1))

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    return atr


def atr_breakout(
    df: pd.DataFrame,
    period: int = 14,
    multiplier: float = 2.0
) -> pd.Series:
    """ATR breakout: price breaks above previous close + multiplier * ATR.

    This indicates strong momentum with volatility expansion.

    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        period: ATR period (default 14)
        multiplier: ATR multiplier for breakout threshold (default 2.0)

    Returns:
        Boolean Series indicating ATR breakout signals
    """
    atr = _calculate_atr(df, period)
    prev_close = df['close'].shift(1)

    breakout_level = prev_close + multiplier * atr

    return df['close'] > breakout_level


def atr_breakdown(
    df: pd.DataFrame,
    period: int = 14,
    multiplier: float = 2.0
) -> pd.Series:
    """ATR breakdown: price breaks below previous close - multiplier * ATR.

    This indicates strong selling pressure with volatility expansion.

    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        period: ATR period (default 14)
        multiplier: ATR multiplier for breakdown threshold (default 2.0)

    Returns:
        Boolean Series indicating ATR breakdown signals
    """
    atr = _calculate_atr(df, period)
    prev_close = df['close'].shift(1)

    breakdown_level = prev_close - multiplier * atr

    return df['close'] < breakdown_level


# =============================================================================
# Candlestick Pattern Factors
# =============================================================================

def bullish_engulfing(df: pd.DataFrame) -> pd.Series:
    """Bullish engulfing: bullish candle completely engulfs previous bearish candle.

    This is a strong bullish reversal signal at the end of a downtrend.

    Args:
        df: DataFrame with 'open', 'high', 'low', 'close' columns

    Returns:
        Boolean Series indicating bullish engulfing patterns
    """
    prev_open = df['open'].shift(1)
    prev_close = df['close'].shift(1)

    # Previous candle is bearish (close < open)
    prev_bearish = prev_close < prev_open

    # Current candle is bullish (close > open)
    current_bullish = df['close'] > df['open']

    # Current candle engulfs previous (open < prev_close and close > prev_open)
    engulfs = (df['open'] < prev_close) & (df['close'] > prev_open)

    return prev_bearish & current_bullish & engulfs


def bearish_engulfing(df: pd.DataFrame) -> pd.Series:
    """Bearish engulfing: bearish candle completely engulfs previous bullish candle.

    This is a strong bearish reversal signal at the end of an uptrend.

    Args:
        df: DataFrame with 'open', 'high', 'low', 'close' columns

    Returns:
        Boolean Series indicating bearish engulfing patterns
    """
    prev_open = df['open'].shift(1)
    prev_close = df['close'].shift(1)

    # Previous candle is bullish (close > open)
    prev_bullish = prev_close > prev_open

    # Current candle is bearish (close < open)
    current_bearish = df['close'] < df['open']

    # Current candle engulfs previous (open > prev_close and close < prev_open)
    engulfs = (df['open'] > prev_close) & (df['close'] < prev_open)

    return prev_bullish & current_bearish & engulfs


def hammer(df: pd.DataFrame) -> pd.Series:
    """Hammer pattern: small body with long lower shadow, appears in downtrend.

    This is a bullish reversal signal after a decline.

    Args:
        df: DataFrame with 'open', 'high', 'low', 'close' columns

    Returns:
        Boolean Series indicating hammer patterns
    """
    body = abs(df['close'] - df['open'])
    lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
    upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)

    # Lower shadow at least 2x body
    long_lower = lower_shadow > (body * 2)

    # Small upper shadow (less than 10% of body)
    small_upper = upper_shadow < (body * 0.1)

    # In downtrend (3-day SMA declining)
    sma3 = df['close'].rolling(window=3).mean()
    downtrend = sma3 < sma3.shift(1)

    return long_lower & small_upper & downtrend


def shooting_star(df: pd.DataFrame) -> pd.Series:
    """Shooting star: small body with long upper shadow, appears in uptrend.

    This is a bearish reversal signal after a rise.

    Args:
        df: DataFrame with 'open', 'high', 'low', 'close' columns

    Returns:
        Boolean Series indicating shooting star patterns
    """
    body = abs(df['close'] - df['open'])
    upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
    lower_shadow = df[['open', 'close']].min(axis=1) - df['low']

    # Upper shadow at least 2x body
    long_upper = upper_shadow > (body * 2)

    # Small lower shadow (less than 10% of body)
    small_lower = lower_shadow < (body * 0.1)

    # In uptrend (3-day SMA rising)
    sma3 = df['close'].rolling(window=3).mean()
    uptrend = sma3 > sma3.shift(1)

    return long_upper & small_lower & uptrend


def doji(df: pd.DataFrame, max_body_pct: float = 0.1) -> pd.Series:
    """Doji: open and close are nearly equal, indicating indecision.

    This signals potential reversal when appearing after a trend.

    Args:
        df: DataFrame with 'open', 'high', 'low', 'close' columns
        max_body_pct: Maximum body size as percentage of range (default 0.1 = 10%)

    Returns:
        Boolean Series indicating doji patterns
    """
    body = abs(df['close'] - df['open'])
    range_size = df['high'] - df['low']

    # Body is very small relative to range
    small_body = body < (range_size * max_body_pct)

    return small_body


def three_white_soldiers(df: pd.DataFrame) -> pd.Series:
    """Three white soldiers: three consecutive bullish candles with higher closes.

    This is a strong bullish continuation signal.

    Args:
        df: DataFrame with 'open', 'high', 'low', 'close' columns

    Returns:
        Boolean Series indicating three white soldiers patterns
    """
    # Three consecutive bullish candles
    c1_bullish = df['close'].shift(2) > df['open'].shift(2)
    c2_bullish = df['close'].shift(1) > df['open'].shift(1)
    c3_bullish = df['close'] > df['open']

    # Each close higher than previous
    higher_closes = (
        (df['close'].shift(1) > df['close'].shift(2)) &
        (df['close'] > df['close'].shift(1))
    )

    return c1_bullish & c2_bullish & c3_bullish & higher_closes


def three_black_crows(df: pd.DataFrame) -> pd.Series:
    """Three black crows: three consecutive bearish candles with lower closes.

    This is a strong bearish continuation signal.

    Args:
        df: DataFrame with 'open', 'high', 'low', 'close' columns

    Returns:
        Boolean Series indicating three black crows patterns
    """
    # Three consecutive bearish candles
    c1_bearish = df['close'].shift(2) < df['open'].shift(2)
    c2_bearish = df['close'].shift(1) < df['open'].shift(1)
    c3_bearish = df['close'] < df['open']

    # Each close lower than previous
    lower_closes = (
        (df['close'].shift(1) < df['close'].shift(2)) &
        (df['close'] < df['close'].shift(1))
    )

    return c1_bearish & c2_bearish & c3_bearish & lower_closes


# List of all built-in factor functions for registration
BUILTIN_FACTORS = [
    # MACD
    macd_golden_cross,
    macd_death_cross,
    macd_zero_golden_cross,
    # RSI
    rsi_oversold,
    rsi_overbought,
    # KDJ
    kdj_golden_cross,
    kdj_death_cross,
    # Williams %R
    williams_r_oversold,
    williams_r_overbought,
    # CCI
    cci_oversold,
    cci_overbought,
    # Moving Average
    golden_cross,
    death_cross,
    price_above_sma,
    price_below_sma,
    # Bollinger
    bollinger_lower_break,
    bollinger_upper_break,
    # ATR
    atr_breakout,
    atr_breakdown,
    # Volume
    volume_breakout,
    # Candlestick Patterns
    bullish_engulfing,
    bearish_engulfing,
    hammer,
    shooting_star,
    doji,
    three_white_soldiers,
    three_black_crows,
    # Price patterns
    close_gt_open,
    gap_up,
    gap_down,
]
