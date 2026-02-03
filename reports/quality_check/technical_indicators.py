#!/usr/bin/env python3
"""
技术指标特征工程模块

包含完整的技术指标计算实现，涵盖:
- 趋势类指标 (MA, EMA, MACD, ADX, Aroon)
- 动量类指标 (RSI, Stochastic, Williams %R, ROC, CCI, MFI)
- 波动率指标 (Bollinger Bands, ATR, Keltner Channel, Historical Volatility)
- 成交量指标 (OBV, Volume MA, Volume Ratio, VWAP)
- 价格结构特征 (相对位置, 影线比例, 缺口检测)

作者: 量化特征工程专家
日期: 2025-01
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# 趋势类指标
# =============================================================================

def calculate_ma(series: pd.Series, periods: List[int] = [5, 10, 20, 60, 120, 250]) -> pd.DataFrame:
    """
    计算简单移动平均线 (Simple Moving Average)

    公式: MA(n) = sum(close[i], i=0 to n-1) / n

    Args:
        series: 价格序列
        periods: 均线周期列表

    Returns:
        DataFrame with MA columns
    """
    result = pd.DataFrame(index=series.index)
    for period in periods:
        result[f'ma_{period}'] = series.rolling(window=period, min_periods=1).mean()
    return result


def calculate_ema(series: pd.Series, periods: List[int] = [12, 26]) -> pd.DataFrame:
    """
    计算指数移动平均线 (Exponential Moving Average)

    公式: EMA(t) = α * price(t) + (1-α) * EMA(t-1)
          其中 α = 2 / (period + 1)

    Args:
        series: 价格序列
        periods: EMA周期列表

    Returns:
        DataFrame with EMA columns
    """
    result = pd.DataFrame(index=series.index)
    for period in periods:
        result[f'ema_{period}'] = series.ewm(span=period, adjust=False).mean()
    return result


def calculate_macd(series: pd.Series,
                   fast_period: int = 12,
                   slow_period: int = 26,
                   signal_period: int = 9) -> pd.DataFrame:
    """
    计算 MACD (Moving Average Convergence Divergence)

    公式:
        MACD Line = EMA(12) - EMA(26)
        Signal Line = EMA(MACD, 9)
        Histogram = MACD Line - Signal Line

    Args:
        series: 价格序列
        fast_period: 快线周期
        slow_period: 慢线周期
        signal_period: 信号线周期

    Returns:
        DataFrame with MACD, Signal, Histogram
    """
    ema_fast = series.ewm(span=fast_period, adjust=False).mean()
    ema_slow = series.ewm(span=slow_period, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line

    return pd.DataFrame({
        'macd': macd_line,
        'macd_signal': signal_line,
        'macd_hist': histogram
    }, index=series.index)


def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series,
                  period: int = 14) -> pd.DataFrame:
    """
    计算 ADX (Average Directional Index) 趋势强度指标

    公式:
        +DM = high(t) - high(t-1) if > 0 and > -DM, else 0
        -DM = low(t-1) - low(t) if > 0 and > +DM, else 0
        TR = max(high-low, |high-close_prev|, |low-close_prev|)
        +DI = 100 * smoothed(+DM) / smoothed(TR)
        -DI = 100 * smoothed(-DM) / smoothed(TR)
        DX = 100 * |+DI - -DI| / (+DI + -DI)
        ADX = smoothed(DX)

    Args:
        high, low, close: OHLC 数据
        period: 计算周期

    Returns:
        DataFrame with ADX, +DI, -DI
    """
    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    # Smoothed values using Wilder's method
    tr_smooth = pd.Series(tr).ewm(alpha=1/period, adjust=False).mean()
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean()
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean()

    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth

    # ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()

    return pd.DataFrame({
        'adx': adx.values,
        'plus_di': plus_di.values,
        'minus_di': minus_di.values
    }, index=high.index)


def calculate_aroon(high: pd.Series, low: pd.Series, period: int = 25) -> pd.DataFrame:
    """
    计算 Aroon 指标 (趋势方向和强度)

    公式:
        Aroon Up = 100 * (period - days since highest high) / period
        Aroon Down = 100 * (period - days since lowest low) / period
        Aroon Oscillator = Aroon Up - Aroon Down

    Args:
        high, low: 最高价/最低价序列
        period: 回溯周期

    Returns:
        DataFrame with Aroon Up, Down, Oscillator
    """
    aroon_up = high.rolling(window=period+1).apply(
        lambda x: ((period - (period - x.argmax())) / period) * 100,
        raw=True
    )
    aroon_down = low.rolling(window=period+1).apply(
        lambda x: ((period - (period - x.argmin())) / period) * 100,
        raw=True
    )

    return pd.DataFrame({
        'aroon_up': aroon_up,
        'aroon_down': aroon_down,
        'aroon_osc': aroon_up - aroon_down
    }, index=high.index)


# =============================================================================
# 动量类指标
# =============================================================================

def calculate_rsi(series: pd.Series, periods: List[int] = [6, 14, 24]) -> pd.DataFrame:
    """
    计算 RSI (Relative Strength Index) 相对强弱指标

    公式:
        RS = avg_gain / avg_loss
        RSI = 100 - (100 / (1 + RS))

    其中 avg_gain/avg_loss 使用 Wilder's smoothing

    Args:
        series: 价格序列
        periods: RSI 周期列表

    Returns:
        DataFrame with RSI columns
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    result = pd.DataFrame(index=series.index)
    for period in periods:
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        result[f'rsi_{period}'] = 100 - (100 / (1 + rs))

    return result


def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                         k_period: int = 14, d_period: int = 3,
                         smooth_k: int = 3) -> pd.DataFrame:
    """
    计算 Stochastic Oscillator (KDJ 指标的国际版)

    公式:
        %K = 100 * (close - lowest_low) / (highest_high - lowest_low)
        %K_smooth = SMA(%K, smooth_k)
        %D = SMA(%K_smooth, d_period)

    Args:
        high, low, close: OHLC 数据
        k_period: %K 计算周期
        d_period: %D 平滑周期
        smooth_k: %K 平滑周期

    Returns:
        DataFrame with %K, %D
    """
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()

    stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
    stoch_k_smooth = stoch_k.rolling(window=smooth_k).mean()
    stoch_d = stoch_k_smooth.rolling(window=d_period).mean()

    return pd.DataFrame({
        'stoch_k': stoch_k_smooth,
        'stoch_d': stoch_d
    }, index=close.index)


def calculate_williams_r(high: pd.Series, low: pd.Series, close: pd.Series,
                         period: int = 14) -> pd.Series:
    """
    计算 Williams %R

    公式: %R = -100 * (highest_high - close) / (highest_high - lowest_low)

    解释: -20 以上超买, -80 以下超卖

    Args:
        high, low, close: OHLC 数据
        period: 计算周期

    Returns:
        Williams %R Series
    """
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()

    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    williams_r.name = 'williams_r'
    return williams_r


def calculate_roc(series: pd.Series, period: int = 12) -> pd.Series:
    """
    计算 ROC (Rate of Change) 变动率

    公式: ROC = (close - close[n]) / close[n] * 100

    Args:
        series: 价格序列
        period: 回溯周期

    Returns:
        ROC Series
    """
    roc = (series - series.shift(period)) / series.shift(period) * 100
    roc.name = 'roc'
    return roc


def calculate_cci(high: pd.Series, low: pd.Series, close: pd.Series,
                  period: int = 20) -> pd.Series:
    """
    计算 CCI (Commodity Channel Index) 商品通道指数

    公式:
        TP = (high + low + close) / 3
        CCI = (TP - SMA(TP)) / (0.015 * MeanDeviation)

    解释: +100 以上超买, -100 以下超卖

    Args:
        high, low, close: OHLC 数据
        period: 计算周期

    Returns:
        CCI Series
    """
    tp = (high + low + close) / 3
    tp_sma = tp.rolling(window=period).mean()

    # Mean deviation
    mean_dev = tp.rolling(window=period).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=True
    )

    cci = (tp - tp_sma) / (0.015 * mean_dev + 1e-10)
    cci.name = 'cci'
    return cci


def calculate_mfi(high: pd.Series, low: pd.Series, close: pd.Series,
                  volume: pd.Series, period: int = 14) -> pd.Series:
    """
    计算 MFI (Money Flow Index) 资金流量指数

    公式:
        TP = (high + low + close) / 3
        Raw Money Flow = TP * volume
        Money Ratio = sum(positive MF) / sum(negative MF)
        MFI = 100 - (100 / (1 + Money Ratio))

    解释: 类似 RSI，但考虑了成交量

    Args:
        high, low, close, volume: OHLC+V 数据
        period: 计算周期

    Returns:
        MFI Series
    """
    tp = (high + low + close) / 3
    raw_mf = tp * volume

    tp_diff = tp.diff()
    positive_mf = np.where(tp_diff > 0, raw_mf, 0)
    negative_mf = np.where(tp_diff < 0, raw_mf, 0)

    positive_mf_sum = pd.Series(positive_mf).rolling(window=period).sum()
    negative_mf_sum = pd.Series(negative_mf).rolling(window=period).sum()

    money_ratio = positive_mf_sum / (negative_mf_sum + 1e-10)
    mfi = 100 - (100 / (1 + money_ratio))
    mfi.name = 'mfi'
    return pd.Series(mfi.values, index=close.index, name='mfi')


# =============================================================================
# 波动率指标
# =============================================================================

def calculate_bollinger_bands(series: pd.Series, period: int = 20,
                              std_dev: float = 2.0) -> pd.DataFrame:
    """
    计算 Bollinger Bands 布林带

    公式:
        Middle Band = SMA(close, 20)
        Upper Band = Middle Band + 2 * std(close, 20)
        Lower Band = Middle Band - 2 * std(close, 20)
        %B = (close - Lower) / (Upper - Lower)
        Bandwidth = (Upper - Lower) / Middle

    Args:
        series: 价格序列
        period: 移动平均周期
        std_dev: 标准差倍数

    Returns:
        DataFrame with bands, %B, bandwidth
    """
    middle = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()

    upper = middle + std_dev * std
    lower = middle - std_dev * std

    percent_b = (series - lower) / (upper - lower + 1e-10)
    bandwidth = (upper - lower) / (middle + 1e-10)

    return pd.DataFrame({
        'bb_middle': middle,
        'bb_upper': upper,
        'bb_lower': lower,
        'bb_percent_b': percent_b,
        'bb_bandwidth': bandwidth
    }, index=series.index)


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series,
                  period: int = 14) -> pd.Series:
    """
    计算 ATR (Average True Range) 真实波动幅度

    公式:
        TR = max(high-low, |high-close_prev|, |low-close_prev|)
        ATR = Wilder's smoothed average of TR

    用途: 衡量波动性，常用于止损和仓位管理

    Args:
        high, low, close: OHLC 数据
        period: 计算周期

    Returns:
        ATR Series
    """
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    atr.name = 'atr'
    return atr


def calculate_keltner_channel(high: pd.Series, low: pd.Series, close: pd.Series,
                               ema_period: int = 20, atr_period: int = 10,
                               atr_mult: float = 2.0) -> pd.DataFrame:
    """
    计算 Keltner Channel 肯特纳通道

    公式:
        Middle = EMA(close, 20)
        Upper = Middle + 2 * ATR(10)
        Lower = Middle - 2 * ATR(10)

    与布林带类似，但使用 ATR 而非标准差

    Args:
        high, low, close: OHLC 数据
        ema_period: EMA 周期
        atr_period: ATR 周期
        atr_mult: ATR 倍数

    Returns:
        DataFrame with Keltner Channel bands
    """
    middle = close.ewm(span=ema_period, adjust=False).mean()
    atr = calculate_atr(high, low, close, atr_period)

    upper = middle + atr_mult * atr
    lower = middle - atr_mult * atr

    return pd.DataFrame({
        'kc_middle': middle,
        'kc_upper': upper,
        'kc_lower': lower
    }, index=close.index)


def calculate_historical_volatility(series: pd.Series,
                                    periods: List[int] = [20, 60]) -> pd.DataFrame:
    """
    计算历史波动率 (Historical Volatility)

    公式:
        log_return = ln(close / close_prev)
        HV = std(log_return, n) * sqrt(252)  # 年化

    Args:
        series: 价格序列
        periods: 波动率计算周期列表

    Returns:
        DataFrame with HV columns
    """
    log_return = np.log(series / series.shift(1))

    result = pd.DataFrame(index=series.index)
    for period in periods:
        hv = log_return.rolling(window=period).std() * np.sqrt(252)
        result[f'hv_{period}'] = hv

    return result


# =============================================================================
# 成交量指标
# =============================================================================

def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    计算 OBV (On-Balance Volume) 能量潮

    公式:
        if close > close_prev: OBV = OBV_prev + volume
        if close < close_prev: OBV = OBV_prev - volume
        if close == close_prev: OBV = OBV_prev

    用途: 通过成交量变化判断资金流向

    Args:
        close: 收盘价序列
        volume: 成交量序列

    Returns:
        OBV Series
    """
    price_change = close.diff()
    obv_direction = np.sign(price_change)
    obv_direction.iloc[0] = 0

    obv = (obv_direction * volume).cumsum()
    obv.name = 'obv'
    return obv


def calculate_volume_ma(volume: pd.Series,
                        periods: List[int] = [5, 20]) -> pd.DataFrame:
    """
    计算成交量移动平均

    Args:
        volume: 成交量序列
        periods: 均线周期列表

    Returns:
        DataFrame with Volume MA columns
    """
    result = pd.DataFrame(index=volume.index)
    for period in periods:
        result[f'vol_ma_{period}'] = volume.rolling(window=period).mean()
    return result


def calculate_volume_ratio(volume: pd.Series, close: pd.Series,
                          period: int = 5) -> pd.Series:
    """
    计算量比 (Volume Ratio)

    公式: VR = 当日成交量 / 过去n日平均成交量

    也计算: 上涨日成交量 / 下跌日成交量

    Args:
        volume: 成交量序列
        close: 收盘价序列
        period: 计算周期

    Returns:
        Volume Ratio Series
    """
    avg_volume = volume.rolling(window=period).mean()
    vr = volume / (avg_volume + 1e-10)
    vr.name = 'volume_ratio'
    return vr


def calculate_vwap_approx(high: pd.Series, low: pd.Series, close: pd.Series,
                          volume: pd.Series, period: int = 20) -> pd.Series:
    """
    计算日线近似 VWAP (Volume Weighted Average Price)

    真正的 VWAP 需要分钟数据，这里用日线近似:
    公式: VWAP ≈ sum(TP * volume) / sum(volume)
    其中 TP = (high + low + close) / 3

    使用滚动窗口计算

    Args:
        high, low, close, volume: OHLC+V 数据
        period: 滚动窗口期

    Returns:
        VWAP approximation Series
    """
    tp = (high + low + close) / 3
    tp_volume = tp * volume

    vwap = tp_volume.rolling(window=period).sum() / (volume.rolling(window=period).sum() + 1e-10)
    vwap.name = 'vwap'
    return vwap


# =============================================================================
# 价格结构特征
# =============================================================================

def calculate_price_position(close: pd.Series,
                             ma_periods: List[int] = [5, 10, 20, 60]) -> pd.DataFrame:
    """
    计算价格相对均线的位置

    公式: position = (close / MA - 1) * 100

    正值表示在均线上方，负值表示在均线下方

    Args:
        close: 收盘价序列
        ma_periods: 均线周期列表

    Returns:
        DataFrame with position features
    """
    result = pd.DataFrame(index=close.index)
    for period in ma_periods:
        ma = close.rolling(window=period).mean()
        result[f'pos_ma_{period}'] = (close / ma - 1) * 100
    return result


def calculate_distance_from_extremes(high: pd.Series, low: pd.Series,
                                     close: pd.Series,
                                     periods: List[int] = [5, 20, 60]) -> pd.DataFrame:
    """
    计算价格离高点/低点的距离

    公式:
        dist_from_high = (highest - close) / highest * 100
        dist_from_low = (close - lowest) / lowest * 100

    Args:
        high, low, close: OHLC 数据
        periods: 回溯周期列表

    Returns:
        DataFrame with distance features
    """
    result = pd.DataFrame(index=close.index)
    for period in periods:
        highest = high.rolling(window=period).max()
        lowest = low.rolling(window=period).min()

        result[f'dist_high_{period}'] = (highest - close) / (highest + 1e-10) * 100
        result[f'dist_low_{period}'] = (close - lowest) / (lowest + 1e-10) * 100

    return result


def calculate_candle_patterns(open_: pd.Series, high: pd.Series,
                              low: pd.Series, close: pd.Series) -> pd.DataFrame:
    """
    计算K线形态特征

    特征:
        - body_ratio: 实体占比 = |close - open| / (high - low)
        - upper_shadow: 上影线比例 = (high - max(open, close)) / (high - low)
        - lower_shadow: 下影线比例 = (min(open, close) - low) / (high - low)
        - is_bullish: 是否阳线 (close > open)

    Args:
        open_, high, low, close: OHLC 数据

    Returns:
        DataFrame with candle pattern features
    """
    body = abs(close - open_)
    range_ = high - low + 1e-10

    body_ratio = body / range_
    upper_shadow = (high - pd.concat([open_, close], axis=1).max(axis=1)) / range_
    lower_shadow = (pd.concat([open_, close], axis=1).min(axis=1) - low) / range_
    is_bullish = (close > open_).astype(int)

    return pd.DataFrame({
        'body_ratio': body_ratio,
        'upper_shadow': upper_shadow,
        'lower_shadow': lower_shadow,
        'is_bullish': is_bullish
    }, index=close.index)


def detect_gaps(open_: pd.Series, high: pd.Series, low: pd.Series,
                close: pd.Series) -> pd.DataFrame:
    """
    检测价格缺口

    定义:
        - 向上缺口 (Gap Up): low > prev_high
        - 向下缺口 (Gap Down): high < prev_low
        - 缺口大小: 缺口边界差占前收盘价的百分比

    Args:
        open_, high, low, close: OHLC 数据

    Returns:
        DataFrame with gap features
    """
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    gap_up = (low > prev_high).astype(int)
    gap_down = (high < prev_low).astype(int)

    # 缺口大小 (百分比)
    gap_up_size = np.where(gap_up, (low - prev_high) / prev_close * 100, 0)
    gap_down_size = np.where(gap_down, (prev_low - high) / prev_close * 100, 0)

    return pd.DataFrame({
        'gap_up': gap_up,
        'gap_down': gap_down,
        'gap_up_size': gap_up_size,
        'gap_down_size': gap_down_size
    }, index=close.index)


# =============================================================================
# 综合特征计算
# =============================================================================

def calculate_all_features(df: pd.DataFrame,
                          price_col: str = 'close',
                          use_adj: bool = True) -> pd.DataFrame:
    """
    计算所有技术指标特征

    Args:
        df: 原始数据 DataFrame，需包含 open, high, low, close, vol, adj_factor
        price_col: 价格列名
        use_adj: 是否使用复权价格

    Returns:
        包含所有特征的 DataFrame
    """
    # 准备数据
    if use_adj and 'adj_factor' in df.columns:
        adj = df['adj_factor'] / df['adj_factor'].iloc[-1]  # 后复权
        open_ = df['open'] * adj
        high = df['high'] * adj
        low = df['low'] * adj
        close = df['close'] * adj
    else:
        open_ = df['open']
        high = df['high']
        low = df['low']
        close = df['close']

    volume = df['vol'] if 'vol' in df.columns else df.get('volume', pd.Series(0, index=df.index))

    # 创建结果 DataFrame
    result = df[['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'vol']].copy()

    # ===== 趋势类指标 =====
    # MA
    ma_df = calculate_ma(close, [5, 10, 20, 60, 120, 250])
    result = pd.concat([result, ma_df], axis=1)

    # EMA
    ema_df = calculate_ema(close, [12, 26])
    result = pd.concat([result, ema_df], axis=1)

    # MACD
    macd_df = calculate_macd(close, 12, 26, 9)
    result = pd.concat([result, macd_df], axis=1)

    # ADX
    adx_df = calculate_adx(high, low, close, 14)
    result = pd.concat([result, adx_df], axis=1)

    # Aroon
    aroon_df = calculate_aroon(high, low, 25)
    result = pd.concat([result, aroon_df], axis=1)

    # ===== 动量类指标 =====
    # RSI
    rsi_df = calculate_rsi(close, [6, 14, 24])
    result = pd.concat([result, rsi_df], axis=1)

    # Stochastic
    stoch_df = calculate_stochastic(high, low, close, 14, 3, 3)
    result = pd.concat([result, stoch_df], axis=1)

    # Williams %R
    result['williams_r'] = calculate_williams_r(high, low, close, 14)

    # ROC
    result['roc'] = calculate_roc(close, 12)

    # CCI
    result['cci'] = calculate_cci(high, low, close, 20)

    # MFI
    result['mfi'] = calculate_mfi(high, low, close, volume, 14)

    # ===== 波动率指标 =====
    # Bollinger Bands
    bb_df = calculate_bollinger_bands(close, 20, 2)
    result = pd.concat([result, bb_df], axis=1)

    # ATR
    result['atr'] = calculate_atr(high, low, close, 14)

    # Keltner Channel
    kc_df = calculate_keltner_channel(high, low, close, 20, 10, 2)
    result = pd.concat([result, kc_df], axis=1)

    # Historical Volatility
    hv_df = calculate_historical_volatility(close, [20, 60])
    result = pd.concat([result, hv_df], axis=1)

    # ===== 成交量指标 =====
    # OBV
    result['obv'] = calculate_obv(close, volume)

    # Volume MA
    vol_ma_df = calculate_volume_ma(volume, [5, 20])
    result = pd.concat([result, vol_ma_df], axis=1)

    # Volume Ratio
    result['volume_ratio'] = calculate_volume_ratio(volume, close, 5)

    # VWAP (近似)
    result['vwap'] = calculate_vwap_approx(high, low, close, volume, 20)

    # ===== 价格结构特征 =====
    # 价格相对位置
    pos_df = calculate_price_position(close, [5, 10, 20, 60])
    result = pd.concat([result, pos_df], axis=1)

    # 离高低点距离
    dist_df = calculate_distance_from_extremes(high, low, close, [5, 20, 60])
    result = pd.concat([result, dist_df], axis=1)

    # K线形态
    candle_df = calculate_candle_patterns(open_, high, low, close)
    result = pd.concat([result, candle_df], axis=1)

    # 缺口检测
    gap_df = detect_gaps(open_, high, low, close)
    result = pd.concat([result, gap_df], axis=1)

    return result


def get_feature_list() -> Dict[str, List[str]]:
    """
    获取所有特征的分类列表

    Returns:
        特征分类字典
    """
    return {
        '趋势类': [
            'ma_5', 'ma_10', 'ma_20', 'ma_60', 'ma_120', 'ma_250',
            'ema_12', 'ema_26',
            'macd', 'macd_signal', 'macd_hist',
            'adx', 'plus_di', 'minus_di',
            'aroon_up', 'aroon_down', 'aroon_osc'
        ],
        '动量类': [
            'rsi_6', 'rsi_14', 'rsi_24',
            'stoch_k', 'stoch_d',
            'williams_r',
            'roc',
            'cci',
            'mfi'
        ],
        '波动率': [
            'bb_middle', 'bb_upper', 'bb_lower', 'bb_percent_b', 'bb_bandwidth',
            'atr',
            'kc_middle', 'kc_upper', 'kc_lower',
            'hv_20', 'hv_60'
        ],
        '成交量': [
            'obv',
            'vol_ma_5', 'vol_ma_20',
            'volume_ratio',
            'vwap'
        ],
        '价格结构': [
            'pos_ma_5', 'pos_ma_10', 'pos_ma_20', 'pos_ma_60',
            'dist_high_5', 'dist_high_20', 'dist_high_60',
            'dist_low_5', 'dist_low_20', 'dist_low_60',
            'body_ratio', 'upper_shadow', 'lower_shadow', 'is_bullish',
            'gap_up', 'gap_down', 'gap_up_size', 'gap_down_size'
        ]
    }


if __name__ == '__main__':
    # 测试代码
    import duckdb

    conn = duckdb.connect('/Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db', read_only=True)

    # 获取测试数据
    df = conn.execute("""
        SELECT d.*, a.adj_factor
        FROM daily d
        LEFT JOIN adj_factor a ON d.ts_code = a.ts_code AND d.trade_date = a.trade_date
        WHERE d.ts_code = '000001.SZ'
        ORDER BY d.trade_date
    """).fetchdf()

    print(f"测试数据: {len(df)} 行")

    # 计算所有特征
    features = calculate_all_features(df)

    print(f"\n生成特征: {len(features.columns)} 列")
    print("\n特征列表:")
    for col in features.columns:
        print(f"  - {col}")

    print("\n特征统计:")
    print(features.describe())
