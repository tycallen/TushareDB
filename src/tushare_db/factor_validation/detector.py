"""Signal detector module with vectorized indicators for Monte Carlo Factor Validation System."""

from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

from src.tushare_db.factor_validation.factor import Factor, FactorRegistry, FactorType


class SignalDetector:
    """Signal detector with vectorized indicator calculations.

    This class provides efficient signal detection using NumPy vectorized
    operations on price matrices for Monte Carlo simulations.

    Attributes:
        vectorized: Always True, indicates vectorized operations are used
    """

    def __init__(self):
        """Initialize the SignalDetector."""
        self.vectorized = True
        self._register_builtin_factors()

    def _register_builtin_factors(self) -> None:
        """Register built-in factors with the FactorRegistry."""
        # Register close_gt_open factor if not already registered
        if "close_gt_open" not in FactorRegistry.list_builtin():
            def close_gt_open_func(df: pd.DataFrame) -> pd.Series:
                return df['close'] > df['open']

            factor = Factor(
                name="close_gt_open",
                description="收盘价大于开盘价",
                definition=close_gt_open_func,
                type=FactorType.FUNCTION
            )
            FactorRegistry.register_builtin(factor)

    def _ema_vectorized(self, data: np.ndarray, span: int = 12) -> np.ndarray:
        """Calculate EMA for a matrix using vectorized operations.

        Args:
            data: Input matrix (n_paths, n_periods)
            span: EMA span (default 12 for fast EMA)

        Returns:
            EMA matrix with same shape as input
        """
        alpha = 2.0 / (span + 1)
        n_paths, n_periods = data.shape

        # Initialize EMA array
        ema = np.zeros_like(data)

        # First value is just the first data point
        ema[:, 0] = data[:, 0]

        # Vectorized EMA calculation along axis 1 (time)
        for t in range(1, n_periods):
            ema[:, t] = alpha * data[:, t] + (1 - alpha) * ema[:, t - 1]

        return ema

    def macd_golden_cross(
        self,
        price_matrix: np.ndarray,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> np.ndarray:
        """Detect MACD golden cross signals using vectorized operations.

        A golden cross occurs when MACD line crosses above the signal line.

        Args:
            price_matrix: Price matrix (n_paths, n_periods)
            fast: Fast EMA period (default 12)
            slow: Slow EMA period (default 26)
            signal: Signal EMA period (default 9)

        Returns:
            Boolean matrix indicating golden cross signals
        """
        n_paths, n_periods = price_matrix.shape

        # Calculate fast and slow EMAs
        ema_fast = self._ema_vectorized(price_matrix, span=fast)
        ema_slow = self._ema_vectorized(price_matrix, span=slow)

        # MACD line
        macd_line = ema_fast - ema_slow

        # Signal line (EMA of MACD)
        signal_line = self._ema_vectorized(macd_line, span=signal)

        # Detect golden cross: MACD crosses above signal line
        # Current: MACD > Signal, Previous: MACD <= Signal
        macd_gt_signal = macd_line > signal_line

        # Shift to get previous state
        prev_macd_gt_signal = np.zeros_like(macd_gt_signal)
        prev_macd_gt_signal[:, 1:] = macd_gt_signal[:, :-1]

        # Golden cross: current MACD > signal, previous MACD <= signal
        golden_cross = macd_gt_signal & (~prev_macd_gt_signal)

        return golden_cross

    def calculate_p_random(self, signal_matrix: np.ndarray) -> float:
        """Calculate the random probability of signals.

        This is the proportion of True values in the signal matrix,
        representing the baseline probability of a signal occurring.

        Args:
            signal_matrix: Boolean matrix of signals

        Returns:
            Probability of random signal occurrence
        """
        total_signals = np.sum(signal_matrix)
        total_positions = signal_matrix.size

        if total_positions == 0:
            return 0.0

        return float(total_signals) / float(total_positions)

    def detect_signals(
        self,
        price_matrix: np.ndarray,
        factor: Factor,
        **extra_data: np.ndarray
    ) -> np.ndarray:
        """Detect signals using a factor definition.

        Converts the price matrix to a DataFrame, evaluates the factor,
        and returns the result as a boolean matrix.

        Args:
            price_matrix: Price matrix (n_paths, n_periods)
            factor: Factor definition to evaluate
            **extra_data: Additional price matrices (e.g., open_price, high_price)

        Returns:
            Boolean matrix indicating factor signals
        """
        n_paths, n_periods = price_matrix.shape

        # Create DataFrame with close prices
        # Each row becomes a separate "stock" in the DataFrame
        data_dict: Dict[str, np.ndarray] = {'close': price_matrix.flatten()}

        # Add extra data if provided
        # Map common price column names to standard OHLCV names
        column_mapping = {
            'open_price': 'open',
            'high_price': 'high',
            'low_price': 'low',
            'close_price': 'close',
            'volume_data': 'volume',
        }
        for key, value in extra_data.items():
            if value.shape == price_matrix.shape:
                # Map the key to standard name if applicable
                mapped_key = column_mapping.get(key, key)
                data_dict[mapped_key] = value.flatten()

        # Create DataFrame
        df = pd.DataFrame(data_dict)

        # Evaluate factor
        if factor.type == FactorType.FUNCTION:
            result = factor.evaluate(df)
        elif factor.type == FactorType.BUILTIN:
            # For builtin factors, try to get the definition and evaluate
            if callable(factor.definition):
                result = factor.definition(df)
            else:
                raise NotImplementedError(f"BUILTIN factor {factor.name} without callable definition")
        else:
            raise NotImplementedError(f"Factor type {factor.type} not supported for matrix detection")

        # Convert back to matrix shape
        signal_matrix = result.values.reshape(n_paths, n_periods)

        return signal_matrix
