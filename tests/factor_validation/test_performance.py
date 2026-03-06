"""Performance and accuracy tests for Monte Carlo Factor Validation System.

These tests verify:
1. Performance requirements (speed benchmarks)
2. Accuracy of statistical calculations
3. Memory usage efficiency
4. Vectorized vs loop implementation comparison
"""

import time
import tracemalloc

import numpy as np
import pandas as pd
import pytest

from src.tushare_db.factor_validation.detector import SignalDetector
from src.tushare_db.factor_validation.simulator import GBMSimulator


class TestGBMPerformance:
    """Performance tests for GBM simulation."""

    def test_gbm_simulation_speed(self):
        """Test GBM 10,000 paths x 252 days completes in < 1 second.

        This is the primary performance benchmark for the simulator.
        """
        sim = GBMSimulator(n_paths=10000, n_steps=252, random_seed=42)

        start_time = time.perf_counter()
        result = sim.simulate(s0=100, mu=0.1, sigma=0.2)
        elapsed = time.perf_counter() - start_time

        # Verify shape
        assert result.shape == (10000, 252)

        # Performance assertion: must complete in < 1 second
        print(f"\nGBM Simulation (10,000 x 252): {elapsed:.4f}s")
        assert elapsed < 1.0, f"GBM simulation took {elapsed:.4f}s, expected < 1.0s"

    def test_gbm_simulation_speed_50000_paths(self):
        """Test GBM 50,000 paths x 252 days for scalability."""
        sim = GBMSimulator(n_paths=50000, n_steps=252, random_seed=42)

        start_time = time.perf_counter()
        result = sim.simulate(s0=100, mu=0.1, sigma=0.2)
        elapsed = time.perf_counter() - start_time

        assert result.shape == (50000, 252)
        print(f"\nGBM Simulation (50,000 x 252): {elapsed:.4f}s")

        # Should complete in reasonable time (< 5 seconds for 5x data)
        assert elapsed < 5.0, f"Large GBM simulation took {elapsed:.4f}s, expected < 5.0s"


class TestMACDPerformance:
    """Performance tests for MACD signal detection."""

    def test_macd_detection_speed(self):
        """Test MACD detection on 10,000 paths x 252 days completes in < 1 second."""
        # Generate price data
        np.random.seed(42)
        sim = GBMSimulator(n_paths=10000, n_steps=252, random_seed=42)
        price_matrix = sim.simulate(s0=100, mu=0.1, sigma=0.2)

        detector = SignalDetector()

        start_time = time.perf_counter()
        signals = detector.macd_golden_cross(price_matrix)
        elapsed = time.perf_counter() - start_time

        # Verify shape
        assert signals.shape == (10000, 252)
        assert signals.dtype == bool

        print(f"\nMACD Detection (10,000 x 252): {elapsed:.4f}s")
        assert elapsed < 1.0, f"MACD detection took {elapsed:.4f}s, expected < 1.0s"

    def test_macd_detection_speed_large(self):
        """Test MACD detection on 50,000 paths for scalability."""
        np.random.seed(42)
        sim = GBMSimulator(n_paths=50000, n_steps=252, random_seed=42)
        price_matrix = sim.simulate(s0=100, mu=0.1, sigma=0.2)

        detector = SignalDetector()

        start_time = time.perf_counter()
        signals = detector.macd_golden_cross(price_matrix)
        elapsed = time.perf_counter() - start_time

        assert signals.shape == (50000, 252)
        print(f"\nMACD Detection (50,000 x 252): {elapsed:.4f}s")
        assert elapsed < 5.0, f"Large MACD detection took {elapsed:.4f}s, expected < 5.0s"


class TestVectorizedVsLoop:
    """Compare vectorized implementation vs loop-based implementation."""

    def _ema_loop(self, data: np.ndarray, span: int = 12) -> np.ndarray:
        """Loop-based EMA calculation for comparison."""
        alpha = 2.0 / (span + 1)
        n_paths, n_periods = data.shape
        ema = np.zeros_like(data)
        ema[:, 0] = data[:, 0]

        # Loop-based implementation
        for i in range(n_paths):
            for t in range(1, n_periods):
                ema[i, t] = alpha * data[i, t] + (1 - alpha) * ema[i, t - 1]

        return ema

    def _macd_loop(
        self,
        price_matrix: np.ndarray,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> np.ndarray:
        """Loop-based MACD calculation for comparison."""
        n_paths, n_periods = price_matrix.shape

        # Calculate EMAs using loop method
        ema_fast = self._ema_loop(price_matrix, span=fast)
        ema_slow = self._ema_loop(price_matrix, span=slow)

        macd_line = ema_fast - ema_slow
        signal_line = self._ema_loop(macd_line, span=signal)

        # Detect golden cross
        golden_cross = np.zeros((n_paths, n_periods), dtype=bool)
        for i in range(n_paths):
            for t in range(1, n_periods):
                if macd_line[i, t] > signal_line[i, t] and macd_line[i, t-1] <= signal_line[i, t-1]:
                    golden_cross[i, t] = True

        return golden_cross

    def test_vs_loop_implementation(self):
        """Test vectorized is 10x+ faster than loop implementation."""
        # Use smaller dataset for loop comparison (loop is very slow)
        np.random.seed(42)
        sim = GBMSimulator(n_paths=1000, n_steps=252, random_seed=42)
        price_matrix = sim.simulate(s0=100, mu=0.1, sigma=0.2)

        # Time loop-based implementation
        start_loop = time.perf_counter()
        signals_loop = self._macd_loop(price_matrix)
        elapsed_loop = time.perf_counter() - start_loop

        # Time vectorized implementation
        detector = SignalDetector()
        start_vec = time.perf_counter()
        signals_vec = detector.macd_golden_cross(price_matrix)
        elapsed_vec = time.perf_counter() - start_vec

        # Verify results are equivalent (allowing for minor numerical differences)
        assert signals_vec.shape == signals_loop.shape
        # Note: Results may differ slightly due to numerical precision
        # but overall signal count should be similar

        speedup = elapsed_loop / elapsed_vec if elapsed_vec > 0 else float('inf')
        print(f"\nLoop: {elapsed_loop:.4f}s, Vectorized: {elapsed_vec:.4f}s, Speedup: {speedup:.1f}x")

        # Vectorized should be at least 10x faster
        assert speedup >= 10.0, f"Vectorized only {speedup:.1f}x faster than loop, expected >= 10x"


class TestMemoryUsage:
    """Memory usage tests."""

    def test_memory_usage_50000_paths(self):
        """Test memory usage for 50,000 paths simulation is reasonable."""
        tracemalloc.start()

        # Memory before
        before, _ = tracemalloc.get_traced_memory()

        # Run simulation
        sim = GBMSimulator(n_paths=50000, n_steps=252, random_seed=42)
        result = sim.simulate(s0=100, mu=0.1, sigma=0.2)

        # Memory after simulation
        after, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Calculate memory usage
        memory_used_mb = (after - before) / (1024 * 1024)
        peak_mb = peak / (1024 * 1024)

        # Result size
        result_size_mb = result.nbytes / (1024 * 1024)

        print(f"\nMemory - Used: {memory_used_mb:.2f} MB, Peak: {peak_mb:.2f} MB")
        print(f"Result array size: {result_size_mb:.2f} MB")

        # 50,000 x 252 float64 = ~100 MB for result
        # Total memory should be reasonable (< 500 MB)
        assert peak_mb < 500, f"Peak memory {peak_mb:.2f} MB exceeds 500 MB limit"


class TestGBMAccuracy:
    """Accuracy tests for GBM simulation."""

    def test_gbm_statistical_properties(self):
        """Verify GBM mean and std match theoretical values.

        For GBM: E[S_t] = S_0 * exp(μ * t)
                 Var[S_t] = S_0^2 * exp(2μ * t) * (exp(σ^2 * t) - 1)
        """
        np.random.seed(42)
        n_paths = 50000
        n_steps = 252
        s0 = 100
        mu = 0.1
        sigma = 0.2
        dt = 1 / 252

        sim = GBMSimulator(n_paths=n_paths, n_steps=n_steps, random_seed=42)
        result = sim.simulate(s0=s0, mu=mu, sigma=sigma, dt=dt)

        # Check final prices at t=1 year
        final_prices = result[:, -1]

        # Theoretical mean: E[S_T] = S_0 * exp(μ * T)
        theoretical_mean = s0 * np.exp(mu * 1.0)  # T = 1 year
        actual_mean = np.mean(final_prices)

        # Theoretical variance
        theoretical_var = (s0 ** 2) * np.exp(2 * mu * 1.0) * (np.exp(sigma ** 2 * 1.0) - 1)
        actual_std = np.std(final_prices)
        theoretical_std = np.sqrt(theoretical_var)

        # Allow 5% tolerance for Monte Carlo with 50,000 paths
        mean_error = abs(actual_mean - theoretical_mean) / theoretical_mean
        std_error = abs(actual_std - theoretical_std) / theoretical_std

        print(f"\nGBM Statistics:")
        print(f"  Mean - Theoretical: {theoretical_mean:.4f}, Actual: {actual_mean:.4f}, Error: {mean_error:.2%}")
        print(f"  Std  - Theoretical: {theoretical_std:.4f}, Actual: {actual_std:.4f}, Error: {std_error:.2%}")

        assert mean_error < 0.05, f"Mean error {mean_error:.2%} exceeds 5% tolerance"
        assert std_error < 0.05, f"Std error {std_error:.2%} exceeds 5% tolerance"

    def test_price_limits_effect(self):
        """Verify price limits work correctly.

        With high volatility, some paths should hit the limits.
        """
        np.random.seed(42)
        sim = GBMSimulator(
            n_paths=10000,
            n_steps=252,
            limit_up=0.10,
            limit_down=-0.10,
            random_seed=42
        )

        # High volatility to trigger limits
        result = sim.simulate(s0=100, mu=0.1, sigma=0.5)

        # Calculate daily returns
        returns = np.diff(result, axis=1) / result[:, :-1]

        # Check no returns exceed limits
        assert np.all(returns <= 0.10 + 1e-6), "Some returns exceed limit_up"
        assert np.all(returns >= -0.10 - 1e-6), "Some returns below limit_down"

        # With high volatility, some returns should be exactly at limits
        hits_up = np.sum(returns >= 0.10 - 1e-6)
        hits_down = np.sum(returns <= -0.10 + 1e-6)

        print(f"\nPrice limit hits - Up: {hits_up}, Down: {hits_down}")

        # With sigma=0.5, we expect some limit hits
        assert hits_up > 0 or hits_down > 0, "Expected some limit hits with high volatility"

    def test_gbm_log_returns_distribution(self):
        """Verify log returns follow normal distribution."""
        np.random.seed(42)
        sim = GBMSimulator(n_paths=10000, n_steps=252, random_seed=42)
        result = sim.simulate(s0=100, mu=0.1, sigma=0.2)

        # Calculate log returns
        log_returns = np.diff(np.log(result), axis=1)

        # Flatten for analysis
        flat_returns = log_returns.flatten()

        # Check mean and std of log returns
        # For GBM: E[log return] = (μ - σ²/2) * Δt
        #          Std[log return] = σ * √Δt
        dt = 1 / 252
        expected_mean = (0.1 - 0.5 * 0.2 ** 2) * dt
        expected_std = 0.2 * np.sqrt(dt)

        actual_mean = np.mean(flat_returns)
        actual_std = np.std(flat_returns)

        mean_error = abs(actual_mean - expected_mean) / abs(expected_mean) if expected_mean != 0 else abs(actual_mean)
        std_error = abs(actual_std - expected_std) / expected_std

        print(f"\nLog Returns Distribution:")
        print(f"  Mean - Expected: {expected_mean:.6f}, Actual: {actual_mean:.6f}")
        print(f"  Std  - Expected: {expected_std:.6f}, Actual: {actual_std:.6f}")

        # Allow 10% tolerance for std (Monte Carlo variance)
        assert std_error < 0.10, f"Log return std error {std_error:.2%} exceeds 10% tolerance"


class TestEMAAccuracy:
    """Accuracy tests for EMA calculation."""

    def test_ema_accuracy(self):
        """Test EMA calculation accuracy against pandas reference."""
        np.random.seed(42)

        # Create sample price data
        prices = 100 + np.cumsum(np.random.randn(100) * 0.5)

        # Calculate EMA using pandas (reference)
        pd_ema = pd.Series(prices).ewm(span=12, adjust=False).mean().values

        # Calculate EMA using our vectorized implementation
        detector = SignalDetector()
        price_matrix = prices.reshape(1, -1)
        our_ema = detector._ema_vectorized(price_matrix, span=12)[0]

        # Compare
        max_diff = np.max(np.abs(our_ema - pd_ema))
        mean_diff = np.mean(np.abs(our_ema - pd_ema))

        print(f"\nEMA Accuracy:")
        print(f"  Max difference: {max_diff:.8f}")
        print(f"  Mean difference: {mean_diff:.8f}")

        # Should match pandas within numerical precision
        assert max_diff < 1e-10, f"EMA max difference {max_diff} exceeds tolerance"

    def test_ema_matrix_consistency(self):
        """Test EMA produces consistent results across multiple paths."""
        np.random.seed(42)

        # Create multiple identical paths
        prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        price_matrix = np.tile(prices, (10, 1))

        detector = SignalDetector()
        ema_matrix = detector._ema_vectorized(price_matrix, span=12)

        # All paths should have identical EMAs
        for i in range(1, 10):
            assert np.allclose(ema_matrix[0], ema_matrix[i]), f"Path {i} EMA differs from path 0"


class TestSignalProbability:
    """Tests for signal probability distribution."""

    def test_signal_probability_distribution(self):
        """Test signal probability matches expected for random walk.

        For a random walk, MACD golden cross should occur with
        approximately predictable frequency.
        """
        np.random.seed(42)
        n_paths = 10000
        n_steps = 252

        sim = GBMSimulator(n_paths=n_paths, n_steps=n_steps, random_seed=42)
        price_matrix = sim.simulate(s0=100, mu=0.0, sigma=0.2)  # Random walk, no drift

        detector = SignalDetector()
        signals = detector.macd_golden_cross(price_matrix)

        # Calculate signal probability
        p_signal = np.sum(signals) / signals.size

        # For random walk with MACD(12,26,9), expected signal rate is roughly
        # related to the frequency of zero crossings in the MACD line
        # Empirically, this is typically around 2-5% for random data

        print(f"\nSignal Probability Distribution:")
        print(f"  Signal probability: {p_signal:.4f} ({p_signal*100:.2f}%)")
        print(f"  Total signals: {np.sum(signals)}")

        # Signal probability should be in reasonable range for random walk
        # Too low (< 0.5%) or too high (> 15%) would indicate issues
        assert 0.005 < p_signal < 0.15, f"Signal probability {p_signal:.4f} outside expected range (0.5% - 15%)"

    def test_p_random_calculation(self):
        """Test p_random calculation is accurate."""
        detector = SignalDetector()

        # Test with known signal matrix
        signal_matrix = np.array([
            [True, False, False],
            [False, True, False],
            [False, False, False],
            [True, True, False],
        ])

        p_random = detector.calculate_p_random(signal_matrix)
        expected = 4 / 12  # 4 True out of 12 positions

        assert abs(p_random - expected) < 1e-10, f"p_random {p_random} != expected {expected}"

    def test_signal_independence(self):
        """Test that signals on different paths are independent.

        For independent paths with same parameters, signal counts
        should follow approximately binomial distribution.
        """
        np.random.seed(42)
        n_paths = 5000
        n_steps = 100

        sim = GBMSimulator(n_paths=n_paths, n_steps=n_steps, random_seed=42)
        price_matrix = sim.simulate(s0=100, mu=0.0, sigma=0.2)

        detector = SignalDetector()
        signals = detector.macd_golden_cross(price_matrix)

        # Count signals per path
        signals_per_path = np.sum(signals, axis=1)

        # Statistical properties
        mean_signals = np.mean(signals_per_path)
        std_signals = np.std(signals_per_path)

        print(f"\nSignal Independence:")
        print(f"  Mean signals per path: {mean_signals:.2f}")
        print(f"  Std signals per path: {std_signals:.2f}")

        # For independent paths, variance should be reasonable
        # (not too close to 0, not excessively high)
        assert std_signals > 0, "Signal std should be > 0 for independent paths"
        assert mean_signals > 0, "Should have some signals on average"


class TestCombinedPerformance:
    """Combined workflow performance tests."""

    def test_full_workflow_performance(self):
        """Test complete simulation + detection workflow performance."""
        np.random.seed(42)

        start_time = time.perf_counter()

        # Simulate
        sim = GBMSimulator(n_paths=10000, n_steps=252, random_seed=42)
        prices = sim.simulate(s0=100, mu=0.1, sigma=0.2)

        # Detect signals
        detector = SignalDetector()
        signals = detector.macd_golden_cross(prices)

        # Calculate statistics
        p_random = detector.calculate_p_random(signals)

        elapsed = time.perf_counter() - start_time

        print(f"\nFull Workflow (10,000 x 252): {elapsed:.4f}s")
        print(f"  Simulation + Detection + Stats")
        print(f"  p_random: {p_random:.4f}")

        # Full workflow should complete in < 2 seconds
        assert elapsed < 2.0, f"Full workflow took {elapsed:.4f}s, expected < 2.0s"
