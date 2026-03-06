"""Factor Validation Demo - Monte Carlo Factor Validation System.

This demo shows how to use the factor validation system to test
technical indicators and identify factors with alpha potential.

Usage:
    python examples/factor_validation_demo.py

Environment Variables:
    DB_PATH: Path to DuckDB database (default: tushare.db)
    RUN_REAL_DATA: Set to 1 to run demos with real data (default: use mock data)
"""

import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tushare_db.factor_validation import FactorFilter, FactorRegistry
from src.tushare_db.factor_validation.factor import Factor, FactorType


def create_mock_reader():
    """Create a mock DataReader for demo purposes."""
    mock_reader = Mock()

    # Create sample daily price data with realistic patterns
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=300, freq="B")

    # Generate price with trend and volatility
    returns = np.random.randn(300) * 0.02 + 0.0003
    prices = 100 * np.cumprod(1 + returns)

    sample_data = pd.DataFrame({
        "ts_code": ["000001.SZ"] * 300,
        "trade_date": dates,
        "open": prices * (1 + np.random.randn(300) * 0.005),
        "high": prices * (1 + abs(np.random.randn(300)) * 0.02),
        "low": prices * (1 - abs(np.random.randn(300)) * 0.02),
        "close": prices,
        "vol": np.random.randint(10000, 50000, 300),
    })

    mock_reader.get_stock_daily.return_value = sample_data
    return mock_reader


def demo_basic_usage():
    """Demo 1: Basic single factor validation.

    Shows how to:
    - Initialize FactorFilter
    - Validate a single factor
    - Access report results
    """
    print("=" * 60)
    print("Demo 1: Basic Single Factor Validation")
    print("=" * 60)

    mock_reader = create_mock_reader()

    with patch("src.tushare_db.reader.DataReader", return_value=mock_reader):
        # Initialize FactorFilter with in-memory database
        filter_obj = FactorFilter(
            db_path=":memory:",
            n_simulations=1000,      # Number of Monte Carlo paths
            simulation_days=252,     # Trading days to simulate
            alpha_threshold=1.5,     # Minimum alpha ratio for KEEP
            window=252,              # Parameter estimation window
        )

        # Validate a built-in factor
        print("\nValidating 'macd_golden_cross' factor...")
        report = filter_obj.filter(
            factor="macd_golden_cross",
            ts_codes=["000001.SZ"],
            lookback_days=252,
            use_sample=True,         # Use subset for faster demo
        )

        # Display results
        print(f"\nFactor: {report.factor}")
        print(f"Timestamp: {report.timestamp}")
        print(f"\nParameters used:")
        for key, value in report.parameters.items():
            print(f"  {key}: {value}")

        print(f"\nSummary:")
        for key, value in report.summary.items():
            print(f"  {key}: {value}")

        if report.results:
            result = report.results[0]
            print(f"\nDetailed Results for {result.ts_code}:")
            print(f"  P(actual): {result.p_actual:.4f}")
            print(f"  P(random): {result.p_random:.4f}")
            print(f"  Alpha Ratio: {result.alpha_ratio:.4f}")
            print(f"  P-value: {result.p_value:.4f}")
            print(f"  Significant: {result.is_significant}")
            print(f"  Recommendation: {result.recommendation}")

        # Show markdown preview
        print(f"\nMarkdown Report Preview (first 500 chars):")
        print("-" * 40)
        print(report.markdown[:500] + "...")

        filter_obj.close()

    print("\n" + "=" * 60)
    print("Demo 1 Complete!")
    print("=" * 60 + "\n")


def demo_batch_filter():
    """Demo 2: Batch validation of multiple factors.

    Shows how to:
    - Compare multiple factors in one run
    - Analyze which factors have the best alpha
    """
    print("=" * 60)
    print("Demo 2: Batch Factor Validation")
    print("=" * 60)

    mock_reader = create_mock_reader()

    with patch("src.tushare_db.reader.DataReader", return_value=mock_reader):
        filter_obj = FactorFilter(
            db_path=":memory:",
            n_simulations=500,
            simulation_days=126,
            alpha_threshold=1.5,
        )

        # List of factors to compare
        factors_to_test = [
            "macd_golden_cross",
            "golden_cross",
            "rsi_oversold",
            "bollinger_lower_break",
        ]

        print(f"\nValidating {len(factors_to_test)} factors...")
        print("Factors:", ", ".join(factors_to_test))

        # Batch validation
        reports = filter_obj.batch_filter(
            factors=factors_to_test,
            ts_codes=["000001.SZ"],
            lookback_days=252,
            use_sample=True,
        )

        # Compare results
        print("\n" + "-" * 60)
        print("Comparison Results:")
        print("-" * 60)
        print(f"{'Factor':<25} {'Alpha Ratio':>12} {'P-Value':>10} {'Rec':>8}")
        print("-" * 60)

        for factor_name, report in reports.items():
            if report.results:
                result = report.results[0]
                print(f"{factor_name:<25} {result.alpha_ratio:>12.4f} "
                      f"{result.p_value:>10.4f} {result.recommendation:>8}")
            else:
                print(f"{factor_name:<25} {'N/A':>12} {'N/A':>10} {'N/A':>8}")

        # Find best factor
        best_factor = None
        best_alpha = 0
        for name, report in reports.items():
            if report.results and report.results[0].alpha_ratio > best_alpha:
                best_alpha = report.results[0].alpha_ratio
                best_factor = name

        if best_factor:
            print(f"\nBest Factor: {best_factor} (Alpha Ratio: {best_alpha:.4f})")

        filter_obj.close()

    print("\n" + "=" * 60)
    print("Demo 2 Complete!")
    print("=" * 60 + "\n")


def demo_custom_factor():
    """Demo 3: Creating and validating custom factors.

    Shows how to:
    - Define a custom factor function
    - Register it for validation
    - Validate the custom factor
    """
    print("=" * 60)
    print("Demo 3: Custom Factor Creation and Validation")
    print("=" * 60)

    mock_reader = create_mock_reader()

    # Define a custom factor function (using only close price for simulation)
    def price_acceleration(df):
        """
        Detect price acceleration:
        - Price increases at an increasing rate (momentum building)
        """
        # Price change
        change = df["close"] - df["close"].shift(1)
        # Acceleration = change is increasing
        acceleration = change > change.shift(1)
        # Price is going up
        going_up = df["close"] > df["close"].shift(1)
        return acceleration & going_up

    # Create Factor object
    custom_factor = Factor(
        name="price_acceleration",
        description="Price increasing at an accelerating rate - momentum signal",
        definition=price_acceleration,
        type=FactorType.FUNCTION,
        parameters={},
    )

    print(f"\nCustom Factor Created:")
    print(f"  Name: {custom_factor.name}")
    print(f"  Description: {custom_factor.description}")
    print(f"  Type: {custom_factor.type}")

    with patch("src.tushare_db.reader.DataReader", return_value=mock_reader):
        filter_obj = FactorFilter(
            db_path=":memory:",
            n_simulations=500,
            simulation_days=126,
            alpha_threshold=1.5,
        )

        print(f"\nValidating custom factor...")
        report = filter_obj.filter(
            factor=custom_factor,
            ts_codes=["000001.SZ"],
            lookback_days=252,
        )

        print(f"\nValidation Results:")
        print(f"  Factor: {report.factor}")
        if report.results:
            result = report.results[0]
            print(f"  Alpha Ratio: {result.alpha_ratio:.4f}")
            print(f"  P-Value: {result.p_value:.4f}")
            print(f"  Recommendation: {result.recommendation}")

            if result.recommendation == "KEEP":
                print(f"\n  ✓ This factor shows significant alpha potential!")
            elif result.recommendation == "OPTIMIZE":
                print(f"\n  ~ This factor shows some potential but needs optimization.")
            else:
                print(f"\n  ✗ This factor does not show significant alpha.")

        filter_obj.close()

    print("\n" + "=" * 60)
    print("Demo 3 Complete!")
    print("=" * 60 + "\n")


def demo_list_builtin_factors():
    """Demo 4: List all available built-in factors.

    Shows how to:
    - Get list of all built-in factors
    - View factor descriptions and parameters
    """
    print("=" * 60)
    print("Demo 4: List Built-in Factors")
    print("=" * 60)

    # Get all built-in factor names
    builtin_names = FactorRegistry.list_builtin()

    print(f"\nTotal Built-in Factors: {len(builtin_names)}")
    print("\nAvailable Factors:")
    print("-" * 60)

    # Group factors by category
    categories = {
        "MACD": [],
        "RSI": [],
        "Moving Average": [],
        "Bollinger Bands": [],
        "Volume": [],
        "Price Pattern": [],
    }

    for name in builtin_names:
        if "macd" in name.lower():
            categories["MACD"].append(name)
        elif "rsi" in name.lower():
            categories["RSI"].append(name)
        elif any(x in name.lower() for x in ["cross", "sma", "ma_"]):
            categories["Moving Average"].append(name)
        elif "bollinger" in name.lower():
            categories["Bollinger Bands"].append(name)
        elif "volume" in name.lower():
            categories["Volume"].append(name)
        else:
            categories["Price Pattern"].append(name)

    for category, factors in categories.items():
        if factors:
            print(f"\n{category}:")
            for factor in sorted(factors):
                try:
                    factor_obj = FactorRegistry.get(factor)
                    print(f"  • {factor:<30} - {factor_obj.description}")
                except KeyError:
                    print(f"  • {factor:<30}")

    print("\n" + "-" * 60)
    print("\nExample usage:")
    print('  report = filter_obj.filter(factor="macd_golden_cross", ...)')

    print("\n" + "=" * 60)
    print("Demo 4 Complete!")
    print("=" * 60 + "\n")


def demo_save_report():
    """Demo 5: Save validation reports to files.

    Shows how to:
    - Save reports as markdown and CSV
    - Organize reports by timestamp
    """
    print("=" * 60)
    print("Demo 5: Save Validation Reports")
    print("=" * 60)

    mock_reader = create_mock_reader()

    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("src.tushare_db.reader.DataReader", return_value=mock_reader):
            filter_obj = FactorFilter(
                db_path=":memory:",
                n_simulations=500,
                simulation_days=126,
                alpha_threshold=1.5,
            )

            # Validate multiple factors
            factors = ["macd_golden_cross", "rsi_oversold"]
            reports = filter_obj.batch_filter(
                factors=factors,
                ts_codes=["000001.SZ"],
                lookback_days=252,
            )

            # Create timestamped directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_dir = os.path.join(tmpdir, f"factor_reports_{timestamp}")
            os.makedirs(report_dir, exist_ok=True)

            print(f"\nSaving reports to: {report_dir}")

            # Save each report
            for factor_name, report in reports.items():
                # Create safe filename
                safe_name = factor_name.replace(" ", "_").lower()
                report_path = os.path.join(report_dir, safe_name)

                report.save(report_path)

                print(f"\n  Saved '{factor_name}':")
                print(f"    Markdown: {safe_name}.md")
                print(f"    CSV: {safe_name}.csv")

                # Show summary from saved markdown
                md_file = f"{report_path}.md"
                if os.path.exists(md_file):
                    with open(md_file, "r") as f:
                        content = f.read()
                        # Extract recommendation line
                        for line in content.split("\n"):
                            if "Recommendation" in line:
                                print(f"    Result: {line.strip()}")
                                break

            # List all saved files
            print(f"\nAll saved files:")
            for filename in sorted(os.listdir(report_dir)):
                filepath = os.path.join(report_dir, filename)
                size = os.path.getsize(filepath)
                print(f"  {filename} ({size} bytes)")

            filter_obj.close()

    print("\n" + "=" * 60)
    print("Demo 5 Complete!")
    print("=" * 60 + "\n")


def demo_context_manager():
    """Bonus Demo: Using FactorFilter as context manager.

    Shows the recommended pattern for using FactorFilter
    with automatic resource cleanup.
    """
    print("=" * 60)
    print("Bonus Demo: Context Manager Pattern")
    print("=" * 60)

    mock_reader = create_mock_reader()

    with patch("src.tushare_db.reader.DataReader", return_value=mock_reader):
        # Using context manager ensures cleanup
        with FactorFilter(
            db_path=":memory:",
            n_simulations=500,
            simulation_days=126,
        ) as filter_obj:
            report = filter_obj.filter(
                factor="golden_cross",
                ts_codes=["000001.SZ"],
                lookback_days=252,
            )

            print(f"\nFactor validated: {report.factor}")
            if report.results:
                print(f"Alpha Ratio: {report.results[0].alpha_ratio:.4f}")

        # Database connection automatically closed here
        print("\nDatabase connection automatically closed.")

    print("\n" + "=" * 60)
    print("Bonus Demo Complete!")
    print("=" * 60 + "\n")


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("Monte Carlo Factor Validation System - Demo")
    print("=" * 60 + "\n")

    # Check if real data should be used
    use_real_data = os.environ.get("RUN_REAL_DATA", "0") == "1"
    if use_real_data:
        print("Note: RUN_REAL_DATA=1, but demos use mock data for consistency.")
        print("Set up real data tests in test_integration.py for actual validation.\n")

    # Run all demos
    try:
        demo_basic_usage()
        demo_batch_filter()
        demo_custom_factor()
        demo_list_builtin_factors()
        demo_save_report()
        demo_context_manager()
    except Exception as e:
        print(f"\nError running demos: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n" + "=" * 60)
    print("All Demos Completed Successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Review the demo code in examples/factor_validation_demo.py")
    print("  2. Run integration tests: pytest tests/factor_validation/test_integration.py -v")
    print("  3. Try with real data: Set DB_PATH and RUN_REAL_DATA_TESTS=1")
    print("=" * 60 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
