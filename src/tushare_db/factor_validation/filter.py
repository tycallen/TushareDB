"""Main controller module for Monte Carlo Factor Validation System.

Integrates all 4 stages of the validation pipeline:
- Stage 1: ParameterEstimator - Extract statistical features from real A-share data
- Stage 2: GBMSimulator - Monte Carlo simulation with price limits
- Stage 3: SignalDetector - Detect signals on simulated paths
- Stage 4: SignificanceTester - Statistical hypothesis testing
"""

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .detector import SignalDetector
from .estimator import ParameterEstimator
from .factor import Factor, FactorRegistry, FactorType
from .simulator import GBMSimulator
from .tester import SignificanceTester, TestResult


@dataclass
class FactorReport:
    """Report generated from factor validation.

    Attributes:
        factor: Factor name
        timestamp: Report generation timestamp
        parameters: Parameters used for validation
        results: List of TestResult objects
        summary: Summary statistics dictionary
        markdown: Markdown formatted report string
        dataframe: Results as a DataFrame
    """

    factor: str
    timestamp: datetime
    parameters: Dict[str, Any]
    results: List[TestResult]
    summary: Dict[str, Any] = field(default_factory=dict)
    markdown: str = ""
    dataframe: Optional[pd.DataFrame] = None

    def __post_init__(self):
        """Initialize derived fields if not provided."""
        if not self.summary:
            self._generate_summary()
        if not self.markdown and self.results:
            self._generate_markdown()
        if self.dataframe is None:
            self._generate_dataframe()

    def _generate_summary(self) -> None:
        """Generate summary statistics from results."""
        total = len(self.results)
        if total == 0:
            self.summary = {"total_stocks": 0}
            return

        keep_count = sum(1 for r in self.results if r.recommendation == "KEEP")
        optimize_count = sum(1 for r in self.results if r.recommendation == "OPTIMIZE")
        discard_count = sum(1 for r in self.results if r.recommendation == "DISCARD")
        significant_count = sum(1 for r in self.results if r.is_significant)

        avg_alpha = np.mean([r.alpha_ratio for r in self.results])
        avg_p_value = np.mean([r.p_value for r in self.results])

        self.summary = {
            "total_stocks": total,
            "significant_count": significant_count,
            "keep_count": keep_count,
            "optimize_count": optimize_count,
            "discard_count": discard_count,
            "keep_rate": keep_count / total if total > 0 else 0,
            "significant_rate": significant_count / total if total > 0 else 0,
            "avg_alpha_ratio": avg_alpha,
            "avg_p_value": avg_p_value,
        }

    def _generate_markdown(self) -> None:
        """Generate markdown report from results."""
        tester = SignificanceTester()
        self.markdown = tester.generate_report(self.factor, self.results)

    def _generate_dataframe(self) -> None:
        """Generate DataFrame from results."""
        if not self.results:
            self.dataframe = pd.DataFrame()
            return

        data = []
        for r in self.results:
            data.append({
                "ts_code": r.ts_code,
                "p_actual": r.p_actual,
                "p_random": r.p_random,
                "alpha_ratio": r.alpha_ratio,
                "n_signals_actual": r.n_signals_actual,
                "n_signals_random": r.n_signals_random,
                "p_value": r.p_value,
                "is_significant": r.is_significant,
                "recommendation": r.recommendation,
            })
        self.dataframe = pd.DataFrame(data)

    def save(self, path: str) -> None:
        """Save report to disk.

        Saves both markdown (.md) and CSV (.csv) files.

        Args:
            path: Base path for output files (without extension)
        """
        # Ensure directory exists
        dir_path = os.path.dirname(path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

        # Save markdown
        md_path = f"{path}.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(self.markdown)

        # Save CSV
        csv_path = f"{path}.csv"
        if self.dataframe is not None and not self.dataframe.empty:
            self.dataframe.to_csv(csv_path, index=False)
        else:
            # Create empty CSV with headers
            pd.DataFrame(
                columns=[
                    "ts_code",
                    "p_actual",
                    "p_random",
                    "alpha_ratio",
                    "n_signals_actual",
                    "n_signals_random",
                    "p_value",
                    "is_significant",
                    "recommendation",
                ]
            ).to_csv(csv_path, index=False)


class FactorFilter:
    """Main controller for Monte Carlo Factor Validation System.

    Orchestrates the 4-stage validation pipeline:
    1. Parameter estimation from real market data
    2. GBM simulation with A-share price limits
    3. Signal detection on simulated paths
    4. Statistical significance testing

    Attributes:
        db_path: Path to DuckDB database
        n_simulations: Number of Monte Carlo simulation paths
        simulation_days: Number of days to simulate per path
        alpha_threshold: Minimum alpha ratio for KEEP recommendation
        window: Rolling window for parameter estimation
    """

    def __init__(
        self,
        db_path: str,
        n_simulations: int = 1000,
        simulation_days: int = 252,
        alpha_threshold: float = 1.5,
        window: int = 252,
    ):
        """Initialize the FactorFilter.

        Args:
            db_path: Path to DuckDB database
            n_simulations: Number of Monte Carlo simulation paths (default 1000)
            simulation_days: Number of days to simulate per path (default 252)
            alpha_threshold: Minimum alpha ratio for KEEP (default 1.5)
            window: Rolling window for parameter estimation (default 252)
        """
        self.db_path = db_path
        self.n_simulations = n_simulations
        self.simulation_days = simulation_days
        self.alpha_threshold = alpha_threshold
        self.window = window

        # Initialize components (lazy initialization for reader)
        self._reader = None
        self._estimator = None
        self._detector = SignalDetector()
        self._tester = SignificanceTester(alpha_threshold=alpha_threshold)

    def _get_reader(self):
        """Lazy initialization of DataReader."""
        if self._reader is None:
            from ..reader import DataReader

            self._reader = DataReader(db_path=self.db_path)
        return self._reader

    def _get_estimator(self):
        """Lazy initialization of ParameterEstimator."""
        if self._estimator is None:
            self._estimator = ParameterEstimator(
                reader=self._get_reader(), window=self.window
            )
        return self._estimator

    def filter(
        self,
        factor: Union[Factor, str],
        ts_codes: List[str],
        lookback_days: int = 252,
        use_sample: bool = False,
    ) -> FactorReport:
        """Run full validation workflow for a factor.

        Args:
            factor: Factor object or factor name string
            ts_codes: List of stock codes to validate
            lookback_days: Days of historical data to use
            use_sample: If True, use subset of stocks for faster testing

        Returns:
            FactorReport with validation results
        """
        # Resolve factor if string
        if isinstance(factor, str):
            factor = FactorRegistry.get(factor)

        # Use sample if requested
        if use_sample and len(ts_codes) > 5:
            ts_codes = ts_codes[:5]

        # Stage 1: Parameter Estimation
        params_df = self._get_estimator().estimate_parameters(
            ts_codes=ts_codes, lookback_days=lookback_days, factor=factor
        )

        if params_df.empty:
            return FactorReport(
                factor=factor.name,
                timestamp=datetime.now(),
                parameters={
                    "db_path": self.db_path,
                    "n_simulations": self.n_simulations,
                    "simulation_days": self.simulation_days,
                    "alpha_threshold": self.alpha_threshold,
                    "window": self.window,
                    "lookback_days": lookback_days,
                },
                results=[],
            )

        # Stages 2-4: Simulation, Detection, Testing
        results = self._run_simulation_test(params_df, factor)

        return FactorReport(
            factor=factor.name,
            timestamp=datetime.now(),
            parameters={
                "db_path": self.db_path,
                "n_simulations": self.n_simulations,
                "simulation_days": self.simulation_days,
                "alpha_threshold": self.alpha_threshold,
                "window": self.window,
                "lookback_days": lookback_days,
            },
            results=results,
        )

    def _run_simulation_test(
        self, params_df: pd.DataFrame, factor: Factor
    ) -> List[TestResult]:
        """Run stages 2-4: Simulation, Detection, and Testing.

        Args:
            params_df: DataFrame with columns [ts_code, mu, sigma, p_actual]
            factor: Factor to test

        Returns:
            List of TestResult objects
        """
        results = []

        # Initialize simulator
        simulator = GBMSimulator(
            n_paths=self.n_simulations, n_steps=self.simulation_days
        )

        for _, row in params_df.iterrows():
            ts_code = row["ts_code"]
            mu = row["mu"]
            sigma = row["sigma"]
            p_actual = row["p_actual"]

            # Stage 2: Simulation
            # Use s0=100 as normalized initial price
            price_matrix = simulator.simulate(s0=100.0, mu=mu, sigma=sigma)

            # Stage 3: Signal Detection
            signal_matrix = self._detector.detect_signals(price_matrix, factor)

            # Calculate p_random
            p_random = self._detector.calculate_p_random(signal_matrix)

            # Stage 4: Significance Testing
            n_total = self.simulation_days
            result = self._tester.test(
                p_actual=p_actual,
                p_random=p_random,
                n_total=n_total,
                ts_code=ts_code,
            )
            results.append(result)

        return results

    def batch_filter(
        self,
        factors: List[Union[Factor, str]],
        ts_codes: List[str],
        lookback_days: int = 252,
        use_sample: bool = False,
    ) -> Dict[str, FactorReport]:
        """Run validation for multiple factors.

        Args:
            factors: List of Factor objects or factor name strings
            ts_codes: List of stock codes to validate
            lookback_days: Days of historical data to use
            use_sample: If True, use subset of stocks for faster testing

        Returns:
            Dictionary mapping factor names to FactorReport objects
        """
        reports = {}
        for factor in factors:
            report = self.filter(
                factor=factor,
                ts_codes=ts_codes,
                lookback_days=lookback_days,
                use_sample=use_sample,
            )
            reports[report.factor] = report
        return reports

    def benchmark_all_builtin(
        self,
        ts_codes: List[str],
        lookback_days: int = 252,
        use_sample: bool = False,
    ) -> Dict[str, FactorReport]:
        """Validate all built-in factors.

        Args:
            ts_codes: List of stock codes to validate
            lookback_days: Days of historical data to use
            use_sample: If True, use subset of stocks for faster testing

        Returns:
            Dictionary mapping factor names to FactorReport objects
        """
        builtin_names = FactorRegistry.list_builtin()
        if not builtin_names:
            return {}

        return self.batch_filter(
            factors=builtin_names,
            ts_codes=ts_codes,
            lookback_days=lookback_days,
            use_sample=use_sample,
        )

    def close(self) -> None:
        """Close database connections."""
        if self._reader is not None:
            self._reader.close()
            self._reader = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
