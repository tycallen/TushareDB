"""Statistical significance testing and report generation for Monte Carlo Factor Validation System."""

from dataclasses import dataclass
from typing import List

from scipy import stats


@dataclass
class TestResult:
    """Result of significance testing for a single stock.

    Attributes:
        ts_code: Stock code (e.g., "000001.SZ")
        p_actual: Actual signal probability from real market data
        p_random: Random signal probability from Monte Carlo simulation
        alpha_ratio: Ratio of p_actual to p_random (p_actual / p_random)
        n_signals_actual: Number of actual signals observed
        n_signals_random: Expected number of random signals
        p_value: P-value from binomial test
        is_significant: Whether the result is statistically significant
        recommendation: KEEP, OPTIMIZE, or DISCARD
    """
    ts_code: str
    p_actual: float
    p_random: float
    alpha_ratio: float
    n_signals_actual: int
    n_signals_random: float
    p_value: float
    is_significant: bool
    recommendation: str


class SignificanceTester:
    """Statistical significance tester for factor validation.

    Performs hypothesis testing comparing actual signal probabilities
    against random walk simulations to determine if a factor has
    alpha generation capability.

    Attributes:
        alpha_threshold: Minimum alpha ratio for KEEP recommendation (default 1.5)
        confidence_level: Confidence level for statistical test (default 0.95)
    """

    def __init__(self, alpha_threshold: float = 1.5, confidence_level: float = 0.95):
        """Initialize the SignificanceTester.

        Args:
            alpha_threshold: Minimum alpha ratio for KEEP recommendation
            confidence_level: Confidence level for statistical significance
        """
        self.alpha_threshold = alpha_threshold
        self.confidence_level = confidence_level

    def test(
        self,
        p_actual: float,
        p_random: float,
        n_total: int,
        ts_code: str
    ) -> TestResult:
        """Perform hypothesis testing on signal probabilities.

        Uses binomial test to determine if the actual signal count
        is significantly higher than expected from random walk.

        Args:
            p_actual: Actual signal probability from real market data
            p_random: Random signal probability from Monte Carlo simulation
            n_total: Total number of observations (days)
            ts_code: Stock code

        Returns:
            TestResult with statistical test results and recommendation
        """
        # Calculate alpha ratio
        alpha_ratio = p_actual / p_random if p_random > 0 else float('inf')

        # Calculate signal counts
        n_signals_actual = int(round(p_actual * n_total))
        n_signals_random = p_random * n_total

        # Perform binomial test
        # H0: p_actual <= p_random (no alpha)
        # H1: p_actual > p_random (has alpha)
        # p_value = P(X >= n_signals_actual | n_total, p_random)
        if n_signals_actual > 0 and p_random > 0:
            p_value = 1 - stats.binom.cdf(n_signals_actual - 1, n_total, p_random)
        elif n_signals_actual == 0:
            p_value = 1.0
        else:
            p_value = 0.0

        # Determine significance
        alpha = 1 - self.confidence_level
        is_significant = bool(p_value < alpha and alpha_ratio >= self.alpha_threshold)

        # Generate recommendation
        recommendation = self._generate_recommendation(alpha_ratio)

        return TestResult(
            ts_code=ts_code,
            p_actual=p_actual,
            p_random=p_random,
            alpha_ratio=alpha_ratio,
            n_signals_actual=n_signals_actual,
            n_signals_random=n_signals_random,
            p_value=p_value,
            is_significant=is_significant,
            recommendation=recommendation
        )

    def _generate_recommendation(self, alpha_ratio: float) -> str:
        """Generate recommendation based on alpha ratio.

        Args:
            alpha_ratio: Ratio of actual to random probability

        Returns:
            "KEEP" if alpha_ratio >= 1.5
            "OPTIMIZE" if 1.2 <= alpha_ratio < 1.5
            "DISCARD" if alpha_ratio < 1.2
        """
        if alpha_ratio < 1.2:
            return "DISCARD"
        elif alpha_ratio < 1.5:
            return "OPTIMIZE"
        else:
            return "KEEP"

    def generate_report(self, factor: str, results: List[TestResult]) -> str:
        """Generate a Markdown report of test results.

        Args:
            factor: Factor name (e.g., "macd_golden_cross")
            results: List of TestResult objects

        Returns:
            Markdown formatted report string
        """
        lines = []
        lines.append(f"# Factor Validation Report: {factor}")
        lines.append("")
        lines.append(f"**Alpha Threshold**: {self.alpha_threshold}")
        lines.append(f"**Confidence Level**: {self.confidence_level}")
        lines.append("")

        # Summary statistics
        total_stocks = len(results)
        keep_count = sum(1 for r in results if r.recommendation == "KEEP")
        optimize_count = sum(1 for r in results if r.recommendation == "OPTIMIZE")
        discard_count = sum(1 for r in results if r.recommendation == "DISCARD")
        significant_count = sum(1 for r in results if r.is_significant)

        lines.append("## Summary")
        lines.append("")
        lines.append(f"- **Total Stocks**: {total_stocks}")
        lines.append(f"- **Significant Results**: {significant_count}")
        lines.append(f"- **KEEP**: {keep_count}")
        lines.append(f"- **OPTIMIZE**: {optimize_count}")
        lines.append(f"- **DISCARD**: {discard_count}")
        lines.append("")

        # Detailed results table
        lines.append("## Detailed Results")
        lines.append("")
        lines.append("| Stock | P(Actual) | P(Random) | Alpha Ratio | N(Actual) | N(Random) | P-Value | Significant | Recommendation |")
        lines.append("|-------|-----------|-----------|-------------|-----------|-----------|---------|-------------|----------------|")

        for r in results:
            sig_marker = "Yes" if r.is_significant else "No"
            lines.append(
                f"| {r.ts_code} | {r.p_actual:.4f} | {r.p_random:.4f} | "
                f"{r.alpha_ratio:.2f} | {r.n_signals_actual} | {r.n_signals_random:.1f} | "
                f"{r.p_value:.4f} | {sig_marker} | {r.recommendation} |"
            )

        lines.append("")

        # Recommendations section
        lines.append("## Recommendations")
        lines.append("")

        keep_results = [r for r in results if r.recommendation == "KEEP"]
        if keep_results:
            lines.append("### KEEP (Strong Alpha)")
            lines.append("")
            for r in keep_results:
                lines.append(f"- **{r.ts_code}**: alpha_ratio={r.alpha_ratio:.2f}, p_value={r.p_value:.4f}")
            lines.append("")

        optimize_results = [r for r in results if r.recommendation == "OPTIMIZE"]
        if optimize_results:
            lines.append("### OPTIMIZE (Weak Alpha)")
            lines.append("")
            for r in optimize_results:
                lines.append(f"- **{r.ts_code}**: alpha_ratio={r.alpha_ratio:.2f}, p_value={r.p_value:.4f}")
            lines.append("")

        discard_results = [r for r in results if r.recommendation == "DISCARD"]
        if discard_results:
            lines.append("### DISCARD (No Alpha)")
            lines.append("")
            for r in discard_results:
                lines.append(f"- **{r.ts_code}**: alpha_ratio={r.alpha_ratio:.2f}, p_value={r.p_value:.4f}")
            lines.append("")

        return "\n".join(lines)
