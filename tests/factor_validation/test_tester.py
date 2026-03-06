import numpy as np
import pytest


def test_tester_import():
    """测试模块可以导入"""
    from src.tushare_db.factor_validation.tester import SignificanceTester, TestResult
    assert SignificanceTester is not None
    assert TestResult is not None


def test_test_result_dataclass():
    """测试 TestResult dataclass 字段"""
    from src.tushare_db.factor_validation.tester import TestResult

    result = TestResult(
        ts_code="000001.SZ",
        p_actual=0.15,
        p_random=0.10,
        alpha_ratio=1.5,
        n_signals_actual=15,
        n_signals_random=10.0,
        p_value=0.03,
        is_significant=True,
        recommendation="KEEP"
    )

    assert result.ts_code == "000001.SZ"
    assert result.p_actual == 0.15
    assert result.p_random == 0.10
    assert result.alpha_ratio == 1.5
    assert result.n_signals_actual == 15
    assert result.n_signals_random == 10.0
    assert result.p_value == 0.03
    assert result.is_significant is True
    assert result.recommendation == "KEEP"


def test_significance_tester_init():
    """测试 SignificanceTester 初始化"""
    from src.tushare_db.factor_validation.tester import SignificanceTester

    tester = SignificanceTester()
    assert tester.alpha_threshold == 1.5
    assert tester.confidence_level == 0.95

    tester2 = SignificanceTester(alpha_threshold=2.0, confidence_level=0.99)
    assert tester2.alpha_threshold == 2.0
    assert tester2.confidence_level == 0.99


def test_generate_recommendation():
    """测试推荐生成逻辑"""
    from src.tushare_db.factor_validation.tester import SignificanceTester

    tester = SignificanceTester()

    # alpha_ratio < 1.2 -> DISCARD
    assert tester._generate_recommendation(1.1) == "DISCARD"
    assert tester._generate_recommendation(1.19) == "DISCARD"

    # 1.2 <= alpha_ratio < 1.5 -> OPTIMIZE
    assert tester._generate_recommendation(1.2) == "OPTIMIZE"
    assert tester._generate_recommendation(1.3) == "OPTIMIZE"
    assert tester._generate_recommendation(1.49) == "OPTIMIZE"

    # alpha_ratio >= 1.5 -> KEEP
    assert tester._generate_recommendation(1.5) == "KEEP"
    assert tester._generate_recommendation(2.0) == "KEEP"


def test_test_method():
    """测试假设检验方法"""
    from src.tushare_db.factor_validation.tester import SignificanceTester

    tester = SignificanceTester(alpha_threshold=1.5, confidence_level=0.95)

    # Test case: p_actual > p_random, should be significant
    # n_total=100, p_actual=0.15, p_random=0.10
    # n_signals_actual = 15, n_signals_random = 10
    # alpha_ratio = 1.5
    result = tester.test(
        p_actual=0.15,
        p_random=0.10,
        n_total=100,
        ts_code="000001.SZ"
    )

    assert result.ts_code == "000001.SZ"
    assert result.p_actual == 0.15
    assert result.p_random == 0.10
    assert abs(result.alpha_ratio - 1.5) < 1e-10
    assert result.n_signals_actual == 15
    assert abs(result.n_signals_random - 10.0) < 1e-10
    assert isinstance(result.p_value, float)
    assert isinstance(result.is_significant, bool)
    assert result.recommendation in ["KEEP", "OPTIMIZE", "DISCARD"]


def test_test_method_not_significant():
    """测试不显著的情况"""
    from src.tushare_db.factor_validation.tester import SignificanceTester

    tester = SignificanceTester(alpha_threshold=1.5, confidence_level=0.95)

    # Test case: p_actual <= p_random, should not be significant
    # n_total=100, p_actual=0.08, p_random=0.10
    result = tester.test(
        p_actual=0.08,
        p_random=0.10,
        n_total=100,
        ts_code="000002.SZ"
    )

    assert result.ts_code == "000002.SZ"
    assert result.p_actual == 0.08
    assert result.p_random == 0.10
    assert abs(result.alpha_ratio - 0.8) < 1e-10
    assert result.n_signals_actual == 8
    assert result.recommendation == "DISCARD"


def test_generate_report():
    """测试报告生成"""
    from src.tushare_db.factor_validation.tester import SignificanceTester, TestResult

    tester = SignificanceTester()

    results = [
        TestResult(
            ts_code="000001.SZ",
            p_actual=0.20,
            p_random=0.10,
            alpha_ratio=2.0,
            n_signals_actual=20,
            n_signals_random=10.0,
            p_value=0.001,
            is_significant=True,
            recommendation="KEEP"
        ),
        TestResult(
            ts_code="000002.SZ",
            p_actual=0.12,
            p_random=0.10,
            alpha_ratio=1.2,
            n_signals_actual=12,
            n_signals_random=10.0,
            p_value=0.20,
            is_significant=False,
            recommendation="OPTIMIZE"
        ),
    ]

    report = tester.generate_report(factor="macd_golden_cross", results=results)

    # Check report is a string and contains expected content
    assert isinstance(report, str)
    assert "macd_golden_cross" in report
    assert "000001.SZ" in report
    assert "000002.SZ" in report
    assert "KEEP" in report
    assert "OPTIMIZE" in report
    assert "Alpha Ratio" in report or "alpha_ratio" in report
