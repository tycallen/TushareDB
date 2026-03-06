"""
增强版因子过滤器 - 集成报告管理功能

在 FactorFilter 基础上增加：
1. 自动保存验证结果
2. 历史记录查询
3. 对比分析
"""
from typing import List, Union, Optional, Dict

from .filter import FactorFilter, FactorReport
from .report_manager import ReportManager, ValidationRecord


class EnhancedFactorFilter(FactorFilter):
    """
    增强版因子过滤器

    在标准 FactorFilter 基础上集成报告管理功能，自动保存每次验证结果
    到本地数据库，支持历史查询和对比分析。

    Example:
        >>> from src.tushare_db.factor_validation import EnhancedFactorFilter
        >>>
        >>> # 创建增强版过滤器
        >>> filter_obj = EnhancedFactorFilter(db_path="tushare.db")
        >>>
        >>> # 运行验证（结果自动保存）
        >>> report = filter_obj.filter(
        ...     factor="macd_golden_cross",
        ...     ts_codes=['000001.SZ'],
        ...     save_report=True,
        ...     notes="测试笔记"
        ... )
        >>>
        >>> # 查询历史
        >>> history = filter_obj.get_factor_history("macd_golden_cross")
        >>>
        >>> # 生成统计报告
        >>> stats = filter_obj.get_statistics(days=30)
    """

    def __init__(
        self,
        db_path: str = "tushare.db",
        n_simulations: int = 10000,
        simulation_days: int = 252,
        alpha_threshold: float = 1.5,
        window: int = 252,
        report_db_path: Optional[str] = None,
        auto_save: bool = True
    ):
        """
        初始化增强版过滤器

        Args:
            db_path: Tushare DuckDB 数据库路径
            n_simulations: 模拟路径数
            simulation_days: 模拟天数
            alpha_threshold: Alpha Ratio 阈值
            window: 滚动窗口
            report_db_path: 报告数据库路径，默认 ~/.factor_validation/reports.db
            auto_save: 是否自动保存每次验证结果
        """
        super().__init__(
            db_path=db_path,
            n_simulations=n_simulations,
            simulation_days=simulation_days,
            alpha_threshold=alpha_threshold,
            window=window
        )

        self.report_manager = ReportManager(db_path=report_db_path)
        self.auto_save = auto_save

    def filter(
        self,
        factor: Union[str, 'Factor'],
        ts_codes: List[str],
        lookback_days: int = 252,
        use_sample: bool = False,
        save_report: Optional[bool] = None,
        notes: str = ""
    ) -> FactorReport:
        """
        运行因子验证（增强版，支持自动保存）

        Args:
            factor: 因子名称或 Factor 对象
            ts_codes: 股票代码列表
            lookback_days: 回看天数
            use_sample: 使用样本数据
            save_report: 是否保存报告（默认使用 auto_save 设置）
            notes: 备注

        Returns:
            FactorReport 对象
        """
        # 运行标准验证
        report = super().filter(
            factor=factor,
            ts_codes=ts_codes,
            lookback_days=lookback_days,
            use_sample=use_sample
        )

        # 保存报告
        should_save = save_report if save_report is not None else self.auto_save
        if should_save and ts_codes:
            for ts_code in ts_codes:
                self.report_manager.save_report(
                    report=report,
                    ts_code=ts_code,
                    stock_name="",  # 可以后续查询补充
                    start_date="",  # 可以从报告中获取
                    end_date="",
                    notes=notes
                )

        return report

    def get_factor_history(self, factor_name: str, ts_code: Optional[str] = None):
        """获取因子的历史验证记录"""
        return self.report_manager.get_factor_history(factor_name, ts_code)

    def get_stock_history(self, ts_code: str):
        """获取股票的所有因子验证记录"""
        return self.report_manager.get_stock_history(ts_code)

    def compare_factor_over_time(self, factor_name: str, ts_code: str):
        """对比同一因子在不同时间的表现"""
        return self.report_manager.compare_factor_over_time(factor_name, ts_code)

    def get_statistics(self, days: int = 30):
        """获取最近N天的统计摘要"""
        return self.report_manager.get_statistics_summary(days)

    def export_reports(self, filepath: str, days: int = 30):
        """导出最近N天的报告到CSV"""
        start_date = (
            __import__('datetime').datetime.now() -
            __import__('datetime').timedelta(days=days)
        ).isoformat()

        return self.report_manager.export_to_csv(
            filepath,
            start_date=start_date
        )

    def generate_report(self, days: int = 30) -> str:
        """生成统计报告"""
        return self.report_manager.generate_report(days)
