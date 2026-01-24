"""
板块关系分析器
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional
from scipy.stats import pearsonr
from .calculator import SectorCalculator
from .config import MAX_LAG_MAP

logger = logging.getLogger(__name__)


class SectorAnalyzer:
    """板块关系分析器"""

    def __init__(self, db_path: str):
        """
        初始化分析器

        Args:
            db_path: 数据库路径
        """
        from ..reader import DataReader
        self.reader = DataReader(db_path)
        self.calculator = SectorCalculator(self.reader)

    def calculate_sector_returns(
        self,
        start_date: str,
        end_date: str,
        level: str = 'L1',
        period: str = 'daily',
        method: str = 'equal'
    ) -> pd.DataFrame:
        """
        计算板块涨跌幅（委托给Calculator）

        Returns:
            DataFrame with columns: [sector_code, sector_name, trade_date, return, stock_count]
        """
        return self.calculator.calculate_returns(
            start_date, end_date, level, period, method
        )

    def calculate_correlation_matrix(
        self,
        start_date: str,
        end_date: str,
        level: str = 'L1',
        period: str = 'daily',
        method: str = 'pearson'
    ) -> pd.DataFrame:
        """
        计算相关性矩阵

        Args:
            start_date: 开始日期
            end_date: 结束日期
            level: 层级
            period: 周期
            method: 相关性计算方法 (pearson)

        Returns:
            相关系数矩阵（行列都是板块代码）
        """
        logger.info(f"计算相关性矩阵: {start_date} ~ {end_date}, level={level}, period={period}")

        # 1. 获取板块涨跌幅数据
        returns_df = self.calculate_sector_returns(
            start_date, end_date, level, period
        )

        # 2. 转换为透视表（日期×板块）
        pivot_df = returns_df.pivot(
            index='trade_date',
            columns='sector_code',
            values='return'
        )

        # 3. 计算相关系数矩阵
        if method == 'pearson':
            corr_matrix = pivot_df.corr(method='pearson')
        else:
            raise ValueError(f"不支持的方法: {method}")

        logger.info(f"相关性矩阵: {corr_matrix.shape[0]}×{corr_matrix.shape[1]}")
        return corr_matrix

    def calculate_correlation_with_pvalue(
        self,
        start_date: str,
        end_date: str,
        level: str = 'L1',
        period: str = 'daily',
        min_correlation: float = 0.0
    ) -> pd.DataFrame:
        """
        计算相关性矩阵（包含p值）

        Args:
            start_date: 开始日期
            end_date: 结束日期
            level: 层级
            period: 周期
            min_correlation: 最小相关系数阈值（绝对值）

        Returns:
            DataFrame with columns: [sector_a, sector_b, correlation, p_value]
        """
        logger.info("计算相关性矩阵（含p值）...")

        # 1. 获取板块涨跌幅数据
        returns_df = self.calculate_sector_returns(
            start_date, end_date, level, period
        )

        # 2. 转换为透视表
        pivot_df = returns_df.pivot(
            index='trade_date',
            columns='sector_code',
            values='return'
        )

        # 3. 计算每对板块的相关性和p值
        sectors = pivot_df.columns.tolist()
        results = []

        for i, sector_a in enumerate(sectors):
            for j, sector_b in enumerate(sectors):
                if i >= j:  # 只计算上三角（避免重复）
                    continue

                # 提取两个板块的涨跌幅序列（去除NaN）
                data_a = pivot_df[sector_a].dropna()
                data_b = pivot_df[sector_b].dropna()

                # 找到共同的日期
                common_dates = data_a.index.intersection(data_b.index)
                if len(common_dates) < 10:  # 至少需要10个样本
                    continue

                series_a = pivot_df.loc[common_dates, sector_a]
                series_b = pivot_df.loc[common_dates, sector_b]

                # 计算相关性和p值
                corr, p_value = pearsonr(series_a, series_b)

                # 过滤低相关性
                if abs(corr) >= min_correlation:
                    results.append({
                        'sector_a': sector_a,
                        'sector_b': sector_b,
                        'correlation': corr,
                        'p_value': p_value,
                        'sample_size': len(common_dates)
                    })

        result_df = pd.DataFrame(results)
        if len(result_df) > 0:
            result_df = result_df.sort_values('correlation', ascending=False, key=abs)

        logger.info(f"找到 {len(result_df)} 对相关板块（|r| >= {min_correlation}）")
        return result_df

    def calculate_lead_lag(
        self,
        start_date: str,
        end_date: str,
        max_lag: Optional[int] = None,
        level: str = 'L1',
        period: str = 'daily'
    ) -> pd.DataFrame:
        """
        计算传导关系

        Returns:
            DataFrame with columns: [sector_lead, sector_lag, lag_days, correlation, p_value]
        """
        raise NotImplementedError("将在Task 4中实现")

    def calculate_linkage_strength(
        self,
        start_date: str,
        end_date: str,
        level: str = 'L1'
    ) -> pd.DataFrame:
        """
        计算联动强度

        Returns:
            DataFrame with columns: [sector_a, sector_b, beta, r_squared]
        """
        raise NotImplementedError("将在Task 5中实现")

    def close(self):
        """关闭数据库连接"""
        self.reader.close()
