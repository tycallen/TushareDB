"""
板块关系分析器
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional
from scipy.stats import pearsonr, linregress
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
        period: str = 'daily',
        min_correlation: float = 0.0
    ) -> pd.DataFrame:
        """
        计算传导关系（滞后相关性）

        Args:
            start_date: 开始日期
            end_date: 结束日期
            max_lag: 最大滞后期（None表示自适应）
            level: 层级
            period: 周期
            min_correlation: 最小相关系数阈值（绝对值）

        Returns:
            DataFrame with columns: [sector_lead, sector_lag, lag_days, correlation, p_value]
        """
        # 自适应窗口
        if max_lag is None:
            max_lag = MAX_LAG_MAP.get(period, 5)

        logger.info(f"计算传导关系: {start_date} ~ {end_date}, max_lag={max_lag}, level={level}, period={period}")

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

        # 3. 计算每对板块的滞后相关性
        sectors = pivot_df.columns.tolist()
        results = []

        from tqdm import tqdm
        for i, sector_lead in enumerate(tqdm(sectors, desc=f"计算传导关系")):
            for j, sector_lag in enumerate(sectors):
                if i == j:  # 跳过自己
                    continue

                # 提取时间序列
                series_lead = pivot_df[sector_lead].dropna()
                series_lag = pivot_df[sector_lag].dropna()

                # 计算不同滞后期的相关性
                for lag in range(0, max_lag + 1):
                    if lag == 0:
                        # 同期相关性（跳过，已在correlation_matrix中计算）
                        continue

                    # 对齐时间序列（lead在前，lag在后）
                    if len(series_lead) <= lag or len(series_lag) <= lag:
                        continue

                    lead_values = series_lead.iloc[:-lag].values
                    lag_values = series_lag.iloc[lag:].values

                    # 确保长度一致
                    min_len = min(len(lead_values), len(lag_values))
                    if min_len < 10:  # 至少需要10个样本
                        continue

                    lead_values = lead_values[:min_len]
                    lag_values = lag_values[:min_len]

                    # 计算相关性和p值
                    try:
                        corr, p_value = pearsonr(lead_values, lag_values)

                        # 过滤低相关性
                        if abs(corr) >= min_correlation:
                            results.append({
                                'sector_lead': sector_lead,
                                'sector_lag': sector_lag,
                                'lag_days': lag,
                                'correlation': corr,
                                'p_value': p_value,
                                'sample_size': min_len
                            })
                    except Exception as e:
                        logger.warning(f"计算失败: {sector_lead} -> {sector_lag} (lag={lag}): {e}")
                        continue

        result_df = pd.DataFrame(results)
        if len(result_df) > 0:
            result_df = result_df.sort_values('correlation', ascending=False, key=abs)

        logger.info(f"找到 {len(result_df)} 对传导关系（|r| >= {min_correlation}）")
        return result_df

    def calculate_linkage_strength(
        self,
        start_date: str,
        end_date: str,
        level: str = 'L1',
        period: str = 'daily',
        min_r_squared: float = 0.0
    ) -> pd.DataFrame:
        """
        计算联动强度（Beta系数）

        Args:
            start_date: 开始日期
            end_date: 结束日期
            level: 层级
            period: 周期
            min_r_squared: 最小R²阈值

        Returns:
            DataFrame with columns: [sector_a, sector_b, beta, r_squared, p_value]
            说明：sector_a涨1%时，sector_b平均涨beta%
        """
        logger.info(f"计算联动强度: {start_date} ~ {end_date}, level={level}, period={period}")

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

        # 3. 计算每对板块的Beta系数
        sectors = pivot_df.columns.tolist()
        results = []

        from tqdm import tqdm
        for i, sector_a in enumerate(tqdm(sectors, desc="计算联动强度")):
            for j, sector_b in enumerate(sectors):
                if i >= j:  # 只计算上三角（避免重复和自己）
                    continue

                # 提取两个板块的涨跌幅序列（去除NaN）
                data_a = pivot_df[sector_a].dropna()
                data_b = pivot_df[sector_b].dropna()

                # 找到共同的日期
                common_dates = data_a.index.intersection(data_b.index)
                if len(common_dates) < 10:  # 至少需要10个样本
                    continue

                series_a = pivot_df.loc[common_dates, sector_a].values
                series_b = pivot_df.loc[common_dates, sector_b].values

                # 线性回归：sector_b = alpha + beta * sector_a
                try:
                    slope, intercept, r_value, p_value, std_err = linregress(
                        series_a, series_b
                    )

                    r_squared = r_value ** 2

                    # 过滤低R²
                    if r_squared >= min_r_squared:
                        results.append({
                            'sector_a': sector_a,
                            'sector_b': sector_b,
                            'beta': slope,
                            'alpha': intercept,
                            'r_squared': r_squared,
                            'p_value': p_value,
                            'std_err': std_err,
                            'sample_size': len(common_dates)
                        })

                    # 反向回归：sector_a = alpha + beta * sector_b
                    slope_rev, intercept_rev, r_value_rev, p_value_rev, std_err_rev = linregress(
                        series_b, series_a
                    )

                    r_squared_rev = r_value_rev ** 2

                    if r_squared_rev >= min_r_squared:
                        results.append({
                            'sector_a': sector_b,
                            'sector_b': sector_a,
                            'beta': slope_rev,
                            'alpha': intercept_rev,
                            'r_squared': r_squared_rev,
                            'p_value': p_value_rev,
                            'std_err': std_err_rev,
                            'sample_size': len(common_dates)
                        })

                except Exception as e:
                    logger.warning(f"计算失败: {sector_a} - {sector_b}: {e}")
                    continue

        result_df = pd.DataFrame(results)
        if len(result_df) > 0:
            result_df = result_df.sort_values('r_squared', ascending=False)

        logger.info(f"找到 {len(result_df)} 对联动关系（R² >= {min_r_squared}）")
        return result_df

    def close(self):
        """关闭数据库连接"""
        self.reader.close()
