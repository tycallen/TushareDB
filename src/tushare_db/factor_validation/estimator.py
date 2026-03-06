"""
参数估计器 - 从真实A股数据提取统计特征

用于蒙特卡洛因子质检系统的参数估计，包括:
- 对数收益率计算
- 年化参数估计 (mu, sigma)
- 实际信号概率计算
"""

from datetime import datetime, timedelta
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from .factor import Factor


class ParameterEstimator:
    """
    参数估计器，从真实A股数据中提取统计特征。

    用于为GBM模拟器提供参数:
    - mu: 年化预期收益率
    - sigma: 年化波动率
    - p_actual: 实际信号触发概率
    """

    def __init__(self, reader, window: int = 252):
        """
        初始化参数估计器。

        Parameters
        ----------
        reader : DataReader
            数据读取器实例，用于从DuckDB读取历史数据
        window : int, default 252
            滚动窗口大小（交易日），默认252个交易日（约1年）
        """
        self.reader = reader
        self.window = window

    def _calculate_log_returns(self, prices: pd.Series) -> pd.Series:
        """
        计算对数收益率。

        公式: R_t = ln(S_t / S_{t-1})

        Parameters
        ----------
        prices : pd.Series
            价格序列

        Returns
        -------
        pd.Series
            对数收益率序列，第一个值为NaN
        """
        return np.log(prices / prices.shift(1))

    def _annualize_parameters(
        self, daily_mu: float, daily_sigma: float, trading_days: int = 252
    ) -> tuple[float, float]:
        """
        将日度参数年化。

        公式:
        - 年化收益率: annual_mu = daily_mu * 252
        - 年化波动率: annual_sigma = daily_sigma * sqrt(252)

        Parameters
        ----------
        daily_mu : float
            日度平均收益率
        daily_sigma : float
            日度收益率标准差
        trading_days : int, default 252
            年化交易日数量

        Returns
        -------
        tuple[float, float]
            (annual_mu, annual_sigma) 年化收益率和年化波动率
        """
        annual_mu = daily_mu * trading_days
        annual_sigma = daily_sigma * np.sqrt(trading_days)
        return annual_mu, annual_sigma

    def calculate_p_actual(self, df: pd.DataFrame, factor: Factor) -> float:
        """
        计算实际信号触发概率。

        Parameters
        ----------
        df : pd.DataFrame
            包含OHLCV数据的DataFrame
        factor : Factor
            因子定义对象

        Returns
        -------
        float
            信号触发概率 (0-1之间)
        """
        signals = factor.evaluate(df)
        if len(signals) == 0:
            return 0.0
        return signals.sum() / len(signals)

    def estimate_parameters(
        self,
        ts_codes: List[str],
        lookback_days: int,
        factor: Factor,
        end_date: Optional[Union[str, datetime]] = None,
    ) -> pd.DataFrame:
        """
        估计参数 (mu, sigma, p_actual) 用于蒙特卡洛模拟。

        Parameters
        ----------
        ts_codes : List[str]
            股票代码列表
        lookback_days : int
            回看天数（用于数据获取）
        factor : Factor
            因子定义对象
        end_date : str or datetime, optional
            结束日期，默认为今天

        Returns
        -------
        pd.DataFrame
            包含以下列的DataFrame:
            - ts_code: 股票代码
            - mu: 年化预期收益率
            - sigma: 年化波动率
            - p_actual: 实际信号触发概率
        """
        if end_date is None:
            end_date = datetime.now()
        elif isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y%m%d")

        # 计算开始日期（多取一些数据用于滚动计算）
        total_days_needed = lookback_days + self.window
        start_date = end_date - timedelta(days=int(total_days_needed * 1.5))  # 1.5倍缓冲
        start_date_str = start_date.strftime("%Y%m%d")

        results = []

        for ts_code in ts_codes:
            try:
                # 获取历史数据
                df = self.reader.get_stock_daily(
                    ts_code=ts_code,
                    start_date=start_date_str,
                    end_date=end_date.strftime("%Y%m%d") if isinstance(end_date, datetime) else end_date,
                )

                if df.empty or len(df) < self.window:
                    # 数据不足，跳过该股票
                    continue

                # 确保数据按日期排序
                if "trade_date" in df.columns:
                    df = df.sort_values("trade_date")

                # 计算对数收益率
                log_returns = self._calculate_log_returns(df["close"])

                # 移除NaN值
                log_returns = log_returns.dropna()

                if len(log_returns) < self.window:
                    continue

                # 取最近lookback_days的数据用于计算参数
                recent_returns = log_returns.iloc[-self.window:]

                # 计算日度参数
                daily_mu = recent_returns.mean()
                daily_sigma = recent_returns.std()

                # 年化参数
                annual_mu, annual_sigma = self._annualize_parameters(daily_mu, daily_sigma)

                # 计算实际信号概率
                recent_df = df.iloc[-self.window:]
                p_actual = self.calculate_p_actual(recent_df, factor)

                results.append({
                    "ts_code": ts_code,
                    "mu": annual_mu,
                    "sigma": annual_sigma,
                    "p_actual": p_actual,
                })

            except Exception:
                # 处理异常，跳过该股票
                continue

        return pd.DataFrame(results)
