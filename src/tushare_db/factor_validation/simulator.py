"""
蒙特卡洛模拟器 - 用于因子验证的几何布朗运动模拟

提供带A股涨跌停限制的价格路径模拟。
"""

import numpy as np
from typing import Optional


class GBMSimulator:
    """
    几何布朗运动模拟器，支持A股涨跌停限制。

    GBM公式: S_t = S_{t-1} × exp((μ - σ²/2) × Δt + σ × √Δt × Z)

    其中:
    - S_t: t时刻的价格
    - μ: 年化漂移率
    - σ: 年化波动率
    - Δt: 时间步长（年化）
    - Z: 标准正态随机变量

    A股涨跌停限制:
    - 普通股票: ±10%
    - ST股票: ±5%
    - 创业板/科创板: ±20%
    """

    def __init__(
        self,
        n_paths: int,
        n_steps: int,
        limit_up: float = 0.10,
        limit_down: float = -0.10,
        random_seed: Optional[int] = None,
    ):
        """
        初始化GBM模拟器。

        Parameters
        ----------
        n_paths : int
            模拟路径数量
        n_steps : int
            每路径的时间步数
        limit_up : float, default 0.10
            涨停限制（日收益率上限），默认10%
        limit_down : float, default -0.10
            跌停限制（日收益率下限），默认-10%
        random_seed : int, optional
            随机种子，用于结果可重复
        """
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.limit_up = limit_up
        self.limit_down = limit_down
        self.random_seed = random_seed

    def simulate(
        self,
        s0: float,
        mu: float,
        sigma: float,
        dt: float = 1 / 252,
    ) -> np.ndarray:
        """
        模拟价格路径。

        Parameters
        ----------
        s0 : float
            初始价格
        mu : float
            年化漂移率（预期收益率）
        sigma : float
            年化波动率
        dt : float, default 1/252
            时间步长（年化），默认1个交易日

        Returns
        -------
        np.ndarray
            形状为 (n_paths, n_steps) 的价格矩阵
        """
        # 设置随机种子以确保可重复性
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        # 初始化价格矩阵
        prices = np.zeros((self.n_paths, self.n_steps))
        prices[:, 0] = s0

        # GBM参数
        # drift term: (μ - σ²/2) × Δt
        drift = (mu - 0.5 * sigma**2) * dt
        # diffusion term: σ × √Δt
        diffusion = sigma * np.sqrt(dt)

        # 生成随机增量
        # 形状: (n_paths, n_steps - 1)
        random_shocks = np.random.standard_normal((self.n_paths, self.n_steps - 1))

        # 模拟价格路径
        for t in range(1, self.n_steps):
            # GBM公式: S_t = S_{t-1} × exp(drift + diffusion × Z)
            price_ratio = np.exp(drift + diffusion * random_shocks[:, t - 1])

            # 计算新价格
            new_prices = prices[:, t - 1] * price_ratio

            # 应用涨跌停限制
            # 计算相对于前一日收盘价的收益率
            daily_returns = (new_prices - prices[:, t - 1]) / prices[:, t - 1]

            # 使用clip限制收益率在涨跌停范围内
            clipped_returns = np.clip(
                daily_returns, self.limit_down, self.limit_up
            )

            # 根据限制后的收益率计算实际价格
            prices[:, t] = prices[:, t - 1] * (1 + clipped_returns)

        return prices
