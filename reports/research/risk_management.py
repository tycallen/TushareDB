#!/usr/bin/env python3
"""
风险管理模块

本模块提供多空对冲策略的风险管理功能，包括：
1. 风险指标计算 (VaR, CVaR, 最大回撤等)
2. 风险监控与预警
3. 组合风险分解
4. 压力测试

使用示例:
    from risk_management import RiskManager

    rm = RiskManager()
    risk_metrics = rm.calculate_risk_metrics(returns)
    rm.print_risk_report(risk_metrics)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from scipy import stats


@dataclass
class RiskMetrics:
    """风险指标数据类"""
    # 基本统计
    mean_return: float
    volatility: float
    skewness: float
    kurtosis: float

    # 收益分布
    max_return: float
    min_return: float
    median_return: float

    # 风险指标
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float

    # 回撤分析
    max_drawdown: float
    avg_drawdown: float
    max_drawdown_duration: int

    # 风险调整收益
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # 其他指标
    win_rate: float
    profit_factor: float
    max_consecutive_losses: int


@dataclass
class PositionRisk:
    """持仓风险数据类"""
    position_id: str
    weight: float
    contribution_to_var: float
    marginal_var: float
    beta: float


class RiskManager:
    """风险管理类"""

    def __init__(self, risk_free_rate: float = 0.03):
        """
        初始化风险管理器

        Args:
            risk_free_rate: 无风险利率 (年化)
        """
        self.risk_free_rate = risk_free_rate
        self.daily_rf = risk_free_rate / 252

    def calculate_risk_metrics(
        self,
        returns: Union[pd.Series, np.ndarray],
        annualize: bool = True
    ) -> RiskMetrics:
        """
        计算风险指标

        Args:
            returns: 收益率序列 (百分比形式)
            annualize: 是否年化
        """
        if isinstance(returns, pd.Series):
            returns = returns.dropna().values
        returns = np.array(returns)

        # 基本统计
        mean_return = np.mean(returns)
        volatility = np.std(returns)
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)

        # 收益分布
        max_return = np.max(returns)
        min_return = np.min(returns)
        median_return = np.median(returns)

        # VaR和CVaR
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        cvar_95 = np.mean(returns[returns <= var_95])
        cvar_99 = np.mean(returns[returns <= var_99])

        # 回撤分析
        cumret = np.cumprod(1 + returns / 100)
        running_max = np.maximum.accumulate(cumret)
        drawdown = (cumret - running_max) / running_max * 100
        max_drawdown = np.min(drawdown)
        avg_drawdown = np.mean(drawdown[drawdown < 0]) if np.any(drawdown < 0) else 0

        # 最长回撤期
        in_drawdown = drawdown < 0
        dd_periods = []
        current_period = 0
        for dd in in_drawdown:
            if dd:
                current_period += 1
            else:
                if current_period > 0:
                    dd_periods.append(current_period)
                current_period = 0
        if current_period > 0:
            dd_periods.append(current_period)
        max_drawdown_duration = max(dd_periods) if dd_periods else 0

        # 年化因子
        ann_factor = np.sqrt(252) if annualize else 1
        ann_return_factor = 252 if annualize else 1

        # 夏普比率
        excess_return = mean_return - self.daily_rf * 100
        sharpe_ratio = excess_return / volatility * ann_factor if volatility > 0 else 0

        # Sortino比率 (只考虑下行波动)
        downside_returns = returns[returns < 0]
        downside_vol = np.std(downside_returns) if len(downside_returns) > 0 else volatility
        sortino_ratio = excess_return / downside_vol * ann_factor if downside_vol > 0 else 0

        # Calmar比率
        ann_return = mean_return * ann_return_factor
        calmar_ratio = ann_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # 胜率
        win_rate = np.mean(returns > 0) * 100

        # 盈亏比
        avg_win = np.mean(returns[returns > 0]) if np.any(returns > 0) else 0
        avg_loss = abs(np.mean(returns[returns < 0])) if np.any(returns < 0) else 1
        profit_factor = avg_win / avg_loss if avg_loss > 0 else float('inf')

        # 最长连续亏损
        max_consecutive_losses = 0
        current_losses = 0
        for r in returns:
            if r < 0:
                current_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, current_losses)
            else:
                current_losses = 0

        return RiskMetrics(
            mean_return=mean_return,
            volatility=volatility,
            skewness=skewness,
            kurtosis=kurtosis,
            max_return=max_return,
            min_return=min_return,
            median_return=median_return,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            max_drawdown=max_drawdown,
            avg_drawdown=avg_drawdown,
            max_drawdown_duration=max_drawdown_duration,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            win_rate=win_rate,
            profit_factor=profit_factor,
            max_consecutive_losses=max_consecutive_losses
        )

    def calculate_rolling_risk(
        self,
        returns: pd.Series,
        window: int = 20
    ) -> pd.DataFrame:
        """
        计算滚动风险指标

        Args:
            returns: 收益率序列
            window: 滚动窗口
        """
        rolling = returns.rolling(window=window)

        risk_df = pd.DataFrame({
            'rolling_mean': rolling.mean(),
            'rolling_vol': rolling.std(),
            'rolling_var_95': rolling.quantile(0.05),
            'rolling_skew': rolling.skew()
        })

        # 滚动最大回撤
        def rolling_max_dd(x):
            cumret = (1 + x / 100).cumprod()
            running_max = cumret.expanding().max()
            drawdown = (cumret - running_max) / running_max
            return drawdown.min() * 100

        risk_df['rolling_max_dd'] = returns.rolling(window=window).apply(
            rolling_max_dd, raw=False
        )

        return risk_df

    def calculate_parametric_var(
        self,
        returns: np.ndarray,
        confidence: float = 0.95,
        method: str = 'normal'
    ) -> float:
        """
        计算参数化VaR

        Args:
            returns: 收益率序列
            confidence: 置信水平
            method: 方法 ('normal', 't')
        """
        mean = np.mean(returns)
        std = np.std(returns)

        if method == 'normal':
            z_score = stats.norm.ppf(1 - confidence)
            var = mean + z_score * std
        elif method == 't':
            # 使用t分布，考虑厚尾
            df = 4  # 自由度
            t_score = stats.t.ppf(1 - confidence, df)
            var = mean + t_score * std * np.sqrt((df - 2) / df)
        else:
            raise ValueError(f"Unknown method: {method}")

        return var

    def stress_test(
        self,
        returns: pd.Series,
        scenarios: Dict[str, float]
    ) -> pd.DataFrame:
        """
        压力测试

        Args:
            returns: 收益率序列
            scenarios: 压力情景 {场景名: 冲击倍数}
        """
        results = []
        base_metrics = self.calculate_risk_metrics(returns)

        for name, shock in scenarios.items():
            shocked_returns = returns * shock
            shocked_metrics = self.calculate_risk_metrics(shocked_returns)

            results.append({
                '情景': name,
                '冲击倍数': shock,
                '日均收益%': shocked_metrics.mean_return,
                '波动率%': shocked_metrics.volatility,
                '95% VaR%': shocked_metrics.var_95,
                '最大回撤%': shocked_metrics.max_drawdown,
                '夏普比率': shocked_metrics.sharpe_ratio
            })

        return pd.DataFrame(results)

    def calculate_drawdown_analysis(
        self,
        returns: pd.Series
    ) -> pd.DataFrame:
        """
        详细回撤分析
        """
        cumret = (1 + returns / 100).cumprod()
        running_max = cumret.expanding().max()
        drawdown = (cumret - running_max) / running_max * 100

        # 找出所有回撤期
        in_drawdown = drawdown < 0
        drawdown_periods = []

        start_idx = None
        for i, (idx, dd) in enumerate(zip(drawdown.index, drawdown.values)):
            if dd < 0 and start_idx is None:
                start_idx = idx
            elif dd >= 0 and start_idx is not None:
                drawdown_periods.append({
                    '开始日期': start_idx,
                    '结束日期': idx,
                    '持续天数': i - list(drawdown.index).index(start_idx),
                    '最大回撤%': drawdown.loc[start_idx:idx].min()
                })
                start_idx = None

        # 如果当前仍在回撤中
        if start_idx is not None:
            drawdown_periods.append({
                '开始日期': start_idx,
                '结束日期': drawdown.index[-1],
                '持续天数': len(drawdown) - list(drawdown.index).index(start_idx),
                '最大回撤%': drawdown.loc[start_idx:].min()
            })

        df = pd.DataFrame(drawdown_periods)
        if len(df) > 0:
            df = df.sort_values('最大回撤%').head(10)  # TOP10最大回撤

        return df

    def calculate_risk_decomposition(
        self,
        returns: pd.DataFrame,
        weights: pd.Series
    ) -> pd.DataFrame:
        """
        风险分解

        Args:
            returns: 各资产收益率矩阵 (columns为资产)
            weights: 权重向量
        """
        # 协方差矩阵
        cov_matrix = returns.cov()

        # 组合波动率
        port_var = np.dot(weights.T, np.dot(cov_matrix, weights))
        port_vol = np.sqrt(port_var)

        # 边际风险贡献
        marginal_risk = np.dot(cov_matrix, weights) / port_vol

        # 风险贡献
        risk_contribution = weights * marginal_risk
        risk_contribution_pct = risk_contribution / port_vol * 100

        result = pd.DataFrame({
            '资产': weights.index,
            '权重%': weights.values * 100,
            '边际风险': marginal_risk,
            '风险贡献%': risk_contribution_pct
        })

        return result

    def generate_risk_alerts(
        self,
        returns: pd.Series,
        thresholds: Optional[Dict[str, float]] = None
    ) -> List[str]:
        """
        生成风险预警

        Args:
            returns: 收益率序列
            thresholds: 预警阈值
        """
        if thresholds is None:
            thresholds = {
                'daily_loss': -3.0,       # 日亏损超过3%
                'weekly_loss': -5.0,      # 周亏损超过5%
                'max_drawdown': -10.0,    # 最大回撤超过10%
                'volatility': 25.0,       # 年化波动率超过25%
                'consecutive_losses': 5   # 连续亏损超过5天
            }

        alerts = []
        metrics = self.calculate_risk_metrics(returns)

        # 检查最近一日
        if len(returns) > 0:
            last_return = returns.iloc[-1]
            if last_return < thresholds['daily_loss']:
                alerts.append(f"[高风险] 日亏损 {last_return:.2f}% 超过阈值 {thresholds['daily_loss']}%")

        # 检查最近一周
        if len(returns) >= 5:
            weekly_return = (1 + returns.tail(5) / 100).prod() - 1
            weekly_return *= 100
            if weekly_return < thresholds['weekly_loss']:
                alerts.append(f"[高风险] 周亏损 {weekly_return:.2f}% 超过阈值 {thresholds['weekly_loss']}%")

        # 检查最大回撤
        if metrics.max_drawdown < thresholds['max_drawdown']:
            alerts.append(f"[高风险] 最大回撤 {metrics.max_drawdown:.2f}% 超过阈值 {thresholds['max_drawdown']}%")

        # 检查波动率
        annual_vol = metrics.volatility * np.sqrt(252)
        if annual_vol > thresholds['volatility']:
            alerts.append(f"[中风险] 年化波动率 {annual_vol:.2f}% 超过阈值 {thresholds['volatility']}%")

        # 检查连续亏损
        if metrics.max_consecutive_losses >= thresholds['consecutive_losses']:
            alerts.append(f"[中风险] 连续亏损天数 {metrics.max_consecutive_losses} 超过阈值 {thresholds['consecutive_losses']}")

        return alerts

    def print_risk_report(self, metrics: RiskMetrics, title: str = "风险分析报告"):
        """打印风险报告"""
        print(f"\n{'='*60}")
        print(f"{title}")
        print('='*60)

        print("\n【收益分布】")
        print(f"  日均收益: {metrics.mean_return:.3f}%")
        print(f"  日收益中位数: {metrics.median_return:.3f}%")
        print(f"  最大单日收益: {metrics.max_return:.2f}%")
        print(f"  最大单日亏损: {metrics.min_return:.2f}%")

        print("\n【波动与分布】")
        print(f"  日波动率: {metrics.volatility:.3f}%")
        print(f"  偏度: {metrics.skewness:.3f}")
        print(f"  峰度: {metrics.kurtosis:.3f}")

        print("\n【风险指标】")
        print(f"  95% VaR: {metrics.var_95:.2f}%")
        print(f"  99% VaR: {metrics.var_99:.2f}%")
        print(f"  95% CVaR: {metrics.cvar_95:.2f}%")
        print(f"  99% CVaR: {metrics.cvar_99:.2f}%")

        print("\n【回撤分析】")
        print(f"  最大回撤: {metrics.max_drawdown:.2f}%")
        print(f"  平均回撤: {metrics.avg_drawdown:.2f}%")
        print(f"  最长回撤期: {metrics.max_drawdown_duration}天")

        print("\n【风险调整收益】")
        print(f"  夏普比率: {metrics.sharpe_ratio:.2f}")
        print(f"  Sortino比率: {metrics.sortino_ratio:.2f}")
        print(f"  Calmar比率: {metrics.calmar_ratio:.2f}")

        print("\n【交易统计】")
        print(f"  胜率: {metrics.win_rate:.1f}%")
        print(f"  盈亏比: {metrics.profit_factor:.2f}")
        print(f"  最长连续亏损: {metrics.max_consecutive_losses}天")

        print('='*60)


def main():
    """主函数: 风险管理示例"""
    import duckdb
    import os

    # 生成模拟数据或从数据库读取
    print("风险管理模块测试")

    # 模拟多空策略收益率
    np.random.seed(42)
    n_days = 252
    returns = pd.Series(
        np.random.normal(0.1, 1.3, n_days),
        index=pd.date_range('2024-01-01', periods=n_days, freq='B')
    )

    # 初始化风险管理器
    rm = RiskManager()

    # 计算风险指标
    metrics = rm.calculate_risk_metrics(returns)
    rm.print_risk_report(metrics, "多空策略风险分析")

    # 滚动风险
    print("\n滚动风险指标 (20日窗口):")
    rolling_risk = rm.calculate_rolling_risk(returns, window=20)
    print(rolling_risk.tail())

    # 压力测试
    print("\n压力测试:")
    scenarios = {
        '正常': 1.0,
        '温和压力': 1.5,
        '严重压力': 2.0,
        '极端压力': 3.0
    }
    stress_results = rm.stress_test(returns, scenarios)
    print(stress_results.to_string(index=False))

    # 风险预警
    print("\n风险预警:")
    alerts = rm.generate_risk_alerts(returns)
    if alerts:
        for alert in alerts:
            print(f"  {alert}")
    else:
        print("  无风险预警")

    # 回撤分析
    print("\nTOP回撤分析:")
    drawdown_analysis = rm.calculate_drawdown_analysis(returns)
    if len(drawdown_analysis) > 0:
        print(drawdown_analysis.to_string(index=False))


if __name__ == '__main__':
    main()
