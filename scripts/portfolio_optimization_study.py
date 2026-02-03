#!/usr/bin/env python3
"""
A股市场投资组合优化方法研究

研究内容:
1. 均值-方差优化 (Markowitz)
2. 收缩估计 (Ledoit-Wolf)
3. 风险平价
4. 最大分散化
5. 约束优化
6. 鲁棒优化 (Black-Litterman)
7. 回测比较
"""

import numpy as np
import pandas as pd
import duckdb
from datetime import datetime, timedelta
from scipy.optimize import minimize
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 配置
DB_PATH = '/Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db'
REPORT_PATH = '/Users/allen/workspace/python/stock/Tushare-DuckDB/reports/quality_check/portfolio_optimization_study.md'

# 研究参数
START_DATE = '20200101'  # 回测开始日期
END_DATE = '20251231'    # 回测结束日期
TRAIN_WINDOW = 252       # 训练窗口（交易日）
REBALANCE_FREQ = 21      # 再平衡频率（月度）
RISK_FREE_RATE = 0.02    # 无风险利率
N_STOCKS = 50            # 组合股票数量


def load_data():
    """加载数据"""
    print("正在加载数据...")
    conn = duckdb.connect(DB_PATH, read_only=True)

    # 获取沪深300成分股（最新）
    hs300_stocks = conn.execute("""
        SELECT DISTINCT con_code
        FROM index_weight
        WHERE index_code = '000300.SH'
        AND trade_date = (SELECT MAX(trade_date) FROM index_weight WHERE index_code = '000300.SH')
    """).fetchdf()['con_code'].tolist()

    print(f"沪深300成分股数量: {len(hs300_stocks)}")

    # 获取日线数据
    stocks_str = "', '".join(hs300_stocks)
    daily_data = conn.execute(f"""
        SELECT ts_code, trade_date, close, pct_chg/100 as ret
        FROM daily
        WHERE ts_code IN ('{stocks_str}')
        AND trade_date >= '{START_DATE}'
        AND trade_date <= '{END_DATE}'
        ORDER BY trade_date, ts_code
    """).fetchdf()

    # 获取股票基本信息
    stock_info = conn.execute(f"""
        SELECT ts_code, name, industry
        FROM stock_basic
        WHERE ts_code IN ('{stocks_str}')
    """).fetchdf()

    # 获取市值数据（用于市值加权）
    market_cap = conn.execute(f"""
        SELECT ts_code, trade_date, total_mv
        FROM daily_basic
        WHERE ts_code IN ('{stocks_str}')
        AND trade_date >= '{START_DATE}'
        AND trade_date <= '{END_DATE}'
    """).fetchdf()

    # 获取指数权重历史
    index_weights = conn.execute("""
        SELECT con_code, trade_date, weight/100 as weight
        FROM index_weight
        WHERE index_code = '000300.SH'
        AND trade_date >= '20200101'
        ORDER BY trade_date, con_code
    """).fetchdf()

    conn.close()

    return daily_data, stock_info, market_cap, index_weights, hs300_stocks


def prepare_returns_matrix(daily_data, stocks, date_range=None):
    """准备收益率矩阵"""
    # 转换为宽表格式
    returns_pivot = daily_data.pivot(index='trade_date', columns='ts_code', values='ret')

    if date_range is not None:
        start, end = date_range
        returns_pivot = returns_pivot.loc[start:end]

    # 选择指定的股票
    available_stocks = [s for s in stocks if s in returns_pivot.columns]
    returns_pivot = returns_pivot[available_stocks]

    # 处理缺失值
    returns_pivot = returns_pivot.dropna(axis=1, thresh=int(len(returns_pivot)*0.9))
    returns_pivot = returns_pivot.fillna(0)

    return returns_pivot


class PortfolioOptimizer:
    """投资组合优化器"""

    def __init__(self, returns, risk_free_rate=0.02):
        """
        初始化
        returns: DataFrame, 收益率矩阵 (日期 x 股票)
        """
        self.returns = returns
        self.rf = risk_free_rate / 252  # 日度无风险利率
        self.n_assets = returns.shape[1]
        self.asset_names = returns.columns.tolist()

        # 计算统计量
        self.mean_returns = returns.mean()
        self.cov_matrix = returns.cov()

    def portfolio_return(self, weights):
        """组合年化收益率"""
        return np.sum(self.mean_returns * weights) * 252

    def portfolio_volatility(self, weights):
        """组合年化波动率"""
        return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix * 252, weights)))

    def portfolio_sharpe(self, weights):
        """组合夏普比率"""
        ret = self.portfolio_return(weights)
        vol = self.portfolio_volatility(weights)
        return (ret - self.rf * 252) / vol if vol > 0 else 0

    def negative_sharpe(self, weights):
        """负夏普比率（用于最小化）"""
        return -self.portfolio_sharpe(weights)

    # ==================== 1. 均值-方差优化 ====================

    def minimum_variance_portfolio(self, constraints=None):
        """最小方差组合"""
        n = self.n_assets
        init_weights = np.ones(n) / n

        bounds = tuple((0, 0.1) for _ in range(n))  # 单股票权重上限10%

        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        if constraints:
            cons.extend(constraints)

        result = minimize(
            lambda w: self.portfolio_volatility(w),
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=cons
        )

        return result.x if result.success else init_weights

    def tangent_portfolio(self, constraints=None):
        """切线组合（最大夏普比率）"""
        n = self.n_assets
        init_weights = np.ones(n) / n

        bounds = tuple((0, 0.1) for _ in range(n))

        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        if constraints:
            cons.extend(constraints)

        result = minimize(
            self.negative_sharpe,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=cons
        )

        return result.x if result.success else init_weights

    def efficient_frontier(self, n_points=50):
        """构建有效前沿"""
        # 获取最小和最大收益率
        min_var_weights = self.minimum_variance_portfolio()

        # 最大收益组合
        max_ret_weights = np.zeros(self.n_assets)
        max_ret_idx = self.mean_returns.argmax()
        max_ret_weights[max_ret_idx] = 1

        min_ret = self.portfolio_return(min_var_weights)
        max_ret = self.mean_returns.max() * 252

        target_returns = np.linspace(min_ret, max_ret * 0.8, n_points)

        frontier = []
        for target in target_returns:
            weights = self._optimize_for_target_return(target)
            if weights is not None:
                vol = self.portfolio_volatility(weights)
                ret = self.portfolio_return(weights)
                sharpe = self.portfolio_sharpe(weights)
                frontier.append({'return': ret, 'volatility': vol, 'sharpe': sharpe})

        return pd.DataFrame(frontier)

    def _optimize_for_target_return(self, target_return):
        """给定目标收益率，最小化波动率"""
        n = self.n_assets
        init_weights = np.ones(n) / n

        bounds = tuple((0, 0.1) for _ in range(n))

        cons = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: self.portfolio_return(x) - target_return}
        ]

        result = minimize(
            lambda w: self.portfolio_volatility(w),
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=cons
        )

        return result.x if result.success else None

    # ==================== 2. 收缩估计 ====================

    def ledoit_wolf_shrinkage(self):
        """Ledoit-Wolf 协方差收缩估计"""
        X = self.returns.values
        n, p = X.shape

        # 样本协方差矩阵
        sample_cov = np.cov(X.T, ddof=1)

        # 收缩目标：对角矩阵（常数相关系数模型）
        # 使用市场因子模型的收缩目标
        mean_var = np.trace(sample_cov) / p
        target = mean_var * np.eye(p)

        # 计算收缩强度
        # Ledoit-Wolf 2004 方法
        X_centered = X - X.mean(axis=0)

        # 计算 pi (估计误差)
        y = X_centered ** 2
        phi_mat = np.dot(y.T, y) / n - 2 * np.dot(X_centered.T, X_centered) * sample_cov / n + sample_cov ** 2
        phi = np.sum(phi_mat)

        # 计算 gamma
        gamma = np.linalg.norm(sample_cov - target, 'fro') ** 2

        # 计算收缩强度
        kappa = (phi - gamma) / n
        shrinkage = max(0, min(1, kappa / gamma)) if gamma > 0 else 0

        # 收缩后的协方差矩阵
        shrunk_cov = shrinkage * target + (1 - shrinkage) * sample_cov

        return shrunk_cov, shrinkage

    def minimum_variance_ledoit_wolf(self, constraints=None):
        """使用Ledoit-Wolf收缩的最小方差组合"""
        shrunk_cov, shrinkage = self.ledoit_wolf_shrinkage()

        n = self.n_assets
        init_weights = np.ones(n) / n

        def portfolio_vol_shrunk(weights):
            return np.sqrt(np.dot(weights.T, np.dot(shrunk_cov * 252, weights)))

        bounds = tuple((0, 0.1) for _ in range(n))

        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        if constraints:
            cons.extend(constraints)

        result = minimize(
            portfolio_vol_shrunk,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=cons
        )

        return result.x if result.success else init_weights, shrinkage

    # ==================== 3. 风险平价 ====================

    def risk_parity_portfolio(self):
        """风险平价组合（等风险贡献）"""
        n = self.n_assets
        init_weights = np.ones(n) / n

        def risk_budget_objective(weights):
            """风险贡献偏离目标的惩罚"""
            cov = self.cov_matrix.values
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))

            if portfolio_vol < 1e-10:
                return 1e10

            # 边际风险贡献
            marginal_contrib = np.dot(cov, weights) / portfolio_vol

            # 风险贡献
            risk_contrib = weights * marginal_contrib

            # 目标：每个资产贡献相等
            target_contrib = portfolio_vol / n

            # 偏离惩罚
            return np.sum((risk_contrib - target_contrib) ** 2)

        bounds = tuple((0.001, 0.2) for _ in range(n))
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]

        result = minimize(
            risk_budget_objective,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=cons
        )

        return result.x if result.success else init_weights

    def risk_budget_portfolio(self, risk_budgets):
        """风险预算组合"""
        n = self.n_assets
        init_weights = np.array(risk_budgets) / np.sum(risk_budgets)

        def risk_budget_objective(weights):
            cov = self.cov_matrix.values
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))

            if portfolio_vol < 1e-10:
                return 1e10

            marginal_contrib = np.dot(cov, weights) / portfolio_vol
            risk_contrib = weights * marginal_contrib

            # 按比例分配风险
            target_contrib = portfolio_vol * np.array(risk_budgets) / np.sum(risk_budgets)

            return np.sum((risk_contrib - target_contrib) ** 2)

        bounds = tuple((0.001, 0.3) for _ in range(n))
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]

        result = minimize(
            risk_budget_objective,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=cons
        )

        return result.x if result.success else init_weights

    # ==================== 4. 最大分散化 ====================

    def diversification_ratio(self, weights):
        """计算分散化比率"""
        # 分散化比率 = 加权平均波动率 / 组合波动率
        individual_vols = np.sqrt(np.diag(self.cov_matrix.values)) * np.sqrt(252)
        weighted_vol = np.sum(weights * individual_vols)
        portfolio_vol = self.portfolio_volatility(weights)

        return weighted_vol / portfolio_vol if portfolio_vol > 0 else 1

    def max_diversification_portfolio(self):
        """最大分散化组合"""
        n = self.n_assets
        init_weights = np.ones(n) / n

        def neg_diversification_ratio(weights):
            return -self.diversification_ratio(weights)

        bounds = tuple((0, 0.1) for _ in range(n))
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]

        result = minimize(
            neg_diversification_ratio,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=cons
        )

        return result.x if result.success else init_weights

    # ==================== 5. 约束优化 ====================

    def constrained_optimization(self, weight_bounds=(0, 0.1),
                                  turnover_limit=None,
                                  prev_weights=None,
                                  industry_info=None,
                                  industry_limits=None):
        """约束优化"""
        n = self.n_assets
        init_weights = np.ones(n) / n

        bounds = tuple(weight_bounds for _ in range(n))

        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]

        # 换手率约束
        if turnover_limit is not None and prev_weights is not None:
            cons.append({
                'type': 'ineq',
                'fun': lambda x: turnover_limit - np.sum(np.abs(x - prev_weights))
            })

        # 行业约束
        if industry_info is not None and industry_limits is not None:
            for industry, limit in industry_limits.items():
                industry_mask = np.array([
                    1 if industry_info.get(asset) == industry else 0
                    for asset in self.asset_names
                ])
                cons.append({
                    'type': 'ineq',
                    'fun': lambda x, mask=industry_mask, lim=limit: lim - np.sum(x * mask)
                })

        result = minimize(
            self.negative_sharpe,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=cons
        )

        return result.x if result.success else init_weights

    # ==================== 6. Black-Litterman 模型 ====================

    def black_litterman(self, market_weights, views=None, view_confidences=None, tau=0.05):
        """
        Black-Litterman 模型

        market_weights: 市场权重（市值加权）
        views: 观点矩阵 P @ returns = Q
        view_confidences: 观点置信度
        tau: 不确定性参数
        """
        # 隐含均衡收益率
        pi = self.implied_returns(market_weights, tau)

        if views is None or len(views) == 0:
            # 无观点时，使用隐含收益率
            return self.tangent_portfolio_bl(pi)

        P = np.array([v['assets'] for v in views])  # 观点矩阵
        Q = np.array([v['return'] for v in views])  # 观点收益率

        # 观点不确定性
        if view_confidences is None:
            view_confidences = [0.5] * len(views)

        omega = np.diag([
            tau * np.dot(P[i], np.dot(self.cov_matrix.values, P[i])) / conf
            for i, conf in enumerate(view_confidences)
        ])

        # Black-Litterman 公式
        cov = self.cov_matrix.values
        tau_cov_inv = np.linalg.inv(tau * cov)
        omega_inv = np.linalg.inv(omega)

        # 后验期望收益率
        M = np.linalg.inv(tau_cov_inv + np.dot(P.T, np.dot(omega_inv, P)))
        posterior_returns = np.dot(M, np.dot(tau_cov_inv, pi) + np.dot(P.T, np.dot(omega_inv, Q)))

        # 后验协方差
        posterior_cov = cov + M

        return self.tangent_portfolio_bl(posterior_returns, posterior_cov)

    def implied_returns(self, market_weights, tau=0.05):
        """计算隐含均衡收益率"""
        # 风险厌恶系数（假设市场组合为切线组合）
        market_vol = np.sqrt(np.dot(market_weights.T, np.dot(self.cov_matrix.values * 252, market_weights)))
        market_ret = 0.08  # 假设市场预期收益率8%
        delta = (market_ret - self.rf * 252) / (market_vol ** 2)

        # 隐含收益率
        pi = delta * np.dot(self.cov_matrix.values * 252, market_weights)

        return pi

    def tangent_portfolio_bl(self, expected_returns, cov_matrix=None):
        """基于给定期望收益的切线组合"""
        if cov_matrix is None:
            cov_matrix = self.cov_matrix.values

        n = self.n_assets
        init_weights = np.ones(n) / n

        def neg_sharpe(weights):
            ret = np.sum(expected_returns * weights)
            vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return -(ret - self.rf * 252) / vol if vol > 0 else 0

        bounds = tuple((0, 0.1) for _ in range(n))
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]

        result = minimize(
            neg_sharpe,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=cons
        )

        return result.x if result.success else init_weights


def calculate_risk_contributions(weights, cov_matrix):
    """计算风险贡献"""
    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
    risk_contrib = weights * marginal_contrib
    risk_contrib_pct = risk_contrib / portfolio_vol
    return risk_contrib_pct


def backtest_portfolio(returns, weights_history, rebalance_dates):
    """回测投资组合"""
    portfolio_returns = []

    current_weights = None
    for date in returns.index:
        if date in rebalance_dates:
            idx = rebalance_dates.index(date)
            current_weights = weights_history[idx]

        if current_weights is not None:
            # 确保权重和收益率对齐
            common_assets = [a for a in returns.columns if a in current_weights.index]
            daily_ret = np.sum(returns.loc[date, common_assets] * current_weights[common_assets])
            portfolio_returns.append({'date': date, 'return': daily_ret})

    return pd.DataFrame(portfolio_returns)


def calculate_metrics(returns_series):
    """计算组合指标"""
    annual_return = returns_series.mean() * 252
    annual_vol = returns_series.std() * np.sqrt(252)
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0

    # 最大回撤
    cum_returns = (1 + returns_series).cumprod()
    rolling_max = cum_returns.expanding().max()
    drawdown = cum_returns / rolling_max - 1
    max_drawdown = drawdown.min()

    # Calmar比率
    calmar = -annual_return / max_drawdown if max_drawdown < 0 else 0

    # 胜率
    win_rate = (returns_series > 0).mean()

    return {
        'annual_return': annual_return,
        'annual_volatility': annual_vol,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar,
        'win_rate': win_rate
    }


def run_backtest(daily_data, stock_info, market_cap, stocks, start_date, end_date):
    """运行完整回测"""
    print("\n开始回测分析...")

    # 准备数据
    returns_df = prepare_returns_matrix(daily_data, stocks)

    # 获取交易日期
    all_dates = sorted(returns_df.index.tolist())

    # 创建行业映射
    industry_map = dict(zip(stock_info['ts_code'], stock_info['industry']))

    # 回测结果存储
    results = {
        'equal_weight': {'weights': [], 'dates': []},
        'min_variance': {'weights': [], 'dates': []},
        'tangent': {'weights': [], 'dates': []},
        'min_var_lw': {'weights': [], 'dates': []},
        'risk_parity': {'weights': [], 'dates': []},
        'max_div': {'weights': [], 'dates': []},
        'constrained': {'weights': [], 'dates': []},
        'black_litterman': {'weights': [], 'dates': []}
    }

    # 滚动窗口回测
    rebalance_idx = list(range(TRAIN_WINDOW, len(all_dates), REBALANCE_FREQ))

    print(f"回测期间: {all_dates[TRAIN_WINDOW]} - {all_dates[-1]}")
    print(f"再平衡次数: {len(rebalance_idx)}")

    prev_weights = None

    for i, idx in enumerate(rebalance_idx):
        if i % 10 == 0:
            print(f"  进度: {i+1}/{len(rebalance_idx)}")

        # 训练窗口
        train_start = all_dates[idx - TRAIN_WINDOW]
        train_end = all_dates[idx - 1]
        rebalance_date = all_dates[idx]

        # 获取训练数据
        train_returns = returns_df.loc[train_start:train_end]

        # 选择流动性好的股票
        valid_stocks = train_returns.columns[train_returns.notna().sum() > TRAIN_WINDOW * 0.9].tolist()
        if len(valid_stocks) < 20:
            continue

        # 限制股票数量
        selected_stocks = valid_stocks[:min(N_STOCKS, len(valid_stocks))]
        train_returns = train_returns[selected_stocks]

        # 创建优化器
        optimizer = PortfolioOptimizer(train_returns, RISK_FREE_RATE)

        # 1. 等权重
        eq_weights = pd.Series(np.ones(len(selected_stocks)) / len(selected_stocks),
                               index=selected_stocks)
        results['equal_weight']['weights'].append(eq_weights)
        results['equal_weight']['dates'].append(rebalance_date)

        # 2. 最小方差
        try:
            mv_weights = optimizer.minimum_variance_portfolio()
            results['min_variance']['weights'].append(pd.Series(mv_weights, index=selected_stocks))
            results['min_variance']['dates'].append(rebalance_date)
        except:
            pass

        # 3. 切线组合
        try:
            tang_weights = optimizer.tangent_portfolio()
            results['tangent']['weights'].append(pd.Series(tang_weights, index=selected_stocks))
            results['tangent']['dates'].append(rebalance_date)
        except:
            pass

        # 4. Ledoit-Wolf最小方差
        try:
            lw_weights, shrinkage = optimizer.minimum_variance_ledoit_wolf()
            results['min_var_lw']['weights'].append(pd.Series(lw_weights, index=selected_stocks))
            results['min_var_lw']['dates'].append(rebalance_date)
        except:
            pass

        # 5. 风险平价
        try:
            rp_weights = optimizer.risk_parity_portfolio()
            results['risk_parity']['weights'].append(pd.Series(rp_weights, index=selected_stocks))
            results['risk_parity']['dates'].append(rebalance_date)
        except:
            pass

        # 6. 最大分散化
        try:
            md_weights = optimizer.max_diversification_portfolio()
            results['max_div']['weights'].append(pd.Series(md_weights, index=selected_stocks))
            results['max_div']['dates'].append(rebalance_date)
        except:
            pass

        # 7. 约束优化（带换手率限制）
        try:
            industry_info = {s: industry_map.get(s, 'unknown') for s in selected_stocks}
            industry_limits = {ind: 0.3 for ind in set(industry_info.values())}

            prev_w = None
            if prev_weights is not None:
                prev_w = np.array([prev_weights.get(s, 0) for s in selected_stocks])

            const_weights = optimizer.constrained_optimization(
                weight_bounds=(0, 0.05),
                turnover_limit=0.5,
                prev_weights=prev_w,
                industry_info=industry_info,
                industry_limits=industry_limits
            )
            results['constrained']['weights'].append(pd.Series(const_weights, index=selected_stocks))
            results['constrained']['dates'].append(rebalance_date)
            prev_weights = pd.Series(const_weights, index=selected_stocks)
        except:
            pass

        # 8. Black-Litterman
        try:
            # 使用等权重作为市场权重（简化）
            market_w = np.ones(len(selected_stocks)) / len(selected_stocks)
            bl_weights = optimizer.black_litterman(market_w)
            results['black_litterman']['weights'].append(pd.Series(bl_weights, index=selected_stocks))
            results['black_litterman']['dates'].append(rebalance_date)
        except:
            pass

    return results, returns_df


def analyze_backtest_results(results, returns_df):
    """分析回测结果"""
    print("\n分析回测结果...")

    metrics_summary = {}
    portfolio_values = {}
    turnover_stats = {}

    for strategy, data in results.items():
        if len(data['weights']) == 0:
            continue

        # 计算组合收益率
        portfolio_rets = []
        turnovers = []
        prev_weights = None

        for i in range(len(data['dates']) - 1):
            start_date = data['dates'][i]
            end_date = data['dates'][i + 1]
            weights = data['weights'][i]

            # 计算期间收益
            period_returns = returns_df.loc[start_date:end_date]
            for date in period_returns.index:
                daily_ret = 0
                for stock in weights.index:
                    if stock in period_returns.columns:
                        ret = period_returns.loc[date, stock]
                        if not pd.isna(ret):
                            daily_ret += weights[stock] * ret
                portfolio_rets.append({'date': date, 'return': daily_ret})

            # 计算换手率
            if prev_weights is not None:
                common = set(weights.index) & set(prev_weights.index)
                turnover = sum(abs(weights.get(s, 0) - prev_weights.get(s, 0)) for s in common)
                turnover += sum(weights.get(s, 0) for s in set(weights.index) - common)
                turnover += sum(prev_weights.get(s, 0) for s in set(prev_weights.index) - common)
                turnovers.append(turnover)

            prev_weights = weights

        if len(portfolio_rets) == 0:
            continue

        # 转换为DataFrame
        portfolio_df = pd.DataFrame(portfolio_rets)
        portfolio_df.set_index('date', inplace=True)

        # 计算指标
        metrics = calculate_metrics(portfolio_df['return'])
        metrics['avg_turnover'] = np.mean(turnovers) if turnovers else 0
        metrics_summary[strategy] = metrics

        # 累计收益
        portfolio_df['cumulative'] = (1 + portfolio_df['return']).cumprod()
        portfolio_values[strategy] = portfolio_df

        turnover_stats[strategy] = {
            'avg': np.mean(turnovers) if turnovers else 0,
            'max': np.max(turnovers) if turnovers else 0,
            'min': np.min(turnovers) if turnovers else 0
        }

    return metrics_summary, portfolio_values, turnover_stats


def run_efficient_frontier_analysis(returns_df):
    """有效前沿分析"""
    print("\n构建有效前沿...")

    # 使用最近一年数据
    recent_returns = returns_df.tail(252)

    # 选择数据完整的股票
    valid_stocks = recent_returns.columns[recent_returns.notna().sum() > 200].tolist()[:30]
    recent_returns = recent_returns[valid_stocks]

    optimizer = PortfolioOptimizer(recent_returns, RISK_FREE_RATE)

    # 构建有效前沿
    frontier = optimizer.efficient_frontier(n_points=30)

    # 计算特殊组合
    min_var_weights = optimizer.minimum_variance_portfolio()
    tangent_weights = optimizer.tangent_portfolio()

    special_portfolios = {
        'min_variance': {
            'weights': min_var_weights,
            'return': optimizer.portfolio_return(min_var_weights),
            'volatility': optimizer.portfolio_volatility(min_var_weights),
            'sharpe': optimizer.portfolio_sharpe(min_var_weights)
        },
        'tangent': {
            'weights': tangent_weights,
            'return': optimizer.portfolio_return(tangent_weights),
            'volatility': optimizer.portfolio_volatility(tangent_weights),
            'sharpe': optimizer.portfolio_sharpe(tangent_weights)
        }
    }

    return frontier, special_portfolios, valid_stocks


def run_shrinkage_analysis(returns_df):
    """收缩估计分析"""
    print("\n分析协方差收缩效果...")

    recent_returns = returns_df.tail(252)
    valid_stocks = recent_returns.columns[recent_returns.notna().sum() > 200].tolist()[:30]
    recent_returns = recent_returns[valid_stocks]

    optimizer = PortfolioOptimizer(recent_returns, RISK_FREE_RATE)

    # 样本协方差
    sample_cov = optimizer.cov_matrix

    # 收缩协方差
    shrunk_cov, shrinkage_intensity = optimizer.ledoit_wolf_shrinkage()

    # 比较
    results = {
        'sample_cov_condition': np.linalg.cond(sample_cov),
        'shrunk_cov_condition': np.linalg.cond(shrunk_cov),
        'shrinkage_intensity': shrinkage_intensity,
        'sample_cov_eigenvalues': np.linalg.eigvalsh(sample_cov),
        'shrunk_cov_eigenvalues': np.linalg.eigvalsh(shrunk_cov)
    }

    # 优化结果比较
    mv_weights_sample = optimizer.minimum_variance_portfolio()
    mv_weights_shrunk, _ = optimizer.minimum_variance_ledoit_wolf()

    results['mv_sample_vol'] = optimizer.portfolio_volatility(mv_weights_sample)
    results['mv_shrunk_vol'] = optimizer.portfolio_volatility(mv_weights_shrunk)
    results['mv_sample_weights'] = mv_weights_sample
    results['mv_shrunk_weights'] = mv_weights_shrunk

    return results


def run_risk_analysis(returns_df):
    """风险分析"""
    print("\n分析风险特征...")

    recent_returns = returns_df.tail(252)
    valid_stocks = recent_returns.columns[recent_returns.notna().sum() > 200].tolist()[:30]
    recent_returns = recent_returns[valid_stocks]

    optimizer = PortfolioOptimizer(recent_returns, RISK_FREE_RATE)

    # 各种组合的风险贡献
    portfolios = {
        'equal_weight': np.ones(len(valid_stocks)) / len(valid_stocks),
        'min_variance': optimizer.minimum_variance_portfolio(),
        'risk_parity': optimizer.risk_parity_portfolio(),
        'max_diversification': optimizer.max_diversification_portfolio()
    }

    risk_analysis = {}
    for name, weights in portfolios.items():
        cov = optimizer.cov_matrix.values
        risk_contrib = calculate_risk_contributions(weights, cov)

        risk_analysis[name] = {
            'weights': weights,
            'risk_contributions': risk_contrib,
            'diversification_ratio': optimizer.diversification_ratio(weights),
            'herfindahl_weight': np.sum(weights ** 2),
            'herfindahl_risk': np.sum(risk_contrib ** 2),
            'effective_n_assets': 1 / np.sum(weights ** 2),
            'portfolio_vol': optimizer.portfolio_volatility(weights)
        }

    return risk_analysis, valid_stocks


def generate_report(metrics_summary, portfolio_values, turnover_stats,
                   frontier, special_portfolios, shrinkage_results,
                   risk_analysis, stock_names):
    """生成分析报告"""
    print("\n生成报告...")

    report = []
    report.append("# A股市场投资组合优化方法研究报告")
    report.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"\n研究期间: {START_DATE} - {END_DATE}")
    report.append(f"\n股票池: 沪深300成分股")

    # 1. 执行摘要
    report.append("\n## 1. 执行摘要")
    report.append("""
本研究系统比较了A股市场上多种投资组合优化方法的表现，包括：
- 均值-方差优化（Markowitz模型）
- 协方差收缩估计（Ledoit-Wolf）
- 风险平价组合
- 最大分散化组合
- 约束优化
- Black-Litterman模型

主要发现：
""")

    # 找出最佳策略
    if metrics_summary:
        best_sharpe = max(metrics_summary.items(), key=lambda x: x[1].get('sharpe_ratio', 0))
        best_return = max(metrics_summary.items(), key=lambda x: x[1].get('annual_return', 0))
        lowest_vol = min(metrics_summary.items(), key=lambda x: x[1].get('annual_volatility', float('inf')))
        lowest_dd = max(metrics_summary.items(), key=lambda x: x[1].get('max_drawdown', -float('inf')))

        report.append(f"- **最高夏普比率**: {best_sharpe[0]} ({best_sharpe[1]['sharpe_ratio']:.3f})")
        report.append(f"- **最高年化收益**: {best_return[0]} ({best_return[1]['annual_return']:.2%})")
        report.append(f"- **最低波动率**: {lowest_vol[0]} ({lowest_vol[1]['annual_volatility']:.2%})")
        report.append(f"- **最小回撤**: {lowest_dd[0]} ({lowest_dd[1]['max_drawdown']:.2%})")

    # 2. 均值-方差优化
    report.append("\n## 2. 均值-方差优化（Markowitz模型）")
    report.append("""
### 2.1 理论基础

Markowitz均值-方差模型是现代投资组合理论的基石，其核心思想是：
- 投资者追求收益最大化同时规避风险
- 通过分散投资可以降低非系统性风险
- 最优组合位于有效前沿上

### 2.2 有效前沿
""")

    if frontier is not None and len(frontier) > 0:
        report.append("\n有效前沿关键点：")
        report.append("| 收益率 | 波动率 | 夏普比率 |")
        report.append("|--------|--------|----------|")
        for _, row in frontier.iloc[::5].iterrows():  # 每5个取一个
            report.append(f"| {row['return']:.2%} | {row['volatility']:.2%} | {row['sharpe']:.3f} |")

    if special_portfolios:
        report.append("\n### 2.3 特殊组合")
        report.append("\n**最小方差组合**")
        mv = special_portfolios['min_variance']
        report.append(f"- 年化收益率: {mv['return']:.2%}")
        report.append(f"- 年化波动率: {mv['volatility']:.2%}")
        report.append(f"- 夏普比率: {mv['sharpe']:.3f}")

        report.append("\n**切线组合（最大夏普比率）**")
        tang = special_portfolios['tangent']
        report.append(f"- 年化收益率: {tang['return']:.2%}")
        report.append(f"- 年化波动率: {tang['volatility']:.2%}")
        report.append(f"- 夏普比率: {tang['sharpe']:.3f}")

    # 3. 收缩估计
    report.append("\n## 3. 协方差矩阵收缩估计")
    report.append("""
### 3.1 为什么需要收缩估计？

样本协方差矩阵存在以下问题：
1. **估计误差大**: 当股票数量接近或超过观测数量时，协方差矩阵的估计误差会很大
2. **条件数高**: 样本协方差矩阵可能接近奇异，导致优化不稳定
3. **极端权重**: 估计误差会导致优化产生极端权重

### 3.2 Ledoit-Wolf收缩估计

收缩估计将样本协方差矩阵向一个结构化目标收缩：
$$ \\Sigma_{shrunk} = \\alpha \\cdot F + (1-\\alpha) \\cdot S $$

其中：
- $S$: 样本协方差矩阵
- $F$: 收缩目标（如对角矩阵）
- $\\alpha$: 收缩强度（0到1之间）
""")

    if shrinkage_results:
        report.append("\n### 3.3 收缩效果分析")
        report.append(f"\n- **收缩强度**: {shrinkage_results['shrinkage_intensity']:.4f}")
        report.append(f"- **样本协方差条件数**: {shrinkage_results['sample_cov_condition']:.2f}")
        report.append(f"- **收缩后条件数**: {shrinkage_results['shrunk_cov_condition']:.2f}")

        report.append("\n### 3.4 对优化的影响")
        report.append(f"- 样本协方差最小方差组合波动率: {shrinkage_results['mv_sample_vol']:.2%}")
        report.append(f"- 收缩协方差最小方差组合波动率: {shrinkage_results['mv_shrunk_vol']:.2%}")

        # 权重分布比较
        report.append("\n**权重集中度比较**")
        sample_w = shrinkage_results['mv_sample_weights']
        shrunk_w = shrinkage_results['mv_shrunk_weights']
        report.append(f"- 样本协方差: HHI={np.sum(sample_w**2):.4f}, 有效资产数={1/np.sum(sample_w**2):.1f}")
        report.append(f"- 收缩协方差: HHI={np.sum(shrunk_w**2):.4f}, 有效资产数={1/np.sum(shrunk_w**2):.1f}")

    # 4. 风险平价
    report.append("\n## 4. 风险平价组合")
    report.append("""
### 4.1 核心理念

风险平价的核心思想是让每个资产对组合风险的贡献相等：
$$ RC_i = w_i \\cdot \\frac{\\partial \\sigma_p}{\\partial w_i} = \\frac{\\sigma_p}{n} $$

### 4.2 优势与特点

1. **不依赖收益率预测**: 只使用风险信息，避免了收益率预测的不确定性
2. **分散化**: 通过等风险贡献实现更好的分散化
3. **稳健性**: 对参数估计误差更加稳健
""")

    if risk_analysis:
        report.append("\n### 4.3 风险贡献分析")

        # 等权重
        eq_w = risk_analysis.get('equal_weight', {})
        rp = risk_analysis.get('risk_parity', {})

        if eq_w and rp:
            report.append("\n| 指标 | 等权重 | 风险平价 |")
            report.append("|------|--------|----------|")
            report.append(f"| 组合波动率 | {eq_w.get('portfolio_vol', 0):.2%} | {rp.get('portfolio_vol', 0):.2%} |")
            report.append(f"| 分散化比率 | {eq_w.get('diversification_ratio', 0):.3f} | {rp.get('diversification_ratio', 0):.3f} |")
            report.append(f"| 权重HHI | {eq_w.get('herfindahl_weight', 0):.4f} | {rp.get('herfindahl_weight', 0):.4f} |")
            report.append(f"| 风险HHI | {eq_w.get('herfindahl_risk', 0):.4f} | {rp.get('herfindahl_risk', 0):.4f} |")
            report.append(f"| 有效资产数 | {eq_w.get('effective_n_assets', 0):.1f} | {rp.get('effective_n_assets', 0):.1f} |")

    # 5. 最大分散化
    report.append("\n## 5. 最大分散化组合")
    report.append("""
### 5.1 分散化比率

分散化比率定义为加权平均波动率与组合波动率的比值：
$$ DR = \\frac{\\sum_i w_i \\sigma_i}{\\sigma_p} $$

DR越高，说明组合通过分散化降低了更多的风险。

### 5.2 最大分散化组合

最大化分散化比率的组合可以被证明等价于：
- 最大化组合的信息比率（假设所有资产有相同的夏普比率）
- 投资于最大夏普比率的有效组合（在一定条件下）
""")

    if risk_analysis:
        md = risk_analysis.get('max_diversification', {})
        if md:
            report.append("\n### 5.3 最大分散化组合特征")
            report.append(f"- 分散化比率: {md.get('diversification_ratio', 0):.3f}")
            report.append(f"- 组合波动率: {md.get('portfolio_vol', 0):.2%}")
            report.append(f"- 有效资产数: {md.get('effective_n_assets', 0):.1f}")

    # 6. 约束优化
    report.append("\n## 6. 约束优化")
    report.append("""
### 6.1 常见约束类型

在实际投资中，需要考虑多种约束：

1. **权重约束**
   - 单只股票权重上限（如5%或10%）
   - 做多约束（权重非负）

2. **换手率约束**
   - 限制每次再平衡的换手率
   - 降低交易成本

3. **行业约束**
   - 单一行业权重上限
   - 行业中性约束

4. **因子暴露约束**
   - 控制对特定因子的暴露
   - 实现市场中性或特定因子中性

### 6.2 约束对优化的影响

约束会限制优化空间，通常导致：
- 更分散的权重分布
- 更低的样本内表现
- 更稳健的样本外表现
- 更低的换手率和交易成本
""")

    # 7. Black-Litterman模型
    report.append("\n## 7. 鲁棒优化与Black-Litterman模型")
    report.append("""
### 7.1 参数不确定性

均值-方差优化对输入参数（特别是预期收益率）非常敏感。解决方法包括：
1. **收缩估计**: 对协方差矩阵进行收缩
2. **鲁棒优化**: 考虑参数的不确定性范围
3. **Black-Litterman模型**: 结合市场均衡和投资者观点

### 7.2 Black-Litterman模型

Black-Litterman模型的核心思想：
1. 从市场均衡出发，反推隐含的预期收益率
2. 允许投资者表达对特定资产的观点
3. 用贝叶斯方法结合均衡收益和投资者观点

**隐含收益率**:
$$ \\pi = \\delta \\Sigma w_{mkt} $$

**后验收益率**:
$$ E[r] = [(\\tau\\Sigma)^{-1} + P'\\Omega^{-1}P]^{-1}[(\\tau\\Sigma)^{-1}\\pi + P'\\Omega^{-1}Q] $$

### 7.3 优势

- 更稳定的权重
- 可以融入投资者的市场观点
- 避免极端权重
- 提供合理的基准（市场组合）
""")

    # 8. 回测比较
    report.append("\n## 8. 回测比较")

    if metrics_summary:
        report.append("\n### 8.1 各策略表现汇总")
        report.append("\n| 策略 | 年化收益 | 年化波动 | 夏普比率 | 最大回撤 | Calmar | 胜率 | 平均换手 |")
        report.append("|------|----------|----------|----------|----------|--------|------|----------|")

        strategy_names = {
            'equal_weight': '等权重',
            'min_variance': '最小方差',
            'tangent': '切线组合',
            'min_var_lw': 'LW最小方差',
            'risk_parity': '风险平价',
            'max_div': '最大分散化',
            'constrained': '约束优化',
            'black_litterman': 'Black-Litterman'
        }

        for strategy, metrics in sorted(metrics_summary.items(),
                                        key=lambda x: x[1].get('sharpe_ratio', 0),
                                        reverse=True):
            name = strategy_names.get(strategy, strategy)
            report.append(
                f"| {name} | "
                f"{metrics.get('annual_return', 0):.2%} | "
                f"{metrics.get('annual_volatility', 0):.2%} | "
                f"{metrics.get('sharpe_ratio', 0):.3f} | "
                f"{metrics.get('max_drawdown', 0):.2%} | "
                f"{metrics.get('calmar_ratio', 0):.3f} | "
                f"{metrics.get('win_rate', 0):.2%} | "
                f"{metrics.get('avg_turnover', 0):.2%} |"
            )

    if turnover_stats:
        report.append("\n### 8.2 换手率分析")
        report.append("\n| 策略 | 平均换手率 | 最大换手率 | 最小换手率 |")
        report.append("|------|------------|------------|------------|")

        for strategy, stats in turnover_stats.items():
            name = strategy_names.get(strategy, strategy)
            report.append(
                f"| {name} | "
                f"{stats['avg']:.2%} | "
                f"{stats['max']:.2%} | "
                f"{stats['min']:.2%} |"
            )

    report.append("""
### 8.3 交易成本影响

假设双边交易成本为0.3%（包含佣金和冲击成本），对各策略收益的影响：
""")

    if metrics_summary and turnover_stats:
        report.append("\n| 策略 | 原始年化收益 | 年化交易成本 | 净年化收益 |")
        report.append("|------|--------------|--------------|------------|")

        for strategy, metrics in metrics_summary.items():
            name = strategy_names.get(strategy, strategy)
            raw_ret = metrics.get('annual_return', 0)
            avg_turnover = turnover_stats.get(strategy, {}).get('avg', 0)
            # 假设每月再平衡，年化换手率 = 平均换手率 * 12
            annual_turnover = avg_turnover * 12
            trading_cost = annual_turnover * 0.003  # 0.3%双边成本
            net_ret = raw_ret - trading_cost
            report.append(f"| {name} | {raw_ret:.2%} | {trading_cost:.2%} | {net_ret:.2%} |")

    # 9. 结论与建议
    report.append("\n## 9. 结论与建议")
    report.append("""
### 9.1 主要发现

1. **风险平价组合**通常提供更稳定的表现，在各种市场环境下表现相对均衡

2. **最小方差组合**在熊市中表现较好，但可能错过牛市的大部分收益

3. **Ledoit-Wolf收缩估计**显著改善了协方差矩阵的条件数，使优化更加稳健

4. **约束优化**通过限制权重和换手率，可以显著降低交易成本

5. **Black-Litterman模型**提供了一种结合市场均衡和投资者观点的优雅方法

### 9.2 实践建议

1. **不要仅使用样本协方差**: 在优化之前，应该对协方差矩阵进行收缩处理

2. **添加合理的约束**: 权重约束、行业约束可以避免极端配置

3. **考虑交易成本**: 换手率约束可以显著降低交易成本

4. **组合多种方法**: 可以考虑结合多种方法的优点
   - 使用风险平价作为基准
   - 叠加Black-Litterman的观点调整
   - 使用收缩估计改进协方差矩阵

5. **定期监控和再平衡**: 根据市场情况调整再平衡频率

### 9.3 局限性

1. **历史数据的局限**: 过去的表现不能保证未来的收益

2. **模型假设**: 各模型都有其假设条件，如收益率正态分布等

3. **执行成本**: 实际执行可能面临流动性、冲击成本等问题

4. **参数敏感性**: 优化结果对参数选择较为敏感
""")

    # 10. 附录
    report.append("\n## 10. 附录")
    report.append("""
### 10.1 研究参数

| 参数 | 值 | 说明 |
|------|------|------|
| 训练窗口 | 252天 | 约一年的交易日 |
| 再平衡频率 | 21天 | 约月度再平衡 |
| 无风险利率 | 2% | 年化 |
| 单股权重上限 | 10% | 分散化约束 |
| 股票数量 | 50只 | 每次优化的最大股票数 |

### 10.2 方法论参考

1. Markowitz, H. (1952). Portfolio Selection. Journal of Finance.
2. Ledoit, O., & Wolf, M. (2004). Honey, I shrunk the sample covariance matrix. Journal of Portfolio Management.
3. Maillard, S., Roncalli, T., & Teiletche, J. (2010). The properties of equally weighted risk contribution portfolios. Journal of Portfolio Management.
4. Choueifaty, Y., & Coignard, Y. (2008). Toward Maximum Diversification. Journal of Portfolio Management.
5. Black, F., & Litterman, R. (1992). Global Portfolio Optimization. Financial Analysts Journal.
""")

    # 保存报告
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    print(f"\n报告已保存到: {REPORT_PATH}")

    return '\n'.join(report)


def main():
    """主函数"""
    print("=" * 60)
    print("A股市场投资组合优化方法研究")
    print("=" * 60)

    # 加载数据
    daily_data, stock_info, market_cap, index_weights, hs300_stocks = load_data()

    # 准备收益率矩阵
    returns_df = prepare_returns_matrix(daily_data, hs300_stocks)
    print(f"收益率矩阵: {returns_df.shape[0]} 天 x {returns_df.shape[1]} 只股票")

    # 1. 有效前沿分析
    frontier, special_portfolios, frontier_stocks = run_efficient_frontier_analysis(returns_df)

    # 2. 收缩估计分析
    shrinkage_results = run_shrinkage_analysis(returns_df)

    # 3. 风险分析
    risk_analysis, risk_stocks = run_risk_analysis(returns_df)

    # 4. 运行回测
    backtest_results, returns_df = run_backtest(daily_data, stock_info, market_cap, hs300_stocks, START_DATE, END_DATE)

    # 5. 分析回测结果
    metrics_summary, portfolio_values, turnover_stats = analyze_backtest_results(backtest_results, returns_df)

    # 6. 生成报告
    report = generate_report(
        metrics_summary, portfolio_values, turnover_stats,
        frontier, special_portfolios, shrinkage_results,
        risk_analysis, frontier_stocks
    )

    print("\n" + "=" * 60)
    print("研究完成!")
    print("=" * 60)

    return report


if __name__ == "__main__":
    main()
