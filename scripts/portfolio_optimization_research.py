#!/usr/bin/env python3
"""
投资组合优化研究报告
====================================
1. 优化方法：均值-方差优化、风险平价、最大分散化、Black-Litterman
2. 约束条件：行业约束、个股权重约束、换手率约束
3. 实证分析：不同方法对比、样本外表现、稳健性分析

Author: Research Team
Date: 2026-02-01
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import duckdb
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import scipy.optimize as sco
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 配置
DB_PATH = '/Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db'
REPORT_DIR = '/Users/allen/workspace/python/stock/Tushare-DuckDB/reports/research'

class DataLoader:
    """数据加载器"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = duckdb.connect(db_path, read_only=True)

    def get_stock_pool(self, trade_date: str, min_mv: float = 50, max_stocks: int = 300) -> pd.DataFrame:
        """
        获取股票池
        筛选条件：上市满1年、流通市值大于min_mv亿、剔除ST
        """
        query = f"""
        WITH latest_basic AS (
            SELECT
                ts_code,
                circ_mv / 10000 as circ_mv_yi,  -- 转换为亿
                turnover_rate_f
            FROM daily_basic
            WHERE trade_date = '{trade_date}'
            AND circ_mv > {min_mv * 10000}  -- 流通市值大于min_mv亿
        ),
        stock_info AS (
            SELECT
                ts_code,
                name,
                industry,
                list_date
            FROM stock_basic
            WHERE list_status = 'L'
            AND name NOT LIKE '%ST%'
            AND list_date <= '{str(int(trade_date) - 10000)}'  -- 上市满1年
        )
        SELECT
            s.ts_code,
            s.name,
            s.industry,
            s.list_date,
            b.circ_mv_yi,
            b.turnover_rate_f
        FROM stock_info s
        INNER JOIN latest_basic b ON s.ts_code = b.ts_code
        ORDER BY b.circ_mv_yi DESC
        LIMIT {max_stocks}
        """
        return self.conn.execute(query).fetchdf()

    def get_industry_mapping(self) -> pd.DataFrame:
        """获取申万行业分类"""
        query = """
        SELECT
            ts_code,
            name,
            l1_name as industry,
            l1_code as industry_code
        FROM index_member_all
        WHERE is_new = 'Y'
        """
        return self.conn.execute(query).fetchdf()

    def get_price_data(self, ts_codes: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """获取价格数据"""
        codes_str = "', '".join(ts_codes)
        query = f"""
        SELECT
            ts_code,
            trade_date,
            close,
            pct_chg
        FROM daily
        WHERE ts_code IN ('{codes_str}')
        AND trade_date >= '{start_date}'
        AND trade_date <= '{end_date}'
        ORDER BY trade_date, ts_code
        """
        return self.conn.execute(query).fetchdf()

    def get_adj_factor(self, ts_codes: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """获取复权因子"""
        codes_str = "', '".join(ts_codes)
        query = f"""
        SELECT
            ts_code,
            trade_date,
            adj_factor
        FROM adj_factor
        WHERE ts_code IN ('{codes_str}')
        AND trade_date >= '{start_date}'
        AND trade_date <= '{end_date}'
        ORDER BY trade_date, ts_code
        """
        return self.conn.execute(query).fetchdf()

    def get_index_daily(self, index_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取指数日线数据"""
        query = f"""
        SELECT
            trade_date,
            close,
            pct_chg
        FROM index_daily
        WHERE ts_code = '{index_code}'
        AND trade_date >= '{start_date}'
        AND trade_date <= '{end_date}'
        ORDER BY trade_date
        """
        return self.conn.execute(query).fetchdf()

    def close(self):
        self.conn.close()


class ReturnCalculator:
    """收益率计算器"""

    @staticmethod
    def calculate_returns(price_df: pd.DataFrame, adj_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        计算收益率矩阵
        返回：行为日期，列为股票代码的收益率矩阵
        """
        # 转换为透视表
        price_pivot = price_df.pivot(index='trade_date', columns='ts_code', values='close')

        # 如果有复权因子，进行后复权
        if adj_df is not None and len(adj_df) > 0:
            adj_pivot = adj_df.pivot(index='trade_date', columns='ts_code', values='adj_factor')
            # 对齐索引和列
            common_dates = price_pivot.index.intersection(adj_pivot.index)
            common_codes = price_pivot.columns.intersection(adj_pivot.columns)
            price_pivot = price_pivot.loc[common_dates, common_codes]
            adj_pivot = adj_pivot.loc[common_dates, common_codes]
            price_pivot = price_pivot * adj_pivot / adj_pivot.iloc[-1]

        # 计算日收益率
        returns = price_pivot.pct_change().dropna()
        return returns

    @staticmethod
    def annualize_returns(returns: pd.DataFrame, periods: int = 252) -> pd.Series:
        """年化收益率"""
        mean_returns = returns.mean()
        return mean_returns * periods

    @staticmethod
    def annualize_cov(returns: pd.DataFrame, periods: int = 252) -> pd.DataFrame:
        """年化协方差矩阵"""
        return returns.cov() * periods


class PortfolioOptimizer:
    """投资组合优化器"""

    def __init__(self, returns: pd.DataFrame, risk_free_rate: float = 0.02):
        self.returns = returns
        self.risk_free_rate = risk_free_rate
        self.n_assets = returns.shape[1]
        self.mean_returns = ReturnCalculator.annualize_returns(returns)
        self.cov_matrix = ReturnCalculator.annualize_cov(returns)
        self.asset_names = returns.columns.tolist()

    def portfolio_performance(self, weights: np.ndarray) -> Tuple[float, float, float]:
        """计算组合表现：收益、波动、夏普"""
        ret = np.dot(weights, self.mean_returns)
        vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        sharpe = (ret - self.risk_free_rate) / vol if vol > 0 else 0
        return ret, vol, sharpe

    def mean_variance_optimize(self,
                                target: str = 'sharpe',
                                industry_constraints: Dict = None,
                                weight_bounds: Tuple = (0, 0.1),
                                turnover_limit: float = None,
                                current_weights: np.ndarray = None) -> np.ndarray:
        """
        均值-方差优化

        Args:
            target: 优化目标 ('sharpe', 'min_vol', 'max_return')
            industry_constraints: 行业约束字典 {industry: (min_weight, max_weight)}
            weight_bounds: 个股权重约束
            turnover_limit: 换手率约束
            current_weights: 当前权重（用于换手率约束）
        """
        n = self.n_assets

        # 目标函数
        def neg_sharpe(weights):
            ret, vol, _ = self.portfolio_performance(weights)
            return -(ret - self.risk_free_rate) / vol if vol > 0 else 0

        def portfolio_vol(weights):
            return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))

        def neg_return(weights):
            return -np.dot(weights, self.mean_returns)

        # 选择目标函数
        if target == 'sharpe':
            objective = neg_sharpe
        elif target == 'min_vol':
            objective = portfolio_vol
        else:
            objective = neg_return

        # 约束条件
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # 权重和为1

        # 换手率约束
        if turnover_limit is not None and current_weights is not None:
            constraints.append({
                'type': 'ineq',
                'fun': lambda x: turnover_limit - np.sum(np.abs(x - current_weights))
            })

        # 行业约束（简化版，假设已有行业映射）
        # 实际使用时需要传入股票的行业分类信息

        # 边界条件
        bounds = tuple(weight_bounds for _ in range(n))

        # 初始权重
        init_weights = np.array([1/n] * n)

        # 优化
        result = sco.minimize(
            objective,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )

        return result.x if result.success else init_weights

    def risk_parity(self, weight_bounds: Tuple = (0, 0.1)) -> np.ndarray:
        """
        风险平价优化
        目标：使每个资产对组合总风险的贡献相等
        """
        n = self.n_assets

        def risk_contribution(weights):
            """计算每个资产的风险贡献"""
            port_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            marginal_risk = np.dot(self.cov_matrix, weights) / port_vol
            risk_contrib = weights * marginal_risk
            return risk_contrib

        def risk_parity_objective(weights):
            """风险平价目标函数：最小化风险贡献的方差"""
            rc = risk_contribution(weights)
            target_rc = np.ones(n) / n  # 目标：每个资产贡献相等
            # 使用相对风险贡献
            total_rc = np.sum(rc)
            if total_rc > 0:
                relative_rc = rc / total_rc
                return np.sum((relative_rc - target_rc) ** 2)
            return 0

        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple(weight_bounds for _ in range(n))
        init_weights = np.array([1/n] * n)

        result = sco.minimize(
            risk_parity_objective,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )

        return result.x if result.success else init_weights

    def max_diversification(self, weight_bounds: Tuple = (0, 0.1)) -> np.ndarray:
        """
        最大分散化优化
        目标：最大化分散化比率 = 加权平均波动率 / 组合波动率
        """
        n = self.n_assets
        asset_vols = np.sqrt(np.diag(self.cov_matrix))

        def neg_diversification_ratio(weights):
            """负的分散化比率"""
            weighted_avg_vol = np.dot(weights, asset_vols)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            if port_vol > 0:
                return -weighted_avg_vol / port_vol
            return 0

        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple(weight_bounds for _ in range(n))
        init_weights = np.array([1/n] * n)

        result = sco.minimize(
            neg_diversification_ratio,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )

        return result.x if result.success else init_weights

    def black_litterman(self,
                        market_weights: np.ndarray,
                        views: Dict[int, float] = None,
                        view_confidence: float = 0.5,
                        tau: float = 0.05,
                        weight_bounds: Tuple = (0, 0.1)) -> np.ndarray:
        """
        Black-Litterman模型

        Args:
            market_weights: 市场均衡权重（通常按市值加权）
            views: 观点字典 {asset_index: expected_return}
            view_confidence: 观点置信度
            tau: 不确定性缩放因子
        """
        n = self.n_assets

        # 计算隐含均衡收益
        risk_aversion = 2.5  # 风险厌恶系数
        implied_returns = risk_aversion * np.dot(self.cov_matrix, market_weights)

        if views is None or len(views) == 0:
            # 没有观点时，使用均衡权重
            combined_returns = implied_returns
        else:
            # 构建观点矩阵P和观点向量Q
            n_views = len(views)
            P = np.zeros((n_views, n))
            Q = np.zeros(n_views)

            for i, (asset_idx, expected_return) in enumerate(views.items()):
                P[i, asset_idx] = 1
                Q[i] = expected_return

            # 观点不确定性矩阵
            Omega = np.diag([view_confidence] * n_views)

            # Black-Litterman公式
            tau_cov = tau * self.cov_matrix

            # 计算后验收益
            inv_tau_cov = np.linalg.inv(tau_cov)
            inv_omega = np.linalg.inv(Omega)

            # BL公式
            M1 = np.linalg.inv(inv_tau_cov + np.dot(P.T, np.dot(inv_omega, P)))
            M2 = np.dot(inv_tau_cov, implied_returns) + np.dot(P.T, np.dot(inv_omega, Q))
            combined_returns = np.dot(M1, M2)

        # 使用后验收益进行均值-方差优化
        def neg_utility(weights):
            ret = np.dot(weights, combined_returns)
            vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            return -(ret - risk_aversion * vol * vol / 2)

        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple(weight_bounds for _ in range(n))
        init_weights = market_weights.copy()

        result = sco.minimize(
            neg_utility,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )

        return result.x if result.success else market_weights


class BacktestEngine:
    """回测引擎"""

    def __init__(self, returns: pd.DataFrame, risk_free_rate: float = 0.02):
        self.returns = returns
        self.risk_free_rate = risk_free_rate
        self.daily_rf = risk_free_rate / 252

    def backtest(self,
                 weights_history: Dict[str, np.ndarray],
                 rebalance_dates: List[str]) -> pd.DataFrame:
        """
        回测组合表现

        Args:
            weights_history: {date: weights} 字典
            rebalance_dates: 调仓日期列表
        """
        # 获取所有交易日
        all_dates = self.returns.index.tolist()

        # 初始化
        portfolio_values = [1.0]
        current_weights = None
        current_date_idx = 0

        rebalance_idx = 0

        for i, date in enumerate(all_dates[:-1]):
            # 检查是否需要调仓
            if rebalance_idx < len(rebalance_dates) and date >= rebalance_dates[rebalance_idx]:
                current_weights = weights_history.get(rebalance_dates[rebalance_idx])
                rebalance_idx += 1

            if current_weights is None:
                current_weights = np.ones(self.returns.shape[1]) / self.returns.shape[1]

            # 计算当日收益
            daily_return = np.dot(current_weights, self.returns.iloc[i+1].fillna(0))
            portfolio_values.append(portfolio_values[-1] * (1 + daily_return))

        # 创建结果DataFrame
        result = pd.DataFrame({
            'date': all_dates,
            'portfolio_value': portfolio_values
        })
        result['date'] = pd.to_datetime(result['date'])
        result.set_index('date', inplace=True)

        return result

    def calculate_metrics(self, portfolio_values: pd.DataFrame) -> Dict:
        """计算组合指标"""
        values = portfolio_values['portfolio_value']
        returns = values.pct_change().dropna()

        # 年化收益
        total_return = values.iloc[-1] / values.iloc[0] - 1
        years = len(values) / 252
        annual_return = (1 + total_return) ** (1/years) - 1

        # 年化波动率
        annual_vol = returns.std() * np.sqrt(252)

        # 夏普比率
        sharpe = (annual_return - self.risk_free_rate) / annual_vol if annual_vol > 0 else 0

        # 最大回撤
        cummax = values.cummax()
        drawdown = (values - cummax) / cummax
        max_drawdown = drawdown.min()

        # Calmar比率
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # 胜率
        win_rate = (returns > 0).sum() / len(returns)

        # 盈亏比
        avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0
        avg_loss = abs(returns[returns < 0].mean()) if (returns < 0).any() else 1
        profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar,
            'win_rate': win_rate,
            'profit_loss_ratio': profit_loss_ratio
        }


class PortfolioResearch:
    """投资组合优化研究"""

    def __init__(self, db_path: str):
        self.loader = DataLoader(db_path)
        self.results = {}

    def run_research(self,
                     end_date: str = '20251231',
                     lookback_years: int = 3,
                     test_years: int = 1,
                     n_stocks: int = 100):
        """
        运行完整研究

        Args:
            end_date: 研究截止日期
            lookback_years: 样本内回看年数
            test_years: 样本外测试年数
            n_stocks: 股票池数量
        """
        print("=" * 60)
        print("投资组合优化研究")
        print("=" * 60)

        # 计算日期
        end_dt = datetime.strptime(end_date, '%Y%m%d')
        test_start_dt = end_dt - timedelta(days=test_years * 365)
        train_start_dt = test_start_dt - timedelta(days=lookback_years * 365)

        train_start = train_start_dt.strftime('%Y%m%d')
        test_start = test_start_dt.strftime('%Y%m%d')

        print(f"\n训练期: {train_start} - {test_start}")
        print(f"测试期: {test_start} - {end_date}")

        # 1. 获取股票池
        print("\n[1] 获取股票池...")
        stock_pool = self.loader.get_stock_pool(test_start, min_mv=30, max_stocks=n_stocks)
        print(f"  股票池数量: {len(stock_pool)}")

        if len(stock_pool) < 20:
            print("  警告：股票池数量不足，尝试放宽条件...")
            stock_pool = self.loader.get_stock_pool(test_start, min_mv=10, max_stocks=n_stocks)
            print(f"  调整后股票池数量: {len(stock_pool)}")

        ts_codes = stock_pool['ts_code'].tolist()

        # 2. 获取价格数据
        print("\n[2] 获取价格数据...")
        price_data = self.loader.get_price_data(ts_codes, train_start, end_date)
        adj_data = self.loader.get_adj_factor(ts_codes, train_start, end_date)

        print(f"  价格数据行数: {len(price_data)}")
        print(f"  日期范围: {price_data['trade_date'].min()} - {price_data['trade_date'].max()}")

        # 3. 计算收益率
        print("\n[3] 计算收益率...")
        returns = ReturnCalculator.calculate_returns(price_data, adj_data)

        # 过滤缺失值过多的股票
        valid_stocks = returns.columns[returns.isna().sum() < len(returns) * 0.1]
        returns = returns[valid_stocks].dropna()
        print(f"  有效股票数: {len(valid_stocks)}")
        print(f"  收益率矩阵形状: {returns.shape}")

        # 分割训练集和测试集
        train_returns = returns[returns.index < test_start]
        test_returns = returns[returns.index >= test_start]

        print(f"  训练集: {train_returns.shape}")
        print(f"  测试集: {test_returns.shape}")

        # 4. 运行优化方法
        print("\n[4] 运行优化方法...")

        optimizer = PortfolioOptimizer(train_returns)

        # 计算市场权重（按等权）
        market_weights = np.ones(len(valid_stocks)) / len(valid_stocks)

        # 各种优化方法
        methods = {}

        # 4.1 均值-方差优化（最大夏普）
        print("  - 均值-方差优化（最大夏普）...")
        methods['MVO_Sharpe'] = optimizer.mean_variance_optimize(target='sharpe')

        # 4.2 均值-方差优化（最小方差）
        print("  - 均值-方差优化（最小方差）...")
        methods['MVO_MinVol'] = optimizer.mean_variance_optimize(target='min_vol')

        # 4.3 风险平价
        print("  - 风险平价...")
        methods['Risk_Parity'] = optimizer.risk_parity()

        # 4.4 最大分散化
        print("  - 最大分散化...")
        methods['Max_Diversification'] = optimizer.max_diversification()

        # 4.5 Black-Litterman（无观点）
        print("  - Black-Litterman...")
        methods['Black_Litterman'] = optimizer.black_litterman(market_weights)

        # 4.6 等权重基准
        methods['Equal_Weight'] = market_weights.copy()

        # 5. 样本内表现
        print("\n[5] 样本内表现分析...")
        train_performance = {}
        for name, weights in methods.items():
            ret, vol, sharpe = optimizer.portfolio_performance(weights)
            train_performance[name] = {
                'return': ret,
                'volatility': vol,
                'sharpe': sharpe,
                'effective_n': 1 / np.sum(weights ** 2)  # 有效资产数
            }
            print(f"  {name:25s}: 收益={ret:.2%}, 波动={vol:.2%}, 夏普={sharpe:.2f}")

        # 6. 样本外回测
        print("\n[6] 样本外回测...")
        backtest_engine = BacktestEngine(test_returns)

        test_performance = {}
        portfolio_values = {}

        for name, weights in methods.items():
            # 创建权重历史（简化：使用固定权重）
            weights_history = {test_start: weights}
            rebalance_dates = [test_start]

            # 运行回测
            pv = backtest_engine.backtest(weights_history, rebalance_dates)
            portfolio_values[name] = pv

            # 计算指标
            metrics = backtest_engine.calculate_metrics(pv)
            test_performance[name] = metrics

            print(f"  {name:25s}: 收益={metrics['annual_return']:.2%}, "
                  f"波动={metrics['annual_volatility']:.2%}, 夏普={metrics['sharpe_ratio']:.2f}, "
                  f"回撤={metrics['max_drawdown']:.2%}")

        # 7. 稳健性分析
        print("\n[7] 稳健性分析...")
        robustness_results = self._robustness_analysis(train_returns, valid_stocks)

        # 8. 生成报告
        print("\n[8] 生成报告...")
        self.results = {
            'stock_pool': stock_pool,
            'train_returns': train_returns,
            'test_returns': test_returns,
            'methods': methods,
            'train_performance': train_performance,
            'test_performance': test_performance,
            'portfolio_values': portfolio_values,
            'robustness': robustness_results,
            'valid_stocks': valid_stocks.tolist(),
            'dates': {
                'train_start': train_start,
                'test_start': test_start,
                'end_date': end_date
            }
        }

        self._generate_report()
        self._generate_plots()

        print("\n研究完成!")
        return self.results

    def _robustness_analysis(self, returns: pd.DataFrame, stocks: pd.Index) -> Dict:
        """稳健性分析"""
        n_bootstrap = 50
        n_samples = len(returns)

        robustness = {method: [] for method in ['MVO_Sharpe', 'Risk_Parity', 'Max_Diversification']}

        for i in range(n_bootstrap):
            # Bootstrap采样
            sample_idx = np.random.choice(n_samples, n_samples, replace=True)
            sample_returns = returns.iloc[sample_idx]

            optimizer = PortfolioOptimizer(sample_returns)

            # 运行各方法
            w_mvo = optimizer.mean_variance_optimize(target='sharpe')
            w_rp = optimizer.risk_parity()
            w_md = optimizer.max_diversification()

            robustness['MVO_Sharpe'].append(w_mvo)
            robustness['Risk_Parity'].append(w_rp)
            robustness['Max_Diversification'].append(w_md)

        # 计算权重稳定性
        stability = {}
        for method, weights_list in robustness.items():
            weights_array = np.array(weights_list)
            mean_weights = weights_array.mean(axis=0)
            std_weights = weights_array.std(axis=0)

            # 权重变异系数
            cv = np.mean(std_weights / (mean_weights + 1e-10))
            stability[method] = {
                'mean_weights': mean_weights,
                'std_weights': std_weights,
                'coefficient_of_variation': cv
            }

        return stability

    def _generate_report(self):
        """生成研究报告"""
        report = []
        report.append("# 投资组合优化研究报告")
        report.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        report.append("\n## 1. 研究概述")
        report.append(f"\n### 1.1 研究期间")
        report.append(f"- 训练期: {self.results['dates']['train_start']} - {self.results['dates']['test_start']}")
        report.append(f"- 测试期: {self.results['dates']['test_start']} - {self.results['dates']['end_date']}")
        report.append(f"- 股票池数量: {len(self.results['valid_stocks'])}")

        report.append("\n### 1.2 优化方法")
        report.append("""
| 方法 | 描述 |
|------|------|
| MVO_Sharpe | 均值-方差优化，最大化夏普比率 |
| MVO_MinVol | 均值-方差优化，最小化波动率 |
| Risk_Parity | 风险平价，使各资产风险贡献相等 |
| Max_Diversification | 最大分散化，最大化分散化比率 |
| Black_Litterman | Black-Litterman模型，结合市场均衡与投资者观点 |
| Equal_Weight | 等权重基准 |
""")

        report.append("\n## 2. 样本内表现")
        report.append("\n| 方法 | 年化收益 | 年化波动 | 夏普比率 | 有效资产数 |")
        report.append("|------|----------|----------|----------|------------|")
        for name, perf in self.results['train_performance'].items():
            report.append(f"| {name} | {perf['return']:.2%} | {perf['volatility']:.2%} | "
                         f"{perf['sharpe']:.2f} | {perf['effective_n']:.1f} |")

        report.append("\n## 3. 样本外表现")
        report.append("\n| 方法 | 年化收益 | 年化波动 | 夏普比率 | 最大回撤 | Calmar比率 | 胜率 |")
        report.append("|------|----------|----------|----------|----------|------------|------|")
        for name, perf in self.results['test_performance'].items():
            report.append(f"| {name} | {perf['annual_return']:.2%} | {perf['annual_volatility']:.2%} | "
                         f"{perf['sharpe_ratio']:.2f} | {perf['max_drawdown']:.2%} | "
                         f"{perf['calmar_ratio']:.2f} | {perf['win_rate']:.2%} |")

        report.append("\n## 4. 稳健性分析")
        report.append("\n### 4.1 权重稳定性（Bootstrap分析）")
        report.append("\n| 方法 | 权重变异系数 | 稳定性评级 |")
        report.append("|------|--------------|------------|")
        for method, stats in self.results['robustness'].items():
            cv = stats['coefficient_of_variation']
            rating = "高" if cv < 0.3 else ("中" if cv < 0.6 else "低")
            report.append(f"| {method} | {cv:.3f} | {rating} |")

        report.append("\n## 5. 约束条件影响分析")
        report.append("""
### 5.1 个股权重约束
- 本研究采用个股权重上限为10%的约束
- 该约束有效防止了组合过度集中于少数股票
- 权重下限为0（不允许做空）

### 5.2 行业约束（待实现）
- 建议单一行业权重不超过30%
- 可根据基准指数设定行业偏离度上限

### 5.3 换手率约束（待实现）
- 建议单次调仓换手率不超过50%
- 可降低交易成本和市场冲击
""")

        report.append("\n## 6. 结论与建议")

        # 找出最佳方法
        best_sharpe = max(self.results['test_performance'].items(),
                         key=lambda x: x[1]['sharpe_ratio'])
        best_return = max(self.results['test_performance'].items(),
                         key=lambda x: x[1]['annual_return'])
        best_drawdown = max(self.results['test_performance'].items(),
                           key=lambda x: x[1]['max_drawdown'])  # 回撤为负，max即最小回撤

        report.append(f"""
### 6.1 方法比较总结
- **最高夏普比率**: {best_sharpe[0]} ({best_sharpe[1]['sharpe_ratio']:.2f})
- **最高年化收益**: {best_return[0]} ({best_return[1]['annual_return']:.2%})
- **最小最大回撤**: {best_drawdown[0]} ({best_drawdown[1]['max_drawdown']:.2%})

### 6.2 投资建议
1. **风险厌恶型投资者**: 建议采用最小方差或风险平价策略
2. **收益导向型投资者**: 建议采用均值-方差优化（最大夏普）
3. **分散化偏好投资者**: 建议采用最大分散化或风险平价策略
4. **机构投资者**: 建议采用Black-Litterman模型，结合自身投资观点

### 6.3 注意事项
- 所有优化方法都依赖历史数据，未来表现可能与历史不同
- 建议定期（如月度或季度）重新优化组合权重
- 实际交易时需考虑交易成本和流动性约束
- 市场环境变化时应及时调整策略参数
""")

        report.append("\n## 7. 附录")
        report.append("\n### 7.1 方法论详解")
        report.append("""
#### 均值-方差优化（Markowitz）
$$\\max_{w} \\frac{w^T\\mu - r_f}{\\sqrt{w^T\\Sigma w}}$$

其中：
- $w$: 资产权重向量
- $\\mu$: 预期收益向量
- $\\Sigma$: 协方差矩阵
- $r_f$: 无风险利率

#### 风险平价
$$\\min_{w} \\sum_{i=1}^{n}(RC_i - \\frac{1}{n})^2$$

其中 $RC_i = w_i \\cdot \\frac{(\\Sigma w)_i}{\\sqrt{w^T\\Sigma w}}$ 为第i个资产的风险贡献。

#### 最大分散化
$$\\max_{w} \\frac{w^T\\sigma}{\\sqrt{w^T\\Sigma w}}$$

其中 $\\sigma$ 为各资产波动率向量。

#### Black-Litterman
$$E[R] = [(\\tau\\Sigma)^{-1} + P^T\\Omega^{-1}P]^{-1}[(\\tau\\Sigma)^{-1}\\Pi + P^T\\Omega^{-1}Q]$$

其中：
- $\\Pi$: 隐含均衡收益
- $P$: 观点矩阵
- $Q$: 观点收益向量
- $\\Omega$: 观点不确定性矩阵
""")

        # 保存报告
        report_path = f"{REPORT_DIR}/portfolio_optimization_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))

        print(f"  报告已保存: {report_path}")

    def _generate_plots(self):
        """生成图表"""
        # 1. 样本外净值曲线
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1.1 净值曲线
        ax = axes[0, 0]
        for name, pv in self.results['portfolio_values'].items():
            ax.plot(pv.index, pv['portfolio_value'], label=name, linewidth=1.5)
        ax.set_title('Portfolio Value (Out-of-Sample)', fontsize=12)
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)

        # 1.2 回撤曲线
        ax = axes[0, 1]
        for name, pv in self.results['portfolio_values'].items():
            values = pv['portfolio_value']
            drawdown = (values - values.cummax()) / values.cummax()
            ax.fill_between(pv.index, drawdown, 0, alpha=0.3, label=name)
        ax.set_title('Drawdown', fontsize=12)
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown')
        ax.legend(loc='lower left', fontsize=8)
        ax.grid(True, alpha=0.3)

        # 1.3 风险-收益散点图
        ax = axes[1, 0]
        for name, perf in self.results['test_performance'].items():
            ax.scatter(perf['annual_volatility'], perf['annual_return'],
                      s=100, label=name, marker='o')
            ax.annotate(name, (perf['annual_volatility'], perf['annual_return']),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        ax.set_title('Risk-Return Profile', fontsize=12)
        ax.set_xlabel('Annual Volatility')
        ax.set_ylabel('Annual Return')
        ax.grid(True, alpha=0.3)

        # 1.4 各方法指标对比
        ax = axes[1, 1]
        methods_names = list(self.results['test_performance'].keys())
        metrics = ['sharpe_ratio', 'calmar_ratio', 'win_rate']
        x = np.arange(len(methods_names))
        width = 0.25

        for i, metric in enumerate(metrics):
            values = [self.results['test_performance'][m][metric] for m in methods_names]
            ax.bar(x + i * width, values, width, label=metric)

        ax.set_title('Performance Metrics Comparison', fontsize=12)
        ax.set_xticks(x + width)
        ax.set_xticklabels(methods_names, rotation=45, ha='right', fontsize=8)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        fig_path = f"{REPORT_DIR}/portfolio_optimization_charts.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  图表已保存: {fig_path}")

        # 2. 权重分布图
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        methods_to_plot = ['MVO_Sharpe', 'MVO_MinVol', 'Risk_Parity',
                          'Max_Diversification', 'Black_Litterman', 'Equal_Weight']

        for ax, method in zip(axes.flat, methods_to_plot):
            weights = self.results['methods'][method]
            # 只显示前20个最大权重
            sorted_idx = np.argsort(weights)[::-1][:20]
            sorted_weights = weights[sorted_idx]

            ax.bar(range(len(sorted_weights)), sorted_weights)
            ax.set_title(f'{method}\n(Top 20 Weights)', fontsize=10)
            ax.set_xlabel('Asset Rank')
            ax.set_ylabel('Weight')
            ax.axhline(y=0.1, color='r', linestyle='--', alpha=0.5, label='Max Limit')

        plt.tight_layout()
        fig_path = f"{REPORT_DIR}/portfolio_weights_distribution.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  权重分布图已保存: {fig_path}")

        # 3. 稳健性分析图
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for ax, (method, stats) in zip(axes, self.results['robustness'].items()):
            mean_w = stats['mean_weights']
            std_w = stats['std_weights']

            # 只显示前20个
            sorted_idx = np.argsort(mean_w)[::-1][:20]

            ax.errorbar(range(20), mean_w[sorted_idx], yerr=std_w[sorted_idx],
                       fmt='o', capsize=3, capthick=1)
            ax.set_title(f'{method}\nCV={stats["coefficient_of_variation"]:.3f}', fontsize=10)
            ax.set_xlabel('Asset Rank')
            ax.set_ylabel('Weight (Mean ± Std)')

        plt.tight_layout()
        fig_path = f"{REPORT_DIR}/portfolio_robustness_analysis.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  稳健性分析图已保存: {fig_path}")


def main():
    """主函数"""
    import os
    os.makedirs(REPORT_DIR, exist_ok=True)

    research = PortfolioResearch(DB_PATH)

    # 运行研究
    results = research.run_research(
        end_date='20251231',
        lookback_years=3,
        test_years=1,
        n_stocks=100
    )

    print(f"\n报告已保存至: {REPORT_DIR}")


if __name__ == '__main__':
    main()
