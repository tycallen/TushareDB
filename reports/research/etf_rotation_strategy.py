#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETF轮动策略研究
================
基于申万行业指数和宽基指数进行ETF轮动策略研究

策略类型：
1. 动量轮动策略 - 买入近期表现最好的ETF
2. 均值回归策略 - 买入近期表现最差但有回归迹象的ETF
3. 多因子轮动策略 - 综合动量、波动率、估值等因子

Author: Claude AI
Date: 2026-02-01
"""

import duckdb
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 尝试设置中文字体
try:
    plt.rcParams['font.sans-serif'] = ['PingFang HK', 'Heiti TC', 'STHeiti', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass

# 数据库路径
DB_PATH = '/Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db'
REPORT_PATH = '/Users/allen/workspace/python/stock/Tushare-DuckDB/reports/research'


class ETFRotationResearch:
    """ETF轮动策略研究类"""

    # 申万一级行业代码映射（使用最新名称）
    SW_L1_INDUSTRIES = {
        '801010.SI': '农林牧渔',
        '801030.SI': '基础化工',
        '801040.SI': '钢铁',
        '801050.SI': '有色金属',
        '801080.SI': '电子',
        '801110.SI': '家用电器',
        '801120.SI': '食品饮料',
        '801130.SI': '纺织服饰',
        '801140.SI': '轻工制造',
        '801150.SI': '医药生物',
        '801160.SI': '公用事业',
        '801170.SI': '交通运输',
        '801180.SI': '房地产',
        '801200.SI': '商贸零售',
        '801210.SI': '社会服务',
        '801230.SI': '综合',
        '801710.SI': '建筑材料',
        '801720.SI': '建筑装饰',
        '801730.SI': '电力设备',
        '801740.SI': '国防军工',
        '801750.SI': '计算机',
        '801760.SI': '传媒',
        '801770.SI': '通信',
        '801780.SI': '银行',
        '801790.SI': '非银金融',
        '801880.SI': '汽车',
        '801890.SI': '机械设备',
    }

    # 宽基指数代码映射
    BROAD_INDICES = {
        '000001.SH': '上证指数',
        '000016.SH': '上证50',
        '000300.SH': '沪深300',
        '000905.SH': '中证500',
        '000852.SH': '中证1000',
        '399001.SZ': '深证成指',
        '399006.SZ': '创业板指',
        '000688.SH': '科创50',
    }

    # 主题分类（基于申万二级行业）
    THEME_CATEGORIES = {
        '新能源': ['801730.SI'],  # 电力设备
        '消费': ['801120.SI', '801110.SI', '801200.SI'],  # 食品饮料、家电、商贸
        '科技': ['801080.SI', '801750.SI', '801770.SI'],  # 电子、计算机、通信
        '金融': ['801780.SI', '801790.SI'],  # 银行、非银
        '周期': ['801030.SI', '801040.SI', '801050.SI'],  # 化工、钢铁、有色
        '医药': ['801150.SI'],  # 医药生物
    }

    def __init__(self, db_path=DB_PATH):
        """初始化"""
        self.conn = duckdb.connect(db_path, read_only=True)
        self.industry_data = None
        self.broad_index_data = None

    def load_data(self, start_date='20220101', end_date='20260130'):
        """加载数据"""
        print(f"加载数据: {start_date} - {end_date}")

        # 加载申万行业数据
        industry_codes = list(self.SW_L1_INDUSTRIES.keys())
        industry_codes_str = "', '".join(industry_codes)

        query = f"""
        SELECT ts_code, trade_date, close, pct_change, vol, amount, pe, pb, float_mv, total_mv
        FROM sw_daily
        WHERE ts_code IN ('{industry_codes_str}')
          AND trade_date >= '{start_date}'
          AND trade_date <= '{end_date}'
        ORDER BY ts_code, trade_date
        """
        self.industry_data = self.conn.execute(query).fetchdf()
        self.industry_data['trade_date'] = pd.to_datetime(self.industry_data['trade_date'])

        # 加载宽基指数数据
        broad_codes = list(self.BROAD_INDICES.keys())
        broad_codes_str = "', '".join(broad_codes)

        query = f"""
        SELECT ts_code, trade_date, close, pct_chg, vol, amount
        FROM index_daily
        WHERE ts_code IN ('{broad_codes_str}')
          AND trade_date >= '{start_date}'
          AND trade_date <= '{end_date}'
        ORDER BY ts_code, trade_date
        """
        self.broad_index_data = self.conn.execute(query).fetchdf()
        self.broad_index_data['trade_date'] = pd.to_datetime(self.broad_index_data['trade_date'])

        print(f"行业数据: {len(self.industry_data)} 条记录, {self.industry_data['ts_code'].nunique()} 个行业")
        print(f"宽基指数数据: {len(self.broad_index_data)} 条记录, {self.broad_index_data['ts_code'].nunique()} 个指数")

        return self

    def prepare_pivot_data(self, data, value_col='close', code_col='ts_code'):
        """将数据转换为宽表格式"""
        pivot = data.pivot(index='trade_date', columns=code_col, values=value_col)
        return pivot.sort_index()

    def calculate_returns(self, prices, periods=[5, 10, 20, 60]):
        """计算不同周期的收益率"""
        returns = {}
        for p in periods:
            returns[f'ret_{p}d'] = prices.pct_change(p)
        return returns

    def calculate_momentum_score(self, prices, lookback=20):
        """计算动量得分"""
        return prices.pct_change(lookback)

    def calculate_volatility(self, prices, window=20):
        """计算波动率"""
        returns = prices.pct_change()
        return returns.rolling(window=window).std() * np.sqrt(252)

    def calculate_mean_reversion_score(self, prices, short_window=5, long_window=20):
        """计算均值回归得分

        使用短期均线偏离长期均线的程度来判断
        负值表示价格低于均线，可能有回归机会
        """
        ma_short = prices.rolling(window=short_window).mean()
        ma_long = prices.rolling(window=long_window).mean()
        deviation = (ma_short - ma_long) / ma_long
        return -deviation  # 取负值，偏离越大（价格越低），得分越高

    def calculate_rsi(self, prices, window=14):
        """计算RSI指标"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)

        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def momentum_rotation_strategy(self, prices, lookback=20, hold_period=20,
                                    top_n=3, transaction_cost=0.001):
        """
        动量轮动策略

        策略逻辑：
        - 每个调仓周期选择过去lookback天收益率最高的top_n个标的
        - 等权重持有hold_period天

        Parameters:
        -----------
        prices : DataFrame
            价格数据，行为日期，列为标的代码
        lookback : int
            动量计算回溯期
        hold_period : int
            持仓周期
        top_n : int
            选择标的数量
        transaction_cost : float
            单边交易成本

        Returns:
        --------
        dict : 包含策略结果的字典
        """
        # 计算动量
        momentum = prices.pct_change(lookback)

        # 初始化
        dates = prices.index[lookback:]
        portfolio_value = [1.0]
        holdings = {}
        rebalance_dates = []
        selected_history = []
        turnover_list = []

        last_rebalance = None

        for i, date in enumerate(dates):
            if i == 0:
                # 首次建仓
                mom_scores = momentum.loc[date].dropna()
                if len(mom_scores) < top_n:
                    portfolio_value.append(portfolio_value[-1])
                    continue

                top_assets = mom_scores.nlargest(top_n).index.tolist()
                holdings = {asset: 1.0/top_n for asset in top_assets}
                last_rebalance = date
                rebalance_dates.append(date)
                selected_history.append({'date': date, 'assets': top_assets, 'scores': mom_scores[top_assets].to_dict()})

                # 扣除建仓成本
                cost = transaction_cost * 2  # 买入成本
                portfolio_value.append(portfolio_value[-1] * (1 - cost))
                continue

            # 计算当日收益
            daily_return = 0
            for asset, weight in holdings.items():
                if asset in prices.columns:
                    prev_price = prices.loc[dates[i-1], asset]
                    curr_price = prices.loc[date, asset]
                    if pd.notna(prev_price) and pd.notna(curr_price) and prev_price > 0:
                        daily_return += weight * (curr_price / prev_price - 1)

            new_value = portfolio_value[-1] * (1 + daily_return)

            # 检查是否需要调仓
            days_since_rebalance = (date - last_rebalance).days
            if days_since_rebalance >= hold_period:
                mom_scores = momentum.loc[date].dropna()
                if len(mom_scores) >= top_n:
                    new_top_assets = mom_scores.nlargest(top_n).index.tolist()

                    # 计算换手率
                    old_assets = set(holdings.keys())
                    new_assets = set(new_top_assets)
                    turnover = len(old_assets - new_assets) / top_n
                    turnover_list.append(turnover)

                    # 扣除交易成本
                    cost = transaction_cost * 2 * turnover
                    new_value = new_value * (1 - cost)

                    # 更新持仓
                    holdings = {asset: 1.0/top_n for asset in new_top_assets}
                    last_rebalance = date
                    rebalance_dates.append(date)
                    selected_history.append({'date': date, 'assets': new_top_assets, 'scores': mom_scores[new_top_assets].to_dict()})

            portfolio_value.append(new_value)

        # 构建结果
        portfolio_series = pd.Series(portfolio_value[1:], index=dates)

        return {
            'portfolio_value': portfolio_series,
            'rebalance_dates': rebalance_dates,
            'selected_history': selected_history,
            'avg_turnover': np.mean(turnover_list) if turnover_list else 0,
            'total_rebalances': len(rebalance_dates),
        }

    def mean_reversion_strategy(self, prices, lookback=20, hold_period=20,
                                 top_n=3, transaction_cost=0.001, rsi_threshold=30):
        """
        均值回归策略

        策略逻辑：
        - 选择RSI低于阈值且价格偏离均线较多的标的
        - 预期价格会回归均值

        Parameters:
        -----------
        prices : DataFrame
            价格数据
        lookback : int
            均值计算窗口
        hold_period : int
            持仓周期
        top_n : int
            选择标的数量
        transaction_cost : float
            单边交易成本
        rsi_threshold : float
            RSI阈值，低于此值认为超卖

        Returns:
        --------
        dict : 包含策略结果的字典
        """
        # 计算均值回归得分
        mr_score = self.calculate_mean_reversion_score(prices, short_window=5, long_window=lookback)
        rsi = self.calculate_rsi(prices, window=14)

        # 初始化
        start_idx = max(lookback, 14)
        dates = prices.index[start_idx:]
        portfolio_value = [1.0]
        holdings = {}
        rebalance_dates = []
        selected_history = []
        turnover_list = []

        last_rebalance = None

        for i, date in enumerate(dates):
            if i == 0:
                # 首次建仓：选择超卖且回归得分高的
                mr_today = mr_score.loc[date].dropna()
                rsi_today = rsi.loc[date].dropna()

                # 结合RSI过滤
                combined_score = mr_today.copy()
                for asset in combined_score.index:
                    if asset in rsi_today.index and rsi_today[asset] > rsi_threshold + 20:
                        combined_score[asset] = -999  # 排除RSI过高的

                if len(combined_score[combined_score > -999]) < top_n:
                    portfolio_value.append(portfolio_value[-1])
                    continue

                top_assets = combined_score.nlargest(top_n).index.tolist()
                holdings = {asset: 1.0/top_n for asset in top_assets}
                last_rebalance = date
                rebalance_dates.append(date)
                selected_history.append({'date': date, 'assets': top_assets})

                cost = transaction_cost * 2
                portfolio_value.append(portfolio_value[-1] * (1 - cost))
                continue

            # 计算当日收益
            daily_return = 0
            for asset, weight in holdings.items():
                if asset in prices.columns:
                    prev_price = prices.loc[dates[i-1], asset]
                    curr_price = prices.loc[date, asset]
                    if pd.notna(prev_price) and pd.notna(curr_price) and prev_price > 0:
                        daily_return += weight * (curr_price / prev_price - 1)

            new_value = portfolio_value[-1] * (1 + daily_return)

            # 检查是否需要调仓
            days_since_rebalance = (date - last_rebalance).days
            if days_since_rebalance >= hold_period:
                mr_today = mr_score.loc[date].dropna()
                rsi_today = rsi.loc[date].dropna()

                combined_score = mr_today.copy()
                for asset in combined_score.index:
                    if asset in rsi_today.index and rsi_today[asset] > rsi_threshold + 20:
                        combined_score[asset] = -999

                if len(combined_score[combined_score > -999]) >= top_n:
                    new_top_assets = combined_score.nlargest(top_n).index.tolist()

                    old_assets = set(holdings.keys())
                    new_assets = set(new_top_assets)
                    turnover = len(old_assets - new_assets) / top_n
                    turnover_list.append(turnover)

                    cost = transaction_cost * 2 * turnover
                    new_value = new_value * (1 - cost)

                    holdings = {asset: 1.0/top_n for asset in new_top_assets}
                    last_rebalance = date
                    rebalance_dates.append(date)
                    selected_history.append({'date': date, 'assets': new_top_assets})

            portfolio_value.append(new_value)

        portfolio_series = pd.Series(portfolio_value[1:], index=dates)

        return {
            'portfolio_value': portfolio_series,
            'rebalance_dates': rebalance_dates,
            'selected_history': selected_history,
            'avg_turnover': np.mean(turnover_list) if turnover_list else 0,
            'total_rebalances': len(rebalance_dates),
        }

    def multi_factor_strategy(self, prices, pe_data=None, pb_data=None,
                               lookback=20, hold_period=20, top_n=3,
                               transaction_cost=0.001):
        """
        多因子轮动策略

        因子：
        1. 动量因子（权重40%）
        2. 波动率因子（权重20%，低波动优先）
        3. 估值因子（权重20%，低PE/PB优先）
        4. 均值回归因子（权重20%）

        Parameters:
        -----------
        prices : DataFrame
            价格数据
        pe_data : DataFrame
            PE数据（可选）
        pb_data : DataFrame
            PB数据（可选）
        lookback : int
            因子计算窗口
        hold_period : int
            持仓周期
        top_n : int
            选择标的数量
        transaction_cost : float
            单边交易成本

        Returns:
        --------
        dict : 包含策略结果的字典
        """
        # 计算各因子
        momentum = self.calculate_momentum_score(prices, lookback)
        volatility = self.calculate_volatility(prices, lookback)
        mr_score = self.calculate_mean_reversion_score(prices, 5, lookback)

        # 标准化因子（截面标准化）
        def cross_sectional_rank(df):
            return df.rank(axis=1, pct=True)

        mom_rank = cross_sectional_rank(momentum)
        vol_rank = 1 - cross_sectional_rank(volatility)  # 低波动率得分高
        mr_rank = cross_sectional_rank(mr_score)

        # 处理估值因子
        if pe_data is not None:
            pe_rank = 1 - cross_sectional_rank(pe_data)  # 低PE得分高
        else:
            pe_rank = pd.DataFrame(0.5, index=prices.index, columns=prices.columns)

        if pb_data is not None:
            pb_rank = 1 - cross_sectional_rank(pb_data)
        else:
            pb_rank = pd.DataFrame(0.5, index=prices.index, columns=prices.columns)

        # 综合估值得分
        val_rank = (pe_rank + pb_rank) / 2

        # 计算综合得分
        composite_score = (
            0.4 * mom_rank +
            0.2 * vol_rank +
            0.2 * val_rank +
            0.2 * mr_rank
        )

        # 初始化
        start_idx = lookback
        dates = prices.index[start_idx:]
        portfolio_value = [1.0]
        holdings = {}
        rebalance_dates = []
        selected_history = []
        turnover_list = []
        factor_contributions = []

        last_rebalance = None

        for i, date in enumerate(dates):
            if i == 0:
                scores = composite_score.loc[date].dropna()
                if len(scores) < top_n:
                    portfolio_value.append(portfolio_value[-1])
                    continue

                top_assets = scores.nlargest(top_n).index.tolist()
                holdings = {asset: 1.0/top_n for asset in top_assets}
                last_rebalance = date
                rebalance_dates.append(date)

                # 记录因子贡献
                factor_contrib = {
                    'date': date,
                    'assets': top_assets,
                    'composite_scores': scores[top_assets].to_dict(),
                    'momentum_scores': mom_rank.loc[date][top_assets].to_dict() if date in mom_rank.index else {},
                    'volatility_scores': vol_rank.loc[date][top_assets].to_dict() if date in vol_rank.index else {},
                }
                factor_contributions.append(factor_contrib)
                selected_history.append({'date': date, 'assets': top_assets})

                cost = transaction_cost * 2
                portfolio_value.append(portfolio_value[-1] * (1 - cost))
                continue

            daily_return = 0
            for asset, weight in holdings.items():
                if asset in prices.columns:
                    prev_price = prices.loc[dates[i-1], asset]
                    curr_price = prices.loc[date, asset]
                    if pd.notna(prev_price) and pd.notna(curr_price) and prev_price > 0:
                        daily_return += weight * (curr_price / prev_price - 1)

            new_value = portfolio_value[-1] * (1 + daily_return)

            days_since_rebalance = (date - last_rebalance).days
            if days_since_rebalance >= hold_period:
                scores = composite_score.loc[date].dropna()
                if len(scores) >= top_n:
                    new_top_assets = scores.nlargest(top_n).index.tolist()

                    old_assets = set(holdings.keys())
                    new_assets = set(new_top_assets)
                    turnover = len(old_assets - new_assets) / top_n
                    turnover_list.append(turnover)

                    cost = transaction_cost * 2 * turnover
                    new_value = new_value * (1 - cost)

                    holdings = {asset: 1.0/top_n for asset in new_top_assets}
                    last_rebalance = date
                    rebalance_dates.append(date)
                    selected_history.append({'date': date, 'assets': new_top_assets})

                    factor_contrib = {
                        'date': date,
                        'assets': new_top_assets,
                        'composite_scores': scores[new_top_assets].to_dict(),
                    }
                    factor_contributions.append(factor_contrib)

            portfolio_value.append(new_value)

        portfolio_series = pd.Series(portfolio_value[1:], index=dates)

        return {
            'portfolio_value': portfolio_series,
            'rebalance_dates': rebalance_dates,
            'selected_history': selected_history,
            'factor_contributions': factor_contributions,
            'avg_turnover': np.mean(turnover_list) if turnover_list else 0,
            'total_rebalances': len(rebalance_dates),
        }

    def calculate_performance_metrics(self, portfolio_series, benchmark_series=None):
        """计算策略绩效指标"""
        returns = portfolio_series.pct_change().dropna()

        # 年化收益率
        total_return = portfolio_series.iloc[-1] / portfolio_series.iloc[0] - 1
        years = len(returns) / 252
        annual_return = (1 + total_return) ** (1/years) - 1

        # 年化波动率
        annual_volatility = returns.std() * np.sqrt(252)

        # 夏普比率（假设无风险利率2%）
        risk_free_rate = 0.02
        sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility > 0 else 0

        # 最大回撤
        cummax = portfolio_series.cummax()
        drawdown = (portfolio_series - cummax) / cummax
        max_drawdown = drawdown.min()

        # 卡尔玛比率
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # 胜率（正收益天数占比）
        win_rate = (returns > 0).sum() / len(returns)

        # 盈亏比
        avg_win = returns[returns > 0].mean() if (returns > 0).sum() > 0 else 0
        avg_loss = abs(returns[returns < 0].mean()) if (returns < 0).sum() > 0 else 0
        profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0

        metrics = {
            '总收益率': f'{total_return:.2%}',
            '年化收益率': f'{annual_return:.2%}',
            '年化波动率': f'{annual_volatility:.2%}',
            '夏普比率': f'{sharpe_ratio:.2f}',
            '最大回撤': f'{max_drawdown:.2%}',
            '卡尔玛比率': f'{calmar_ratio:.2f}',
            '胜率': f'{win_rate:.2%}',
            '盈亏比': f'{profit_loss_ratio:.2f}',
        }

        # 如果有基准，计算超额收益
        if benchmark_series is not None:
            # 对齐时间
            common_index = portfolio_series.index.intersection(benchmark_series.index)
            if len(common_index) > 0:
                port_aligned = portfolio_series.loc[common_index]
                bench_aligned = benchmark_series.loc[common_index]

                bench_return = bench_aligned.iloc[-1] / bench_aligned.iloc[0] - 1
                excess_return = total_return - bench_return

                # 信息比率
                active_returns = port_aligned.pct_change() - bench_aligned.pct_change()
                active_returns = active_returns.dropna()
                tracking_error = active_returns.std() * np.sqrt(252)
                information_ratio = (annual_return - (bench_return / years)) / tracking_error if tracking_error > 0 else 0

                metrics['基准收益率'] = f'{bench_return:.2%}'
                metrics['超额收益'] = f'{excess_return:.2%}'
                metrics['信息比率'] = f'{information_ratio:.2f}'

        return metrics

    def run_backtest_comparison(self, prices, pe_data=None, pb_data=None,
                                 periods=[10, 20, 40, 60], top_n=3):
        """运行不同周期的回测比较"""
        results = {}

        for period in periods:
            print(f"\n回测周期: {period}天")

            # 动量策略
            mom_result = self.momentum_rotation_strategy(
                prices, lookback=period, hold_period=period, top_n=top_n
            )

            # 均值回归策略
            mr_result = self.mean_reversion_strategy(
                prices, lookback=period, hold_period=period, top_n=top_n
            )

            # 多因子策略
            mf_result = self.multi_factor_strategy(
                prices, pe_data, pb_data, lookback=period, hold_period=period, top_n=top_n
            )

            results[period] = {
                'momentum': mom_result,
                'mean_reversion': mr_result,
                'multi_factor': mf_result,
            }

        return results

    def analyze_industry_data(self):
        """分析行业ETF数据"""
        if self.industry_data is None:
            raise ValueError("请先加载数据")

        analysis = {}

        # 按行业分组分析
        for code, name in self.SW_L1_INDUSTRIES.items():
            data = self.industry_data[self.industry_data['ts_code'] == code].copy()
            if len(data) == 0:
                continue

            data = data.sort_values('trade_date')

            # 计算收益率
            total_return = data['close'].iloc[-1] / data['close'].iloc[0] - 1

            # 计算波动率
            returns = data['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)

            # 计算最大回撤
            cummax = data['close'].cummax()
            drawdown = (data['close'] - cummax) / cummax
            max_drawdown = drawdown.min()

            # 平均估值
            avg_pe = data['pe'].mean()
            avg_pb = data['pb'].mean()

            analysis[code] = {
                'name': name,
                'total_return': total_return,
                'volatility': volatility,
                'max_drawdown': max_drawdown,
                'avg_pe': avg_pe,
                'avg_pb': avg_pb,
                'data_points': len(data),
            }

        return pd.DataFrame(analysis).T

    def plot_strategy_comparison(self, results, benchmark=None, save_path=None):
        """绘制策略比较图"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        colors = {'momentum': 'blue', 'mean_reversion': 'green', 'multi_factor': 'red'}
        labels = {'momentum': '动量策略', 'mean_reversion': '均值回归', 'multi_factor': '多因子'}

        for idx, (period, strats) in enumerate(results.items()):
            ax = axes[idx // 2, idx % 2]

            for strat_name, strat_result in strats.items():
                pv = strat_result['portfolio_value']
                ax.plot(pv.index, pv.values, color=colors[strat_name],
                       label=labels[strat_name], linewidth=1.5)

            if benchmark is not None:
                # 对齐基准数据
                bench_aligned = benchmark.reindex(strats['momentum']['portfolio_value'].index)
                bench_normalized = bench_aligned / bench_aligned.iloc[0]
                ax.plot(bench_normalized.index, bench_normalized.values,
                       color='gray', linestyle='--', label='基准(沪深300)', linewidth=1)

            ax.set_title(f'持仓周期: {period}天', fontsize=12)
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
            ax.set_ylabel('净值')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"图表已保存至: {save_path}")

        plt.close()

    def generate_report(self, save_path=None):
        """生成完整研究报告"""
        if self.industry_data is None:
            self.load_data()

        report = []
        report.append("=" * 80)
        report.append("ETF轮动策略研究报告")
        report.append("=" * 80)
        report.append(f"\n报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"数据范围: {self.industry_data['trade_date'].min().strftime('%Y-%m-%d')} - {self.industry_data['trade_date'].max().strftime('%Y-%m-%d')}")

        # 1. ETF数据分析
        report.append("\n" + "=" * 80)
        report.append("一、ETF数据分析")
        report.append("=" * 80)

        # 1.1 行业ETF分析
        report.append("\n1.1 行业ETF分析（基于申万一级行业指数）")
        report.append("-" * 60)

        industry_analysis = self.analyze_industry_data()
        industry_analysis = industry_analysis.sort_values('total_return', ascending=False)

        report.append("\n行业表现排名（按总收益率排序）:")
        report.append("-" * 60)
        for code, row in industry_analysis.iterrows():
            report.append(f"  {row['name']:8s}: 收益{row['total_return']:7.2%}, "
                         f"波动{row['volatility']:7.2%}, 回撤{row['max_drawdown']:7.2%}, "
                         f"PE:{row['avg_pe']:6.1f}, PB:{row['avg_pb']:5.2f}")

        # 1.2 宽基ETF分析
        report.append("\n\n1.2 宽基ETF分析")
        report.append("-" * 60)

        for code, name in self.BROAD_INDICES.items():
            data = self.broad_index_data[self.broad_index_data['ts_code'] == code]
            if len(data) == 0:
                continue
            data = data.sort_values('trade_date')
            total_return = data['close'].iloc[-1] / data['close'].iloc[0] - 1
            returns = data['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)
            report.append(f"  {name:8s}: 收益{total_return:7.2%}, 波动{volatility:7.2%}")

        # 1.3 主题分类
        report.append("\n\n1.3 主题ETF分类")
        report.append("-" * 60)

        for theme, codes in self.THEME_CATEGORIES.items():
            theme_returns = []
            for code in codes:
                if code in industry_analysis.index:
                    theme_returns.append(industry_analysis.loc[code, 'total_return'])
            if theme_returns:
                avg_return = np.mean(theme_returns)
                report.append(f"  {theme:6s}: 平均收益{avg_return:7.2%}, 包含行业: {', '.join([self.SW_L1_INDUSTRIES.get(c, c) for c in codes])}")

        # 2. 轮动策略回测
        report.append("\n\n" + "=" * 80)
        report.append("二、轮动策略研究")
        report.append("=" * 80)

        # 准备价格数据
        prices = self.prepare_pivot_data(self.industry_data, 'close')
        pe_data = self.prepare_pivot_data(self.industry_data, 'pe')
        pb_data = self.prepare_pivot_data(self.industry_data, 'pb')

        # 准备基准数据
        hs300 = self.broad_index_data[self.broad_index_data['ts_code'] == '000300.SH'].copy()
        hs300 = hs300.set_index('trade_date')['close']

        # 运行不同周期回测
        periods = [10, 20, 40, 60]
        backtest_results = self.run_backtest_comparison(prices, pe_data, pb_data, periods)

        # 2.1 动量轮动策略
        report.append("\n2.1 动量轮动策略")
        report.append("-" * 60)
        report.append("策略逻辑: 选择过去N天收益率最高的前3个行业，等权持有")
        report.append("\n不同持仓周期表现:")

        for period in periods:
            mom_result = backtest_results[period]['momentum']
            metrics = self.calculate_performance_metrics(mom_result['portfolio_value'], hs300)
            report.append(f"\n  周期{period}天:")
            for key, value in metrics.items():
                report.append(f"    {key}: {value}")
            report.append(f"    平均换手率: {mom_result['avg_turnover']:.2%}")
            report.append(f"    调仓次数: {mom_result['total_rebalances']}")

        # 2.2 均值回归策略
        report.append("\n\n2.2 均值回归轮动策略")
        report.append("-" * 60)
        report.append("策略逻辑: 选择价格偏离均线较多且RSI较低的行业，预期均值回归")
        report.append("\n不同持仓周期表现:")

        for period in periods:
            mr_result = backtest_results[period]['mean_reversion']
            metrics = self.calculate_performance_metrics(mr_result['portfolio_value'], hs300)
            report.append(f"\n  周期{period}天:")
            for key, value in metrics.items():
                report.append(f"    {key}: {value}")
            report.append(f"    平均换手率: {mr_result['avg_turnover']:.2%}")

        # 2.3 多因子轮动策略
        report.append("\n\n2.3 多因子轮动策略")
        report.append("-" * 60)
        report.append("策略逻辑: 综合动量(40%)、波动率(20%)、估值(20%)、均值回归(20%)因子")
        report.append("\n不同持仓周期表现:")

        for period in periods:
            mf_result = backtest_results[period]['multi_factor']
            metrics = self.calculate_performance_metrics(mf_result['portfolio_value'], hs300)
            report.append(f"\n  周期{period}天:")
            for key, value in metrics.items():
                report.append(f"    {key}: {value}")
            report.append(f"    平均换手率: {mf_result['avg_turnover']:.2%}")

        # 3. 策略对比分析
        report.append("\n\n" + "=" * 80)
        report.append("三、策略对比与分析")
        report.append("=" * 80)

        # 3.1 不同周期对比
        report.append("\n3.1 不同周期对比")
        report.append("-" * 60)

        comparison_data = []
        for period in periods:
            for strat_name in ['momentum', 'mean_reversion', 'multi_factor']:
                result = backtest_results[period][strat_name]
                pv = result['portfolio_value']
                total_ret = pv.iloc[-1] / pv.iloc[0] - 1
                returns = pv.pct_change().dropna()
                volatility = returns.std() * np.sqrt(252)
                sharpe = (total_ret / (len(returns)/252) - 0.02) / volatility if volatility > 0 else 0

                comparison_data.append({
                    '周期': period,
                    '策略': {'momentum': '动量', 'mean_reversion': '均值回归', 'multi_factor': '多因子'}[strat_name],
                    '总收益': total_ret,
                    '年化波动': volatility,
                    '夏普比率': sharpe,
                })

        comparison_df = pd.DataFrame(comparison_data)

        report.append("\n策略表现汇总表:")
        report.append(comparison_df.to_string())

        # 3.2 交易成本敏感性分析
        report.append("\n\n3.2 交易成本敏感性分析")
        report.append("-" * 60)

        cost_levels = [0.0, 0.001, 0.002, 0.003]
        report.append("\n动量策略(20天周期)在不同交易成本下的表现:")

        for cost in cost_levels:
            result = self.momentum_rotation_strategy(prices, lookback=20, hold_period=20,
                                                      top_n=3, transaction_cost=cost)
            pv = result['portfolio_value']
            total_ret = pv.iloc[-1] / pv.iloc[0] - 1
            report.append(f"  交易成本 {cost:.1%}: 总收益 {total_ret:.2%}")

        # 3.3 风险控制建议
        report.append("\n\n3.3 风险控制建议")
        report.append("-" * 60)
        report.append("""
  1. 止损机制: 建议设置单个行业持仓-10%止损，组合-15%止损
  2. 仓位管理: 根据市场波动率动态调整仓位，高波动时降低仓位
  3. 分散投资: 建议持有3-5个行业，避免过度集中
  4. 调仓频率: 20天调仓周期在收益和成本间取得较好平衡
  5. 择时考虑: 在市场整体下跌趋势时可适当降低股票仓位
        """)

        # 4. 结论与建议
        report.append("\n" + "=" * 80)
        report.append("四、结论与建议")
        report.append("=" * 80)

        # 找出最优策略
        best_results = []
        for period in periods:
            for strat_name, strat_result in backtest_results[period].items():
                pv = strat_result['portfolio_value']
                total_ret = pv.iloc[-1] / pv.iloc[0] - 1
                returns = pv.pct_change().dropna()
                volatility = returns.std() * np.sqrt(252)
                sharpe = (total_ret / (len(returns)/252) - 0.02) / volatility if volatility > 0 else 0
                best_results.append({
                    'period': period,
                    'strategy': strat_name,
                    'return': total_ret,
                    'sharpe': sharpe,
                })

        best_by_return = max(best_results, key=lambda x: x['return'])
        best_by_sharpe = max(best_results, key=lambda x: x['sharpe'])

        strat_names = {'momentum': '动量策略', 'mean_reversion': '均值回归策略', 'multi_factor': '多因子策略'}

        report.append(f"""
  研究结论:

  1. 最高收益策略: {strat_names[best_by_return['strategy']]}(周期{best_by_return['period']}天)
     - 总收益率: {best_by_return['return']:.2%}

  2. 最高夏普比率策略: {strat_names[best_by_sharpe['strategy']]}(周期{best_by_sharpe['period']}天)
     - 夏普比率: {best_by_sharpe['sharpe']:.2f}

  3. 策略特点总结:
     - 动量策略: 趋势市场表现好，但震荡市场容易追高
     - 均值回归策略: 震荡市场表现好，但趋势市场容易抄底被套
     - 多因子策略: 综合表现稳定，风险收益比较平衡

  4. 实操建议:
     - 建议使用20-40天的调仓周期
     - 多因子策略更适合长期投资者
     - 动量策略更适合趋势跟踪型投资者
     - 注意控制交易成本，频繁调仓会显著侵蚀收益
        """)

        report.append("\n" + "=" * 80)
        report.append("报告结束")
        report.append("=" * 80)

        # 保存报告
        report_text = "\n".join(report)

        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"\n报告已保存至: {save_path}")

        # 生成图表
        chart_path = save_path.replace('.txt', '_chart.png') if save_path else None
        if chart_path:
            self.plot_strategy_comparison(backtest_results, hs300, chart_path)

        return report_text, backtest_results


def main():
    """主函数"""
    print("=" * 60)
    print("ETF轮动策略研究")
    print("=" * 60)

    # 创建研究实例
    research = ETFRotationResearch()

    # 加载数据
    research.load_data(start_date='20220101', end_date='20260130')

    # 生成报告
    report_path = f"{REPORT_PATH}/etf_rotation_report_{datetime.now().strftime('%Y%m%d')}.txt"
    report_text, results = research.generate_report(save_path=report_path)

    print("\n研究完成!")
    print(f"报告路径: {report_path}")


if __name__ == '__main__':
    main()
