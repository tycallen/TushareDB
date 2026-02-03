#!/usr/bin/env python3
"""
多空对冲策略回测框架

本模块提供A股多空对冲策略的回测和分析功能，包括：
1. 因子多空组合
2. 行业中性策略
3. 市值中性策略
4. 双重中性策略

使用示例:
    from long_short_strategy import LongShortBacktest

    backtest = LongShortBacktest('tushare.db')
    results = backtest.run_multifactor_ls(start_date='20240101', end_date='20241231')
    backtest.print_performance(results)
"""

import duckdb
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class BacktestResult:
    """回测结果数据类"""
    daily_returns: pd.DataFrame
    long_returns: pd.Series
    short_returns: pd.Series
    ls_spread: pd.Series
    cumulative_return: float
    annual_return: float
    annual_volatility: float
    sharpe_ratio: float
    information_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    var_95: float
    cvar_95: float


class LongShortBacktest:
    """多空对冲策略回测类"""

    def __init__(self, db_path: str):
        """
        初始化回测框架

        Args:
            db_path: DuckDB数据库路径
        """
        self.db_path = db_path
        self.conn = duckdb.connect(db_path, read_only=True)

    def __del__(self):
        """关闭数据库连接"""
        if hasattr(self, 'conn'):
            self.conn.close()

    def get_factor_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取因子数据

        Args:
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)

        Returns:
            因子数据DataFrame
        """
        query = f"""
        SELECT
            ts_code,
            trade_date,
            pct_chg,
            close_qfq,
            total_mv,
            circ_mv,
            pe_ttm,
            pb,
            turnover_rate_f
        FROM stk_factor_pro
        WHERE trade_date >= '{start_date}' AND trade_date <= '{end_date}'
          AND total_mv > 0 AND pe_ttm IS NOT NULL
        ORDER BY trade_date, ts_code
        """
        return self.conn.execute(query).fetchdf()

    def get_industry_data(self) -> pd.DataFrame:
        """获取行业分类数据"""
        query = """
        SELECT ts_code, l1_name as industry
        FROM index_member_all
        WHERE is_new = 'Y'
        """
        return self.conn.execute(query).fetchdf()

    def get_index_returns(self, index_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取指数收益率

        Args:
            index_code: 指数代码 (e.g., '000300.SH')
            start_date: 开始日期
            end_date: 结束日期
        """
        query = f"""
        SELECT trade_date, pct_chg as index_return
        FROM index_daily
        WHERE ts_code = '{index_code}'
          AND trade_date >= '{start_date}'
          AND trade_date <= '{end_date}'
        ORDER BY trade_date
        """
        return self.conn.execute(query).fetchdf()

    def calculate_factor_scores(self, df: pd.DataFrame, factors: List[str]) -> pd.DataFrame:
        """
        计算因子得分

        Args:
            df: 原始数据
            factors: 因子列表 ['size', 'value', 'momentum', 'reversal']
        """
        df = df.copy()

        for date in df['trade_date'].unique():
            mask = df['trade_date'] == date
            date_df = df.loc[mask]

            if 'size' in factors:
                # 小市值高分
                df.loc[mask, 'size_score'] = 1 - date_df['total_mv'].rank(pct=True)

            if 'value' in factors:
                # 低估值(高EP)高分
                valid_pe = (date_df['pe_ttm'] > 0) & (date_df['pe_ttm'] < 500)
                df.loc[mask & valid_pe, 'value_score'] = (1.0 / date_df.loc[valid_pe, 'pe_ttm']).rank(pct=True)

            if 'reversal' in factors:
                # 低换手率高分(反转)
                df.loc[mask, 'reversal_score'] = 1 - date_df['turnover_rate_f'].rank(pct=True)

        return df

    def run_multifactor_ls(
        self,
        start_date: str,
        end_date: str,
        factors: List[str] = ['size', 'value', 'reversal'],
        long_pct: float = 0.1,
        short_pct: float = 0.1
    ) -> BacktestResult:
        """
        运行多因子多空策略回测

        Args:
            start_date: 开始日期
            end_date: 结束日期
            factors: 使用的因子列表
            long_pct: 多头持仓比例
            short_pct: 空头持仓比例
        """
        # 获取数据
        df = self.get_factor_data(start_date, end_date)

        # 计算因子得分
        df = self.calculate_factor_scores(df, factors)

        # 计算复合因子
        score_cols = [f'{f}_score' for f in factors if f'{f}_score' in df.columns]
        df['composite_score'] = df[score_cols].mean(axis=1)

        # 计算下期收益
        df = df.sort_values(['ts_code', 'trade_date'])
        df['next_return'] = df.groupby('ts_code')['pct_chg'].shift(-1)

        # 按日期计算多空收益
        daily_results = []
        for date in df['trade_date'].unique():
            date_df = df[(df['trade_date'] == date) & df['next_return'].notna()]

            if len(date_df) < 100:
                continue

            # 多头: 复合因子Top
            long_threshold = date_df['composite_score'].quantile(1 - long_pct)
            long_mask = date_df['composite_score'] >= long_threshold
            long_return = date_df.loc[long_mask, 'next_return'].mean()

            # 空头: 复合因子Bottom
            short_threshold = date_df['composite_score'].quantile(short_pct)
            short_mask = date_df['composite_score'] <= short_threshold
            short_return = date_df.loc[short_mask, 'next_return'].mean()

            if pd.notna(long_return) and pd.notna(short_return):
                daily_results.append({
                    'trade_date': date,
                    'long_return': long_return,
                    'short_return': short_return,
                    'ls_spread': long_return - short_return
                })

        results_df = pd.DataFrame(daily_results)
        return self._calculate_performance(results_df)

    def run_industry_neutral(
        self,
        start_date: str,
        end_date: str,
        factor: str = 'value',
        long_pct: float = 0.2,
        short_pct: float = 0.2
    ) -> BacktestResult:
        """
        运行行业中性策略回测
        """
        # 获取数据
        df = self.get_factor_data(start_date, end_date)
        industry_df = self.get_industry_data()
        df = df.merge(industry_df, on='ts_code', how='inner')

        # 计算因子得分
        df = self.calculate_factor_scores(df, [factor])
        score_col = f'{factor}_score'

        # 计算下期收益
        df = df.sort_values(['ts_code', 'trade_date'])
        df['next_return'] = df.groupby('ts_code')['pct_chg'].shift(-1)

        # 按日期和行业计算多空收益
        daily_results = []
        for date in df['trade_date'].unique():
            date_df = df[(df['trade_date'] == date) & df['next_return'].notna()]

            industry_returns = []
            for industry in date_df['industry'].unique():
                ind_df = date_df[date_df['industry'] == industry]

                if len(ind_df) < 10:
                    continue

                long_threshold = ind_df[score_col].quantile(1 - long_pct)
                short_threshold = ind_df[score_col].quantile(short_pct)

                long_ret = ind_df.loc[ind_df[score_col] >= long_threshold, 'next_return'].mean()
                short_ret = ind_df.loc[ind_df[score_col] <= short_threshold, 'next_return'].mean()

                if pd.notna(long_ret) and pd.notna(short_ret):
                    industry_returns.append({
                        'long_return': long_ret,
                        'short_return': short_ret
                    })

            if industry_returns:
                ind_df = pd.DataFrame(industry_returns)
                daily_results.append({
                    'trade_date': date,
                    'long_return': ind_df['long_return'].mean(),
                    'short_return': ind_df['short_return'].mean(),
                    'ls_spread': ind_df['long_return'].mean() - ind_df['short_return'].mean()
                })

        results_df = pd.DataFrame(daily_results)
        return self._calculate_performance(results_df)

    def run_size_neutral(
        self,
        start_date: str,
        end_date: str,
        factor: str = 'value',
        n_groups: int = 5,
        long_pct: float = 0.2,
        short_pct: float = 0.2
    ) -> BacktestResult:
        """
        运行市值中性策略回测
        """
        df = self.get_factor_data(start_date, end_date)
        df = self.calculate_factor_scores(df, [factor])
        score_col = f'{factor}_score'

        df = df.sort_values(['ts_code', 'trade_date'])
        df['next_return'] = df.groupby('ts_code')['pct_chg'].shift(-1)

        daily_results = []
        for date in df['trade_date'].unique():
            date_df = df[(df['trade_date'] == date) & df['next_return'].notna()]

            if len(date_df) < 100:
                continue

            # 市值分组
            date_df['mv_group'] = pd.qcut(date_df['total_mv'], q=n_groups, labels=False, duplicates='drop')

            group_returns = []
            for group in date_df['mv_group'].unique():
                group_df = date_df[date_df['mv_group'] == group]

                if len(group_df) < 10:
                    continue

                long_threshold = group_df[score_col].quantile(1 - long_pct)
                short_threshold = group_df[score_col].quantile(short_pct)

                long_ret = group_df.loc[group_df[score_col] >= long_threshold, 'next_return'].mean()
                short_ret = group_df.loc[group_df[score_col] <= short_threshold, 'next_return'].mean()

                if pd.notna(long_ret) and pd.notna(short_ret):
                    group_returns.append({
                        'long_return': long_ret,
                        'short_return': short_ret
                    })

            if group_returns:
                grp_df = pd.DataFrame(group_returns)
                daily_results.append({
                    'trade_date': date,
                    'long_return': grp_df['long_return'].mean(),
                    'short_return': grp_df['short_return'].mean(),
                    'ls_spread': grp_df['long_return'].mean() - grp_df['short_return'].mean()
                })

        results_df = pd.DataFrame(daily_results)
        return self._calculate_performance(results_df)

    def run_double_neutral(
        self,
        start_date: str,
        end_date: str,
        factor: str = 'value',
        n_mv_groups: int = 3,
        long_pct: float = 0.25,
        short_pct: float = 0.25
    ) -> BacktestResult:
        """
        运行双重中性策略 (行业+市值)
        """
        df = self.get_factor_data(start_date, end_date)
        industry_df = self.get_industry_data()
        df = df.merge(industry_df, on='ts_code', how='inner')

        df = self.calculate_factor_scores(df, [factor])
        score_col = f'{factor}_score'

        df = df.sort_values(['ts_code', 'trade_date'])
        df['next_return'] = df.groupby('ts_code')['pct_chg'].shift(-1)

        daily_results = []
        for date in df['trade_date'].unique():
            date_df = df[(df['trade_date'] == date) & df['next_return'].notna()]

            cell_returns = []
            for industry in date_df['industry'].unique():
                ind_df = date_df[date_df['industry'] == industry]

                if len(ind_df) < 15:
                    continue

                # 行业内市值分组
                ind_df = ind_df.copy()
                ind_df['mv_group'] = pd.qcut(ind_df['total_mv'], q=n_mv_groups, labels=False, duplicates='drop')

                for group in ind_df['mv_group'].unique():
                    cell_df = ind_df[ind_df['mv_group'] == group]

                    if len(cell_df) < 5:
                        continue

                    long_threshold = cell_df[score_col].quantile(1 - long_pct)
                    short_threshold = cell_df[score_col].quantile(short_pct)

                    long_ret = cell_df.loc[cell_df[score_col] >= long_threshold, 'next_return'].mean()
                    short_ret = cell_df.loc[cell_df[score_col] <= short_threshold, 'next_return'].mean()

                    if pd.notna(long_ret) and pd.notna(short_ret):
                        cell_returns.append({
                            'long_return': long_ret,
                            'short_return': short_ret
                        })

            if cell_returns:
                cell_df = pd.DataFrame(cell_returns)
                daily_results.append({
                    'trade_date': date,
                    'long_return': cell_df['long_return'].mean(),
                    'short_return': cell_df['short_return'].mean(),
                    'ls_spread': cell_df['long_return'].mean() - cell_df['short_return'].mean()
                })

        results_df = pd.DataFrame(daily_results)
        return self._calculate_performance(results_df)

    def _calculate_performance(self, results_df: pd.DataFrame) -> BacktestResult:
        """计算绩效指标"""
        if len(results_df) == 0:
            raise ValueError("No valid trading days found")

        long_returns = results_df['long_return']
        short_returns = results_df['short_return']
        ls_spread = results_df['ls_spread']

        # 基本统计
        daily_mean = ls_spread.mean()
        daily_std = ls_spread.std()
        annual_return = daily_mean * 252
        annual_volatility = daily_std * np.sqrt(252)

        # 夏普比率 (假设无风险利率3%)
        sharpe_ratio = (annual_return - 3) / annual_volatility if annual_volatility > 0 else 0

        # 信息比率
        information_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0

        # 累计收益和最大回撤
        cumret = (1 + ls_spread / 100).cumprod()
        cumulative_return = (cumret.iloc[-1] - 1) * 100
        running_max = cumret.expanding().max()
        drawdown = (cumret - running_max) / running_max
        max_drawdown = drawdown.min() * 100

        # Calmar比率
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # 胜率
        win_rate = (ls_spread > 0).mean() * 100

        # VaR和CVaR
        var_95 = ls_spread.quantile(0.05)
        cvar_95 = ls_spread[ls_spread <= var_95].mean()

        return BacktestResult(
            daily_returns=results_df,
            long_returns=long_returns,
            short_returns=short_returns,
            ls_spread=ls_spread,
            cumulative_return=cumulative_return,
            annual_return=annual_return,
            annual_volatility=annual_volatility,
            sharpe_ratio=sharpe_ratio,
            information_ratio=information_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            win_rate=win_rate,
            var_95=var_95,
            cvar_95=cvar_95
        )

    def print_performance(self, result: BacktestResult, title: str = "策略绩效"):
        """打印绩效报告"""
        print(f"\n{'='*50}")
        print(f"{title}")
        print('='*50)
        print(f"交易日数: {len(result.daily_returns)}")
        print(f"累计收益: {result.cumulative_return:.2f}%")
        print(f"年化收益: {result.annual_return:.2f}%")
        print(f"年化波动率: {result.annual_volatility:.2f}%")
        print(f"夏普比率: {result.sharpe_ratio:.2f}")
        print(f"信息比率: {result.information_ratio:.2f}")
        print(f"最大回撤: {result.max_drawdown:.2f}%")
        print(f"Calmar比率: {result.calmar_ratio:.2f}")
        print(f"日胜率: {result.win_rate:.1f}%")
        print(f"95% VaR: {result.var_95:.2f}%")
        print(f"95% CVaR: {result.cvar_95:.2f}%")
        print('='*50)

    def compare_strategies(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        比较不同策略的绩效
        """
        strategies = {
            '多因子多空': lambda: self.run_multifactor_ls(start_date, end_date),
            '行业中性': lambda: self.run_industry_neutral(start_date, end_date),
            '市值中性': lambda: self.run_size_neutral(start_date, end_date),
            '双重中性': lambda: self.run_double_neutral(start_date, end_date),
        }

        results = []
        for name, func in strategies.items():
            try:
                result = func()
                results.append({
                    '策略': name,
                    '年化收益%': result.annual_return,
                    '年化波动%': result.annual_volatility,
                    '信息比率': result.information_ratio,
                    '最大回撤%': result.max_drawdown,
                    '日胜率%': result.win_rate
                })
                print(f"策略 {name} 计算完成")
            except Exception as e:
                print(f"策略 {name} 计算失败: {e}")

        return pd.DataFrame(results)


def main():
    """主函数: 运行策略回测示例"""
    import os

    # 数据库路径
    db_path = os.path.join(os.path.dirname(__file__), '..', '..', 'tushare.db')

    print("初始化回测框架...")
    backtest = LongShortBacktest(db_path)

    print("\n运行多因子多空策略回测 (2024年)...")
    result = backtest.run_multifactor_ls('20240101', '20241231')
    backtest.print_performance(result, "多因子多空策略")

    print("\n运行行业中性策略回测...")
    result = backtest.run_industry_neutral('20240101', '20241231')
    backtest.print_performance(result, "行业中性策略")

    print("\n运行市值中性策略回测...")
    result = backtest.run_size_neutral('20240101', '20241231')
    backtest.print_performance(result, "市值中性策略")

    print("\n运行双重中性策略回测...")
    result = backtest.run_double_neutral('20240101', '20241231')
    backtest.print_performance(result, "双重中性策略")

    print("\n策略对比:")
    comparison = backtest.compare_strategies('20240101', '20241231')
    print(comparison.to_string(index=False))


if __name__ == '__main__':
    main()
