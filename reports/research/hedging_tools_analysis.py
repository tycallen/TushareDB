#!/usr/bin/env python3
"""
对冲工具分析模块

本模块提供A股市场对冲工具的分析功能，包括：
1. 股指期货对冲分析
2. 融券卖空可行性分析
3. ETF对冲分析

使用示例:
    from hedging_tools_analysis import HedgingToolsAnalyzer

    analyzer = HedgingToolsAnalyzer('tushare.db')
    analyzer.analyze_futures_hedging()
    analyzer.analyze_short_selling()
    analyzer.analyze_etf_hedging()
"""

import duckdb
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class FuturesHedgingResult:
    """期货对冲分析结果"""
    index_returns: pd.DataFrame
    correlation: float
    spread_mean: float
    spread_std: float
    hedge_ratio: float
    alpha_return: float
    alpha_volatility: float
    alpha_sharpe: float


@dataclass
class ShortSellingResult:
    """融券卖空分析结果"""
    total_stocks: int
    margin_stocks: int
    coverage_rate: float
    mv_coverage: pd.DataFrame
    liquidity_stats: Dict[str, float]
    top_short_stocks: pd.DataFrame


@dataclass
class ETFHedgingResult:
    """ETF对冲分析结果"""
    etf_pairs: List[Tuple[str, str]]
    correlation_matrix: pd.DataFrame
    spread_analysis: pd.DataFrame


class HedgingToolsAnalyzer:
    """对冲工具分析类"""

    def __init__(self, db_path: str):
        """
        初始化分析器

        Args:
            db_path: DuckDB数据库路径
        """
        self.db_path = db_path
        self.conn = duckdb.connect(db_path, read_only=True)

    def __del__(self):
        if hasattr(self, 'conn'):
            self.conn.close()

    def analyze_futures_hedging(
        self,
        portfolio_returns: Optional[pd.Series] = None,
        index_code: str = '000300.SH',
        start_date: str = '20240101',
        end_date: str = '20241231'
    ) -> FuturesHedgingResult:
        """
        分析股指期货对冲效果

        Args:
            portfolio_returns: 投资组合收益率序列
            index_code: 对冲指数代码
            start_date: 开始日期
            end_date: 结束日期
        """
        # 获取指数收益率
        index_returns = self._get_index_returns(index_code, start_date, end_date)

        if portfolio_returns is None:
            # 使用默认的多因子组合
            portfolio_returns = self._get_default_portfolio_returns(start_date, end_date)

        # 合并数据
        merged = pd.merge(
            portfolio_returns.reset_index(),
            index_returns,
            left_on='index' if 'index' in portfolio_returns.reset_index().columns else 'trade_date',
            right_on='trade_date'
        )

        # 计算相关性
        correlation = merged['portfolio_return'].corr(merged['index_return'])

        # 计算最优对冲比率 (Beta)
        cov = np.cov(merged['portfolio_return'], merged['index_return'])
        hedge_ratio = cov[0, 1] / cov[1, 1]

        # 计算对冲后收益
        merged['hedged_return'] = merged['portfolio_return'] - hedge_ratio * merged['index_return']

        alpha_return = merged['hedged_return'].mean() * 252
        alpha_volatility = merged['hedged_return'].std() * np.sqrt(252)
        alpha_sharpe = alpha_return / alpha_volatility if alpha_volatility > 0 else 0

        # 价差分析
        spread_mean = (merged['portfolio_return'] - merged['index_return']).mean()
        spread_std = (merged['portfolio_return'] - merged['index_return']).std()

        return FuturesHedgingResult(
            index_returns=index_returns,
            correlation=correlation,
            spread_mean=spread_mean,
            spread_std=spread_std,
            hedge_ratio=hedge_ratio,
            alpha_return=alpha_return,
            alpha_volatility=alpha_volatility,
            alpha_sharpe=alpha_sharpe
        )

    def analyze_short_selling(self) -> ShortSellingResult:
        """分析融券卖空可行性"""
        # 融券覆盖率
        coverage = self.conn.execute("""
            WITH latest_margin AS (
                SELECT DISTINCT ts_code
                FROM margin_detail
                WHERE trade_date = (SELECT MAX(trade_date) FROM margin_detail)
            ),
            all_stocks AS (
                SELECT ts_code
                FROM stock_basic
                WHERE list_status = 'L'
            )
            SELECT
                COUNT(DISTINCT a.ts_code) as total_stocks,
                COUNT(DISTINCT m.ts_code) as margin_stocks
            FROM all_stocks a
            LEFT JOIN latest_margin m ON a.ts_code = m.ts_code
        """).fetchone()

        total_stocks = coverage[0]
        margin_stocks = coverage[1]
        coverage_rate = margin_stocks / total_stocks * 100

        # 不同市值组覆盖率
        mv_coverage = self.conn.execute("""
            WITH margin_data AS (
                SELECT ts_code
                FROM margin_detail
                WHERE trade_date = (SELECT MAX(trade_date) FROM margin_detail)
            ),
            mv_groups AS (
                SELECT
                    ts_code,
                    CASE
                        WHEN total_mv < 5000000 THEN '1. <50亿'
                        WHEN total_mv < 10000000 THEN '2. 50-100亿'
                        WHEN total_mv < 30000000 THEN '3. 100-300亿'
                        WHEN total_mv < 100000000 THEN '4. 300-1000亿'
                        ELSE '5. >1000亿'
                    END as mv_group
                FROM daily_basic
                WHERE trade_date = (SELECT MAX(trade_date) FROM daily_basic)
            )
            SELECT
                g.mv_group,
                COUNT(DISTINCT g.ts_code) as total_stocks,
                COUNT(DISTINCT m.ts_code) as margin_stocks,
                COUNT(DISTINCT m.ts_code) * 100.0 / NULLIF(COUNT(DISTINCT g.ts_code), 0) as coverage_pct
            FROM mv_groups g
            LEFT JOIN margin_data m ON g.ts_code = m.ts_code
            GROUP BY g.mv_group
            ORDER BY g.mv_group
        """).fetchdf()

        # 融券流动性
        liquidity = self.conn.execute("""
            WITH margin_data AS (
                SELECT ts_code, rqyl as short_qty
                FROM margin_detail
                WHERE trade_date = (SELECT MAX(trade_date) FROM margin_detail)
                  AND rqyl > 0
            ),
            daily_data AS (
                SELECT ts_code, vol
                FROM daily
                WHERE trade_date = (SELECT MAX(trade_date) FROM daily)
            )
            SELECT
                AVG(m.short_qty / NULLIF(d.vol, 0) * 100) as avg_short_to_vol,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY m.short_qty / NULLIF(d.vol, 0) * 100) as median_short_to_vol,
                MAX(m.short_qty / NULLIF(d.vol, 0) * 100) as max_short_to_vol,
                COUNT(*) as count
            FROM margin_data m
            JOIN daily_data d ON m.ts_code = d.ts_code
            WHERE d.vol > 0
        """).fetchone()

        liquidity_stats = {
            'avg_short_to_vol': liquidity[0],
            'median_short_to_vol': liquidity[1],
            'max_short_to_vol': liquidity[2],
            'sample_count': liquidity[3]
        }

        # 融券卖空比例TOP股票
        top_short = self.conn.execute("""
            WITH latest_margin AS (
                SELECT ts_code, rqye, trade_date
                FROM margin_detail
                WHERE trade_date = (SELECT MAX(trade_date) FROM margin_detail)
            ),
            latest_mv AS (
                SELECT ts_code, circ_mv
                FROM daily_basic
                WHERE trade_date = (SELECT MAX(trade_date) FROM daily_basic)
            )
            SELECT
                m.ts_code,
                s.name,
                m.rqye / 10000 as rq_balance_wan,
                v.circ_mv / 10000 as circ_mv_billion,
                m.rqye / NULLIF(v.circ_mv, 0) * 100 as short_ratio_pct
            FROM latest_margin m
            JOIN latest_mv v ON m.ts_code = v.ts_code
            JOIN stock_basic s ON m.ts_code = s.ts_code
            WHERE m.rqye > 0 AND v.circ_mv > 0
            ORDER BY short_ratio_pct DESC
            LIMIT 20
        """).fetchdf()

        return ShortSellingResult(
            total_stocks=total_stocks,
            margin_stocks=margin_stocks,
            coverage_rate=coverage_rate,
            mv_coverage=mv_coverage,
            liquidity_stats=liquidity_stats,
            top_short_stocks=top_short
        )

    def analyze_etf_hedging(
        self,
        start_date: str = '20240101',
        end_date: str = '20241231'
    ) -> Dict:
        """
        分析ETF对冲效果

        主要分析指数之间的相关性和配对交易机会
        """
        # 获取主要指数收益率
        indices = ['000300.SH', '000905.SH', '000016.SH', '399006.SZ']
        index_data = {}

        for idx in indices:
            df = self._get_index_returns(idx, start_date, end_date)
            if len(df) > 0:
                index_data[idx] = df.set_index('trade_date')['index_return']

        if len(index_data) < 2:
            return {'error': 'Insufficient index data'}

        # 合并数据
        merged = pd.DataFrame(index_data)
        merged = merged.dropna()

        # 计算相关性矩阵
        correlation_matrix = merged.corr()

        # 配对分析: 沪深300 vs 中证500
        if '000300.SH' in merged.columns and '000905.SH' in merged.columns:
            hs300 = merged['000300.SH']
            zz500 = merged['000905.SH']

            spread = zz500 - hs300
            spread_stats = {
                'correlation': hs300.corr(zz500),
                'spread_mean': spread.mean(),
                'spread_std': spread.std(),
                'spread_sharpe': spread.mean() / spread.std() * np.sqrt(252) if spread.std() > 0 else 0
            }
        else:
            spread_stats = None

        return {
            'correlation_matrix': correlation_matrix,
            'spread_stats': spread_stats,
            'index_returns': merged
        }

    def calculate_beta(
        self,
        ts_codes: List[str],
        index_code: str = '000300.SH',
        start_date: str = '20240101',
        end_date: str = '20241231'
    ) -> pd.DataFrame:
        """
        计算个股Beta

        Args:
            ts_codes: 股票代码列表
            index_code: 基准指数代码
            start_date: 开始日期
            end_date: 结束日期
        """
        # 获取股票收益率
        ts_codes_str = "', '".join(ts_codes)
        stock_returns = self.conn.execute(f"""
            SELECT ts_code, trade_date, pct_chg
            FROM daily
            WHERE ts_code IN ('{ts_codes_str}')
              AND trade_date >= '{start_date}'
              AND trade_date <= '{end_date}'
            ORDER BY ts_code, trade_date
        """).fetchdf()

        # 获取指数收益率
        index_returns = self._get_index_returns(index_code, start_date, end_date)

        # 计算每只股票的Beta
        betas = []
        for ts_code in ts_codes:
            stock_df = stock_returns[stock_returns['ts_code'] == ts_code]
            merged = pd.merge(stock_df, index_returns, on='trade_date')

            if len(merged) >= 20:
                cov = np.cov(merged['pct_chg'], merged['index_return'])
                beta = cov[0, 1] / cov[1, 1] if cov[1, 1] != 0 else 0
                corr = merged['pct_chg'].corr(merged['index_return'])

                betas.append({
                    'ts_code': ts_code,
                    'beta': beta,
                    'correlation': corr,
                    'observations': len(merged)
                })

        return pd.DataFrame(betas)

    def recommend_hedge_ratio(
        self,
        portfolio_composition: Dict[str, float],
        index_code: str = '000300.SH',
        start_date: str = '20240101',
        end_date: str = '20241231'
    ) -> Dict:
        """
        推荐对冲比率

        Args:
            portfolio_composition: 投资组合组成 {ts_code: weight}
            index_code: 对冲指数
        """
        ts_codes = list(portfolio_composition.keys())
        betas = self.calculate_beta(ts_codes, index_code, start_date, end_date)

        if len(betas) == 0:
            return {'error': 'No valid beta calculations'}

        # 计算组合Beta
        portfolio_beta = 0
        for _, row in betas.iterrows():
            weight = portfolio_composition.get(row['ts_code'], 0)
            portfolio_beta += weight * row['beta']

        return {
            'portfolio_beta': portfolio_beta,
            'recommended_hedge_ratio': portfolio_beta,
            'individual_betas': betas
        }

    def _get_index_returns(self, index_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取指数收益率"""
        return self.conn.execute(f"""
            SELECT trade_date, pct_chg as index_return
            FROM index_daily
            WHERE ts_code = '{index_code}'
              AND trade_date >= '{start_date}'
              AND trade_date <= '{end_date}'
            ORDER BY trade_date
        """).fetchdf()

    def _get_default_portfolio_returns(self, start_date: str, end_date: str) -> pd.Series:
        """获取默认多因子组合收益率"""
        result = self.conn.execute(f"""
            WITH factor_scores AS (
                SELECT
                    ts_code,
                    trade_date,
                    pct_chg,
                    -1 * PERCENT_RANK() OVER (PARTITION BY trade_date ORDER BY LOG(total_mv)) as size_score,
                    PERCENT_RANK() OVER (PARTITION BY trade_date ORDER BY 1.0/NULLIF(pe_ttm, 0)) as value_score,
                    LEAD(pct_chg, 1) OVER (PARTITION BY ts_code ORDER BY trade_date) as next_return
                FROM stk_factor_pro
                WHERE trade_date >= '{start_date}' AND trade_date <= '{end_date}'
                  AND total_mv > 0 AND pe_ttm > 0 AND pe_ttm < 500
            ),
            composite AS (
                SELECT
                    trade_date,
                    next_return,
                    (size_score + value_score) / 2 as composite_score,
                    NTILE(10) OVER (PARTITION BY trade_date ORDER BY (size_score + value_score) / 2) as decile
                FROM factor_scores
                WHERE next_return IS NOT NULL
            )
            SELECT
                trade_date,
                AVG(CASE WHEN decile = 10 THEN next_return END) as portfolio_return
            FROM composite
            GROUP BY trade_date
            HAVING AVG(CASE WHEN decile = 10 THEN next_return END) IS NOT NULL
            ORDER BY trade_date
        """).fetchdf()

        return result.set_index('trade_date')['portfolio_return']

    def print_short_selling_report(self, result: ShortSellingResult):
        """打印融券卖空分析报告"""
        print("\n" + "="*60)
        print("融券卖空可行性分析报告")
        print("="*60)

        print(f"\n【整体覆盖情况】")
        print(f"  上市股票总数: {result.total_stocks}")
        print(f"  可融券股票数: {result.margin_stocks}")
        print(f"  覆盖率: {result.coverage_rate:.1f}%")

        print(f"\n【不同市值组覆盖率】")
        print(result.mv_coverage.to_string(index=False))

        print(f"\n【融券流动性统计】")
        print(f"  融券余量/日成交量(平均): {result.liquidity_stats['avg_short_to_vol']:.2f}%")
        print(f"  融券余量/日成交量(中位): {result.liquidity_stats['median_short_to_vol']:.2f}%")
        print(f"  有效样本数: {result.liquidity_stats['sample_count']}")

        print(f"\n【融券卖空比例TOP10】")
        print(result.top_short_stocks.head(10).to_string(index=False))

        print("="*60)


def main():
    """主函数"""
    import os

    db_path = os.path.join(os.path.dirname(__file__), '..', '..', 'tushare.db')

    print("初始化对冲工具分析器...")
    analyzer = HedgingToolsAnalyzer(db_path)

    print("\n分析融券卖空可行性...")
    short_result = analyzer.analyze_short_selling()
    analyzer.print_short_selling_report(short_result)

    print("\n分析ETF对冲效果...")
    etf_result = analyzer.analyze_etf_hedging()
    if 'correlation_matrix' in etf_result:
        print("\n指数相关性矩阵:")
        print(etf_result['correlation_matrix'].round(3))

    if etf_result.get('spread_stats'):
        print("\n沪深300 vs 中证500 配对统计:")
        for k, v in etf_result['spread_stats'].items():
            print(f"  {k}: {v:.4f}")

    print("\n分析期货对冲效果...")
    futures_result = analyzer.analyze_futures_hedging()
    print(f"\n期货对冲分析结果:")
    print(f"  组合与指数相关性: {futures_result.correlation:.3f}")
    print(f"  最优对冲比率(Beta): {futures_result.hedge_ratio:.3f}")
    print(f"  对冲后Alpha年化: {futures_result.alpha_return:.2f}%")
    print(f"  对冲后波动率: {futures_result.alpha_volatility:.2f}%")
    print(f"  Alpha夏普比率: {futures_result.alpha_sharpe:.2f}")


if __name__ == '__main__':
    main()
