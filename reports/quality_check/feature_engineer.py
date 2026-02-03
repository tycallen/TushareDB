"""
横截面与时序特征工程实现
数据库: /Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db

特征类别:
1. 横截面特征 (Cross-sectional): 同一时点比较不同股票
2. 时序特征 (Time-series): 单只股票历史数据特征
3. 筹码特征 (Chip): 基于筹码分布的特征
"""

import duckdb
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from pathlib import Path


class FeatureEngineer:
    """横截面和时序特征计算器"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = duckdb.connect(db_path, read_only=True)
        print(f"Connected to database: {db_path}")

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # ==================== 横截面特征 ====================

    def compute_market_rank_features(self, trade_date: str) -> pd.DataFrame:
        """
        计算市场排名特征

        Returns:
            DataFrame with columns:
            - pct_chg_rank: 涨幅排名百分位 [0,1]
            - vol_rank: 成交量排名百分位 [0,1]
            - turnover_rank: 换手率排名百分位 [0,1]
            - mktcap_rank: 市值排名百分位 [0,1]
        """
        query = """
        SELECT
            d.ts_code,
            d.trade_date,
            d.pct_chg,
            d.vol,
            d.amount,
            db.turnover_rate,
            db.total_mv,
            db.circ_mv,
            PERCENT_RANK() OVER (ORDER BY d.pct_chg) AS pct_chg_rank,
            PERCENT_RANK() OVER (ORDER BY d.vol) AS vol_rank,
            PERCENT_RANK() OVER (ORDER BY d.amount) AS amount_rank,
            PERCENT_RANK() OVER (ORDER BY db.turnover_rate) AS turnover_rank,
            PERCENT_RANK() OVER (ORDER BY db.total_mv) AS mktcap_rank,
            PERCENT_RANK() OVER (ORDER BY db.circ_mv) AS circ_mktcap_rank
        FROM daily d
        JOIN daily_basic db ON d.ts_code = db.ts_code AND d.trade_date = db.trade_date
        WHERE d.trade_date = ?
          AND d.pct_chg IS NOT NULL
        """
        return self.conn.execute(query, [trade_date]).fetchdf()

    def compute_industry_relative_features(self, trade_date: str) -> pd.DataFrame:
        """
        计算行业相对特征

        Returns:
            DataFrame with columns:
            - excess_return_industry: 相对行业超额收益
            - relative_turnover_industry: 相对行业换手率
            - rank_in_industry: 行业内排名百分位
            - industry_strength: 行业强弱度
            - is_industry_leader: 是否行业领涨
        """
        query = """
        WITH industry_mapping AS (
            SELECT ts_code, l1_code, l1_name
            FROM index_member_all
            WHERE (out_date IS NULL OR out_date > ?)
        ),
        daily_with_industry AS (
            SELECT
                d.ts_code,
                d.trade_date,
                d.pct_chg,
                db.turnover_rate,
                im.l1_code,
                im.l1_name
            FROM daily d
            JOIN daily_basic db ON d.ts_code = db.ts_code AND d.trade_date = db.trade_date
            JOIN industry_mapping im ON d.ts_code = im.ts_code
            WHERE d.trade_date = ?
              AND d.pct_chg IS NOT NULL
        )
        SELECT
            ts_code,
            trade_date,
            pct_chg,
            turnover_rate,
            l1_code,
            l1_name,
            -- 行业平均涨幅
            AVG(pct_chg) OVER (PARTITION BY l1_code) AS industry_avg_pct_chg,
            -- 相对行业超额收益
            pct_chg - AVG(pct_chg) OVER (PARTITION BY l1_code) AS excess_return_industry,
            -- 行业平均换手率
            AVG(turnover_rate) OVER (PARTITION BY l1_code) AS industry_avg_turnover,
            -- 相对行业换手率
            turnover_rate / NULLIF(AVG(turnover_rate) OVER (PARTITION BY l1_code), 0) AS relative_turnover_industry,
            -- 行业内涨幅排名
            PERCENT_RANK() OVER (PARTITION BY l1_code ORDER BY pct_chg) AS rank_in_industry,
            -- 行业内股票数量
            COUNT(*) OVER (PARTITION BY l1_code) AS industry_stock_count,
            -- 市场平均涨幅
            AVG(pct_chg) OVER () AS market_avg_pct_chg,
            -- 行业强弱度
            AVG(pct_chg) OVER (PARTITION BY l1_code) - AVG(pct_chg) OVER () AS industry_strength,
            -- 行业内涨幅标准差
            STDDEV(pct_chg) OVER (PARTITION BY l1_code) AS industry_volatility,
            -- 是否行业领涨
            CASE WHEN RANK() OVER (PARTITION BY l1_code ORDER BY pct_chg DESC) = 1 THEN 1 ELSE 0 END AS is_industry_leader
        FROM daily_with_industry
        """
        return self.conn.execute(query, [trade_date, trade_date]).fetchdf()

    def compute_market_sentiment_features(self, trade_date: str) -> pd.DataFrame:
        """
        计算市场情绪特征

        Returns:
            DataFrame with columns:
            - limit_up_count: 涨停数量
            - limit_down_count: 跌停数量
            - advance_count: 上涨家数
            - decline_count: 下跌家数
            - advance_decline_ratio: 涨跌比
            - market_breadth_pct: 市场宽度百分比
        """
        query = """
        SELECT
            trade_date,
            -- 涨停数量 (区分创业板/科创板20%涨跌幅)
            COUNT(CASE WHEN (ts_code LIKE '30%' OR ts_code LIKE '68%') AND pct_chg >= 19.5 THEN 1
                       WHEN NOT (ts_code LIKE '30%' OR ts_code LIKE '68%') AND pct_chg >= 9.5 THEN 1 END) AS limit_up_count,
            -- 跌停数量
            COUNT(CASE WHEN (ts_code LIKE '30%' OR ts_code LIKE '68%') AND pct_chg <= -19.5 THEN 1
                       WHEN NOT (ts_code LIKE '30%' OR ts_code LIKE '68%') AND pct_chg <= -9.5 THEN 1 END) AS limit_down_count,
            -- 上涨家数
            COUNT(CASE WHEN pct_chg > 0 THEN 1 END) AS advance_count,
            -- 下跌家数
            COUNT(CASE WHEN pct_chg < 0 THEN 1 END) AS decline_count,
            -- 平盘家数
            COUNT(CASE WHEN pct_chg = 0 THEN 1 END) AS unchanged_count,
            -- 总股票数
            COUNT(*) AS total_stocks,
            -- 涨跌比
            COUNT(CASE WHEN pct_chg > 0 THEN 1 END) * 1.0 /
                NULLIF(COUNT(CASE WHEN pct_chg < 0 THEN 1 END), 0) AS advance_decline_ratio,
            -- 市场宽度百分比
            (COUNT(CASE WHEN pct_chg > 0 THEN 1 END) - COUNT(CASE WHEN pct_chg < 0 THEN 1 END)) * 1.0 /
                COUNT(*) AS market_breadth_pct,
            -- 涨幅超过3%的股票数
            COUNT(CASE WHEN pct_chg > 3 THEN 1 END) AS strong_up_count,
            -- 跌幅超过3%的股票数
            COUNT(CASE WHEN pct_chg < -3 THEN 1 END) AS strong_down_count,
            -- 市场平均涨幅
            AVG(pct_chg) AS market_avg_pct_chg,
            -- 市场涨幅中位数
            MEDIAN(pct_chg) AS market_median_pct_chg
        FROM daily
        WHERE trade_date = ?
          AND pct_chg IS NOT NULL
        GROUP BY trade_date
        """
        return self.conn.execute(query, [trade_date]).fetchdf()

    def compute_moneyflow_features(self, trade_date: str) -> pd.DataFrame:
        """
        计算资金流向特征

        Returns:
            DataFrame with columns:
            - net_main_inflow: 主力资金净流入
            - main_buy_ratio: 主力买入占比
        """
        query = """
        SELECT
            ts_code,
            trade_date,
            -- 主力买入金额 (大单+超大单)
            (buy_lg_amount + COALESCE(buy_elg_amount, 0)) AS main_buy_amount,
            -- 主力卖出金额
            (sell_lg_amount + COALESCE(sell_elg_amount, 0)) AS main_sell_amount,
            -- 主力净流入
            (buy_lg_amount + COALESCE(buy_elg_amount, 0)) -
                (sell_lg_amount + COALESCE(sell_elg_amount, 0)) AS net_main_inflow,
            -- 小单净流入
            (buy_sm_amount - sell_sm_amount) AS net_small_flow,
            -- 总成交额
            (buy_sm_amount + buy_md_amount + buy_lg_amount + COALESCE(buy_elg_amount, 0)) AS total_amount,
            -- 主力买入占比
            (buy_lg_amount + COALESCE(buy_elg_amount, 0)) /
                NULLIF(buy_sm_amount + buy_md_amount + buy_lg_amount + COALESCE(buy_elg_amount, 0), 0) AS main_buy_ratio
        FROM moneyflow
        WHERE trade_date = ?
        ORDER BY net_main_inflow DESC
        """
        return self.conn.execute(query, [trade_date]).fetchdf()

    # ==================== 时序特征 ====================

    def compute_momentum_features(self, ts_code: str, end_date: str, lookback: int = 252) -> pd.DataFrame:
        """
        计算动量特征

        Args:
            ts_code: 股票代码
            end_date: 结束日期
            lookback: 回看天数

        Returns:
            DataFrame with columns:
            - return_1d/5d/10d/20d/60d/120d: 不同周期收益率
            - momentum_reversal: 短期与长期动量差
        """
        query = """
        WITH price_history AS (
            SELECT
                ts_code,
                trade_date,
                close,
                pct_chg,
                LAG(close, 1) OVER (ORDER BY trade_date) AS close_1d,
                LAG(close, 5) OVER (ORDER BY trade_date) AS close_5d,
                LAG(close, 10) OVER (ORDER BY trade_date) AS close_10d,
                LAG(close, 20) OVER (ORDER BY trade_date) AS close_20d,
                LAG(close, 60) OVER (ORDER BY trade_date) AS close_60d,
                LAG(close, 120) OVER (ORDER BY trade_date) AS close_120d
            FROM daily
            WHERE ts_code = ?
            ORDER BY trade_date
        )
        SELECT
            ts_code,
            trade_date,
            close,
            pct_chg AS return_1d,
            (close / NULLIF(close_5d, 0) - 1) * 100 AS return_5d,
            (close / NULLIF(close_10d, 0) - 1) * 100 AS return_10d,
            (close / NULLIF(close_20d, 0) - 1) * 100 AS return_20d,
            (close / NULLIF(close_60d, 0) - 1) * 100 AS return_60d,
            (close / NULLIF(close_120d, 0) - 1) * 100 AS return_120d,
            -- 动量反转: 短期动量 - 长期动量
            ((close / NULLIF(close_5d, 0)) - (close / NULLIF(close_20d, 0))) * 100 AS momentum_reversal_5_20,
            ((close / NULLIF(close_20d, 0)) - (close / NULLIF(close_60d, 0))) * 100 AS momentum_reversal_20_60
        FROM price_history
        WHERE trade_date <= ?
        ORDER BY trade_date DESC
        LIMIT ?
        """
        return self.conn.execute(query, [ts_code, end_date, lookback]).fetchdf()

    def compute_volatility_features(self, ts_code: str, end_date: str, lookback: int = 252) -> pd.DataFrame:
        """
        计算波动率特征

        Returns:
            DataFrame with columns:
            - vol_5d/10d/20d/60d: 不同周期波动率
            - annualized_vol: 年化波动率
            - vol_change_ratio: 波动率变化率
        """
        query = """
        WITH volatility_calc AS (
            SELECT
                ts_code,
                trade_date,
                pct_chg,
                STDDEV(pct_chg) OVER (ORDER BY trade_date ROWS 4 PRECEDING) AS vol_5d,
                STDDEV(pct_chg) OVER (ORDER BY trade_date ROWS 9 PRECEDING) AS vol_10d,
                STDDEV(pct_chg) OVER (ORDER BY trade_date ROWS 19 PRECEDING) AS vol_20d,
                STDDEV(pct_chg) OVER (ORDER BY trade_date ROWS 59 PRECEDING) AS vol_60d,
                AVG(pct_chg) OVER (ORDER BY trade_date ROWS 19 PRECEDING) AS mean_20d
            FROM daily
            WHERE ts_code = ?
            ORDER BY trade_date
        )
        SELECT
            ts_code,
            trade_date,
            pct_chg,
            vol_5d,
            vol_10d,
            vol_20d,
            vol_60d,
            -- 年化波动率 (按20日计算)
            vol_20d * SQRT(252) AS annualized_vol_20d,
            -- 波动率变化: 短期/长期
            vol_5d / NULLIF(vol_20d, 0) AS vol_ratio_5_20,
            vol_20d / NULLIF(vol_60d, 0) AS vol_ratio_20_60,
            -- 收益率均值
            mean_20d,
            -- 夏普比率 (简化版)
            mean_20d / NULLIF(vol_20d, 0) AS sharpe_ratio_20d
        FROM volatility_calc
        WHERE trade_date <= ?
        ORDER BY trade_date DESC
        LIMIT ?
        """
        return self.conn.execute(query, [ts_code, end_date, lookback]).fetchdf()

    def compute_state_features(self, ts_code: str, end_date: str, lookback: int = 252) -> pd.DataFrame:
        """
        计算状态特征

        Returns:
            DataFrame with columns:
            - ma5/10/20/60: 均线值
            - ma_bull_alignment: 多头排列标志
            - bollinger_position: 布林带位置 [0,1]
            - position_in_range: 价格在区间内位置 [0,1]
        """
        query = """
        WITH state_calc AS (
            SELECT
                ts_code,
                trade_date,
                close,
                high,
                low,
                open,
                -- 均线
                AVG(close) OVER (ORDER BY trade_date ROWS 4 PRECEDING) AS ma5,
                AVG(close) OVER (ORDER BY trade_date ROWS 9 PRECEDING) AS ma10,
                AVG(close) OVER (ORDER BY trade_date ROWS 19 PRECEDING) AS ma20,
                AVG(close) OVER (ORDER BY trade_date ROWS 59 PRECEDING) AS ma60,
                AVG(close) OVER (ORDER BY trade_date ROWS 119 PRECEDING) AS ma120,
                -- 标准差
                STDDEV(close) OVER (ORDER BY trade_date ROWS 19 PRECEDING) AS std_20,
                -- 区间高低点
                MAX(high) OVER (ORDER BY trade_date ROWS 19 PRECEDING) AS high_20d,
                MIN(low) OVER (ORDER BY trade_date ROWS 19 PRECEDING) AS low_20d,
                MAX(high) OVER (ORDER BY trade_date ROWS 59 PRECEDING) AS high_60d,
                MIN(low) OVER (ORDER BY trade_date ROWS 59 PRECEDING) AS low_60d
            FROM daily
            WHERE ts_code = ?
            ORDER BY trade_date
        )
        SELECT
            ts_code,
            trade_date,
            close,
            -- 均线值
            ma5, ma10, ma20, ma60, ma120,
            -- 均线多头排列
            CASE WHEN ma5 > ma10 AND ma10 > ma20 AND ma20 > ma60 THEN 1 ELSE 0 END AS ma_bull_alignment,
            -- 均线空头排列
            CASE WHEN ma5 < ma10 AND ma10 < ma20 AND ma20 < ma60 THEN 1 ELSE 0 END AS ma_bear_alignment,
            -- 均线得分 (0-3)
            (CASE WHEN ma5 > ma10 THEN 1 ELSE 0 END +
             CASE WHEN ma10 > ma20 THEN 1 ELSE 0 END +
             CASE WHEN ma20 > ma60 THEN 1 ELSE 0 END) AS ma_score,
            -- 价格偏离MA20
            (close / ma20 - 1) * 100 AS price_ma20_deviation,
            -- 价格在20日区间位置
            (close - low_20d) / NULLIF(high_20d - low_20d, 0) AS position_in_20d_range,
            -- 价格在60日区间位置
            (close - low_60d) / NULLIF(high_60d - low_60d, 0) AS position_in_60d_range,
            -- 布林带
            ma20 AS bb_middle,
            ma20 + 2 * std_20 AS bb_upper,
            ma20 - 2 * std_20 AS bb_lower,
            -- 布林带位置 [0,1]
            (close - (ma20 - 2 * std_20)) / NULLIF(4 * std_20, 0) AS bollinger_position,
            -- Z-score
            (close - ma20) / NULLIF(std_20, 0) AS z_score,
            -- 是否创20日新高/新低
            CASE WHEN close >= high_20d THEN 1 ELSE 0 END AS at_20d_high,
            CASE WHEN close <= low_20d THEN 1 ELSE 0 END AS at_20d_low
        FROM state_calc
        WHERE trade_date <= ?
        ORDER BY trade_date DESC
        LIMIT ?
        """
        return self.conn.execute(query, [ts_code, end_date, lookback]).fetchdf()

    def compute_autocorr_features(self, ts_code: str, end_date: str, window: int = 60) -> pd.DataFrame:
        """
        计算自相关特征

        Returns:
            DataFrame with autocorrelation coefficients
        """
        query = """
        WITH lagged_returns AS (
            SELECT
                ts_code,
                trade_date,
                pct_chg AS return_t,
                vol,
                LAG(pct_chg, 1) OVER (ORDER BY trade_date) AS return_lag1,
                LAG(pct_chg, 2) OVER (ORDER BY trade_date) AS return_lag2,
                LAG(pct_chg, 3) OVER (ORDER BY trade_date) AS return_lag3,
                LAG(pct_chg, 5) OVER (ORDER BY trade_date) AS return_lag5,
                LAG(vol, 1) OVER (ORDER BY trade_date) AS vol_lag1,
                ABS(pct_chg) AS abs_return,
                LAG(ABS(pct_chg), 1) OVER (ORDER BY trade_date) AS abs_return_lag1
            FROM daily
            WHERE ts_code = ? AND trade_date <= ?
        )
        SELECT
            ts_code,
            MAX(trade_date) AS as_of_date,
            -- 收益率自相关
            CORR(return_t, return_lag1) AS return_autocorr_lag1,
            CORR(return_t, return_lag2) AS return_autocorr_lag2,
            CORR(return_t, return_lag3) AS return_autocorr_lag3,
            CORR(return_t, return_lag5) AS return_autocorr_lag5,
            -- 成交量自相关
            CORR(vol, vol_lag1) AS volume_autocorr_lag1,
            -- 波动率聚类
            CORR(abs_return, abs_return_lag1) AS volatility_clustering
        FROM lagged_returns
        WHERE return_lag5 IS NOT NULL
        GROUP BY ts_code
        """
        return self.conn.execute(query, [ts_code, end_date]).fetchdf()

    # ==================== 筹码特征 ====================

    def compute_chip_features(self, ts_code: str, end_date: str, lookback: int = 252) -> pd.DataFrame:
        """
        计算筹码特征

        Returns:
            DataFrame with columns:
            - winner_rate: 获利盘比例
            - chip_concentration: 筹码集中度
            - price_vs_avg_cost: 价格相对平均成本偏离
        """
        query = """
        SELECT
            c.ts_code,
            c.trade_date,
            d.close,
            -- 获利盘比例
            c.winner_rate,
            -- 套牢盘比例
            (1 - c.winner_rate) AS loser_rate,
            -- 历史最高/最低获利比例
            c.his_high AS his_high_winner,
            c.his_low AS his_low_winner,
            -- 成本分布
            c.weight_avg AS avg_cost,
            c.cost_5pct,
            c.cost_15pct,
            c.cost_50pct AS median_cost,
            c.cost_85pct,
            c.cost_95pct,
            -- 价格相对平均成本偏离
            (d.close / c.weight_avg - 1) * 100 AS price_vs_avg_cost,
            -- 价格相对中位成本偏离
            (d.close / c.cost_50pct - 1) * 100 AS price_vs_median_cost,
            -- 筹码集中度 (90%筹码分布区间/平均成本)
            (c.cost_95pct - c.cost_5pct) / NULLIF(c.weight_avg, 0) AS chip_concentration,
            -- 70%筹码集中度
            (c.cost_85pct - c.cost_15pct) / NULLIF(c.weight_avg, 0) AS chip_concentration_70,
            -- 获利比例变化
            c.winner_rate - LAG(c.winner_rate, 1) OVER (PARTITION BY c.ts_code ORDER BY c.trade_date) AS winner_rate_change_1d,
            c.winner_rate - LAG(c.winner_rate, 5) OVER (PARTITION BY c.ts_code ORDER BY c.trade_date) AS winner_rate_change_5d,
            -- 成本变化
            c.weight_avg - LAG(c.weight_avg, 5) OVER (PARTITION BY c.ts_code ORDER BY c.trade_date) AS cost_change_5d,
            -- 价格位置相对筹码
            CASE
                WHEN d.close > c.cost_85pct THEN '价格在筹码上方'
                WHEN d.close > c.cost_50pct THEN '价格在中位成本上方'
                WHEN d.close > c.cost_15pct THEN '价格在中位成本下方'
                ELSE '价格在筹码下方'
            END AS price_chip_relation
        FROM cyq_perf c
        JOIN daily d ON c.ts_code = d.ts_code AND c.trade_date = d.trade_date
        WHERE c.ts_code = ? AND c.trade_date <= ?
        ORDER BY c.trade_date DESC
        LIMIT ?
        """
        return self.conn.execute(query, [ts_code, end_date, lookback]).fetchdf()

    # ==================== 综合计算 ====================

    def compute_all_cross_sectional_features(self, trade_date: str) -> Dict[str, pd.DataFrame]:
        """计算某一天所有横截面特征"""
        result = {
            'market_rank': self.compute_market_rank_features(trade_date),
            'industry_relative': self.compute_industry_relative_features(trade_date),
            'market_sentiment': self.compute_market_sentiment_features(trade_date),
            'moneyflow': self.compute_moneyflow_features(trade_date)
        }
        return result

    def compute_all_timeseries_features(self, ts_code: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """计算某只股票所有时序特征"""
        result = {
            'momentum': self.compute_momentum_features(ts_code, end_date),
            'volatility': self.compute_volatility_features(ts_code, end_date),
            'state': self.compute_state_features(ts_code, end_date),
            'autocorr': self.compute_autocorr_features(ts_code, end_date),
            'chip': self.compute_chip_features(ts_code, end_date)
        }
        return result

    def get_combined_features(self, ts_code: str, trade_date: str) -> pd.DataFrame:
        """获取单只股票某日的所有特征 (横截面+时序)"""
        # 横截面特征
        rank_df = self.compute_market_rank_features(trade_date)
        industry_df = self.compute_industry_relative_features(trade_date)

        # 筛选当前股票
        stock_rank = rank_df[rank_df['ts_code'] == ts_code].copy()
        stock_industry = industry_df[industry_df['ts_code'] == ts_code].copy()

        # 时序特征 (最新一天)
        momentum_df = self.compute_momentum_features(ts_code, trade_date, lookback=1)
        volatility_df = self.compute_volatility_features(ts_code, trade_date, lookback=1)
        state_df = self.compute_state_features(ts_code, trade_date, lookback=1)
        chip_df = self.compute_chip_features(ts_code, trade_date, lookback=1)

        # 合并所有特征
        if len(stock_rank) > 0:
            result = stock_rank.copy()

            if len(stock_industry) > 0:
                for col in ['l1_code', 'l1_name', 'excess_return_industry',
                           'relative_turnover_industry', 'rank_in_industry', 'industry_strength']:
                    if col in stock_industry.columns:
                        result[col] = stock_industry[col].values[0]

            if len(momentum_df) > 0:
                for col in ['return_5d', 'return_20d', 'return_60d', 'momentum_reversal_5_20']:
                    if col in momentum_df.columns:
                        result[col] = momentum_df[col].values[0]

            if len(volatility_df) > 0:
                for col in ['vol_20d', 'annualized_vol_20d']:
                    if col in volatility_df.columns:
                        result[col] = volatility_df[col].values[0]

            if len(state_df) > 0:
                for col in ['ma_bull_alignment', 'bollinger_position', 'z_score', 'position_in_20d_range']:
                    if col in state_df.columns:
                        result[col] = state_df[col].values[0]

            if len(chip_df) > 0:
                for col in ['winner_rate', 'chip_concentration', 'price_vs_avg_cost']:
                    if col in chip_df.columns:
                        result[col] = chip_df[col].values[0]

            return result
        else:
            return pd.DataFrame()


def demo_features():
    """演示特征计算"""
    db_path = "/Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db"

    print("=" * 80)
    print("横截面与时序特征工程演示")
    print("=" * 80)

    with FeatureEngineer(db_path) as fe:
        # 获取最新交易日
        latest_date = fe.conn.execute(
            "SELECT MAX(trade_date) FROM daily"
        ).fetchone()[0]
        print(f"\n最新交易日: {latest_date}")

        # ==================== 横截面特征演示 ====================
        print("\n" + "=" * 40)
        print("一、横截面特征")
        print("=" * 40)

        # 1. 市场排名特征
        print("\n1. 市场排名特征 (前10名涨幅最高):")
        rank_df = fe.compute_market_rank_features(latest_date)
        display_cols = ['ts_code', 'pct_chg', 'pct_chg_rank', 'vol_rank', 'turnover_rank', 'mktcap_rank']
        print(rank_df.nlargest(10, 'pct_chg')[display_cols].to_string(index=False))

        # 2. 行业相对特征
        print("\n2. 行业相对特征 (行业内领涨股):")
        industry_df = fe.compute_industry_relative_features(latest_date)
        leaders = industry_df[industry_df['is_industry_leader'] == 1].sort_values('pct_chg', ascending=False)
        display_cols = ['ts_code', 'l1_name', 'pct_chg', 'excess_return_industry', 'industry_strength']
        print(leaders[display_cols].head(10).to_string(index=False))

        # 3. 市场情绪特征
        print("\n3. 市场情绪特征:")
        sentiment_df = fe.compute_market_sentiment_features(latest_date)
        for col in sentiment_df.columns:
            print(f"   {col}: {sentiment_df[col].values[0]:.4f}" if isinstance(sentiment_df[col].values[0], float)
                  else f"   {col}: {sentiment_df[col].values[0]}")

        # 4. 资金流向特征 (前10)
        print("\n4. 资金流向特征 (主力净流入前10):")
        moneyflow_df = fe.compute_moneyflow_features(latest_date)
        display_cols = ['ts_code', 'net_main_inflow', 'main_buy_ratio']
        print(moneyflow_df[display_cols].head(10).to_string(index=False))

        # ==================== 时序特征演示 ====================
        print("\n" + "=" * 40)
        print("二、时序特征 (以平安银行 000001.SZ 为例)")
        print("=" * 40)

        test_stock = '000001.SZ'

        # 1. 动量特征
        print("\n1. 动量特征 (最近10天):")
        momentum_df = fe.compute_momentum_features(test_stock, latest_date, lookback=10)
        display_cols = ['trade_date', 'close', 'return_1d', 'return_5d', 'return_20d', 'momentum_reversal_5_20']
        print(momentum_df[display_cols].to_string(index=False))

        # 2. 波动率特征
        print("\n2. 波动率特征 (最近10天):")
        volatility_df = fe.compute_volatility_features(test_stock, latest_date, lookback=10)
        display_cols = ['trade_date', 'vol_5d', 'vol_20d', 'annualized_vol_20d', 'vol_ratio_5_20']
        print(volatility_df[display_cols].to_string(index=False))

        # 3. 状态特征
        print("\n3. 状态特征 (最近10天):")
        state_df = fe.compute_state_features(test_stock, latest_date, lookback=10)
        display_cols = ['trade_date', 'close', 'ma_score', 'bollinger_position', 'z_score', 'position_in_20d_range']
        print(state_df[display_cols].to_string(index=False))

        # 4. 自相关特征
        print("\n4. 自相关特征:")
        autocorr_df = fe.compute_autocorr_features(test_stock, latest_date)
        for col in autocorr_df.columns:
            val = autocorr_df[col].values[0]
            if isinstance(val, float):
                print(f"   {col}: {val:.4f}")
            else:
                print(f"   {col}: {val}")

        # 5. 筹码特征
        print("\n5. 筹码特征 (最近10天):")
        chip_df = fe.compute_chip_features(test_stock, latest_date, lookback=10)
        display_cols = ['trade_date', 'close', 'winner_rate', 'chip_concentration', 'price_vs_avg_cost']
        print(chip_df[display_cols].to_string(index=False))

        # ==================== 综合特征 ====================
        print("\n" + "=" * 40)
        print("三、综合特征视图")
        print("=" * 40)

        combined = fe.get_combined_features(test_stock, latest_date)
        if len(combined) > 0:
            print(f"\n{test_stock} 在 {latest_date} 的综合特征:")
            for col in combined.columns:
                val = combined[col].values[0]
                if isinstance(val, float):
                    print(f"   {col}: {val:.4f}")
                else:
                    print(f"   {col}: {val}")

    print("\n" + "=" * 80)
    print("特征计算演示完成!")
    print("=" * 80)


if __name__ == "__main__":
    demo_features()
