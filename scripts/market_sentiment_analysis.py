#!/usr/bin/env python3
"""
A股市场情绪指标体系分析
=====================

本模块实现完整的市场情绪指标体系，包括：
1. 市场宽度指标 (Market Breadth)
2. 资金情绪指标 (Money Flow Sentiment)
3. 波动率情绪指标 (Volatility Sentiment)
4. 换手率情绪指标 (Turnover Sentiment)
5. 综合情绪指数 (Composite Sentiment Index)
6. 情绪与收益关系分析 (Sentiment-Return Relationship)

Author: Market Sentiment Analysis System
Date: 2026-01-31
"""

import duckdb
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')

# 数据库路径
DB_PATH = "/Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db"
REPORT_PATH = "/Users/allen/workspace/python/stock/Tushare-DuckDB/reports/quality_check/market_sentiment_indicators.md"

class MarketSentimentAnalyzer:
    """市场情绪分析器"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = duckdb.connect(db_path, read_only=True)

    def close(self):
        self.conn.close()

    # ==================== 1. 市场宽度指标 ====================

    def calc_advance_decline_ratio(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        计算涨跌家数比 (Advance/Decline Ratio)

        涨跌家数比 = 上涨家数 / 下跌家数
        - >1.5: 强势市场
        - 0.67-1.5: 中性市场
        - <0.67: 弱势市场
        """
        query = f"""
        SELECT
            trade_date,
            COUNT(*) as total_stocks,
            SUM(CASE WHEN pct_chg > 0 THEN 1 ELSE 0 END) as advance_count,
            SUM(CASE WHEN pct_chg < 0 THEN 1 ELSE 0 END) as decline_count,
            SUM(CASE WHEN pct_chg = 0 THEN 1 ELSE 0 END) as unchanged_count,
            SUM(CASE WHEN pct_chg > 0 THEN 1 ELSE 0 END) * 1.0 /
                NULLIF(SUM(CASE WHEN pct_chg < 0 THEN 1 ELSE 0 END), 0) as ad_ratio,
            AVG(pct_chg) as avg_pct_chg
        FROM daily
        WHERE trade_date BETWEEN '{start_date}' AND '{end_date}'
          AND ts_code NOT LIKE '%BJ%'  -- 排除北交所
        GROUP BY trade_date
        ORDER BY trade_date
        """
        return self.conn.execute(query).fetchdf()

    def calc_limit_up_down(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        计算涨停/跌停数量

        A股涨跌停限制：
        - 主板/中小板：±10%
        - 科创板/创业板：±20%
        - ST股票：±5%
        """
        query = f"""
        SELECT
            trade_date,
            -- 涨停：涨幅接近涨停限制
            SUM(CASE
                WHEN ts_code LIKE '68%' OR ts_code LIKE '30%' THEN  -- 科创板/创业板
                    CASE WHEN pct_chg >= 19.5 THEN 1 ELSE 0 END
                WHEN ts_code LIKE 'ST%' THEN  -- ST股票
                    CASE WHEN pct_chg >= 4.8 THEN 1 ELSE 0 END
                ELSE  -- 主板
                    CASE WHEN pct_chg >= 9.8 THEN 1 ELSE 0 END
            END) as limit_up_count,
            -- 跌停：跌幅接近跌停限制
            SUM(CASE
                WHEN ts_code LIKE '68%' OR ts_code LIKE '30%' THEN
                    CASE WHEN pct_chg <= -19.5 THEN 1 ELSE 0 END
                WHEN ts_code LIKE 'ST%' THEN
                    CASE WHEN pct_chg <= -4.8 THEN 1 ELSE 0 END
                ELSE
                    CASE WHEN pct_chg <= -9.8 THEN 1 ELSE 0 END
            END) as limit_down_count,
            COUNT(*) as total_stocks
        FROM daily
        WHERE trade_date BETWEEN '{start_date}' AND '{end_date}'
          AND ts_code NOT LIKE '%BJ%'
        GROUP BY trade_date
        ORDER BY trade_date
        """
        df = self.conn.execute(query).fetchdf()
        df['limit_up_ratio'] = df['limit_up_count'] / df['total_stocks']
        df['limit_down_ratio'] = df['limit_down_count'] / df['total_stocks']
        df['net_limit'] = df['limit_up_count'] - df['limit_down_count']
        return df

    def calc_new_highs_lows(self, start_date: str, end_date: str, lookback: int = 250) -> pd.DataFrame:
        """
        计算创新高/新低数量 (52周新高新低)

        - 新高/新低比 > 2: 牛市信号
        - 新高/新低比 < 0.5: 熊市信号
        """
        query = f"""
        WITH price_history AS (
            SELECT
                ts_code,
                trade_date,
                high,
                low,
                MAX(high) OVER (
                    PARTITION BY ts_code
                    ORDER BY trade_date
                    ROWS BETWEEN {lookback} PRECEDING AND 1 PRECEDING
                ) as max_high_prev,
                MIN(low) OVER (
                    PARTITION BY ts_code
                    ORDER BY trade_date
                    ROWS BETWEEN {lookback} PRECEDING AND 1 PRECEDING
                ) as min_low_prev
            FROM daily
            WHERE ts_code NOT LIKE '%BJ%'
        )
        SELECT
            trade_date,
            SUM(CASE WHEN high > max_high_prev THEN 1 ELSE 0 END) as new_high_count,
            SUM(CASE WHEN low < min_low_prev THEN 1 ELSE 0 END) as new_low_count,
            COUNT(*) as total_stocks
        FROM price_history
        WHERE trade_date BETWEEN '{start_date}' AND '{end_date}'
          AND max_high_prev IS NOT NULL
        GROUP BY trade_date
        ORDER BY trade_date
        """
        df = self.conn.execute(query).fetchdf()
        df['nh_nl_ratio'] = df['new_high_count'] / df['new_low_count'].replace(0, 1)
        df['net_new_high'] = df['new_high_count'] - df['new_low_count']
        return df

    def calc_adl(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        计算腾落指数 (Advance-Decline Line, ADL)

        ADL = 累计(上涨家数 - 下跌家数)

        ADL与指数背离是重要的市场见顶/见底信号
        """
        ad_df = self.calc_advance_decline_ratio(start_date, end_date)
        ad_df['ad_diff'] = ad_df['advance_count'] - ad_df['decline_count']
        ad_df['adl'] = ad_df['ad_diff'].cumsum()

        # 计算ADL的移动平均
        ad_df['adl_ma10'] = ad_df['adl'].rolling(10).mean()
        ad_df['adl_ma30'] = ad_df['adl'].rolling(30).mean()

        return ad_df

    def calc_mcclellan_oscillator(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        计算McClellan Oscillator

        McClellan Oscillator = EMA(AD_diff, 19) - EMA(AD_diff, 39)

        - > 0: 市场强势
        - < 0: 市场弱势
        - 极端值 (>100 或 <-100): 超买/超卖信号
        """
        ad_df = self.calc_advance_decline_ratio(start_date, end_date)
        ad_df['ad_diff'] = ad_df['advance_count'] - ad_df['decline_count']

        # 计算EMA
        ad_df['ema_19'] = ad_df['ad_diff'].ewm(span=19, adjust=False).mean()
        ad_df['ema_39'] = ad_df['ad_diff'].ewm(span=39, adjust=False).mean()
        ad_df['mcclellan_osc'] = ad_df['ema_19'] - ad_df['ema_39']

        # McClellan Summation Index (累积)
        ad_df['mcclellan_sum'] = ad_df['mcclellan_osc'].cumsum()

        return ad_df

    # ==================== 2. 资金情绪指标 ====================

    def calc_money_flow_sentiment(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        计算资金流向情绪指标

        包括：
        - 主力净流入市场总量
        - 大单/超大单净流入
        - 中小单净流入
        """
        query = f"""
        SELECT
            trade_date,
            -- 主力资金 = 大单 + 超大单
            SUM(buy_lg_amount + buy_elg_amount - sell_lg_amount - sell_elg_amount) as main_net_inflow,
            SUM(buy_elg_amount - sell_elg_amount) as elg_net_inflow,
            SUM(buy_lg_amount - sell_lg_amount) as lg_net_inflow,
            SUM(buy_md_amount - sell_md_amount) as md_net_inflow,
            SUM(buy_sm_amount - sell_sm_amount) as sm_net_inflow,
            SUM(net_mf_amount) as total_net_flow,
            COUNT(*) as stock_count,
            -- 资金流入/流出家数
            SUM(CASE WHEN net_mf_amount > 0 THEN 1 ELSE 0 END) as inflow_stocks,
            SUM(CASE WHEN net_mf_amount < 0 THEN 1 ELSE 0 END) as outflow_stocks
        FROM moneyflow
        WHERE trade_date BETWEEN '{start_date}' AND '{end_date}'
        GROUP BY trade_date
        ORDER BY trade_date
        """
        df = self.conn.execute(query).fetchdf()

        # 计算移动平均
        df['main_net_ma5'] = df['main_net_inflow'].rolling(5).mean()
        df['main_net_ma20'] = df['main_net_inflow'].rolling(20).mean()

        # 资金流入比
        df['inflow_ratio'] = df['inflow_stocks'] / df['outflow_stocks'].replace(0, 1)

        return df

    def calc_margin_sentiment(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        计算融资融券情绪指标

        - 融资余额变化率：反映杠杆资金情绪
        - 融券余额变化：反映做空情绪
        """
        query = f"""
        SELECT
            trade_date,
            SUM(rzye) as total_rzye,           -- 融资余额
            SUM(rqye) as total_rqye,           -- 融券余额
            SUM(rzmre) as total_rzmre,         -- 融资买入额
            SUM(rzche) as total_rzche,         -- 融资偿还额
            SUM(rqyl) as total_rqyl,           -- 融券余量
            SUM(rqmcl) as total_rqmcl,         -- 融券卖出量
            SUM(rzrqye) as total_rzrqye,       -- 融资融券余额
            COUNT(DISTINCT ts_code) as stock_count
        FROM margin_detail
        WHERE trade_date BETWEEN '{start_date}' AND '{end_date}'
        GROUP BY trade_date
        ORDER BY trade_date
        """
        df = self.conn.execute(query).fetchdf()

        # 计算变化率
        df['rzye_chg'] = df['total_rzye'].pct_change()
        df['rqye_chg'] = df['total_rqye'].pct_change()

        # 融资净买入
        df['rz_net_buy'] = df['total_rzmre'] - df['total_rzche']

        # 融资/融券比
        df['rz_rq_ratio'] = df['total_rzye'] / df['total_rqye'].replace(0, 1)

        # 移动平均
        df['rzye_ma5'] = df['total_rzye'].rolling(5).mean()
        df['rzye_ma20'] = df['total_rzye'].rolling(20).mean()

        return df

    # ==================== 3. 波动率情绪指标 ====================

    def calc_volatility_sentiment(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        计算波动率情绪指标

        包括：
        - 市场整体波动率
        - 波动率变化率
        - 恐慌指数代理
        - 波动率偏度
        """
        query = f"""
        SELECT
            trade_date,
            AVG(ABS(pct_chg)) as avg_abs_return,
            STDDEV(pct_chg) as return_std,
            MAX(pct_chg) as max_return,
            MIN(pct_chg) as min_return,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY pct_chg) as pct_95,
            PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY pct_chg) as pct_05,
            PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY pct_chg) as pct_75,
            PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY pct_chg) as pct_25,
            AVG(pct_chg) as avg_return,
            COUNT(*) as stock_count
        FROM daily
        WHERE trade_date BETWEEN '{start_date}' AND '{end_date}'
          AND ts_code NOT LIKE '%BJ%'
          AND pct_chg IS NOT NULL
        GROUP BY trade_date
        ORDER BY trade_date
        """
        df = self.conn.execute(query).fetchdf()

        # 市场波动率（标准差）
        df['volatility'] = df['return_std']

        # 波动率变化率
        df['vol_chg'] = df['volatility'].pct_change()

        # 恐慌指数代理：下行波动率 / 上行波动率
        df['downside_vol'] = df['avg_return'] - df['pct_05']
        df['upside_vol'] = df['pct_95'] - df['avg_return']
        df['fear_index'] = df['downside_vol'] / df['upside_vol'].replace(0, 1)

        # 波动率偏度：衡量收益分布的不对称性
        df['vol_skew'] = (df['pct_95'] + df['pct_05'] - 2 * df['avg_return']) / (df['pct_95'] - df['pct_05']).replace(0, 1)

        # 日内波动幅度
        df['intraday_range'] = df['max_return'] - df['min_return']

        # 波动率移动平均
        df['vol_ma5'] = df['volatility'].rolling(5).mean()
        df['vol_ma20'] = df['volatility'].rolling(20).mean()

        # 波动率相对水平
        df['vol_percentile'] = df['volatility'].rolling(60).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5
        )

        return df

    def calc_intraday_volatility(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        计算日内波动率指标
        """
        query = f"""
        SELECT
            trade_date,
            AVG((high - low) / NULLIF(pre_close, 0) * 100) as avg_intraday_range,
            AVG((high - open) / NULLIF(open, 0) * 100) as avg_upper_shadow,
            AVG((open - low) / NULLIF(open, 0) * 100) as avg_lower_shadow,
            AVG(ABS(close - open) / NULLIF(open, 0) * 100) as avg_body_size
        FROM daily
        WHERE trade_date BETWEEN '{start_date}' AND '{end_date}'
          AND ts_code NOT LIKE '%BJ%'
          AND pre_close > 0 AND open > 0
        GROUP BY trade_date
        ORDER BY trade_date
        """
        return self.conn.execute(query).fetchdf()

    # ==================== 4. 换手率情绪指标 ====================

    def calc_turnover_sentiment(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        计算换手率情绪指标

        包括：
        - 市场平均换手率
        - 换手率分布
        - 极端换手率股票比例
        """
        query = f"""
        SELECT
            trade_date,
            AVG(turnover_rate) as avg_turnover,
            AVG(turnover_rate_f) as avg_turnover_free,
            STDDEV(turnover_rate) as turnover_std,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY turnover_rate) as turnover_median,
            PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY turnover_rate) as turnover_p90,
            PERCENTILE_CONT(0.1) WITHIN GROUP (ORDER BY turnover_rate) as turnover_p10,
            AVG(volume_ratio) as avg_volume_ratio,
            COUNT(*) as stock_count,
            -- 高换手率股票比例 (>10%)
            SUM(CASE WHEN turnover_rate > 10 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as high_turnover_pct,
            -- 低换手率股票比例 (<1%)
            SUM(CASE WHEN turnover_rate < 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as low_turnover_pct
        FROM daily_basic
        WHERE trade_date BETWEEN '{start_date}' AND '{end_date}'
          AND ts_code NOT LIKE '%BJ%'
          AND turnover_rate IS NOT NULL
        GROUP BY trade_date
        ORDER BY trade_date
        """
        df = self.conn.execute(query).fetchdf()

        # 换手率变化
        df['turnover_chg'] = df['avg_turnover'].pct_change()

        # 换手率离散度
        df['turnover_dispersion'] = df['turnover_std'] / df['avg_turnover'].replace(0, 1)

        # 换手率移动平均
        df['turnover_ma5'] = df['avg_turnover'].rolling(5).mean()
        df['turnover_ma20'] = df['avg_turnover'].rolling(20).mean()

        # 换手率相对水平
        df['turnover_rel'] = df['avg_turnover'] / df['turnover_ma20']

        return df

    def calc_volume_sentiment(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        计算成交量情绪指标
        """
        query = f"""
        SELECT
            trade_date,
            SUM(amount) as total_amount,
            SUM(vol) as total_vol,
            COUNT(*) as stock_count,
            AVG(amount) as avg_amount,
            STDDEV(amount) as amount_std
        FROM daily
        WHERE trade_date BETWEEN '{start_date}' AND '{end_date}'
          AND ts_code NOT LIKE '%BJ%'
        GROUP BY trade_date
        ORDER BY trade_date
        """
        df = self.conn.execute(query).fetchdf()

        # 成交额变化
        df['amount_chg'] = df['total_amount'].pct_change()

        # 成交额移动平均
        df['amount_ma5'] = df['total_amount'].rolling(5).mean()
        df['amount_ma20'] = df['total_amount'].rolling(20).mean()

        # 量比
        df['volume_ratio'] = df['total_amount'] / df['amount_ma5']

        return df

    # ==================== 5. 综合情绪指数 ====================

    def calc_composite_sentiment_index(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        计算综合情绪指数

        采用多指标加权方法：
        1. 市场宽度指标权重: 30%
        2. 资金流向指标权重: 25%
        3. 波动率指标权重: 20%
        4. 换手率指标权重: 15%
        5. 融资融券指标权重: 10%
        """
        # 获取各类指标
        ad_df = self.calc_advance_decline_ratio(start_date, end_date)
        mcclellan_df = self.calc_mcclellan_oscillator(start_date, end_date)
        money_flow_df = self.calc_money_flow_sentiment(start_date, end_date)
        vol_df = self.calc_volatility_sentiment(start_date, end_date)
        turnover_df = self.calc_turnover_sentiment(start_date, end_date)
        margin_df = self.calc_margin_sentiment(start_date, end_date)

        # 合并数据
        result = ad_df[['trade_date', 'ad_ratio', 'avg_pct_chg']].copy()

        # 标准化函数
        def normalize(series, window=60):
            """将指标标准化到0-100范围"""
            rolling_min = series.rolling(window, min_periods=20).min()
            rolling_max = series.rolling(window, min_periods=20).max()
            normalized = (series - rolling_min) / (rolling_max - rolling_min + 1e-10) * 100
            return normalized

        # 1. 市场宽度指标 (30%)
        result = result.merge(
            mcclellan_df[['trade_date', 'mcclellan_osc']],
            on='trade_date', how='left'
        )
        result['breadth_score'] = normalize(result['ad_ratio'].fillna(1))
        result['mcclellan_score'] = normalize(result['mcclellan_osc'].fillna(0))
        result['width_sentiment'] = 0.6 * result['breadth_score'] + 0.4 * result['mcclellan_score']

        # 2. 资金流向指标 (25%)
        result = result.merge(
            money_flow_df[['trade_date', 'main_net_inflow', 'inflow_ratio']],
            on='trade_date', how='left'
        )
        result['money_flow_score'] = normalize(result['main_net_inflow'].fillna(0))
        result['inflow_ratio_score'] = normalize(result['inflow_ratio'].fillna(1))
        result['money_sentiment'] = 0.7 * result['money_flow_score'] + 0.3 * result['inflow_ratio_score']

        # 3. 波动率指标 (20%) - 低波动率对应高情绪
        result = result.merge(
            vol_df[['trade_date', 'volatility', 'fear_index']],
            on='trade_date', how='left'
        )
        result['vol_score'] = 100 - normalize(result['volatility'].fillna(result['volatility'].median()))
        result['fear_score'] = 100 - normalize(result['fear_index'].fillna(1))
        result['vol_sentiment'] = 0.6 * result['vol_score'] + 0.4 * result['fear_score']

        # 4. 换手率指标 (15%) - 适度换手率最好
        result = result.merge(
            turnover_df[['trade_date', 'avg_turnover', 'turnover_rel']],
            on='trade_date', how='left'
        )
        # 换手率偏离度评分（过高或过低都不好）
        result['turnover_deviation'] = abs(result['turnover_rel'].fillna(1) - 1)
        result['turnover_score'] = 100 - normalize(result['turnover_deviation'])
        result['turnover_sentiment'] = result['turnover_score']

        # 5. 融资融券指标 (10%)
        result = result.merge(
            margin_df[['trade_date', 'rzye_chg', 'rz_rq_ratio']],
            on='trade_date', how='left'
        )
        result['margin_chg_score'] = normalize(result['rzye_chg'].fillna(0))
        result['margin_sentiment'] = result['margin_chg_score']

        # 综合情绪指数
        result['composite_sentiment'] = (
            0.30 * result['width_sentiment'].fillna(50) +
            0.25 * result['money_sentiment'].fillna(50) +
            0.20 * result['vol_sentiment'].fillna(50) +
            0.15 * result['turnover_sentiment'].fillna(50) +
            0.10 * result['margin_sentiment'].fillna(50)
        )

        # 情绪移动平均
        result['sentiment_ma5'] = result['composite_sentiment'].rolling(5).mean()
        result['sentiment_ma20'] = result['composite_sentiment'].rolling(20).mean()

        # 情绪变化率
        result['sentiment_chg'] = result['composite_sentiment'].pct_change()

        return result

    def identify_sentiment_cycles(self, sentiment_df: pd.DataFrame) -> Dict:
        """
        识别情绪周期

        基于综合情绪指数识别：
        - 贪婪阶段 (>70)
        - 乐观阶段 (55-70)
        - 中性阶段 (45-55)
        - 悲观阶段 (30-45)
        - 恐惧阶段 (<30)
        """
        df = sentiment_df.copy()

        # 定义情绪阶段
        def classify_sentiment(score):
            if pd.isna(score):
                return 'Unknown'
            elif score >= 70:
                return 'Greed'
            elif score >= 55:
                return 'Optimism'
            elif score >= 45:
                return 'Neutral'
            elif score >= 30:
                return 'Pessimism'
            else:
                return 'Fear'

        df['sentiment_phase'] = df['composite_sentiment'].apply(classify_sentiment)

        # 统计各阶段分布
        phase_dist = df['sentiment_phase'].value_counts(normalize=True) * 100

        # 识别情绪拐点
        df['sentiment_trend'] = np.where(
            df['sentiment_ma5'] > df['sentiment_ma20'], 'Bullish', 'Bearish'
        )

        # 检测极端情绪
        df['extreme_sentiment'] = np.where(
            df['composite_sentiment'] >= 75, 'Extreme Greed',
            np.where(df['composite_sentiment'] <= 25, 'Extreme Fear', 'Normal')
        )

        return {
            'data': df,
            'phase_distribution': phase_dist.to_dict(),
            'current_phase': df['sentiment_phase'].iloc[-1] if len(df) > 0 else 'Unknown',
            'current_trend': df['sentiment_trend'].iloc[-1] if len(df) > 0 else 'Unknown',
            'current_extreme': df['extreme_sentiment'].iloc[-1] if len(df) > 0 else 'Normal'
        }

    def generate_sentiment_alerts(self, sentiment_df: pd.DataFrame) -> List[Dict]:
        """
        生成情绪预警信号
        """
        alerts = []
        df = sentiment_df.copy()

        if len(df) < 5:
            return alerts

        latest = df.iloc[-1]
        prev_5d = df.iloc[-5:]

        # 1. 极端情绪预警
        if latest['composite_sentiment'] >= 75:
            alerts.append({
                'type': 'EXTREME_GREED',
                'level': 'HIGH',
                'message': f"市场处于极度贪婪状态 (情绪指数: {latest['composite_sentiment']:.1f})，注意回调风险",
                'value': latest['composite_sentiment']
            })
        elif latest['composite_sentiment'] <= 25:
            alerts.append({
                'type': 'EXTREME_FEAR',
                'level': 'HIGH',
                'message': f"市场处于极度恐惧状态 (情绪指数: {latest['composite_sentiment']:.1f})，可能存在抄底机会",
                'value': latest['composite_sentiment']
            })

        # 2. 情绪快速变化预警
        sentiment_chg = prev_5d['composite_sentiment'].iloc[-1] - prev_5d['composite_sentiment'].iloc[0]
        if abs(sentiment_chg) > 15:
            direction = '上升' if sentiment_chg > 0 else '下降'
            alerts.append({
                'type': 'RAPID_CHANGE',
                'level': 'MEDIUM',
                'message': f"情绪指数5日内快速{direction} {abs(sentiment_chg):.1f} 点",
                'value': sentiment_chg
            })

        # 3. 趋势反转预警
        if len(df) >= 20:
            ma5 = df['sentiment_ma5'].iloc[-1]
            ma20 = df['sentiment_ma20'].iloc[-1]
            prev_ma5 = df['sentiment_ma5'].iloc[-2]
            prev_ma20 = df['sentiment_ma20'].iloc[-2]

            # 金叉
            if prev_ma5 <= prev_ma20 and ma5 > ma20:
                alerts.append({
                    'type': 'GOLDEN_CROSS',
                    'level': 'MEDIUM',
                    'message': "情绪指数MA5上穿MA20，市场情绪转暖",
                    'value': ma5 - ma20
                })
            # 死叉
            elif prev_ma5 >= prev_ma20 and ma5 < ma20:
                alerts.append({
                    'type': 'DEATH_CROSS',
                    'level': 'MEDIUM',
                    'message': "情绪指数MA5下穿MA20，市场情绪转冷",
                    'value': ma5 - ma20
                })

        return alerts

    # ==================== 6. 情绪与收益关系 ====================

    def analyze_sentiment_return_relationship(self, start_date: str, end_date: str) -> Dict:
        """
        分析情绪指标与未来收益的关系
        """
        # 获取综合情绪指数
        sentiment_df = self.calc_composite_sentiment_index(start_date, end_date)

        # 获取市场收益（使用沪深300或全市场平均）
        market_return_query = f"""
        SELECT
            trade_date,
            AVG(pct_chg) as market_return,
            SUM(amount) as total_amount
        FROM daily
        WHERE trade_date BETWEEN '{start_date}' AND '{end_date}'
          AND ts_code NOT LIKE '%BJ%'
        GROUP BY trade_date
        ORDER BY trade_date
        """
        market_df = self.conn.execute(market_return_query).fetchdf()

        # 合并数据
        df = sentiment_df.merge(market_df, on='trade_date', how='left')

        # 计算未来收益
        for period in [1, 5, 10, 20]:
            df[f'future_return_{period}d'] = df['market_return'].rolling(period).sum().shift(-period)

        # 分析不同情绪水平下的未来收益
        def analyze_by_sentiment_level(df, sentiment_col, return_col):
            """分析不同情绪水平对应的平均未来收益"""
            result = {}
            for level, (low, high) in {
                'Very Low (0-20)': (0, 20),
                'Low (20-40)': (20, 40),
                'Medium (40-60)': (40, 60),
                'High (60-80)': (60, 80),
                'Very High (80-100)': (80, 100)
            }.items():
                mask = (df[sentiment_col] >= low) & (df[sentiment_col] < high)
                subset = df.loc[mask, return_col]
                if len(subset) > 10:
                    result[level] = {
                        'mean': subset.mean(),
                        'std': subset.std(),
                        'count': len(subset),
                        'sharpe': subset.mean() / (subset.std() + 1e-10)
                    }
            return result

        analysis_results = {}
        for period in [1, 5, 10, 20]:
            analysis_results[f'{period}d_return'] = analyze_by_sentiment_level(
                df, 'composite_sentiment', f'future_return_{period}d'
            )

        # 计算相关性
        correlations = {}
        for period in [1, 5, 10, 20]:
            valid_df = df.dropna(subset=['composite_sentiment', f'future_return_{period}d'])
            if len(valid_df) > 30:
                corr = valid_df['composite_sentiment'].corr(valid_df[f'future_return_{period}d'])
                correlations[f'{period}d'] = corr

        # 逆向策略回测
        df['sentiment_quintile'] = pd.qcut(df['composite_sentiment'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
        contrarian_results = {}
        for q in [1, 5]:  # 极端情绪组
            mask = df['sentiment_quintile'] == q
            for period in [5, 10, 20]:
                subset = df.loc[mask, f'future_return_{period}d'].dropna()
                if len(subset) > 10:
                    contrarian_results[f'Q{q}_{period}d'] = {
                        'mean': subset.mean(),
                        'std': subset.std(),
                        'count': len(subset)
                    }

        return {
            'data': df,
            'sentiment_level_analysis': analysis_results,
            'correlations': correlations,
            'contrarian_strategy': contrarian_results
        }

    def identify_sentiment_turning_points(self, sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """
        识别情绪拐点
        """
        df = sentiment_df.copy()

        # 使用简单的峰谷检测
        df['is_local_max'] = (
            (df['composite_sentiment'] > df['composite_sentiment'].shift(1)) &
            (df['composite_sentiment'] > df['composite_sentiment'].shift(-1)) &
            (df['composite_sentiment'] > df['composite_sentiment'].shift(2)) &
            (df['composite_sentiment'] > df['composite_sentiment'].shift(-2))
        )

        df['is_local_min'] = (
            (df['composite_sentiment'] < df['composite_sentiment'].shift(1)) &
            (df['composite_sentiment'] < df['composite_sentiment'].shift(-1)) &
            (df['composite_sentiment'] < df['composite_sentiment'].shift(2)) &
            (df['composite_sentiment'] < df['composite_sentiment'].shift(-2))
        )

        df['turning_point'] = np.where(
            df['is_local_max'], 'Peak',
            np.where(df['is_local_min'], 'Trough', None)
        )

        return df

    # ==================== 报告生成 ====================

    def generate_report(self, lookback_days: int = 365) -> str:
        """
        生成完整的市场情绪分析报告
        """
        # 计算日期范围
        end_date = self.conn.execute("SELECT MAX(trade_date) FROM daily").fetchone()[0]

        # 计算开始日期（约1年前）
        from datetime import datetime, timedelta
        end_dt = datetime.strptime(end_date, '%Y%m%d')
        start_dt = end_dt - timedelta(days=lookback_days)
        start_date = start_dt.strftime('%Y%m%d')

        report = []
        report.append("# A股市场情绪指标体系分析报告")
        report.append(f"\n**报告生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"\n**分析期间**: {start_date} 至 {end_date}")
        report.append(f"\n**数据来源**: Tushare DuckDB 数据库")

        # ===== 1. 市场宽度指标 =====
        report.append("\n\n---")
        report.append("\n## 1. 市场宽度指标 (Market Breadth)")

        # 1.1 涨跌家数比
        report.append("\n### 1.1 涨跌家数比 (Advance/Decline Ratio)")
        ad_df = self.calc_advance_decline_ratio(start_date, end_date)
        if len(ad_df) > 0:
            latest_ad = ad_df.iloc[-1]
            report.append(f"\n**最新数据 ({latest_ad['trade_date']})**:")
            report.append(f"- 上涨家数: {int(latest_ad['advance_count'])}")
            report.append(f"- 下跌家数: {int(latest_ad['decline_count'])}")
            report.append(f"- 平盘家数: {int(latest_ad['unchanged_count'])}")
            report.append(f"- 涨跌比: {latest_ad['ad_ratio']:.2f}")

            # 统计分析
            avg_ad_ratio = ad_df['ad_ratio'].mean()
            report.append(f"\n**期间统计**:")
            report.append(f"- 平均涨跌比: {avg_ad_ratio:.2f}")
            report.append(f"- 上涨占优天数: {(ad_df['ad_ratio'] > 1).sum()} ({(ad_df['ad_ratio'] > 1).mean()*100:.1f}%)")
            report.append(f"- 下跌占优天数: {(ad_df['ad_ratio'] < 1).sum()} ({(ad_df['ad_ratio'] < 1).mean()*100:.1f}%)")

            # 近期趋势
            recent_5d = ad_df.tail(5)['ad_ratio'].mean()
            recent_20d = ad_df.tail(20)['ad_ratio'].mean()
            report.append(f"\n**近期趋势**:")
            report.append(f"- 近5日平均涨跌比: {recent_5d:.2f}")
            report.append(f"- 近20日平均涨跌比: {recent_20d:.2f}")

            # 最近10个交易日数据
            report.append("\n**近10个交易日涨跌家数**:")
            report.append("\n| 日期 | 上涨 | 下跌 | 平盘 | 涨跌比 |")
            report.append("|------|------|------|------|--------|")
            for _, row in ad_df.tail(10).iterrows():
                report.append(f"| {row['trade_date']} | {int(row['advance_count'])} | {int(row['decline_count'])} | {int(row['unchanged_count'])} | {row['ad_ratio']:.2f} |")

        # 1.2 涨停/跌停数量
        report.append("\n\n### 1.2 涨停/跌停数量")
        limit_df = self.calc_limit_up_down(start_date, end_date)
        if len(limit_df) > 0:
            latest_limit = limit_df.iloc[-1]
            report.append(f"\n**最新数据 ({latest_limit['trade_date']})**:")
            report.append(f"- 涨停家数: {int(latest_limit['limit_up_count'])}")
            report.append(f"- 跌停家数: {int(latest_limit['limit_down_count'])}")
            report.append(f"- 涨停比例: {latest_limit['limit_up_ratio']*100:.2f}%")
            report.append(f"- 跌停比例: {latest_limit['limit_down_ratio']*100:.2f}%")

            report.append(f"\n**期间统计**:")
            report.append(f"- 日均涨停家数: {limit_df['limit_up_count'].mean():.0f}")
            report.append(f"- 日均跌停家数: {limit_df['limit_down_count'].mean():.0f}")
            report.append(f"- 最多涨停日: {limit_df.loc[limit_df['limit_up_count'].idxmax(), 'trade_date']} ({int(limit_df['limit_up_count'].max())}家)")
            report.append(f"- 最多跌停日: {limit_df.loc[limit_df['limit_down_count'].idxmax(), 'trade_date']} ({int(limit_df['limit_down_count'].max())}家)")

        # 1.3 创新高/新低
        report.append("\n\n### 1.3 创新高/新低数量 (52周)")
        nh_nl_df = self.calc_new_highs_lows(start_date, end_date)
        if len(nh_nl_df) > 0:
            latest_nhnl = nh_nl_df.iloc[-1]
            report.append(f"\n**最新数据 ({latest_nhnl['trade_date']})**:")
            report.append(f"- 创新高家数: {int(latest_nhnl['new_high_count'])}")
            report.append(f"- 创新低家数: {int(latest_nhnl['new_low_count'])}")
            report.append(f"- 新高/新低比: {latest_nhnl['nh_nl_ratio']:.2f}")

            report.append(f"\n**期间统计**:")
            report.append(f"- 日均创新高: {nh_nl_df['new_high_count'].mean():.0f}")
            report.append(f"- 日均创新低: {nh_nl_df['new_low_count'].mean():.0f}")

        # 1.4 腾落指数
        report.append("\n\n### 1.4 腾落指数 (ADL)")
        adl_df = self.calc_adl(start_date, end_date)
        if len(adl_df) > 0:
            latest_adl = adl_df.iloc[-1]
            report.append(f"\n**最新数据 ({latest_adl['trade_date']})**:")
            report.append(f"- ADL值: {latest_adl['adl']:.0f}")
            report.append(f"- ADL MA10: {latest_adl['adl_ma10']:.0f}" if pd.notna(latest_adl['adl_ma10']) else "- ADL MA10: N/A")
            report.append(f"- ADL MA30: {latest_adl['adl_ma30']:.0f}" if pd.notna(latest_adl['adl_ma30']) else "- ADL MA30: N/A")

            # ADL趋势判断
            if pd.notna(latest_adl['adl_ma10']) and pd.notna(latest_adl['adl_ma30']):
                if latest_adl['adl_ma10'] > latest_adl['adl_ma30']:
                    report.append("\n**ADL趋势**: 短期均线在长期均线之上，市场宽度走强")
                else:
                    report.append("\n**ADL趋势**: 短期均线在长期均线之下，市场宽度走弱")

        # 1.5 McClellan Oscillator
        report.append("\n\n### 1.5 McClellan Oscillator")
        mcclellan_df = self.calc_mcclellan_oscillator(start_date, end_date)
        if len(mcclellan_df) > 0:
            latest_mc = mcclellan_df.iloc[-1]
            report.append(f"\n**最新数据 ({latest_mc['trade_date']})**:")
            report.append(f"- McClellan Oscillator: {latest_mc['mcclellan_osc']:.2f}")
            report.append(f"- McClellan Summation Index: {latest_mc['mcclellan_sum']:.0f}")

            mc_osc = latest_mc['mcclellan_osc']
            if mc_osc > 100:
                report.append("\n**信号**: 超买区域，市场可能过热")
            elif mc_osc > 0:
                report.append("\n**信号**: 正值区域，市场偏强")
            elif mc_osc > -100:
                report.append("\n**信号**: 负值区域，市场偏弱")
            else:
                report.append("\n**信号**: 超卖区域，可能存在反弹机会")

        # ===== 2. 资金情绪指标 =====
        report.append("\n\n---")
        report.append("\n## 2. 资金情绪指标 (Money Flow Sentiment)")

        # 2.1 主力资金流向
        report.append("\n### 2.1 主力资金流向")
        money_flow_df = self.calc_money_flow_sentiment(start_date, end_date)
        if len(money_flow_df) > 0:
            latest_mf = money_flow_df.iloc[-1]
            report.append(f"\n**最新数据 ({latest_mf['trade_date']})**:")
            report.append(f"- 主力净流入: {latest_mf['main_net_inflow']/10000:.2f}亿元")
            report.append(f"- 超大单净流入: {latest_mf['elg_net_inflow']/10000:.2f}亿元")
            report.append(f"- 大单净流入: {latest_mf['lg_net_inflow']/10000:.2f}亿元")
            report.append(f"- 中单净流入: {latest_mf['md_net_inflow']/10000:.2f}亿元")
            report.append(f"- 小单净流入: {latest_mf['sm_net_inflow']/10000:.2f}亿元")
            report.append(f"- 资金流入家数: {int(latest_mf['inflow_stocks'])}")
            report.append(f"- 资金流出家数: {int(latest_mf['outflow_stocks'])}")

            report.append(f"\n**期间统计**:")
            total_main_flow = money_flow_df['main_net_inflow'].sum()
            report.append(f"- 期间主力净流入合计: {total_main_flow/10000:.2f}亿元")
            report.append(f"- 主力净流入天数: {(money_flow_df['main_net_inflow'] > 0).sum()}")
            report.append(f"- 主力净流出天数: {(money_flow_df['main_net_inflow'] < 0).sum()}")

            # 近期趋势
            recent_5d_flow = money_flow_df.tail(5)['main_net_inflow'].sum()
            recent_20d_flow = money_flow_df.tail(20)['main_net_inflow'].sum()
            report.append(f"\n**近期趋势**:")
            report.append(f"- 近5日主力净流入: {recent_5d_flow/10000:.2f}亿元")
            report.append(f"- 近20日主力净流入: {recent_20d_flow/10000:.2f}亿元")

            # 近10日数据表
            report.append("\n**近10个交易日主力资金流向** (单位: 亿元):")
            report.append("\n| 日期 | 主力净流入 | 超大单 | 大单 | 流入家数 |")
            report.append("|------|-----------|--------|------|---------|")
            for _, row in money_flow_df.tail(10).iterrows():
                report.append(f"| {row['trade_date']} | {row['main_net_inflow']/10000:.2f} | {row['elg_net_inflow']/10000:.2f} | {row['lg_net_inflow']/10000:.2f} | {int(row['inflow_stocks'])} |")

        # 2.2 融资融券
        report.append("\n\n### 2.2 融资融券情绪")
        margin_df = self.calc_margin_sentiment(start_date, end_date)
        if len(margin_df) > 0:
            latest_margin = margin_df.iloc[-1]
            report.append(f"\n**最新数据 ({latest_margin['trade_date']})**:")
            report.append(f"- 融资余额: {latest_margin['total_rzye']/100000000:.2f}亿元")
            report.append(f"- 融券余额: {latest_margin['total_rqye']/100000000:.2f}亿元")
            report.append(f"- 融资融券余额: {latest_margin['total_rzrqye']/100000000:.2f}亿元")
            report.append(f"- 融资净买入: {latest_margin['rz_net_buy']/100000000:.2f}亿元")

            if pd.notna(latest_margin['rzye_chg']):
                report.append(f"- 融资余额日变化: {latest_margin['rzye_chg']*100:.2f}%")

            report.append(f"\n**期间统计**:")
            start_rzye = margin_df['total_rzye'].iloc[0] if len(margin_df) > 0 else 0
            end_rzye = margin_df['total_rzye'].iloc[-1] if len(margin_df) > 0 else 0
            rzye_change = (end_rzye - start_rzye) / 100000000
            report.append(f"- 期间融资余额变化: {rzye_change:.2f}亿元")
            report.append(f"- 融资余额增加天数: {(margin_df['rzye_chg'] > 0).sum()}")
            report.append(f"- 融资余额减少天数: {(margin_df['rzye_chg'] < 0).sum()}")

        # ===== 3. 波动率情绪指标 =====
        report.append("\n\n---")
        report.append("\n## 3. 波动率情绪指标 (Volatility Sentiment)")

        vol_df = self.calc_volatility_sentiment(start_date, end_date)
        if len(vol_df) > 0:
            latest_vol = vol_df.iloc[-1]
            report.append(f"\n**最新数据 ({latest_vol['trade_date']})**:")
            report.append(f"- 市场波动率(标准差): {latest_vol['volatility']:.2f}%")
            report.append(f"- 平均绝对收益: {latest_vol['avg_abs_return']:.2f}%")
            report.append(f"- 收益极差: {latest_vol['max_return']:.2f}% ~ {latest_vol['min_return']:.2f}%")
            report.append(f"- 恐慌指数代理: {latest_vol['fear_index']:.2f}")
            report.append(f"- 波动率偏度: {latest_vol['vol_skew']:.2f}")

            report.append(f"\n**期间统计**:")
            report.append(f"- 平均波动率: {vol_df['volatility'].mean():.2f}%")
            report.append(f"- 最高波动率: {vol_df['volatility'].max():.2f}% ({vol_df.loc[vol_df['volatility'].idxmax(), 'trade_date']})")
            report.append(f"- 最低波动率: {vol_df['volatility'].min():.2f}% ({vol_df.loc[vol_df['volatility'].idxmin(), 'trade_date']})")

            # 波动率分位数
            vol_percentile = vol_df['volatility'].rank(pct=True).iloc[-1] * 100
            report.append(f"\n**当前波动率分位**: {vol_percentile:.1f}% (历史分位)")

            if vol_percentile > 80:
                report.append("\n**波动率状态**: 高波动期，市场情绪不稳定")
            elif vol_percentile < 20:
                report.append("\n**波动率状态**: 低波动期，市场可能酝酿变盘")
            else:
                report.append("\n**波动率状态**: 正常波动范围")

        # 日内波动
        intraday_vol_df = self.calc_intraday_volatility(start_date, end_date)
        if len(intraday_vol_df) > 0:
            latest_intra = intraday_vol_df.iloc[-1]
            report.append(f"\n**日内波动 ({latest_intra['trade_date']})**:")
            report.append(f"- 平均日内振幅: {latest_intra['avg_intraday_range']:.2f}%")
            report.append(f"- 平均上影线: {latest_intra['avg_upper_shadow']:.2f}%")
            report.append(f"- 平均下影线: {latest_intra['avg_lower_shadow']:.2f}%")
            report.append(f"- 平均实体大小: {latest_intra['avg_body_size']:.2f}%")

        # ===== 4. 换手率情绪指标 =====
        report.append("\n\n---")
        report.append("\n## 4. 换手率情绪指标 (Turnover Sentiment)")

        turnover_df = self.calc_turnover_sentiment(start_date, end_date)
        if len(turnover_df) > 0:
            latest_to = turnover_df.iloc[-1]
            report.append(f"\n**最新数据 ({latest_to['trade_date']})**:")
            report.append(f"- 平均换手率: {latest_to['avg_turnover']:.2f}%")
            report.append(f"- 换手率中位数: {latest_to['turnover_median']:.2f}%")
            report.append(f"- 换手率标准差: {latest_to['turnover_std']:.2f}%")
            report.append(f"- 平均量比: {latest_to['avg_volume_ratio']:.2f}")
            report.append(f"- 高换手率(>10%)股票比例: {latest_to['high_turnover_pct']:.2f}%")
            report.append(f"- 低换手率(<1%)股票比例: {latest_to['low_turnover_pct']:.2f}%")

            report.append(f"\n**期间统计**:")
            report.append(f"- 平均换手率: {turnover_df['avg_turnover'].mean():.2f}%")
            report.append(f"- 最高平均换手率: {turnover_df['avg_turnover'].max():.2f}% ({turnover_df.loc[turnover_df['avg_turnover'].idxmax(), 'trade_date']})")
            report.append(f"- 最低平均换手率: {turnover_df['avg_turnover'].min():.2f}% ({turnover_df.loc[turnover_df['avg_turnover'].idxmin(), 'trade_date']})")

            # 换手率相对水平
            if pd.notna(latest_to['turnover_rel']):
                report.append(f"\n**换手率相对水平**: {latest_to['turnover_rel']:.2f}x (相对20日均值)")
                if latest_to['turnover_rel'] > 1.3:
                    report.append("**状态**: 交投活跃，市场情绪高涨")
                elif latest_to['turnover_rel'] < 0.7:
                    report.append("**状态**: 交投清淡，市场情绪低迷")
                else:
                    report.append("**状态**: 交投正常")

        # 成交量情绪
        volume_df = self.calc_volume_sentiment(start_date, end_date)
        if len(volume_df) > 0:
            latest_vol_s = volume_df.iloc[-1]
            report.append(f"\n**成交额统计 ({latest_vol_s['trade_date']})**:")
            report.append(f"- 全市场成交额: {latest_vol_s['total_amount']/10000:.2f}亿元")
            report.append(f"- 期间日均成交额: {volume_df['total_amount'].mean()/10000:.2f}亿元")
            report.append(f"- 最高成交额: {volume_df['total_amount'].max()/10000:.2f}亿元 ({volume_df.loc[volume_df['total_amount'].idxmax(), 'trade_date']})")

        # ===== 5. 综合情绪指数 =====
        report.append("\n\n---")
        report.append("\n## 5. 综合情绪指数 (Composite Sentiment Index)")

        sentiment_df = self.calc_composite_sentiment_index(start_date, end_date)
        cycle_result = self.identify_sentiment_cycles(sentiment_df)

        if len(sentiment_df) > 0:
            latest_sent = sentiment_df.iloc[-1]
            report.append(f"\n**最新综合情绪指数 ({latest_sent['trade_date']})**:")
            report.append(f"- **综合情绪指数**: {latest_sent['composite_sentiment']:.2f}")
            report.append(f"- 情绪指数MA5: {latest_sent['sentiment_ma5']:.2f}" if pd.notna(latest_sent['sentiment_ma5']) else "- 情绪指数MA5: N/A")
            report.append(f"- 情绪指数MA20: {latest_sent['sentiment_ma20']:.2f}" if pd.notna(latest_sent['sentiment_ma20']) else "- 情绪指数MA20: N/A")

            report.append(f"\n**分项指标得分**:")
            report.append(f"- 市场宽度情绪 (30%): {latest_sent['width_sentiment']:.2f}")
            report.append(f"- 资金流向情绪 (25%): {latest_sent['money_sentiment']:.2f}")
            report.append(f"- 波动率情绪 (20%): {latest_sent['vol_sentiment']:.2f}")
            report.append(f"- 换手率情绪 (15%): {latest_sent['turnover_sentiment']:.2f}")
            report.append(f"- 融资融券情绪 (10%): {latest_sent['margin_sentiment']:.2f}")

            # 情绪周期
            report.append(f"\n**当前市场情绪状态**:")
            report.append(f"- 情绪阶段: {cycle_result['current_phase']}")
            report.append(f"- 情绪趋势: {cycle_result['current_trend']}")
            report.append(f"- 极端情绪: {cycle_result['current_extreme']}")

            # 情绪分布
            report.append(f"\n**期间情绪分布**:")
            for phase, pct in cycle_result['phase_distribution'].items():
                report.append(f"- {phase}: {pct:.1f}%")

            # 情绪指数统计
            report.append(f"\n**期间情绪指数统计**:")
            report.append(f"- 平均值: {sentiment_df['composite_sentiment'].mean():.2f}")
            report.append(f"- 最高值: {sentiment_df['composite_sentiment'].max():.2f} ({sentiment_df.loc[sentiment_df['composite_sentiment'].idxmax(), 'trade_date']})")
            report.append(f"- 最低值: {sentiment_df['composite_sentiment'].min():.2f} ({sentiment_df.loc[sentiment_df['composite_sentiment'].idxmin(), 'trade_date']})")

            # 预警信号
            alerts = self.generate_sentiment_alerts(sentiment_df)
            if alerts:
                report.append(f"\n**情绪预警信号**:")
                for alert in alerts:
                    report.append(f"- [{alert['level']}] {alert['message']}")
            else:
                report.append(f"\n**情绪预警信号**: 无异常信号")

            # 近期情绪指数表
            report.append("\n**近10个交易日情绪指数**:")
            report.append("\n| 日期 | 综合情绪 | 宽度 | 资金 | 波动 | 换手 | 阶段 |")
            report.append("|------|---------|------|------|------|------|------|")
            cycle_data = cycle_result['data']
            for _, row in cycle_data.tail(10).iterrows():
                report.append(f"| {row['trade_date']} | {row['composite_sentiment']:.1f} | {row['width_sentiment']:.1f} | {row['money_sentiment']:.1f} | {row['vol_sentiment']:.1f} | {row['turnover_sentiment']:.1f} | {row['sentiment_phase']} |")

        # ===== 6. 情绪与收益关系 =====
        report.append("\n\n---")
        report.append("\n## 6. 情绪与收益关系分析")

        sr_result = self.analyze_sentiment_return_relationship(start_date, end_date)

        # 相关性分析
        report.append("\n### 6.1 情绪指数与未来收益相关性")
        if sr_result['correlations']:
            report.append("\n| 未来周期 | 相关系数 | 解读 |")
            report.append("|---------|---------|------|")
            for period, corr in sr_result['correlations'].items():
                if abs(corr) > 0.3:
                    interpret = "强相关" if corr > 0 else "强负相关"
                elif abs(corr) > 0.1:
                    interpret = "中等相关" if corr > 0 else "中等负相关"
                else:
                    interpret = "弱相关"
                report.append(f"| {period} | {corr:.4f} | {interpret} |")

        # 不同情绪水平的未来收益
        report.append("\n### 6.2 不同情绪水平的未来收益表现")
        if '5d_return' in sr_result['sentiment_level_analysis']:
            report.append("\n**未来5日平均收益 (按情绪水平分组)**:")
            report.append("\n| 情绪水平 | 平均收益(%) | 标准差(%) | 样本数 | 夏普比 |")
            report.append("|---------|------------|-----------|--------|--------|")
            for level, stats in sr_result['sentiment_level_analysis']['5d_return'].items():
                report.append(f"| {level} | {stats['mean']:.2f} | {stats['std']:.2f} | {stats['count']} | {stats['sharpe']:.3f} |")

        # 逆向策略分析
        report.append("\n### 6.3 逆向策略分析")
        report.append("\n**极端情绪组的未来收益表现**:")
        if sr_result['contrarian_strategy']:
            report.append("\n| 策略 | 平均收益(%) | 标准差(%) | 样本数 |")
            report.append("|------|------------|-----------|--------|")
            for strategy, stats in sr_result['contrarian_strategy'].items():
                parts = strategy.split('_')
                quintile = parts[0]
                period = parts[1]
                label = f"极度恐惧后{period}" if quintile == 'Q1' else f"极度贪婪后{period}"
                report.append(f"| {label} | {stats['mean']:.2f} | {stats['std']:.2f} | {stats['count']} |")

        report.append("\n\n**逆向策略结论**:")
        if sr_result['contrarian_strategy']:
            q1_5d = sr_result['contrarian_strategy'].get('Q1_5d', {})
            q5_5d = sr_result['contrarian_strategy'].get('Q5_5d', {})
            if q1_5d and q5_5d:
                if q1_5d.get('mean', 0) > q5_5d.get('mean', 0):
                    report.append("- 极度恐惧时买入的平均收益高于极度贪婪时买入，支持逆向投资策略")
                else:
                    report.append("- 极度贪婪时买入的平均收益高于极度恐惧时买入，趋势跟随策略可能更有效")

        # 情绪拐点分析
        report.append("\n### 6.4 情绪拐点识别")
        turning_df = self.identify_sentiment_turning_points(sentiment_df)
        peaks = turning_df[turning_df['turning_point'] == 'Peak'].tail(5)
        troughs = turning_df[turning_df['turning_point'] == 'Trough'].tail(5)

        if len(peaks) > 0:
            report.append("\n**近期情绪峰值 (潜在卖出信号)**:")
            for _, row in peaks.iterrows():
                report.append(f"- {row['trade_date']}: 情绪指数 {row['composite_sentiment']:.2f}")

        if len(troughs) > 0:
            report.append("\n**近期情绪谷值 (潜在买入信号)**:")
            for _, row in troughs.iterrows():
                report.append(f"- {row['trade_date']}: 情绪指数 {row['composite_sentiment']:.2f}")

        # ===== 7. 总结与建议 =====
        report.append("\n\n---")
        report.append("\n## 7. 总结与投资建议")

        # 综合判断
        if len(sentiment_df) > 0:
            latest = sentiment_df.iloc[-1]
            current_sentiment = latest['composite_sentiment']

            report.append(f"\n### 7.1 当前市场情绪总结")
            report.append(f"\n**综合情绪指数: {current_sentiment:.2f}**")

            if current_sentiment >= 70:
                report.append("\n**市场状态**: 极度乐观/贪婪")
                report.append("- 市场情绪过热，短期可能面临回调压力")
                report.append("- 建议: 谨慎追高，适当减仓，关注风险控制")
            elif current_sentiment >= 55:
                report.append("\n**市场状态**: 偏乐观")
                report.append("- 市场情绪积极，但需警惕过热风险")
                report.append("- 建议: 维持仓位，关注结构性机会")
            elif current_sentiment >= 45:
                report.append("\n**市场状态**: 中性")
                report.append("- 市场情绪平稳，多空博弈均衡")
                report.append("- 建议: 精选个股，控制仓位")
            elif current_sentiment >= 30:
                report.append("\n**市场状态**: 偏悲观")
                report.append("- 市场情绪低迷，可能存在错杀机会")
                report.append("- 建议: 逢低关注优质标的")
            else:
                report.append("\n**市场状态**: 极度悲观/恐惧")
                report.append("- 市场情绪极度低迷，可能接近底部区域")
                report.append("- 建议: 左侧布局机会，但需控制节奏")

        report.append("\n\n### 7.2 指标体系说明")
        report.append("""
| 指标类别 | 核心指标 | 说明 |
|---------|---------|------|
| 市场宽度 | 涨跌比、ADL、McClellan | 衡量市场参与广度 |
| 资金流向 | 主力净流入、融资余额 | 反映资金情绪 |
| 波动率 | 市场波动率、恐慌指数 | 衡量市场不确定性 |
| 换手率 | 平均换手率、量比 | 反映交易活跃度 |
| 综合指数 | 加权综合 | 0-100，50为中性 |
""")

        report.append("\n### 7.3 使用建议")
        report.append("""
1. **情绪指数解读**:
   - 0-25: 极度恐惧，潜在买入机会
   - 25-45: 悲观，关注抄底时机
   - 45-55: 中性，观望为主
   - 55-75: 乐观，趋势跟随
   - 75-100: 极度贪婪，注意风险

2. **信号确认**:
   - 单一指标信号需要其他指标确认
   - 关注情绪指标与价格的背离
   - 极端情绪往往是反转信号

3. **风险提示**:
   - 情绪指标是参考工具，非绝对预测
   - 需结合基本面和技术面综合判断
   - 市场可能在极端情绪下持续更长时间
""")

        report.append("\n\n---")
        report.append(f"\n*报告生成完毕 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

        return '\n'.join(report)


def main():
    """主函数"""
    print("=" * 60)
    print("A股市场情绪指标体系分析")
    print("=" * 60)

    analyzer = MarketSentimentAnalyzer(DB_PATH)

    try:
        print("\n正在生成市场情绪分析报告...")
        report = analyzer.generate_report(lookback_days=365)

        # 保存报告
        with open(REPORT_PATH, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"\n报告已保存至: {REPORT_PATH}")
        print("\n" + "=" * 60)
        print("分析完成!")
        print("=" * 60)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        analyzer.close()


if __name__ == "__main__":
    main()
