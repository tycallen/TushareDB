#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
量价和资金流特征工程模块

本模块实现基于 DuckDB 数据库的量价特征和资金流特征计算。

数据源:
- daily: 日线行情 (open, high, low, close, vol, amount)
- daily_basic: 每日指标 (turnover_rate, volume_ratio, pe, pb, ps, etc.)
- moneyflow: 资金流向 (buy_sm_vol, sell_sm_vol, buy_md_vol, sell_md_vol, buy_lg_vol, sell_lg_vol, buy_elg_vol, sell_elg_vol)
- moneyflow_dc: 东财资金流向
- stk_factor_pro: 技术因子
- index_member_all: 行业分类

特征分类:
1. 成交量特征: 量比、换手率变化、量突变、量价背离、连续放量/缩量
2. 价格特征: 振幅、连涨连跌、动量、相对强度
3. 估值特征: PE/PB/PS分位数、行业相对估值、市值排名
4. 主力资金特征: 主力净流入、流入率、连续流入天数、累计流入
5. 大中小单分布: 各类型单占比、净流入比例、变化趋势
6. 资金流动量: 资金流强度、动量、转折点

作者: Claude
日期: 2025-01-31
"""

import duckdb
import pandas as pd
import numpy as np
from typing import Optional, Tuple, List, Dict, Union
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


class VolumeMoneyflowFeatures:
    """量价和资金流特征计算类"""

    def __init__(self, db_path: str):
        """
        初始化特征计算器

        Args:
            db_path: DuckDB数据库路径
        """
        self.db_path = db_path
        self.conn = None

    def connect(self):
        """连接数据库"""
        if self.conn is None:
            self.conn = duckdb.connect(self.db_path, read_only=True)
        return self.conn

    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()
            self.conn = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # ==================== 数据加载 ====================

    def load_daily_data(self, ts_code: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        加载日线行情数据

        Args:
            ts_code: 股票代码
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)

        Returns:
            包含日线数据的DataFrame
        """
        conn = self.connect()
        query = f"""
        SELECT d.ts_code, d.trade_date, d.open, d.high, d.low, d.close,
               d.pre_close, d.change, d.pct_chg, d.vol, d.amount
        FROM daily d
        WHERE d.ts_code = '{ts_code}'
        """
        if start_date:
            query += f" AND d.trade_date >= '{start_date}'"
        if end_date:
            query += f" AND d.trade_date <= '{end_date}'"
        query += " ORDER BY d.trade_date"

        df = conn.execute(query).fetchdf()
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        return df

    def load_daily_basic(self, ts_code: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """加载每日指标数据"""
        conn = self.connect()
        query = f"""
        SELECT ts_code, trade_date, close, turnover_rate, turnover_rate_f,
               volume_ratio, pe, pe_ttm, pb, ps, ps_ttm,
               dv_ratio, dv_ttm, total_share, float_share, free_share,
               total_mv, circ_mv
        FROM daily_basic
        WHERE ts_code = '{ts_code}'
        """
        if start_date:
            query += f" AND trade_date >= '{start_date}'"
        if end_date:
            query += f" AND trade_date <= '{end_date}'"
        query += " ORDER BY trade_date"

        df = conn.execute(query).fetchdf()
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        return df

    def load_moneyflow(self, ts_code: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """加载资金流向数据"""
        conn = self.connect()
        query = f"""
        SELECT ts_code, trade_date,
               buy_sm_vol, buy_sm_amount, sell_sm_vol, sell_sm_amount,
               buy_md_vol, buy_md_amount, sell_md_vol, sell_md_amount,
               buy_lg_vol, buy_lg_amount, sell_lg_vol, sell_lg_amount,
               buy_elg_vol, buy_elg_amount, sell_elg_vol, sell_elg_amount,
               net_mf_vol, net_mf_amount
        FROM moneyflow
        WHERE ts_code = '{ts_code}'
        """
        if start_date:
            query += f" AND trade_date >= '{start_date}'"
        if end_date:
            query += f" AND trade_date <= '{end_date}'"
        query += " ORDER BY trade_date"

        df = conn.execute(query).fetchdf()
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        return df

    def load_merged_data(self, ts_code: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        加载合并后的完整数据

        Args:
            ts_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            合并后的DataFrame
        """
        daily = self.load_daily_data(ts_code, start_date, end_date)
        basic = self.load_daily_basic(ts_code, start_date, end_date)
        mf = self.load_moneyflow(ts_code, start_date, end_date)

        # 合并数据
        df = daily.merge(basic[['ts_code', 'trade_date', 'turnover_rate', 'turnover_rate_f',
                                'volume_ratio', 'pe', 'pe_ttm', 'pb', 'ps', 'ps_ttm',
                                'total_mv', 'circ_mv']],
                        on=['ts_code', 'trade_date'], how='left')
        df = df.merge(mf, on=['ts_code', 'trade_date'], how='left')

        return df.sort_values('trade_date').reset_index(drop=True)

    # ==================== 成交量特征 ====================

    def calc_volume_ratio(self, vol: pd.Series, n: int = 5) -> pd.Series:
        """
        计算量比

        量比 = 当日成交量 / N日平均成交量

        Args:
            vol: 成交量序列
            n: 平均天数，默认5日

        Returns:
            量比序列
        """
        ma_vol = vol.rolling(n).mean().shift(1)  # 使用前N日均量
        return vol / ma_vol

    def calc_turnover_change(self, turnover_rate: pd.Series, n: int = 5) -> pd.Series:
        """
        计算换手率变化

        换手率变化 = 当日换手率 / N日平均换手率

        Args:
            turnover_rate: 换手率序列
            n: 平均天数

        Returns:
            换手率变化序列
        """
        ma_turnover = turnover_rate.rolling(n).mean().shift(1)
        return turnover_rate / ma_turnover

    def detect_volume_spike(self, vol: pd.Series, n: int = 20, threshold: float = 2.0) -> pd.Series:
        """
        检测成交量突变

        当成交量超过N日均量的threshold倍时，认为是放量

        Args:
            vol: 成交量序列
            n: 基准周期
            threshold: 突变阈值

        Returns:
            布尔序列，True表示放量
        """
        ma_vol = vol.rolling(n).mean().shift(1)
        std_vol = vol.rolling(n).std().shift(1)

        # 方法1: 简单阈值
        spike_simple = vol > (ma_vol * threshold)

        # 方法2: 基于标准差 (超过均值+2倍标准差)
        spike_std = vol > (ma_vol + 2 * std_vol)

        return spike_simple

    def calc_volume_price_divergence(self, close: pd.Series, vol: pd.Series, n: int = 20) -> pd.Series:
        """
        计算量价背离

        量价背离 = 价格创N日新高但成交量萎缩 (或价格新低但放量)

        返回值:
        - 1: 顶背离 (价格新高但量萎缩)
        - -1: 底背离 (价格新低但量放大)
        - 0: 无背离

        Args:
            close: 收盘价序列
            vol: 成交量序列
            n: 观察周期

        Returns:
            背离信号序列
        """
        # 价格创N日新高
        price_high = close == close.rolling(n).max()
        # 价格创N日新低
        price_low = close == close.rolling(n).min()

        # 成交量低于N日均量
        vol_ma = vol.rolling(n).mean()
        vol_shrink = vol < vol_ma * 0.7
        vol_expand = vol > vol_ma * 1.3

        divergence = pd.Series(0, index=close.index)
        divergence[price_high & vol_shrink] = 1    # 顶背离
        divergence[price_low & vol_expand] = -1    # 底背离

        return divergence

    def calc_consecutive_volume_days(self, vol: pd.Series, n: int = 5) -> Tuple[pd.Series, pd.Series]:
        """
        计算连续放量/缩量天数

        放量定义: 当日成交量 > 前一日成交量
        缩量定义: 当日成交量 < 前一日成交量

        Args:
            vol: 成交量序列
            n: 均线周期 (用于比较)

        Returns:
            (连续放量天数, 连续缩量天数)
        """
        vol_ma = vol.rolling(n).mean()
        vol_change = vol.diff()

        # 连续放量
        increase = (vol_change > 0).astype(int)
        # 连续缩量
        decrease = (vol_change < 0).astype(int)

        # 计算连续天数
        def count_consecutive(series):
            result = []
            count = 0
            for val in series:
                if val == 1:
                    count += 1
                else:
                    count = 0
                result.append(count)
            return pd.Series(result, index=series.index)

        return count_consecutive(increase), count_consecutive(decrease)

    # ==================== 价格特征 ====================

    def calc_amplitude(self, high: pd.Series, low: pd.Series, pre_close: pd.Series) -> pd.Series:
        """
        计算振幅

        振幅 = (最高价 - 最低价) / 前收盘价 * 100

        Args:
            high: 最高价序列
            low: 最低价序列
            pre_close: 前收盘价序列

        Returns:
            振幅序列 (百分比)
        """
        return (high - low) / pre_close * 100

    def calc_consecutive_days(self, pct_chg: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        计算连涨/连跌天数

        Args:
            pct_chg: 涨跌幅序列

        Returns:
            (连涨天数, 连跌天数)
        """
        up = (pct_chg > 0).astype(int)
        down = (pct_chg < 0).astype(int)

        def count_consecutive(series):
            result = []
            count = 0
            for val in series:
                if val == 1:
                    count += 1
                else:
                    count = 0
                result.append(count)
            return pd.Series(result, index=series.index)

        return count_consecutive(up), count_consecutive(down)

    def calc_momentum(self, close: pd.Series, periods: List[int] = [5, 10, 20, 60]) -> pd.DataFrame:
        """
        计算价格动量 (不同周期的收益率)

        动量 = (当前价格 / N日前价格 - 1) * 100

        Args:
            close: 收盘价序列
            periods: 动量周期列表

        Returns:
            包含各周期动量的DataFrame
        """
        result = pd.DataFrame(index=close.index)
        for n in periods:
            result[f'momentum_{n}d'] = (close / close.shift(n) - 1) * 100
        return result

    def calc_relative_strength(self, stock_close: pd.Series, index_close: pd.Series,
                                n: int = 20) -> pd.Series:
        """
        计算相对强度 (相对指数)

        相对强度 = 个股N日涨幅 - 指数N日涨幅

        Args:
            stock_close: 个股收盘价序列
            index_close: 指数收盘价序列
            n: 计算周期

        Returns:
            相对强度序列
        """
        stock_ret = (stock_close / stock_close.shift(n) - 1) * 100
        index_ret = (index_close / index_close.shift(n) - 1) * 100
        return stock_ret - index_ret

    # ==================== 估值特征 ====================

    def calc_valuation_percentile(self, series: pd.Series, window: int = 250) -> pd.Series:
        """
        计算估值指标的历史分位数

        分位数 = 当前值在过去N日内的排名 / N

        Args:
            series: 估值指标序列 (PE, PB, PS等)
            window: 历史窗口

        Returns:
            分位数序列 (0-1)
        """
        def rolling_percentile(x):
            if len(x) < window:
                return np.nan
            return (x.iloc[-1] > x.iloc[:-1]).sum() / (len(x) - 1)

        return series.rolling(window, min_periods=window).apply(rolling_percentile, raw=False)

    def calc_market_cap_rank(self, total_mv: pd.Series, market_mv: pd.DataFrame) -> pd.Series:
        """
        计算市值排名 (需要全市场数据)

        Args:
            total_mv: 个股总市值序列
            market_mv: 全市场市值DataFrame

        Returns:
            市值排名分位数 (0-1, 1表示最大)
        """
        # 这个需要全市场数据，这里返回一个占位实现
        pass

    # ==================== 资金流特征 ====================

    def calc_main_net_inflow(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算主力资金净流入

        主力资金 = 大单 + 特大单
        主力净流入 = (大单买入 + 特大单买入) - (大单卖出 + 特大单卖出)

        Args:
            df: 包含资金流数据的DataFrame

        Returns:
            添加主力资金特征后的DataFrame
        """
        result = df.copy()

        # 主力净流入量
        result['main_buy_vol'] = df['buy_lg_vol'].fillna(0) + df['buy_elg_vol'].fillna(0)
        result['main_sell_vol'] = df['sell_lg_vol'].fillna(0) + df['sell_elg_vol'].fillna(0)
        result['main_net_vol'] = result['main_buy_vol'] - result['main_sell_vol']

        # 主力净流入金额
        result['main_buy_amount'] = df['buy_lg_amount'].fillna(0) + df['buy_elg_amount'].fillna(0)
        result['main_sell_amount'] = df['sell_lg_amount'].fillna(0) + df['sell_elg_amount'].fillna(0)
        result['main_net_amount'] = result['main_buy_amount'] - result['main_sell_amount']

        return result

    def calc_main_net_inflow_rate(self, df: pd.DataFrame) -> pd.Series:
        """
        计算主力净流入率

        主力净流入率 = 主力净流入金额 / 总成交额 * 100

        Args:
            df: 包含资金流数据的DataFrame

        Returns:
            主力净流入率序列
        """
        main_net = (df['buy_lg_amount'].fillna(0) + df['buy_elg_amount'].fillna(0) -
                   df['sell_lg_amount'].fillna(0) - df['sell_elg_amount'].fillna(0))

        # 总成交额 (万元)
        total_amount = df['amount'].fillna(0) / 10  # 假设daily的amount单位是千元

        return main_net / total_amount * 100

    def calc_consecutive_inflow_days(self, net_inflow: pd.Series) -> pd.Series:
        """
        计算连续流入天数

        Args:
            net_inflow: 净流入序列

        Returns:
            连续流入天数 (负数表示连续流出天数)
        """
        result = []
        count = 0
        prev_sign = 0

        for val in net_inflow:
            if pd.isna(val):
                result.append(0)
                count = 0
                prev_sign = 0
                continue

            current_sign = 1 if val > 0 else (-1 if val < 0 else 0)

            if current_sign == prev_sign and current_sign != 0:
                count = count + current_sign
            elif current_sign != 0:
                count = current_sign
            else:
                count = 0

            result.append(count)
            prev_sign = current_sign

        return pd.Series(result, index=net_inflow.index)

    def calc_cumulative_inflow(self, net_inflow: pd.Series, periods: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """
        计算累计净流入

        Args:
            net_inflow: 净流入序列
            periods: 累计周期列表

        Returns:
            包含各周期累计净流入的DataFrame
        """
        result = pd.DataFrame(index=net_inflow.index)
        for n in periods:
            result[f'cum_inflow_{n}d'] = net_inflow.rolling(n).sum()
        return result

    # ==================== 大中小单分布特征 ====================

    def calc_order_type_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算各类型单占比

        Args:
            df: 包含资金流数据的DataFrame

        Returns:
            添加各类型单占比的DataFrame
        """
        result = df.copy()

        # 总成交量
        total_vol = (df['buy_sm_vol'].fillna(0) + df['sell_sm_vol'].fillna(0) +
                    df['buy_md_vol'].fillna(0) + df['sell_md_vol'].fillna(0) +
                    df['buy_lg_vol'].fillna(0) + df['sell_lg_vol'].fillna(0) +
                    df['buy_elg_vol'].fillna(0) + df['sell_elg_vol'].fillna(0))

        # 小单占比
        result['sm_vol_ratio'] = (df['buy_sm_vol'].fillna(0) + df['sell_sm_vol'].fillna(0)) / total_vol * 100
        # 中单占比
        result['md_vol_ratio'] = (df['buy_md_vol'].fillna(0) + df['sell_md_vol'].fillna(0)) / total_vol * 100
        # 大单占比
        result['lg_vol_ratio'] = (df['buy_lg_vol'].fillna(0) + df['sell_lg_vol'].fillna(0)) / total_vol * 100
        # 特大单占比
        result['elg_vol_ratio'] = (df['buy_elg_vol'].fillna(0) + df['sell_elg_vol'].fillna(0)) / total_vol * 100
        # 主力单占比 (大单+特大单)
        result['main_vol_ratio'] = result['lg_vol_ratio'] + result['elg_vol_ratio']

        return result

    def calc_net_inflow_by_type(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算各类型单净流入

        Args:
            df: 包含资金流数据的DataFrame

        Returns:
            添加各类型净流入的DataFrame
        """
        result = df.copy()

        # 各类型净流入量
        result['sm_net_vol'] = df['buy_sm_vol'].fillna(0) - df['sell_sm_vol'].fillna(0)
        result['md_net_vol'] = df['buy_md_vol'].fillna(0) - df['sell_md_vol'].fillna(0)
        result['lg_net_vol'] = df['buy_lg_vol'].fillna(0) - df['sell_lg_vol'].fillna(0)
        result['elg_net_vol'] = df['buy_elg_vol'].fillna(0) - df['sell_elg_vol'].fillna(0)

        # 各类型净流入金额
        result['sm_net_amount'] = df['buy_sm_amount'].fillna(0) - df['sell_sm_amount'].fillna(0)
        result['md_net_amount'] = df['buy_md_amount'].fillna(0) - df['sell_md_amount'].fillna(0)
        result['lg_net_amount'] = df['buy_lg_amount'].fillna(0) - df['sell_lg_amount'].fillna(0)
        result['elg_net_amount'] = df['buy_elg_amount'].fillna(0) - df['sell_elg_amount'].fillna(0)

        return result

    # ==================== 资金流动量特征 ====================

    def calc_moneyflow_intensity(self, net_amount: pd.Series, total_amount: pd.Series) -> pd.Series:
        """
        计算资金流强度

        资金流强度 = 净流入金额 / 总成交额 * 100

        Args:
            net_amount: 净流入金额序列
            total_amount: 总成交额序列

        Returns:
            资金流强度序列
        """
        return net_amount / total_amount * 100

    def calc_moneyflow_momentum(self, net_inflow: pd.Series, n: int = 5) -> pd.Series:
        """
        计算资金流动量

        资金流动量 = N日净流入之和

        Args:
            net_inflow: 净流入序列
            n: 动量周期

        Returns:
            资金流动量序列
        """
        return net_inflow.rolling(n).sum()

    def detect_moneyflow_turning_point(self, net_inflow: pd.Series, n: int = 5) -> pd.Series:
        """
        检测资金流转折点

        转折点定义:
        - 1: 从净流出转为净流入 (N日累计从负转正)
        - -1: 从净流入转为净流出 (N日累计从正转负)
        - 0: 无转折

        Args:
            net_inflow: 净流入序列
            n: 累计周期

        Returns:
            转折点信号序列
        """
        cum_flow = net_inflow.rolling(n).sum()
        cum_flow_prev = cum_flow.shift(1)

        turning = pd.Series(0, index=net_inflow.index)
        # 从负转正
        turning[(cum_flow_prev < 0) & (cum_flow > 0)] = 1
        # 从正转负
        turning[(cum_flow_prev > 0) & (cum_flow < 0)] = -1

        return turning

    # ==================== 综合特征计算 ====================

    def calculate_all_features(self, ts_code: str, start_date: str = None,
                               end_date: str = None) -> pd.DataFrame:
        """
        计算所有特征

        Args:
            ts_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            包含所有特征的DataFrame
        """
        # 加载数据
        df = self.load_merged_data(ts_code, start_date, end_date)

        if df.empty:
            return pd.DataFrame()

        # ===== 成交量特征 =====
        # 量比
        df['vol_ratio_5d'] = self.calc_volume_ratio(df['vol'], 5)
        df['vol_ratio_10d'] = self.calc_volume_ratio(df['vol'], 10)
        df['vol_ratio_20d'] = self.calc_volume_ratio(df['vol'], 20)

        # 换手率变化
        if 'turnover_rate' in df.columns:
            df['turnover_change_5d'] = self.calc_turnover_change(df['turnover_rate'], 5)

        # 量突变检测
        df['vol_spike'] = self.detect_volume_spike(df['vol'], 20, 2.0).astype(int)

        # 量价背离
        df['vol_price_divergence'] = self.calc_volume_price_divergence(df['close'], df['vol'], 20)

        # 连续放量/缩量天数
        vol_up, vol_down = self.calc_consecutive_volume_days(df['vol'], 5)
        df['consecutive_vol_up'] = vol_up
        df['consecutive_vol_down'] = vol_down

        # ===== 价格特征 =====
        # 振幅
        df['amplitude'] = self.calc_amplitude(df['high'], df['low'], df['pre_close'])

        # 连涨/连跌天数
        up_days, down_days = self.calc_consecutive_days(df['pct_chg'])
        df['consecutive_up_days'] = up_days
        df['consecutive_down_days'] = down_days

        # 动量
        momentum_df = self.calc_momentum(df['close'], [5, 10, 20, 60])
        for col in momentum_df.columns:
            df[col] = momentum_df[col]

        # ===== 估值特征 =====
        if 'pe_ttm' in df.columns:
            df['pe_percentile_250d'] = self.calc_valuation_percentile(df['pe_ttm'], 250)
        if 'pb' in df.columns:
            df['pb_percentile_250d'] = self.calc_valuation_percentile(df['pb'], 250)
        if 'ps_ttm' in df.columns:
            df['ps_percentile_250d'] = self.calc_valuation_percentile(df['ps_ttm'], 250)

        # ===== 资金流特征 =====
        if 'buy_lg_vol' in df.columns and df['buy_lg_vol'].notna().any():
            # 主力资金
            df = self.calc_main_net_inflow(df)
            df['main_net_inflow_rate'] = self.calc_main_net_inflow_rate(df)

            # 连续流入天数
            df['consecutive_inflow_days'] = self.calc_consecutive_inflow_days(df['main_net_amount'])

            # 累计流入
            cum_inflow = self.calc_cumulative_inflow(df['main_net_amount'], [5, 10, 20])
            for col in cum_inflow.columns:
                df[col] = cum_inflow[col]

            # 大中小单分布
            df = self.calc_order_type_ratio(df)
            df = self.calc_net_inflow_by_type(df)

            # 资金流动量
            df['mf_intensity'] = self.calc_moneyflow_intensity(df['net_mf_amount'], df['amount'] / 10)
            df['mf_momentum_5d'] = self.calc_moneyflow_momentum(df['main_net_amount'], 5)
            df['mf_momentum_10d'] = self.calc_moneyflow_momentum(df['main_net_amount'], 10)

            # 资金流转折点
            df['mf_turning_point'] = self.detect_moneyflow_turning_point(df['main_net_amount'], 5)

        return df

    def get_feature_summary(self, df: pd.DataFrame) -> Dict:
        """
        获取特征统计摘要

        Args:
            df: 特征DataFrame

        Returns:
            特征统计字典
        """
        feature_cols = [col for col in df.columns if col not in
                       ['ts_code', 'trade_date', 'open', 'high', 'low', 'close',
                        'pre_close', 'change', 'vol', 'amount']]

        summary = {}
        for col in feature_cols:
            if col in df.columns and df[col].notna().any():
                summary[col] = {
                    'count': int(df[col].notna().sum()),
                    'mean': float(df[col].mean()) if df[col].notna().any() else None,
                    'std': float(df[col].std()) if df[col].notna().any() else None,
                    'min': float(df[col].min()) if df[col].notna().any() else None,
                    'max': float(df[col].max()) if df[col].notna().any() else None,
                    'median': float(df[col].median()) if df[col].notna().any() else None,
                }

        return summary


def analyze_feature_distribution(db_path: str, sample_stocks: List[str] = None,
                                  start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    分析特征分布

    Args:
        db_path: 数据库路径
        sample_stocks: 样本股票列表
        start_date: 开始日期
        end_date: 结束日期

    Returns:
        特征分布统计DataFrame
    """
    with VolumeMoneyflowFeatures(db_path) as calc:
        conn = calc.connect()

        # 如果没有指定股票，选取有完整资金流数据的股票
        if sample_stocks is None:
            query = """
            SELECT ts_code, COUNT(*) as cnt
            FROM moneyflow
            WHERE buy_lg_vol IS NOT NULL AND buy_lg_vol > 0
            GROUP BY ts_code
            HAVING cnt > 200
            ORDER BY cnt DESC
            LIMIT 100
            """
            stocks_df = conn.execute(query).fetchdf()
            sample_stocks = stocks_df['ts_code'].tolist()

        all_summaries = []
        for ts_code in sample_stocks[:20]:  # 只处理前20只股票
            try:
                df = calc.calculate_all_features(ts_code, start_date, end_date)
                if not df.empty:
                    summary = calc.get_feature_summary(df)
                    for feat, stats in summary.items():
                        stats['ts_code'] = ts_code
                        stats['feature'] = feat
                        all_summaries.append(stats)
            except Exception as e:
                print(f"Error processing {ts_code}: {e}")
                continue

        if all_summaries:
            return pd.DataFrame(all_summaries)
        return pd.DataFrame()


def main():
    """主函数 - 演示特征计算"""
    db_path = "/Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db"
    output_dir = Path("/Users/allen/workspace/python/stock/Tushare-DuckDB/reports/quality_check")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("量价和资金流特征工程")
    print("=" * 60)

    # 1. 计算单只股票特征
    print("\n1. 计算单只股票特征示例 (000001.SZ - 平安银行)")
    with VolumeMoneyflowFeatures(db_path) as calc:
        df = calc.calculate_all_features('000001.SZ', '20240101', '20260130')

        print(f"\n数据形状: {df.shape}")
        print(f"日期范围: {df['trade_date'].min()} ~ {df['trade_date'].max()}")

        # 显示最新的特征值
        print("\n最新5日特征值:")
        feature_cols = ['vol_ratio_5d', 'amplitude', 'consecutive_up_days',
                       'momentum_5d', 'main_net_amount', 'main_net_inflow_rate',
                       'consecutive_inflow_days', 'mf_turning_point']
        print(df[['trade_date'] + [c for c in feature_cols if c in df.columns]].tail())

        # 保存完整特征数据
        output_file = output_dir / "features_000001_SZ.csv"
        df.to_csv(output_file, index=False)
        print(f"\n特征数据已保存到: {output_file}")

    # 2. 分析特征分布
    print("\n2. 分析特征分布")
    dist_df = analyze_feature_distribution(db_path, start_date='20240101', end_date='20260130')

    if not dist_df.empty:
        # 按特征聚合统计
        agg_stats = dist_df.groupby('feature').agg({
            'mean': ['mean', 'std'],
            'min': 'min',
            'max': 'max',
            'count': 'mean'
        }).round(4)

        print("\n特征分布统计:")
        print(agg_stats.head(30))

        # 保存分布统计
        output_file = output_dir / "feature_distribution_stats.csv"
        dist_df.to_csv(output_file, index=False)
        print(f"\n分布统计已保存到: {output_file}")

    print("\n" + "=" * 60)
    print("特征计算完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
