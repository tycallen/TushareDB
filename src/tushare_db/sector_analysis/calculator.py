"""
板块涨跌幅计算器
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, List, Tuple
from tqdm import tqdm
from ..reader import DataReader

logger = logging.getLogger(__name__)


class SectorCalculator:
    """板块涨跌幅计算器"""

    def __init__(self, reader: DataReader):
        """
        初始化计算器

        Args:
            reader: DataReader实例
        """
        self.reader = reader

    def calculate_returns(
        self,
        start_date: str,
        end_date: str,
        level: str = 'L1',
        period: str = 'daily',
        method: str = 'equal'
    ) -> pd.DataFrame:
        """
        计算板块涨跌幅

        Args:
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)
            level: 层级 (L1/L2/L3)
            period: 周期 (daily/weekly/monthly)
            method: 计算方法 (equal/weighted/index)

        Returns:
            DataFrame with columns: [sector_code, sector_name, trade_date, return, stock_count]
        """
        if method != 'equal':
            raise NotImplementedError(f"方法 '{method}' 暂未实现，当前仅支持 'equal'")

        logger.info(f"开始计算板块涨跌幅: {start_date} ~ {end_date}, level={level}, period={period}")

        # 1. 计算日线涨跌幅
        daily_returns = self._calculate_daily_returns(start_date, end_date, level)

        # 2. 周期转换
        if period == 'daily':
            result = daily_returns
        elif period == 'weekly':
            result = self._convert_to_weekly(daily_returns)
        elif period == 'monthly':
            result = self._convert_to_monthly(daily_returns)
        else:
            raise ValueError(f"不支持的周期: {period}")

        logger.info(f"计算完成: 共 {len(result)} 条记录")
        return result

    def _calculate_daily_returns(
        self,
        start_date: str,
        end_date: str,
        level: str
    ) -> pd.DataFrame:
        """计算日线等权平均涨跌幅"""

        # 获取所有板块列表
        sectors = self._get_sectors(level)
        logger.info(f"找到 {len(sectors)} 个{level}板块")

        # 获取交易日列表
        trade_dates = self._get_trade_dates(start_date, end_date)
        logger.info(f"交易日范围: {len(trade_dates)} 个交易日")

        # 计算每个板块每日涨跌幅
        results = []
        for sector_code, sector_name in tqdm(sectors, desc=f"计算{level}板块日线涨跌"):
            for trade_date in trade_dates:
                ret = self._calculate_equal_weighted_return(
                    sector_code, trade_date, level
                )
                if ret is not None:
                    results.append({
                        'sector_code': sector_code,
                        'sector_name': sector_name,
                        'trade_date': trade_date,
                        'return': ret['return'],
                        'stock_count': ret['stock_count'],
                        'valid_count': ret['valid_count']
                    })

        return pd.DataFrame(results)

    def _get_sectors(self, level: str) -> List[Tuple[str, str]]:
        """获取指定层级的所有板块"""
        level_code_map = {
            'L1': 'l1_code',
            'L2': 'l2_code',
            'L3': 'l3_code'
        }
        level_name_map = {
            'L1': 'l1_name',
            'L2': 'l2_name',
            'L3': 'l3_name'
        }

        code_col = level_code_map.get(level)
        name_col = level_name_map.get(level)

        if not code_col:
            raise ValueError(f"不支持的层级: {level}")

        query = f"""
            SELECT DISTINCT {code_col} as sector_code, {name_col} as sector_name
            FROM index_member_all
            WHERE is_new = 'Y'
            ORDER BY sector_code
        """

        df = self.reader.db.con.execute(query).fetchdf()
        return list(zip(df['sector_code'], df['sector_name']))

    def _get_trade_dates(self, start_date: str, end_date: str) -> List[str]:
        """获取交易日列表"""
        query = f"""
            SELECT DISTINCT trade_date
            FROM daily
            WHERE trade_date >= '{start_date}'
              AND trade_date <= '{end_date}'
            ORDER BY trade_date
        """
        df = self.reader.db.con.execute(query).fetchdf()
        return df['trade_date'].tolist()

    def _calculate_equal_weighted_return(
        self,
        sector_code: str,
        trade_date: str,
        level: str
    ) -> Optional[dict]:
        """计算单个板块在指定日期的等权平均涨跌幅"""

        level_code_map = {
            'L1': 'l1_code',
            'L2': 'l2_code',
            'L3': 'l3_code'
        }
        code_col = level_code_map.get(level)

        # 1. 获取成分股（确保无未来函数）
        query = f"""
            SELECT ts_code
            FROM index_member_all
            WHERE {code_col} = '{sector_code}'
              AND in_date <= '{trade_date}'
              AND (out_date IS NULL OR out_date > '{trade_date}')
        """
        members_df = self.reader.db.con.execute(query).fetchdf()

        if len(members_df) == 0:
            return None

        stock_codes = members_df['ts_code'].tolist()

        # 2. 获取成分股涨跌幅
        placeholders = ','.join(['?' for _ in stock_codes])
        query = f"""
            SELECT pct_chg
            FROM daily
            WHERE ts_code IN ({placeholders})
              AND trade_date = ?
        """
        returns_df = self.reader.db.con.execute(
            query, stock_codes + [trade_date]
        ).fetchdf()

        # 3. 计算等权平均（剔除NaN）
        valid_returns = returns_df['pct_chg'].dropna()

        if len(valid_returns) == 0:
            return None

        sector_return = valid_returns.mean()

        return {
            'return': sector_return,
            'stock_count': len(stock_codes),
            'valid_count': len(valid_returns)
        }

    def _convert_to_weekly(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        """日线转周线"""
        df = daily_df.copy()
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
        df = df.set_index('trade_date')

        results = []
        for sector_code in df['sector_code'].unique():
            sector_data = df[df['sector_code'] == sector_code].copy()
            sector_name = sector_data['sector_name'].iloc[0]

            # 按周分组，计算累计收益率
            weekly = sector_data.resample('W-SUN').apply({
                'return': lambda x: ((1 + x / 100).prod() - 1) * 100,
                'stock_count': 'last',
                'valid_count': 'mean'
            })

            weekly['sector_code'] = sector_code
            weekly['sector_name'] = sector_name
            weekly = weekly.reset_index()
            weekly['trade_date'] = weekly['trade_date'].dt.strftime('%Y%m%d')

            results.append(weekly)

        return pd.concat(results, ignore_index=True)

    def _convert_to_monthly(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        """日线转月线"""
        df = daily_df.copy()
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
        df = df.set_index('trade_date')

        results = []
        for sector_code in df['sector_code'].unique():
            sector_data = df[df['sector_code'] == sector_code].copy()
            sector_name = sector_data['sector_name'].iloc[0]

            # 按月分组，计算累计收益率
            monthly = sector_data.resample('ME').apply({
                'return': lambda x: ((1 + x / 100).prod() - 1) * 100,
                'stock_count': 'last',
                'valid_count': 'mean'
            })

            monthly['sector_code'] = sector_code
            monthly['sector_name'] = sector_name
            monthly = monthly.reset_index()
            monthly['trade_date'] = monthly['trade_date'].dt.strftime('%Y%m%d')

            results.append(monthly)

        return pd.concat(results, ignore_index=True)
