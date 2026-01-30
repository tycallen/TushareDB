# -*- coding: utf-8 -*-
"""
数据查询模块

职责：只从 DuckDB 读取数据，不触发任何网络请求
设计理念：高性能、零开销、假设数据已经存在
"""
import pandas as pd
from typing import Optional, List, Union
from datetime import datetime

from .logger import get_logger
from .duckdb_manager import DuckDBManager

logger = get_logger(__name__)


class DataReaderError(Exception):
    """数据查询器异常"""
    pass


class DataReader:
    """
    数据查询器：专门负责从 DuckDB 读取数据

    特点：
    - 只读模式：不会触发任何网络请求或数据写入
    - 高性能：直接 SQL 查询，毫秒级响应
    - 简单直接：如果数据不存在，返回空 DataFrame 或抛出异常

    使用场景：
    - 回测系统（高频读取）
    - Web API 服务（实时查询）
    - 数据分析（探索性分析）
    """

    def __init__(self, db_path: str = "tushare.db", strict_mode: bool = False):
        """
        初始化查询器

        Args:
            db_path: DuckDB 数据库文件路径
            strict_mode: 严格模式，数据不存在时抛出异常而不是返回空 DataFrame
        """
        self.db = DuckDBManager(db_path, read_only=True)
        self.strict_mode = strict_mode
        logger.info(f"DataReader 初始化完成: db={db_path}, strict_mode={strict_mode}")

    # ==================== 基础信息查询 ====================

    def get_stock_basic(
        self,
        ts_code: Optional[str] = None,
        list_status: Optional[str] = 'L'
    ) -> pd.DataFrame:
        """
        查询股票基础信息

        Args:
            ts_code: 股票代码（可选），不指定则返回所有
            list_status: 上市状态 ('L'=上市, 'D'=退市, 'P'=暂停)，如果表中没有此字段则忽略

        Returns:
            股票基础信息 DataFrame
        """
        # 检查表中是否有 list_status 字段（兼容旧数据）
        has_list_status = 'list_status' in self.db.get_table_columns('stock_basic')

        if ts_code:
            if has_list_status and list_status:
                query = "SELECT * FROM stock_basic WHERE ts_code = ? AND list_status = ?"
                params = [ts_code, list_status]
            else:
                query = "SELECT * FROM stock_basic WHERE ts_code = ?"
                params = [ts_code]
        else:
            if has_list_status and list_status:
                query = "SELECT * FROM stock_basic WHERE list_status = ?"
                params = [list_status]
            else:
                query = "SELECT * FROM stock_basic"
                params = []

        df = self.db.execute_query(query, params if params else None)
        self._check_empty(df, f"stock_basic(ts_code={ts_code}, list_status={list_status})")
        return df

    def get_trade_calendar(
        self,
        start_date: str,
        end_date: Optional[str] = None,
        is_open: Optional[str] = None
    ) -> pd.DataFrame:
        """
        查询交易日历

        Args:
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)，默认为今天
            is_open: 是否交易日 ('1'=是, '0'=否)，不指定则返回所有

        Returns:
            交易日历 DataFrame
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')

        if is_open:
            query = """
                SELECT * FROM trade_cal
                WHERE cal_date BETWEEN ? AND ?
                AND (is_open = ? OR CAST(is_open AS VARCHAR) = ?)
                ORDER BY cal_date
            """
            params = [start_date, end_date, is_open, is_open]
        else:
            query = """
                SELECT * FROM trade_cal
                WHERE cal_date BETWEEN ? AND ?
                ORDER BY cal_date
            """
            params = [start_date, end_date]

        return self.db.execute_query(query, params)

    # ==================== 行情数据查询 ====================

    def get_stock_daily(
        self,
        ts_code: str,
        start_date: str,
        end_date: Optional[str] = None,
        adj: Optional[str] = None
    ) -> pd.DataFrame:
        """
        查询股票日线数据

        Args:
            ts_code: 股票代码 (如 '000001.SZ')
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)，默认为今天
            adj: 复权类型 (None=不复权, 'qfq'=前复权, 'hfq'=后复权)

        Returns:
            日线数据 DataFrame，包含 open/high/low/close/vol 等字段
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')

        # 1. 查询原始日线数据
        query = """
            SELECT * FROM daily
            WHERE ts_code = ?
            AND trade_date BETWEEN ? AND ?
            ORDER BY trade_date
        """
        df = self.db.execute_query(query, [ts_code, start_date, end_date])

        self._check_empty(df, f"daily(ts_code={ts_code}, {start_date}-{end_date})")

        # 2. 如果需要复权，动态计算
        if adj in ['qfq', 'hfq'] and not df.empty:
            df = self._apply_adjustment(df, ts_code, start_date, end_date, adj)

        return df

    def get_multiple_stocks_daily(
        self,
        ts_codes: List[str],
        start_date: str,
        end_date: Optional[str] = None,
        adj: Optional[str] = None
    ) -> pd.DataFrame:
        """
        批量查询多只股票的日线数据

        Args:
            ts_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            adj: 复权类型

        Returns:
            合并后的日线数据 DataFrame
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')

        if not ts_codes:
            return pd.DataFrame()

        # 使用 IN 查询批量获取
        placeholders = ','.join(['?'] * len(ts_codes))
        query = f"""
            SELECT * FROM daily
            WHERE ts_code IN ({placeholders})
            AND trade_date BETWEEN ? AND ?
            ORDER BY ts_code, trade_date
        """
        params = ts_codes + [start_date, end_date]
        df = self.db.execute_query(query, params)

        # 如果需要复权
        if adj in ['qfq', 'hfq'] and not df.empty:
            # 批量获取复权因子
            adj_query = f"""
                SELECT ts_code, trade_date, adj_factor FROM adj_factor
                WHERE ts_code IN ({placeholders})
                AND trade_date BETWEEN ? AND ?
            """
            adj_params = ts_codes + [start_date, end_date]
            adj_df = self.db.execute_query(adj_query, adj_params)

            if not adj_df.empty:
                df = df.merge(adj_df, on=['ts_code', 'trade_date'], how='left')
                # 传入 ts_codes 以正确计算前复权
                df = self._adjust_prices(df, adj, ts_codes)

        return df

    def get_daily_basic(
        self,
        ts_code: str,
        start_date: str,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        查询每日基本面指标

        Args:
            ts_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            每日基本面数据 DataFrame (包含 PE、PB、市值等)
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')

        query = """
            SELECT * FROM daily_basic
            WHERE ts_code = ?
            AND trade_date BETWEEN ? AND ?
            ORDER BY trade_date
        """
        return self.db.execute_query(query, [ts_code, start_date, end_date])

    # ==================== 复权因子查询 ====================

    def get_adj_factor(
        self,
        ts_code: str,
        start_date: str,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        查询复权因子

        Args:
            ts_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            复权因子 DataFrame
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')

        query = """
            SELECT * FROM adj_factor
            WHERE ts_code = ?
            AND trade_date BETWEEN ? AND ?
            ORDER BY trade_date
        """
        return self.db.execute_query(query, [ts_code, start_date, end_date])

    # ==================== 上市首日信息查询 ====================

    def get_all_listing_first_day_info(
        self,
        list_status: Optional[str] = None,
        market: Optional[str] = None,
        include_no_data: bool = False
    ) -> pd.DataFrame:
        """
        获取所有股票的上市首日信息（股票基本信息 + 上市首日交易数据）

        Args:
            list_status: 上市状态筛选 ('L'=上市, 'D'=退市, 'P'=暂停)。None 表示所有状态
            market: 市场筛选（如 '主板', '创业板', '科创板', '北交所'）。None 表示所有市场
            include_no_data: 是否包含没有上市首日交易数据的股票

        Returns:
            包含以下字段的 DataFrame:
            - ts_code: 股票代码
            - name: 股票名称
            - list_date: 上市日期
            - market: 市场
            - list_status: 上市状态
            - open, high, low, close: 上市首日价格
            - vol, amount: 上市首日成交量、成交额
        """
        # 检查必要的表是否存在
        if not self.db.table_exists('stock_basic'):
            logger.warning(
                "stock_basic 表不存在。请先使用 DataDownloader 下载数据：\n"
                "示例代码：\n"
                "  from tushare_db import DataDownloader\n"
                "  downloader = DataDownloader()\n"
                "  downloader.download_stock_basic('L')  # 下载股票列表\n"
                "  downloader.download_all_stocks_listing_first_day('L')  # 下载首日数据\n"
                "  downloader.close()"
            )
            return pd.DataFrame()
        
        # 构建 WHERE 条件
        conditions = ["s.list_date IS NOT NULL"]
        params = []

        if list_status:
            # 检查是否有 list_status 字段
            has_list_status = 'list_status' in self.db.get_table_columns('stock_basic')
            if has_list_status:
                conditions.append("s.list_status = ?")
                params.append(list_status)

        if market:
            conditions.append("s.market = ?")
            params.append(market)

        where_clause = " AND ".join(conditions)

        # SQL 查询：左连接 stock_basic 和 pro_bar
        query = f"""
            SELECT 
                s.ts_code,
                s.name,
                s.list_date,
                s.market,
                {'s.list_status,' if 'list_status' in self.db.get_table_columns('stock_basic') else ''}
                p.open,
                p.high,
                p.low,
                p.close,
                p.pre_close,
                p.vol,
                p.amount
            FROM stock_basic s
            LEFT JOIN daily p ON s.ts_code = p.ts_code AND s.list_date = p.trade_date
            WHERE {where_clause}
            ORDER BY s.list_date DESC
        """

        df = self.db.execute_query(query, params if params else None)

        # 如果不包含无交易数据的股票，过滤掉
        if not include_no_data and not df.empty:
            df = df[df['close'].notna()]

        # 友好提示：如果结果为空，提醒用户下载数据
        if df.empty:
            logger.warning(
                "未找到上市首日数据。请先使用 DataDownloader 下载数据：\n"
                "示例代码：\n"
                "  from tushare_db import DataDownloader\n"
                "  downloader = DataDownloader()\n"
                "  downloader.download_stock_basic('L')  # 下载股票列表\n"
                "  downloader.download_all_stocks_listing_first_day('L')  # 下载首日数据\n"
                "  downloader.close()"
            )

        return df

    def get_listing_first_day_info(
        self,
        ts_code: str
    ) -> pd.DataFrame:
        """
        获取单只股票的上市首日信息

        Args:
            ts_code: 股票代码

        Returns:
            包含股票基本信息和上市首日交易数据的 DataFrame
        """
        # 检查必要的表是否存在
        if not self.db.table_exists('stock_basic'):
            logger.warning(
                f"stock_basic 表不存在。请先使用 DataDownloader 下载股票列表数据。"
            )
            return pd.DataFrame()
        
        query = """
            SELECT 
                s.*,
                p.open,
                p.high,
                p.low,
                p.close,
                p.pre_close,
                p.vol,
                p.amount
            FROM stock_basic s
            LEFT JOIN daily p ON s.ts_code = p.ts_code AND s.list_date = p.trade_date
            WHERE s.ts_code = ?
        """

        df = self.db.execute_query(query, [ts_code])
        
        # 友好提示：如果结果为空，提醒用户下载数据
        if df.empty:
            logger.warning(
                f"未找到股票 {ts_code} 的信息。请先使用 DataDownloader 下载数据：\n"
                "示例代码：\n"
                "  from tushare_db import DataDownloader\n"
                "  downloader = DataDownloader()\n"
                "  downloader.download_stock_basic('L')  # 下载股票列表\n"
                f"  downloader.download_stock_daily('{ts_code}', start_date, end_date)  # 下载行情\n"
                "  downloader.close()"
            )

        return df

    # ==================== 其他常用接口 ====================

    def get_stock_company(self, ts_code: str) -> pd.DataFrame:
        """查询上市公司基本信息"""
        query = "SELECT * FROM stock_company WHERE ts_code = ?"
        return self.db.execute_query(query, [ts_code])

    def get_cyq_perf(
        self,
        ts_code: str,
        start_date: str,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """查询筹码平均成本"""
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')

        query = """
            SELECT * FROM cyq_perf
            WHERE ts_code = ?
            AND trade_date BETWEEN ? AND ?
            ORDER BY trade_date
        """
        return self.db.execute_query(query, [ts_code, start_date, end_date])

    def get_stk_factor_pro(
        self,
        ts_code: str,
        start_date: str,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """查询技术因子"""
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')

        query = """
            SELECT * FROM stk_factor_pro
            WHERE ts_code = ?
            AND trade_date BETWEEN ? AND ?
            ORDER BY trade_date
        """
        return self.db.execute_query(query, [ts_code, start_date, end_date])

    def get_moneyflow_ind_dc(
        self,
        start_date: str,
        end_date: Optional[str] = None,
        ts_code: Optional[str] = None
    ) -> pd.DataFrame:
        """
        查询董财板块资金流向

        Args:
            start_date: 开始日期
            end_date: 结束日期
            ts_code: 板块代码（可选）

        Returns:
            资金流向数据
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')

        if ts_code:
            query = """
                SELECT * FROM moneyflow_ind_dc
                WHERE ts_code = ?
                AND trade_date BETWEEN ? AND ?
                ORDER BY trade_date
            """
            params = [ts_code, start_date, end_date]
        else:
            query = """
                SELECT * FROM moneyflow_ind_dc
                WHERE trade_date BETWEEN ? AND ?
                ORDER BY trade_date, ts_code
            """
            params = [start_date, end_date]

        return self.db.execute_query(query, params)

    def get_moneyflow_dc(
        self,
        ts_code: Optional[str] = None,
        trade_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        查询个股资金流向数据（东方财富DC接口）

        数据说明：
        - 包含主力资金、超大单、大单、中单、小单的净流入额和占比
        - 每日盘后更新
        - 数据开始于20230911

        Args:
            ts_code: 股票代码（可选）
            trade_date: 交易日期 YYYYMMDD（可选）
            start_date: 开始日期（可选）
            end_date: 结束日期（可选）

        Returns:
            个股资金流向数据，包含以下字段：
            - trade_date: 交易日期
            - ts_code: 股票代码
            - name: 股票名称
            - pct_change: 涨跌幅
            - close: 最新价
            - net_amount: 今日主力净流入额（万元）
            - net_amount_rate: 今日主力净流入净占比（%）
            - buy_elg_amount: 今日超大单净流入额（万元）
            - buy_elg_amount_rate: 今日超大单净流入占比（%）
            - buy_lg_amount: 今日大单净流入额（万元）
            - buy_lg_amount_rate: 今日大单净流入占比（%）
            - buy_md_amount: 今日中单净流入额（万元）
            - buy_md_amount_rate: 今日中单净流入占比（%）
            - buy_sm_amount: 今日小单净流入额（万元）
            - buy_sm_amount_rate: 今日小单净流入占比（%）

        Examples:
            >>> # 获取单日全部股票数据
            >>> df = reader.get_moneyflow_dc(trade_date='20241011')
            >>>
            >>> # 获取单个股票数据
            >>> df = reader.get_moneyflow_dc(
            ...     ts_code='002149.SZ',
            ...     start_date='20240901',
            ...     end_date='20240913'
            ... )
        """
        conditions = []
        params = []

        # 按优先级处理查询条件
        if trade_date:
            # 查询特定日期的数据
            conditions.append("trade_date = ?")
            params.append(trade_date)
        elif start_date or end_date:
            # 查询日期范围
            if start_date and end_date:
                conditions.append("trade_date BETWEEN ? AND ?")
                params.extend([start_date, end_date])
            elif start_date:
                conditions.append("trade_date >= ?")
                params.append(start_date)
            elif end_date:
                conditions.append("trade_date <= ?")
                params.append(end_date)

        # 添加股票代码条件
        if ts_code:
            conditions.append("ts_code = ?")
            params.append(ts_code)

        # 构建查询语句
        query = "SELECT * FROM moneyflow_dc"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY trade_date DESC, ts_code"

        return self.db.execute_query(query, params)

    def get_moneyflow(
        self,
        ts_code: Optional[str] = None,
        trade_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        查询个股资金流向数据（标准接口）

        数据说明：
        - 获取沪深A股票资金流向数据，分析大单小单成交情况
        - 包含成交量和金额，买入和卖出分开统计
        - 数据开始于2010年

        资金分类：
        - 小单：5万以下
        - 中单：5万～20万
        - 大单：20万～100万
        - 特大单：成交额>=100万

        Args:
            ts_code: 股票代码（可选）
            trade_date: 交易日期 YYYYMMDD（可选）
            start_date: 开始日期（可选）
            end_date: 结束日期（可选）

        Returns:
            个股资金流向数据，包含以下字段：
            - ts_code: 股票代码
            - trade_date: 交易日期
            - buy_sm_vol: 小单买入量（手）
            - buy_sm_amount: 小单买入金额（万元）
            - sell_sm_vol: 小单卖出量（手）
            - sell_sm_amount: 小单卖出金额（万元）
            - buy_md_vol: 中单买入量（手）
            - buy_md_amount: 中单买入金额（万元）
            - sell_md_vol: 中单卖出量（手）
            - sell_md_amount: 中单卖出金额（万元）
            - buy_lg_vol: 大单买入量（手）
            - buy_lg_amount: 大单买入金额（万元）
            - sell_lg_vol: 大单卖出量（手）
            - sell_lg_amount: 大单卖出金额（万元）
            - buy_elg_vol: 特大单买入量（手）
            - buy_elg_amount: 特大单买入金额（万元）
            - sell_elg_vol: 特大单卖出量（手）
            - sell_elg_amount: 特大单卖出金额（万元）
            - net_mf_vol: 净流入量（手）
            - net_mf_amount: 净流入额（万元）

        Examples:
            >>> # 获取单日全部股票数据
            >>> df = reader.get_moneyflow(trade_date='20190315')
            >>>
            >>> # 获取单个股票数据
            >>> df = reader.get_moneyflow(
            ...     ts_code='002149.SZ',
            ...     start_date='20190115',
            ...     end_date='20190315'
            ... )
        """
        conditions = []
        params = []

        # 按优先级处理查询条件
        if trade_date:
            # 查询特定日期的数据
            conditions.append("trade_date = ?")
            params.append(trade_date)
        elif start_date or end_date:
            # 查询日期范围
            if start_date and end_date:
                conditions.append("trade_date BETWEEN ? AND ?")
                params.extend([start_date, end_date])
            elif start_date:
                conditions.append("trade_date >= ?")
                params.append(start_date)
            elif end_date:
                conditions.append("trade_date <= ?")
                params.append(end_date)

        # 添加股票代码条件
        if ts_code:
            conditions.append("ts_code = ?")
            params.append(ts_code)

        # 构建查询语句
        query = "SELECT * FROM moneyflow"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY trade_date DESC, ts_code"

        return self.db.execute_query(query, params)

    def get_index_classify(
        self,
        index_code: Optional[str] = None,
        level: Optional[str] = None,
        parent_code: Optional[str] = None,
        src: Optional[str] = None
    ) -> pd.DataFrame:
        """
        查询申万行业分类数据

        数据说明：
        - 申万行业分类数据，支持申万2014版和申万2021版
        - 包含L1（一级）、L2（二级）、L3（三级）三个层级
        - 每个层级都有对应的行业代码和行业名称

        Args:
            index_code: 指数代码（可选）
            level: 行业等级（L1/L2/L3）（可选）
            parent_code: 父级代码（可选），L1的父级代码为0
            src: 来源（SW2014/SW2021）（可选）

        Returns:
            申万行业分类数据，包含以下字段：
            - index_code: 指数代码
            - industry_name: 行业名称
            - parent_code: 父级代码
            - level: 行业级别（L1/L2/L3）
            - industry_code: 行业代码（唯一标识）
            - is_pub: 是否发布指数（1=是，0=否）
            - src: 分类来源（SW）

        Examples:
            >>> # 获取所有L1一级行业
            >>> df = reader.get_index_classify(level='L1')
            >>>
            >>> # 获取申万2021版的所有行业
            >>> df = reader.get_index_classify(src='SW2021')
            >>>
            >>> # 获取特定一级行业下的二级行业
            >>> df = reader.get_index_classify(level='L2', parent_code='801010')
        """
        conditions = []
        params = []

        # 添加各种查询条件
        if index_code:
            conditions.append("index_code = ?")
            params.append(index_code)

        if level:
            conditions.append("level = ?")
            params.append(level)

        if parent_code:
            conditions.append("parent_code = ?")
            params.append(parent_code)

        if src:
            conditions.append("src = ?")
            params.append(src)

        # 构建查询语句
        query = "SELECT * FROM index_classify"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY level, industry_code"

        return self.db.execute_query(query, params if params else None)

    def get_index_weight(
        self,
        index_code: Optional[str] = None,
        con_code: Optional[str] = None,
        trade_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        查询指数成分和权重数据

        数据说明：
        - 获取指数的成分股及权重信息
        - 可用于获取申万行业指数的成分股

        Args:
            index_code: 指数代码（可选），如 '801780.SI'
            con_code: 成分股代码（可选）
            trade_date: 交易日期 YYYYMMDD（可选）
            start_date: 开始日期（可选）
            end_date: 结束日期（可选）

        Returns:
            指数成分权重数据，包含以下字段：
            - index_code: 指数代码
            - con_code: 成分股代码
            - trade_date: 交易日期
            - weight: 权重（%）
            - in_date: 纳入日期
            - out_date: 剔除日期

        Examples:
            >>> # 获取申万银行行业的成分股
            >>> df = reader.get_index_weight(
            ...     index_code='801780.SI',
            ...     trade_date='20241201'
            ... )
            >>>
            >>> # 获取某个时间段的指数成分变化
            >>> df = reader.get_index_weight(
            ...     index_code='000300.SH',
            ...     start_date='20240101',
            ...     end_date='20241231'
            ... )
        """
        conditions = []
        params = []

        # 添加指数代码条件
        if index_code:
            conditions.append("index_code = ?")
            params.append(index_code)

        # 添加成分股代码条件
        if con_code:
            conditions.append("con_code = ?")
            params.append(con_code)

        # 添加日期条件
        if trade_date:
            conditions.append("trade_date = ?")
            params.append(trade_date)
        elif start_date or end_date:
            if start_date and end_date:
                conditions.append("trade_date BETWEEN ? AND ?")
                params.extend([start_date, end_date])
            elif start_date:
                conditions.append("trade_date >= ?")
                params.append(start_date)
            elif end_date:
                conditions.append("trade_date <= ?")
                params.append(end_date)

        # 构建查询语句
        query = "SELECT * FROM index_weight"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY index_code, trade_date, con_code"

        return self.db.execute_query(query, params if params else None)

    def get_index_member_all(
        self,
        l1_code: Optional[str] = None,
        l2_code: Optional[str] = None,
        l3_code: Optional[str] = None,
        ts_code: Optional[str] = None,
        is_new: Optional[str] = None,
        trade_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        查询申万行业成分构成数据（分级）

        数据说明：
        - 支持按行业代码查询成分股
        - 支持按股票代码查询所属行业
        - 支持历史回测（通过trade_date参数）

        Args:
            l1_code: 一级行业代码（可选）
            l2_code: 二级行业代码（可选）
            l3_code: 三级行业代码（可选）
            ts_code: 股票代码（可选）
            is_new: 是否最新（'Y'=是，'N'=否，None=全部，可选）
            trade_date: 交易日期 YYYYMMDD（可选，用于历史回测）

        Returns:
            申万行业成分数据，包含以下字段：
            - l1_code: 一级行业代码
            - l1_name: 一级行业名称
            - l2_code: 二级行业代码
            - l2_name: 二级行业名称
            - l3_code: 三级行业代码
            - l3_name: 三级行业名称
            - ts_code: 成分股票代码
            - name: 成分股票名称
            - in_date: 纳入日期
            - out_date: 剔除日期
            - is_new: 是否最新（Y/N）

        Examples:
            >>> # 查询黄金行业的当前成分股
            >>> df = reader.get_index_member_all(l3_code='850531.SI', is_new='Y')
            >>>
            >>> # 查询某只股票所属的所有行业
            >>> df = reader.get_index_member_all(ts_code='000001.SZ', is_new='Y')
            >>>
            >>> # 历史回测：查询2023年1月时的银行行业成分
            >>> df = reader.get_index_member_all(
            ...     l2_code='801780.SI',
            ...     trade_date='20230115'
            ... )
        """
        conditions = []
        params = []

        # 添加行业代码条件
        if l1_code:
            conditions.append("l1_code = ?")
            params.append(l1_code)

        if l2_code:
            conditions.append("l2_code = ?")
            params.append(l2_code)

        if l3_code:
            conditions.append("l3_code = ?")
            params.append(l3_code)

        # 添加股票代码条件
        if ts_code:
            conditions.append("ts_code = ?")
            params.append(ts_code)

        # 添加is_new条件
        if is_new:
            conditions.append("is_new = ?")
            params.append(is_new)

        # 添加历史回测条件
        if trade_date:
            # 在指定日期时，股票必须已经纳入且尚未剔除
            conditions.append("in_date <= ?")
            params.append(trade_date)
            conditions.append("(out_date IS NULL OR out_date > ?)")
            params.append(trade_date)

        # 构建查询语句
        query = "SELECT * FROM index_member_all"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY l1_code, l2_code, l3_code, ts_code"

        return self.db.execute_query(query, params if params else None)

    # ==================== 自定义 SQL 查询 ====================

    def query(self, sql: str, params: Optional[List] = None) -> pd.DataFrame:
        """
        执行自定义 SQL 查询

        Args:
            sql: SQL 查询语句
            params: 参数列表（用于参数化查询）

        Returns:
            查询结果 DataFrame

        Example:
            >>> reader.query("SELECT * FROM stock_basic WHERE name LIKE ?", ['%科技%'])
        """
        return self.db.execute_query(sql, params)

    # ==================== 辅助方法 ====================

    def _apply_adjustment(
        self,
        df: pd.DataFrame,
        ts_code: str,
        start_date: str,
        end_date: str,
        adj_type: str
    ) -> pd.DataFrame:
        """
        对日线数据应用复权计算

        Args:
            df: 原始日线数据
            ts_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            adj_type: 复权类型 ('qfq' 或 'hfq')

        Returns:
            复权后的数据
        """
        # 查询复权因子
        adj_query = """
            SELECT trade_date, adj_factor FROM adj_factor
            WHERE ts_code = ?
            AND trade_date BETWEEN ? AND ?
        """
        adj_df = self.db.execute_query(adj_query, [ts_code, start_date, end_date])

        if adj_df.empty:
            logger.warning(f"未找到复权因子: {ts_code}，返回不复权数据")
            return df

        # 合并复权因子
        df = df.merge(adj_df, on='trade_date', how='left')

        # 应用复权（传入 ts_code 以正确计算前复权）
        return self._adjust_prices(df, adj_type, ts_code)

    def _adjust_prices(self, df: pd.DataFrame, adj_type: str, ts_codes: Optional[Union[str, List[str]]] = None) -> pd.DataFrame:
        """
        对价格字段应用复权因子

        Args:
            df: 包含 adj_factor 列的 DataFrame
            adj_type: 复权类型
            ts_codes: 股票代码（单个或列表），前复权时必须提供以获取最新复权因子

        Returns:
            复权后的 DataFrame
        """
        if 'adj_factor' not in df.columns:
            return df

        # 需要复权的价格字段
        price_cols = ['open', 'high', 'low', 'close', 'pre_close']

        if adj_type == 'qfq':
            # 前复权：价格 × (最新复权因子 / 当日复权因子)
            # 特点：最新日期的价格不变，等于真实市场价格
            if not df.empty:
                # 获取每只股票的最新复权因子
                if ts_codes is None:
                    # 如果未提供 ts_codes，尝试从 df 中提取
                    if 'ts_code' in df.columns:
                        ts_codes = df['ts_code'].unique().tolist()
                    else:
                        # 降级：使用查询结果中的最后一天因子（旧行为，可能不准确）
                        logger.warning("前复权计算缺少 ts_codes 参数，使用查询结果中的最后一天因子")
                        latest_factor = df['adj_factor'].iloc[-1]
                        for col in price_cols:
                            if col in df.columns:
                                df[col] = df[col] * (df['adj_factor'] / latest_factor)
                        return df

                # 确保 ts_codes 是列表
                if isinstance(ts_codes, str):
                    ts_codes = [ts_codes]

                # 批量查询所有股票的最新复权因子
                placeholders = ','.join(['?'] * len(ts_codes))
                latest_adj_query = f"""
                    SELECT ts_code, adj_factor
                    FROM (
                        SELECT ts_code, adj_factor,
                               ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY trade_date DESC) as rn
                        FROM adj_factor
                        WHERE ts_code IN ({placeholders})
                    )
                    WHERE rn = 1
                """
                latest_adj_df = self.db.execute_query(latest_adj_query, ts_codes)

                if latest_adj_df.empty:
                    logger.warning(f"未找到股票的最新复权因子: {ts_codes}")
                    return df

                # 将最新复权因子映射到每只股票
                latest_factors = dict(zip(latest_adj_df['ts_code'], latest_adj_df['adj_factor']))

                # 应用前复权
                # 前复权公式：qfq_price = raw_price × (adj_factor / latest_factor)
                # 等价于：qfq_price = hfq_price / latest_factor
                if 'ts_code' in df.columns:
                    # 多股票情况
                    for col in price_cols:
                        if col in df.columns:
                            df[col] = df.apply(
                                lambda row: row[col] * (row['adj_factor'] / latest_factors.get(row['ts_code'], 1.0))
                                if pd.notna(row['adj_factor']) and latest_factors.get(row['ts_code'], 1.0) != 0 else row[col],
                                axis=1
                            )
                else:
                    # 单股票情况
                    latest_factor = latest_factors.get(ts_codes[0], df['adj_factor'].iloc[-1])
                    for col in price_cols:
                        if col in df.columns:
                            df[col] = df[col] * (df['adj_factor'] / latest_factor)

        elif adj_type == 'hfq':
            # 后复权：价格 × 复权因子
            # 特点：历史某时点价格不变，最新价格会被调整
            for col in price_cols:
                if col in df.columns:
                    df[col] = df[col] * df['adj_factor']

        return df

    def _check_empty(self, df: pd.DataFrame, query_desc: str):
        """
        检查查询结果是否为空

        Args:
            df: 查询结果
            query_desc: 查询描述（用于错误提示）

        Raises:
            DataReaderError: 如果启用了严格模式且结果为空
        """
        if df.empty and self.strict_mode:
            raise DataReaderError(
                f"数据不存在: {query_desc}\n"
                f"提示：请先使用 DataDownloader 下载数据"
            )

    # ==================== 申万行业指数 ====================

    def get_sw_daily(
        self,
        ts_code: Optional[str] = None,
        trade_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        查询申万行业指数日线行情

        Args:
            ts_code: 行业指数代码（可选，如 801010.SI）
            trade_date: 交易日期 YYYYMMDD（可选）
            start_date: 开始日期（可选）
            end_date: 结束日期（可选）

        Returns:
            DataFrame，包含字段：
            - ts_code: 指数代码
            - trade_date: 交易日期
            - name: 指数名称
            - open/high/low/close: OHLC数据
            - change: 涨跌点位
            - pct_change: 涨跌幅
            - vol: 成交量（万股）
            - amount: 成交额（万元）
            - pe: 市盈率
            - pb: 市净率
            - float_mv: 流通市值（万元）
            - total_mv: 总市值（万元）
        """
        conditions = []
        params = []

        if ts_code:
            conditions.append("ts_code = ?")
            params.append(ts_code)

        if trade_date:
            conditions.append("trade_date = ?")
            params.append(trade_date)

        if start_date:
            conditions.append("trade_date >= ?")
            params.append(start_date)

        if end_date:
            conditions.append("trade_date <= ?")
            params.append(end_date)

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        query = f"""
            SELECT * FROM sw_daily
            WHERE {where_clause}
            ORDER BY trade_date DESC, ts_code
        """

        df = self.db.execute_query(query, params)
        self._check_empty(df, f"申万行业日线 ts_code={ts_code}, trade_date={trade_date}")
        return df

    def get_sw_daily_pivot(
        self,
        start_date: str,
        end_date: str,
        field: str = 'pct_change'
    ) -> pd.DataFrame:
        """
        获取申万行业指数的透视表格式（行=日期，列=行业）

        适用于行业轮动分析、相关性分析等

        Args:
            start_date: 开始日期 YYYYMMDD
            end_date: 结束日期 YYYYMMDD
            field: 要提取的字段（pct_change, close, pe, pb, vol, amount）

        Returns:
            透视 DataFrame，index=trade_date, columns=行业名称
        """
        query = f"""
            SELECT trade_date, name, {field}
            FROM sw_daily
            WHERE trade_date >= ? AND trade_date <= ?
            ORDER BY trade_date, name
        """

        df = self.db.execute_query(query, [start_date, end_date])

        if df.empty:
            return df

        # 转为透视表
        pivot_df = df.pivot(index='trade_date', columns='name', values=field)
        pivot_df = pivot_df.sort_index()

        return pivot_df

    def table_exists(self, table_name: str) -> bool:
        """检查表是否存在"""
        return self.db.table_exists(table_name)

    def get_table_info(self, table_name: str) -> pd.DataFrame:
        """获取表结构信息"""
        return self.db.execute_query(f"PRAGMA table_info('{table_name}')")

    def close(self):
        """关闭数据库连接"""
        self.db.close()
        logger.info("DataReader 已关闭")
