# -*- coding: utf-8 -*-
"""
数据下载模块

职责：从 Tushare API 下载数据并存入 DuckDB，不提供查询功能
设计理念：简单直接，不做复杂的缓存策略判断
"""
import os
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from tqdm import tqdm

from .logger import get_logger
from .duckdb_manager import DuckDBManager
from .tushare_fetcher import TushareFetcher
from .rate_limit_config import PRESET_PROFILES, STANDARD_PROFILE

logger = get_logger(__name__)


class DataDownloaderError(Exception):
    """数据下载器异常"""
    pass


class DataDownloader:
    """
    数据下载器：专门负责从 Tushare 下载数据到 DuckDB

    特点：
    - 职责单一：只下载，不查询
    - 逻辑简单：不做复杂的增量判断和追溯性检测
    - 易于理解：每个方法都是直接的 fetch -> write 流程

    使用场景：
    - 初始化历史数据
    - 每日定时更新
    - 手动补充缺失数据
    """

    def __init__(
        self,
        tushare_token: Optional[str] = None,
        db_path: str = "tushare.db",
        rate_limit_profile: str = "standard"
    ):
        """
        初始化下载器

        Args:
            tushare_token: Tushare API token（优先使用此参数，其次从环境变量读取）
            db_path: DuckDB 数据库文件路径
            rate_limit_profile: 限速配置档位（'trial', 'standard', 'pro'）
        """
        self.tushare_token = tushare_token or os.getenv("TUSHARE_TOKEN")
        if not self.tushare_token:
            raise DataDownloaderError("请提供 Tushare token 或设置 TUSHARE_TOKEN 环境变量")

        # 初始化核心组件
        rate_config = PRESET_PROFILES.get(rate_limit_profile, STANDARD_PROFILE)
        self.fetcher = TushareFetcher(self.tushare_token, rate_config)
        self.db = DuckDBManager(db_path)

        logger.info(f"DataDownloader 初始化完成: db={db_path}, profile={rate_limit_profile}")

    # ==================== 基础数据下载 ====================

    def download_trade_calendar(self, start_date: str = '19900101', end_date: str = '20301231'):
        """
        下载交易日历

        Args:
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)

        Returns:
            下载的行数
        """
        logger.info(f"开始下载交易日历: {start_date} -> {end_date}")
        df = self.fetcher.fetch('trade_cal', start_date=start_date, end_date=end_date)

        if df.empty:
            logger.warning("交易日历数据为空")
            return 0

        self.db.write_dataframe(df, 'trade_cal', mode='replace')
        logger.info(f"交易日历下载完成: {len(df)} 行")
        return len(df)

    def download_stock_basic(self, list_status: str = 'L'):
        """
        下载股票基础信息

        Args:
            list_status: 上市状态 ('L'=上市, 'D'=退市, 'P'=暂停上市)

        Returns:
            下载的行数
        """
        logger.info(f"开始下载股票列表: list_status={list_status}")
        # 确保 fields 包含 list_status 字段，以便后续查询时可以筛选
        df = self.fetcher.fetch(
            'stock_basic',
            list_status=list_status,
            fields='ts_code,symbol,name,area,industry,list_date,market,list_status,exchange'
        )

        if df.empty:
            logger.warning(f"股票列表为空: list_status={list_status}")
            return 0

        # 使用 append 模式以支持多次调用（如分别下载 L/D/P）
        self.db.write_dataframe(df, 'stock_basic', mode='append')
        logger.info(f"股票列表下载完成: {len(df)} 行")
        return len(df)

    # ==================== 单股票数据下载 ====================

    def download_stock_daily(
        self,
        ts_code: str,
        start_date: str,
        end_date: Optional[str] = None
    ) -> int:
        """
        下载单只股票的日线数据（不复权）

        Args:
            ts_code: 股票代码 (如 '000001.SZ')
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)，默认为今天

        Returns:
            下载的行数
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')

        logger.debug(f"下载日线数据: {ts_code}, {start_date}-{end_date}")
        df = self.fetcher.fetch(
            'pro_bar',
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            asset='E',
            freq='D'
        )

        if df.empty:
            logger.debug(f"无数据: {ts_code}")
            return 0

        self.db.write_dataframe(df, 'pro_bar', mode='append')
        return len(df)

    def download_adj_factor(
        self,
        ts_code: str,
        start_date: str,
        end_date: Optional[str] = None
    ) -> int:
        """
        下载单只股票的复权因子

        Args:
            ts_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            下载的行数
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')

        logger.debug(f"下载复权因子: {ts_code}, {start_date}-{end_date}")
        df = self.fetcher.fetch(
            'adj_factor',
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date
        )

        if df.empty:
            logger.debug(f"无复权因子: {ts_code}")
            return 0

        self.db.write_dataframe(df, 'adj_factor', mode='append')
        return len(df)

    def download_daily_basic(
        self,
        ts_code: str,
        start_date: str,
        end_date: Optional[str] = None
    ) -> int:
        """
        下载单只股票的每日基本面指标

        Args:
            ts_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            下载的行数
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')

        logger.debug(f"下载每日基本面: {ts_code}, {start_date}-{end_date}")
        df = self.fetcher.fetch(
            'daily_basic',
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date
        )

        if df.empty:
            return 0

        self.db.write_dataframe(df, 'daily_basic', mode='append')
        return len(df)

    # ==================== 批量下载 ====================

    def download_all_stocks_daily(
        self,
        start_date: str,
        end_date: Optional[str] = None,
        list_status: str = 'L'
    ):
        """
        批量下载所有股票的日线数据

        Args:
            start_date: 开始日期
            end_date: 结束日期
            list_status: 股票状态筛选
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')

        # 获取股票列表
        # 检查表中是否有 list_status 字段（兼容旧数据）
        has_list_status = 'list_status' in self.db.get_table_columns('stock_basic')

        if has_list_status:
            stocks_df = self.db.execute_query(
                "SELECT ts_code FROM stock_basic WHERE list_status = ?",
                [list_status]
            )
        else:
            logger.warning(f"stock_basic 表中没有 list_status 字段，将查询所有股票")
            stocks_df = self.db.execute_query("SELECT ts_code FROM stock_basic")

        if stocks_df.empty:
            logger.warning(f"未找到股票列表，请先运行 download_stock_basic('{list_status}')")
            return

        total_stocks = len(stocks_df)
        logger.info(f"开始批量下载 {total_stocks} 只股票的日线数据: {start_date}-{end_date}")

        success_count = 0
        fail_count = 0

        for ts_code in tqdm(stocks_df['ts_code'], desc="下载日线数据"):
            try:
                rows = self.download_stock_daily(ts_code, start_date, end_date)
                if rows > 0:
                    success_count += 1
            except Exception as e:
                logger.error(f"下载失败: {ts_code}, 错误: {e}")
                fail_count += 1

        logger.info(f"批量下载完成: 成功 {success_count}, 失败 {fail_count}")

    def download_all_adj_factors(
        self,
        start_date: str,
        end_date: Optional[str] = None,
        list_status: str = 'L'
    ):
        """
        批量下载所有股票的复权因子

        Args:
            start_date: 开始日期
            end_date: 结束日期
            list_status: 股票状态筛选
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')

        # 检查表中是否有 list_status 字段（兼容旧数据）
        has_list_status = 'list_status' in self.db.get_table_columns('stock_basic')

        if has_list_status:
            stocks_df = self.db.execute_query(
                "SELECT ts_code FROM stock_basic WHERE list_status = ?",
                [list_status]
            )
        else:
            logger.warning(f"stock_basic 表中没有 list_status 字段，将查询所有股票")
            stocks_df = self.db.execute_query("SELECT ts_code FROM stock_basic")

        if stocks_df.empty:
            logger.warning(f"未找到股票列表")
            return

        total_stocks = len(stocks_df)
        logger.info(f"开始批量下载 {total_stocks} 只股票的复权因子: {start_date}-{end_date}")

        success_count = 0
        for ts_code in tqdm(stocks_df['ts_code'], desc="下载复权因子"):
            try:
                rows = self.download_adj_factor(ts_code, start_date, end_date)
                if rows > 0:
                    success_count += 1
            except Exception as e:
                logger.error(f"下载失败: {ts_code}, 错误: {e}")

        logger.info(f"批量下载完成: 成功 {success_count}")

    def download_all_stocks_listing_first_day(
        self,
        list_status: str = 'L',
        include_adj_factor: bool = True
    ):
        """
        批量下载所有股票的上市首日数据（优化稀疏数据场景）

        这个方法专门用于只需要上市首日数据的场景，避免下载完整历史数据。
        
        Args:
            list_status: 股票状态筛选 ('L'=上市, 'D'=退市, 'P'=暂停)
            include_adj_factor: 是否同时下载复权因子（建议开启）
        """
        # 检查表中是否有 list_status 字段
        has_list_status = 'list_status' in self.db.get_table_columns('stock_basic')

        if has_list_status:
            stocks_df = self.db.execute_query(
                """
                SELECT ts_code, name, list_date FROM stock_basic 
                WHERE list_status = ? AND list_date IS NOT NULL
                ORDER BY list_date
                """,
                [list_status]
            )
        else:
            logger.warning(f"stock_basic 表中没有 list_status 字段，将查询所有股票")
            stocks_df = self.db.execute_query(
                "SELECT ts_code, name, list_date FROM stock_basic WHERE list_date IS NOT NULL ORDER BY list_date"
            )

        if stocks_df.empty:
            logger.warning(f"未找到股票列表，请先运行 download_stock_basic('{list_status}')")
            return

        total_stocks = len(stocks_df)
        logger.info(f"开始批量下载 {total_stocks} 只股票的上市首日数据")

        success_count = 0
        fail_count = 0

        for _, stock in tqdm(stocks_df.iterrows(), total=total_stocks, desc="下载上市首日数据"):
            ts_code = stock['ts_code']
            list_date = stock['list_date']
            
            try:
                # 下载上市首日行情
                rows = self.download_stock_daily(ts_code, list_date, list_date)
                
                # 同时下载复权因子（如果需要）
                if include_adj_factor:
                    self.download_adj_factor(ts_code, list_date, list_date)
                
                if rows > 0:
                    success_count += 1
                else:
                    logger.debug(f"{ts_code} ({stock['name']}) 上市首日 {list_date} 无数据")
                    
            except Exception as e:
                logger.error(f"下载失败: {ts_code}, 错误: {e}")
                fail_count += 1

        logger.info(f"批量下载上市首日数据完成: 成功 {success_count}, 失败 {fail_count}")
        return {
            'total': total_stocks,
            'success': success_count,
            'fail': fail_count
        }

    # ==================== 按日期下载（适合每日更新）====================

    def download_daily_data_by_date(self, trade_date: str):
        """
        按日期下载所有股票的数据（适合每日更新场景）

        下载内容：
        - 当日所有股票的日线数据
        - 当日所有股票的复权因子
        - 当日所有股票的每日基本面指标

        Args:
            trade_date: 交易日期 (YYYYMMDD)
        """
        logger.info(f"开始按日期下载数据: {trade_date}")

        # 检查是否为交易日
        cal_df = self.db.execute_query(
            "SELECT is_open FROM trade_cal WHERE cal_date = ?",
            [trade_date]
        )

        if cal_df.empty:
            logger.warning(f"交易日历中未找到日期 {trade_date}，请先下载交易日历")
            return

        # 兼容不同类型：字符串 '1' 或整数 1
        is_open = cal_df.iloc[0]['is_open']
        if str(is_open) != '1':
            logger.info(f"{trade_date} 不是交易日，跳过")
            return

        # 1. 下载日线数据（pro_bar 不支持 trade_date，使用 start_date=end_date）
        logger.info(f"下载日线数据: {trade_date}")
        df_daily = self.fetcher.fetch(
            'pro_bar',
            start_date=trade_date,
            end_date=trade_date,
            asset='E'
        )
        if not df_daily.empty:
            self.db.write_dataframe(df_daily, 'pro_bar', mode='append')
            logger.info(f"日线数据: {len(df_daily)} 行")

        # 2. 下载复权因子（支持 trade_date）
        logger.info(f"下载复权因子: {trade_date}")
        df_adj = self.fetcher.fetch('adj_factor', trade_date=trade_date)
        if not df_adj.empty:
            self.db.write_dataframe(df_adj, 'adj_factor', mode='append')
            logger.info(f"复权因子: {len(df_adj)} 行")

        # 3. 下载每日基本面（支持 trade_date）
        logger.info(f"下载每日基本面: {trade_date}")
        df_basic = self.fetcher.fetch('daily_basic', trade_date=trade_date)
        if not df_basic.empty:
            self.db.write_dataframe(df_basic, 'daily_basic', mode='append')
            logger.info(f"每日基本面: {len(df_basic)} 行")


        logger.info(f"按日期下载完成: {trade_date}")

    def download_cyq_perf(self, trade_date: str) -> int:
        """
        下载筹码分布（每日获利筹码）数据

        Args:
            trade_date: 交易日期 (YYYYMMDD)

        Returns:
            下载的行数
        """
        logger.debug(f"下载筹码分布: {trade_date}")
        df = self.fetcher.fetch(
            'cyq_perf',
            trade_date=trade_date
        )

        if df.empty:
            logger.debug(f"无筹码分布数据: {trade_date}")
            return 0

        self.db.write_dataframe(df, 'cyq_perf', mode='append')
        logger.info(f"筹码分布数据: {len(df)} 行 ({trade_date})")
        return len(df)

    def download_dc_member(self, trade_date: str) -> int:
        """
        下载龙虎榜机构席位交易明细数据

        Args:
            trade_date: 交易日期 (YYYYMMDD)

        Returns:
            下载的行数
        """
        logger.debug(f"下载龙虎榜机构席位: {trade_date}")
        df = self.fetcher.fetch(
            'dc_member',
            trade_date=trade_date
        )

        if df.empty:
            logger.debug(f"无龙虎榜机构席位数据: {trade_date}")
            return 0

        self.db.write_dataframe(df, 'dc_member', mode='append')
        logger.info(f"龙虎榜机构席位数据: {len(df)} 行 ({trade_date})")
        return len(df)

    def download_moneyflow_ind_dc(self, trade_date: str) -> int:
        """
        下载行业资金流向（沪深通）数据

        Args:
            trade_date: 交易日期 (YYYYMMDD)

        Returns:
            下载的行数
        """
        logger.debug(f"下载行业资金流向: {trade_date}")
        df = self.fetcher.fetch(
            'moneyflow_ind_dc',
            trade_date=trade_date
        )

        if df.empty:
            logger.debug(f"无行业资金流向数据: {trade_date}")
            return 0

        self.db.write_dataframe(df, 'moneyflow_ind_dc', mode='append')
        logger.info(f"行业资金流向数据: {len(df)} 行 ({trade_date})")
        return len(df)

    def download_moneyflow_dc(
        self,
        ts_code: Optional[str] = None,
        trade_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> int:
        """
        下载个股资金流向数据（东方财富DC接口）

        数据说明：
        - 每日盘后更新
        - 数据开始于20230911
        - 单次最大获取6000条数据
        - 需要至少5000积分

        Args:
            ts_code: 股票代码（可选）
            trade_date: 交易日期 YYYYMMDD（可选）
            start_date: 开始日期（可选）
            end_date: 结束日期（可选）

        Returns:
            下载的行数
        """
        logger.debug(f"下载个股资金流向: ts_code={ts_code}, trade_date={trade_date}, "
                    f"start_date={start_date}, end_date={end_date}")

        df = self.fetcher.fetch(
            'moneyflow_dc',
            ts_code=ts_code,
            trade_date=trade_date,
            start_date=start_date,
            end_date=end_date
        )

        if df.empty:
            logger.debug(f"无个股资金流向数据")
            return 0

        self.db.write_dataframe(df, 'moneyflow_dc', mode='append')
        logger.info(f"个股资金流向数据: {len(df)} 行")
        return len(df)

    def download_moneyflow(
        self,
        ts_code: Optional[str] = None,
        trade_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> int:
        """
        下载个股资金流向数据（标准接口）

        数据说明：
        - 获取沪深A股票资金流向数据，分析大单小单成交情况
        - 数据开始于2010年
        - 单次最大获取6000条数据
        - 需要至少2000积分

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
            注意：股票代码和时间参数至少输入一个

        Returns:
            下载的行数
        """
        logger.debug(f"下载个股资金流向(标准): ts_code={ts_code}, trade_date={trade_date}, "
                    f"start_date={start_date}, end_date={end_date}")

        df = self.fetcher.fetch(
            'moneyflow',
            ts_code=ts_code,
            trade_date=trade_date,
            start_date=start_date,
            end_date=end_date
        )

        if df.empty:
            logger.debug(f"无个股资金流向数据")
            return 0

        self.db.write_dataframe(df, 'moneyflow', mode='append')
        logger.info(f"个股资金流向数据(标准): {len(df)} 行")
        return len(df)

    def download_index_classify(
        self,
        index_code: Optional[str] = None,
        level: Optional[str] = None,
        parent_code: Optional[str] = None,
        src: Optional[str] = None
    ) -> int:
        """
        下载申万行业分类数据

        数据说明：
        - 获取申万行业分类列表
        - 支持L1/L2/L3三个层级
        - 支持SW2014和SW2021两个版本

        Args:
            index_code: 指数代码（可选）
            level: 行业分级 L1/L2/L3（可选）
            parent_code: 父级代码，一级为0（可选）
            src: 指数来源 SW2014/SW2021（可选）

        Returns:
            下载的行数

        Examples:
            >>> # 获取申万一级行业列表
            >>> downloader.download_index_classify(level='L1', src='SW2021')
            >>>
            >>> # 获取申万二级行业列表
            >>> downloader.download_index_classify(level='L2', src='SW2021')
        """
        logger.debug(f"下载申万行业分类: level={level}, src={src}, "
                    f"index_code={index_code}, parent_code={parent_code}")

        df = self.fetcher.fetch(
            'index_classify',
            index_code=index_code,
            level=level,
            parent_code=parent_code,
            src=src
        )

        if df.empty:
            logger.debug(f"无行业分类数据")
            return 0

        self.db.write_dataframe(df, 'index_classify', mode='append')
        logger.info(f"行业分类数据: {len(df)} 行")
        return len(df)

    def download_index_weight(
        self,
        index_code: Optional[str] = None,
        trade_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> int:
        """
        下载指数成分和权重数据

        数据说明：
        - 获取指数的成分股及权重信息
        - 可用于获取申万行业指数的成分股

        Args:
            index_code: 指数代码（可选），如 '000300.SH', '801780.SI' 等
            trade_date: 交易日期 YYYYMMDD（可选）
            start_date: 开始日期（可选）
            end_date: 结束日期（可选）

        Returns:
            下载的行数

        Examples:
            >>> # 下载沪深300成分股
            >>> downloader.download_index_weight(
            ...     index_code='000300.SH',
            ...     trade_date='20241201'
            ... )
            >>>
            >>> # 下载申万银行行业成分股
            >>> downloader.download_index_weight(
            ...     index_code='801780.SI',
            ...     start_date='20240101',
            ...     end_date='20241231'
            ... )
        """
        logger.debug(f"下载指数成分数据: index_code={index_code}, "
                    f"trade_date={trade_date}, start_date={start_date}, end_date={end_date}")

        df = self.fetcher.fetch(
            'index_weight',
            index_code=index_code,
            trade_date=trade_date,
            start_date=start_date,
            end_date=end_date
        )

        if df.empty:
            logger.debug(f"无指数成分数据")
            return 0

        self.db.write_dataframe(df, 'index_weight', mode='append')
        logger.info(f"指数成分数据: {len(df)} 行")
        return len(df)

    def download_index_member_all(
        self,
        l1_code: Optional[str] = None,
        l2_code: Optional[str] = None,
        l3_code: Optional[str] = None,
        ts_code: Optional[str] = None,
        is_new: Optional[str] = None
    ) -> int:
        """
        下载申万行业成分构成数据（分级）

        数据说明：
        - 按三级分类提取申万行业成分
        - 可提供某个分类的所有成分，也可按股票代码提取所属分类
        - 支持完整历史记录（用于历史回测）

        Args:
            l1_code: 一级行业代码（可选）
            l2_code: 二级行业代码（可选）
            l3_code: 三级行业代码（可选）
            ts_code: 股票代码（可选）
            is_new: 是否最新（'Y'=是，'N'=否，None=全部，可选）

        Returns:
            下载的行数

        Examples:
            >>> # 下载黄金行业的所有成分股（含历史）
            >>> downloader.download_index_member_all(l3_code='850531.SI')
            >>>
            >>> # 下载某只股票所属的所有行业
            >>> downloader.download_index_member_all(ts_code='000001.SZ')
            >>>
            >>> # 只下载当前最新成分（不含历史）
            >>> downloader.download_index_member_all(l3_code='850531.SI', is_new='Y')
        """
        logger.debug(f"下载申万行业成分: l1_code={l1_code}, l2_code={l2_code}, "
                    f"l3_code={l3_code}, ts_code={ts_code}, is_new={is_new}")

        df = self.fetcher.fetch(
            'index_member_all',
            l1_code=l1_code,
            l2_code=l2_code,
            l3_code=l3_code,
            ts_code=ts_code,
            is_new=is_new
        )

        if df.empty:
            logger.debug(f"无申万行业成分数据")
            return 0

        self.db.write_dataframe(df, 'index_member_all', mode='append')
        logger.info(f"申万行业成分数据: {len(df)} 行")
        return len(df)


    # ==================== 数据完整性验证 ====================

    def validate_data_integrity(
        self,
        start_date: str,
        end_date: str,
        sample_size: int = 10
    ) -> Dict[str, Any]:
        """
        验证数据完整性，检查是否有缺失的日期

        Args:
            start_date: 开始日期
            end_date: 结束日期
            sample_size: 抽样检查的股票数量

        Returns:
            验证结果字典，包含:
            - is_valid: 是否完整
            - missing_dates: 缺失的交易日
            - sample_stocks: 抽样检查的股票及其缺失日期
        """
        logger.info(f"开始验证数据完整性: {start_date} -> {end_date}")

        # 1. 获取该期间的所有交易日
        trading_dates_df = self.db.execute_query(
            """
            SELECT cal_date FROM trade_cal
            WHERE cal_date BETWEEN ? AND ?
            AND (is_open = 1 OR is_open = '1')
            ORDER BY cal_date
            """,
            [start_date, end_date]
        )

        if trading_dates_df.empty:
            return {
                'is_valid': False,
                'error': '交易日历数据不存在，请先下载交易日历'
            }

        expected_dates = set(trading_dates_df['cal_date'].astype(str))
        logger.info(f"期间内交易日总数: {len(expected_dates)}")

        # 2. 抽样检查部分股票
        # 检查表中是否有 list_status 字段（兼容旧数据）
        has_list_status = 'list_status' in self.db.get_table_columns('stock_basic')

        if has_list_status:
            stocks_df = self.db.execute_query(
                "SELECT ts_code FROM stock_basic WHERE list_status = 'L' LIMIT ?",
                [sample_size]
            )
        else:
            stocks_df = self.db.execute_query(
                "SELECT ts_code FROM stock_basic LIMIT ?",
                [sample_size]
            )

        if stocks_df.empty:
            return {
                'is_valid': False,
                'error': '股票列表为空，请先下载股票基础信息'
            }

        sample_results = []
        for ts_code in stocks_df['ts_code']:
            actual_dates_df = self.db.execute_query(
                """
                SELECT trade_date FROM pro_bar
                WHERE ts_code = ?
                AND trade_date BETWEEN ? AND ?
                """,
                [ts_code, start_date, end_date]
            )

            actual_dates = set(actual_dates_df['trade_date'].astype(str))
            missing_dates = expected_dates - actual_dates

            sample_results.append({
                'ts_code': ts_code,
                'expected': len(expected_dates),
                'actual': len(actual_dates),
                'missing': len(missing_dates),
                'missing_dates': sorted(list(missing_dates))[:10]  # 只显示前10个
            })

        # 3. 汇总结果
        total_missing = sum(r['missing'] for r in sample_results)
        is_valid = total_missing == 0

        result = {
            'is_valid': is_valid,
            'trading_days': len(expected_dates),
            'sample_size': len(sample_results),
            'sample_stocks': sample_results,
            'total_missing_in_sample': total_missing
        }

        if is_valid:
            logger.info("✓ 数据完整性验证通过")
        else:
            logger.warning(f"✗ 数据不完整，抽样发现 {total_missing} 个缺失")

        return result

    def close(self):
        """关闭数据库连接"""
        self.db.close()
        logger.info("DataDownloader 已关闭")
