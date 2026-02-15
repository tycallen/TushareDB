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
        end_date: Optional[str] = None,
        asset: str = 'E'
    ) -> int:
        """
        下载单只股票或其它资产的日线数据（不复权）

        Args:
            ts_code: 证券代码 (如 '000001.SZ')
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)，默认为今天
            asset: 资产类别：E股票 FD基金 I指数 CB可转债 FT期货

        Returns:
            下载的行数
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')

        logger.debug(f"下载日线数据: {ts_code}, {start_date}-{end_date}, asset={asset}")
        df = self.fetcher.fetch(
            'daily',
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date
        )

        if df.empty:
            logger.debug(f"无数据: {ts_code}")
            return 0

        self.db.write_dataframe(df, 'daily', mode='append')
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

    def download_daily_data_by_date(self, trade_date: str, asset: str = 'E'):
        """
        按日期下载所有证券的数据（适合每日更新场景）

        下载内容：
        - 当日所有资产类型的日线数据
        - 当日所有股票的复权因子（仅当 asset='E' 时）
        - 当日所有股票的每日基本面指标（仅当 asset='E' 时）

        Args:
            trade_date: 交易日期 (YYYYMMDD)
            asset: 资产类别：E股票 FD基金 I指数 CB可转债 FT期货
        """
        logger.info(f"开始按日期下载数据: {trade_date}, asset={asset}")

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

        # 1. 下载日线数据
        logger.info(f"下载日线数据: {trade_date}, asset={asset}")
        df_daily = self.fetcher.fetch(
            'daily',
            trade_date=trade_date
        )
        if not df_daily.empty:
            self.db.write_dataframe(df_daily, 'daily', mode='append')
            logger.info(f"日线数据: {len(df_daily)} 行")

        # 只有在下载股票数据时才下载复权因子和基本面
        if asset == 'E':
            # 2. 下载复权因子（支持 trade_date）
            logger.info(f"下载复权因子: {trade_date}")
            df_adj = self.fetcher.fetch('adj_factor', trade_date=trade_date)
            if not df_adj.empty:
                self.db.write_dataframe(df_adj, 'adj_factor', mode='append')
                logger.info(f"复权因子: {len(df_adj)} 行")
            else:
                logger.warning(f"复权因子数据为空: {trade_date}，可能需要后续补充")

            # 3. 下载每日基本面（支持 trade_date）
            logger.info(f"下载每日基本面: {trade_date}")
            df_basic = self.fetcher.fetch('daily_basic', trade_date=trade_date)
            if not df_basic.empty:
                self.db.write_dataframe(df_basic, 'daily_basic', mode='append')
                logger.info(f"每日基本面: {len(df_basic)} 行")
            else:
                logger.warning(f"每日基本面数据为空: {trade_date}，可能需要后续补充")


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

    def download_cyq_chips(self, ts_code: str, trade_date: str) -> int:
        """
        下载筹码分布详情数据（单只股票单日）

        Args:
            ts_code: 股票代码
            trade_date: 交易日期 (YYYYMMDD)

        Returns:
            下载的行数

        说明:
            - 数据从2018年开始
            - 同一天同一股票有多个价格的筹码占比记录
            - 需要 5000+ 积分
        """
        logger.debug(f"下载筹码分布详情: {ts_code} {trade_date}")
        df = self.fetcher.fetch(
            'cyq_chips',
            ts_code=ts_code,
            trade_date=trade_date
        )

        if df.empty:
            logger.debug(f"无筹码分布详情数据: {ts_code} {trade_date}")
            return 0

        self.db.write_dataframe(df, 'cyq_chips', mode='append')
        logger.debug(f"筹码分布详情: {len(df)} 行 ({ts_code} {trade_date})")
        return len(df)

    def download_cyq_chips_by_stock(
        self,
        ts_code: str,
        start_date: str,
        end_date: str
    ) -> int:
        """
        按股票+日期范围下载筹码分布详情数据

        Args:
            ts_code: 股票代码
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)

        Returns:
            下载的行数

        说明:
            - 一次 API 调用可获取多日数据
            - 单次最大 2000 条（约 15 个交易日）
            - 比逐日下载效率高很多
        """
        logger.debug(f"下载筹码分布详情: {ts_code} {start_date}~{end_date}")
        df = self.fetcher.fetch(
            'cyq_chips',
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date
        )

        if df.empty:
            logger.debug(f"无筹码分布详情数据: {ts_code}")
            return 0

        self.db.write_dataframe(df, 'cyq_chips', mode='append')
        logger.debug(f"筹码分布详情: {len(df)} 行 ({ts_code})")
        return len(df)

    def download_cyq_chips_incremental(
        self,
        start_date: str,
        end_date: str
    ) -> int:
        """
        增量下载所有股票的筹码分布详情数据（按股票遍历）

        Args:
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)

        Returns:
            下载的总行数

        说明:
            - 遍历所有上市股票，每只股票一次性下载日期范围内的所有数据
            - 比按日期遍历效率高很多（减少 API 调用次数）
            - 例如：更新 10 天数据
              - 按日期遍历：10天 × 5000股票 = 50000 次 API
              - 按股票遍历：5000股票 × 1次 = 5000 次 API
        """
        logger.info(f"增量下载筹码分布详情: {start_date} ~ {end_date}")

        # 获取所有上市股票
        stocks_df = self.db.execute_query(
            "SELECT ts_code FROM stock_basic WHERE list_status = 'L'"
        )

        if stocks_df.empty:
            logger.warning("没有找到上市股票列表")
            return 0

        total_rows = 0
        success_count = 0
        stock_list = stocks_df['ts_code'].tolist()

        for i, ts_code in enumerate(stock_list):
            try:
                rows = self.download_cyq_chips_by_stock(ts_code, start_date, end_date)
                if rows > 0:
                    total_rows += rows
                    success_count += 1

                # 每 500 只股票打印一次进度
                if (i + 1) % 500 == 0:
                    logger.info(f"  进度: {i + 1}/{len(stock_list)}, 已下载 {total_rows} 行")

            except Exception as e:
                logger.debug(f"  {ts_code} 下载失败: {e}")
                continue

        logger.info(f"筹码分布详情增量下载完成: {success_count}/{len(stock_list)} 只股票, 共 {total_rows} 行")
        return total_rows

    def download_cyq_chips_by_date(self, trade_date: str) -> int:
        """
        [已废弃] 按日期批量下载所有股票的筹码分布详情数据

        推荐使用 download_cyq_chips_incremental 方法，效率更高
        """
        logger.info(f"批量下载筹码分布详情: {trade_date}")

        # 获取所有上市股票
        stocks_df = self.db.execute_query(
            "SELECT ts_code FROM stock_basic WHERE list_status = 'L'"
        )

        if stocks_df.empty:
            logger.warning("没有找到上市股票列表")
            return 0

        total_rows = 0
        success_count = 0
        stock_list = stocks_df['ts_code'].tolist()

        for i, ts_code in enumerate(stock_list):
            try:
                rows = self.download_cyq_chips(ts_code, trade_date)
                if rows > 0:
                    total_rows += rows
                    success_count += 1

                # 每100只股票打印一次进度
                if (i + 1) % 100 == 0:
                    logger.info(f"  进度: {i + 1}/{len(stock_list)}, 已下载 {total_rows} 行")

            except Exception as e:
                logger.debug(f"  {ts_code} 下载失败: {e}")
                continue

        logger.info(f"筹码分布详情批量下载完成: {success_count}/{len(stock_list)} 只股票, 共 {total_rows} 行")
        return total_rows

    def download_stk_factor_pro(self, ts_code: str, trade_date: str) -> int:
        """
        下载股票技术因子数据（专业版）

        Args:
            ts_code: 股票代码
            trade_date: 交易日期 (YYYYMMDD)

        Returns:
            下载的行数

        说明:
            - 包含 MACD、KDJ、RSI、BOLL 等技术指标
            - 提供不复权(_bfq)、前复权(_qfq)、后复权(_hfq)三种价格
            - 需要 5000+ 积分
        """
        logger.debug(f"下载技术因子: {ts_code} {trade_date}")
        df = self.fetcher.fetch(
            'stk_factor_pro',
            ts_code=ts_code,
            trade_date=trade_date
        )

        if df.empty:
            logger.debug(f"无技术因子数据: {ts_code} {trade_date}")
            return 0

        self.db.write_dataframe(df, 'stk_factor_pro', mode='append')
        logger.debug(f"技术因子: {len(df)} 行 ({ts_code} {trade_date})")
        return len(df)

    def download_stk_factor_pro_by_date(self, trade_date: str) -> int:
        """
        按日期批量下载所有股票的技术因子数据

        Args:
            trade_date: 交易日期 (YYYYMMDD)

        Returns:
            下载的总行数

        说明:
            - 使用 trade_date 参数一次性获取当日所有股票的技术因子
            - 单次 API 调用即可获取约 5000+ 只股票的数据
            - 比逐股票下载效率高 5000x
        """
        logger.info(f"批量下载技术因子: {trade_date}")

        df = self.fetcher.fetch(
            'stk_factor_pro',
            trade_date=trade_date
        )

        if df.empty:
            logger.info(f"无技术因子数据: {trade_date}")
            return 0

        self.db.write_dataframe(df, 'stk_factor_pro', mode='append')
        logger.info(f"技术因子: {len(df)} 行 ({trade_date})")
        return len(df)

    def _download_stk_factor_pro_by_date_legacy(self, trade_date: str) -> int:
        """
        [已废弃] 按日期逐股票下载技术因子数据

        保留此方法作为备用，新代码请使用 download_stk_factor_pro_by_date
        """
        logger.info(f"批量下载技术因子 (逐股票模式): {trade_date}")

        # 获取所有上市股票
        stocks_df = self.db.execute_query(
            "SELECT ts_code FROM stock_basic WHERE list_status = 'L'"
        )

        if stocks_df.empty:
            logger.warning("没有找到上市股票列表")
            return 0

        total_rows = 0
        success_count = 0
        stock_list = stocks_df['ts_code'].tolist()

        for i, ts_code in enumerate(stock_list):
            try:
                rows = self.download_stk_factor_pro(ts_code, trade_date)
                if rows > 0:
                    total_rows += rows
                    success_count += 1

                # 每100只股票打印一次进度
                if (i + 1) % 100 == 0:
                    logger.info(f"  进度: {i + 1}/{len(stock_list)}, 已下载 {total_rows} 行")

            except Exception as e:
                logger.debug(f"  {ts_code} 下载失败: {e}")
                continue

        logger.info(f"技术因子批量下载完成: {success_count}/{len(stock_list)} 只股票, 共 {total_rows} 行")
        return total_rows

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

    def download_dc_index(self, trade_date: str) -> int:
        """
        下载龙虎榜每日上榜个股明细数据

        Args:
            trade_date: 交易日期 (YYYYMMDD)

        Returns:
            下载的行数

        说明:
            - 包含上榜原因、上榜次数、净买入额等
            - 每日盘后更新
        """
        logger.debug(f"下载龙虎榜个股明细: {trade_date}")
        df = self.fetcher.fetch(
            'dc_index',
            trade_date=trade_date
        )

        if df.empty:
            logger.debug(f"无龙虎榜个股明细数据: {trade_date}")
            return 0

        self.db.write_dataframe(df, 'dc_index', mode='append')
        logger.info(f"龙虎榜个股明细数据: {len(df)} 行 ({trade_date})")
        return len(df)

    def download_limit_list_d(
        self,
        trade_date: Optional[str] = None,
        ts_code: Optional[str] = None,
        limit_type: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> int:
        """
        下载涨跌停和炸板数据

        Args:
            trade_date: 交易日期 (YYYYMMDD)
            ts_code: 股票代码（可选）
            limit_type: 涨跌停类型 U涨停 D跌停 Z炸板（可选）
            start_date: 开始日期（可选）
            end_date: 结束日期（可选）

        Returns:
            下载的行数

        说明:
            - 数据从2020年开始
            - 不包含ST股票
            - 单次最大2500条
            - 需要 5000+ 积分

        字段说明:
            - limit: D跌停 U涨停 Z炸板
            - fd_amount: 封单金额
            - first_time/last_time: 首次/最后封板时间
            - open_times: 炸板次数
            - up_stat: 涨停统计（N/T 表示T天有N次涨停）
            - limit_times: 连板数
        """
        logger.debug(f"下载涨跌停数据: trade_date={trade_date}, limit_type={limit_type}")
        df = self.fetcher.fetch(
            'limit_list_d',
            trade_date=trade_date,
            ts_code=ts_code,
            limit_type=limit_type,
            start_date=start_date,
            end_date=end_date
        )

        if df.empty:
            logger.debug(f"无涨跌停数据: {trade_date}")
            return 0

        self.db.write_dataframe(df, 'limit_list_d', mode='append')
        logger.info(f"涨跌停数据: {len(df)} 行 ({trade_date})")
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


    # ==================== 财务指标数据 ====================

    def download_fina_indicator_vip(self, period: str) -> int:
        """
        下载财务指标数据（VIP接口，按报告期批量获取全部股票）

        数据说明：
        - 获取某一季度全部上市公司财务指标数据
        - 包含ROE、ROA、毛利率、净利率等关键指标
        - 需要至少5000积分

        Args:
            period: 报告期 YYYYMMDD，如 20231231（年报）、20230630（半年报）

        Returns:
            下载的行数
        """
        logger.debug(f"下载财务指标(VIP): period={period}")

        df = self.fetcher.fetch('fina_indicator_vip', period=period)

        if df.empty:
            logger.debug(f"无财务指标数据: period={period}")
            return 0

        self.db.write_dataframe(df, 'fina_indicator_vip', mode='append')
        logger.info(f"财务指标数据(VIP): {len(df)} 行 (period={period})")
        return len(df)

    # ==================== 财务报表数据 ====================

    def download_income_vip(self, period: str) -> int:
        """
        下载利润表数据（VIP接口，按报告期批量获取全部股票）

        数据说明：
        - 获取某一季度全部上市公司利润表数据
        - 单次最大获取5000条数据
        - 需要至少5000积分

        Args:
            period: 报告期 YYYYMMDD，如 20231231（年报）、20230630（半年报）

        Returns:
            下载的行数
        """
        logger.debug(f"下载利润表(VIP): period={period}")

        df = self.fetcher.fetch('income_vip', period=period)

        if df.empty:
            logger.debug(f"无利润表数据: period={period}")
            return 0

        self.db.write_dataframe(df, 'income', mode='append')
        logger.info(f"利润表数据(VIP): {len(df)} 行 (period={period})")
        return len(df)

    def download_balancesheet_vip(self, period: str) -> int:
        """
        下载资产负债表数据（VIP接口，按报告期批量获取全部股票）

        数据说明：
        - 获取某一季度全部上市公司资产负债表数据
        - 单次最大获取5000条数据
        - 需要至少5000积分

        Args:
            period: 报告期 YYYYMMDD，如 20231231（年报）、20230630（半年报）

        Returns:
            下载的行数
        """
        logger.debug(f"下载资产负债表(VIP): period={period}")

        df = self.fetcher.fetch('balancesheet_vip', period=period)

        if df.empty:
            logger.debug(f"无资产负债表数据: period={period}")
            return 0

        self.db.write_dataframe(df, 'balancesheet', mode='append')
        logger.info(f"资产负债表数据(VIP): {len(df)} 行 (period={period})")
        return len(df)

    def download_cashflow_vip(self, period: str) -> int:
        """
        下载现金流量表数据（VIP接口，按报告期批量获取全部股票）

        数据说明：
        - 获取某一季度全部上市公司现金流量表数据
        - 单次最大获取5000条数据
        - 需要至少5000积分

        Args:
            period: 报告期 YYYYMMDD，如 20231231（年报）、20230630（半年报）

        Returns:
            下载的行数
        """
        logger.debug(f"下载现金流量表(VIP): period={period}")

        df = self.fetcher.fetch('cashflow_vip', period=period)

        if df.empty:
            logger.debug(f"无现金流量表数据: period={period}")
            return 0

        self.db.write_dataframe(df, 'cashflow', mode='append')
        logger.info(f"现金流量表数据(VIP): {len(df)} 行 (period={period})")
        return len(df)

    def download_income(
        self,
        ts_code: Optional[str] = None,
        ann_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: Optional[str] = None,
        report_type: Optional[str] = None
    ) -> int:
        """
        下载利润表数据（普通接口，按单只股票获取）

        数据说明：
        - 获取单只股票的利润表历史数据
        - 单次最大获取5000条数据
        - 需要至少2000积分
        - 如需批量获取全部股票，请使用 download_income_vip

        Args:
            ts_code: 股票代码（必填参数）
            ann_date: 公告日期 YYYYMMDD（可选）
            start_date: 公告开始日期（可选）
            end_date: 公告结束日期（可选）
            period: 报告期 YYYYMMDD，如 20231231（可选）
            report_type: 报告类型（1合并报表 2单季合并 等）

        Returns:
            下载的行数
        """
        logger.debug(f"下载利润表: ts_code={ts_code}, period={period}, "
                    f"ann_date={ann_date}, start_date={start_date}, end_date={end_date}")

        df = self.fetcher.fetch(
            'income',
            ts_code=ts_code,
            ann_date=ann_date,
            start_date=start_date,
            end_date=end_date,
            period=period,
            report_type=report_type
        )

        if df.empty:
            logger.debug(f"无利润表数据")
            return 0

        self.db.write_dataframe(df, 'income', mode='append')
        logger.info(f"利润表数据: {len(df)} 行")
        return len(df)

    def download_balancesheet(
        self,
        ts_code: Optional[str] = None,
        ann_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: Optional[str] = None,
        report_type: Optional[str] = None
    ) -> int:
        """
        下载资产负债表数据（普通接口，按单只股票获取）

        数据说明：
        - 获取单只股票的资产负债表历史数据
        - 单次最大获取5000条数据
        - 需要至少2000积分
        - 如需批量获取全部股票，请使用 download_balancesheet_vip

        Args:
            ts_code: 股票代码（必填参数）
            ann_date: 公告日期 YYYYMMDD（可选）
            start_date: 公告开始日期（可选）
            end_date: 公告结束日期（可选）
            period: 报告期 YYYYMMDD，如 20231231（可选）
            report_type: 报告类型（1合并报表 2单季合并 等）

        Returns:
            下载的行数
        """
        logger.debug(f"下载资产负债表: ts_code={ts_code}, period={period}, "
                    f"ann_date={ann_date}, start_date={start_date}, end_date={end_date}")

        df = self.fetcher.fetch(
            'balancesheet',
            ts_code=ts_code,
            ann_date=ann_date,
            start_date=start_date,
            end_date=end_date,
            period=period,
            report_type=report_type
        )

        if df.empty:
            logger.debug(f"无资产负债表数据")
            return 0

        self.db.write_dataframe(df, 'balancesheet', mode='append')
        logger.info(f"资产负债表数据: {len(df)} 行")
        return len(df)

    def download_cashflow(
        self,
        ts_code: Optional[str] = None,
        ann_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: Optional[str] = None,
        report_type: Optional[str] = None
    ) -> int:
        """
        下载现金流量表数据（普通接口，按单只股票获取）

        数据说明：
        - 获取单只股票的现金流量表历史数据
        - 单次最大获取5000条数据
        - 需要至少2000积分
        - 如需批量获取全部股票，请使用 download_cashflow_vip

        Args:
            ts_code: 股票代码（必填参数）
            ann_date: 公告日期 YYYYMMDD（可选）
            start_date: 公告开始日期（可选）
            end_date: 公告结束日期（可选）
            period: 报告期 YYYYMMDD，如 20231231（可选）
            report_type: 报告类型（1合并报表 2单季合并 等）

        Returns:
            下载的行数
        """
        logger.debug(f"下载现金流量表: ts_code={ts_code}, period={period}, "
                    f"ann_date={ann_date}, start_date={start_date}, end_date={end_date}")

        df = self.fetcher.fetch(
            'cashflow',
            ts_code=ts_code,
            ann_date=ann_date,
            start_date=start_date,
            end_date=end_date,
            period=period,
            report_type=report_type
        )

        if df.empty:
            logger.debug(f"无现金流量表数据")
            return 0

        self.db.write_dataframe(df, 'cashflow', mode='append')
        logger.info(f"现金流量表数据: {len(df)} 行")
        return len(df)

    def download_dividend(
        self,
        ts_code: Optional[str] = None,
        ann_date: Optional[str] = None,
        record_date: Optional[str] = None,
        ex_date: Optional[str] = None,
        imp_ann_date: Optional[str] = None
    ) -> int:
        """
        下载分红送股数据

        数据说明：
        - 获取上市公司分红送股数据
        - 包括现金分红、送股、转增等
        - 需要至少120积分

        Args:
            ts_code: 股票代码（可选）
            ann_date: 公告日期 YYYYMMDD（可选）
            record_date: 股权登记日期（可选）
            ex_date: 除权除息日（可选）
            imp_ann_date: 实施公告日（可选）

        Returns:
            下载的行数
        """
        logger.debug(f"下载分红送股: ts_code={ts_code}, ann_date={ann_date}, "
                    f"record_date={record_date}, ex_date={ex_date}")

        df = self.fetcher.fetch(
            'dividend',
            ts_code=ts_code,
            ann_date=ann_date,
            record_date=record_date,
            ex_date=ex_date,
            imp_ann_date=imp_ann_date
        )

        if df.empty:
            logger.debug(f"无分红送股数据")
            return 0

        self.db.write_dataframe(df, 'dividend', mode='append')
        logger.info(f"分红送股数据: {len(df)} 行")
        return len(df)

    def download_margin_detail(
        self,
        trade_date: Optional[str] = None,
        ts_code: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> int:
        """
        下载融资融券交易明细数据

        数据说明：
        - 获取沪深两市每日融资融券明细
        - 单次最大获取6000条数据
        - 需要至少2000积分

        Args:
            trade_date: 交易日期 YYYYMMDD（可选）
            ts_code: 股票代码（可选）
            start_date: 开始日期（可选）
            end_date: 结束日期（可选）

        Returns:
            下载的行数
        """
        logger.debug(f"下载融资融券明细: trade_date={trade_date}, ts_code={ts_code}, "
                    f"start_date={start_date}, end_date={end_date}")

        df = self.fetcher.fetch(
            'margin_detail',
            trade_date=trade_date,
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date
        )

        if df.empty:
            logger.debug(f"无融资融券明细数据")
            return 0

        self.db.write_dataframe(df, 'margin_detail', mode='append')
        logger.info(f"融资融券明细数据: {len(df)} 行")
        return len(df)

    # ==================== 申万行业指数 ====================

    def download_sw_daily(
        self,
        ts_code: Optional[str] = None,
        trade_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> int:
        """
        下载申万行业指数日线行情

        数据说明：
        - 获取申万行业指数日线行情（申万2021版）
        - 单次最大获取4000条数据
        - 需要5000积分

        字段说明：
        - ts_code: 指数代码（如 801010.SI）
        - trade_date: 交易日期
        - name: 指数名称
        - open/high/low/close: OHLC数据
        - pct_change: 涨跌幅
        - vol: 成交量（万股）
        - amount: 成交额（万元）
        - pe: 市盈率
        - pb: 市净率
        - float_mv: 流通市值（万元）
        - total_mv: 总市值（万元）

        Args:
            ts_code: 行业指数代码（可选）
            trade_date: 交易日期 YYYYMMDD（可选）
            start_date: 开始日期（可选）
            end_date: 结束日期（可选）

        Returns:
            下载的行数
        """
        logger.debug(f"下载申万行业日线: ts_code={ts_code}, trade_date={trade_date}, "
                    f"start_date={start_date}, end_date={end_date}")

        df = self.fetcher.fetch(
            'sw_daily',
            ts_code=ts_code,
            trade_date=trade_date,
            start_date=start_date,
            end_date=end_date
        )

        if df.empty:
            logger.debug(f"无申万行业日线数据")
            return 0

        self.db.write_dataframe(df, 'sw_daily', mode='append')
        logger.info(f"申万行业日线数据: {len(df)} 行")
        return len(df)

    def download_sw_daily_by_date_range(
        self,
        start_date: str,
        end_date: str,
        ts_code: Optional[str] = None
    ) -> int:
        """
        按日期范围下载申万行业指数日线（适合大量历史数据）

        由于单次最大4000条限制，按日期逐日下载

        Args:
            start_date: 开始日期 YYYYMMDD
            end_date: 结束日期 YYYYMMDD
            ts_code: 指定行业代码（可选，不指定则下载所有行业）

        Returns:
            下载的总行数
        """
        logger.info(f"批量下载申万行业日线: {start_date} -> {end_date}")

        # 获取交易日列表
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
            logger.warning(f"无交易日数据，请先下载交易日历")
            return 0

        trading_dates = trading_dates_df['cal_date'].tolist()
        total_rows = 0

        for trade_date in tqdm(trading_dates, desc="下载申万行业日线"):
            rows = self.download_sw_daily(ts_code=ts_code, trade_date=trade_date)
            total_rows += rows

        logger.info(f"申万行业日线下载完成: 共 {total_rows} 行")
        return total_rows

    # ==================== 指数日线数据 ====================

    def download_index_daily(
        self,
        ts_code: Optional[str] = None,
        trade_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> int:
        """
        下载指数日线行情

        数据说明：
        - 获取指数每日行情（上证指数、深证成指、创业板指等）
        - 单次最大获取8000条数据

        字段说明：
        - ts_code: 指数代码（如 000001.SH 上证指数）
        - trade_date: 交易日期
        - open/high/low/close: OHLC数据
        - pre_close: 昨收盘
        - change: 涨跌额
        - pct_chg: 涨跌幅（%）
        - vol: 成交量（手）
        - amount: 成交额（千元）

        常用指数代码：
        - 000001.SH: 上证指数
        - 399001.SZ: 深证成指
        - 399006.SZ: 创业板指
        - 000300.SH: 沪深300
        - 000905.SH: 中证500
        - 000852.SH: 中证1000

        Args:
            ts_code: 指数代码（可选）
            trade_date: 交易日期 YYYYMMDD（可选）
            start_date: 开始日期（可选）
            end_date: 结束日期（可选）

        Returns:
            下载的行数
        """
        logger.debug(f"下载指数日线: ts_code={ts_code}, trade_date={trade_date}, "
                    f"start_date={start_date}, end_date={end_date}")

        df = self.fetcher.fetch(
            'index_daily',
            ts_code=ts_code,
            trade_date=trade_date,
            start_date=start_date,
            end_date=end_date
        )

        if df.empty:
            logger.debug(f"无指数日线数据")
            return 0

        self.db.write_dataframe(df, 'index_daily', mode='append')
        logger.info(f"指数日线数据: {len(df)} 行")
        return len(df)

    def download_index_daily_by_date_range(
        self,
        ts_code: str,
        start_date: str,
        end_date: str
    ) -> int:
        """
        按日期范围下载指数日线（适合大量历史数据）

        Args:
            ts_code: 指数代码（如 000001.SH）
            start_date: 开始日期 YYYYMMDD
            end_date: 结束日期 YYYYMMDD

        Returns:
            下载的总行数
        """
        logger.info(f"下载指数日线: {ts_code} {start_date} -> {end_date}")

        # 直接使用 start_date/end_date 下载，单次最大8000条
        df = self.fetcher.fetch(
            'index_daily',
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date
        )

        if df.empty:
            logger.debug(f"无指数日线数据: {ts_code}")
            return 0

        self.db.write_dataframe(df, 'index_daily', mode='append')
        logger.info(f"指数日线数据: {len(df)} 行 ({ts_code})")
        return len(df)

    # ==================== 开盘啦题材库 ====================

    def download_kpl_concept(self, trade_date: str) -> int:
        """
        下载开盘啦题材列表数据（按日期）

        Args:
            trade_date: 交易日期 (YYYYMMDD)

        Returns:
            下载的行数

        说明:
            - 包含题材代码、名称、涨停数量、排名变化
            - 每日盘后更新
            - 需要 5000+ 积分
        """
        logger.debug(f"下载开盘啦题材列表: {trade_date}")
        df = self.fetcher.fetch(
            'kpl_concept',
            trade_date=trade_date
        )

        if df.empty:
            logger.debug(f"无开盘啦题材数据: {trade_date}")
            return 0

        self.db.write_dataframe(df, 'kpl_concept', mode='append')
        logger.info(f"开盘啦题材列表: {len(df)} 行 ({trade_date})")
        return len(df)

    def download_kpl_concept_batch(self, target_date: Optional[str] = None) -> int:
        """
        批量下载开盘啦题材列表数据（使用 offset 分页）

        Args:
            target_date: 目标日期，只保留该日期之后的数据（可选）
                        如果指定，会过滤掉早于该日期的数据

        Returns:
            下载的行数

        说明:
            - 使用 limit/offset 分页批量获取
            - 单次最大 5000 条
            - 比逐日下载效率高很多
        """
        logger.info(f"批量下载开盘啦题材列表 (目标日期: {target_date or '全部'})...")

        total_rows = 0
        offset = 0
        limit = 5000

        while True:
            df = self.fetcher.fetch(
                'kpl_concept',
                limit=limit,
                offset=offset
            )

            if df.empty:
                break

            # 如果指定了目标日期，过滤掉早于该日期的数据
            if target_date and 'trade_date' in df.columns:
                df = df[df['trade_date'] >= target_date]

            if df.empty:
                # 如果过滤后没有数据，说明已经获取到目标日期之前的数据了
                break

            self.db.write_dataframe(df, 'kpl_concept', mode='append')
            total_rows += len(df)

            logger.info(f"  offset={offset}: {len(df)} 行, 日期范围: {df['trade_date'].min()} ~ {df['trade_date'].max()}")

            # 如果返回的数据少于 limit，说明没有更多数据了
            if len(df) < limit:
                break

            offset += limit

        logger.info(f"开盘啦题材列表批量下载完成: 共 {total_rows} 行")
        return total_rows

    def download_kpl_concept_cons(self, trade_date: str) -> int:
        """
        下载开盘啦题材成分股数据（按日期）

        Args:
            trade_date: 交易日期 (YYYYMMDD)

        Returns:
            下载的行数

        说明:
            - 包含题材与股票的关联关系
            - 包含股票在该题材中的描述和人气值
            - 每日盘后更新
            - 需要 5000+ 积分
        """
        logger.debug(f"下载开盘啦题材成分: {trade_date}")
        df = self.fetcher.fetch(
            'kpl_concept_cons',
            trade_date=trade_date
        )

        if df.empty:
            logger.debug(f"无开盘啦题材成分数据: {trade_date}")
            return 0

        self.db.write_dataframe(df, 'kpl_concept_cons', mode='append')
        logger.info(f"开盘啦题材成分: {len(df)} 行 ({trade_date})")
        return len(df)

    def download_kpl_concept_cons_batch(self, target_date: Optional[str] = None) -> int:
        """
        批量下载开盘啦题材成分股数据（使用 offset 分页）

        Args:
            target_date: 目标日期，只保留该日期之后的数据（可选）
                        如果指定，会过滤掉早于该日期的数据

        Returns:
            下载的行数

        说明:
            - 使用 limit/offset 分页批量获取
            - 单次最大 3000 条
            - 比逐日下载效率高很多
        """
        logger.info(f"批量下载开盘啦题材成分 (目标日期: {target_date or '全部'})...")

        total_rows = 0
        offset = 0
        limit = 3000

        while True:
            df = self.fetcher.fetch(
                'kpl_concept_cons',
                limit=limit,
                offset=offset
            )

            if df.empty:
                break

            # 如果指定了目标日期，过滤掉早于该日期的数据
            if target_date and 'trade_date' in df.columns:
                df = df[df['trade_date'] >= target_date]

            if df.empty:
                # 如果过滤后没有数据，说明已经获取到目标日期之前的数据了
                break

            self.db.write_dataframe(df, 'kpl_concept_cons', mode='append')
            total_rows += len(df)

            logger.info(f"  offset={offset}: {len(df)} 行, 日期范围: {df['trade_date'].min()} ~ {df['trade_date'].max()}")

            # 如果返回的数据少于 limit，说明没有更多数据了
            if len(df) < limit:
                break

            offset += limit

        logger.info(f"开盘啦题材成分批量下载完成: 共 {total_rows} 行")
        return total_rows

    # ==================== 同花顺板块 ====================

    def download_ths_index(
        self,
        ts_code: Optional[str] = None,
        exchange: Optional[str] = None,
        type: Optional[str] = None
    ) -> int:
        """
        下载同花顺板块指数信息

        Args:
            ts_code: 指数代码（可选）
            exchange: 交易所代码（可选）
            type: 指数类型（可选）
                - N: 概念板块
                - I: 行业板块
                - R: 地域板块
                - S: 特色指数
                - ST: 风格指数
                - TH: 同花顺特色
                - BB: 宽基指数

        Returns:
            下载的行数

        说明:
            - 不指定参数时下载所有板块信息
            - 本项目默认只下载 I(行业)、N(概念)、R(地域)、BB(宽基) 四类
            - 需要 6000 积分
        """
        logger.debug(f"下载同花顺板块指数: ts_code={ts_code}, type={type}")
        df = self.fetcher.fetch(
            'ths_index',
            ts_code=ts_code,
            exchange=exchange,
            type=type
        )

        if df.empty:
            logger.debug(f"无同花顺板块数据")
            return 0

        self.db.write_dataframe(df, 'ths_index', mode='append')
        logger.info(f"同花顺板块指数: {len(df)} 行")
        return len(df)

    def download_ths_member(
        self,
        ts_code: Optional[str] = None,
        code: Optional[str] = None
    ) -> int:
        """
        下载同花顺板块成分股数据

        Args:
            ts_code: 板块指数代码（可选）
            code: 股票代码（可选，查询股票所属板块）

        Returns:
            下载的行数

        说明:
            - ts_code 和 code 必须输入一个
            - ts_code 获取该板块的所有成分股
            - code 获取该股票所属的所有板块
            - 需要 6000 积分
        """
        logger.debug(f"下载同花顺板块成分: ts_code={ts_code}, code={code}")
        df = self.fetcher.fetch(
            'ths_member',
            ts_code=ts_code,
            code=code
        )

        if df.empty:
            logger.debug(f"无同花顺板块成分数据")
            return 0

        self.db.write_dataframe(df, 'ths_member', mode='append')
        logger.info(f"同花顺板块成分: {len(df)} 行")
        return len(df)

    def download_ths_daily(
        self,
        ts_code: Optional[str] = None,
        trade_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> int:
        """
        下载同花顺板块指数日行情数据

        Args:
            ts_code: 板块指数代码（可选）
            trade_date: 交易日期 YYYYMMDD（可选）
            start_date: 开始日期（可选）
            end_date: 结束日期（可选）

        Returns:
            下载的行数

        说明:
            - 数据从 20180102 开始
            - 单次最大 3000 条
            - 需要 6000 积分

        字段说明:
            - ts_code: 板块代码
            - trade_date: 交易日期
            - open/high/low/close: OHLC
            - pct_change: 涨跌幅
            - vol: 成交量
            - turnover_rate: 换手率
            - pe/pb: 市盈率/市净率
            - total_mv/float_mv: 总市值/流通市值
        """
        logger.debug(f"下载同花顺板块日行情: ts_code={ts_code}, trade_date={trade_date}, "
                    f"start_date={start_date}, end_date={end_date}")
        df = self.fetcher.fetch(
            'ths_daily',
            ts_code=ts_code,
            trade_date=trade_date,
            start_date=start_date,
            end_date=end_date
        )

        if df.empty:
            logger.debug(f"无同花顺板块日行情数据")
            return 0

        self.db.write_dataframe(df, 'ths_daily', mode='append')
        logger.info(f"同花顺板块日行情: {len(df)} 行")
        return len(df)

    # ==================== 基金数据 ====================

    def download_fund_basic(
        self,
        market: Optional[str] = None,
        status: Optional[str] = None
    ) -> int:
        """
        下载基金列表

        数据说明：
        - 获取公募基金基础信息
        - 需要至少2000积分
        - 单次最大15000条

        Args:
            market: 交易市场 E=场内 O=场外（默认E）
            status: 存续状态 D=摘牌 I=发行中 L=已上市

        Returns:
            下载的行数
        """
        logger.info(f"开始下载基金列表: market={market}, status={status}")
        df = self.fetcher.fetch(
            'fund_basic',
            market=market,
            status=status
        )

        if df.empty:
            logger.warning(f"基金列表为空: market={market}, status={status}")
            return 0

        self.db.write_dataframe(df, 'fund_basic', mode='append')
        logger.info(f"基金列表下载完成: {len(df)} 行")
        return len(df)

    def download_fund_daily(
        self,
        ts_code: Optional[str] = None,
        trade_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> int:
        """
        下载场内基金日线行情

        数据说明：
        - 获取场内基金（ETF/LOF等）日线行情数据
        - 需要至少5000积分
        - 单次最大2000条

        Args:
            ts_code: 基金代码（可选）
            trade_date: 交易日期 YYYYMMDD（可选）
            start_date: 开始日期（可选）
            end_date: 结束日期（可选）

        Returns:
            下载的行数
        """
        logger.debug(f"下载场内基金日线: ts_code={ts_code}, trade_date={trade_date}, "
                    f"start_date={start_date}, end_date={end_date}")
        df = self.fetcher.fetch(
            'fund_daily',
            ts_code=ts_code,
            trade_date=trade_date,
            start_date=start_date,
            end_date=end_date
        )

        if df.empty:
            logger.debug(f"无场内基金日线数据")
            return 0

        self.db.write_dataframe(df, 'fund_daily', mode='append')
        logger.info(f"场内基金日线: {len(df)} 行")
        return len(df)

    def download_fund_nav(
        self,
        ts_code: Optional[str] = None,
        nav_date: Optional[str] = None,
        market: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> int:
        """
        下载基金净值数据

        数据说明：
        - 获取公募基金净值数据
        - ts_code 和 nav_date 至少输入一个
        - 需要至少2000积分

        字段说明：
        - unit_nav: 单位净值
        - accum_nav: 累计净值
        - accum_div: 累计分红
        - net_asset: 资产净值
        - total_netasset: 合计资产净值
        - adj_nav: 复权净值

        Args:
            ts_code: 基金代码（可选）
            nav_date: 净值日期 YYYYMMDD（可选）
            market: E=场内 O=场外（可选）
            start_date: 开始日期（可选）
            end_date: 结束日期（可选）

        Returns:
            下载的行数
        """
        logger.debug(f"下载基金净值: ts_code={ts_code}, nav_date={nav_date}, "
                    f"market={market}, start_date={start_date}, end_date={end_date}")
        df = self.fetcher.fetch(
            'fund_nav',
            ts_code=ts_code,
            nav_date=nav_date,
            market=market,
            start_date=start_date,
            end_date=end_date
        )

        if df.empty:
            logger.debug(f"无基金净值数据")
            return 0

        self.db.write_dataframe(df, 'fund_nav', mode='append')
        logger.info(f"基金净值数据: {len(df)} 行")
        return len(df)

    def download_fund_div(
        self,
        ts_code: Optional[str] = None,
        ann_date: Optional[str] = None,
        ex_date: Optional[str] = None,
        pay_date: Optional[str] = None
    ) -> int:
        """
        下载基金分红数据

        数据说明：
        - 获取公募基金分红数据
        - ts_code/ann_date/ex_date/pay_date 至少输入一个
        - 需要至少400积分

        Args:
            ts_code: 基金代码（可选）
            ann_date: 公告日期 YYYYMMDD（可选）
            ex_date: 除息日 YYYYMMDD（可选）
            pay_date: 派息日 YYYYMMDD（可选）

        Returns:
            下载的行数
        """
        logger.debug(f"下载基金分红: ts_code={ts_code}, ann_date={ann_date}")
        df = self.fetcher.fetch(
            'fund_div',
            ts_code=ts_code,
            ann_date=ann_date,
            ex_date=ex_date,
            pay_date=pay_date
        )

        if df.empty:
            logger.debug(f"无基金分红数据")
            return 0

        self.db.write_dataframe(df, 'fund_div', mode='append')
        logger.info(f"基金分红数据: {len(df)} 行")
        return len(df)

    def download_fund_portfolio(
        self,
        ts_code: Optional[str] = None,
        ann_date: Optional[str] = None,
        period: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> int:
        """
        下载基金持仓数据

        数据说明：
        - 获取公募基金持仓数据（十大重仓股）
        - ts_code/ann_date/period 至少输入一个
        - 需要至少5000积分

        Args:
            ts_code: 基金代码（可选）
            ann_date: 公告日期 YYYYMMDD（可选）
            period: 报告期 YYYYMMDD，如 20231231（可选）
            start_date: 报告期开始日期（可选）
            end_date: 报告期结束日期（可选）

        Returns:
            下载的行数
        """
        logger.debug(f"下载基金持仓: ts_code={ts_code}, ann_date={ann_date}, period={period}")
        df = self.fetcher.fetch(
            'fund_portfolio',
            ts_code=ts_code,
            ann_date=ann_date,
            period=period,
            start_date=start_date,
            end_date=end_date
        )

        if df.empty:
            logger.debug(f"无基金持仓数据")
            return 0

        self.db.write_dataframe(df, 'fund_portfolio', mode='append')
        logger.info(f"基金持仓数据: {len(df)} 行")
        return len(df)

    def download_fund_share(
        self,
        ts_code: Optional[str] = None,
        trade_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        market: Optional[str] = None
    ) -> int:
        """
        下载基金份额数据

        数据说明：
        - 获取基金份额变动数据
        - 需要至少2000积分
        - 单次最大2000条

        Args:
            ts_code: 基金代码（可选）
            trade_date: 交易日期 YYYYMMDD（可选）
            start_date: 开始日期（可选）
            end_date: 结束日期（可选）
            market: 市场 SH=上交所 SZ=深交所（可选）

        Returns:
            下载的行数
        """
        logger.debug(f"下载基金份额: ts_code={ts_code}, trade_date={trade_date}")
        df = self.fetcher.fetch(
            'fund_share',
            ts_code=ts_code,
            trade_date=trade_date,
            start_date=start_date,
            end_date=end_date,
            market=market
        )

        if df.empty:
            logger.debug(f"无基金份额数据")
            return 0

        self.db.write_dataframe(df, 'fund_share', mode='append')
        logger.info(f"基金份额数据: {len(df)} 行")
        return len(df)

    def download_fund_manager(
        self,
        ts_code: Optional[str] = None,
        ann_date: Optional[str] = None,
        name: Optional[str] = None
    ) -> int:
        """
        下载基金经理数据

        数据说明：
        - 获取基金经理信息及任职情况
        - 需要至少500积分
        - 单次最大5000条

        Args:
            ts_code: 基金代码（可选，支持逗号分隔多个）
            ann_date: 公告日期 YYYYMMDD（可选）
            name: 基金经理名称（可选）

        Returns:
            下载的行数
        """
        logger.debug(f"下载基金经理: ts_code={ts_code}, ann_date={ann_date}, name={name}")
        df = self.fetcher.fetch(
            'fund_manager',
            ts_code=ts_code,
            ann_date=ann_date,
            name=name
        )

        if df.empty:
            logger.debug(f"无基金经理数据")
            return 0

        self.db.write_dataframe(df, 'fund_manager', mode='append')
        logger.info(f"基金经理数据: {len(df)} 行")
        return len(df)

    def download_fund_adj(
        self,
        ts_code: Optional[str] = None,
        trade_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> int:
        """
        下载基金复权因子

        数据说明：
        - 获取基金复权因子数据
        - 需要至少600积分
        - 单次最大2000条

        Args:
            ts_code: 基金代码（可选，支持多个）
            trade_date: 交易日期 YYYYMMDD（可选）
            start_date: 开始日期（可选）
            end_date: 结束日期（可选）

        Returns:
            下载的行数
        """
        logger.debug(f"下载基金复权因子: ts_code={ts_code}, trade_date={trade_date}")
        df = self.fetcher.fetch(
            'fund_adj',
            ts_code=ts_code,
            trade_date=trade_date,
            start_date=start_date,
            end_date=end_date
        )

        if df.empty:
            logger.debug(f"无基金复权因子数据")
            return 0

        self.db.write_dataframe(df, 'fund_adj', mode='append')
        logger.info(f"基金复权因子: {len(df)} 行")
        return len(df)

    def download_all_fund_daily(
        self,
        start_date: str,
        end_date: Optional[str] = None
    ) -> int:
        """
        按日期范围批量下载场内基金日线（逐日下载，因为单次限2000条）

        Args:
            start_date: 开始日期 YYYYMMDD
            end_date: 结束日期 YYYYMMDD

        Returns:
            下载的总行数
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')

        logger.info(f"批量下载场内基金日线: {start_date} -> {end_date}")

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
            logger.warning("无交易日数据，请先下载交易日历")
            return 0

        trading_dates = trading_dates_df['cal_date'].tolist()
        total_rows = 0

        for trade_date in tqdm(trading_dates, desc="下载场内基金日线"):
            rows = self.download_fund_daily(trade_date=trade_date)
            total_rows += rows

        logger.info(f"场内基金日线下载完成: 共 {total_rows} 行")
        return total_rows

    def download_all_fund_nav(
        self,
        start_date: str,
        end_date: Optional[str] = None,
        market: Optional[str] = None
    ) -> int:
        """
        按日期范围批量下载基金净值（逐日下载）

        Args:
            start_date: 开始日期 YYYYMMDD
            end_date: 结束日期 YYYYMMDD
            market: E=场内 O=场外（可选）

        Returns:
            下载的总行数
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')

        logger.info(f"批量下载基金净值: {start_date} -> {end_date}, market={market}")

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
            logger.warning("无交易日数据，请先下载交易日历")
            return 0

        trading_dates = trading_dates_df['cal_date'].tolist()
        total_rows = 0

        for nav_date in tqdm(trading_dates, desc="下载基金净值"):
            rows = self.download_fund_nav(nav_date=nav_date, market=market)
            total_rows += rows

        logger.info(f"基金净值下载完成: 共 {total_rows} 行")
        return total_rows

    def download_all_fund_portfolio(self, period: str) -> int:
        """
        按报告期批量下载所有基金持仓（使用 offset 分页）

        Args:
            period: 报告期 YYYYMMDD，如 20231231（年报）、20230630（半年报）

        Returns:
            下载的总行数
        """
        logger.info(f"批量下载基金持仓: period={period}")

        total_rows = 0
        offset = 0
        limit = 5000

        while True:
            df = self.fetcher.fetch(
                'fund_portfolio',
                period=period,
                offset=offset,
                limit=limit
            )

            if df.empty:
                break

            self.db.write_dataframe(df, 'fund_portfolio', mode='append')
            total_rows += len(df)

            logger.info(f"  offset={offset}: {len(df)} 行")

            if len(df) < limit:
                break

            offset += limit

        logger.info(f"基金持仓下载完成: 共 {total_rows} 行 (period={period})")
        return total_rows

    # ==================== 沪深港通 ====================

    def download_moneyflow_hsgt(
        self,
        trade_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> int:
        """
        下载沪深港通资金流向

        数据说明：
        - 获取沪深港通每日资金流向
        - 每日18:00~20:00更新
        - 需要至少2000积分
        - 单次最大300条

        字段说明：
        - ggt_ss: 港股通（沪）
        - ggt_sz: 港股通（深）
        - hgt: 沪股通（百万元）
        - sgt: 深股通（百万元）
        - north_money: 北向资金（百万元）
        - south_money: 南向资金（百万元）

        Args:
            trade_date: 交易日期 YYYYMMDD（可选）
            start_date: 开始日期（可选）
            end_date: 结束日期（可选）

        Returns:
            下载的行数
        """
        logger.debug(f"下载沪深港通资金流向: trade_date={trade_date}, "
                    f"start_date={start_date}, end_date={end_date}")
        df = self.fetcher.fetch(
            'moneyflow_hsgt',
            trade_date=trade_date,
            start_date=start_date,
            end_date=end_date
        )

        if df.empty:
            logger.debug(f"无沪深港通资金流向数据")
            return 0

        self.db.write_dataframe(df, 'moneyflow_hsgt', mode='append')
        logger.info(f"沪深港通资金流向: {len(df)} 行")
        return len(df)

    def download_hsgt_top10(
        self,
        ts_code: Optional[str] = None,
        trade_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        market_type: Optional[str] = None
    ) -> int:
        """
        下载沪深股通十大成交股

        数据说明：
        - 获取沪深股通每日十大成交个股数据
        - 每日18:00~20:00更新
        - ts_code 和 trade_date 至少输入一个

        Args:
            ts_code: 股票代码（可选）
            trade_date: 交易日期 YYYYMMDD（可选）
            start_date: 开始日期（可选）
            end_date: 结束日期（可选）
            market_type: 市场类型 1=沪股通 3=深股通（可选）

        Returns:
            下载的行数
        """
        logger.debug(f"下载沪深股通十大成交: trade_date={trade_date}, ts_code={ts_code}")
        df = self.fetcher.fetch(
            'hsgt_top10',
            ts_code=ts_code,
            trade_date=trade_date,
            start_date=start_date,
            end_date=end_date,
            market_type=market_type
        )

        if df.empty:
            logger.debug(f"无沪深股通十大成交数据")
            return 0

        self.db.write_dataframe(df, 'hsgt_top10', mode='append')
        logger.info(f"沪深股通十大成交: {len(df)} 行")
        return len(df)

    def download_ggt_top10(
        self,
        ts_code: Optional[str] = None,
        trade_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        market_type: Optional[str] = None
    ) -> int:
        """
        下载港股通十大成交股

        数据说明：
        - 获取港股通每日十大成交个股数据
        - 每日18:00~20:00更新
        - ts_code 和 trade_date 至少输入一个

        Args:
            ts_code: 股票代码（可选）
            trade_date: 交易日期 YYYYMMDD（可选）
            start_date: 开始日期（可选）
            end_date: 结束日期（可选）
            market_type: 市场类型 2=港股通(沪) 4=港股通(深)（可选）

        Returns:
            下载的行数
        """
        logger.debug(f"下载港股通十大成交: trade_date={trade_date}, ts_code={ts_code}")
        df = self.fetcher.fetch(
            'ggt_top10',
            ts_code=ts_code,
            trade_date=trade_date,
            start_date=start_date,
            end_date=end_date,
            market_type=market_type
        )

        if df.empty:
            logger.debug(f"无港股通十大成交数据")
            return 0

        self.db.write_dataframe(df, 'ggt_top10', mode='append')
        logger.info(f"港股通十大成交: {len(df)} 行")
        return len(df)

    def download_ggt_daily(
        self,
        trade_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> int:
        """
        下载港股通每日成交统计

        数据说明：
        - 获取港股通每日成交汇总数据
        - 数据从2014年开始
        - 需要至少2000积分
        - 单次最大1000条

        字段说明：
        - buy_amount: 买入成交金额（亿元）
        - buy_volume: 买入成交笔数（万笔）
        - sell_amount: 卖出成交金额（亿元）
        - sell_volume: 卖出成交笔数（万笔）

        Args:
            trade_date: 交易日期 YYYYMMDD（可选）
            start_date: 开始日期（可选）
            end_date: 结束日期（可选）

        Returns:
            下载的行数
        """
        logger.debug(f"下载港股通每日成交: trade_date={trade_date}, "
                    f"start_date={start_date}, end_date={end_date}")
        df = self.fetcher.fetch(
            'ggt_daily',
            trade_date=trade_date,
            start_date=start_date,
            end_date=end_date
        )

        if df.empty:
            logger.debug(f"无港股通每日成交数据")
            return 0

        self.db.write_dataframe(df, 'ggt_daily', mode='append')
        logger.info(f"港股通每日成交: {len(df)} 行")
        return len(df)

    def download_hk_hold(
        self,
        code: Optional[str] = None,
        ts_code: Optional[str] = None,
        trade_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        exchange: Optional[str] = None
    ) -> int:
        """
        下载沪深港通持股明细

        数据说明：
        - 获取沪深港通持股明细数据
        - 需要至少2000积分
        - 单次最大3800条

        Args:
            code: 交易所代码（可选）
            ts_code: TS股票代码（可选）
            trade_date: 交易日期 YYYYMMDD（可选）
            start_date: 开始日期（可选）
            end_date: 结束日期（可选）
            exchange: 类型 SH=沪股通(北向) SZ=深股通(北向) HK=港股通(南向)（可选）

        Returns:
            下载的行数
        """
        logger.debug(f"下载沪深港通持股明细: trade_date={trade_date}, ts_code={ts_code}, "
                    f"exchange={exchange}")
        df = self.fetcher.fetch(
            'hk_hold',
            code=code,
            ts_code=ts_code,
            trade_date=trade_date,
            start_date=start_date,
            end_date=end_date,
            exchange=exchange
        )

        if df.empty:
            logger.debug(f"无沪深港通持股明细数据")
            return 0

        self.db.write_dataframe(df, 'hk_hold', mode='append')
        logger.info(f"沪深港通持股明细: {len(df)} 行")
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
                SELECT trade_date FROM daily
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
