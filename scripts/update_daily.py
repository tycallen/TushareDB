#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
每日数据更新脚本（新架构版本）

功能：
1. 更新交易日历
2. 更新股票列表
3. 更新所有股票的日线数据、复权因子、每日基本面
4. 更新财务指标数据（VIP接口）
5. 更新筹码分布数据
6. 更新龙虎榜机构席位数据
7. 更新行业资金流向（沪深通）数据

使用方法：
    python scripts/update_daily.py

环境变量：
    TUSHARE_TOKEN: Tushare API Token（必须）
    DB_PATH: 数据库路径（可选，默认为 tushare.db）

作者：根据新架构重构
日期：2025-12-04
更新：2025-12-11 - 添加龙虎榜和资金流向数据更新
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tushare_db import DataDownloader, get_logger

# 配置日志
logger = get_logger(__name__)


def get_env_config():
    """获取环境变量配置"""
    tushare_token = os.getenv("TUSHARE_TOKEN")
    if not tushare_token:
        raise ValueError("请设置 TUSHARE_TOKEN 环境变量")
    
    db_path = os.getenv("DB_PATH", str(project_root / "tushare.db"))
    
    return tushare_token, db_path


def update_trade_calendar(downloader: DataDownloader):
    """
    更新交易日历
    
    策略：每次更新最近30天到未来1年的数据，确保覆盖节假日调整
    """
    logger.info("=" * 60)
    logger.info("开始更新交易日历...")
    
    thirty_days_ago = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
    one_year_later = (datetime.now() + timedelta(days=365)).strftime('%Y%m%d')
    
    try:
        rows = downloader.download_trade_calendar(
            start_date=thirty_days_ago,
            end_date=one_year_later
        )
        logger.info(f"✓ 交易日历更新完成，更新 {rows} 行")
    except Exception as e:
        logger.error(f"✗ 更新交易日历失败: {e}")
        raise


def update_stock_basic(downloader: DataDownloader):
    """
    更新股票基础信息
    
    策略：每次全量更新所有状态的股票（上市、退市、暂停）
    """
    logger.info("=" * 60)
    logger.info("开始更新股票基础信息...")
    
    try:
        total_rows = 0
        for status, desc in [('L', '上市'), ('D', '退市'), ('P', '暂停上市')]:
            rows = downloader.download_stock_basic(list_status=status)
            logger.info(f"  - {desc}股票: {rows} 只")
            total_rows += rows
        
        logger.info(f"✓ 股票基础信息更新完成，总计 {total_rows} 只")
    except Exception as e:
        logger.error(f"✗ 更新股票基础信息失败: {e}")
        raise


def update_daily_data(downloader: DataDownloader):
    """
    智能增量更新每日数据（日线、复权因子、每日基本面）
    
    策略：
        1. 查询数据库中最新的交易日期
        2. 从最新日期的下一天开始，到今天为止
        3. 逐日更新所有缺失的交易日数据
        4. 自动跳过非交易日
    
    说明：
        使用 download_daily_data_by_date 方法一次性下载：
        1. 所有股票的日线数据（pro_bar）
        2. 所有股票的复权因子（adj_factor）
        3. 所有股票的每日基本面（daily_basic）
    """
    logger.info("=" * 60)
    logger.info("开始智能增量更新每日数据...")
    
    try:
        # 1. 获取数据库中最新的交易日期
        latest_date = downloader.db.get_latest_date('pro_bar', 'trade_date')
        today = datetime.now().strftime('%Y%m%d')
        
        if latest_date is None:
            logger.info("数据库中没有历史数据，将从今天开始更新")
            start_date = today
        else:
            # 从最新日期的下一天开始
            latest_dt = datetime.strptime(latest_date, '%Y%m%d')
            start_date = (latest_dt + timedelta(days=1)).strftime('%Y%m%d')
            logger.info(f"数据库最新日期: {latest_date}")
        
        logger.info(f"更新范围: {start_date} → {today}")
        
        # 2. 查询这段时间内的所有交易日
        trading_dates_df = downloader.db.execute_query('''
            SELECT cal_date 
            FROM trade_cal 
            WHERE cal_date >= ? AND cal_date <= ? AND is_open = 1
            ORDER BY cal_date
        ''', [start_date, today])
        
        if trading_dates_df.empty:
            logger.info(f"期间 {start_date} → {today} 没有交易日，无需更新")
            return
        
        trading_dates = trading_dates_df['cal_date'].tolist()
        logger.info(f"发现 {len(trading_dates)} 个交易日需要更新")
        
        # 3. 逐日更新
        success_count = 0
        fail_count = 0
        
        for trade_date in trading_dates:
            try:
                logger.info(f"  [{success_count + fail_count + 1}/{len(trading_dates)}] 更新 {trade_date}...")
                downloader.download_daily_data_by_date(trade_date)
                success_count += 1
            except Exception as e:
                logger.error(f"    ✗ 失败: {e}")
                fail_count += 1
                # 继续处理其他日期
                continue
        
        # 4. 汇总结果
        logger.info("=" * 60)
        logger.info(f"✓ 每日数据增量更新完成")
        logger.info(f"  - 成功: {success_count} 个交易日")
        if fail_count > 0:
            logger.warning(f"  - 失败: {fail_count} 个交易日")
        logger.info(f"  - 更新范围: {start_date} → {today}")
        
    except Exception as e:
        logger.error(f"✗ 每日数据增量更新失败: {e}")
        raise


def update_financial_indicators(downloader: DataDownloader):
    """
    更新财务指标数据（VIP接口）
    
    策略：获取最近8个季度的数据，确保捕获所有延迟披露的财报
    
    注意：此功能需要 DataDownloader 扩展 download_fina_indicator_vip 方法
    TODO: 等待 DataDownloader 实现此方法后启用
    """
    logger.info("=" * 60)
    logger.info("开始更新财务指标数据...")
    
    try:
        # 生成最近8个季度的结束日期
        quarters = _generate_recent_quarters(8)
        
        logger.info(f"将更新 {len(quarters)} 个季度的财务数据: {quarters}")
        
        # TODO: 目前 DataDownloader 还没有 download_fina_indicator_vip 方法
        # 需要在 downloader.py 中添加类似这样的方法：
        #
        # def download_fina_indicator_vip(self, period: str):
        #     """下载财务指标数据（VIP接口）"""
        #     df = self.fetcher.fetch('fina_indicator_vip', period=period)
        #     if not df.empty:
        #         self.db.write_dataframe(df, 'fina_indicator_vip', mode='append')
        #     return len(df)
        
        logger.warning("⚠ 财务指标更新功能需要扩展 DataDownloader，当前跳过")
        logger.info("  建议：在 src/tushare_db/downloader.py 中添加 download_fina_indicator_vip 方法")
        
        # 临时方案：直接使用 fetcher
        for period in quarters:
            logger.info(f"  正在获取 {period} 的财务指标...")
            df = downloader.fetcher.fetch('fina_indicator_vip', period=period)
            if not df.empty:
                downloader.db.write_dataframe(df, 'fina_indicator_vip', mode='append')
                logger.info(f"    ✓ {period}: {len(df)} 行")
            else:
                logger.info(f"    - {period}: 无数据")
        
        logger.info("✓ 财务指标数据更新完成")
        
    except Exception as e:
        logger.error(f"✗ 更新财务指标数据失败: {e}")
        # 财务数据不阻塞主流程
        logger.warning("  继续执行其他更新任务...")


def update_cyq_perf(downloader: DataDownloader):
    """
    增量更新筹码分布数据 (cyq_perf)

    策略：
        1. 获取数据库中 cyq_perf 的最新日期
        2. 从下一天开始更新到今天
    """
    logger.info("=" * 60)
    logger.info("开始增量更新筹码分布数据 (cyq_perf)...")

    try:
        # 1. 获取最新日期
        latest_date = downloader.db.get_latest_date('cyq_perf', 'trade_date')
        today = datetime.now().strftime('%Y%m%d')

        if latest_date is None:
            logger.info("数据库中没有筹码历史数据，将从今天开始更新")
            # 实际上如果没有数据，可能需要更长时间的历史，这里先简单设为今天
            # 或者设置为一个默认的起点，例如最近30天
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
            logger.info(f"  (初始化: 从30天前 {start_date} 开始)")
        else:
            latest_dt = datetime.strptime(latest_date, '%Y%m%d')
            start_date = (latest_dt + timedelta(days=1)).strftime('%Y%m%d')
            logger.info(f"数据库最新日期: {latest_date}")

        logger.info(f"更新范围: {start_date} → {today}")

        # 2. 获取交易日
        if start_date > today:
            logger.info("无需更新")
            return

        trading_dates_df = downloader.db.execute_query('''
            SELECT cal_date
            FROM trade_cal
            WHERE cal_date >= ? AND cal_date <= ? AND is_open = 1
            ORDER BY cal_date
        ''', [start_date, today])

        if trading_dates_df.empty:
            logger.info("期间无交易日")
            return

        trading_dates = trading_dates_df['cal_date'].tolist()
        logger.info(f"需要更新 {len(trading_dates)} 个交易日")

        # 3. 逐日更新
        success_count = 0
        for trade_date in trading_dates:
            try:
                rows = downloader.download_cyq_perf(trade_date)
                if rows > 0:
                    success_count += 1
            except Exception as e:
                logger.error(f"  ✗ {trade_date} 更新失败: {e}")
                # 不中断，继续下一个

        logger.info(f"✓ 筹码分布更新完成: 成功 {success_count}/{len(trading_dates)}")

    except Exception as e:
        logger.error(f"✗ 更新筹码分布数据失败: {e}")
        logger.warning("  继续执行其他更新任务...")


def update_dc_member(downloader: DataDownloader):
    """
    增量更新龙虎榜机构席位交易明细数据 (dc_member)

    策略：
        1. 获取数据库中 dc_member 的最新日期
        2. 从下一天开始更新到今天
    """
    logger.info("=" * 60)
    logger.info("开始增量更新龙虎榜机构席位数据 (dc_member)...")

    try:
        # 1. 获取最新日期
        latest_date = downloader.db.get_latest_date('dc_member', 'trade_date')
        today = datetime.now().strftime('%Y%m%d')

        if latest_date is None:
            logger.info("数据库中没有龙虎榜历史数据，将从30天前开始更新")
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
            logger.info(f"  (初始化: 从30天前 {start_date} 开始)")
        else:
            latest_dt = datetime.strptime(latest_date, '%Y%m%d')
            start_date = (latest_dt + timedelta(days=1)).strftime('%Y%m%d')
            logger.info(f"数据库最新日期: {latest_date}")

        logger.info(f"更新范围: {start_date} → {today}")

        # 2. 获取交易日
        if start_date > today:
            logger.info("无需更新")
            return

        trading_dates_df = downloader.db.execute_query('''
            SELECT cal_date
            FROM trade_cal
            WHERE cal_date >= ? AND cal_date <= ? AND is_open = 1
            ORDER BY cal_date
        ''', [start_date, today])

        if trading_dates_df.empty:
            logger.info("期间无交易日")
            return

        trading_dates = trading_dates_df['cal_date'].tolist()
        logger.info(f"需要更新 {len(trading_dates)} 个交易日")

        # 3. 逐日更新
        success_count = 0
        for trade_date in trading_dates:
            try:
                rows = downloader.download_dc_member(trade_date)
                if rows > 0:
                    success_count += 1
            except Exception as e:
                logger.error(f"  ✗ {trade_date} 更新失败: {e}")
                # 不中断，继续下一个

        logger.info(f"✓ 龙虎榜机构席位更新完成: 成功 {success_count}/{len(trading_dates)}")

    except Exception as e:
        logger.error(f"✗ 更新龙虎榜机构席位数据失败: {e}")
        logger.warning("  继续执行其他更新任务...")


def update_moneyflow_ind_dc(downloader: DataDownloader):
    """
    增量更新行业资金流向（沪深通）数据 (moneyflow_ind_dc)

    策略：
        1. 获取数据库中 moneyflow_ind_dc 的最新日期
        2. 从下一天开始更新到今天
    """
    logger.info("=" * 60)
    logger.info("开始增量更新行业资金流向数据 (moneyflow_ind_dc)...")

    try:
        # 1. 获取最新日期
        latest_date = downloader.db.get_latest_date('moneyflow_ind_dc', 'trade_date')
        today = datetime.now().strftime('%Y%m%d')

        if latest_date is None:
            logger.info("数据库中没有行业资金流向历史数据，将从30天前开始更新")
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
            logger.info(f"  (初始化: 从30天前 {start_date} 开始)")
        else:
            latest_dt = datetime.strptime(latest_date, '%Y%m%d')
            start_date = (latest_dt + timedelta(days=1)).strftime('%Y%m%d')
            logger.info(f"数据库最新日期: {latest_date}")

        logger.info(f"更新范围: {start_date} → {today}")

        # 2. 获取交易日
        if start_date > today:
            logger.info("无需更新")
            return

        trading_dates_df = downloader.db.execute_query('''
            SELECT cal_date
            FROM trade_cal
            WHERE cal_date >= ? AND cal_date <= ? AND is_open = 1
            ORDER BY cal_date
        ''', [start_date, today])

        if trading_dates_df.empty:
            logger.info("期间无交易日")
            return

        trading_dates = trading_dates_df['cal_date'].tolist()
        logger.info(f"需要更新 {len(trading_dates)} 个交易日")

        # 3. 逐日更新
        success_count = 0
        for trade_date in trading_dates:
            try:
                rows = downloader.download_moneyflow_ind_dc(trade_date)
                if rows > 0:
                    success_count += 1
            except Exception as e:
                logger.error(f"  ✗ {trade_date} 更新失败: {e}")
                # 不中断，继续下一个

        logger.info(f"✓ 行业资金流向更新完成: 成功 {success_count}/{len(trading_dates)}")

    except Exception as e:
        logger.error(f"✗ 更新行业资金流向数据失败: {e}")
        logger.warning("  继续执行其他更新任务...")


def _generate_recent_quarters(count: int = 8) -> list:
    """
    生成最近 N 个季度的结束日期
    
    参数：
        count: 要生成的季度数量
    
    返回：
        季度结束日期列表，格式 YYYYMMDD，例如 ['20231231', '20230930', ...]
    """
    quarters = []
    current_date = datetime.now()
    
    for _ in range(count):
        year = current_date.year
        month = current_date.month
        
        # 确定当前季度的结束月份
        if month >= 10:  # Q4
            quarter_end_month = 12
        elif month >= 7:  # Q3
            quarter_end_month = 9
        elif month >= 4:  # Q2
            quarter_end_month = 6
        else:  # Q1
            quarter_end_month = 3
        
        # 构造季度结束日期
        if quarter_end_month in [3, 12]:
            period_str = f"{year}{quarter_end_month:02d}31"
        else:
            period_str = f"{year}{quarter_end_month:02d}30"
        
        quarters.append(period_str)
        
        # 移动到上一个季度
        if quarter_end_month == 3:
            current_date = datetime(year - 1, 12, 31)
        else:
            current_date = datetime(year, quarter_end_month - 3, 1)
    
    # 去重并按降序排序（最新的在前）
    quarters = sorted(list(set(quarters)), reverse=True)
    
    # 只返回不晚于今天的季度
    today = datetime.now().strftime('%Y%m%d')
    quarters = [q for q in quarters if q <= today]
    
    return quarters


def main():
    """
    主函数：执行所有每日更新任务

    执行顺序：
    1. 交易日历（基础数据，其他任务依赖）
    2. 股票列表（基础数据）
    3. 每日数据（日线、复权、基本面）
    4. 财务指标（季度数据）
    5. 筹码分布
    6. 龙虎榜机构席位数据
    7. 行业资金流向（沪深通）数据
    """
    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info("开始每日数据更新任务")
    logger.info(f"启动时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    
    try:
        # 1. 获取配置
        tushare_token, db_path = get_env_config()
        logger.info(f"数据库路径: {db_path}")
        
        # 2. 初始化下载器
        downloader = DataDownloader(
            tushare_token=tushare_token,
            db_path=db_path,
            rate_limit_profile="standard"  # 根据你的账号权限调整：trial/standard/pro
        )
        
        try:
            # 3. 执行更新任务
            update_trade_calendar(downloader)
            update_stock_basic(downloader)
            update_daily_data(downloader)
            update_financial_indicators(downloader)
            update_cyq_perf(downloader)
            update_dc_member(downloader)
            update_moneyflow_ind_dc(downloader)

            # 4. 完成
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.info("=" * 60)
            logger.info("✓ 所有每日数据更新任务完成！")
            logger.info(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"总耗时: {duration:.1f} 秒")
            logger.info("=" * 60)
            
        finally:
            # 5. 关闭连接
            downloader.close()
            logger.info("数据库连接已关闭")
    
    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"✗ 每日更新任务失败: {e}")
        logger.error("=" * 60)
        raise


if __name__ == "__main__":
    main()