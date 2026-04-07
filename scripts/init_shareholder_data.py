#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
股东数据初始化脚本

功能：
1. 初始化前十大流通股东数据 (top10_floatholders)
2. 初始化股东人数数据 (stk_holdernumber)
3. 初始化高管薪酬数据 (stk_rewards)

使用方法：
    python scripts/init_shareholder_data.py

环境变量：
    TUSHARE_TOKEN: Tushare API Token（必须）
    DB_PATH: 数据库路径（可选，默认为 tushare.db）
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tushare_db import DataDownloader, DataReader
from tushare_db.logger import get_logger

# 配置日志
logger = get_logger(__name__)


def get_env_config():
    """获取环境变量配置"""
    tushare_token = os.getenv("TUSHARE_TOKEN")
    if not tushare_token:
        raise ValueError("请设置 TUSHARE_TOKEN 环境变量")

    db_path = os.getenv("DB_PATH", str(project_root / "tushare.db"))

    return tushare_token, db_path


def init_top10_floatholders(downloader: DataDownloader, start_date: str = '20100101'):
    """
    初始化前十大流通股东数据

    策略：
        按公告期(ann_date)批量获取，从start_date到最新
        每个公告期获取当日所有股票的流通股东数据

    Args:
        downloader: DataDownloader 实例
        start_date: 开始日期，默认2010年1月1日
    """
    logger.info("=" * 60)
    logger.info("开始初始化前十大流通股东数据...")

    try:
        # 获取所有公告期（从财报日期推算）
        # 季报公告期：0331, 0630, 0930, 1231
        current_year = datetime.now().year
        start_year = int(start_date[:4])

        periods = []
        for year in range(start_year, current_year + 1):
            for quarter in ['0331', '0630', '0930', '1231']:
                period = f"{year}{quarter}"
                if datetime.strptime(period, '%Y%m%d') <= datetime.now():
                    periods.append(period)

        logger.info(f"将处理 {len(periods)} 个报告期")

        total_rows = 0
        for period in tqdm(periods, desc="初始化流通股东数据"):
            try:
                # 使用 end_date 参数获取该报告期的数据
                rows = downloader.download_top10_floatholders(end_date=period)
                if rows > 0:
                    total_rows += rows
                    logger.info(f"  {period}: {rows} 条记录")
            except Exception as e:
                logger.warning(f"  {period} 获取失败: {e}")
                continue

        logger.info(f"✓ 前十大流通股东数据初始化完成: 共 {total_rows} 条记录")

    except Exception as e:
        logger.error(f"✗ 初始化前十大流通股东数据失败: {e}")
        raise


def init_stk_holdernumber(downloader: DataDownloader, start_date: str = '20100101'):
    """
    初始化股东人数数据

    策略：
        按股票代码逐个获取历史数据
        从上市日期或start_date开始

    Args:
        downloader: DataDownloader 实例
        start_date: 开始日期，默认2010年1月1日
    """
    logger.info("=" * 60)
    logger.info("开始初始化股东人数数据...")

    try:
        # 获取所有股票列表 (直接使用 downloader.db 查询避免连接冲突)
        stocks_df = downloader.db.execute_query(
            "SELECT ts_code, list_date FROM stock_basic WHERE list_status = 'L'"
        )
        if stocks_df.empty:
            logger.warning("未找到股票列表，请先初始化 stock_basic 表")
            return
        stocks = stocks_df.to_dict('records')
        logger.info(f"将处理 {len(stocks)} 只股票")

        total_rows = 0
        for stock in tqdm(stocks, total=len(stocks), desc="初始化股东人数"):
            ts_code = stock['ts_code']
            list_date = stock.get('list_date', start_date)

            # 使用较晚的日期
            stock_start = max(start_date, list_date) if list_date else start_date

            try:
                rows = downloader.download_stk_holdernumber(
                    ts_code=ts_code,
                    start_date=stock_start
                )
                if rows > 0:
                    total_rows += rows
            except Exception as e:
                logger.warning(f"  {ts_code} 获取失败: {e}")
                continue

        logger.info(f"✓ 股东人数数据初始化完成: 共 {total_rows} 条记录")

    except Exception as e:
        logger.error(f"✗ 初始化股东人数数据失败: {e}")
        raise


def init_stk_rewards(downloader: DataDownloader, start_year: int = 2010):
    """
    初始化高管薪酬数据

    策略：
        按股票代码逐个获取，从start_year到最新年份
        每年获取所有上市公司的高管薪酬数据

    Args:
        downloader: DataDownloader 实例
        start_year: 开始年份，默认2010年
    """
    logger.info("=" * 60)
    logger.info("开始初始化高管薪酬数据...")

    try:
        # 获取所有股票列表
        stocks_df = downloader.db.execute_query(
            "SELECT ts_code, list_date FROM stock_basic WHERE list_status = 'L'"
        )
        if stocks_df.empty:
            logger.warning("未找到股票列表，请先初始化 stock_basic 表")
            return
        stocks = stocks_df.to_dict('records')
        logger.info(f"将处理 {len(stocks)} 只股票")

        current_year = datetime.now().year
        total_rows = 0

        for stock in tqdm(stocks, total=len(stocks), desc="初始化高管薪酬"):
            ts_code = stock['ts_code']

            for year in range(start_year, current_year + 1):
                end_date = f"{year}1231"
                try:
                    rows = downloader.download_stk_rewards(
                        ts_code=ts_code,
                        end_date=end_date
                    )
                    if rows > 0:
                        total_rows += rows
                except Exception as e:
                    logger.debug(f"  {ts_code} {year}年获取失败: {e}")
                    continue

        logger.info(f"✓ 高管薪酬数据初始化完成: 共 {total_rows} 条记录")

    except Exception as e:
        logger.error(f"✗ 初始化高管薪酬数据失败: {e}")
        raise


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="股东数据初始化脚本")
    parser.add_argument('--top10-floatholders', action='store_true',
                        help='初始化前十大流通股东数据')
    parser.add_argument('--stk-holdernumber', action='store_true',
                        help='初始化股东人数数据')
    parser.add_argument('--stk-rewards', action='store_true',
                        help='初始化高管薪酬数据')
    parser.add_argument('--all', action='store_true',
                        help='初始化所有股东数据')
    parser.add_argument('--start-date', type=str, default='20100101',
                        help='开始日期 (YYYYMMDD)')
    parser.add_argument('--start-year', type=int, default=2010,
                        help='开始年份（用于高管薪酬）')

    args = parser.parse_args()

    # 如果没有指定任何选项，默认初始化全部
    if not any([args.top10_floatholders, args.stk_holdernumber, args.stk_rewards, args.all]):
        args.all = True

    # 获取配置
    tushare_token, db_path = get_env_config()

    logger.info("=" * 60)
    logger.info("开始股东数据初始化...")
    logger.info(f"数据库路径: {db_path}")
    logger.info("=" * 60)

    # 初始化下载器
    downloader = DataDownloader(
        tushare_token=tushare_token,
        db_path=db_path,
        rate_limit_profile="standard"
    )

    try:
        if args.all or args.top10_floatholders:
            init_top10_floatholders(downloader, start_date=args.start_date)

        if args.all or args.stk_holdernumber:
            init_stk_holdernumber(downloader, start_date=args.start_date)

        if args.all or args.stk_rewards:
            init_stk_rewards(downloader, start_year=args.start_year)

        logger.info("=" * 60)
        logger.info("✓ 所有股东数据初始化任务完成！")
        logger.info("=" * 60)

    finally:
        downloader.close()
        logger.info("数据库连接已关闭")


if __name__ == "__main__":
    main()
