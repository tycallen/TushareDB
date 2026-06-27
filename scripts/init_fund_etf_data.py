#!/usr/bin/env python3
"""
基金/ETF数据初始化脚本

一键初始化以下表：
- etf_basic: ETF基础信息
- etf_share: ETF份额规模
- fund_company: 基金公司信息
- etf_index: ETF基准指数

使用方法：
    python scripts/init_fund_etf_data.py --all
    python scripts/init_fund_etf_data.py --etf-basic --etf-share
    python scripts/init_fund_etf_data.py --help
"""

import os
import sys
import argparse
import logging
from datetime import datetime, timedelta

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tushare_db import DataDownloader

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def init_etf_basic(downloader: DataDownloader) -> int:
    """初始化ETF基础信息"""
    logger.info("=" * 60)
    logger.info("开始初始化ETF基础信息(etf_basic)...")
    logger.info("=" * 60)

    total_rows = 0
    markets = ['SH', 'SZ']

    for market in markets:
        logger.info(f"下载 {market} 市场ETF基础信息...")
        try:
            rows = downloader.download_etf_basic(market=market)
            total_rows += rows
            logger.info(f"{market} 市场ETF: {rows} 行")
        except Exception as e:
            logger.error(f"下载 {market} 市场ETF失败: {e}")

    logger.info(f"ETF基础信息初始化完成，总计: {total_rows} 行")
    return total_rows


def init_etf_share(downloader: DataDownloader, start_date: str = None, end_date: str = None) -> int:
    """
    初始化ETF份额规模数据

    Args:
        start_date: 开始日期 YYYYMMDD，默认一年前
        end_date: 结束日期 YYYYMMDD，默认今天
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y%m%d')
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')

    logger.info("=" * 60)
    logger.info(f"开始初始化ETF份额规模(etf_share): {start_date} -> {end_date}")
    logger.info("=" * 60)

    total_rows = 0
    exchanges = ['SSE', 'SZSE']

    for exchange in exchanges:
        logger.info(f"下载 {exchange} 交易所ETF份额规模...")
        try:
            rows = downloader.download_etf_share(
                start_date=start_date,
                end_date=end_date,
                exchange=exchange
            )
            total_rows += rows
            logger.info(f"{exchange} 交易所: {rows} 行")
        except Exception as e:
            error_msg = str(e)
            if "没有接口访问权限" in error_msg or "permission" in error_msg.lower():
                print(f"\n⚠️  {exchange} 交易所: 权限不足 (需要8000积分)")
                print("    请访问 https://tushare.pro/document/1?doc_id=108 提升积分")
            else:
                logger.error(f"下载 {exchange} 交易所ETF份额失败: {e}")

    logger.info(f"ETF份额规模初始化完成，总计: {total_rows} 行")
    return total_rows


def init_fund_company(downloader: DataDownloader) -> int:
    """初始化基金公司信息"""
    logger.info("=" * 60)
    logger.info("开始初始化基金公司信息(fund_company)...")
    logger.info("=" * 60)

    try:
        rows = downloader.download_fund_company()
        logger.info(f"基金公司信息初始化完成: {rows} 行")
        return rows
    except Exception as e:
        logger.error(f"下载基金公司信息失败: {e}")
        return 0


def init_etf_index(downloader: DataDownloader) -> int:
    """初始化ETF基准指数"""
    logger.info("=" * 60)
    logger.info("开始初始化ETF基准指数(etf_index)...")
    logger.info("=" * 60)

    try:
        rows = downloader.download_etf_index()
        logger.info(f"ETF基准指数初始化完成: {rows} 行")
        return rows
    except Exception as e:
        logger.error(f"下载ETF基准指数失败: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(
        description='初始化基金/ETF数据',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 初始化所有数据
    python scripts/init_fund_etf_data.py --all

    # 只初始化ETF基础信息
    python scripts/init_fund_etf_data.py --etf-basic

    # 初始化ETF份额（指定日期范围）
    python scripts/init_fund_etf_data.py --etf-share --start-date 20240101 --end-date 20241231

    # 初始化基金公司和ETF指数
    python scripts/init_fund_etf_data.py --fund-company --etf-index
        """
    )

    parser.add_argument('--etf-basic', action='store_true', help='初始化ETF基础信息')
    parser.add_argument('--etf-share', action='store_true', help='初始化ETF份额规模')
    parser.add_argument('--fund-company', action='store_true', help='初始化基金公司信息')
    parser.add_argument('--etf-index', action='store_true', help='初始化ETF基准指数')
    parser.add_argument('--all', action='store_true', help='初始化所有数据')
    parser.add_argument('--start-date', type=str, help='ETF份额开始日期 YYYYMMDD')
    parser.add_argument('--end-date', type=str, help='ETF份额结束日期 YYYYMMDD')
    parser.add_argument('--db-path', type=str, default='tushare.db', help='数据库路径')

    args = parser.parse_args()

    # 如果没有指定任何参数，显示帮助
    if not any([args.etf_basic, args.etf_share, args.fund_company, args.etf_index, args.all]):
        parser.print_help()
        return

    # 初始化下载器
    try:
        downloader = DataDownloader(db_path=args.db_path)
    except Exception as e:
        logger.error(f"初始化下载器失败: {e}")
        sys.exit(1)

    try:
        total_stats = {}

        # ETF基础信息
        if args.all or args.etf_basic:
            total_stats['etf_basic'] = init_etf_basic(downloader)

        # ETF份额规模
        if args.all or args.etf_share:
            total_stats['etf_share'] = init_etf_share(
                downloader,
                start_date=args.start_date,
                end_date=args.end_date
            )

        # 基金公司信息
        if args.all or args.fund_company:
            total_stats['fund_company'] = init_fund_company(downloader)

        # ETF基准指数
        if args.all or args.etf_index:
            total_stats['etf_index'] = init_etf_index(downloader)

        # 打印汇总
        logger.info("\n" + "=" * 60)
        logger.info("初始化完成汇总")
        logger.info("=" * 60)
        for table, rows in total_stats.items():
            logger.info(f"{table}: {rows} 行")
        logger.info("=" * 60)

    except KeyboardInterrupt:
        logger.info("\n用户中断初始化")
    except Exception as e:
        logger.error(f"初始化过程中出错: {e}", exc_info=True)
    finally:
        downloader.close()


if __name__ == '__main__':
    main()
