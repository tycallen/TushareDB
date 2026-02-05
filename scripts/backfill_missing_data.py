#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据补充脚本

功能：检测并补充缺失的数据
1. adj_factor - 复权因子（与 daily 表对比）
2. daily_basic - 每日基本面（与 daily 表对比）
3. stk_factor_pro - 技术因子
4. dc_index - 龙虎榜个股明细

使用方法：
    # 补充所有表的缺失数据
    python scripts/backfill_missing_data.py --all

    # 只补充特定表
    python scripts/backfill_missing_data.py --adj-factor
    python scripts/backfill_missing_data.py --daily-basic
    python scripts/backfill_missing_data.py --stk-factor-pro
    python scripts/backfill_missing_data.py --dc-index

    # 指定日期范围
    python scripts/backfill_missing_data.py --all --start-date 20240101 --end-date 20240131

环境变量：
    TUSHARE_TOKEN: Tushare API Token（必须）
    DB_PATH: 数据库路径（可选，默认为 tushare.db）

作者：根据数据完整性修复计划创建
日期：2026-02-04
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Set, Optional

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


def get_trading_dates(downloader: DataDownloader, start_date: str, end_date: str) -> List[str]:
    """获取指定日期范围内的所有交易日"""
    df = downloader.db.execute_query('''
        SELECT cal_date
        FROM trade_cal
        WHERE cal_date >= ? AND cal_date <= ? AND is_open = 1
        ORDER BY cal_date
    ''', [start_date, end_date])

    if df.empty:
        return []

    return df['cal_date'].tolist()


def get_dates_in_table(downloader: DataDownloader, table_name: str, start_date: str, end_date: str) -> Set[str]:
    """获取表中指定日期范围内已有的日期"""
    if not downloader.db.table_exists(table_name):
        return set()

    df = downloader.db.execute_query(f'''
        SELECT DISTINCT trade_date
        FROM {table_name}
        WHERE trade_date >= ? AND trade_date <= ?
    ''', [start_date, end_date])

    if df.empty:
        return set()

    return set(df['trade_date'].astype(str).tolist())


def get_dates_in_daily(downloader: DataDownloader, start_date: str, end_date: str) -> Set[str]:
    """获取 daily 表中指定日期范围内已有的日期"""
    return get_dates_in_table(downloader, 'daily', start_date, end_date)


def backfill_adj_factor(downloader: DataDownloader, start_date: str, end_date: str) -> dict:
    """
    补充 adj_factor 缺失的日期

    策略：与 daily 表对比，找出在 daily 表中有数据但 adj_factor 中缺失的日期
    """
    logger.info("=" * 60)
    logger.info("开始检测并补充 adj_factor 缺失数据...")

    # 获取 daily 表中的日期
    daily_dates = get_dates_in_daily(downloader, start_date, end_date)
    if not daily_dates:
        logger.warning("daily 表中没有数据，无法确定缺失日期")
        return {'total': 0, 'missing': 0, 'success': 0, 'failed': 0}

    # 获取 adj_factor 表中的日期
    adj_dates = get_dates_in_table(downloader, 'adj_factor', start_date, end_date)

    # 计算缺失日期
    missing_dates = sorted(daily_dates - adj_dates)

    logger.info(f"daily 表日期数: {len(daily_dates)}")
    logger.info(f"adj_factor 表日期数: {len(adj_dates)}")
    logger.info(f"缺失日期数: {len(missing_dates)}")

    if not missing_dates:
        logger.info("adj_factor 数据完整，无需补充")
        return {'total': len(daily_dates), 'missing': 0, 'success': 0, 'failed': 0}

    logger.info(f"缺失日期: {missing_dates[:10]}{'...' if len(missing_dates) > 10 else ''}")

    # 逐日补充
    success_count = 0
    failed_count = 0

    for i, trade_date in enumerate(missing_dates):
        try:
            logger.info(f"  [{i+1}/{len(missing_dates)}] 补充 adj_factor: {trade_date}")
            df = downloader.fetcher.fetch('adj_factor', trade_date=trade_date)
            if not df.empty:
                downloader.db.write_dataframe(df, 'adj_factor', mode='append')
                logger.info(f"    ✓ 下载 {len(df)} 行")
                success_count += 1
            else:
                logger.warning(f"    - API 返回空数据")
                failed_count += 1
        except Exception as e:
            logger.error(f"    ✗ 失败: {e}")
            failed_count += 1

    logger.info(f"✓ adj_factor 补充完成: 成功 {success_count}, 失败 {failed_count}")
    return {
        'total': len(daily_dates),
        'missing': len(missing_dates),
        'success': success_count,
        'failed': failed_count
    }


def backfill_daily_basic(downloader: DataDownloader, start_date: str, end_date: str) -> dict:
    """
    补充 daily_basic 缺失的日期

    策略：与 daily 表对比，找出在 daily 表中有数据但 daily_basic 中缺失的日期
    """
    logger.info("=" * 60)
    logger.info("开始检测并补充 daily_basic 缺失数据...")

    # 获取 daily 表中的日期
    daily_dates = get_dates_in_daily(downloader, start_date, end_date)
    if not daily_dates:
        logger.warning("daily 表中没有数据，无法确定缺失日期")
        return {'total': 0, 'missing': 0, 'success': 0, 'failed': 0}

    # 获取 daily_basic 表中的日期
    basic_dates = get_dates_in_table(downloader, 'daily_basic', start_date, end_date)

    # 计算缺失日期
    missing_dates = sorted(daily_dates - basic_dates)

    logger.info(f"daily 表日期数: {len(daily_dates)}")
    logger.info(f"daily_basic 表日期数: {len(basic_dates)}")
    logger.info(f"缺失日期数: {len(missing_dates)}")

    if not missing_dates:
        logger.info("daily_basic 数据完整，无需补充")
        return {'total': len(daily_dates), 'missing': 0, 'success': 0, 'failed': 0}

    logger.info(f"缺失日期: {missing_dates[:10]}{'...' if len(missing_dates) > 10 else ''}")

    # 逐日补充
    success_count = 0
    failed_count = 0

    for i, trade_date in enumerate(missing_dates):
        try:
            logger.info(f"  [{i+1}/{len(missing_dates)}] 补充 daily_basic: {trade_date}")
            df = downloader.fetcher.fetch('daily_basic', trade_date=trade_date)
            if not df.empty:
                downloader.db.write_dataframe(df, 'daily_basic', mode='append')
                logger.info(f"    ✓ 下载 {len(df)} 行")
                success_count += 1
            else:
                logger.warning(f"    - API 返回空数据")
                failed_count += 1
        except Exception as e:
            logger.error(f"    ✗ 失败: {e}")
            failed_count += 1

    logger.info(f"✓ daily_basic 补充完成: 成功 {success_count}, 失败 {failed_count}")
    return {
        'total': len(daily_dates),
        'missing': len(missing_dates),
        'success': success_count,
        'failed': failed_count
    }


def backfill_stk_factor_pro(downloader: DataDownloader, start_date: str, end_date: str) -> dict:
    """
    补充 stk_factor_pro 缺失的日期

    策略：与交易日历对比，找出缺失的交易日
    """
    logger.info("=" * 60)
    logger.info("开始检测并补充 stk_factor_pro 缺失数据...")

    # 获取交易日
    trading_dates = set(get_trading_dates(downloader, start_date, end_date))
    if not trading_dates:
        logger.warning("交易日历中没有数据")
        return {'total': 0, 'missing': 0, 'success': 0, 'failed': 0}

    # 获取 stk_factor_pro 表中的日期
    factor_dates = get_dates_in_table(downloader, 'stk_factor_pro', start_date, end_date)

    # 计算缺失日期
    missing_dates = sorted(trading_dates - factor_dates)

    logger.info(f"交易日数: {len(trading_dates)}")
    logger.info(f"stk_factor_pro 表日期数: {len(factor_dates)}")
    logger.info(f"缺失日期数: {len(missing_dates)}")

    if not missing_dates:
        logger.info("stk_factor_pro 数据完整，无需补充")
        return {'total': len(trading_dates), 'missing': 0, 'success': 0, 'failed': 0}

    logger.info(f"缺失日期: {missing_dates[:10]}{'...' if len(missing_dates) > 10 else ''}")

    # 逐日补充（使用按日期批量下载）
    success_count = 0
    failed_count = 0

    for i, trade_date in enumerate(missing_dates):
        try:
            logger.info(f"  [{i+1}/{len(missing_dates)}] 补充 stk_factor_pro: {trade_date}")
            rows = downloader.download_stk_factor_pro_by_date(trade_date)
            if rows > 0:
                logger.info(f"    ✓ 下载 {rows} 行")
                success_count += 1
            else:
                logger.warning(f"    - API 返回空数据")
                failed_count += 1
        except Exception as e:
            logger.error(f"    ✗ 失败: {e}")
            failed_count += 1

    logger.info(f"✓ stk_factor_pro 补充完成: 成功 {success_count}, 失败 {failed_count}")
    return {
        'total': len(trading_dates),
        'missing': len(missing_dates),
        'success': success_count,
        'failed': failed_count
    }


def backfill_dc_index(downloader: DataDownloader, start_date: str, end_date: str) -> dict:
    """
    补充 dc_index 缺失的日期

    策略：与交易日历对比，找出缺失的交易日
    """
    logger.info("=" * 60)
    logger.info("开始检测并补充 dc_index 缺失数据...")

    # 获取交易日
    trading_dates = set(get_trading_dates(downloader, start_date, end_date))
    if not trading_dates:
        logger.warning("交易日历中没有数据")
        return {'total': 0, 'missing': 0, 'success': 0, 'failed': 0}

    # 获取 dc_index 表中的日期
    dc_dates = get_dates_in_table(downloader, 'dc_index', start_date, end_date)

    # 计算缺失日期
    missing_dates = sorted(trading_dates - dc_dates)

    logger.info(f"交易日数: {len(trading_dates)}")
    logger.info(f"dc_index 表日期数: {len(dc_dates)}")
    logger.info(f"缺失日期数: {len(missing_dates)}")

    if not missing_dates:
        logger.info("dc_index 数据完整，无需补充")
        return {'total': len(trading_dates), 'missing': 0, 'success': 0, 'failed': 0}

    logger.info(f"缺失日期: {missing_dates[:10]}{'...' if len(missing_dates) > 10 else ''}")

    # 逐日补充
    success_count = 0
    failed_count = 0

    for i, trade_date in enumerate(missing_dates):
        try:
            logger.info(f"  [{i+1}/{len(missing_dates)}] 补充 dc_index: {trade_date}")
            rows = downloader.download_dc_index(trade_date)
            if rows > 0:
                logger.info(f"    ✓ 下载 {rows} 行")
                success_count += 1
            else:
                # 龙虎榜数据不是每天都有，空数据是正常的
                logger.info(f"    - 当日无龙虎榜数据（正常）")
        except Exception as e:
            logger.error(f"    ✗ 失败: {e}")
            failed_count += 1

    logger.info(f"✓ dc_index 补充完成: 成功 {success_count}, 失败 {failed_count}")
    return {
        'total': len(trading_dates),
        'missing': len(missing_dates),
        'success': success_count,
        'failed': failed_count
    }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='数据补充脚本 - 检测并补充缺失的数据',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
    # 补充所有表的缺失数据
    python scripts/backfill_missing_data.py --all

    # 只补充特定表
    python scripts/backfill_missing_data.py --adj-factor
    python scripts/backfill_missing_data.py --daily-basic

    # 指定日期范围
    python scripts/backfill_missing_data.py --all --start-date 20240101 --end-date 20240131
        '''
    )

    parser.add_argument('--all', action='store_true', help='补充所有表的缺失数据')
    parser.add_argument('--adj-factor', action='store_true', help='补充 adj_factor 缺失数据')
    parser.add_argument('--daily-basic', action='store_true', help='补充 daily_basic 缺失数据')
    parser.add_argument('--stk-factor-pro', action='store_true', help='补充 stk_factor_pro 缺失数据')
    parser.add_argument('--dc-index', action='store_true', help='补充 dc_index 缺失数据')
    parser.add_argument('--start-date', type=str, help='开始日期 (YYYYMMDD)，默认为一年前')
    parser.add_argument('--end-date', type=str, help='结束日期 (YYYYMMDD)，默认为今天')

    args = parser.parse_args()

    # 如果没有指定任何表，显示帮助
    if not (args.all or args.adj_factor or args.daily_basic or args.stk_factor_pro or args.dc_index):
        parser.print_help()
        return

    # 设置日期范围
    end_date = args.end_date or datetime.now().strftime('%Y%m%d')
    start_date = args.start_date or (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')

    logger.info("=" * 60)
    logger.info("数据补充脚本")
    logger.info(f"日期范围: {start_date} → {end_date}")
    logger.info("=" * 60)

    try:
        # 获取配置
        tushare_token, db_path = get_env_config()
        logger.info(f"数据库路径: {db_path}")

        # 初始化下载器
        downloader = DataDownloader(
            tushare_token=tushare_token,
            db_path=db_path,
            rate_limit_profile="standard"
        )

        results = {}

        try:
            # 执行补充任务
            if args.all or args.adj_factor:
                results['adj_factor'] = backfill_adj_factor(downloader, start_date, end_date)

            if args.all or args.daily_basic:
                results['daily_basic'] = backfill_daily_basic(downloader, start_date, end_date)

            if args.all or args.stk_factor_pro:
                results['stk_factor_pro'] = backfill_stk_factor_pro(downloader, start_date, end_date)

            if args.all or args.dc_index:
                results['dc_index'] = backfill_dc_index(downloader, start_date, end_date)

            # 输出汇总
            logger.info("=" * 60)
            logger.info("补充结果汇总:")
            logger.info("-" * 60)

            for table_name, stats in results.items():
                logger.info(f"  {table_name}:")
                logger.info(f"    - 总日期数: {stats['total']}")
                logger.info(f"    - 缺失日期数: {stats['missing']}")
                logger.info(f"    - 成功补充: {stats['success']}")
                logger.info(f"    - 失败: {stats['failed']}")

            logger.info("=" * 60)
            logger.info("✓ 数据补充完成!")

        finally:
            downloader.close()
            logger.info("数据库连接已关闭")

    except Exception as e:
        logger.error(f"✗ 数据补充失败: {e}")
        raise


if __name__ == "__main__":
    main()
