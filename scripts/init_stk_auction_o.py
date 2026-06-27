#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
股票开盘集合竞价数据初始化脚本

功能：
    批量下载 stk_auction_o（开盘集合竞价）历史数据

用法：
    python scripts/init_stk_auction_o.py --start-date 20241101 --end-date 20241231
    python scripts/init_stk_auction_o.py --start-date 20240101          # 下载到最新
    python scripts/init_stk_auction_o.py --recent 30                      # 下载最近30个交易日

环境变量：
    TUSHARE_TOKEN: Tushare API Token（必须）
    DB_PATH: 数据库路径（可选，默认 tushare.db）
"""

import os
import sys
import argparse
from datetime import datetime, timedelta
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tushare_db import DataDownloader, get_logger

logger = get_logger(__name__)


def get_trading_dates(downloader: DataDownloader, start_date: str, end_date: str) -> list:
    """获取指定范围内的交易日列表"""
    df = downloader.db.execute_query('''
        SELECT cal_date
        FROM trade_cal
        WHERE cal_date >= ? AND cal_date <= ? AND is_open = 1
        ORDER BY cal_date
    ''', [start_date, end_date])

    if df.empty:
        return []
    return df['cal_date'].tolist()


def init_stk_auction_o(start_date: str, end_date: str):
    """
    批量下载开盘集合竞价历史数据

    Args:
        start_date: 开始日期 YYYYMMDD
        end_date: 结束日期 YYYYMMDD
    """
    token = os.getenv("TUSHARE_TOKEN")
    if not token:
        raise ValueError("请设置 TUSHARE_TOKEN 环境变量")

    db_path = os.getenv("DB_PATH", str(project_root / "tushare.db"))

    downloader = DataDownloader(
        tushare_token=token,
        db_path=db_path,
        rate_limit_profile="standard"
    )

    try:
        # 检查交易日历是否存在
        if not downloader.db.table_exists('trade_cal'):
            logger.error("交易日历表不存在，请先运行: python scripts/init_data.py --trade-cal")
            return

        trading_dates = get_trading_dates(downloader, start_date, end_date)
        if not trading_dates:
            logger.info(f"期间 {start_date} → {end_date} 无交易日")
            return

        logger.info(f"=" * 60)
        logger.info(f"开始下载开盘集合竞价数据")
        logger.info(f"日期范围: {start_date} → {end_date}")
        logger.info(f"交易日数: {len(trading_dates)}")
        logger.info(f"=" * 60)

        success_count = 0
        total_rows = 0

        for i, trade_date in enumerate(trading_dates, 1):
            try:
                rows = downloader.download_stk_auction_o_by_date(trade_date)
                if rows > 0:
                    success_count += 1
                    total_rows += rows
                    logger.info(f"[{i}/{len(trading_dates)}] {trade_date}: {rows} 行")
                else:
                    logger.warning(f"[{i}/{len(trading_dates)}] {trade_date}: 无数据")
            except Exception as e:
                logger.error(f"[{i}/{len(trading_dates)}] {trade_date}: 失败 - {e}")
                continue

        logger.info(f"=" * 60)
        logger.info(f"下载完成")
        logger.info(f"  成功: {success_count}/{len(trading_dates)} 个交易日")
        logger.info(f"  总数据: {total_rows:,} 行")
        logger.info(f"=" * 60)

    finally:
        downloader.close()


def probe_earliest_date(downloader: DataDownloader) -> str:
    """
    二分探测 stk_auction_o 最早有数据的日期

    从2024-01-01开始试探，逐步向前/向后二分查找最早有数据的日期
    """
    import bisect

    logger.info("探测最早数据日期...")

    # 生成候选日期列表（从2024-01-01到今天的所有交易日）
    candidate_start = '20240101'
    today = datetime.now().strftime('%Y%m%d')

    df = downloader.db.execute_query('''
        SELECT cal_date
        FROM trade_cal
        WHERE cal_date >= ? AND cal_date <= ? AND is_open = 1
        ORDER BY cal_date
    ''', [candidate_start, today])

    if df.empty:
        logger.error("无法获取交易日历")
        return None

    dates = df['cal_date'].tolist()

    # 二分查找最早有数据的日期
    left, right = 0, len(dates) - 1
    earliest = None

    while left <= right:
        mid = (left + right) // 2
        test_date = dates[mid]

        try:
            df_test = downloader.download_stk_auction_o(trade_date=test_date)
            has_data = len(df_test) > 0
        except Exception:
            has_data = False

        if has_data:
            earliest = test_date
            right = mid - 1  # 尝试更早的日期
        else:
            left = mid + 1   # 尝试更晚的日期

    if earliest:
        logger.info(f"最早数据日期: {earliest}")
    else:
        logger.warning("未找到有数据的日期，可能接口权限未开通或数据尚未上线")

    return earliest


def main():
    parser = argparse.ArgumentParser(description="初始化开盘集合竞价数据")
    parser.add_argument("--start-date", type=str, help="开始日期 YYYYMMDD")
    parser.add_argument("--end-date", type=str, help="结束日期 YYYYMMDD，默认今天")
    parser.add_argument("--recent", type=int, help="下载最近 N 个交易日（优先于 start-date）")
    parser.add_argument("--probe-earliest", action="store_true", help="探测最早数据日期")

    args = parser.parse_args()

    if args.probe_earliest:
        token = os.getenv("TUSHARE_TOKEN")
        db_path = os.getenv("DB_PATH", str(project_root / "tushare.db"))
        downloader = DataDownloader(tushare_token=token, db_path=db_path)
        try:
            earliest = probe_earliest_date(downloader)
            if earliest:
                print(f"\n最早数据日期: {earliest}")
                print(f"建议下载命令: python scripts/init_stk_auction_o.py --start-date {earliest}")
        finally:
            downloader.close()
        return

    if args.recent:
        # 下载最近 N 个交易日
        token = os.getenv("TUSHARE_TOKEN")
        db_path = os.getenv("DB_PATH", str(project_root / "tushare.db"))
        downloader = DataDownloader(tushare_token=token, db_path=db_path)
        try:
            today = datetime.now().strftime('%Y%m%d')
            df = downloader.db.execute_query('''
                SELECT cal_date FROM trade_cal
                WHERE cal_date <= ? AND is_open = 1
                ORDER BY cal_date DESC
                LIMIT ?
            ''', [today, args.recent])
            if df.empty:
                logger.error("无法获取交易日历")
                return
            dates = sorted(df['cal_date'].tolist())
            start_date = dates[0]
            end_date = dates[-1]
        finally:
            downloader.close()
    elif args.start_date:
        start_date = args.start_date
        end_date = args.end_date or datetime.now().strftime('%Y%m%d')
    else:
        parser.print_help()
        return

    init_stk_auction_o(start_date, end_date)


if __name__ == "__main__":
    main()
