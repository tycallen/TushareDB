#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
回填历史三大财务报表数据（利润表、资产负债表、现金流量表）

当前数据库只有 2024Q2 之后的财报数据，需要回填 2010~2024 的历史数据。
使用 VIP 接口按报告期批量下载，每个季度一次 API 调用获取全部股票。

用法:
    # 回填 2010~2024 全部季度
    python scripts/backfill_financial_statements.py

    # 指定起止年份
    python scripts/backfill_financial_statements.py --start-year 2015 --end-year 2020

    # 只回填某张表
    python scripts/backfill_financial_statements.py --only income
    python scripts/backfill_financial_statements.py --only balancesheet
    python scripts/backfill_financial_statements.py --only cashflow

环境变量:
    TUSHARE_TOKEN: Tushare API Token（必须）
    DB_PATH: 数据库路径（可选，默认为 tushare.db）
"""

import argparse
import os
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tushare_db import DataDownloader, get_logger

logger = get_logger(__name__)


def generate_quarters(start_year: int, end_year: int) -> list:
    """生成指定年份范围内所有季度的报告期 (YYYYMMDD)"""
    quarters = []
    for year in range(start_year, end_year + 1):
        quarters.append(f"{year}0331")
        quarters.append(f"{year}0630")
        quarters.append(f"{year}0930")
        quarters.append(f"{year}1231")
    return quarters


def backfill(downloader: DataDownloader, quarters: list, only: str = None,
             sleep_sec: float = 0.5):
    """回填指定季度的财务报表数据"""
    tables = ["income", "balancesheet", "cashflow"]
    if only:
        assert only in tables, f"--only must be one of {tables}"
        tables = [only]

    totals = {t: 0 for t in tables}

    for i, period in enumerate(quarters):
        logger.info(f"[{i+1}/{len(quarters)}] 正在获取 {period} ...")

        for table in tables:
            try:
                method = getattr(downloader, f"download_{table}_vip")
                rows = method(period=period)
                totals[table] += rows
            except Exception as e:
                logger.error(f"  ✗ {table} {period} 失败: {e}")

        # API 限流保护
        time.sleep(sleep_sec)

    logger.info("=" * 50)
    logger.info("回填完成:")
    for table, total in totals.items():
        logger.info(f"  {table}: {total} 行")


def main():
    parser = argparse.ArgumentParser(description="回填历史三大财务报表")
    parser.add_argument("--start-year", type=int, default=2010)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--only", type=str, default=None,
                        choices=["income", "balancesheet", "cashflow"])
    parser.add_argument("--sleep", type=float, default=0.5,
                        help="API 调用间隔秒数")
    args = parser.parse_args()

    token = os.getenv("TUSHARE_TOKEN")
    if not token:
        print("请设置 TUSHARE_TOKEN 环境变量")
        sys.exit(1)

    db_path = os.getenv("DB_PATH", str(project_root / "tushare.db"))
    logger.info(f"数据库: {db_path}")
    logger.info(f"回填范围: {args.start_year}Q1 ~ {args.end_year}Q4")

    quarters = generate_quarters(args.start_year, args.end_year)
    logger.info(f"共 {len(quarters)} 个季度")

    downloader = DataDownloader(tushare_token=token, db_path=db_path)
    try:
        backfill(downloader, quarters, only=args.only, sleep_sec=args.sleep)
    finally:
        downloader.close()


if __name__ == "__main__":
    main()
