#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
补充缺失的 cyq_perf（筹码分布）数据

功能：
识别 cyq_perf 表中缺失的交易日，并补充下载对应的数据

使用方法：
    python scripts/backfill_cyq_perf.py

环境变量：
    TUSHARE_TOKEN: Tushare API Token（必须）
    DB_PATH: 数据库路径（可选，默认为 tushare.db）

作者：Claude Code
日期：2025-12-12
"""

import os
import sys
from datetime import datetime
from pathlib import Path
import time

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


def get_missing_dates(downloader: DataDownloader, start_date: str, end_date: str):
    """
    获取 cyq_perf 表中缺失的交易日

    Args:
        downloader: DataDownloader 实例
        start_date: 起始日期 (YYYYMMDD)
        end_date: 结束日期 (YYYYMMDD)

    Returns:
        缺失日期列表
    """
    logger.info(f"检查 {start_date} - {end_date} 期间的缺失数据...")

    # 查询缺失的交易日
    sql = """
    WITH pro_bar_dates AS (
        SELECT DISTINCT trade_date
        FROM pro_bar
        WHERE trade_date BETWEEN ? AND ?
    ),
    cyq_dates AS (
        SELECT DISTINCT trade_date
        FROM cyq_perf
        WHERE trade_date BETWEEN ? AND ?
    )
    SELECT p.trade_date as missing_date
    FROM pro_bar_dates p
    WHERE NOT EXISTS (SELECT 1 FROM cyq_dates c WHERE c.trade_date = p.trade_date)
    ORDER BY p.trade_date
    """

    missing_df = downloader.db.execute_query(sql, [start_date, end_date, start_date, end_date])

    if missing_df.empty:
        logger.info("没有发现缺失数据")
        return []

    missing_dates = missing_df['missing_date'].tolist()
    logger.info(f"发现 {len(missing_dates)} 个缺失的交易日")

    return missing_dates


def backfill_cyq_perf(downloader: DataDownloader, start_date: str = "20250101", end_date: str = "20251231"):
    """
    补充缺失的 cyq_perf 数据

    Args:
        downloader: DataDownloader 实例
        start_date: 起始日期 (YYYYMMDD)
        end_date: 结束日期 (YYYYMMDD)
    """
    logger.info("=" * 60)
    logger.info("开始补充缺失的 cyq_perf 数据...")
    logger.info(f"检查范围: {start_date} → {end_date}")

    try:
        # 1. 获取缺失日期
        missing_dates = get_missing_dates(downloader, start_date, end_date)

        if not missing_dates:
            logger.info("✓ 没有需要补充的数据")
            return

        # 2. 逐日下载
        success_count = 0
        failed_dates = []

        logger.info(f"开始下载 {len(missing_dates)} 个交易日的数据...")

        for i, trade_date in enumerate(missing_dates, 1):
            try:
                logger.info(f"[{i}/{len(missing_dates)}] 下载 {trade_date} 的数据...")
                rows = downloader.download_cyq_perf(trade_date)

                if rows > 0:
                    success_count += 1
                    logger.info(f"  ✓ 成功下载 {rows} 条记录")
                else:
                    logger.warning(f"  ⚠ {trade_date} 无数据")

                # API 限流：每次请求后暂停
                time.sleep(0.5)

            except Exception as e:
                logger.error(f"  ✗ {trade_date} 下载失败: {e}")
                failed_dates.append(trade_date)
                # 继续下一个日期
                continue

        # 3. 汇总结果
        logger.info("=" * 60)
        logger.info(f"✓ 补充完成: 成功 {success_count}/{len(missing_dates)}")

        if failed_dates:
            logger.warning(f"失败日期 ({len(failed_dates)} 个):")
            for date in failed_dates:
                logger.warning(f"  - {date}")

    except Exception as e:
        logger.error(f"✗ 补充数据失败: {e}")
        raise


def main():
    """主函数"""
    try:
        # 1. 获取配置
        tushare_token, db_path = get_env_config()
        logger.info(f"数据库路径: {db_path}")

        # 2. 初始化下载器
        downloader = DataDownloader(
            tushare_token=tushare_token,
            db_path=db_path
        )
        logger.info("下载器初始化完成")

        # 3. 补充缺失数据（2025年的数据）
        backfill_cyq_perf(downloader, start_date="20250101", end_date="20251231")

        logger.info("=" * 60)
        logger.info("全部任务完成！")

    except Exception as e:
        logger.error(f"执行失败: {e}")
        sys.exit(1)
    finally:
        # 清理资源
        if 'downloader' in locals():
            downloader.close()


if __name__ == "__main__":
    main()
