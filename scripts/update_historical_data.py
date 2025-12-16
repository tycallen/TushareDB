#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
历史数据更新脚本

功能：
1. 更新交易日历（2000-2026年完整数据）
2. 更新股票基础信息（所有状态）
3. 可选：更新历史日线数据（需要大量时间和API调用）

使用方法：
    # 仅更新交易日历和股票基础信息
    python scripts/update_historical_data.py

    # 更新包括历史日线数据（警告：耗时很长）
    python scripts/update_historical_data.py --include-daily

环境变量：
    TUSHARE_TOKEN: Tushare API Token（必须）
    DB_PATH: 数据库路径（可选，默认为 tushare.db）

作者：Claude Code
日期：2025-12-11
"""

import os
import sys
import argparse
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


def update_full_trade_calendar(downloader: DataDownloader):
    """
    更新完整交易日历（2000-2026年）

    策略：一次性获取从2000年到2026年的所有交易日历数据
    """
    logger.info("=" * 60)
    logger.info("开始更新完整交易日历（2000-2026年）...")

    start_date = '20000101'
    end_date = '20261231'

    try:
        rows = downloader.download_trade_calendar(
            start_date=start_date,
            end_date=end_date
        )
        logger.info(f"✓ 交易日历更新完成，共 {rows} 行")
        logger.info(f"  覆盖范围: {start_date} → {end_date}")
    except Exception as e:
        logger.error(f"✗ 更新交易日历失败: {e}")
        raise


def update_stock_basic(downloader: DataDownloader):
    """
    更新股票基础信息

    策略：全量更新所有状态的股票（上市、退市、暂停）
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


def update_historical_daily_data(downloader: DataDownloader, start_year: int = 2000):
    """
    更新历史日线数据（警告：耗时很长，API调用量大）

    参数：
        downloader: DataDownloader实例
        start_year: 起始年份，默认2000年

    策略：
        1. 从指定年份开始，到今天为止
        2. 逐日更新所有交易日的数据
        3. 自动跳过非交易日

    注意：
        - 这个操作可能需要数小时到数天完成
        - 需要大量API调用，请确保账号权限足够
        - 建议分批执行，避免API限流
    """
    logger.info("=" * 60)
    logger.info(f"开始更新历史日线数据（{start_year}年至今）...")
    logger.warning("⚠ 警告：这将是一个非常耗时的操作，可能需要数小时！")

    try:
        # 1. 设置时间范围
        start_date = f'{start_year}0101'
        today = datetime.now().strftime('%Y%m%d')

        logger.info(f"更新范围: {start_date} → {today}")

        # 2. 查询所有交易日
        trading_dates_df = downloader.db.execute_query('''
            SELECT cal_date
            FROM trade_cal
            WHERE cal_date >= ? AND cal_date <= ? AND is_open = 1
            ORDER BY cal_date
        ''', [start_date, today])

        if trading_dates_df.empty:
            logger.warning(f"未找到 {start_date} → {today} 期间的交易日")
            logger.info("请先运行交易日历更新")
            return

        trading_dates = trading_dates_df['cal_date'].tolist()
        total_days = len(trading_dates)

        logger.info(f"发现 {total_days} 个交易日需要更新")

        # 3. 确认是否继续
        logger.warning(f"预计需要更新 {total_days} 个交易日的数据")
        logger.warning("这可能需要数小时甚至更长时间，且会消耗大量API调用次数")

        # 4. 逐日更新
        success_count = 0
        fail_count = 0
        skip_count = 0

        for i, trade_date in enumerate(trading_dates, 1):
            try:
                # 检查是否已有数据（避免重复下载）
                existing_df = downloader.db.execute_query(
                    "SELECT COUNT(*) as cnt FROM pro_bar WHERE trade_date = ?",
                    [trade_date]
                )

                if not existing_df.empty and existing_df['cnt'].iloc[0] > 0:
                    logger.info(f"  [{i}/{total_days}] {trade_date} 已有数据，跳过")
                    skip_count += 1
                    continue

                logger.info(f"  [{i}/{total_days}] 更新 {trade_date}...")
                downloader.download_daily_data_by_date(trade_date)
                success_count += 1

                # 每100个交易日输出一次进度
                if i % 100 == 0:
                    progress = (i / total_days) * 100
                    logger.info(f"  进度: {progress:.1f}% ({i}/{total_days})")

            except Exception as e:
                logger.error(f"    ✗ 失败: {e}")
                fail_count += 1
                # 继续处理其他日期
                continue

        # 5. 汇总结果
        logger.info("=" * 60)
        logger.info(f"✓ 历史日线数据更新完成")
        logger.info(f"  - 成功: {success_count} 个交易日")
        logger.info(f"  - 跳过: {skip_count} 个交易日（已有数据）")
        if fail_count > 0:
            logger.warning(f"  - 失败: {fail_count} 个交易日")
        logger.info(f"  - 更新范围: {start_date} → {today}")

    except Exception as e:
        logger.error(f"✗ 历史日线数据更新失败: {e}")
        raise


def main():
    """
    主函数：执行历史数据更新任务
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='更新Tushare历史数据（2000-2026年）')
    parser.add_argument(
        '--include-daily',
        action='store_true',
        help='是否包括历史日线数据更新（警告：非常耗时）'
    )
    parser.add_argument(
        '--start-year',
        type=int,
        default=2000,
        help='历史数据起始年份（默认：2000）'
    )
    args = parser.parse_args()

    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info("开始历史数据更新任务")
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
            rate_limit_profile="standard"  # 根据你的账号权限调整
        )

        try:
            # 3. 执行更新任务
            # 3.1 更新完整交易日历（2000-2026年）
            update_full_trade_calendar(downloader)

            # 3.2 更新股票基础信息
            update_stock_basic(downloader)

            # 3.3 可选：更新历史日线数据
            if args.include_daily:
                logger.info("=" * 60)
                logger.info("将要开始更新历史日线数据...")
                logger.warning("⚠ 这个操作可能需要数小时到数天完成")
                logger.warning("⚠ 建议先测试小范围数据，确认无误后再全量更新")

                # 给用户5秒思考时间
                import time
                for i in range(5, 0, -1):
                    logger.info(f"  {i} 秒后开始...")
                    time.sleep(1)

                update_historical_daily_data(downloader, start_year=args.start_year)
            else:
                logger.info("=" * 60)
                logger.info("跳过历史日线数据更新")
                logger.info("如需更新历史日线数据，请使用 --include-daily 参数")

            # 4. 完成
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            logger.info("=" * 60)
            logger.info("✓ 历史数据更新任务完成！")
            logger.info(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"总耗时: {duration:.1f} 秒 ({duration/60:.1f} 分钟)")
            logger.info("=" * 60)

        finally:
            # 5. 关闭连接
            downloader.close()
            logger.info("数据库连接已关闭")

    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"✗ 历史数据更新任务失败: {e}")
        logger.error("=" * 60)
        raise


if __name__ == "__main__":
    main()
