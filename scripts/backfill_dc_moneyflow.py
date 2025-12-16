#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
龙虎榜和资金流向数据回填脚本

功能：
1. 回填 dc_member（龙虎榜机构席位交易明细）历史数据
2. 回填 moneyflow_ind_dc（行业资金流向）历史数据

策略：从新到旧往回填充，直到：
    - 达到指定的回填天数
    - 或达到指定的起始日期
    - 或 API 返回空数据（说明没有更早的数据）

使用方法：
    # 回填最近365天的数据
    python scripts/backfill_dc_moneyflow.py --days 365

    # 回填到指定日期（例如从2020年开始）
    python scripts/backfill_dc_moneyflow.py --start-date 20200101

    # 只回填龙虎榜数据
    python scripts/backfill_dc_moneyflow.py --days 365 --table dc_member

    # 只回填资金流向数据
    python scripts/backfill_dc_moneyflow.py --days 365 --table moneyflow_ind_dc

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


def backfill_dc_member(downloader: DataDownloader, start_date: str, end_date: str):
    """
    回填龙虎榜机构席位交易明细数据

    Args:
        downloader: DataDownloader 实例
        start_date: 回填起始日期（早）
        end_date: 回填结束日期（晚）
    """
    logger.info("=" * 60)
    logger.info("开始回填 dc_member（龙虎榜机构席位）数据...")
    logger.info(f"回填范围: {start_date} → {end_date}")

    try:
        # 1. 获取这段时间内的所有交易日
        trading_dates_df = downloader.db.execute_query('''
            SELECT cal_date
            FROM trade_cal
            WHERE cal_date >= ? AND cal_date <= ? AND is_open = 1
            ORDER BY cal_date DESC
        ''', [start_date, end_date])

        if trading_dates_df.empty:
            logger.warning(f"期间 {start_date} → {end_date} 没有交易日")
            return

        trading_dates = trading_dates_df['cal_date'].tolist()
        total_dates = len(trading_dates)
        logger.info(f"发现 {total_dates} 个交易日需要回填")

        # 2. 逐日回填（从新到旧）
        success_count = 0
        skip_count = 0
        fail_count = 0
        empty_count = 0

        for i, trade_date in enumerate(trading_dates, 1):
            try:
                # 检查是否已有数据
                existing_df = downloader.db.execute_query(
                    "SELECT COUNT(*) as cnt FROM dc_member WHERE trade_date = ?",
                    [trade_date]
                )

                if not existing_df.empty and existing_df['cnt'].iloc[0] > 0:
                    logger.debug(f"  [{i}/{total_dates}] {trade_date} 已有数据，跳过")
                    skip_count += 1
                    continue

                # 下载数据
                logger.info(f"  [{i}/{total_dates}] 回填 {trade_date}...")
                rows = downloader.download_dc_member(trade_date)

                if rows > 0:
                    success_count += 1
                    logger.info(f"    ✓ 成功: {rows} 行")
                else:
                    empty_count += 1
                    logger.debug(f"    - 无数据")

                # 每50个交易日输出一次进度
                if i % 50 == 0:
                    progress = (i / total_dates) * 100
                    logger.info(f"  进度: {progress:.1f}% ({i}/{total_dates}) - 成功:{success_count} 跳过:{skip_count} 空:{empty_count}")

            except Exception as e:
                logger.error(f"    ✗ 失败: {e}")
                fail_count += 1
                continue

        # 3. 汇总结果
        logger.info("=" * 60)
        logger.info(f"✓ dc_member 回填完成")
        logger.info(f"  - 成功: {success_count} 个交易日")
        logger.info(f"  - 跳过: {skip_count} 个交易日（已有数据）")
        logger.info(f"  - 空数据: {empty_count} 个交易日")
        if fail_count > 0:
            logger.warning(f"  - 失败: {fail_count} 个交易日")

    except Exception as e:
        logger.error(f"✗ dc_member 回填失败: {e}")
        raise


def backfill_moneyflow_ind_dc(downloader: DataDownloader, start_date: str, end_date: str):
    """
    回填行业资金流向（沪深通）数据

    Args:
        downloader: DataDownloader 实例
        start_date: 回填起始日期（早）
        end_date: 回填结束日期（晚）
    """
    logger.info("=" * 60)
    logger.info("开始回填 moneyflow_ind_dc（行业资金流向）数据...")
    logger.info(f"回填范围: {start_date} → {end_date}")

    try:
        # 1. 获取这段时间内的所有交易日
        trading_dates_df = downloader.db.execute_query('''
            SELECT cal_date
            FROM trade_cal
            WHERE cal_date >= ? AND cal_date <= ? AND is_open = 1
            ORDER BY cal_date DESC
        ''', [start_date, end_date])

        if trading_dates_df.empty:
            logger.warning(f"期间 {start_date} → {end_date} 没有交易日")
            return

        trading_dates = trading_dates_df['cal_date'].tolist()
        total_dates = len(trading_dates)
        logger.info(f"发现 {total_dates} 个交易日需要回填")

        # 2. 逐日回填（从新到旧）
        success_count = 0
        skip_count = 0
        fail_count = 0
        empty_count = 0

        for i, trade_date in enumerate(trading_dates, 1):
            try:
                # 检查是否已有数据
                existing_df = downloader.db.execute_query(
                    "SELECT COUNT(*) as cnt FROM moneyflow_ind_dc WHERE trade_date = ?",
                    [trade_date]
                )

                if not existing_df.empty and existing_df['cnt'].iloc[0] > 0:
                    logger.debug(f"  [{i}/{total_dates}] {trade_date} 已有数据，跳过")
                    skip_count += 1
                    continue

                # 下载数据
                logger.info(f"  [{i}/{total_dates}] 回填 {trade_date}...")
                rows = downloader.download_moneyflow_ind_dc(trade_date)

                if rows > 0:
                    success_count += 1
                    logger.info(f"    ✓ 成功: {rows} 行")
                else:
                    empty_count += 1
                    logger.debug(f"    - 无数据")

                # 每50个交易日输出一次进度
                if i % 50 == 0:
                    progress = (i / total_dates) * 100
                    logger.info(f"  进度: {progress:.1f}% ({i}/{total_dates}) - 成功:{success_count} 跳过:{skip_count} 空:{empty_count}")

            except Exception as e:
                logger.error(f"    ✗ 失败: {e}")
                fail_count += 1
                continue

        # 3. 汇总结果
        logger.info("=" * 60)
        logger.info(f"✓ moneyflow_ind_dc 回填完成")
        logger.info(f"  - 成功: {success_count} 个交易日")
        logger.info(f"  - 跳过: {skip_count} 个交易日（已有数据）")
        logger.info(f"  - 空数据: {empty_count} 个交易日")
        if fail_count > 0:
            logger.warning(f"  - 失败: {fail_count} 个交易日")

    except Exception as e:
        logger.error(f"✗ moneyflow_ind_dc 回填失败: {e}")
        raise


def main():
    """
    主函数：执行历史数据回填任务
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='回填龙虎榜和资金流向历史数据（从新到旧）')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--days',
        type=int,
        help='从今天往回回填的天数（例如：365）'
    )
    group.add_argument(
        '--start-date',
        type=str,
        help='回填起始日期 YYYYMMDD（例如：20200101），将从今天回填到该日期'
    )

    parser.add_argument(
        '--table',
        choices=['dc_member', 'moneyflow_ind_dc', 'both'],
        default='both',
        help='要回填的表：dc_member（龙虎榜）、moneyflow_ind_dc（资金流向）或 both（默认）'
    )

    args = parser.parse_args()

    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info("开始历史数据回填任务（从新到旧）")
    logger.info(f"启动时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    try:
        # 1. 获取配置
        tushare_token, db_path = get_env_config()
        logger.info(f"数据库路径: {db_path}")

        # 2. 计算日期范围
        end_date = datetime.now().strftime('%Y%m%d')

        if args.days:
            start_date = (datetime.now() - timedelta(days=args.days)).strftime('%Y%m%d')
            logger.info(f"回填范围: 最近 {args.days} 天 ({start_date} → {end_date})")
        else:
            start_date = args.start_date
            # 验证日期格式
            try:
                datetime.strptime(start_date, '%Y%m%d')
            except ValueError:
                logger.error(f"日期格式错误: {start_date}，应为 YYYYMMDD")
                return
            logger.info(f"回填范围: {start_date} → {end_date}")

        # 3. 初始化下载器
        downloader = DataDownloader(
            tushare_token=tushare_token,
            db_path=db_path,
            rate_limit_profile="standard"
        )

        try:
            # 4. 执行回填任务
            if args.table in ['dc_member', 'both']:
                backfill_dc_member(downloader, start_date, end_date)

            if args.table in ['moneyflow_ind_dc', 'both']:
                backfill_moneyflow_ind_dc(downloader, start_date, end_date)

            # 5. 完成
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            logger.info("=" * 60)
            logger.info("✓ 历史数据回填任务完成！")
            logger.info(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"总耗时: {duration:.1f} 秒 ({duration/60:.1f} 分钟)")
            logger.info("=" * 60)

        finally:
            # 6. 关闭连接
            downloader.close()
            logger.info("数据库连接已关闭")

    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"✗ 历史数据回填任务失败: {e}")
        logger.error("=" * 60)
        raise


if __name__ == "__main__":
    main()
