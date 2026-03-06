#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
每日数据更新脚本（新架构版本）

功能：
1. 更新交易日历
2. 更新股票列表
3. 更新申万行业分类（仅初始化时下载）
4. 更新同花顺板块指数（仅初始化时下载）
5. 更新所有股票的日线数据、复权因子、每日基本面
6. 更新财务指标数据（VIP接口）
7. 更新三大财务报表（利润表、资产负债表、现金流量表）
8. 更新分红送股数据
9. 更新融资融券交易明细
10. 更新筹码分布数据 (cyq_perf)
11. 更新筹码分布详情 (cyq_chips) - 各价位占比
12. 更新技术因子 (stk_factor_pro) - MACD、KDJ、RSI等
13. 更新龙虎榜机构席位数据
14. 更新行业资金流向（沪深通）数据
15. 更新个股资金流向数据
16. 更新申万行业指数日线数据
17. 更新同花顺板块日行情 (ths_daily)
18. 更新开盘啦题材库 (kpl_concept, kpl_concept_cons)

使用方法：
    python scripts/update_daily.py

环境变量：
    TUSHARE_TOKEN: Tushare API Token（必须）
    DB_PATH: 数据库路径（可选，默认为 tushare.db）

作者：根据新架构重构
日期：2025-12-04
更新：2025-12-11 - 添加龙虎榜和资金流向数据更新
更新：2025-12-26 - 添加申万行业分类和个股资金流向更新
更新：2026-01-19 - 添加财务报表、分红送股、融资融券明细更新
更新：2026-01-31 - 添加申万行业指数日线数据更新
更新：2026-02-03 - 添加筹码分布详情、技术因子、开盘啦题材库更新
更新：2026-02-05 - 添加同花顺板块指数、成分、日行情更新
更新：2026-02-14 - 添加基金模块和沪深港通数据更新
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def _ensure_table_primary_keys(downloader: DataDownloader):
    """
    检查并修复缺失主键的表（一次性操作）

    这是为了修复在 TABLE_PRIMARY_KEYS 添加主键定义之前创建的表。
    修复后的表将具有正确的主键约束，支持 UPSERT 操作。

    注意：stk_factor_pro 表数据量较大（1400万+行），重建可能需要一些时间，
    但 DuckDB 能够高效处理，且这只是一次性操作。
    """
    # 需要检查的表（已知可能缺失主键的表）
    tables_to_check = [
        'fina_indicator_vip',
        'cyq_perf',
        'moneyflow_ind_dc',
        'stk_factor_pro',  # 技术因子表，数据量大但需要主键支持 UPSERT
    ]

    rebuilt_count = 0
    for table_name in tables_to_check:
        try:
            if downloader.db.ensure_primary_key(table_name):
                rebuilt_count += 1
        except Exception as e:
            logger.warning(f"修复表 {table_name} 主键时出错: {e}")
            # 不阻塞主流程

    if rebuilt_count > 0:
        logger.info(f"已修复 {rebuilt_count} 个表的主键约束")


def update_trade_calendar(downloader: DataDownloader):
    """
    更新交易日历

    策略：每次更新完整的交易日历（2000年至未来2年），确保数据完整性
    由于底层使用 mode='replace'，必须每次获取完整范围以避免历史数据丢失
    """
    logger.info("=" * 60)
    logger.info("开始更新交易日历...")

    start_date = '20000101'  # 从2000年开始
    two_years_later = (datetime.now() + timedelta(days=730)).strftime('%Y%m%d')

    try:
        rows = downloader.download_trade_calendar(
            start_date=start_date,
            end_date=two_years_later
        )
        logger.info(f"✓ 交易日历更新完成，更新 {rows} 行")
        logger.info(f"  覆盖范围: {start_date} → {two_years_later}")
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
        1. 所有股票的日线数据（daily）
        2. 所有股票的复权因子（adj_factor）
        3. 所有股票的每日基本面（daily_basic）
    """
    logger.info("=" * 60)
    logger.info("开始智能增量更新每日数据...")
    
    try:
        # 1. 获取数据库中最新的交易日期
        latest_date = downloader.db.get_latest_date('daily', 'trade_date')
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


def update_index_daily(downloader: DataDownloader):
    """
    更新常见指数日线数据 (index_daily)

    常见指数包括：
    - 000001.SH 上证指数
    - 399001.SZ 深证成指
    - 399006.SZ 创业板指
    - 000300.SH 沪深300
    - 000905.SH 中证500
    - 000852.SH 中证1000
    - 000688.SH 科创50
    - 000016.SH 上证50

    数据存储在 index_daily 表，使用 Tushare index_daily 接口
    """
    logger.info("=" * 60)
    logger.info("开始更新常见指数日线数据 (index_daily)...")

    # 常见指数列表
    indices = [
        ('000001.SH', '上证指数'),
        ('399001.SZ', '深证成指'),
        ('399006.SZ', '创业板指'),
        ('000300.SH', '沪深300'),
        ('000905.SH', '中证500'),
        ('000852.SH', '中证1000'),
        ('000688.SH', '科创50'),
        ('000016.SH', '上证50'),
    ]

    try:
        today = datetime.now().strftime('%Y%m%d')
        total_rows = 0

        # 检查表是否存在
        table_exists = downloader.db.table_exists('index_daily')
        if not table_exists:
            logger.info("  index_daily 表不存在，将进行初始化下载")

        for ts_code, name in indices:
            try:
                start_date = '20100101'  # 默认起始日期

                # 如果表存在，获取该指数的最新日期
                if table_exists:
                    latest_date_df = downloader.db.execute_query(
                        "SELECT MAX(trade_date) as max_date FROM index_daily WHERE ts_code = ?",
                        [ts_code]
                    )
                    if not latest_date_df.empty and latest_date_df.iloc[0]['max_date']:
                        latest_date = str(latest_date_df.iloc[0]['max_date'])
                        latest_dt = datetime.strptime(latest_date, '%Y%m%d')
                        start_date = (latest_dt + timedelta(days=1)).strftime('%Y%m%d')

                if start_date > today:
                    logger.info(f"  {name} ({ts_code}): 已是最新，跳过")
                    continue

                logger.info(f"  更新 {name} ({ts_code}): {start_date} -> {today}")
                rows = downloader.download_index_daily(
                    ts_code=ts_code,
                    start_date=start_date,
                    end_date=today
                )
                if rows > 0:
                    logger.info(f"    ✓ 下载 {rows} 行")
                    total_rows += rows
                else:
                    logger.info(f"    - 无新数据")

            except Exception as e:
                logger.error(f"  ✗ 更新 {name} ({ts_code}) 失败: {e}")
                continue

        logger.info(f"✓ 常见指数数据更新完成，总计 {total_rows} 行")

    except Exception as e:
        logger.error(f"✗ 更新常见指数数据失败: {e}")
        # 不阻塞主流程
        logger.warning("  继续执行其他更新任务...")


def update_financial_indicators(downloader: DataDownloader):
    """
    更新财务指标数据（VIP接口）

    策略：
        使用 VIP 接口按报告期批量下载全部股票的财务指标
        - 财报季（1-4/7-8/10-11月）：每日更新，捕获新披露财报
        - 非财报季（5-6/9/12月）：每周更新，数据几乎无变化
        - 需要5000积分
    """
    logger.info("=" * 60)

    try:
        import time as _time

        if not _is_earnings_season():
            last_update = downloader.db.get_cache_metadata('fina_indicator_vip_update')
            if last_update is not None:
                days_since = (_time.time() - last_update) / 86400
                if days_since < 7:
                    logger.info(f"非财报季，距上次更新仅 {days_since:.1f} 天，跳过财务指标（每周更新）")
                    return

        logger.info("开始更新财务指标数据（VIP批量模式）...")
        quarters = _generate_recent_quarters(8)
        logger.info(f"将更新 {len(quarters)} 个季度的财务指标: {quarters}")

        total_rows = 0
        for period in quarters:
            try:
                rows = downloader.download_fina_indicator_vip(period=period)
                total_rows += rows
            except Exception as e:
                logger.error(f"  ✗ {period} 更新失败: {e}")
                continue

        downloader.db.update_cache_metadata('fina_indicator_vip_update', _time.time())
        logger.info(f"✓ 财务指标数据更新完成: 共 {total_rows} 行")

    except Exception as e:
        logger.error(f"✗ 更新财务指标数据失败: {e}")
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


def update_cyq_chips(downloader: DataDownloader):
    """
    增量更新筹码分布详情数据 (cyq_chips)

    策略（周更新）：
        1. 使用元数据表记录上次更新时间
        2. 如果距离上次更新 < 7天，跳过（周更新策略）
        3. 否则获取期间的交易日列表，按日期逐日下载

    性能说明：
        - cyq_chips 是更新耗时最长的表（~5500次API调用，限速200/min，约27分钟）
        - 筹码分布变化缓慢，周级粒度足够支撑技术分析
        - 改为周更新后，每日更新总耗时从 ~30分钟 降至 ~3分钟
    """
    logger.info("=" * 60)
    logger.info("开始检查筹码分布详情数据 (cyq_chips, 周更新)...")

    try:
        import time as _time

        # 1. 检查上次更新时间（周更新策略）
        last_update = downloader.db.get_cache_metadata('cyq_chips')
        if last_update is not None:
            days_since_update = (_time.time() - last_update) / 86400
            if days_since_update < 7:
                logger.info(f"  距离上次更新仅 {days_since_update:.1f} 天，跳过（每周更新一次）")
                return
            else:
                logger.info(f"  距离上次更新 {days_since_update:.1f} 天，开始更新...")

        # 2. 获取最新日期
        latest_date = downloader.db.get_latest_date('cyq_chips', 'trade_date')
        today = datetime.now().strftime('%Y%m%d')

        if latest_date is None:
            start_date = '20180101'
            logger.info("数据库中没有筹码分布详情历史数据，将进行完整初始化")
            logger.info(f"  (数据起始日期: {start_date})")
            logger.warning("  提示: 完整初始化数据量很大，可能需要较长时间")
        else:
            latest_dt = datetime.strptime(latest_date, '%Y%m%d')
            start_date = (latest_dt + timedelta(days=1)).strftime('%Y%m%d')
            logger.info(f"数据库最新日期: {latest_date}")

        logger.info(f"更新范围: {start_date} → {today}")

        if start_date > today:
            logger.info("无需更新")
            return

        # 3. 获取交易日列表
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

        # 4. 按交易日逐日下载
        total_rows = 0
        for trade_date in trading_dates:
            try:
                rows = downloader.download_cyq_chips_by_date(trade_date)
                total_rows += rows
            except Exception as e:
                logger.error(f"  ✗ {trade_date} 更新失败: {e}")

        # 4. 更新元数据
        downloader.db.update_cache_metadata('cyq_chips', _time.time())

        logger.info(f"✓ 筹码分布详情更新完成: 共 {total_rows} 行")

    except Exception as e:
        logger.error(f"✗ 更新筹码分布详情数据失败: {e}")
        logger.warning("  继续执行其他更新任务...")


def update_stk_factor_pro(downloader: DataDownloader):
    """
    增量更新技术因子数据 (stk_factor_pro)

    策略：
        1. 获取数据库中 stk_factor_pro 的最新日期
        2. 从下一天开始更新到今天
        3. 按日期批量下载所有股票的技术因子

    注意：
        - 包含 MACD、KDJ、RSI、BOLL 等技术指标
        - 需要 5000+ 积分
    """
    logger.info("=" * 60)
    logger.info("开始增量更新技术因子数据 (stk_factor_pro)...")

    try:
        # 1. 获取最新日期
        latest_date = downloader.db.get_latest_date('stk_factor_pro', 'trade_date')
        today = datetime.now().strftime('%Y%m%d')

        if latest_date is None:
            # 技术因子覆盖全历史，但为避免初始化太慢，默认从近期开始
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
            logger.info("数据库中没有技术因子历史数据，将从近30天开始初始化")
            logger.info(f"  (数据起始日期: {start_date})")
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
        total_rows = 0
        for trade_date in trading_dates:
            try:
                rows = downloader.download_stk_factor_pro_by_date(trade_date)
                if rows > 0:
                    success_count += 1
                    total_rows += rows
                    logger.info(f"  ✓ {trade_date}: {rows} 行")
            except Exception as e:
                logger.error(f"  ✗ {trade_date} 更新失败: {e}")
                # 不中断，继续下一个

        logger.info(f"✓ 技术因子更新完成: 成功 {success_count}/{len(trading_dates)}, 共 {total_rows} 行")

    except Exception as e:
        logger.error(f"✗ 更新技术因子数据失败: {e}")
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


def update_dc_index(downloader: DataDownloader):
    """
    增量更新龙虎榜每日上榜个股明细数据 (dc_index)

    策略：
        1. 获取数据库中 dc_index 的最新日期
        2. 从下一天开始更新到今天
    """
    logger.info("=" * 60)
    logger.info("开始增量更新龙虎榜个股明细数据 (dc_index)...")

    try:
        # 1. 获取最新日期
        latest_date = downloader.db.get_latest_date('dc_index', 'trade_date')
        today = datetime.now().strftime('%Y%m%d')

        if latest_date is None:
            logger.info("数据库中没有龙虎榜个股明细历史数据，将从30天前开始更新")
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
                rows = downloader.download_dc_index(trade_date)
                if rows > 0:
                    success_count += 1
            except Exception as e:
                logger.error(f"  ✗ {trade_date} 更新失败: {e}")
                # 不中断，继续下一个

        logger.info(f"✓ 龙虎榜个股明细更新完成: 成功 {success_count}/{len(trading_dates)}")

    except Exception as e:
        logger.error(f"✗ 更新龙虎榜个股明细数据失败: {e}")
        logger.warning("  继续执行其他更新任务...")


def update_limit_list_d(downloader: DataDownloader):
    """
    增量更新涨跌停和炸板数据 (limit_list_d)

    策略：
        1. 获取数据库中 limit_list_d 的最新日期
        2. 从下一天开始更新到今天

    数据说明：
        - 数据从2020年开始
        - 包含涨停(U)、跌停(D)、炸板(Z)三种类型
        - 不包含ST股票
        - 需要 5000+ 积分
    """
    logger.info("=" * 60)
    logger.info("开始增量更新涨跌停数据 (limit_list_d)...")

    try:
        # 1. 获取最新日期
        latest_date = downloader.db.get_latest_date('limit_list_d', 'trade_date')
        today = datetime.now().strftime('%Y%m%d')

        if latest_date is None:
            # 数据从2020年开始
            start_date = '20200101'
            logger.info("数据库中没有涨跌停历史数据，将进行完整初始化")
            logger.info(f"  (数据起始日期: {start_date})")
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
        total_rows = 0
        for trade_date in trading_dates:
            try:
                rows = downloader.download_limit_list_d(trade_date=trade_date)
                if rows > 0:
                    success_count += 1
                    total_rows += rows
            except Exception as e:
                logger.error(f"  ✗ {trade_date} 更新失败: {e}")
                # 不中断，继续下一个

        logger.info(f"✓ 涨跌停数据更新完成: 成功 {success_count}/{len(trading_dates)}, 共 {total_rows} 行")

    except Exception as e:
        logger.error(f"✗ 更新涨跌停数据失败: {e}")
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


def update_moneyflow_dc(downloader: DataDownloader):
    """
    增量更新个股资金流向数据 (moneyflow_dc)

    策略：
        1. 获取数据库中 moneyflow_dc 的最新日期
        2. 从下一天开始更新到今天
    """
    logger.info("=" * 60)
    logger.info("开始增量更新个股资金流向数据 (moneyflow_dc)...")

    try:
        # 1. 获取最新日期
        latest_date = downloader.db.get_latest_date('moneyflow_dc', 'trade_date')
        today = datetime.now().strftime('%Y%m%d')

        if latest_date is None:
            # moneyflow_dc 数据开始于 2023-09-11
            start_date = '20230911'
            logger.info("数据库中没有个股资金流向历史数据，将进行完整初始化")
            logger.info(f"  (数据起始日期: {start_date})")
            logger.warning("  提示: 完整初始化可能需要较长时间，建议使用 scripts/init_moneyflow_dc_history.py")
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
                rows = downloader.download_moneyflow_dc(trade_date=trade_date)
                if rows > 0:
                    success_count += 1
                    logger.info(f"  ✓ {trade_date}: {rows} 行")
            except Exception as e:
                logger.error(f"  ✗ {trade_date} 更新失败: {e}")
                # 不中断，继续下一个

        logger.info(f"✓ 个股资金流向更新完成: 成功 {success_count}/{len(trading_dates)}")

    except Exception as e:
        logger.error(f"✗ 更新个股资金流向数据失败: {e}")
        logger.warning("  继续执行其他更新任务...")


def update_moneyflow(downloader: DataDownloader):
    """
    增量更新个股资金流向数据 (moneyflow 标准接口)

    策略：
        1. 获取数据库中 moneyflow 的最新日期
        2. 从下一天开始更新到今天
    """
    logger.info("=" * 60)
    logger.info("开始增量更新个股资金流向数据 (moneyflow)...")

    try:
        # 1. 获取最新日期
        latest_date = downloader.db.get_latest_date('moneyflow', 'trade_date')
        today = datetime.now().strftime('%Y%m%d')

        if latest_date is None:
            # moneyflow 数据开始于 2010年
            start_date = '20100101'
            logger.info("数据库中没有个股资金流向历史数据，将进行完整初始化")
            logger.info(f"  (数据起始日期: {start_date})")
            logger.warning("  提示: 完整初始化需要较长时间，从2010年开始下载所有历史数据")
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
                rows = downloader.download_moneyflow(trade_date=trade_date)
                if rows > 0:
                    success_count += 1
                    logger.info(f"  ✓ {trade_date}: {rows} 行")
            except Exception as e:
                logger.error(f"  ✗ {trade_date} 更新失败: {e}")
                # 不中断，继续下一个

        logger.info(f"✓ 个股资金流向更新完成: 成功 {success_count}/{len(trading_dates)}")

    except Exception as e:
        logger.error(f"✗ 更新个股资金流向数据失败: {e}")
        logger.warning("  继续执行其他更新任务...")


def update_index_classify(downloader: DataDownloader):
    """
    更新申万行业分类数据 (index_classify)

    策略：
        1. 检查数据库中是否已有数据
        2. 如果没有数据，下载所有版本和层级的数据
        3. 如果已有数据，跳过（行业分类数据相对稳定，不需要频繁更新）

    注意：行业分类数据相对静态，通常只需要初始化时下载一次
    """
    logger.info("=" * 60)
    logger.info("开始检查申万行业分类数据 (index_classify)...")

    try:
        # 检查是否已有数据
        if downloader.db.table_exists('index_classify'):
            result = downloader.db.execute_query("SELECT COUNT(*) as count FROM index_classify")
            if not result.empty and result.iloc[0]['count'] > 0:
                logger.info(f"数据库中已有 {result.iloc[0]['count']} 条行业分类数据，跳过更新")
                return

        logger.info("数据库中没有行业分类数据，开始初始化...")

        total_rows = 0

        # 下载申万2021版所有层级
        logger.info("  下载申万2021版...")
        for level in ['L1', 'L2', 'L3']:
            rows = downloader.download_index_classify(level=level, src='SW2021')
            logger.info(f"    - {level}: {rows} 条")
            total_rows += rows

        # 下载申万2014版所有层级
        logger.info("  下载申万2014版...")
        for level in ['L1', 'L2', 'L3']:
            rows = downloader.download_index_classify(level=level, src='SW2014')
            logger.info(f"    - {level}: {rows} 条")
            total_rows += rows

        logger.info(f"✓ 申万行业分类初始化完成: 总计 {total_rows} 条")

    except Exception as e:
        logger.error(f"✗ 更新申万行业分类数据失败: {e}")
        logger.warning("  继续执行其他更新任务...")


def update_index_member_all(downloader: DataDownloader):
    """
    更新申万行业成分构成数据 (index_member_all)

    策略：
        1. 使用元数据表记录上次更新时间
        2. 如果距离上次更新 < 7天，跳过更新（周更新策略）
        3. 否则下载所有L2和L3行业的成分股（完整历史）
        4. UPSERT机制会自动处理新增/更新，同时保留历史记录

    注意：
        - index_member_all 包含完整历史记录（支持回测）
        - 每周更新一次以获取最新的成分变动
        - 每条记录包含 in_date 和 out_date，可精确定位历史成分
    """
    logger.info("=" * 60)
    logger.info("开始更新申万行业成分构成数据 (index_member_all)...")

    try:
        # 1. 检查上次更新时间（使用元数据表）
        import time
        last_update = downloader.db.get_cache_metadata('index_member_all')
        if last_update is not None:
            days_since_update = (time.time() - last_update) / 86400  # 转换为天数
            if days_since_update < 7:
                logger.info(f"  距离上次更新仅 {days_since_update:.1f} 天，跳过（每周更新一次）")
                return
            else:
                logger.info(f"  距离上次更新 {days_since_update:.1f} 天，开始更新...")
        else:
            # 首次运行，检查表是否已有数据
            if downloader.db.table_exists('index_member_all'):
                result = downloader.db.execute_query("SELECT COUNT(*) as count FROM index_member_all")
                if not result.empty and result.iloc[0]['count'] > 0:
                    logger.info(f"  数据库中已有 {result.iloc[0]['count']} 条记录，但无更新记录")
                else:
                    logger.info("  数据库中没有申万行业成分数据，开始初始化...")
            else:
                logger.info("  数据库中没有申万行业成分数据，开始初始化...")

        # 2. 获取所有申万L2和L3行业指数代码
        industries_df = downloader.db.execute_query(
            "SELECT index_code, industry_name, level FROM index_classify "
            "WHERE src = 'SW2021' AND level IN ('L2', 'L3') "
            "ORDER BY level, index_code"
        )

        if industries_df.empty:
            logger.warning("  未找到申万行业分类，跳过")
            logger.warning("  提示：请先运行 update_index_classify() 下载行业分类数据")
            return

        logger.info(f"  共 {len(industries_df)} 个申万行业（L2+L3）")

        # 3. 逐个行业下载成分股数据（不指定is_new，下载完整历史）
        success_count = 0
        total_rows = 0

        for idx, row in industries_df.iterrows():
            index_code = row['index_code']  # 使用 index_code(如 801780.SI)而非 industry_code
            industry_name = row['industry_name']
            level = row['level']

            try:
                # 根据level决定使用哪个参数
                if level == 'L2':
                    rows = downloader.download_index_member_all(l2_code=index_code)
                else:  # L3
                    rows = downloader.download_index_member_all(l3_code=index_code)

                if rows > 0:
                    success_count += 1
                    total_rows += rows
                    logger.info(f"  [{idx+1}/{len(industries_df)}] {level} {industry_name} ({index_code}): {rows} 条")

            except Exception as e:
                logger.error(f"  [{idx+1}/{len(industries_df)}] {level} {industry_name} ({index_code}): 失败 - {e}")
                # 继续下载其他行业
                continue

        logger.info(f"✓ 申万行业成分构成更新完成: 成功 {success_count}/{len(industries_df)}, 总计 {total_rows} 条")

        # 更新元数据，记录本次更新时间
        downloader.db.update_cache_metadata('index_member_all', time.time())

    except Exception as e:
        logger.error(f"✗ 更新申万行业成分构成数据失败: {e}")
        logger.warning("  继续执行其他更新任务...")


def update_financial_statements(downloader: DataDownloader):
    """
    更新三大财务报表数据（利润表、资产负债表、现金流量表）

    策略：
        使用 VIP 接口按报告期批量下载全部股票数据（需要5000积分）
        - 财报季（1-4/7-8/10-11月）：每日更新，捕获新披露财报
        - 非财报季（5-6/9/12月）：每周更新，数据几乎无变化

    说明：
        - income_vip / balancesheet_vip / cashflow_vip 按报告期批量获取
        - UPSERT 机制确保数据不会重复
    """
    logger.info("=" * 60)

    try:
        import time as _time

        if not _is_earnings_season():
            last_update = downloader.db.get_cache_metadata('financial_statements_update')
            if last_update is not None:
                days_since = (_time.time() - last_update) / 86400
                if days_since < 7:
                    logger.info(f"非财报季，距上次更新仅 {days_since:.1f} 天，跳过三大报表（每周更新）")
                    return

        logger.info("开始更新三大财务报表数据（VIP批量模式）...")
        quarters = _generate_recent_quarters(8)
        logger.info(f"将更新 {len(quarters)} 个季度的财务报表: {quarters}")

        income_total = 0
        balance_total = 0
        cashflow_total = 0

        for period in quarters:
            logger.info(f"  正在获取 {period} 的财务报表...")

            try:
                rows = downloader.download_income_vip(period=period)
                income_total += rows

                rows = downloader.download_balancesheet_vip(period=period)
                balance_total += rows

                rows = downloader.download_cashflow_vip(period=period)
                cashflow_total += rows

            except Exception as e:
                logger.error(f"    ✗ {period} 更新失败: {e}")
                continue

        downloader.db.update_cache_metadata('financial_statements_update', _time.time())
        logger.info(f"✓ 三大财务报表更新完成:")
        logger.info(f"  - 利润表: {income_total} 行")
        logger.info(f"  - 资产负债表: {balance_total} 行")
        logger.info(f"  - 现金流量表: {cashflow_total} 行")

    except Exception as e:
        logger.error(f"✗ 更新财务报表数据失败: {e}")
        logger.warning("  继续执行其他更新任务...")


def update_dividend(downloader: DataDownloader):
    """
    更新分红送股数据

    策略：
        按公告日期批量下载最近7天的分红数据
        - dividend 接口支持按 ann_date 批量获取当日所有公告
        - 7天窗口足以捕获绝大多数分红公告（原60天过于冗余）
        - UPSERT 机制确保数据不会重复
    """
    logger.info("=" * 60)
    logger.info("开始更新分红送股数据（按公告日期批量模式）...")

    try:
        today = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=7)).strftime('%Y%m%d')

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
        logger.info(f"将查询 {len(trading_dates)} 个交易日的分红公告")

        total_rows = 0
        for ann_date in trading_dates:
            try:
                rows = downloader.download_dividend(ann_date=ann_date)
                if rows > 0:
                    total_rows += rows

            except Exception as e:
                logger.error(f"  {ann_date} 更新失败: {e}")
                continue

        logger.info(f"✓ 分红送股数据更新完成: 共 {total_rows} 条记录")

    except Exception as e:
        logger.error(f"✗ 更新分红送股数据失败: {e}")
        logger.warning("  继续执行其他更新任务...")


def update_margin_detail(downloader: DataDownloader):
    """
    增量更新融资融券交易明细数据 (margin_detail)

    策略：
        1. 获取数据库中 margin_detail 的最新日期
        2. 从下一天开始更新到今天
    """
    logger.info("=" * 60)
    logger.info("开始增量更新融资融券交易明细数据 (margin_detail)...")

    try:
        # 1. 获取最新日期
        latest_date = downloader.db.get_latest_date('margin_detail', 'trade_date')
        today = datetime.now().strftime('%Y%m%d')

        if latest_date is None:
            # 融资融券数据开始于 2010年
            start_date = '20100101'
            logger.info("数据库中没有融资融券历史数据，将进行完整初始化")
            logger.info(f"  (数据起始日期: {start_date})")
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
                rows = downloader.download_margin_detail(trade_date=trade_date)
                if rows > 0:
                    success_count += 1
                    logger.info(f"  ✓ {trade_date}: {rows} 行")
            except Exception as e:
                logger.error(f"  ✗ {trade_date} 更新失败: {e}")
                # 不中断，继续下一个

        logger.info(f"✓ 融资融券明细更新完成: 成功 {success_count}/{len(trading_dates)}")

    except Exception as e:
        logger.error(f"✗ 更新融资融券明细数据失败: {e}")
        logger.warning("  继续执行其他更新任务...")


def update_sw_daily(downloader: DataDownloader):
    """
    增量更新申万行业指数日线数据 (sw_daily)

    策略：
        1. 获取数据库中 sw_daily 的最新日期
        2. 从下一天开始更新到今天

    数据说明：
        - 申万2021版行业指数日线行情
        - 包含 OHLC、涨跌幅、成交量/额、PE/PB、市值等
        - 需要 5000 积分
    """
    logger.info("=" * 60)
    logger.info("开始增量更新申万行业指数日线数据 (sw_daily)...")

    try:
        # 1. 获取最新日期
        latest_date = downloader.db.get_latest_date('sw_daily', 'trade_date')
        today = datetime.now().strftime('%Y%m%d')

        if latest_date is None:
            # sw_daily 数据从2021年开始（申万2021版）
            start_date = '20210101'
            logger.info("数据库中没有申万行业指数历史数据，将进行完整初始化")
            logger.info(f"  (数据起始日期: {start_date})")
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
        total_rows = 0
        for trade_date in trading_dates:
            try:
                rows = downloader.download_sw_daily(trade_date=trade_date)
                if rows > 0:
                    success_count += 1
                    total_rows += rows
                    logger.info(f"  ✓ {trade_date}: {rows} 行")
            except Exception as e:
                logger.error(f"  ✗ {trade_date} 更新失败: {e}")
                # 不中断，继续下一个

        logger.info(f"✓ 申万行业指数日线更新完成: 成功 {success_count}/{len(trading_dates)}, 共 {total_rows} 行")

    except Exception as e:
        logger.error(f"✗ 更新申万行业指数日线数据失败: {e}")
        logger.warning("  继续执行其他更新任务...")


def update_kpl_concept(downloader: DataDownloader):
    """
    增量更新开盘啦题材列表数据 (kpl_concept)

    策略：
        1. 获取数据库中 kpl_concept 的最新日期
        2. 使用批量下载方法（offset 分页）获取该日期之后的所有数据

    说明：
        - 使用 limit/offset 批量下载，比逐日下载效率高很多
        - 包含题材代码、名称、涨停数量、排名变化
        - 每日盘后更新
        - 需要 5000+ 积分
    """
    logger.info("=" * 60)
    logger.info("开始增量更新开盘啦题材列表数据 (kpl_concept)...")

    try:
        # 1. 获取最新日期
        latest_date = downloader.db.get_latest_date('kpl_concept', 'trade_date')
        today = datetime.now().strftime('%Y%m%d')

        if latest_date is None:
            # 初始化：使用批量下载获取所有可用数据
            target_date = None
            logger.info("数据库中没有开盘啦题材历史数据，将批量下载所有可用数据")
        else:
            # 增量更新：只获取最新日期之后的数据
            latest_dt = datetime.strptime(latest_date, '%Y%m%d')
            target_date = (latest_dt + timedelta(days=1)).strftime('%Y%m%d')
            logger.info(f"数据库最新日期: {latest_date}")

            if target_date > today:
                logger.info("无需更新")
                return

            logger.info(f"更新范围: {target_date} → 最新")

        # 2. 批量下载
        total_rows = downloader.download_kpl_concept_batch(target_date=target_date)

        logger.info(f"✓ 开盘啦题材列表更新完成: 共 {total_rows} 行")

    except Exception as e:
        logger.error(f"✗ 更新开盘啦题材列表数据失败: {e}")
        logger.warning("  继续执行其他更新任务...")


def update_kpl_concept_cons(downloader: DataDownloader):
    """
    增量更新开盘啦题材成分股数据 (kpl_concept_cons)

    策略：
        1. 获取数据库中 kpl_concept_cons 的最新日期
        2. 获取期间的交易日列表
        3. 按交易日逐日下载

    说明：
        - 包含题材与股票的关联关系
        - 包含股票在该题材中的描述和人气值
        - 每日盘后更新
        - 需要 5000+ 积分
    """
    logger.info("=" * 60)
    logger.info("开始增量更新开盘啦题材成分股数据 (kpl_concept_cons)...")

    try:
        # 1. 获取最新日期
        latest_date = downloader.db.get_latest_date('kpl_concept_cons', 'trade_date')
        today = datetime.now().strftime('%Y%m%d')

        if latest_date is None:
            # 初始化：从近30天开始
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
            logger.info("数据库中没有开盘啦题材成分历史数据，将从近30天开始初始化")
        else:
            latest_dt = datetime.strptime(latest_date, '%Y%m%d')
            start_date = (latest_dt + timedelta(days=1)).strftime('%Y%m%d')
            logger.info(f"数据库最新日期: {latest_date}")

        if start_date > today:
            logger.info("无需更新")
            return

        logger.info(f"更新范围: {start_date} → {today}")

        # 2. 获取交易日列表
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

        # 3. 按交易日逐日下载
        total_rows = 0
        for trade_date in trading_dates:
            try:
                rows = downloader.download_kpl_concept_cons(trade_date)
                total_rows += rows
            except Exception as e:
                logger.error(f"  ✗ {trade_date} 更新失败: {e}")

        logger.info(f"✓ 开盘啦题材成分更新完成: 共 {total_rows} 行")

    except Exception as e:
        logger.error(f"✗ 更新开盘啦题材成分数据失败: {e}")
        logger.warning("  继续执行其他更新任务...")


def update_ths_index(downloader: DataDownloader):
    """
    更新同花顺板块指数信息 (ths_index)

    策略：
        1. 检查数据库中是否已有数据
        2. 如果没有数据，下载指定类型的板块信息
        3. 如果已有数据，跳过（板块信息相对稳定，不需要频繁更新）

    下载的类型：
        - I: 行业板块 (1077个)
        - N: 概念板块 (411个)
        - R: 地域板块 (33个)
        - BB: 宽基指数 (46个)

    不下载的类型：
        - S: 特色指数 (126个) - 技术面筛选，实际使用少
        - ST: 风格指数 (21个) - 可用其他方式构建
        - TH: 同花顺特色 (10个) - 专有指数，可替代性强
    """
    logger.info("=" * 60)
    logger.info("开始检查同花顺板块指数数据 (ths_index)...")

    try:
        # 检查是否已有数据
        if downloader.db.table_exists('ths_index'):
            result = downloader.db.execute_query("SELECT COUNT(*) as count FROM ths_index")
            if not result.empty and result.iloc[0]['count'] > 0:
                logger.info(f"数据库中已有 {result.iloc[0]['count']} 条板块指数数据，跳过更新")
                return

        logger.info("数据库中没有同花顺板块数据，开始初始化...")

        total_rows = 0

        # 只下载 I(行业)、N(概念)、R(地域)、BB(宽基) 四类
        types_to_download = [
            ('I', '行业板块'),
            ('N', '概念板块'),
            ('R', '地域板块'),
            ('BB', '宽基指数'),
        ]

        for type_code, type_name in types_to_download:
            rows = downloader.download_ths_index(type=type_code)
            logger.info(f"  - {type_name} ({type_code}): {rows} 条")
            total_rows += rows

        logger.info(f"✓ 同花顺板块指数初始化完成: 总计 {total_rows} 条")

    except Exception as e:
        logger.error(f"✗ 更新同花顺板块指数数据失败: {e}")
        logger.warning("  继续执行其他更新任务...")


def update_ths_member(downloader: DataDownloader):
    """
    更新同花顺板块成分股数据 (ths_member)

    策略：
        1. 使用元数据表记录上次更新时间
        2. 如果距离上次更新 < 7天，跳过更新（周更新策略）
        3. 否则下载所有板块的成分股
    """
    logger.info("=" * 60)
    logger.info("开始更新同花顺板块成分股数据 (ths_member)...")

    try:
        # 1. 检查上次更新时间
        import time
        last_update = downloader.db.get_cache_metadata('ths_member')
        if last_update is not None:
            days_since_update = (time.time() - last_update) / 86400
            if days_since_update < 7:
                logger.info(f"  距离上次更新仅 {days_since_update:.1f} 天，跳过（每周更新一次）")
                return
            else:
                logger.info(f"  距离上次更新 {days_since_update:.1f} 天，开始更新...")
        else:
            logger.info("  首次运行，开始初始化...")

        # 2. 获取所有板块代码
        if not downloader.db.table_exists('ths_index'):
            logger.warning("  ths_index 表不存在，请先运行 update_ths_index()")
            return

        indices_df = downloader.db.execute_query(
            "SELECT ts_code, name, type FROM ths_index ORDER BY type, ts_code"
        )

        if indices_df.empty:
            logger.warning("  未找到同花顺板块数据，请先运行 update_ths_index()")
            return

        logger.info(f"  共 {len(indices_df)} 个板块")

        # 3. 逐个板块下载成分股
        success_count = 0
        total_rows = 0

        for idx, row in indices_df.iterrows():
            ts_code = row['ts_code']

            try:
                rows = downloader.download_ths_member(ts_code=ts_code)
                if rows > 0:
                    success_count += 1
                    total_rows += rows

                # 每 100 个板块打印一次进度
                if (idx + 1) % 100 == 0:
                    logger.info(f"  进度: {idx + 1}/{len(indices_df)}, 已下载 {total_rows} 条")

            except Exception as e:
                logger.debug(f"  {ts_code} 下载失败: {e}")
                continue

        logger.info(f"✓ 同花顺板块成分股更新完成: 成功 {success_count}/{len(indices_df)}, 总计 {total_rows} 条")

        # 更新元数据
        downloader.db.update_cache_metadata('ths_member', time.time())

    except Exception as e:
        logger.error(f"✗ 更新同花顺板块成分股数据失败: {e}")
        logger.warning("  继续执行其他更新任务...")


def update_ths_daily(downloader: DataDownloader):
    """
    增量更新同花顺板块指数日行情数据 (ths_daily)

    策略：
        1. 获取数据库中 ths_daily 的最新日期
        2. 从下一天开始更新到今天
        3. 按日期批量下载所有板块的日行情

    数据说明：
        - 数据从 20180102 开始
        - 包含 OHLC、涨跌幅、成交量、换手率、PE/PB、市值等
        - 需要 6000 积分
    """
    logger.info("=" * 60)
    logger.info("开始增量更新同花顺板块日行情数据 (ths_daily)...")

    try:
        # 1. 获取最新日期
        latest_date = downloader.db.get_latest_date('ths_daily', 'trade_date')
        today = datetime.now().strftime('%Y%m%d')

        if latest_date is None:
            # 数据从 2018-01-02 开始
            start_date = '20180102'
            logger.info("数据库中没有同花顺板块日行情历史数据，将进行完整初始化")
            logger.info(f"  (数据起始日期: {start_date})")
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
        total_rows = 0
        for trade_date in trading_dates:
            try:
                rows = downloader.download_ths_daily(trade_date=trade_date)
                if rows > 0:
                    success_count += 1
                    total_rows += rows
                    logger.info(f"  ✓ {trade_date}: {rows} 行")
            except Exception as e:
                logger.error(f"  ✗ {trade_date} 更新失败: {e}")
                # 不中断，继续下一个

        logger.info(f"✓ 同花顺板块日行情更新完成: 成功 {success_count}/{len(trading_dates)}, 共 {total_rows} 行")

    except Exception as e:
        logger.error(f"✗ 更新同花顺板块日行情数据失败: {e}")
        logger.warning("  继续执行其他更新任务...")


def update_fund_basic(downloader: DataDownloader):
    """
    更新基金列表 (fund_basic)

    策略：
        每次全量更新所有市场和状态的基金
    """
    logger.info("=" * 60)
    logger.info("开始更新基金列表 (fund_basic)...")

    try:
        total_rows = 0
        for market, market_name in [('E', '场内'), ('O', '场外')]:
            for status, status_name in [('L', '上市'), ('D', '摘牌'), ('I', '发行中')]:
                rows = downloader.download_fund_basic(market=market, status=status)
                if rows > 0:
                    logger.info(f"  - {market_name}{status_name}: {rows} 只")
                    total_rows += rows

        logger.info(f"✓ 基金列表更新完成: 总计 {total_rows} 只")

    except Exception as e:
        logger.error(f"✗ 更新基金列表失败: {e}")
        logger.warning("  继续执行其他更新任务...")


def update_fund_daily(downloader: DataDownloader):
    """
    增量更新场内基金日线行情 (fund_daily)

    策略：
        1. 获取数据库中 fund_daily 的最新日期
        2. 逐日下载（单次限2000条）
    """
    logger.info("=" * 60)
    logger.info("开始增量更新场内基金日线 (fund_daily)...")

    try:
        latest_date = downloader.db.get_latest_date('fund_daily', 'trade_date')
        today = datetime.now().strftime('%Y%m%d')

        if latest_date is None:
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
            logger.info(f"数据库中没有场内基金日线数据，从30天前 {start_date} 开始")
        else:
            latest_dt = datetime.strptime(latest_date, '%Y%m%d')
            start_date = (latest_dt + timedelta(days=1)).strftime('%Y%m%d')
            logger.info(f"数据库最新日期: {latest_date}")

        if start_date > today:
            logger.info("无需更新")
            return

        total_rows = downloader.download_all_fund_daily(start_date, today)
        logger.info(f"✓ 场内基金日线更新完成: 共 {total_rows} 行")

    except Exception as e:
        logger.error(f"✗ 更新场内基金日线失败: {e}")
        logger.warning("  继续执行其他更新任务...")


def update_fund_nav(downloader: DataDownloader):
    """
    增量更新基金净值 (fund_nav)

    策略：
        1. 获取数据库中 fund_nav 的最新日期
        2. 逐日下载
    """
    logger.info("=" * 60)
    logger.info("开始增量更新基金净值 (fund_nav)...")

    try:
        latest_date = downloader.db.get_latest_date('fund_nav', 'nav_date')
        today = datetime.now().strftime('%Y%m%d')

        if latest_date is None:
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
            logger.info(f"数据库中没有基金净值数据，从30天前 {start_date} 开始")
        else:
            latest_dt = datetime.strptime(latest_date, '%Y%m%d')
            start_date = (latest_dt + timedelta(days=1)).strftime('%Y%m%d')
            logger.info(f"数据库最新日期: {latest_date}")

        if start_date > today:
            logger.info("无需更新")
            return

        total_rows = downloader.download_all_fund_nav(start_date, today)
        logger.info(f"✓ 基金净值更新完成: 共 {total_rows} 行")

    except Exception as e:
        logger.error(f"✗ 更新基金净值失败: {e}")
        logger.warning("  继续执行其他更新任务...")


def update_fund_share(downloader: DataDownloader):
    """
    增量更新基金份额 (fund_share)

    策略：
        1. 获取数据库中 fund_share 的最新日期
        2. 逐日下载
    """
    logger.info("=" * 60)
    logger.info("开始增量更新基金份额 (fund_share)...")

    try:
        latest_date = downloader.db.get_latest_date('fund_share', 'trade_date')
        today = datetime.now().strftime('%Y%m%d')

        if latest_date is None:
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
            logger.info(f"数据库中没有基金份额数据，从30天前 {start_date} 开始")
        else:
            latest_dt = datetime.strptime(latest_date, '%Y%m%d')
            start_date = (latest_dt + timedelta(days=1)).strftime('%Y%m%d')
            logger.info(f"数据库最新日期: {latest_date}")

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

        total_rows = 0
        for trade_date in trading_dates:
            try:
                rows = downloader.download_fund_share(trade_date=trade_date)
                total_rows += rows
            except Exception as e:
                logger.error(f"  ✗ {trade_date} 更新失败: {e}")

        logger.info(f"✓ 基金份额更新完成: 共 {total_rows} 行")

    except Exception as e:
        logger.error(f"✗ 更新基金份额失败: {e}")
        logger.warning("  继续执行其他更新任务...")


def update_moneyflow_hsgt(downloader: DataDownloader):
    """
    增量更新沪深港通资金流向 (moneyflow_hsgt)

    策略：
        1. 获取数据库中最新日期
        2. 使用 start_date/end_date 批量下载（单次最大300条）
    """
    logger.info("=" * 60)
    logger.info("开始增量更新沪深港通资金流向 (moneyflow_hsgt)...")

    try:
        latest_date = downloader.db.get_latest_date('moneyflow_hsgt', 'trade_date')
        today = datetime.now().strftime('%Y%m%d')

        if latest_date is None:
            start_date = '20141117'  # 沪港通开通日期
            logger.info(f"数据库中没有沪深港通数据，从 {start_date} 开始初始化")
        else:
            latest_dt = datetime.strptime(latest_date, '%Y%m%d')
            start_date = (latest_dt + timedelta(days=1)).strftime('%Y%m%d')
            logger.info(f"数据库最新日期: {latest_date}")

        if start_date > today:
            logger.info("无需更新")
            return

        logger.info(f"更新范围: {start_date} → {today}")
        rows = downloader.download_moneyflow_hsgt(start_date=start_date, end_date=today)
        logger.info(f"✓ 沪深港通资金流向更新完成: {rows} 行")

    except Exception as e:
        logger.error(f"✗ 更新沪深港通资金流向失败: {e}")
        logger.warning("  继续执行其他更新任务...")


def update_hsgt_top10(downloader: DataDownloader):
    """
    增量更新沪深股通十大成交股 (hsgt_top10)

    策略：
        1. 获取数据库中最新日期
        2. 逐日下载
    """
    logger.info("=" * 60)
    logger.info("开始增量更新沪深股通十大成交 (hsgt_top10)...")

    try:
        latest_date = downloader.db.get_latest_date('hsgt_top10', 'trade_date')
        today = datetime.now().strftime('%Y%m%d')

        if latest_date is None:
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
            logger.info(f"数据库中没有数据，从30天前 {start_date} 开始")
        else:
            latest_dt = datetime.strptime(latest_date, '%Y%m%d')
            start_date = (latest_dt + timedelta(days=1)).strftime('%Y%m%d')
            logger.info(f"数据库最新日期: {latest_date}")

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

        success_count = 0
        for trade_date in trading_dates:
            try:
                rows = downloader.download_hsgt_top10(trade_date=trade_date)
                if rows > 0:
                    success_count += 1
            except Exception as e:
                logger.error(f"  ✗ {trade_date} 更新失败: {e}")

        logger.info(f"✓ 沪深股通十大成交更新完成: 成功 {success_count}/{len(trading_dates)}")

    except Exception as e:
        logger.error(f"✗ 更新沪深股通十大成交失败: {e}")
        logger.warning("  继续执行其他更新任务...")


def update_ggt_top10(downloader: DataDownloader):
    """
    增量更新港股通十大成交股 (ggt_top10)
    """
    logger.info("=" * 60)
    logger.info("开始增量更新港股通十大成交 (ggt_top10)...")

    try:
        latest_date = downloader.db.get_latest_date('ggt_top10', 'trade_date')
        today = datetime.now().strftime('%Y%m%d')

        if latest_date is None:
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
            logger.info(f"数据库中没有数据，从30天前 {start_date} 开始")
        else:
            latest_dt = datetime.strptime(latest_date, '%Y%m%d')
            start_date = (latest_dt + timedelta(days=1)).strftime('%Y%m%d')
            logger.info(f"数据库最新日期: {latest_date}")

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

        success_count = 0
        for trade_date in trading_dates:
            try:
                rows = downloader.download_ggt_top10(trade_date=trade_date)
                if rows > 0:
                    success_count += 1
            except Exception as e:
                logger.error(f"  ✗ {trade_date} 更新失败: {e}")

        logger.info(f"✓ 港股通十大成交更新完成: 成功 {success_count}/{len(trading_dates)}")

    except Exception as e:
        logger.error(f"✗ 更新港股通十大成交失败: {e}")
        logger.warning("  继续执行其他更新任务...")


def update_ggt_daily(downloader: DataDownloader):
    """
    增量更新港股通每日成交统计 (ggt_daily)
    """
    logger.info("=" * 60)
    logger.info("开始增量更新港股通每日成交 (ggt_daily)...")

    try:
        latest_date = downloader.db.get_latest_date('ggt_daily', 'trade_date')
        today = datetime.now().strftime('%Y%m%d')

        if latest_date is None:
            start_date = '20141117'
            logger.info(f"数据库中没有数据，从 {start_date} 开始初始化")
        else:
            latest_dt = datetime.strptime(latest_date, '%Y%m%d')
            start_date = (latest_dt + timedelta(days=1)).strftime('%Y%m%d')
            logger.info(f"数据库最新日期: {latest_date}")

        if start_date > today:
            logger.info("无需更新")
            return

        logger.info(f"更新范围: {start_date} → {today}")
        rows = downloader.download_ggt_daily(start_date=start_date, end_date=today)
        logger.info(f"✓ 港股通每日成交更新完成: {rows} 行")

    except Exception as e:
        logger.error(f"✗ 更新港股通每日成交失败: {e}")
        logger.warning("  继续执行其他更新任务...")


def update_hk_hold(downloader: DataDownloader):
    """
    增量更新沪深港通持股明细 (hk_hold)

    策略：
        1. 获取数据库中最新日期
        2. 逐日下载（单次最大3800条）
    """
    logger.info("=" * 60)
    logger.info("开始增量更新沪深港通持股明细 (hk_hold)...")

    try:
        latest_date = downloader.db.get_latest_date('hk_hold', 'trade_date')
        today = datetime.now().strftime('%Y%m%d')

        if latest_date is None:
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
            logger.info(f"数据库中没有数据，从30天前 {start_date} 开始")
        else:
            latest_dt = datetime.strptime(latest_date, '%Y%m%d')
            start_date = (latest_dt + timedelta(days=1)).strftime('%Y%m%d')
            logger.info(f"数据库最新日期: {latest_date}")

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

        success_count = 0
        total_rows = 0
        for trade_date in trading_dates:
            try:
                rows = downloader.download_hk_hold(trade_date=trade_date)
                if rows > 0:
                    success_count += 1
                    total_rows += rows
            except Exception as e:
                logger.error(f"  ✗ {trade_date} 更新失败: {e}")

        logger.info(f"✓ 沪深港通持股明细更新完成: 成功 {success_count}/{len(trading_dates)}, 共 {total_rows} 行")

    except Exception as e:
        logger.error(f"✗ 更新沪深港通持股明细失败: {e}")
        logger.warning("  继续执行其他更新任务...")


def _is_earnings_season() -> bool:
    """
    判断当前是否处于财报季

    财报季定义（A股财报披露规则）：
        - 1月~4月：年报季（上年年报 + 一季报密集披露）
        - 7月~8月：中报季
        - 10月~11月：三季报季

    非财报季（5-6月、9月、12月）财务数据几乎无变化，
    可降低更新频率至每周一次。
    """
    month = datetime.now().month
    return month in (1, 2, 3, 4, 7, 8, 10, 11)


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


def _run_parallel_tasks(downloader: DataDownloader, tasks: list, max_workers: int = 4):
    """
    并行执行多个互不依赖的更新任务

    当一个任务因 API 限速而等待时，其他任务可以同时调用不同的 API，
    从而大幅提升整体更新速度。

    Args:
        downloader: DataDownloader 实例
        tasks: (函数, 描述) 元组的列表
        max_workers: 最大并发线程数
    """
    failed_tasks = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_name = {}
        for func, name in tasks:
            future = executor.submit(func, downloader)
            future_to_name[future] = name

        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                future.result()
            except Exception as e:
                logger.error(f"✗ 并行任务 [{name}] 失败: {e}")
                failed_tasks.append(name)

    if failed_tasks:
        logger.warning(f"以下并行任务执行失败: {', '.join(failed_tasks)}")


def main():
    """
    主函数：执行所有每日更新任务

    执行策略（分阶段并行）：
    阶段1（串行）：基础数据，其他任务的依赖
        - 交易日历、股票列表
        - 申万行业分类、同花顺板块指数（仅初始化时下载）
    阶段2（并行）：所有互不依赖的数据更新任务
        - 利用不同 API 各自独立的限速窗口，当一个 API 限速等待时，
          其他 API 的任务可以同时执行，从而大幅提升整体速度
    """
    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info("开始每日数据更新任务（并行模式）")
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
            # 2.5 修复缺失主键的表（一次性操作）
            _ensure_table_primary_keys(downloader)

            # ========== 阶段 1：串行执行基础数据（其他任务的依赖） ==========
            logger.info("")
            logger.info("=" * 60)
            logger.info("阶段 1/2：更新基础数据（串行）...")
            logger.info("=" * 60)

            update_trade_calendar(downloader)
            update_stock_basic(downloader)
            update_index_classify(downloader)      # 申万行业分类（仅初始化时下载）
            update_ths_index(downloader)            # 同花顺板块指数（仅初始化时下载）
            update_fund_basic(downloader)           # 基金列表

            # ========== 阶段 2：并行执行所有独立的更新任务 ==========
            logger.info("")
            logger.info("=" * 60)
            logger.info("阶段 2/2：并行更新所有独立数据...")
            logger.info("=" * 60)

            parallel_tasks = [
                # 依赖 index_classify 的任务
                (update_index_member_all, "申万行业成分股"),
                # 依赖 ths_index 的任务
                (update_ths_member, "同花顺板块成分股"),
                # 每日行情数据
                (update_daily_data, "每日数据(日线/复权/基本面)"),
                (update_index_daily, "指数日线"),
                # 财务数据
                (update_financial_indicators, "财务指标(VIP)"),
                (update_financial_statements, "三大财务报表"),
                (update_dividend, "分红送股"),
                # 交易数据
                (update_margin_detail, "融资融券明细"),
                (update_cyq_perf, "筹码分布"),
                (update_cyq_chips, "筹码分布详情"),
                (update_stk_factor_pro, "技术因子"),
                # 龙虎榜
                (update_dc_member, "龙虎榜机构席位"),
                (update_dc_index, "龙虎榜个股明细"),
                (update_limit_list_d, "涨跌停炸板"),
                # 资金流向
                (update_moneyflow_ind_dc, "行业资金流向"),
                (update_moneyflow_dc, "个股资金流向(DC)"),
                (update_moneyflow, "个股资金流向"),
                # 行业/板块
                (update_sw_daily, "申万行业指数日线"),
                (update_ths_daily, "同花顺板块日行情"),
                # 题材
                (update_kpl_concept, "开盘啦题材列表"),
                (update_kpl_concept_cons, "开盘啦题材成分股"),
                # 基金
                (update_fund_daily, "场内基金日线"),
                (update_fund_nav, "基金净值"),
                (update_fund_share, "基金份额"),
                # 沪深港通
                (update_moneyflow_hsgt, "沪深港通资金流向"),
                (update_hsgt_top10, "沪深股通十大成交"),
                (update_ggt_top10, "港股通十大成交"),
                (update_ggt_daily, "港股通每日成交"),
                (update_hk_hold, "沪深港通持股明细"),
            ]

            _run_parallel_tasks(downloader, parallel_tasks, max_workers=4)

            # 4. 完成
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            logger.info("=" * 60)
            logger.info("✓ 所有每日数据更新任务完成！")
            logger.info(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"总耗时: {duration:.1f} 秒 ({duration/60:.1f} 分钟)")
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