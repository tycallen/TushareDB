#!/usr/bin/env python3
"""
初始化 moneyflow_dc 完整历史数据

数据说明：
- moneyflow_dc (个股资金流向) 数据开始于 2023-09-11
- 本脚本下载从 2023-09-11 至今的所有交易日数据
"""
import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tushare_db.downloader import DataDownloader

load_dotenv()


def init_moneyflow_dc_history():
    """初始化 moneyflow_dc 完整历史数据"""

    tushare_token = os.getenv('TUSHARE_TOKEN')
    db_path = os.getenv('DUCKDB_PATH', 'tushare.db')

    if not tushare_token:
        print("错误: 未找到 TUSHARE_TOKEN 环境变量")
        return

    print("=" * 60)
    print("moneyflow_dc 历史数据初始化")
    print("=" * 60)

    downloader = DataDownloader(
        tushare_token=tushare_token,
        db_path=db_path,
        rate_limit_profile="standard"
    )

    try:
        # 1. 检查当前数据状态
        print("\n【步骤1】检查当前数据状态...")
        print("-" * 60)

        latest_date = downloader.db.get_latest_date('moneyflow_dc', 'trade_date')

        if latest_date:
            result = downloader.db.execute_query(
                "SELECT COUNT(DISTINCT trade_date) as days FROM moneyflow_dc"
            )
            existing_days = result.iloc[0]['days']
            print(f"数据库中已有数据:")
            print(f"  - 最新日期: {latest_date}")
            print(f"  - 交易日数: {existing_days} 天")

            response = input(f"\n是否继续初始化（会补充缺失的历史数据）? (yes/no): ")
            # print(f"\n是否继续初始化（会补充缺失的历史数据）? (yes/no): yes (auto)")
            # response = 'yes'
            if response.lower() not in ['yes', 'y']:
                print("已取消操作")
                return

        else:
            print("数据库中暂无 moneyflow_dc 数据")

        # 2. 设置下载范围
        start_date = '20230911'  # moneyflow_dc 数据开始日期
        end_date = datetime.now().strftime('%Y%m%d')

        print(f"\n【步骤2】获取交易日历...")
        print("-" * 60)
        print(f"下载范围: {start_date} → {end_date}")

        if not downloader.db.table_exists('trade_cal') or downloader.db.execute_query("SELECT COUNT(*) FROM trade_cal").iloc[0][0] == 0:
            print("⚠ 交易日历不存在或为空，正在下载...")
            downloader.download_trade_calendar(start_date='19900101', end_date='20301231')

        # 3. 获取所有交易日
        trading_dates_df = downloader.db.execute_query('''
            SELECT cal_date
            FROM trade_cal
            WHERE cal_date >= ? AND cal_date <= ? AND is_open = 1
            ORDER BY cal_date
        ''', [start_date, end_date])

        if trading_dates_df.empty:
            print("✗ 未找到交易日数据，请先运行 update_daily.py 初始化交易日历")
            return

        trading_dates = trading_dates_df['cal_date'].tolist()
        print(f"✓ 需要初始化 {len(trading_dates)} 个交易日")

        # 4. 检查哪些日期已有数据
        if latest_date:
            existing_dates_df = downloader.db.execute_query(
                "SELECT DISTINCT trade_date FROM moneyflow_dc ORDER BY trade_date"
            )
            existing_dates = set(existing_dates_df['trade_date'].tolist())

            # 只下载缺失的日期
            missing_dates = [d for d in trading_dates if d not in existing_dates]

            if not missing_dates:
                print(f"\n✓ 所有交易日数据完整，无需初始化")
                return

            print(f"✓ 已有 {len(existing_dates)} 个交易日数据")
            print(f"✓ 需要补充 {len(missing_dates)} 个交易日数据")

            dates_to_download = missing_dates
        else:
            dates_to_download = trading_dates

        # 5. 确认下载
        print(f"\n准备下载 {len(dates_to_download)} 个交易日的数据")
        print(f"时间范围: {dates_to_download[0]} → {dates_to_download[-1]}")

        response = input(f"\n确认开始下载? (yes/no): ")
        # print(f"\n确认开始下载? (yes/no): no (auto-stop for test)")
        # response = 'no'
        if response.lower() not in ['yes', 'y']:
            print("已取消操作")
            return

        # 6. 逐日下载
        print(f"\n【步骤3】开始下载历史数据...")
        print("-" * 60)

        success_count = 0
        fail_count = 0
        no_data_count = 0

        for idx, trade_date in enumerate(dates_to_download, 1):
            try:
                rows = downloader.download_moneyflow_dc(trade_date=trade_date)
                if rows > 0:
                    success_count += 1
                    print(f"  [{idx}/{len(dates_to_download)}] ✓ {trade_date}: {rows} 条")
                else:
                    no_data_count += 1
                    print(f"  [{idx}/{len(dates_to_download)}] ⚠ {trade_date}: 无数据")
            except Exception as e:
                fail_count += 1
                print(f"  [{idx}/{len(dates_to_download)}] ✗ {trade_date}: 失败 - {e}")

        # 7. 验证结果
        print(f"\n【步骤4】验证下载结果...")
        print("-" * 60)

        result = downloader.db.execute_query('''
            SELECT
                COUNT(DISTINCT trade_date) as total_days,
                MIN(trade_date) as earliest_date,
                MAX(trade_date) as latest_date,
                COUNT(*) as total_records
            FROM moneyflow_dc
        ''')

        if not result.empty:
            row = result.iloc[0]
            print(f"\n最终数据统计:")
            print(f"  - 交易日数: {row['total_days']} 天")
            print(f"  - 起始日期: {row['earliest_date']}")
            print(f"  - 最新日期: {row['latest_date']}")
            print(f"  - 总记录数: {row['total_records']:,} 条")
            print(f"  - 平均每天: {row['total_records'] / row['total_days']:.0f} 条")

        print(f"\n下载统计:")
        print(f"  - 成功: {success_count} 天")
        print(f"  - 无数据: {no_data_count} 天")
        print(f"  - 失败: {fail_count} 天")

        print("\n" + "=" * 60)
        print("✓ 历史数据初始化完成!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ 初始化失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        downloader.close()


if __name__ == "__main__":
    init_moneyflow_dc_history()
