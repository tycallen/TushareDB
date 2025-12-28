#!/usr/bin/env python3
"""
修复 moneyflow_dc 表中不完整的历史数据

问题：早期下载的数据可能每天只有1条记录（不完整）
解决：删除记录数异常少的日期数据，然后重新下载
"""
import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tushare_db.downloader import DataDownloader
from src.tushare_db.duckdb_manager import DuckDBManager

load_dotenv()

def fix_incomplete_moneyflow_data():
    """修复不完整的 moneyflow_dc 数据"""

    tushare_token = os.getenv('TUSHARE_TOKEN')
    db_path = os.getenv('DUCKDB_PATH', 'tushare.db')

    if not tushare_token:
        print("错误: 未找到 TUSHARE_TOKEN 环境变量")
        return

    print("=" * 60)
    print("moneyflow_dc 数据修复工具")
    print("=" * 60)

    db = DuckDBManager(db_path)

    try:
        # 1. 检查每个交易日的记录数
        print("\n【步骤1】检查数据完整性...")
        print("-" * 60)

        result = db.execute_query('''
            SELECT trade_date, COUNT(*) as count
            FROM moneyflow_dc
            GROUP BY trade_date
            ORDER BY trade_date
        ''')

        if result.empty:
            print("✓ moneyflow_dc 表为空，无需修复")
            return

        print(f"共有 {len(result)} 个交易日的数据\n")

        # 2. 识别不完整的数据（正常应该每天5000+条记录）
        incomplete_dates = result[result['count'] < 5000]['trade_date'].tolist()

        if not incomplete_dates:
            # 额外检查：是否有日期缺失
            start_date = '20230911'
            end_date = datetime.now().strftime('%Y%m%d')
            
            # 确保 trade_cal 存在
            if not db.table_exists('trade_cal'):
                 print("⚠ trade_cal 表不存在，无法检查缺失日期")
                 print("✓ 现有数据完整性检查通过 (但无法检查历史缺失)")
                 return

            trading_days = db.execute_query(
                "SELECT cal_date FROM trade_cal WHERE cal_date BETWEEN ? AND ? AND is_open=1",
                [start_date, end_date]
            )['cal_date'].tolist()
            
            existing_days_set = set(result['trade_date'].tolist())
            missing_days = [d for d in trading_days if d not in existing_days_set]
            
            if missing_days:
                print(f"⚠ 现有数据完整，但发现缺失 {len(missing_days)} 个交易日的数据")
                print(f"  建议运行: python scripts/init_moneyflow_dc_history.py 来补充历史数据")
            else:
                print("✓ 所有数据完整，无需修复（且无历史缺失）")

            print(f"\n数据统计:")
            print(f"  - 最小记录数: {result['count'].min()}")
            print(f"  - 最大记录数: {result['count'].max()}")
            print(f"  - 平均记录数: {result['count'].mean():.0f}")
            return

        print(f"⚠ 发现 {len(incomplete_dates)} 个交易日的数据不完整:\n")
        incomplete_data = result[result['count'] < 5000]
        for _, row in incomplete_data.iterrows():
            print(f"  - {row['trade_date']}: {row['count']} 条 (异常)")

        # 3. 询问是否删除并重新下载
        print("\n" + "-" * 60)
        response = input(f"\n是否删除这 {len(incomplete_dates)} 个交易日的数据并重新下载? (yes/no): ")

        if response.lower() not in ['yes', 'y']:
            print("已取消操作")
            return

        # 4. 删除不完整的数据
        print("\n【步骤2】删除不完整的数据...")
        print("-" * 60)

        for trade_date in incomplete_dates:
            deleted = db.delete_data(
                'moneyflow_dc',
                f"trade_date='{trade_date}'"
            )
            print(f"  ✓ 删除 {trade_date}: {deleted} 条")

        # 5. 重新下载
        print("\n【步骤3】重新下载完整数据...")
        print("-" * 60)

        downloader = DataDownloader(
            tushare_token=tushare_token,
            db_path=db_path,
            rate_limit_profile="standard"
        )

        try:
            success_count = 0
            for trade_date in incomplete_dates:
                try:
                    rows = downloader.download_moneyflow_dc(trade_date=trade_date)
                    if rows > 0:
                        success_count += 1
                        print(f"  ✓ {trade_date}: {rows} 条")
                    else:
                        print(f"  ⚠ {trade_date}: 无数据")
                except Exception as e:
                    print(f"  ✗ {trade_date}: 下载失败 - {e}")

            print(f"\n✓ 修复完成: {success_count}/{len(incomplete_dates)} 个交易日")

            # 6. 验证修复结果
            print("\n【步骤4】验证修复结果...")
            print("-" * 60)

            result_after = db.execute_query('''
                SELECT trade_date, COUNT(*) as count
                FROM moneyflow_dc
                WHERE trade_date IN ({})
                GROUP BY trade_date
                ORDER BY trade_date
            '''.format(','.join([f"'{d}'" for d in incomplete_dates])))

            print("\n修复后的数据:")
            for _, row in result_after.iterrows():
                status = "✓" if row['count'] >= 5000 else "⚠"
                print(f"  {status} {row['trade_date']}: {row['count']} 条")

        finally:
            downloader.close()

        print("\n" + "=" * 60)
        print("✓ 数据修复完成!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ 修复失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()


if __name__ == "__main__":
    fix_incomplete_moneyflow_data()
