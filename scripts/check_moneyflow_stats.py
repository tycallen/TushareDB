#!/usr/bin/env python3
"""
统计 moneyflow_dc 数据分布
"""
import os
import sys
import pandas as pd
from dotenv import load_dotenv

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tushare_db.duckdb_manager import DuckDBManager

load_dotenv()

def check_stats():
    db_path = os.getenv('DUCKDB_PATH', 'tushare.db')
    print(f"Connecting to {db_path}...")
    db = DuckDBManager(db_path)

    try:
        if not db.table_exists('moneyflow_dc'):
            print("Table 'moneyflow_dc' does not exist.")
            return

        print("=" * 60)
        print("moneyflow_dc 数据统计")
        print("=" * 60)

        # 1. 总体统计
        summary = db.execute_query('''
            SELECT 
                COUNT(*) as total_records,
                COUNT(DISTINCT trade_date) as total_days,
                MIN(trade_date) as start_date,
                MAX(trade_date) as end_date
            FROM moneyflow_dc
        ''').iloc[0]

        print(f"总记录数: {summary['total_records']:,}")
        print(f"交易日数: {summary['total_days']}")
        print(f"时间范围: {summary['start_date']} - {summary['end_date']}")
        
        # 2. 每日记录数分布
        print("-" * 60)
        print("每日记录数分布:")
        daily_counts = db.execute_query('''
            SELECT trade_date, COUNT(*) as count
            FROM moneyflow_dc
            GROUP BY trade_date
            ORDER BY trade_date
        ''')
        
        print(f"  - 最小每日记录: {daily_counts['count'].min()}")
        print(f"  - 最大每日记录: {daily_counts['count'].max()}")
        print(f"  - 平均每日记录: {daily_counts['count'].mean():.2f}")
        print(f"  - 中位数记录: {daily_counts['count'].median()}")

        # 3. 检查是否有异常少的日期 (例如少于 1000 条)
        low_count_days = daily_counts[daily_counts['count'] < 1000]
        if not low_count_days.empty:
            print(f"\n⚠ 发现 {len(low_count_days)} 个日期记录数异常偏少 (<1000):")
            print(low_count_days.head(10).to_string(index=False))
            if len(low_count_days) > 10:
                print(f"... 等共 {len(low_count_days)} 天")
        else:
             print("\n✓ 每日记录数看起来正常 (均 >= 1000)")

        # 4. 检查日期连续性
        if db.table_exists('trade_cal'):
            print("-" * 60)
            print("日期连续性检查:")
            trading_days = db.execute_query('''
                SELECT cal_date 
                FROM trade_cal 
                WHERE cal_date BETWEEN ? AND ? AND is_open = 1
            ''', [summary['start_date'], summary['end_date']])['cal_date'].tolist()
            
            existing_days = set(daily_counts['trade_date'].tolist())
            missing_days = [d for d in trading_days if d not in existing_days]
            
            if missing_days:
                print(f"⚠ 在 {summary['start_date']} - {summary['end_date']} 期间缺失 {len(missing_days)} 个交易日:")
                print(missing_days[:10])
                if len(missing_days) > 10:
                    print(f"... 等共 {len(missing_days)} 天")
            else:
                print(f"✓ 在 {summary['start_date']} - {summary['end_date']} 期间数据连续无缺失")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    check_stats()
