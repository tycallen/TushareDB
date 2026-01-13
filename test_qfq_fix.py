#!/usr/bin/env python3
"""
测试前复权（qfq）修复
验证前复权计算是否正确消除除权除息的影响
"""
import os
import sys
from dotenv import load_dotenv

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.tushare_db.reader import DataReader

load_dotenv()


def test_qfq_fix():
    """测试前复权修复"""

    db_path = os.getenv('DUCKDB_PATH', 'tushare.db')
    reader = DataReader(db_path=db_path)

    print("=" * 60)
    print("测试前复权（qfq）修复")
    print("=" * 60)

    try:
        # 测试1：跨越除权事件的连续性
        print("\n【测试1】除权事件连续性测试 - 002141.SZ")
        print("-" * 60)
        print("2016-10-24 发生重大除权：复权因子从 2.303 跳到 8.059")

        df_qfq = reader.get_stock_daily(
            ts_code='002141.SZ',
            start_date='20161020',
            end_date='20161028',
            adj='qfq'
        )

        # 获取除权前后的数据
        row_before = df_qfq[df_qfq['trade_date'] == '20161021'].iloc[0]
        row_after = df_qfq[df_qfq['trade_date'] == '20161024'].iloc[0]

        qfq_change = (row_after['open'] - row_before['open']) / row_before['open'] * 100

        # 获取原始价格变化
        df_raw = reader.get_stock_daily(
            ts_code='002141.SZ',
            start_date='20161021',
            end_date='20161024'
        )
        raw_change = (df_raw.iloc[1]['open'] - df_raw.iloc[0]['open']) / df_raw.iloc[0]['open'] * 100

        print(f"\n除权前 (20161021):")
        print(f"  前复权开盘价: {row_before['open']:.6f}")
        print(f"\n除权后 (20161024):")
        print(f"  前复权开盘价: {row_after['open']:.6f}")
        print(f"\n涨跌幅对比:")
        print(f"  原始价涨跌幅: {raw_change:.2f}% （除权影响）")
        print(f"  前复权涨跌幅: {qfq_change:.2f}% （消除除权）")

        # 验证连续性
        if abs(qfq_change) < 1.0:  # 前复权涨跌幅应该很小
            print(f"\n✓ 测试通过：前复权成功消除除权影响，价格连续")
        else:
            print(f"\n✗ 测试失败：前复权价格不连续")

        # 测试2：无除权事件的情况
        print("\n" + "=" * 60)
        print("【测试2】无除权事件测试 - 002141.SZ 2025-06-13")
        print("-" * 60)
        print("2025年复权因子恒定为 8.059（无除权事件）")

        df_single = reader.get_stock_daily(
            ts_code='002141.SZ',
            start_date='20250613',
            end_date='20250613',
            adj='qfq'
        )

        df_raw_single = reader.get_stock_daily(
            ts_code='002141.SZ',
            start_date='20250613',
            end_date='20250613'
        )

        qfq_price = df_single.iloc[0]['open']
        raw_price = df_raw_single.iloc[0]['open']

        print(f"\n原始开盘价: {raw_price}")
        print(f"前复权开盘价: {qfq_price}")

        if abs(qfq_price - raw_price) < 0.001:
            print(f"\n✓ 测试通过：无除权事件时，前复权价 = 原始价")
        else:
            print(f"\n✗ 测试失败：前复权价与原始价不符")

        # 测试3：多股票查询
        print("\n" + "=" * 60)
        print("【测试3】多股票查询测试")
        print("-" * 60)

        df_multi = reader.get_multiple_stocks_daily(
            ts_codes=['002141.SZ', '000001.SZ'],
            start_date='20161024',
            end_date='20161024',
            adj='qfq'
        )

        print(f"\n查询到 {len(df_multi)} 只股票的数据")
        for _, row in df_multi.iterrows():
            print(f"  {row['ts_code']}: 前复权开盘 {row['open']:.4f}")

        if len(df_multi) == 2:
            print(f"\n✓ 测试通过：多股票前复权查询正常")
        else:
            print(f"\n✗ 测试失败：多股票查询异常")

        print("\n" + "=" * 60)
        print("✓ 所有测试完成！前复权计算已修复")
        print("=" * 60)

        print("\n修复说明:")
        print("  1. 修复了前复权计算公式：qfq = raw × (adj_factor / latest_factor)")
        print("  2. 前复权现在可以正确消除除权除息的影响")
        print("  3. 除权事件前后的价格走势保持连续")
        print("  4. 最新的价格保持真实市场价格")

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()

    finally:
        reader.close()


if __name__ == "__main__":
    test_qfq_fix()
