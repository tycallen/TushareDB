#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试新架构

验证 DataDownloader 和 DataReader 是否正常工作
"""
import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.tushare_db import DataDownloader, DataReader


def test_basic_functionality():
    """测试基本功能"""
    print("="*60)
    print("测试新架构基本功能")
    print("="*60)

    db_path = 'test_new_arch.db'

    try:
        # 1. 测试下载器
        print("\n[1/4] 测试 DataDownloader...")
        downloader = DataDownloader(db_path=db_path)
        print("  ✓ DataDownloader 初始化成功")

        # 下载交易日历
        print("\n  下载交易日历（2023年）...")
        rows = downloader.download_trade_calendar('20230101', '20231231')
        print(f"  ✓ 下载完成: {rows} 行")

        # 下载股票列表
        print("\n  下载股票列表...")
        rows = downloader.download_stock_basic('L')
        print(f"  ✓ 下载完成: {rows} 只股票")

        # 下载单只股票数据
        print("\n  下载平安银行日线数据...")
        rows = downloader.download_stock_daily('000001.SZ', '20230101', '20230131')
        print(f"  ✓ 下载完成: {rows} 行")

        # 下载复权因子
        print("\n  下载复权因子...")
        rows = downloader.download_adj_factor('000001.SZ', '20230101', '20230131')
        print(f"  ✓ 下载完成: {rows} 行")

        downloader.close()
        print("\n  ✓ DataDownloader 测试通过")

        # 2. 测试查询器
        print("\n[2/4] 测试 DataReader...")
        reader = DataReader(db_path=db_path)
        print("  ✓ DataReader 初始化成功")

        # 查询股票列表
        print("\n  查询股票列表...")
        df = reader.get_stock_basic(list_status='L')
        print(f"  ✓ 查询成功: {len(df)} 只股票")

        # 查询交易日历
        print("\n  查询交易日历...")
        df = reader.get_trade_calendar('20230101', '20230131', is_open='1')
        print(f"  ✓ 查询成功: {len(df)} 个交易日")

        # 查询日线数据（不复权）
        print("\n  查询日线数据（不复权）...")
        df = reader.get_stock_daily('000001.SZ', '20230101', '20230131')
        print(f"  ✓ 查询成功: {len(df)} 行")
        if not df.empty:
            print(f"    日期范围: {df['trade_date'].min()} -> {df['trade_date'].max()}")
            print(f"    收盘价: {df['close'].iloc[0]:.2f} -> {df['close'].iloc[-1]:.2f}")

        # 查询日线数据（前复权）
        print("\n  查询日线数据（前复权）...")
        df_adj = reader.get_stock_daily('000001.SZ', '20230101', '20230131', adj='qfq')
        print(f"  ✓ 查询成功: {len(df_adj)} 行")
        if not df_adj.empty:
            print(f"    复权后收盘价: {df_adj['close'].iloc[0]:.2f} -> {df_adj['close'].iloc[-1]:.2f}")

        reader.close()
        print("\n  ✓ DataReader 测试通过")

        # 3. 测试性能
        print("\n[3/4] 测试查询性能...")
        import time

        reader = DataReader(db_path=db_path)
        start = time.time()
        for _ in range(50):
            df = reader.get_stock_daily('000001.SZ', '20230101', '20230131')
        elapsed = time.time() - start

        print(f"  ✓ 50次查询耗时: {elapsed:.3f}秒 (平均 {elapsed*20:.1f}ms/次)")
        reader.close()

        # 4. 清理测试数据库
        print("\n[4/4] 清理测试数据...")
        if os.path.exists(db_path):
            os.remove(db_path)
            print(f"  ✓ 已删除测试数据库: {db_path}")

        print("\n" + "="*60)
        print("✓ 所有测试通过！新架构工作正常")
        print("="*60)

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()

        # 清理
        if os.path.exists(db_path):
            os.remove(db_path)

        sys.exit(1)


if __name__ == '__main__':
    test_basic_functionality()
