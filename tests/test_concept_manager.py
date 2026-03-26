#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
概念板块数据管理器测试脚本

演示如何使用 ConceptDataManager 接入 jquant_data_sync 数据
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tushare_db import ConceptDataManager

def main():
    # 初始化管理器（自动确定缓存目录）
    manager = ConceptDataManager(db_path="tushare.db")

    # 查看缓存信息
    print("=" * 60)
    print("📂 缓存信息")
    print("=" * 60)
    info = manager.get_cache_info()
    for key, value in info.items():
        print(f"  {key}: {value}")

    # 拉取数据（自动检查缓存）
    print("\n" + "=" * 60)
    print("📥 拉取数据")
    print("=" * 60)
    success = manager.pull_data()
    if not success:
        print("❌ 数据拉取失败")
        return

    # 查看所有概念板块
    print("\n" + "=" * 60)
    print("📋 概念板块列表 (前10个)")
    print("=" * 60)
    concepts = manager.get_all_concepts()
    print(concepts.head(10).to_string(index=False))
    print(f"\n总计: {len(concepts)} 个概念板块")

    # 搜索概念
    print("\n" + "=" * 60)
    print("🔍 搜索含'人工智能'的概念")
    print("=" * 60)
    ai_concepts = manager.search_concepts("人工智能")
    print(ai_concepts.to_string(index=False))

    # 获取某概念的成分股
    print("\n" + "=" * 60)
    print("🤖 人工智能板块成分股 (2024-01-15)")
    print("=" * 60)
    try:
        ai_stocks = manager.get_concept_stocks(
            trade_date='20240115',
            concept_name='人工智能'
        )
        if not ai_stocks.empty:
            print(ai_stocks.head(10).to_string(index=False))
            print(f"\n总计: {len(ai_stocks)} 只股票")
        else:
            print("⚠️ 未找到数据")
    except Exception as e:
        print(f"⚠️ 查询失败: {e}")

    # 获取某股票的概念
    print("\n" + "=" * 60)
    print("📈 000001.SZ 所属概念 (2024-01-15)")
    print("=" * 60)
    try:
        stock_concepts = manager.get_stock_concepts(
            trade_date='20240115',
            ts_code='000001.SZ'
        )
        if not stock_concepts.empty:
            print(stock_concepts.to_string(index=False))
            print(f"\n总计: {len(stock_concepts)} 个概念")
        else:
            print("⚠️ 未找到数据")
    except Exception as e:
        print(f"⚠️ 查询失败: {e}")

    # PIT 查询示例
    print("\n" + "=" * 60)
    print("⏰ PIT 查询对比")
    print("=" * 60)
    test_dates = ['20230115', '20230615', '20240115']
    for date in test_dates:
        try:
            df = manager.get_concept_stocks(
                trade_date=date,
                concept_name='人工智能'
            )
            print(f"  {date}: {len(df)} 只股票")
        except Exception as e:
            print(f"  {date}: 查询失败 - {e}")

    print("\n" + "=" * 60)
    print("✅ 测试完成")
    print("=" * 60)

if __name__ == "__main__":
    main()
