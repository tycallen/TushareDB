#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试读取聚宽概念板块数据
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tushare_db import DataReader, ConceptDataManager

def test_with_data_reader():
    """通过 DataReader 测试（推荐方式）"""
    print("=" * 70)
    print("测试 1: 通过 DataReader 读取概念数据")
    print("=" * 70)

    reader = DataReader(db_path="tushare.db")

    # 1. 查看缓存信息
    print("\n📂 概念数据缓存信息:")
    info = reader.get_concept_cache_info()
    for key, value in info.items():
        print(f"  {key}: {value}")

    # 2. 获取所有概念列表
    print("\n📋 获取所有概念板块 (前10个):")
    try:
        concepts = reader.get_all_concepts()
        print(concepts.head(10).to_string(index=False))
        print(f"\n  总计: {len(concepts)} 个概念板块")
    except Exception as e:
        print(f"  ⚠️ 获取失败: {e}")

    # 3. 搜索概念
    print("\n🔍 搜索含'芯片'的概念:")
    try:
        chips = reader.search_concepts("芯片")
        print(chips.to_string(index=False) if not chips.empty else "  无匹配结果")
    except Exception as e:
        print(f"  ⚠️ 搜索失败: {e}")

    # 4. 获取概念成分股
    print("\n🤖 获取'人工智能'概念成分股 (2024-01-15):")
    try:
        stocks = reader.get_concept_stocks('20240115', concept_name='人工智能')
        if not stocks.empty:
            print(stocks.head(10).to_string(index=False))
            print(f"\n  总计: {len(stocks)} 只股票")
        else:
            print("  ⚠️ 无数据")
    except Exception as e:
        print(f"  ⚠️ 获取失败: {e}")

    # 5. 获取股票所属概念
    print("\n📈 获取 000001.SZ 所属概念 (2024-01-15):")
    try:
        concepts = reader.get_stock_concepts('20240115', ts_code='000001.SZ')
        if not concepts.empty:
            print(concepts.to_string(index=False))
            print(f"\n  总计: {len(concepts)} 个概念")
        else:
            print("  ⚠️ 无数据")
    except Exception as e:
        print(f"  ⚠️ 获取失败: {e}")

    # 6. PIT 查询 - 获取截面数据
    print("\n📊 获取 2024-01-15 概念板块截面数据:")
    try:
        df = reader.get_concept_cross_section('20240115')
        if not df.empty:
            print(df.head(10)[['concept_code', 'concept_name', 'ts_code']].to_string(index=False))
            print(f"\n  总计: {len(df)} 条记录")
            print(f"  概念数: {df['concept_code'].nunique()}")
            print(f"  股票数: {df['ts_code'].nunique()}")
        else:
            print("  ⚠️ 无数据")
    except Exception as e:
        print(f"  ⚠️ 获取失败: {e}")

    reader.close()
    print("\n✅ DataReader 测试完成")


def test_direct_manager():
    """直接测试 ConceptDataManager"""
    print("\n" + "=" * 70)
    print("测试 2: 直接使用 ConceptDataManager")
    print("=" * 70)

    manager = ConceptDataManager(db_path="tushare.db")

    # 拉取数据
    print("\n📥 拉取概念数据:")
    success = manager.pull_data()
    if not success:
        print("  ⚠️ 数据拉取失败，尝试使用历史缓存")

    # 获取缓存信息
    print("\n📂 缓存信息:")
    info = manager.get_cache_info()
    for key, value in info.items():
        print(f"  {key}: {value}")

    # 如果数据已加载，进行一些查询
    if info.get('loaded'):
        print("\n📋 概念板块统计:")
        print(f"  总记录数: {info.get('total_records')}")
        print(f"  概念数量: {info.get('total_concepts')}")
        print(f"  股票数量: {info.get('total_stocks')}")

        # 随机查询一个概念
        concepts = manager.get_all_concepts()
        if not concepts.empty:
            random_concept = concepts.sample(1).iloc[0]
            print(f"\n🎲 随机查询概念 '{random_concept['concept_name']}':")
            stocks = manager.get_concept_stocks(
                '20240115',
                concept_code=random_concept['concept_code']
            )
            print(f"  2024-01-15 有 {len(stocks)} 只成分股")

    print("\n✅ ConceptDataManager 测试完成")


def main():
    print("\n" + "=" * 70)
    print("聚宽概念板块数据读取测试")
    print("数据源: https://github.com/tycallen/jquant_data_sync")
    print("=" * 70)

    try:
        test_with_data_reader()
    except Exception as e:
        print(f"\n❌ DataReader 测试失败: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_direct_manager()
    except Exception as e:
        print(f"\n❌ ConceptDataManager 测试失败: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 70)
    print("测试完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
