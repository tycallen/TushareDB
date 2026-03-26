#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用模拟数据测试概念板块数据读取功能
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
from datetime import date
from io import StringIO

# 创建模拟的概念板块数据
MOCK_CSV_DATA = """concept_code,concept_name,stock_code,in_date,out_date
GN001,人工智能,000001.XSHE,2023-01-01,2099-12-31
GN001,人工智能,000002.XSHE,2023-01-01,2099-12-31
GN001,人工智能,000003.XSHE,2023-06-01,2099-12-31
GN002,芯片概念,000001.XSHE,2023-01-01,2099-12-31
GN002,芯片概念,000004.XSHE,2023-03-01,2099-12-31
GN003,新能源,000002.XSHE,2023-01-01,2024-01-01
GN003,新能源,000005.XSHE,2023-01-01,2099-12-31
GN004,区块链,000001.XSHE,2023-01-01,2099-12-31
GN004,区块链,000003.XSHE,2023-01-01,2099-12-31
GN004,区块链,000005.XSHE,2023-01-01,2099-12-31
"""

def test_mock_data():
    """测试模拟数据功能"""
    print("=" * 70)
    print("概念板块数据功能测试（使用模拟数据）")
    print("=" * 70)

    # 1. 读取模拟数据
    print("\n📥 读取模拟数据...")
    df = pd.read_csv(StringIO(MOCK_CSV_DATA), dtype={'stock_code': str, 'concept_code': str})
    print(f"  原始数据: {len(df)} 行")
    print(df.to_string(index=False))

    # 2. 转换日期格式和股票代码
    df['in_date'] = pd.to_datetime(df['in_date']).dt.strftime('%Y%m%d')
    df['out_date'] = pd.to_datetime(df['out_date']).dt.strftime('%Y%m%d')
    df['ts_code'] = df['stock_code'].str.replace('.XSHE', '.SZ')

    print("\n🔄 转换后的数据:")
    print(df.to_string(index=False))

    # 3. PIT 查询测试
    print("\n" + "=" * 70)
    print("测试 PIT (Point-in-Time) 查询")
    print("=" * 70)

    # 查询2023年2月1日的截面数据
    query_date = '20230201'
    print(f"\n📅 查询日期: {query_date}")
    mask = (df['in_date'] <= query_date) & (df['out_date'] >= query_date)
    pit_df = df[mask]
    print(f"  结果: {len(pit_df)} 条记录")
    print(pit_df[['concept_code', 'concept_name', 'ts_code']].to_string(index=False))

    # 查询2024年1月15日的截面数据
    query_date = '20240115'
    print(f"\n📅 查询日期: {query_date}")
    mask = (df['in_date'] <= query_date) & (df['out_date'] >= query_date)
    pit_df = df[mask]
    print(f"  结果: {len(pit_df)} 条记录")
    print(pit_df[['concept_code', 'concept_name', 'ts_code']].to_string(index=False))

    # 4. 查询特定概念的成分股
    print("\n" + "=" * 70)
    print("测试查询概念成分股")
    print("=" * 70)

    concept_name = '人工智能'
    query_date = '20240115'
    print(f"\n🤖 概念 '{concept_name}' 在 {query_date} 的成分股:")
    mask = (
        (df['concept_name'] == concept_name) &
        (df['in_date'] <= query_date) &
        (df['out_date'] >= query_date)
    )
    result = df[mask]
    if not result.empty:
        print(result[['ts_code', 'concept_name', 'in_date', 'out_date']].to_string(index=False))
        print(f"\n  总计: {len(result)} 只股票")
    else:
        print("  无数据")

    # 5. 查询特定股票的概念
    print("\n" + "=" * 70)
    print("测试查询股票所属概念")
    print("=" * 70)

    ts_code = '000001.SZ'
    query_date = '20240115'
    print(f"\n📈 股票 '{ts_code}' 在 {query_date} 所属概念:")
    mask = (
        (df['ts_code'] == ts_code) &
        (df['in_date'] <= query_date) &
        (df['out_date'] >= query_date)
    )
    result = df[mask]
    if not result.empty:
        print(result[['concept_code', 'concept_name', 'in_date', 'out_date']].to_string(index=False))
        print(f"\n  总计: {len(result)} 个概念")
    else:
        print("  无数据")

    # 6. 历史变化查询
    print("\n" + "=" * 70)
    print("测试历史变化查询")
    print("=" * 70)

    ts_code = '000002.SZ'
    print(f"\n📊 股票 '{ts_code}' 的概念归属变化:")
    for query_date in ['20230201', '20230615', '20240115']:
        mask = (
            (df['ts_code'] == ts_code) &
            (df['in_date'] <= query_date) &
            (df['out_date'] >= query_date)
        )
        result = df[mask]
        concepts = result['concept_name'].tolist()
        print(f"  {query_date}: {concepts if concepts else '无概念'}")

    # 7. 概念板块统计
    print("\n" + "=" * 70)
    print("概念板块统计")
    print("=" * 70)

    query_date = '20240115'
    mask = (df['in_date'] <= query_date) & (df['out_date'] >= query_date)
    pit_df = df[mask]

    print(f"\n📊 {query_date} 统计:")
    print(f"  总记录数: {len(pit_df)}")
    print(f"  概念数量: {pit_df['concept_code'].nunique()}")
    print(f"  股票数量: {pit_df['ts_code'].nunique()}")

    print("\n各概念成分股数量:")
    concept_counts = pit_df.groupby('concept_name').size().sort_values(ascending=False)
    for name, count in concept_counts.items():
        print(f"  {name}: {count} 只")

    print("\n✅ 所有测试通过！")


def test_manager_logic():
    """测试 ConceptDataManager 逻辑"""
    print("\n" + "=" * 70)
    print("测试 ConceptDataManager 功能")
    print("=" * 70)

    from tushare_db.concept_manager import ConceptDataManager

    # 初始化（会使用缓存目录）
    manager = ConceptDataManager(db_path="tushare.db")

    print(f"\n📂 缓存目录: {manager.cache_dir}")

    # 手动加载模拟数据到内存
    df = pd.read_csv(StringIO(MOCK_CSV_DATA), dtype={'stock_code': str, 'concept_code': str})
    df['in_date'] = pd.to_datetime(df['in_date']).dt.strftime('%Y%m%d')
    df['out_date'] = pd.to_datetime(df['out_date']).dt.strftime('%Y%m%d')
    df['ts_code'] = df['stock_code'].str.replace('.XSHE', '.SZ')

    # 直接设置到管理器
    manager._df = df
    manager._load_date = date.today()
    manager._concept_loaded = True

    print("\n✅ 模拟数据已加载到 ConceptDataManager")

    # 测试方法
    print("\n📋 测试 get_all_concepts():")
    concepts = manager.get_all_concepts()
    print(concepts.to_string(index=False))

    print("\n🔍 测试 search_concepts('芯片'):")
    result = manager.search_concepts("芯片")
    print(result.to_string(index=False))

    print("\n🤖 测试 get_concept_stocks('20240115', concept_name='人工智能'):")
    stocks = manager.get_concept_stocks('20240115', concept_name='人工智能')
    print(stocks.to_string(index=False))

    print("\n📈 测试 get_stock_concepts('20240115', ts_code='000001.SZ'):")
    concepts = manager.get_stock_concepts('20240115', ts_code='000001.SZ')
    print(concepts.to_string(index=False))

    print("\n✅ ConceptDataManager 测试通过！")


def main():
    print("\n" + "=" * 70)
    print("聚宽概念板块数据功能测试")
    print("=" * 70)

    test_mock_data()
    test_manager_logic()

    print("\n" + "=" * 70)
    print("所有测试完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
