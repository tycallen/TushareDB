#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用新增 API 获取上市首日信息的示例

演示如何使用 DataReader 的新方法：
1. get_all_listing_first_day_info() - 获取所有股票的上市首日信息
2. get_listing_first_day_info() - 获取单只股票的上市首日信息
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from tushare_db import DataReader

def example_1_get_all_listing_first_day_info():
    """示例1: 获取所有股票的上市首日信息"""
    print("=" * 60)
    print("示例1: 获取所有上市股票的首日信息")
    print("=" * 60)
    
    reader = DataReader()
    
    try:
        # 方法1: 获取所有上市股票（默认不包括无交易数据的股票）
        df = reader.get_all_listing_first_day_info(list_status='L')
        
        print(f"\n共 {len(df)} 只上市股票")
        print("\n前5条记录:")
        print(df.head())
        
        # 统计信息
        if not df.empty and 'close' in df.columns:
            avg_return = ((df['close'] / df['open'] - 1) * 100).mean()
            print(f"\n平均上市首日涨跌幅: {avg_return:.2f}%")
    
    finally:
        reader.close()


def example_2_filter_by_market():
    """示例2: 按市场筛选"""
    print("\n" + "=" * 60)
    print("示例2: 获取科创板股票的上市首日信息")
    print("=" * 60)
    
    reader = DataReader()
    
    try:
        # 只查询科创板股票
        df = reader.get_all_listing_first_day_info(
            list_status='L',
            market='科创板'
        )
        
        print(f"\n科创板上市股票: {len(df)} 只")
        if not df.empty:
            print(df[['ts_code', 'name', 'list_date', 'close']].head(10))
    
    finally:
        reader.close()


def example_3_include_delisted():
    """示例3: 包含退市股票"""
    print("\n" + "=" * 60)
    print("示例3: 获取所有股票（含退市）的上市首日信息")
    print("=" * 60)
    
    reader = DataReader()
    
    try:
        # 不指定 list_status，获取所有股票
        df = reader.get_all_listing_first_day_info(
            list_status=None,  # 所有状态
            include_no_data=True  # 包含无交易数据的股票
        )
        
        print(f"\n总计: {len(df)} 只股票")
        
        # 按状态统计
        if 'list_status' in df.columns:
            status_counts = df['list_status'].value_counts()
            print("\n按状态分类:")
            for status, count in status_counts.items():
                status_name = {'L': '上市', 'D': '退市', 'P': '暂停'}.get(status, status)
                print(f"  {status_name}: {count} 只")
        
        # 有交易数据的股票数量
        has_data_count = df['close'].notna().sum()
        print(f"\n有上市首日交易数据: {has_data_count} 只")
        print(f"无上市首日交易数据: {len(df) - has_data_count} 只")
    
    finally:
        reader.close()


def example_4_get_single_stock():
    """示例4: 获取单只股票的上市首日信息"""
    print("\n" + "=" * 60)
    print("示例4: 获取单只股票的上市首日信息")
    print("=" * 60)
    
    reader = DataReader()
    
    try:
        # 查询平安银行
        ts_code = '000001.SZ'
        df = reader.get_listing_first_day_info(ts_code)
        
        if not df.empty:
            stock = df.iloc[0]
            print(f"\n股票代码: {stock['ts_code']}")
            print(f"股票名称: {stock['name']}")
            print(f"上市日期: {stock['list_date']}")
            
            if pd.notna(stock.get('close')):
                print(f"\n上市首日:")
                print(f"  开盘价: {stock['open']:.2f}")
                print(f"  收盘价: {stock['close']:.2f}")
                print(f"  最高价: {stock['high']:.2f}")
                print(f"  最低价: {stock['low']:.2f}")
                print(f"  成交量: {stock['vol']:.0f} 手")
                
                if stock['open'] > 0:
                    change_pct = (stock['close'] / stock['open'] - 1) * 100
                    print(f"  涨跌幅: {change_pct:+.2f}%")
            else:
                print("\n无上市首日交易数据")
    
    finally:
        reader.close()


def example_5_performance_comparison():
    """示例5: 性能对比 - 新API vs 传统方法"""
    print("\n" + "=" * 60)
    print("示例5: 性能对比")
    print("=" * 60)
    
    import time
    reader = DataReader()
    
    try:
        # 方法1: 使用新API（一次JOIN查询）
        start_time = time.time()
        df_new = reader.get_all_listing_first_day_info(list_status='L')
        time_new = time.time() - start_time
        
        print(f"\n新API方法（JOIN查询）:")
        print(f"  结果数量: {len(df_new)}")
        print(f"  耗时: {time_new:.3f} 秒")
        
        # 方法2: 传统方法（循环查询，仅演示少量股票）
        start_time = time.time()
        stocks = reader.get_stock_basic(list_status='L')
        stocks = stocks.head(10)  # 只测试前10只
        results = []
        for _, stock in stocks.iterrows():
            first_day = reader.get_stock_daily(
                stock['ts_code'],
                stock['list_date'],
                stock['list_date']
            )
            if not first_day.empty:
                results.append(first_day.iloc[0])
        time_old = time.time() - start_time
        
        print(f"\n传统方法（循环查询，仅10只股票）:")
        print(f"  结果数量: {len(results)}")
        print(f"  耗时: {time_old:.3f} 秒")
        
        if time_old > 0:
            estimated_full = time_old * len(df_new) / 10
            print(f"\n传统方法处理全部股票预计耗时: {estimated_full:.1f} 秒")
            print(f"性能提升: {estimated_full / time_new:.1f}x")
    
    finally:
        reader.close()


if __name__ == '__main__':
    import pandas as pd
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 120)
    
    # 运行所有示例
    example_1_get_all_listing_first_day_info()
    example_2_filter_by_market()
    example_3_include_delisted()
    example_4_get_single_stock()
    example_5_performance_comparison()
    
    print("\n" + "=" * 60)
    print("所有示例运行完成！")
    print("=" * 60)
