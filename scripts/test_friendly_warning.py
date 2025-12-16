#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试友好提示功能
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from tushare_db import DataReader

print("测试：查询一个不存在的数据库")
print("=" * 60)

# 使用一个不存在的数据库文件来触发空结果
reader = DataReader(db_path='test_empty.db')

try:
    # 测试1: 查询所有股票（应该显示友好提示）
    print("\n测试1: get_all_listing_first_day_info()")
    print("-" * 60)
    df = reader.get_all_listing_first_day_info()
    print(f"返回结果: {len(df)} 条")
    
    # 测试2: 查询单只股票（应该显示友好提示）
    print("\n测试2: get_listing_first_day_info('000001.SZ')")
    print("-" * 60)
    df = reader.get_listing_first_day_info('000001.SZ')
    print(f"返回结果: {len(df)} 条")
    
finally:
    reader.close()
    # 清理测试数据库
    if os.path.exists('test_empty.db'):
        os.remove('test_empty.db')

print("\n" + "=" * 60)
print("✅ 友好提示功能测试完成")
