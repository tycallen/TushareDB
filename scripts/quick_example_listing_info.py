#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速示例：获取上市首日信息

这是一个最精简的示例，演示如何快速获取股票上市首日信息
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from tushare_db import DataReader

# 初始化读取器
reader = DataReader()

try:
    # 1. 获取所有股票的基本信息（含上市日期）
    print("正在查询所有股票...")
    stocks = reader.query("""
        SELECT ts_code, name, list_date, market, list_status 
        FROM stock_basic
        WHERE list_date IS NOT NULL
        ORDER BY list_date DESC
        LIMIT 10
    """)
    
    print(f"\n找到 {len(stocks)} 只最近上市的股票:\n")
    
    # 2. 对每只股票，获取其上市首日的交易数据
    for idx, stock in stocks.iterrows():
        ts_code = stock['ts_code']
        name = stock['name']
        list_date = stock['list_date']
        
        # 查询上市首日行情
        first_day = reader.get_stock_daily(
            ts_code=ts_code,
            start_date=list_date,
            end_date=list_date
        )
        
        if not first_day.empty:
            data = first_day.iloc[0]
            change_pct = ((data['close'] / data['open']) - 1) * 100 if data['open'] > 0 else 0
            
            print(f"{idx+1}. {name} ({ts_code})")
            print(f"   上市日期: {list_date}")
            print(f"   开盘: {data['open']:.2f}, 收盘: {data['close']:.2f}")
            print(f"   最高: {data['high']:.2f}, 最低: {data['low']:.2f}")
            print(f"   涨跌幅: {change_pct:+.2f}%")
            print(f"   成交量: {data['vol']:.0f} 手")
            print()
        else:
            print(f"{idx+1}. {name} ({ts_code}) - 上市首日无交易数据")
            print()
    
finally:
    reader.close()
    print("完成!")
