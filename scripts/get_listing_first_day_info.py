#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
获取所有A股（包括退市股票）的上市首日信息

本脚本演示如何利用 Tushare-DuckDB 项目获取所有A股的上市首日数据，包括：
1. 股票基本信息（含上市日期）
2. 上市首日的交易数据
"""

import sys
import os
from datetime import datetime
import pandas as pd

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from tushare_db import DataDownloader, DataReader


def ensure_data_downloaded():
    """
    确保数据已下载（如果数据库中没有数据，则下载）
    """
    print("=" * 60)
    print("步骤 1: 确保基础数据已下载")
    print("=" * 60)
    
    downloader = DataDownloader()
    reader = DataReader()
    
    try:
        # 检查是否已有股票基本信息
        stock_basic = reader.get_stock_basic(list_status=None)
        
        if stock_basic.empty:
            print("未找到股票基本信息，开始下载...")
            # 下载上市股票
            downloader.download_stock_basic('L')
            # 下载退市股票
            downloader.download_stock_basic('D')
            print("股票基本信息下载完成")
        else:
            print(f"已有 {len(stock_basic)} 只股票的基本信息")
            
    finally:
        downloader.close()
        reader.close()


def get_all_stocks_with_list_date():
    """
    获取所有股票（含退市股票）的基本信息，包括上市日期
    
    Returns:
        pd.DataFrame: 包含 ts_code, name, list_date, market, list_status 等信息
    """
    print("\n" + "=" * 60)
    print("步骤 2: 获取所有股票的基本信息（含上市日期）")
    print("=" * 60)
    
    reader = DataReader()
    
    try:
        # 获取所有股票（不区分上市状态）
        # 注意：由于 stock_basic 表结构可能不同，我们使用 SQL 查询来确保兼容性
        stocks = reader.query("""
            SELECT * FROM stock_basic
            ORDER BY list_date
        """)
        
        print(f"\n共查询到 {len(stocks)} 只股票")
        
        # 显示统计信息
        if 'list_status' in stocks.columns:
            print("\n按上市状态分类:")
            status_counts = stocks['list_status'].value_counts()
            for status, count in status_counts.items():
                status_name = {
                    'L': '上市',
                    'D': '退市',
                    'P': '暂停上市'
                }.get(status, status)
                print(f"  {status_name} ({status}): {count} 只")
        
        # 显示上市日期范围
        if 'list_date' in stocks.columns:
            stocks_with_date = stocks[stocks['list_date'].notna()]
            if not stocks_with_date.empty:
                print(f"\n上市日期范围:")
                print(f"  最早: {stocks_with_date['list_date'].min()}")
                print(f"  最晚: {stocks_with_date['list_date'].max()}")
        
        return stocks
        
    finally:
        reader.close()


def get_first_trading_day_data(stocks_df: pd.DataFrame, sample_size: int = None):
    """
    获取每只股票上市首日的交易数据
    
    Args:
        stocks_df: 股票基本信息 DataFrame
        sample_size: 抽样数量（如果为 None，处理所有股票；建议先用小样本测试）
    
    Returns:
        pd.DataFrame: 包含上市首日交易数据的 DataFrame
    """
    print("\n" + "=" * 60)
    print(f"步骤 3: 获取{'抽样 ' + str(sample_size) + ' 只' if sample_size else '所有'}股票的上市首日交易数据")
    print("=" * 60)
    
    reader = DataReader()
    downloader = DataDownloader()
    
    results = []
    
    try:
        # 过滤出有上市日期的股票
        stocks_with_date = stocks_df[stocks_df['list_date'].notna()].copy()
        
        if sample_size:
            stocks_with_date = stocks_with_date.head(sample_size)
        
        print(f"\n处理 {len(stocks_with_date)} 只股票...")
        
        for idx, stock in stocks_with_date.iterrows():
            ts_code = stock['ts_code']
            list_date = stock['list_date']
            name = stock.get('name', '')
            
            # 查询上市首日数据
            try:
                # 先查询数据库
                first_day_data = reader.get_stock_daily(
                    ts_code=ts_code,
                    start_date=list_date,
                    end_date=list_date,
                    adj=None  # 不复权
                )
                
                # 如果数据库中没有，尝试下载
                if first_day_data.empty:
                    print(f"  {ts_code} ({name}) 数据缺失，尝试下载...")
                    downloader.download_stock_daily(ts_code, list_date, list_date)
                    
                    # 再次查询
                    first_day_data = reader.get_stock_daily(
                        ts_code=ts_code,
                        start_date=list_date,
                        end_date=list_date,
                        adj=None
                    )
                
                if not first_day_data.empty:
                    # 合并基本信息和交易数据
                    info = {
                        'ts_code': ts_code,
                        'name': name,
                        'list_date': list_date,
                        'market': stock.get('market', ''),
                        'list_status': stock.get('list_status', ''),
                        'open': first_day_data.iloc[0].get('open'),
                        'high': first_day_data.iloc[0].get('high'),
                        'low': first_day_data.iloc[0].get('low'),
                        'close': first_day_data.iloc[0].get('close'),
                        'vol': first_day_data.iloc[0].get('vol'),
                        'amount': first_day_data.iloc[0].get('amount'),
                    }
                    results.append(info)
                    
                    if len(results) % 1000 == 0:
                        print(f"  已处理 {len(results)} 只股票...")
                else:
                    print(f"  {ts_code} ({name}) 上市首日 {list_date} 无交易数据")
                    
            except Exception as e:
                print(f"  {ts_code} ({name}) 处理失败: {e}")
                continue
        
        result_df = pd.DataFrame(results)
        print(f"\n成功获取 {len(result_df)} 只股票的上市首日数据")
        
        return result_df
        
    finally:
        reader.close()
        downloader.close()


def export_results(data: pd.DataFrame, output_path: str = None):
    """
    导出结果到 CSV 文件
    
    Args:
        data: 结果数据
        output_path: 输出文件路径（如果为 None，使用默认路径）
    """
    if output_path is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f'listing_first_day_data_{timestamp}.csv'
    
    print("\n" + "=" * 60)
    print(f"步骤 4: 导出结果到 {output_path}")
    print("=" * 60)
    
    data.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"✓ 成功导出 {len(data)} 条记录到 {output_path}")
    
    # 显示前几条记录
    print("\n前 10 条记录预览:")
    print(data.head(10).to_string())


def main():
    """
    主函数：完整示例
    """
    print("\n获取所有A股上市首日信息")
    print("=" * 60)
    
    # 步骤 1: 确保数据已下载
    ensure_data_downloaded()
    
    # 步骤 2: 获取所有股票基本信息
    all_stocks = get_all_stocks_with_list_date()
    
    # 步骤 3: 获取上市首日交易数据
    # 建议先用小样本测试（如 sample_size=10）
    # 完整处理所有股票可能需要较长时间
    first_day_data = get_first_trading_day_data(
        all_stocks,
        sample_size=None  # 修改为 None 可处理所有股票
    )
    
    # 步骤 4: 导出结果
    if not first_day_data.empty:
        export_results(first_day_data)
    
    print("\n" + "=" * 60)
    print("处理完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
