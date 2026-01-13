#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证Tushare接口数据
对比daily接口(未复权)和adj_factor接口与本地数据库
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.tushare_db.reader import DataReader
from src.tushare_db.tushare_fetcher import TushareFetcher
from src.tushare_db.rate_limit_config import STANDARD_PROFILE
import pandas as pd

def verify_tushare_interfaces(ts_code: str, start_date: str, end_date: str):
    """
    验证Tushare接口数据
    
    Args:
        ts_code: 股票代码
        start_date: 开始日期 (YYYYMMDD)
        end_date: 结束日期 (YYYYMMDD)
    """
    reader = DataReader()
    
    # 获取token
    token = os.getenv('TUSHARE_TOKEN')
    if not token:
        print("错误: 未设置TUSHARE_TOKEN环境变量")
        return
    
    fetcher = TushareFetcher(token=token, rate_limit_config=STANDARD_PROFILE)
    
    print('=' * 80)
    print(f'验证Tushare接口: {ts_code}, 日期范围: {start_date} ~ {end_date}')
    print('=' * 80)
    
    # 1. 验证daily接口(未复权)
    print('\n【1. Tushare daily接口 - 未复权数据】')
    print('-' * 80)
    try:
        df_ts_daily = fetcher.fetch('daily',
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date
        )
        if df_ts_daily is not None and not df_ts_daily.empty:
            df_ts_daily = df_ts_daily.sort_values('trade_date')
            print(f"获取到 {len(df_ts_daily)} 条数据")
            print(df_ts_daily[['trade_date', 'open', 'high', 'low', 'close', 'vol', 'amount']])
        else:
            print('无数据')
    except Exception as e:
        print(f'查询失败: {e}')
        df_ts_daily = None
    
    # 2. 验证adj_factor接口
    print('\n【2. Tushare adj_factor接口 - 复权因子】')
    print('-' * 80)
    try:
        df_ts_adj = fetcher.fetch('adj_factor',
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date
        )
        if df_ts_adj is not None and not df_ts_adj.empty:
            df_ts_adj = df_ts_adj.sort_values('trade_date')
            print(f"获取到 {len(df_ts_adj)} 条数据")
            print(df_ts_adj)
        else:
            print('无数据')
    except Exception as e:
        print(f'查询失败: {e}')
        df_ts_adj = None
    
    # 3. 本地数据库 - 未复权数据
    print('\n【3. 本地数据库 - 未复权数据(pro_bar表)】')
    print('-' * 80)
    df_local_daily = reader.get_stock_daily(
        ts_code=ts_code,
        start_date=start_date,
        end_date=end_date,
        adj=None
    )
    if not df_local_daily.empty:
        print(f"获取到 {len(df_local_daily)} 条数据")
        print(df_local_daily[['trade_date', 'open', 'high', 'low', 'close', 'vol', 'amount']])
    else:
        print('无数据')
    
    # 4. 本地数据库 - 复权因子
    print('\n【4. 本地数据库 - 复权因子(adj_factor表)】')
    print('-' * 80)
    df_local_adj = reader.get_adj_factor(
        ts_code=ts_code,
        start_date=start_date,
        end_date=end_date
    )
    if not df_local_adj.empty:
        print(f"获取到 {len(df_local_adj)} 条数据")
        print(df_local_adj)
    else:
        print('无数据')
    
    # 5. 数据对比
    print('\n' + '=' * 80)
    print('【数据对比分析】')
    print('=' * 80)
    
    # 5.1 对比daily数据
    if df_ts_daily is not None and not df_ts_daily.empty and not df_local_daily.empty:
        print('\n▶ Daily数据对比:')
        print('-' * 80)
        
        # 合并数据进行对比
        comparison = df_ts_daily.merge(
            df_local_daily,
            on='trade_date',
            suffixes=('_tushare', '_local')
        )
        
        if not comparison.empty:
            print(f"共有 {len(comparison)} 个交易日的数据")
            
            # 检查关键字段是否一致
            fields_to_check = ['open', 'high', 'low', 'close', 'vol']
            all_match = True
            
            for field in fields_to_check:
                ts_col = f'{field}_tushare'
                local_col = f'{field}_local'
                
                if ts_col in comparison.columns and local_col in comparison.columns:
                    # 计算差异
                    diff = (comparison[ts_col] - comparison[local_col]).abs()
                    max_diff = diff.max()
                    mean_diff = diff.mean()
                    
                    # 判断是否匹配(允许小的浮点数误差)
                    matches = (diff < 0.01).all()
                    
                    status = "✓ 一致" if matches else "✗ 不一致"
                    print(f"  {field:10s}: {status} (最大差异: {max_diff:.6f}, 平均差异: {mean_diff:.6f})")
                    
                    if not matches:
                        all_match = False
                        # 显示不一致的记录
                        mismatches = comparison[diff >= 0.01]
                        if not mismatches.empty:
                            print(f"    不一致的记录数: {len(mismatches)}")
                            print(f"    示例:")
                            print(mismatches[['trade_date', ts_col, local_col]].head())
            
            if all_match:
                print("\n✅ Daily接口数据与本地数据库完全一致!")
            else:
                print("\n⚠️  Daily接口数据与本地数据库存在差异!")
        else:
            print("⚠️  没有重叠的交易日数据")
    
    # 5.2 对比adj_factor数据
    if df_ts_adj is not None and not df_ts_adj.empty and not df_local_adj.empty:
        print('\n▶ 复权因子对比:')
        print('-' * 80)
        
        # 合并数据进行对比
        adj_comparison = df_ts_adj.merge(
            df_local_adj,
            on='trade_date',
            suffixes=('_tushare', '_local')
        )
        
        if not adj_comparison.empty:
            print(f"共有 {len(adj_comparison)} 个交易日的复权因子")
            
            # 检查复权因子是否一致
            diff = (adj_comparison['adj_factor_tushare'] - adj_comparison['adj_factor_local']).abs()
            max_diff = diff.max()
            mean_diff = diff.mean()
            matches = (diff < 0.0001).all()
            
            status = "✓ 一致" if matches else "✗ 不一致"
            print(f"  adj_factor: {status} (最大差异: {max_diff:.6f}, 平均差异: {mean_diff:.6f})")
            
            if not matches:
                # 显示不一致的记录
                mismatches = adj_comparison[diff >= 0.0001]
                if not mismatches.empty:
                    print(f"  不一致的记录数: {len(mismatches)}")
                    print(f"  示例:")
                    print(mismatches[['trade_date', 'adj_factor_tushare', 'adj_factor_local']].head())
            else:
                print("\n✅ 复权因子数据与本地数据库完全一致!")
        else:
            print("⚠️  没有重叠的交易日数据")
    
    # 6. 总结
    print('\n' + '=' * 80)
    print('【验证总结】')
    print('=' * 80)
    print("\n根据Tushare官方文档:")
    print("- daily接口: 未复权行情数据")
    print("- adj_factor接口: 复权因子数据")
    print("- 前复权价格 = 未复权价格 × (当日复权因子 / 最新复权因子)")
    print("\n验证结果:")
    if df_ts_daily is not None and not df_ts_daily.empty:
        print("✓ daily接口可以正常获取未复权数据")
    else:
        print("✗ daily接口无法获取数据")
    
    if df_ts_adj is not None and not df_ts_adj.empty:
        print("✓ adj_factor接口可以正常获取复权因子")
    else:
        print("✗ adj_factor接口无法获取数据")

if __name__ == '__main__':
    # 默认验证002632.SZ在20240102-20240105的数据
    ts_code = '002632.SZ'
    start_date = '20240102'
    end_date = '20240105'
    
    if len(sys.argv) > 1:
        ts_code = sys.argv[1]
    if len(sys.argv) > 2:
        start_date = sys.argv[2]
    if len(sys.argv) > 3:
        end_date = sys.argv[3]
    
    verify_tushare_interfaces(ts_code, start_date, end_date)
