#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查复权数据的脚本
对比本地数据库和Tushare在线数据
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.tushare_db.reader import DataReader
from src.tushare_db.tushare_fetcher import TushareFetcher
from src.tushare_db.rate_limit_config import STANDARD_PROFILE
import pandas as pd
import os

def check_adj_data(ts_code: str, trade_date: str):
    """
    检查指定股票在指定日期的复权数据
    
    Args:
        ts_code: 股票代码
        trade_date: 交易日期 (YYYYMMDD)
    """
    reader = DataReader()
    
    # 获取token
    token = os.getenv('TUSHARE_TOKEN')
    if not token:
        print("警告: 未设置TUSHARE_TOKEN环境变量,将跳过Tushare在线数据查询")
        fetcher = None
    else:
        fetcher = TushareFetcher(token=token, rate_limit_config=STANDARD_PROFILE)
    
    # 计算日期范围(前后各2天)
    from datetime import datetime, timedelta
    date_obj = datetime.strptime(trade_date, '%Y%m%d')
    start_date = (date_obj - timedelta(days=2)).strftime('%Y%m%d')
    end_date = (date_obj + timedelta(days=2)).strftime('%Y%m%d')
    
    print('=' * 80)
    print(f'检查股票: {ts_code}, 日期: {trade_date}')
    print('=' * 80)
    
    # 1. 本地数据库 - 未复权
    print('\n【本地数据库 - 未复权】')
    df_local_none = reader.get_stock_daily(
        ts_code=ts_code,
        start_date=start_date,
        end_date=end_date,
        adj=None
    )
    if not df_local_none.empty:
        print(df_local_none[['trade_date', 'open', 'high', 'low', 'close', 'vol']])
        target_data = df_local_none[df_local_none['trade_date'] == trade_date]
        if not target_data.empty:
            print(f"\n{trade_date}详情:")
            print(f"  开盘价: {target_data.iloc[0]['open']:.6f}")
            print(f"  最高价: {target_data.iloc[0]['high']:.6f}")
            print(f"  最低价: {target_data.iloc[0]['low']:.6f}")
            print(f"  收盘价: {target_data.iloc[0]['close']:.6f}")
    else:
        print('无数据')
    
    # 2. 本地数据库 - 前复权
    print('\n【本地数据库 - 前复权】')
    df_local_qfq = reader.get_stock_daily(
        ts_code=ts_code,
        start_date=start_date,
        end_date=end_date,
        adj='qfq'
    )
    if not df_local_qfq.empty:
        print(df_local_qfq[['trade_date', 'open', 'high', 'low', 'close', 'vol']])
        target_data = df_local_qfq[df_local_qfq['trade_date'] == trade_date]
        if not target_data.empty:
            print(f"\n{trade_date}详情:")
            print(f"  开盘价: {target_data.iloc[0]['open']:.6f}")
            print(f"  最高价: {target_data.iloc[0]['high']:.6f}")
            print(f"  最低价: {target_data.iloc[0]['low']:.6f}")
            print(f"  收盘价: {target_data.iloc[0]['close']:.6f}")
    else:
        print('无数据')
    
    # 3. 本地数据库 - 复权因子
    print('\n【本地数据库 - 复权因子】')
    df_adj = reader.get_adj_factor(
        ts_code=ts_code,
        start_date=start_date,
        end_date=end_date
    )
    if not df_adj.empty:
        print(df_adj)
    else:
        print('无复权因子数据')
    
    # 4. Tushare在线 - 未复权
    print('\n【Tushare在线 - 未复权】')
    if fetcher is None:
        print('跳过(未设置TUSHARE_TOKEN)')
    else:
        try:
            df_ts_none = fetcher.fetch('daily',
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date
            )
            if df_ts_none is not None and not df_ts_none.empty:
                df_ts_none = df_ts_none.sort_values('trade_date')
                print(df_ts_none[['trade_date', 'open', 'high', 'low', 'close', 'vol']])
                target_data = df_ts_none[df_ts_none['trade_date'] == trade_date]
                if not target_data.empty:
                    print(f"\n{trade_date}详情:")
                    print(f"  开盘价: {target_data.iloc[0]['open']:.6f}")
                    print(f"  最高价: {target_data.iloc[0]['high']:.6f}")
                    print(f"  最低价: {target_data.iloc[0]['low']:.6f}")
                    print(f"  收盘价: {target_data.iloc[0]['close']:.6f}")
            else:
                print('无数据')
        except Exception as e:
            print(f'查询失败: {e}')
    
    # 5. Tushare在线 - 前复权 (使用pro_bar)
    print('\n【Tushare在线 - 前复权 (pro_bar)】')
    try:
        import tushare as ts
        df_ts_qfq = ts.pro_bar(
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            adj='qfq'
        )
        if df_ts_qfq is not None and not df_ts_qfq.empty:
            df_ts_qfq = df_ts_qfq.sort_values('trade_date')
            print(df_ts_qfq[['trade_date', 'open', 'high', 'low', 'close', 'vol']])
            target_data = df_ts_qfq[df_ts_qfq['trade_date'] == trade_date]
            if not target_data.empty:
                print(f"\n{trade_date}详情:")
                print(f"  开盘价: {target_data.iloc[0]['open']:.6f}")
                print(f"  最高价: {target_data.iloc[0]['high']:.6f}")
                print(f"  最低价: {target_data.iloc[0]['low']:.6f}")
                print(f"  收盘价: {target_data.iloc[0]['close']:.6f}")
        else:
            print('无数据')
    except Exception as e:
        print(f'查询失败: {e}')
    
    # 6. Tushare在线 - 复权因子
    print('\n【Tushare在线 - 复权因子】')
    if fetcher is None:
        print('跳过(未设置TUSHARE_TOKEN)')
    else:
        try:
            df_ts_adj = fetcher.fetch('adj_factor',
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date
            )
            if df_ts_adj is not None and not df_ts_adj.empty:
                df_ts_adj = df_ts_adj.sort_values('trade_date')
                print(df_ts_adj)
            else:
                print('无数据')
        except Exception as e:
            print(f'查询失败: {e}')
    
    # 7. 对比分析
    print('\n' + '=' * 80)
    print('【对比分析】')
    print('=' * 80)
    
    if (not df_local_none.empty and not df_local_qfq.empty and 
        not df_adj.empty):
        
        local_none = df_local_none[df_local_none['trade_date'] == trade_date]
        local_qfq = df_local_qfq[df_local_qfq['trade_date'] == trade_date]
        local_adj = df_adj[df_adj['trade_date'] == trade_date]
        
        if not local_none.empty and not local_qfq.empty and not local_adj.empty:
            none_open = local_none.iloc[0]['open']
            qfq_open = local_qfq.iloc[0]['open']
            adj_factor = local_adj.iloc[0]['adj_factor']
            
            print(f"\n本地数据:")
            print(f"  未复权开盘价: {none_open:.6f}")
            print(f"  前复权开盘价: {qfq_open:.6f}")
            print(f"  复权因子: {adj_factor:.6f}")
            print(f"  计算验证: {none_open:.6f} / {adj_factor:.6f} = {none_open / adj_factor:.6f}")
            print(f"  与数据库前复权的差异: {abs(qfq_open - none_open / adj_factor):.10f}")
            
            # 如果有Tushare在线数据,也进行对比
            try:
                if 'df_ts_qfq' in locals() and df_ts_qfq is not None and not df_ts_qfq.empty:
                    ts_qfq = df_ts_qfq[df_ts_qfq['trade_date'] == trade_date]
                    if not ts_qfq.empty:
                        ts_qfq_open = ts_qfq.iloc[0]['open']
                        print(f"\nTushare在线前复权开盘价: {ts_qfq_open:.6f}")
                        print(f"本地与Tushare的差异: {abs(qfq_open - ts_qfq_open):.10f}")
                        print(f"差异百分比: {abs(qfq_open - ts_qfq_open) / ts_qfq_open * 100:.6f}%")
            except:
                pass

if __name__ == '__main__':
    # 默认检查002632.SZ在20240103的数据
    ts_code = '002632.SZ'
    trade_date = '20240103'
    
    if len(sys.argv) > 1:
        ts_code = sys.argv[1]
    if len(sys.argv) > 2:
        trade_date = sys.argv[2]
    
    check_adj_data(ts_code, trade_date)
