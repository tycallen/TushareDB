"""
测试TushareDBClient和API函数的功能。
"""
import os
import pandas as pd
from tushare_db.client import TushareDBClient
from tushare_db.api import *


def test_dc_member():
    """
    测试dc_member接口的功能。
    """
    # 初始化客户端
    token = os.getenv('TUSHARE_TOKEN')
    client = TushareDBClient(tushare_token=token)
    
    # 测试获取单日数据
    print("测试获取单日数据:")
    df = dc_member(client, trade_date='20250102', ts_code='BK1184.DC')
    print(f"获取到{len(df)}条记录:")
    print(df.head())
    
    # 测试获取多日数据
    print("\n测试获取多日数据:")
    start_date = '20250101'
    end_date = '20250110'
    
    # 获取交易日历
    trade_dates = trade_cal(client, start_date=start_date, end_date=end_date, is_open='1')
    trade_dates = trade_dates['cal_date'].tolist()
    
    # 逐日获取数据
    all_data = []
    for date in trade_dates:
        daily_data = dc_member(client, trade_date=date, ts_code='BK1184.DC')
        if not daily_data.empty:
            all_data.append(daily_data)
    
    if all_data:
        combined_df = pd.concat(all_data)
        print(f"共获取到{len(combined_df)}条记录:")
        print(combined_df.head())
    else:
        print("没有获取到数据")

def test_get_top_n_sector_members():
    token = os.getenv('TUSHARE_TOKEN')
    client = TushareDBClient(tushare_token=token)
    result = get_top_n_sector_members(client=client, start_date='20250101', end_date='20250704')
    print(result)


if __name__ == '__main__':
    # test_dc_member()
    test_get_top_n_sector_members()