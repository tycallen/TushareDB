from tushare_db import TushareDBClient, stock_basic, StockBasic, trade_cal, TradeCal, hs_const, HsConst, stock_company, StockCompany, pro_bar, ProBar
from tushare_db.client import TushareDBClientError

try:
    client = TushareDBClient()
    
    # 测试 stock_basic 接口
    df_sb = stock_basic(client, ts_code='000001.SZ')
    print("stock_basic call successful")

    # 测试 TradeCal 接口
    df_tc = trade_cal(client, start_date='20230101', end_date='20230131')
    print("trade_cal call successful")

    # 测试 HsConst 接口
    df_hc = hs_const(client, hs_type='SH')
    print("hs_const call successful")

    # 测试 StockCompany 接口
    df_sc = stock_company(client, exchange='SZSE')
    print("stock_company call successful")

    # 测试 ProBar 接口
    df_pb = pro_bar(client, ts_code='000001.SZ', start_date='20230101', end_date='20230131')
    print("pro_bar call successful")

    # import tushare as ts
    # df = ts.pro_bar(ts_code='000001.SZ', start_date='20230101', end_date='20230131')
    # print(df)
    # print("pro_bar call successful")

except TushareDBClientError as e:
    print(f"Successfully caught expected error: {e}")
