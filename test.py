from tushare_db import TushareDBClient, stock_basic, StockBasic, trade_cal, TradeCal, hs_const, HsConst, stock_company, StockCompany, pro_bar, ProBar
from tushare_db.client import TushareDBClientError

try:
    client = TushareDBClient()
    
    # base init 
    # client.initialize_basic_data()

    # download stock basic data
    client.get_all_stock_qfq_daily_bar(start_date="20000101", end_date="20250628")

except TushareDBClientError as e:
    print(f"Successfully caught expected error: {e}")
