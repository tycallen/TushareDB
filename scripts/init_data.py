
import os
import warnings
import tushare_db
from datetime import datetime, timedelta

# 忽略来自 tushare 库内部的特定 FutureWarning
# Tushare 库使用了即将被废弃的 pandas 功能，此代码可以在不修改库源码的情况下抑制警告
warnings.filterwarnings("ignore", category=FutureWarning, module="tushare.pro.data_pro")


# 初始化 Tushare Pro API
# 请确保您已经设置了 TUSHARE_TOKEN 环境变量
# 例如: export TUSHARE_TOKEN='your_token_here'
tushare_token = os.getenv("TUSHARE_TOKEN")
if not tushare_token:
    raise ValueError("请设置 TUSHARE_TOKEN 环境变量")

client = tushare_db.TushareDBClient(tushare_token=tushare_token, db_path="/Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db")

def init_trade_cal():
    """初始化交易日历"""
    print("开始初始化交易日历...")
    tushare_db.api.trade_cal(client, start_date='19900101', end_date='20301231')

    print("交易日历初始化完成。")

def init_stock_basic():
    """初始化股票列表"""
    print("开始初始化股票列表...")
    tushare_db.api.stock_basic(list_status='L')
    tushare_db.api.stock_basic(list_status='D')
    tushare_db.api.stock_basic(list_status='P')
    print("股票列表初始化完成。")

def init_pro_bar():
    """初始化所有股票的历史日线数据"""
    print("开始初始化所有股票的历史日线数据...")
    # 获取所有股票代码
    all_stocks = tushare_db.api.stock_basic(list_status='L')
    for ts_code in all_stocks['ts_code']:
        print(f"正在获取 {ts_code} 的历史数据...")
        try:
            tushare_db.api.pro_bar(ts_code=ts_code, asset='E', freq='D', start_date='19900101')
        except Exception as e:
            print(f"获取 {ts_code} 数据时出错: {e}")
    print("所有股票的历史日线数据初始化完成。")

def main():
    """主函数，执行所有初始化任务"""
    print("开始数据初始化...")
    init_trade_cal()
    # init_stock_basic()
    # init_pro_bar()
    client.get_all_stock_qfq_daily_bar(start_date='20000101', end_date=datetime.now().strftime('%Y%m%d'))
    print("所有数据初始化任务完成！")

if __name__ == "__main__":
    main()
